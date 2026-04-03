"""
langchain_explainer.py — LangChain Explanation Engine for FluxAgent
====================================================================
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | GSOM SPBU | 2026

PURPOSE
-------
Generates natural-language explanations of FluxAgent predictions and decisions
for different audiences: field engineers, NOC managers, and executive stakeholders.

Uses LangChain chains with RAG context injection. This provides a higher-quality
explanation layer than the rule-based system in the DiagnosticAgent, because it:
  1. Pulls relevant documentation from the RAG corpus at generation time
  2. Adapts tone and technical depth to the target audience
  3. Generates structured reports suitable for maintenance logs
  4. Supports multi-language output (French, Arabic, English — for African operators)

ARCHITECTURE
------------
  AlertJSON (from InterpreterAgent)
    ↓
  EvidenceBundle (from RAGPipeline)
    ↓
  LangChainExplainer
    ├── AudienceRouter (selects prompt template)
    ├── RAGContextInjector (inserts top-5 evidence chunks)
    ├── LLM Chain (LLaMA via Groq / Claude Haiku / GPT-4o-mini)
    └── ExplanationOutput (structured dict + rendered text)

USAGE
-----
  # Simple explanation for a field engineer
  from langchain_explainer import LangChainExplainer
  explainer = LangChainExplainer(api_key="your-groq-key", provider="groq")
  result = explainer.explain(alert_dict, evidence_bundle, audience="engineer")
  print(result["explanation"])

  # Executive summary
  result = explainer.explain(alert_dict, evidence_bundle, audience="executive")

  # Generate full maintenance report
  result = explainer.generate_report(alert_dict, evidence_bundle, ticket_dict)

SUPPORTED LLM PROVIDERS (all free/low-cost)
-------------------------------------------
  • Groq  — LLaMA 3.3 70B (fastest, free 14k req/day)
  • Anthropic — Claude Haiku (best quality)
  • OpenRouter — DeepSeek/Mistral free tier
  • HuggingFace Inference API — Zephyr-7B (free with HF token)
  • Ollama (local) — llama3 / mistral / phi3 (fully offline)
"""

import os, json, re, time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import logging

log = logging.getLogger("langchain_explainer")

# ── Prompt Templates by Audience ─────────────────────────────────────────────
PROMPTS = {

"engineer": """\
You are an expert telecom base station maintenance engineer.
A field engineer needs a clear, actionable explanation of the following predictive maintenance alert.
Be specific, cite alarm codes, use technical terminology, and tell them exactly what to check first.
Keep it under 300 words.

ALERT:
{alert_summary}

RELEVANT DOCUMENTATION:
{evidence_context}

TASK: Write a technical explanation for the field engineer covering:
1. What is failing and why (fault mechanism)
2. What they will find on-site (expected symptoms)
3. What to do first (step-by-step, most important action first)
4. What spare parts to bring
Cite sources as [DOC-ID] where relevant.""",

"noc_manager": """\
You are an AI assistant for a Network Operations Centre manager.
Explain the following predictive maintenance alert in operational terms.
Focus on service impact, SLA compliance, cost of inaction, and resource requirements.
Be concise (under 200 words) and business-oriented.

ALERT:
{alert_summary}

DOCUMENTATION:
{evidence_context}

Produce:
- One-sentence severity summary
- Service impact if unaddressed
- Required action and timeline (SLA)
- Estimated dispatch cost and time""",

"executive": """\
You are preparing a briefing for a senior telecoms executive with no technical background.
Explain what is happening, why it matters to the business, and what decision is needed.
Use plain language. No alarm codes. No technical jargon. Maximum 150 words.

SITUATION:
{alert_summary}

PRODUCE:
- What is happening (1 sentence, plain language)
- Business risk if ignored
- What action has been recommended
- Expected cost and time to resolve""",

"report": """\
You are generating a formal maintenance incident report for a telecom operator's records.
Be precise, structured, and complete. Use professional engineering language.

ALERT DATA:
{alert_summary}

EVIDENCE:
{evidence_context}

MAINTENANCE TICKET (if available):
{ticket_data}

PRODUCE A FORMAL REPORT WITH THESE SECTIONS:
1. Incident Summary
2. Predicted Fault Mechanism
3. AI Model Prediction Details (RUL, confidence, top features)
4. Evidence Chain (cite [DOC-ID] for each claim)
5. Actions Taken / Recommended
6. Root Cause Assessment
7. Lessons Learned / Follow-up
Format: structured prose, not bullet points.""",

}

# ── LLM Client wrappers (no LangChain dependency required — uses raw HTTP) ───
def _call_groq(key: str, prompt: str, max_tokens: int = 600) -> str:
    import urllib.request
    payload = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.25,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()

def _call_anthropic(key: str, prompt: str, max_tokens: int = 600) -> str:
    import urllib.request
    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["content"][0]["text"].strip()

def _call_openrouter(key: str, prompt: str, max_tokens: int = 600) -> str:
    import urllib.request
    payload = json.dumps({
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.25,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://fluxagent.noc",
                 "X-Title": "FluxAgent NOC"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()

def _call_hf(token: str, prompt: str, max_tokens: int = 500) -> str:
    import urllib.request
    full = f"<|user|>\n{prompt}\n<|assistant|>\n"
    payload = json.dumps({
        "inputs": full,
        "parameters": {"max_new_tokens": max_tokens, "temperature": 0.25,
                       "return_full_text": False}
    }).encode()
    req = urllib.request.Request(
        "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        data = json.loads(r.read())
        if isinstance(data, list):
            return data[0].get("generated_text","").strip()
        return data.get("generated_text","").strip()

def _call_ollama(model: str, prompt: str) -> str:
    """Fully local inference via Ollama (no API key required)."""
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()).get("response","").strip()

# ── Output dataclass ──────────────────────────────────────────────────────────
@dataclass
class ExplanationOutput:
    station_id:   str
    audience:     str
    explanation:  str
    provider:     str
    latency_ms:   float
    alert_id:     str = ""
    language:     str = "en"
    error:        Optional[str] = None

# ── Main Explainer ────────────────────────────────────────────────────────────
class LangChainExplainer:
    """
    Generates audience-adapted explanations from FluxAgent alert + evidence bundles.

    Providers (in priority order if auto):
      groq → anthropic → openrouter → huggingface → ollama → rule_based

    Example:
        explainer = LangChainExplainer()  # reads keys from env vars automatically
        result = explainer.explain(alert, bundle, audience="engineer")
    """

    PROVIDERS = {
        "groq":        ("GROQ_API_KEY",        _call_groq),
        "anthropic":   ("ANTHROPIC_API_KEY",    _call_anthropic),
        "openrouter":  ("OPENROUTER_API_KEY",   _call_openrouter),
        "huggingface": ("HF_TOKEN",             _call_hf),
    }

    def __init__(self,
                 api_key: str = None,
                 provider: str = "auto",
                 ollama_model: str = "llama3"):
        self.provider     = provider
        self.api_key      = api_key
        self.ollama_model = ollama_model
        self._resolved    = self._resolve_provider()
        log.info(f"LangChainExplainer initialised — provider: {self._resolved[0]}")

    def _resolve_provider(self):
        """Find the first working provider and its key."""
        if self.provider == "ollama":
            return ("ollama", None, lambda k, p: _call_ollama(self.ollama_model, p))
        if self.provider != "auto" and self.provider in self.PROVIDERS:
            env_k, fn = self.PROVIDERS[self.provider]
            key = self.api_key or os.environ.get(env_k,"").strip()
            if len(key) > 10:
                return (self.provider, key, fn)
        # Auto mode: try each in order
        for name, (env_k, fn) in self.PROVIDERS.items():
            key = self.api_key or os.environ.get(env_k,"").strip()
            if len(key) > 10:
                return (name, key, fn)
        # Try Ollama (local, no key)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return ("ollama", None, lambda k, p: _call_ollama(self.ollama_model, p))
        except Exception:
            pass
        return ("rule_based", None, None)

    def _build_alert_summary(self, alert: dict) -> str:
        return (
            f"Station: {alert.get('station_id','?')}\n"
            f"Predicted RUL: {alert.get('rul_cycles', alert.get('rul','?'))} cycles\n"
            f"Urgency: {alert.get('urgency','?')}\n"
            f"Subsystem: {alert.get('primary_subsystem', alert.get('sub','?'))}\n"
            f"Hypothesis: {alert.get('fault_hypothesis', alert.get('hyp','?'))}\n"
            f"Top feature: {alert.get('top_feat','?')} (imp={alert.get('top_imp','?')})\n"
            f"Alarm expected: {alert.get('alm','unknown')}\n"
            f"SLA: {alert.get('sla_hours', alert.get('sla','?'))}h\n"
            f"Model: XGBoost v2 Final · RMSE=14.60 · R²=0.874"
        )

    def _build_evidence_context(self, bundle: dict) -> str:
        chunks = bundle.get("chunks", [])
        if not chunks:
            return "No evidence chunks available."
        lines = []
        for c in chunks[:5]:
            cid   = c.get("citation_ref", c.get("chunk_id","?"))
            title = c.get("title","")
            text  = c.get("text","")[:400]
            lines.append(f"[{cid}] {title}\n{text}")
        return "\n\n".join(lines)

    def _rule_based_explanation(self, alert: dict, audience: str) -> str:
        """Fallback when no LLM is available."""
        rul  = alert.get("rul_cycles", alert.get("rul", "?"))
        urg  = alert.get("urgency","?")
        sub  = alert.get("primary_subsystem", alert.get("sub","?")).replace("_"," ")
        hyp  = alert.get("fault_hypothesis", alert.get("hyp","?"))
        a1   = alert.get("a1","perform inspection")
        sla  = alert.get("sla_hours", alert.get("sla","?"))

        if audience == "executive":
            return (f"Station {alert.get('station_id','?')} has {rul} operational cycles remaining "
                    f"before a maintenance event is required. The {sub} shows degradation. "
                    f"Action is required within {sla} hours to avoid service disruption.")
        elif audience == "noc_manager":
            return (f"{urg} alert — Station {alert.get('station_id','?')} | RUL={rul} cycles | "
                    f"Subsystem: {sub} | SLA: {sla}h\n"
                    f"Fault hypothesis: {hyp}\n"
                    f"Recommended action: {a1}")
        else:  # engineer / default
            return (f"FAULT: {hyp}\n"
                    f"RUL: {rul} cycles remaining | Urgency: {urg} | SLA: {sla}h\n"
                    f"Subsystem: {sub}\n"
                    f"Action: {a1}\n"
                    f"[Source: FluxAgent rule-based fallback — no LLM available]")

    def explain(self,
                alert: dict,
                evidence_bundle: dict,
                audience: str = "engineer",
                language: str = "en",
                max_tokens: int = 600) -> ExplanationOutput:
        """
        Generate an explanation for the given alert and evidence bundle.

        Parameters
        ----------
        alert : dict
            AlertJSON dict from InterpreterAgent (or STATIONS dict entry).
        evidence_bundle : dict
            EvidenceBundle dict from RAGPipeline.
        audience : str
            "engineer" | "noc_manager" | "executive" | "report"
        language : str
            "en" | "fr" | "ar" — adds language instruction to prompt.
        """
        t0 = time.time()
        provider_name, key, fn = self._resolved

        template = PROMPTS.get(audience, PROMPTS["engineer"])
        lang_note = "" if language == "en" else f"\n\nIMPORTANT: Respond in {'French' if language=='fr' else 'Arabic'}."

        prompt = template.format(
            alert_summary    = self._build_alert_summary(alert),
            evidence_context = self._build_evidence_context(evidence_bundle),
            ticket_data      = "N/A",
        ) + lang_note

        explanation = None
        error       = None

        if fn:
            try:
                explanation = fn(key, prompt, max_tokens)
            except Exception as e:
                error = str(e)[:200]
                log.warning(f"LLM call failed ({provider_name}): {error}")

        if not explanation:
            explanation = self._rule_based_explanation(alert, audience)
            provider_name = "rule_based"

        return ExplanationOutput(
            station_id  = str(alert.get("station_id", alert.get("id","?"))),
            audience    = audience,
            explanation = explanation,
            provider    = provider_name,
            latency_ms  = round((time.time()-t0)*1000, 1),
            alert_id    = str(alert.get("alert_id","")),
            language    = language,
            error       = error,
        )

    def generate_report(self, alert: dict, evidence_bundle: dict,
                        ticket: dict = None, language: str = "en") -> ExplanationOutput:
        """Generate a full formal maintenance incident report."""
        t0 = time.time()
        provider_name, key, fn = self._resolved

        template = PROMPTS["report"]
        lang_note = "" if language == "en" else f"\n\nRespond in {'French' if language=='fr' else 'Arabic'}."

        ticket_str = json.dumps(ticket, indent=2, default=str) if ticket else "No ticket data provided."
        prompt = template.format(
            alert_summary    = self._build_alert_summary(alert),
            evidence_context = self._build_evidence_context(evidence_bundle),
            ticket_data      = ticket_str,
        ) + lang_note

        explanation = None
        error = None
        if fn:
            try:
                explanation = fn(key, prompt, max_tokens=1200)
            except Exception as e:
                error = str(e)[:200]

        if not explanation:
            explanation = self._rule_based_explanation(alert, "engineer")
            provider_name = "rule_based"

        return ExplanationOutput(
            station_id  = str(alert.get("station_id", alert.get("id","?"))),
            audience    = "report",
            explanation = explanation,
            provider    = provider_name,
            latency_ms  = round((time.time()-t0)*1000, 1),
            alert_id    = str(alert.get("alert_id","")),
            language    = language,
            error       = error,
        )

    def batch_explain(self, alerts: list, bundles: list,
                      audience: str = "engineer") -> list:
        """Explain multiple alerts in batch."""
        return [self.explain(a, b, audience) for a, b in zip(alerts, bundles)]

# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DEMO_ALERT = {
        "station_id": "FD002_47", "urgency": "Critical",
        "rul_cycles": 14.7, "sla": 4,
        "primary_subsystem": "power_subsystem",
        "fault_hypothesis": "Power unit degradation — voltage instability or rectifier wear",
        "top_feat": "voltage_rolling_mean", "top_imp": 0.0744,
        "alm": "PWR-001 (undervoltage) or PWR-004 (mains failure)",
        "a1": "Execute remote rectifier reset via OMC",
    }
    DEMO_BUNDLE = {
        "chunks": [
            {"citation_ref": "[SOP-PWR-001]", "title": "SOP: Power Unit Fault Response",
             "text": "Step 1: Query OMC rectifier status. Step 2: Remote reset. Step 3: Dispatch if unresolved 30min."},
            {"citation_ref": "[MAN-PWR-001]", "title": "Power Unit Rectifier Specs",
             "text": "Rectifier output nominal 47.5–51.5V. Alarm threshold <44V. Replacement at 7yr or >5% ripple."},
        ]
    }

    print("=" * 60)
    print("FluxAgent LangChain Explainer — Demo")
    print("=" * 60)

    explainer = LangChainExplainer()  # auto-detects available API keys

    for audience in ["engineer", "noc_manager", "executive"]:
        result = explainer.explain(DEMO_ALERT, DEMO_BUNDLE, audience=audience)
        print(f"\n── {audience.upper()} ({result.provider}, {result.latency_ms}ms) ──")
        print(result.explanation[:500])

    print("\n" + "="*60)
