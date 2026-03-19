"""
LangChain Explainer — Plain-English Anomaly Explanation
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | March 2026

Produces a short, jargon-free explanation of a maintenance alert for:
  • Non-technical stakeholders (operations managers, site owners)
  • Field engineers in the field needing a quick brief
  • Executive summaries in reports

CHAIN DESIGN (LangChain LCEL):
  AlertJSON + DiagnosticReport → PromptTemplate → LLM → OutputParser → Plain text

PRODUCTION PATH (USE_LLM=true):
  Uses langchain-anthropic with claude-sonnet-4-6.
  Falls back to direct urllib call (same API) if langchain not installed.
  Falls back to rule-based template if no API key.

INSTALL (when network available):
  pip install langchain langchain-anthropic

OUTPUT CONTRACT:
  {
    "headline":     "One sentence — what is happening",
    "business_impact": "Two sentences — what this means operationally/financially",
    "recommended_action": "One sentence — what to do next",
    "confidence_plain": "High/Medium/Low with plain explanation",
    "full_explanation": "Full 3-4 sentence paragraph for reports",
    "engine":       "langchain | anthropic_direct | rule_based"
  }
"""

import os, json, re
from typing import Optional

USE_LLM  = os.environ.get("USE_LLM", "false").lower() == "true"
API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_ID = "claude-sonnet-4-6"

# ── Urgency → business impact mapping (rule-based fallback) ───────────────
URGENCY_BUSINESS_IMPACT = {
    "Critical": (
        "This station requires immediate attention — failure is expected within "
        "the next {rul:.0f} operational cycles (approximately {rul_h:.0f} hours). "
        "Without intervention within {sla}h, service outage and SLA breach are likely."
    ),
    "Warning": (
        "This station shows early degradation signs with {rul:.0f} cycles remaining "
        "(approximately {rul_h:.0f} hours). Scheduling a preventive inspection within "
        "{sla} hours will avoid unplanned downtime and emergency repair costs."
    ),
    "Monitor": (
        "This station is operating within acceptable bounds with {rul:.0f} cycles "
        "of remaining life. No immediate action required — add to the next planned "
        "maintenance cycle within {sla} hours."
    ),
}

SUBSYSTEM_PLAIN = {
    "power_subsystem":        "the power supply unit",
    "thermal_management":     "the cooling system",
    "rf_antenna":             "the antenna and RF chain",
    "backhaul_connectivity":  "the backhaul network connection",
    "baseband_processing":    "the baseband processing unit (BBU)",
    "sensor_array":           "the sensor measurement system",
    "degradation_trend":      "multiple subsystems (general wear)",
    "variability_index":      "measurement stability indicators",
    "health_trend":           "the overall station health index",
    "operational_age":        "equipment lifecycle (end-of-life approach)",
}

CONFIDENCE_PLAIN = {
    "high":     "High — the model is confident. Multiple corroborating sensor features confirm this diagnosis.",
    "moderate": "Moderate — the diagnosis is likely but one or two sensor readings are ambiguous. An on-site check will confirm.",
    "low":      "Low — limited evidence available. Additional data or a sensor calibration check is recommended before acting.",
}

# ── Prompt template ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior telecom maintenance manager explaining a predictive 
maintenance alert to a mixed audience — some are engineers, some are operations managers 
with no technical background.

Your explanation must be:
  • Written in plain English with NO technical jargon (no acronyms like VSWR, OMC, BBU 
    unless immediately explained in parentheses)
  • Concise: headline ≤ 15 words, business_impact ≤ 2 sentences, action ≤ 1 sentence
  • Honest about confidence level
  • Grounded in the evidence provided — do not invent information

Respond ONLY with a JSON object. No markdown, no preamble. Fields:
  headline, business_impact, recommended_action, confidence_plain, full_explanation"""

def _build_prompt(alert: dict, report: dict) -> str:
    rul      = alert.get("rul_cycles", 0)
    urgency  = alert.get("urgency", "Monitor")
    subsys   = SUBSYSTEM_PLAIN.get(alert.get("primary_subsystem",""), "the equipment")
    hyp      = alert.get("fault_hypothesis", "Equipment degradation detected")
    actions  = report.get("action_recommendations", [])
    first_action = actions[0]["action"] if actions else "Schedule inspection"
    conf     = report.get("diagnostic_confidence", 0.5)
    conf_word = "high" if conf > 0.78 else ("moderate" if conf > 0.55 else "low")

    return f"""MAINTENANCE ALERT SUMMARY — EXPLAIN IN PLAIN ENGLISH

Station:          {alert.get('station_id')}
Urgency:          {urgency}
Predicted RUL:    {rul:.1f} cycles (1 cycle ≈ 1 operational hour)
Affected system:  {subsys}
Fault hypothesis: {hyp}
Diagnosis conf:   {conf:.1%} ({conf_word})
Top action:       {first_action}
SLA window:       {alert.get('sla_hours', 48)} hours

Evidence summary (from maintenance documents):
{report.get('root_cause_primary','')[:400]}

Write a plain-English explanation for a mixed technical/non-technical audience.
JSON fields: headline, business_impact, recommended_action, confidence_plain, full_explanation"""


# ── LangChain path ────────────────────────────────────────────────────────

def _explain_langchain(alert: dict, report: dict) -> dict:
    """Uses LangChain LCEL chain: ChatPromptTemplate | ChatAnthropic | JsonOutputParser"""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    llm = ChatAnthropic(
        model=MODEL_ID, api_key=API_KEY,
        max_tokens=600, temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  "{user_input}"),
    ])

    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"user_input": _build_prompt(alert, report)})
    result["engine"] = "langchain"
    return result


# ── Direct Anthropic API path (urllib, no extra deps) ─────────────────────

def _explain_anthropic_direct(alert: dict, report: dict) -> dict:
    import urllib.request
    payload = json.dumps({
        "model": MODEL_ID, "max_tokens": 600,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": _build_prompt(alert, report)}]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())
        text = data["content"][0]["text"]
        text = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        result = json.loads(text)
        result["engine"] = "anthropic_direct"
        return result


# ── Rule-based fallback ───────────────────────────────────────────────────

def _explain_rule_based(alert: dict, report: dict) -> dict:
    rul     = alert.get("rul_cycles", 0)
    urgency = alert.get("urgency", "Monitor")
    subsys  = SUBSYSTEM_PLAIN.get(alert.get("primary_subsystem",""), "the equipment")
    sla     = alert.get("sla_hours", 48)
    hyp     = alert.get("fault_hypothesis", "Equipment degradation detected")
    station = alert.get("station_id", "unknown")
    conf    = report.get("diagnostic_confidence", 0.5)
    conf_word = "high" if conf > 0.78 else ("moderate" if conf > 0.55 else "low")
    actions = report.get("action_recommendations", [])
    first   = actions[0]["action"] if actions else "Schedule a preventive inspection"

    impact_template = URGENCY_BUSINESS_IMPACT[urgency]
    impact = impact_template.format(rul=rul, rul_h=rul, sla=sla)

    headline = {
        "Critical": f"⚠ Station {station} requires emergency maintenance within {sla}h",
        "Warning":  f"Station {station} showing early degradation — inspection due in {sla}h",
        "Monitor":  f"Station {station} healthy — schedule routine check within {sla}h",
    }[urgency]

    full = (
        f"The AI predictive maintenance system has detected signs of wear in {subsys} "
        f"at station {station}, with an estimated {rul:.0f} hours of remaining useful "
        f"life before a maintenance intervention is required. "
        f"The most likely cause is: {hyp.lower()}. "
        f"Diagnostic confidence is {conf_word} ({conf:.0%}). "
        f"The recommended first action is: {first.lower()}."
    )

    return {
        "headline":           headline,
        "business_impact":    impact,
        "recommended_action": first,
        "confidence_plain":   CONFIDENCE_PLAIN.get(conf_word, CONFIDENCE_PLAIN["moderate"]),
        "full_explanation":   full,
        "engine":             "rule_based",
    }


# ── Public entry point ────────────────────────────────────────────────────

def explain(alert: dict, report: dict) -> dict:
    """
    Generate a plain-English explanation of a maintenance alert.
    
    Priority order:
      1. LangChain + Claude (if USE_LLM and langchain installed)
      2. Direct Anthropic API (if USE_LLM and API key present)
      3. Rule-based template (always works offline)
    
    Returns: dict with headline, business_impact, recommended_action,
             confidence_plain, full_explanation, engine
    """
    if USE_LLM and API_KEY:
        # Try LangChain first
        try:
            return _explain_langchain(alert, report)
        except ImportError:
            pass  # langchain not installed — try direct API
        except Exception as e:
            print(f"  [LangChain] Failed: {e}. Trying direct API.")

        # Try direct Anthropic API
        try:
            return _explain_anthropic_direct(alert, report)
        except Exception as e:
            print(f"  [Anthropic direct] Failed: {e}. Using rule-based fallback.")

    # Rule-based fallback
    return _explain_rule_based(alert, report)


def explain_batch(alerts: list, reports: list) -> list:
    """Explain a batch of alerts. reports can be dicts or DiagnosticReport objects."""
    results = []
    for alert, report in zip(alerts, reports):
        a = alert.__dict__ if hasattr(alert, "__dict__") else alert
        r = report.__dict__ if hasattr(report, "__dict__") else report
        results.append(explain(a, r))
    return results


# ── Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_alert = {
        "station_id": "FD002_47", "urgency": "Critical", "rul_cycles": 14.7,
        "sla_hours": 4, "primary_subsystem": "power_subsystem",
        "fault_hypothesis": "Power unit degradation — voltage instability or rectifier wear",
    }
    demo_report = {
        "root_cause_primary": "Power unit degradation confirmed by SOP-PWR-001. "
            "Alarm PWR-001 (undervoltage) expected imminently per ALM-DICT-001.",
        "diagnostic_confidence": 0.880,
        "action_recommendations": [
            {"action": "Execute remote rectifier reset via OMC"},
            {"action": "Dispatch field engineer with power specialisation"},
        ],
    }
    result = explain(demo_alert, demo_report)
    print(json.dumps(result, indent=2))
