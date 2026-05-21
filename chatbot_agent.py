"""
chatbot_agent.py — OrchestrAI NOC Standalone Chatbot Agent
===========================================================
Self-contained LangChain agent — no RunnableWithMessageHistory.

Memory pattern (explicit, no magic):
    1. load  → get_windowed_messages(uid)  reads last 10 msgs from SQLite
    2. invoke → AgentExecutor receives {"input": question, "chat_history": msgs}
    3. save  → save_turn(uid, question, answer) appends to SQLite

Architecture
------------
NOCChatAgent(user_id, groq_key, ant_key, stations_fn)
  │
  ├── 4 LangChain @tool functions
  │     get_station_status(id)         live RUL, urgency, hypothesis
  │     get_fleet_summary()            counts + top 3 critical
  │     get_alarm_info(code)           PWR/COOL/RF/BKH/BBU alarm KB
  │     get_maintenance_procedure(sub) numbered field procedures
  │
  ├── Tier 1 · ChatGroq  LLaMA 3.3 70B + tools
  ├── Tier 2 · ChatAnthropic  Claude Haiku + tools
  └── Tier 3 · Rule-based KB  (no API key, always available)

Public API
----------
    agent.chat(question)  → {answer, engine, tools_used}
    agent.get_history()   → list[{role, content}]
    agent.clear()         → None
"""

from __future__ import annotations

from typing import Callable

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from chat_memory import (
    get_windowed_messages,
    save_turn,
    clear_history,
    load_history_for_display,
)

# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = (
    "You are an expert BTS maintenance engineer for OrchestrAI NOC, "
    "covering 25 West African stations (Senegal and Mali).\n\n"
    "Your role:\n"
    "- Diagnose BTS faults and interpret RUL (Remaining Useful Life) predictions\n"
    "- Provide clear, actionable maintenance recommendations\n"
    "- Use your tools to look up live station status and alarm codes\n"
    "- Cite document references [DOC-ID] when referencing SOPs or manuals\n\n"
    "Ground rules:\n"
    "- Always call a tool when asked about a specific station or alarm code\n"
    "- Be concise and field-engineer-friendly\n"
    "- Urgency: Critical ≤ 20 cycles (SLA 4 h), Warning ≤ 50 (SLA 48 h), Monitor > 50\n"
    "- If you don't know, say so — never guess"
)

# ─────────────────────────────────────────────────────────────────────────────
#  KNOWLEDGE BASES  (shared by tools and rule-based fallback)
# ─────────────────────────────────────────────────────────────────────────────

ALARM_KB: dict[str, str] = {
    "PWR-001": "DC undervoltage (<44 V). Cause: mains failure, rectifier fault, MCB tripped. "
               "Action: query OMC → remote reset → dispatch if unresolved 30 min. [SOP-PWR-001]",
    "PWR-003": "Rectifier module failure. Swap module, test battery backup. SLA 4 h. [MAN-PWR-001]",
    "PWR-004": "Mains power failure. Activate generator, check MCB/fuse. [SOP-PWR-001]",
    "COOL-001": "Fan < 2 000 RPM — Critical. Reduce TX 50 %, dispatch for fan replacement. SLA 4 h. [SOP-COOL-001]",
    "COOL-002": "Cabinet temp > 60 °C — Warning. Check airflow, clean filters. SLA 48 h. [SOP-COOL-001]",
    "COOL-003": "Cabinet temp > 70 °C — shutdown imminent. Immediate dispatch. SLA 4 h. [SOP-COOL-001]",
    "RF-001":   "VSWR > 2.0. Inspect feeder connectors for corrosion. Run PIM test. [SOP-RF-001]",
    "RF-002":   "PA output power low. Check PA bias, feeder loss, antenna gain. [MAN-RF-001]",
    "BKH-001":  "Backhaul latency high. Check fiber splices (OTDR) or microwave alignment. [SOP-BKH-001]",
    "BKH-002":  "Throughput low. Verify capacity reservation and QoS policy. [SOP-BKH-001]",
    "BKH-003":  "Fade margin < 10 dB. Re-align dish, inspect waveguide and ODU. [SOP-BKH-001]",
    "BBU-003":  "CPU overload. Check capacity license vs active users. [SOP-BBU-001]",
    "BBU-MEM-001": "Memory pressure. Release unused capacity, check for SW memory leaks. [SOP-BBU-001]",
}

PROCEDURE_KB: dict[str, str] = {
    "power": (
        "Power subsystem [SOP-PWR-001]\n"
        "1. Query OMC for active rectifier alarms\n"
        "2. Remote reset rectifier module\n"
        "3. If unresolved after 30 min → dispatch\n"
        "4. On-site: check MCB, replace rectifier, load-test battery\n"
        "5. Verify DC bus 47.5–51.5 V\n"
        "Fix: 2–4 h · SLA 4 h Critical"
    ),
    "thermal": (
        "Thermal/cooling [SOP-COOL-001]\n"
        "1. Reduce TX 50 % if temp > 60 °C\n"
        "2. Check fan RPM via OMC (alarm threshold: 2 000 RPM)\n"
        "3. Dispatch if RPM < 2 000 or temp > 70 °C\n"
        "4. On-site: replace fan, clean filters, inspect HVAC\n"
        "Fix: 1–3 h · SLA 4 h Critical / 48 h Warning"
    ),
    "rf": (
        "RF/Antenna [SOP-RF-001]\n"
        "1. Check VSWR via OMC (threshold: 2.0)\n"
        "2. Run remote PIM test\n"
        "3. If VSWR > 2.0 → dispatch\n"
        "4. Inspect feeder connectors and DIN connections for corrosion\n"
        "5. Re-align antenna if mechanical damage suspected\n"
        "Fix: 3–6 h · SLA 48 h Warning"
    ),
    "backhaul": (
        "Backhaul connectivity [SOP-BKH-001]\n"
        "1. Check latency and throughput via NMS\n"
        "2. Test fiber with OTDR\n"
        "3. Check microwave RSSI and fade margin (target ≥ 20 dB)\n"
        "4. If < 10 dB → re-align dish\n"
        "5. Inspect SFP modules and ODU\n"
        "Fix: 2–5 h · SLA 48 h Warning"
    ),
    "baseband": (
        "Baseband/BBU [SOP-BBU-001]\n"
        "1. Check CPU and memory utilisation via OMC\n"
        "2. Remote reboot BBU if utilisation > 90 %\n"
        "3. Apply available SW patch\n"
        "4. Verify capacity license vs active user count\n"
        "5. On-site: replace BBU board if hardware fault confirmed\n"
        "Fix: 1–3 h · SLA 48 h Warning"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  RULE-BASED FALLBACK (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based(question: str) -> str:
    q = question.lower()
    for code, text in ALARM_KB.items():
        if code.lower() in q:
            return f"**{code}** — {text}"
    if any(k in q for k in ("pwr", "power", "rectifier", "battery", "mains")):
        return ("**Power alarms** — PWR-001 (undervoltage), PWR-003 (rectifier fail), "
                "PWR-004 (mains fail). Check rectifier modules and battery backup. [SOP-PWR-001]")
    if any(k in q for k in ("cool", "thermal", "temp", "fan", "hvac")):
        return ("**Thermal alarms** — COOL-001 (fan < 2 000 RPM, Critical), COOL-002 (> 60 °C), "
                "COOL-003 (> 70 °C, shutdown). Reduce TX 50 % on COOL-001. [SOP-COOL-001]")
    if any(k in q for k in ("rf", "antenna", "vswr", "pim", "feeder", "pa ")):
        return ("**RF alarms** — RF-001 (VSWR > 2.0), RF-002 (PA power low). "
                "Inspect feeder connectors for corrosion, run PIM test. [SOP-RF-001]")
    if any(k in q for k in ("bkh", "backhaul", "microwave", "fiber", "fibre", "latency")):
        return ("**Backhaul alarms** — BKH-001 (latency), BKH-002 (throughput), "
                "BKH-003 (fade margin < 10 dB). Check fiber splices or re-align dish. [SOP-BKH-001]")
    if any(k in q for k in ("bbu", "baseband", "cpu", "memory")):
        return ("**Baseband alarms** — BBU-003 (CPU overload), BBU-MEM-001 (memory). "
                "Check capacity license vs user count. [SOP-BBU-001]")
    if any(k in q for k in ("rul", "remaining useful life", "urgency", "critical", "warning")):
        return ("**RUL urgency tiers** — Critical ≤ 20 cycles (SLA 4 h, immediate dispatch), "
                "Warning ≤ 50 (SLA 48 h), Monitor > 50 (SLA 168 h). "
                "Phase 2 Ensemble+BC model, RMSE 15.11.")
    if any(k in q for k in ("itu", "g.826", "esr", "ber", "threshold")):
        return ("**ITU-T G.826** — ESR (Errored Second Ratio) ≤ 0.04 for 64 kbit/s paths. "
                "BER threshold for PDH: 10⁻³ for degraded minutes. Monitor via NMS counters.")
    if any(k in q for k in ("fleet", "all station", "summary", "overview")):
        return ("Use 'Show me fleet status' to trigger the fleet tool, or go to "
                "**Fleet Overview** / **Live Fleet Monitor** for real-time data.")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────────────────────────────────────

class NOCChatAgent:
    """
    Standalone NOC chatbot agent.

    Parameters
    ----------
    user_id     : str   — SQLite file name; isolates each user's history
    groq_key    : str   — Groq API key (Tier 1, optional)
    ant_key     : str   — Anthropic API key (Tier 2, optional)
    stations_fn : ()→list[dict]
                  Called fresh on every tool invoke to get live station data.
                  Each dict must have: id, city, country, rul, urgency, sub, hyp
    """

    def __init__(
        self,
        user_id: str,
        groq_key: str = "",
        ant_key: str = "",
        stations_fn: Callable[[], list[dict]] | None = None,
    ) -> None:
        self.user_id     = user_id
        self.groq_key    = groq_key
        self.ant_key     = ant_key
        self.stations_fn = stations_fn or (lambda: [])

    # ── Tools ──────────────────────────────────────────────────────────────

    def _make_tools(self) -> list:
        sf = self.stations_fn  # captured by closure; fresh call each time

        @tool
        def get_station_status(station_id: str) -> str:
            """Get current RUL, urgency, subsystem and fault hypothesis for a BTS station.
            Pass the station ID exactly, e.g. FD002_47, ML_BKO_15, SN_DKR_08."""
            stations = sf()
            s = next((x for x in stations if x.get("id") == station_id.strip()), None)
            if not s:
                sample = ", ".join(x["id"] for x in stations[:5])
                return f"Station '{station_id}' not found. Sample IDs: {sample}…"
            urg  = s.get("urgency", "?")
            rul  = s.get("rul", 0)
            sla  = {"Critical": "4 h", "Warning": "48 h"}.get(urg, "168 h")
            return (
                f"Station {station_id} | {s.get('city','')}, {s.get('country','')}\n"
                f"  Urgency  : {urg}  (SLA {sla})\n"
                f"  RUL      : {rul:.1f} cycles\n"
                f"  Subsystem: {s.get('sub','').replace('_',' ')}\n"
                f"  Hypothesis: {s.get('hyp','N/A')}"
            )

        @tool
        def get_fleet_summary() -> str:
            """Summarise all 25 West African BTS stations: urgency counts,
            mean RUL, and the three most critical stations."""
            stations = sf()
            if not stations:
                return "Fleet data not available."
            crit = [s for s in stations if s.get("urgency") == "Critical"]
            warn = [s for s in stations if s.get("urgency") == "Warning"]
            mon  = [s for s in stations if s.get("urgency") == "Monitor"]
            mean_rul = sum(s.get("rul", 0) for s in stations) / len(stations)
            top3 = sorted(crit, key=lambda x: x.get("rul", 999))[:3]
            top_str = " | ".join(
                f"{s['id']} ({s.get('city','')}) RUL={s.get('rul',0):.1f}" for s in top3
            ) or "none"
            return (
                f"Fleet ({len(stations)} stations):\n"
                f"  🔴 Critical : {len(crit)}\n"
                f"  🟡 Warning  : {len(warn)}\n"
                f"  🟢 Monitor  : {len(mon)}\n"
                f"  Mean RUL   : {mean_rul:.1f} cycles\n"
                f"  Most urgent: {top_str}"
            )

        @tool
        def get_alarm_info(alarm_code: str) -> str:
            """Look up description and corrective actions for a BTS alarm code.
            Examples: PWR-001, COOL-003, RF-002, BKH-001, BBU-003."""
            code = alarm_code.upper().strip()
            if code in ALARM_KB:
                return f"[{code}] {ALARM_KB[code]}"
            prefix = code.split("-")[0]
            matches = [(k, v) for k, v in ALARM_KB.items() if k.startswith(prefix)]
            if matches:
                return "\n".join(f"[{k}] {v}" for k, v in matches[:4])
            return (
                f"'{alarm_code}' not found. "
                f"Available codes: {', '.join(ALARM_KB.keys())}"
            )

        @tool
        def get_maintenance_procedure(subsystem: str) -> str:
            """Return a numbered field procedure for a BTS subsystem.
            Accepts: power, thermal, rf, backhaul, baseband (partial names ok)."""
            key = subsystem.lower().strip()
            for sub_key, proc in PROCEDURE_KB.items():
                if sub_key in key or key in sub_key:
                    return proc
            return (
                f"Subsystem '{subsystem}' not recognised. "
                f"Available: {', '.join(PROCEDURE_KB.keys())}"
            )

        return [get_station_status, get_fleet_summary, get_alarm_info, get_maintenance_procedure]

    # ── Prompt & executor ─────────────────────────────────────────────────

    def _make_executor(self, llm) -> AgentExecutor:
        tools  = self._make_tools()
        prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True,
        )

    # ── Core invoke (explicit history, no RunnableWithMessageHistory) ──────

    def _invoke(self, llm, question: str, context: list) -> tuple[str, list[str]]:
        """
        Run the AgentExecutor with an explicit chat_history list.
        Returns (answer_text, list_of_tool_names_called).
        """
        executor = self._make_executor(llm)
        result   = executor.invoke({
            "input"       : question,
            "chat_history": context,       # list[BaseMessage] — windowed from DB
        })
        answer     = (result.get("output") or "").strip()
        tools_used = [
            step[0].tool
            for step in result.get("intermediate_steps", [])
            if hasattr(step[0], "tool")
        ]
        return answer, tools_used

    # ── Public API ─────────────────────────────────────────────────────────

    def chat(self, question: str) -> dict:
        """
        Process a question through the 3-tier pipeline.

        Flow
        ----
        1. Load last 10 messages from SQLite  →  context list
        2. Try Groq → Anthropic → Rule-based
        3. Save the new (question, answer) pair to SQLite
        4. Return {answer, engine, tools_used}
        """
        # Step 1 — load windowed context (never triggers a rerun; pure DB read)
        context    = get_windowed_messages(self.user_id)
        answer     = None
        engine     = "Rule-based KB"
        tools_used : list[str] = []
        groq_error : str | None = None

        # Step 2a — Tier 1: Groq LLaMA 3.3 70B
        if self.groq_key and len(self.groq_key) > 10:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    api_key=self.groq_key,
                    model="llama-3.3-70b-versatile",
                    max_tokens=700,
                    temperature=0.7,
                )
                answer, tools_used = self._invoke(llm, question, context)
                engine = "LLaMA 3.3 70B · Groq"
                if tools_used:
                    engine += f" + {', '.join(dict.fromkeys(tools_used))}"
            except Exception as exc:
                groq_error = str(exc)[:140]

        # Step 2b — Tier 2: Claude Haiku
        if not answer and self.ant_key:
            try:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    api_key=self.ant_key,
                    model="claude-haiku-4-5-20251001",
                    max_tokens=700,
                    temperature=0.7,
                )
                answer, tools_used = self._invoke(llm, question, context)
                engine = "Claude Haiku · Anthropic"
                if tools_used:
                    engine += f" + {', '.join(dict.fromkeys(tools_used))}"
            except Exception:
                pass

        # Step 2c — Tier 3: Rule-based KB (always works)
        if not answer:
            rb     = _rule_based(question)
            answer = rb if rb else (
                "No specific rule matched. Ask about alarm codes (PWR, COOL, RF, BKH, BBU), "
                "RUL urgency tiers, station status, or maintenance procedures."
            )
            engine     = "Rule-based KB"
            tools_used = []
            if groq_error:
                answer = f"⚠ {groq_error}\n\n{answer}"

        # Step 3 — persist to SQLite (single write point, always executed)
        save_turn(self.user_id, question, answer)

        return {"answer": answer, "engine": engine, "tools_used": tools_used}

    def get_history(self) -> list[dict]:
        """Return [{role, content}] for UI rendering (loaded from SQLite)."""
        return load_history_for_display(self.user_id)

    def clear(self) -> None:
        """Wipe this user's SQLite history."""
        clear_history(self.user_id)
