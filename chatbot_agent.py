"""
chatbot_agent.py — OrchestrAI NOC Standalone Chatbot Agent
===========================================================
A self-contained LangChain agent that the Streamlit page imports as a single
object.  All LLM, tool, memory, and fallback logic lives here.

Architecture
------------
NOCChatAgent
  ├── Tools (LangChain @tool)
  │     get_station_status(station_id)       — live RUL, urgency, subsystem
  │     get_fleet_summary()                  — fleet-wide urgency + top critical
  │     get_alarm_info(alarm_code)           — PWR/COOL/RF/BKH/BBU alarm KB
  │     get_maintenance_procedure(subsystem) — step-by-step field procedures
  │
  ├── Tier 1 · ChatGroq  (LLaMA 3.3 70B, tool-calling, windowed memory)
  ├── Tier 2 · ChatAnthropic (Claude Haiku, tool-calling, same memory)
  └── Tier 3 · Rule-based KB  (always available, no API key needed)

Memory
------
  Per-user SQLite at data/chat_history/{uid}.db via chat_memory.py.
  Last 10 messages (5 turns) injected into every request.

Usage (from Streamlit page)
---------------------------
    agent = NOCChatAgent(
        user_id   = UID,
        groq_key  = st.session_state.get("_groq_key", ""),
        ant_key   = _get_ant_key(),
        stations_fn = lambda: [{"id":..., "rul":..., ...} for each station]
    )
    result = agent.chat("What does COOL-003 mean?")
    # result = {"answer": "...", "engine": "...", "tools_used": [...]}

    history = agent.get_history()   # list[{"role", "content", "engine"}]
    agent.clear()                   # wipe SQLite + resets display
"""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

from chat_memory import get_windowed_history, load_history_for_display, clear_history

# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are an expert BTS maintenance engineer for OrchestrAI NOC, \
covering 25 West African stations (Senegal and Mali).

Your role:
- Diagnose BTS faults and interpret RUL (Remaining Useful Life) predictions
- Provide clear, actionable maintenance recommendations
- Look up live station status and alarm codes using your tools
- Cite document references [DOC-ID] when referencing SOPs or manuals

Ground rules:
- Always use tools when asked about a specific station or alarm code
- Be concise and field-engineer-friendly
- Urgency tiers: Critical ≤ 20 cycles (SLA 4 h), Warning ≤ 50 cycles (SLA 48 h), Monitor > 50
- When you don't know, say so clearly rather than guessing"""

# ─────────────────────────────────────────────────────────────────────────────
#  ALARM KNOWLEDGE BASE  (used by tool + rule-based fallback)
# ─────────────────────────────────────────────────────────────────────────────

ALARM_KB: dict[str, str] = {
    "PWR-001": "DC undervoltage (<44 V). Cause: mains failure, rectifier fault, MCB tripped. "
               "Action: query OMC rectifier → remote reset → dispatch if unresolved in 30 min. [SOP-PWR-001]",
    "PWR-003": "Rectifier module failure. Swap rectifier module, test battery backup. SLA: 4 h Critical. [MAN-PWR-001]",
    "PWR-004": "Mains power failure. Activate generator. Check MCB/fuse. [SOP-PWR-001]",
    "COOL-001": "Cooling fan < 2 000 RPM — Critical. Reduce TX power 50 %, dispatch for fan replacement immediately. SLA: 4 h. [SOP-COOL-001]",
    "COOL-002": "Cabinet temperature > 60 °C — Warning. Check airflow, clean filters. SLA: 48 h. [SOP-COOL-001]",
    "COOL-003": "Cabinet temperature > 70 °C — shutdown imminent. Immediate dispatch required. SLA: 4 h. [SOP-COOL-001]",
    "RF-001":   "VSWR > 2.0. Inspect antenna feeder connectors for corrosion/damage. Run PIM test. [SOP-RF-001]",
    "RF-002":   "PA output power low. Check PA bias, feeder insertion loss, antenna gain. [MAN-RF-001]",
    "BKH-001":  "Backhaul latency high. Check fiber splices (OTDR) or microwave alignment. [SOP-BKH-001]",
    "BKH-002":  "Backhaul throughput low. Verify capacity reservation and QoS policy. [SOP-BKH-001]",
    "BKH-003":  "Microwave fade margin < 10 dB. Re-align dish, inspect waveguide and ODU. [SOP-BKH-001]",
    "BBU-003":  "BBU CPU overload. Check capacity license vs active users. Consider load balancing. [SOP-BBU-001]",
    "BBU-MEM-001": "BBU memory pressure. Release unused capacity, check for software memory leaks. [SOP-BBU-001]",
}

PROCEDURE_KB: dict[str, str] = {
    "power": (
        "Power subsystem procedure [SOP-PWR-001]:\n"
        "1. Query OMC for active rectifier alarms\n"
        "2. Remote reset rectifier module\n"
        "3. If unresolved after 30 min → dispatch field engineer\n"
        "4. On-site: check MCB, replace rectifier module, test battery backup (load test)\n"
        "5. Verify DC bus voltage 47.5–51.5 V\n"
        "Estimated fix: 2–4 h · SLA: 4 h Critical"
    ),
    "thermal": (
        "Thermal/cooling procedure [SOP-COOL-001]:\n"
        "1. Reduce TX power 50 % if temperature > 60 °C\n"
        "2. Check fan RPM via OMC (threshold: 2 000 RPM)\n"
        "3. Dispatch if RPM < 2 000 or temperature > 70 °C\n"
        "4. On-site: replace failed fan, clean air filters, inspect HVAC unit\n"
        "Estimated fix: 1–3 h · SLA: 4 h Critical / 48 h Warning"
    ),
    "rf": (
        "RF/Antenna procedure [SOP-RF-001]:\n"
        "1. Check VSWR via OMC (alarm threshold: 2.0)\n"
        "2. Run remote PIM test\n"
        "3. If VSWR > 2.0 → dispatch\n"
        "4. On-site: inspect feeder connectors and DIN connections for corrosion\n"
        "5. Re-align antenna if mechanical damage suspected\n"
        "Estimated fix: 3–6 h · SLA: 48 h Warning"
    ),
    "backhaul": (
        "Backhaul connectivity procedure [SOP-BKH-001]:\n"
        "1. Check latency and throughput via NMS\n"
        "2. Test fiber continuity with OTDR\n"
        "3. Check microwave RSSI and fade margin (target ≥ 20 dB)\n"
        "4. If fade margin < 10 dB → re-align microwave dish\n"
        "5. Inspect SFP modules and ODU for hardware faults\n"
        "Estimated fix: 2–5 h · SLA: 48 h Warning"
    ),
    "baseband": (
        "Baseband/BBU procedure [SOP-BBU-001]:\n"
        "1. Check CPU and memory utilisation via OMC\n"
        "2. Remote reboot BBU if utilisation > 90 %\n"
        "3. Apply available software patch\n"
        "4. Verify capacity license matches active user count\n"
        "5. On-site: replace BBU board if hardware fault confirmed\n"
        "Estimated fix: 1–3 h · SLA: 48 h Warning"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  RULE-BASED FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based(question: str) -> str:
    q = question.lower()
    # Alarm code lookup
    for code, text in ALARM_KB.items():
        if code.lower() in q:
            return f"**{code}** — {text}"
    # Subsystem keywords
    if any(k in q for k in ("pwr", "power", "rectifier", "battery", "mains")):
        return ("**Power alarms** — PWR-001 (undervoltage), PWR-003 (rectifier fail), "
                "PWR-004 (mains fail). Check rectifier modules and battery backup. [SOP-PWR-001]")
    if any(k in q for k in ("cool", "thermal", "temp", "fan", "hvac")):
        return ("**Thermal alarms** — COOL-001 (fan < 2 000 RPM, Critical), COOL-002 (> 60 °C), "
                "COOL-003 (> 70 °C, shutdown). Reduce TX 50 % on COOL-001. [SOP-COOL-001]")
    if any(k in q for k in ("rf", "antenna", "vswr", "pim", "feeder", "pa ")):
        return ("**RF alarms** — RF-001 (VSWR > 2.0), RF-002 (PA power low). "
                "Check feeder connectors for corrosion, run PIM test. [SOP-RF-001]")
    if any(k in q for k in ("bkh", "backhaul", "microwave", "fiber", "fibre", "latency")):
        return ("**Backhaul alarms** — BKH-001 (latency), BKH-002 (throughput), "
                "BKH-003 (fade margin < 10 dB). Check fiber splices or microwave alignment. [SOP-BKH-001]")
    if any(k in q for k in ("bbu", "baseband", "cpu", "memory")):
        return ("**Baseband alarms** — BBU-003 (CPU overload), BBU-MEM-001 (memory pressure). "
                "Check capacity license vs user count. [SOP-BBU-001]")
    if any(k in q for k in ("rul", "remaining useful life", "urgency", "critical", "warning")):
        return ("**RUL urgency tiers**: Critical ≤ 20 cycles (SLA 4 h, immediate dispatch), "
                "Warning ≤ 50 cycles (SLA 48 h), Monitor > 50 cycles (SLA 168 h). "
                "Predictions from Phase 2 Ensemble+BC model (RMSE 15.11).")
    if any(k in q for k in ("itu", "g.826", "esr", "ber", "threshold")):
        return ("**ITU-T G.826** — ESR (Errored Second Ratio): ≤ 0.04 for 64 kbit/s paths. "
                "BER threshold for PDH: 10⁻³ for degraded minutes. Monitor via NMS counters.")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class NOCChatAgent:
    """
    Standalone NOC chatbot agent.

    Parameters
    ----------
    user_id     : str  — used as the SQLite file name for conversation history
    groq_key    : str  — Groq API key (Tier 1)
    ant_key     : str  — Anthropic API key (Tier 2)
    stations_fn : callable → list[dict]
                  Called on every tool invocation to get live station data.
                  Expected keys per dict: id, city, country, rul, urgency, sub, hyp
    """

    def __init__(
        self,
        user_id: str,
        groq_key: str = "",
        ant_key: str = "",
        stations_fn: Callable[[], list[dict]] | None = None,
    ):
        self.user_id     = user_id
        self.groq_key    = groq_key
        self.ant_key     = ant_key
        self.stations_fn = stations_fn or (lambda: [])

    # ── Tools ──────────────────────────────────────────────────────────────

    def _make_tools(self) -> list:
        """Build LangChain tools that close over live station data."""
        sf = self.stations_fn  # capture reference; called fresh on each tool invoke

        @tool
        def get_station_status(station_id: str) -> str:
            """Get current RUL, urgency tier, subsystem, and fault hypothesis for a
            specific BTS station by ID (e.g. FD002_47, ML_BKO_15, SN_DKR_08)."""
            stations = sf()
            s = next((x for x in stations if x.get("id") == station_id.strip()), None)
            if not s:
                ids = ", ".join(x["id"] for x in stations[:6])
                return f"Station '{station_id}' not found. Sample IDs: {ids}…"
            urg   = s.get("urgency", "?")
            rul   = s.get("rul", 0)
            sub   = s.get("sub", "").replace("_", " ")
            hyp   = s.get("hyp", "N/A")
            city  = s.get("city", "")
            cntry = s.get("country", "")
            sla   = "4 h" if urg == "Critical" else ("48 h" if urg == "Warning" else "168 h")
            return (
                f"Station {station_id} | {city}, {cntry}\n"
                f"  Urgency : {urg} (SLA {sla})\n"
                f"  RUL     : {rul:.1f} cycles\n"
                f"  Subsystem: {sub}\n"
                f"  Hypothesis: {hyp}"
            )

        @tool
        def get_fleet_summary() -> str:
            """Return a summary of all 25 West African BTS stations: urgency counts,
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
                f"Fleet status ({len(stations)} stations):\n"
                f"  🔴 Critical : {len(crit)}\n"
                f"  🟡 Warning  : {len(warn)}\n"
                f"  🟢 Monitor  : {len(mon)}\n"
                f"  Mean RUL   : {mean_rul:.1f} cycles\n"
                f"  Most urgent: {top_str}"
            )

        @tool
        def get_alarm_info(alarm_code: str) -> str:
            """Look up description and corrective actions for a BTS alarm code.
            Accepts codes like PWR-001, COOL-003, RF-002, BKH-001, BBU-003."""
            code = alarm_code.upper().strip()
            # Exact match
            if code in ALARM_KB:
                return f"[{code}] {ALARM_KB[code]}"
            # Partial / prefix match
            matches = [(k, v) for k, v in ALARM_KB.items()
                       if code in k or k.startswith(code.split("-")[0])]
            if matches:
                return "\n".join(f"[{k}] {v}" for k, v in matches[:4])
            return (
                f"Alarm code '{alarm_code}' not in KB. "
                f"Available: {', '.join(ALARM_KB.keys())}"
            )

        @tool
        def get_maintenance_procedure(subsystem: str) -> str:
            """Return a step-by-step field maintenance procedure for a BTS subsystem.
            Accepts: power, thermal, rf, backhaul, baseband (or partial names)."""
            key = subsystem.lower().strip()
            for sub_key, proc in PROCEDURE_KB.items():
                if sub_key in key or key in sub_key:
                    return proc
            return (
                f"Subsystem '{subsystem}' not recognised. "
                f"Available: {', '.join(PROCEDURE_KB.keys())}"
            )

        return [get_station_status, get_fleet_summary, get_alarm_info, get_maintenance_procedure]

    # ── Chain / executor builders ──────────────────────────────────────────

    @staticmethod
    def _make_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

    def _build_executor(self, llm) -> AgentExecutor:
        tools = self._make_tools()
        agent = create_tool_calling_agent(llm, tools, self._make_prompt())
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True,
        )

    def _history_factory(self):
        uid = self.user_id
        return lambda _session_id: get_windowed_history(uid)

    def _run_executor(self, llm, question: str) -> tuple[str, list[str]]:
        """Invoke the AgentExecutor with memory. Returns (answer, tools_used)."""
        executor = self._build_executor(llm)
        runnable = RunnableWithMessageHistory(
            executor,
            self._history_factory(),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output",
        )
        result = runnable.invoke(
            {"input": question},
            config={"configurable": {"session_id": "main"}},
        )
        answer = result.get("output", "").strip()
        tools_used = [
            step[0].tool
            for step in result.get("intermediate_steps", [])
            if hasattr(step[0], "tool")
        ]
        return answer, tools_used

    # ── Public API ─────────────────────────────────────────────────────────

    def chat(self, question: str) -> dict:
        """
        Process a question through the 3-tier agent pipeline.

        Returns
        -------
        dict with keys:
            answer     : str   — response text
            engine     : str   — which tier answered
            tools_used : list  — tool names called (empty for rule-based)
        """
        answer: str | None = None
        engine = "Rule-based KB"
        tools_used: list[str] = []
        groq_error: str | None = None

        # ── Tier 1 : Groq LLaMA 3.3 70B + tools ─────────────────────────
        if self.groq_key and len(self.groq_key) > 10:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    api_key=self.groq_key,
                    model="llama-3.3-70b-versatile",
                    max_tokens=700,
                    temperature=0.7,
                )
                answer, tools_used = self._run_executor(llm, question)
                engine = "LLaMA 3.3 70B · Groq"
                if tools_used:
                    engine += f" + {', '.join(set(tools_used))}"
            except Exception as exc:
                groq_error = str(exc)[:140]

        # ── Tier 2 : Claude Haiku + tools ────────────────────────────────
        if not answer and self.ant_key:
            try:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    api_key=self.ant_key,
                    model="claude-haiku-4-5-20251001",
                    max_tokens=700,
                    temperature=0.7,
                )
                answer, tools_used = self._run_executor(llm, question)
                engine = "Claude Haiku · Anthropic"
                if tools_used:
                    engine += f" + {', '.join(set(tools_used))}"
            except Exception:
                pass

        # ── Tier 3 : Rule-based KB (always available) ─────────────────────
        if not answer:
            rb = _rule_based(question)
            answer = rb if rb else (
                "No specific rule matched. Ask about alarm codes (PWR, COOL, RF, BKH, BBU), "
                "RUL urgency tiers, or maintenance procedures."
            )
            engine = "Rule-based KB"
            tools_used = []
            if groq_error:
                answer = f"⚠ {groq_error}\n\n{answer}"
            # Manually persist to SQLite since we bypassed LangChain
            from langchain_core.messages import HumanMessage, AIMessage
            from chat_memory import get_history as _gh
            h = _gh(self.user_id)
            h.add_messages([HumanMessage(content=question), AIMessage(content=answer)])

        return {"answer": answer, "engine": engine, "tools_used": tools_used}

    def get_history(self) -> list[dict]:
        """Return conversation history as list of {role, content} dicts for UI rendering."""
        return load_history_for_display(self.user_id)

    def clear(self) -> None:
        """Wipe the SQLite conversation history for this user."""
        clear_history(self.user_id)
