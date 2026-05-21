"""
agents.py — OrchestrAI NOC
Lightweight agent module: Interpreter, Diagnostic, Planning, Execution agents.
No heavy ML dependencies — works standalone for UI purposes.
Full pipeline agents (interpreter_agent.py etc.) take precedence if available.
"""

import json
import time
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    station_id:        str
    urgency:           str
    rul:               float
    ci_low:            float
    ci_high:           float
    subsystem:         str
    top_feature:       str
    top_importance:    float
    confidence:        float
    hyp:               str
    ts:                str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class DiagnosticOutput:
    station_id:     str
    urgency:        str
    hypothesis:     str
    fault_component:str
    alarm_code:     str
    fault_mechanism:str
    confidence:     float
    grounding_rate: float
    hallucination:  float
    evidence_ids:   list
    top_feature:    str
    alternatives:   list
    ts:             str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class ActionPlan:
    station_id:     str
    urgency:        str
    sla_hours:      int
    actions:        list       # list of {seq, description, tier, tool, cost}
    governance_tier:int
    total_cost_eur: float
    ts:             str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


# ─────────────────────────────────────────────────────────────────────────────
#  SUBSYSTEM KNOWLEDGE BASE (rules, alarm codes, components)
# ─────────────────────────────────────────────────────────────────────────────

SUBSYSTEM_KB = {
    "power_subsystem": {
        "alarm_codes":   ["PWR-001","PWR-003","PWR-004"],
        "components":    ["48V DC rectifier module","VRLA battery string","MCB distribution"],
        "mechanisms":    ["Rectifier voltage decay below 44V","Battery capacity decline","MCB trip on overload"],
        "sop":           "SOP-PWR-001",
        "tier_critical": 3, "tier_warning": 2, "tier_monitor": 1,
    },
    "thermal_management": {
        "alarm_codes":   ["COOL-001","COOL-002","COOL-003"],
        "components":    ["Cooling fan bearing assembly","Heat exchanger unit","Cabinet air filter"],
        "mechanisms":    ["Bearing fatigue → fan speed <2000 RPM","Heat exchanger fouling","Filter blockage"],
        "sop":           "SOP-THM-001",
        "tier_critical": 3, "tier_warning": 2, "tier_monitor": 1,
    },
    "rf_antenna": {
        "alarm_codes":   ["RF-001","RF-002","RF-004"],
        "components":    ["7/16 DIN feeder connector","Power amplifier PA stage","Antenna feeder cable"],
        "mechanisms":    ["Galvanic corrosion → VSWR >2.0","PA efficiency falling 25% below nominal","Cable delamination"],
        "sop":           "SOP-RF-001",
        "tier_critical": 3, "tier_warning": 2, "tier_monitor": 1,
    },
    "backhaul_connectivity": {
        "alarm_codes":   ["BKH-001","BKH-002","BKH-003"],
        "components":    ["Fibre splice point","Microwave ODU alignment","SFP+ transceiver"],
        "mechanisms":    ["Splice loss → latency >10ms","Azimuth drift → fade margin drop","SFP degradation"],
        "sop":           "SOP-BKH-001",
        "tier_critical": 3, "tier_warning": 2, "tier_monitor": 1,
    },
    "baseband_processing": {
        "alarm_codes":   ["BBU-003","BBU-MEM-001","BBU-MEM-002"],
        "components":    ["BBU CPU subsystem","DDR4 memory bank","eRAN processing blade"],
        "mechanisms":    ["CPU load →85% threshold","Memory leak → swap >50%","Blade processing overload"],
        "sop":           "MAN-BBU-001",
        "tier_critical": 3, "tier_warning": 2, "tier_monitor": 1,
    },
}

URGENCY_ACTIONS = {
    "Critical": [
        {"seq":1,"tier":"AUTO",   "tool":"remote_command",   "description":"Execute immediate remote protective action via OMC","cost":0},
        {"seq":2,"tier":"HUMAN",  "tool":"schedule_dispatch","description":"Emergency field dispatch — specialist + spares ≤4h SLA","cost":800},
    ],
    "Warning": [
        {"seq":1,"tier":"AUTO",   "tool":"remote_command",   "description":"Apply protective configuration change via OMC","cost":0},
        {"seq":2,"tier":"TIMEOUT","tool":"schedule_dispatch","description":"Schedule field inspection within 48h SLA","cost":600},
    ],
    "Monitor": [
        {"seq":1,"tier":"AUTO",   "tool":"open_ticket",      "description":"Open monitoring ticket — collect 7-day trend","cost":0},
        {"seq":2,"tier":"AUTO",   "tool":"query_cmdb",       "description":"Query CMDB for asset history + last PM date","cost":0},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  AGENTS
# ─────────────────────────────────────────────────────────────────────────────

class InterpreterAgent:
    """Parses raw station telemetry into a structured Alert."""

    def parse(self, station: dict, live_rul: float) -> Alert:
        urg = "Critical" if live_rul <= 20 else "Warning" if live_rul <= 50 else "Monitor"
        return Alert(
            station_id=station["id"],
            urgency=urg,
            rul=round(live_rul, 2),
            ci_low=station.get("cl", live_rul * 0.78),
            ci_high=station.get("ch", live_rul * 1.22),
            subsystem=station.get("sub","power_subsystem"),
            top_feature=station.get("top_feat","voltage_rolling_mean"),
            top_importance=station.get("top_imp", 0.07),
            confidence=station.get("conf", 0.75),
            hyp=station.get("hyp","Degradation pattern detected"),
        )


class DiagnosticAgent:
    """
    Generates root-cause diagnosis from an Alert + RAG evidence.
    Rule-based (no LLM) for offline/fallback mode.
    """

    def diagnose(self, alert: Alert, rag_chunks: list = None) -> DiagnosticOutput:
        kb = SUBSYSTEM_KB.get(alert.subsystem, SUBSYSTEM_KB["power_subsystem"])
        evidence_ids = [c.get("citation_ref","DOC-?") for c in (rag_chunks or [])][:5]
        if not evidence_ids:
            evidence_ids = [kb["sop"]]

        return DiagnosticOutput(
            station_id=alert.station_id,
            urgency=alert.urgency,
            hypothesis=alert.hyp,
            fault_component=kb["components"][0],
            alarm_code=" / ".join(kb["alarm_codes"][:2]),
            fault_mechanism=kb["mechanisms"][0],
            confidence=alert.confidence,
            grounding_rate=1.0 if evidence_ids else 0.0,
            hallucination=0.0,
            evidence_ids=evidence_ids,
            top_feature=alert.top_feature,
            alternatives=[
                {"hypothesis":"Mains supply failure","confidence":0.35},
                {"hypothesis":"End-of-life degradation","confidence":0.25},
            ],
        )


class PlanningAgent:
    """Selects and prioritises maintenance actions based on urgency + governance tier."""

    def plan(self, diag: DiagnosticOutput, sla_hours: int) -> ActionPlan:
        actions    = URGENCY_ACTIONS.get(diag.urgency, URGENCY_ACTIONS["Monitor"])
        gov_tier   = 3 if diag.urgency=="Critical" else 2 if diag.urgency=="Warning" else 1
        total_cost = sum(a["cost"] for a in actions)
        return ActionPlan(
            station_id=diag.station_id,
            urgency=diag.urgency,
            sla_hours=sla_hours,
            actions=actions,
            governance_tier=gov_tier,
            total_cost_eur=total_cost,
        )


class ExecutionAgent:
    """
    Executes approved actions, respecting governance tier.
    In production this would call real OMC/NMS APIs.
    """

    def execute(self, plan: ActionPlan, approved: bool = False) -> dict:
        results = []
        for action in plan.actions:
            tier = action["tier"]
            if tier == "AUTO":
                status = "executed"
            elif tier == "TIMEOUT" and approved:
                status = "executed"
            elif tier == "HUMAN" and approved:
                status = "executed"
            else:
                status = "pending_approval"
            results.append({**action, "status": status, "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return {
            "station_id": plan.station_id,
            "actions":    results,
            "total_executed": sum(1 for r in results if r["status"] == "executed"),
            "pipeline_latency_ms": 33,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE RUNNER (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class AgentPipeline:
    """Run the full Observe → Diagnose → Plan → Execute pipeline."""

    def __init__(self):
        self.interpreter = InterpreterAgent()
        self.diagnostic  = DiagnosticAgent()
        self.planning    = PlanningAgent()
        self.execution   = ExecutionAgent()

    def run(self, station: dict, live_rul: float,
            sla_hours: int = 48, rag_chunks: list = None,
            auto_execute: bool = False) -> dict:
        t0    = time.time()
        alert = self.interpreter.parse(station, live_rul)
        diag  = self.diagnostic.diagnose(alert, rag_chunks)
        plan  = self.planning.plan(diag, sla_hours)
        exec_ = self.execution.execute(plan, approved=auto_execute)
        latency_ms = round((time.time() - t0) * 1000, 1)
        return {
            "alert":        asdict(alert),
            "diagnostic":   asdict(diag),
            "plan":         asdict(plan),
            "execution":    exec_,
            "latency_ms":   latency_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CHATBOT RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

CHAT_RULES = {
    ("pwr-001","undervoltage","rectifier","pwr001","pwr001"):
        "<strong>PWR-001 — Rectifier Undervoltage</strong> | Critical | SLA 4h<br><br>"
        "<strong>Threshold:</strong> DC bus &lt;44V (nominal 47.5–51.5V).<br>"
        "<strong>Cause:</strong> Mains failure, rectifier fault, or MCB tripped.<br><br>"
        "<strong>Steps:</strong><br>1. Verify AC input via OMC telemetry<br>"
        "2. Remote rectifier reset → wait 5min → verify voltage<br>"
        "3. Activate generator if AC fault confirmed<br>"
        "4. Dispatch engineer if unresolved after 30min<br><br>"
        "<em>Source: [ALM-DICT-001], [SOP-PWR-001]</em>",

    ("cool-001","cool001","fan","bearing","cooling fan"):
        "<strong>COOL-001 — Cooling Fan Failure</strong> | Critical | SLA 4h<br><br>"
        "<strong>Threshold:</strong> Fan speed &lt; 2,000 RPM (nominal 3,200 RPM).<br>"
        "<strong>Immediate action:</strong> Reduce TX power 50% via OMC.<br>"
        "<strong>Bearing service interval:</strong> 40,000 operating hours.<br>"
        "<strong>Spares needed:</strong> 2× fans (replace both) + 1× air filter.<br><br>"
        "<em>Source: [ALM-DICT-003], [MAN-THM-001], [SOP-THM-001]</em>",

    ("cool-003","cool003","thermal runaway","70°c",">70"):
        "<strong>COOL-003 — Internal Temperature Critical (&gt;70°C)</strong><br><br>"
        "1. Reduce TX 50% <strong>immediately</strong><br>"
        "2. At 75°C: graceful shutdown via OMC (BBU → PA → PSU)<br>"
        "3. Do not restore until cabinet &lt;45°C sustained 15min<br>"
        "4. Inspect PCB for discoloration before restart<br><br>"
        "<em>Source: [ALM-DICT-003], [MAN-THM-002]</em>",

    ("vswr","pim","rf-001","rf001","connector","7/16"):
        "<strong>VSWR / PIM Investigation</strong><br><br>"
        "RF-001: VSWR &gt; 2.0:1 · RF-005 critical: &gt; 3.0:1<br>"
        "<strong>PIM test:</strong> 2×43W carriers → pass: &lt;−150 dBc<br>"
        "<strong>Torque:</strong> 7/16 DIN at 30 Nm; N-type at 20 Nm<br>"
        "<strong>Tools:</strong> PIM analyser, torque wrench, IPA spray, self-amalgamating tape<br><br>"
        "<em>Source: [SOP-RF-001], [MAN-RF-002], [FMEA-002]</em>",

    ("g.826","esr","backhaul","bkh","latency >10","otdr","itu"):
        "<strong>ITU-T G.826 Backhaul Thresholds</strong><br><br>"
        "ESR &lt; 4%/month · SESR &lt; 0.2%/month · BBER &lt; 3×10⁻⁴/month<br>"
        "BKH-001 triggers at latency &gt;10ms. ESR →1% → OTDR test (fault within 5m accuracy).<br>"
        "Microwave: fade margin &lt;10dB → check dish alignment (±0.1° tolerance).<br><br>"
        "<em>Source: [SPEC-ITU-001], [SOP-BKH-001]</em>",

    ("bbu","upgrade","software","cpu overload","mem","swap"):
        "<strong>BBU Software Upgrade / CPU-MEM Overload</strong><br><br>"
        "Upgrade window: 02:00–04:00 local time, &lt;20% traffic load<br>"
        "Duration: 15–20min + 30min KPI recovery window<br>"
        "Rollback: 10min via OMC (keep previous SW package on flash)<br>"
        "CPU overload: check licence capacity vs active users; restart non-critical L2 processes first.<br><br>"
        "<em>Source: [MAN-BBU-001], [SOP-BBU-001]</em>",

    ("14.7","rul 14","fd002_47","fd002_14","rmse","15.11","ensemble"):
        "<strong>RUL ≤15 cycles — CRITICAL Threshold</strong><br><br>"
        "Phase2 Ensemble+BC: All-4 RMSE=15.11 · R²=0.8663<br>"
        "CI computed via conformal prediction: CONFIDENCE_ALPHA=0.2206 (90% coverage)<br>"
        "Governance Tier 3 — human approval required for field dispatch.<br><br>"
        "<strong>Recommended actions:</strong><br>"
        "1. [AUTO] Query CMDB — verify rectifier model + last service<br>"
        "2. [AUTO] Check active alarms (PWR-001/003/004)<br>"
        "3. [TIMEOUT 6h] Dispatch power specialist + rectifier spare<br><br>"
        "<em>Phase2 Ensemble+BC · TransV2(α=0.70)+XGB(α=0.30) · C-MAPSS all 4 subsets</em>",
}


def rule_based_answer(question: str) -> Optional[str]:
    """Return a rule-based answer if a keyword matches, else None."""
    q_lo = question.lower()
    for keys, answer in CHAT_RULES.items():
        if any(k in q_lo for k in keys):
            return answer
    return None


async def call_llm_async(api_key: str, messages: list, system: str,
                          model: str = "claude-haiku-4-5-20251001",
                          max_tokens: int = 800) -> tuple[Optional[str], str]:
    """Call Anthropic Claude async. Returns (answer, engine_label)."""
    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=api_key)
        clean    = []
        for m in messages:
            txt = re.sub(r"<[^>]+>","",str(m["content"])).strip()
            if txt and m["role"] in ("user","assistant"):
                if clean and clean[-1]["role"] == m["role"]:
                    clean[-1]["content"] += " " + txt
                else:
                    clean.append({"role":m["role"],"content":txt})
        if not clean or clean[0]["role"] != "user":
            clean.insert(0,{"role":"user","content":"Hello"})
        resp = client.messages.create(model=model, max_tokens=max_tokens,
                                       system=system, messages=clean)
        return resp.content[0].text, f"Claude {model} · Anthropic"
    except ImportError:
        return None, "anthropic package not installed"
    except Exception as e:
        return None, str(e)[:200]
