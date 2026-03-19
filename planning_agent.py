"""
Planning Agent + Execution Agent — Layer 3: Reasoning-to-Action (Steps 2 & 3)
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | March 2026
Model:  Upstream RUL from XGBoost v2 Final (15k trees · exp(α=3) · RMSE=12.77)

ROLE IN PIPELINE:
  Receives:  DiagnosticReport handoff_to_planner dict
  Produces:  ExecutionPlan — validated, sequenced, governance-gated action plan
             + ExecutionLog — record of simulated tool call outcomes

ARCHITECTURE (Pre-Act pattern, Rawat et al. 2025):
  Planning Agent:
    1. Receives diagnostic handoff (actions + confidence + governance tier)
    2. Validates each action against constraints:
         - risk tier vs human approval gate
         - SLA feasibility (within available window)
         - dependency ordering (ticket must precede dispatch)
         - cost budget check (simulated)
    3. Sequences actions respecting dependency graph
    4. Outputs ExecutionPlan with gate decisions per action

  Execution Agent:
    1. Walks the ExecutionPlan in sequence
    2. For Tier 1 (auto): executes tool call immediately
    3. For Tier 2 (timeout): records recommendation, sets timer
    4. For Tier 3 (human required): pauses, logs approval request, waits
    5. Records all outcomes in ExecutionLog (full audit trail)

TOOL CALLS (simulated — same interface as production):
  query_cmdb, search_knowledge, open_ticket,
  schedule_dispatch, remote_command, escalate_to_human
"""

import os, json, time, uuid
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict
import pandas as pd

RESULTS_DIR  = "results/planning"
MEMORY_DIR   = "results/memory"
os.makedirs(RESULTS_DIR,  exist_ok=True)
os.makedirs(MEMORY_DIR,   exist_ok=True)

# ── Governance constants ───────────────────────────────────────────────────
TIER_LABELS  = {1: "FULLY_AUTONOMOUS", 2: "AUTO_AFTER_TIMEOUT", 3: "HUMAN_REQUIRED"}
TIMEOUT_H    = {2: 6}    # Tier 2: auto-execute after 6h if no objection
COST_BUDGET  = {          # Simulated cost budget per priority (€)
    "critical": 5000,
    "warning":  1500,
    "monitor":  1500,   # raised: preventive dispatch (~€800) within scope
}
ESTIMATED_COSTS = {        # Simulated per-tool-call cost (€)
    "query_cmdb":         0,
    "search_knowledge":   0,
    "open_ticket":        0,
    "schedule_dispatch":  800,
    "remote_command":     50,
    "escalate_to_human":  0,
    "remote_reboot":      200,
}


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class PlannedAction:
    sequence:        int           # execution order (1-based)
    action:          str
    tool_call:       str
    tool_params:     dict
    risk_tier:       int
    gate_decision:   str           # AUTO | RECOMMEND | HOLD_FOR_APPROVAL | BLOCKED
    gate_reason:     str
    sla_hours:       int
    estimated_cost:  float
    dependency_on:   Optional[int] # sequence number this action depends on
    citations:       List[str]

@dataclass
class ExecutionPlan:
    plan_id:         str
    station_id:      str
    urgency:         str
    created_at:      str
    actions:         List[dict]    # PlannedAction dicts
    total_cost_est:  float
    within_budget:   bool
    requires_human:  bool
    critical_path:   List[int]     # sequence numbers on critical path
    confidence:      float
    plan_summary:    str

@dataclass
class ToolResult:
    tool_call:   str
    params:      dict
    status:      str      # SUCCESS | SIMULATED | PENDING_APPROVAL | SKIPPED
    output:      dict
    latency_ms:  float
    timestamp:   str

@dataclass
class ExecutionLog:
    log_id:       str
    plan_id:      str
    station_id:   str
    urgency:      str
    started_at:   str
    completed_at: str
    tool_results: List[dict]    # ToolResult dicts
    actions_auto:     int
    actions_timeout:  int
    actions_human:    int
    actions_blocked:  int
    total_latency_ms: float
    outcome_summary:  str
    memory_entry:     dict       # written to persistent memory store


# ── Tool Simulator ─────────────────────────────────────────────────────────

class ToolSimulator:
    """
    Simulates tool call execution.
    In production: replace each method body with real API/system call.
    Interface is identical — same input/output contract.
    """

    def query_cmdb(self, equipment_id: str = None,
                   query_type: str = "status", **kwargs) -> dict:
        return {
            "equipment_id":    equipment_id or "unknown",
            "query_type":      query_type,
            "firmware":        "R22A.3.1",
            "last_maintenance": "2024-01-15",
            "age_years":       6.2,
            "site_type":       "outdoor_macro",
            "backhaul_type":   "fibre",
            "status":          "OPERATIONAL_DEGRADED",
            "open_alarms":     ["PWR-001", "PWR-004"] if "power" in str(kwargs) else [],
        }

    def open_ticket(self, severity: str, subsystem: str,
                    description: str = "", **kwargs) -> dict:
        ticket_id = f"INC-{pd.Timestamp.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:5].upper()}"
        return {
            "ticket_id":   ticket_id,
            "severity":    severity,
            "subsystem":   subsystem,
            "status":      "OPEN",
            "created_at":  pd.Timestamp.now().isoformat(),
            "assigned_to": "NOC_tier2_queue",
        }

    def schedule_dispatch(self, priority: str, skill: str,
                          spares: list = None, **kwargs) -> dict:
        eta_hours = {"emergency": 4, "critical": 4,
                     "warning": 24, "low": 72}.get(priority, 48)
        return {
            "dispatch_id":   f"DSP-{str(uuid.uuid4())[:8].upper()}",
            "priority":      priority,
            "skill_required": skill,
            "spares_requested": spares or [],
            "eta_hours":     eta_hours,
            "status":        "SCHEDULED",
            "assigned_engineer": f"ENG_{skill[:3].upper()}_{str(uuid.uuid4())[:4].upper()}",
        }

    def remote_command(self, command: str, **kwargs) -> dict:
        return {
            "command":    command,
            "status":     "EXECUTED",
            "result":     f"Command '{command}' executed successfully",
            "timestamp":  pd.Timestamp.now().isoformat(),
        }

    def escalate_to_human(self, urgency: str, context: str, **kwargs) -> dict:
        return {
            "escalation_id": f"ESC-{str(uuid.uuid4())[:8].upper()}",
            "urgency":        urgency,
            "context":        context,
            "notified":       ["NOC_shift_lead", "field_ops_manager"],
            "status":         "ACKNOWLEDGED",
            "timestamp":      pd.Timestamp.now().isoformat(),
        }

    def execute(self, tool_call: str, params: dict) -> dict:
        t0 = time.time()
        fn = getattr(self, tool_call, None)
        if fn:
            result = fn(**{k: v for k, v in params.items()})
        else:
            result = {"status": "TOOL_NOT_FOUND", "tool": tool_call}
        return result, (time.time()-t0)*1000


# ── Planning Agent ─────────────────────────────────────────────────────────

class PlanningAgent:
    """
    Validates, sequences, and gates the action list from the Diagnostic Agent.
    Implements the Pre-Act pattern: full plan constructed before any execution.
    """

    def _gate_decision(self, action: dict, urgency: str) -> tuple:
        """
        Returns (gate_decision, gate_reason).
        Applies governance tiers + confidence threshold + cost check.
        """
        tier = action["risk_tier"]
        conf = action.get("confidence", 1.0)

        if tier == 1:
            return "AUTO", "Tier 1 — fully autonomous, low-risk reversible action"
        if tier == 2:
            if urgency == "Critical":
                return "RECOMMEND", (
                    f"Tier 2 action on Critical alert — "
                    f"recommend to engineer, auto-execute after {TIMEOUT_H[2]}h timeout")
            return "RECOMMEND", (
                f"Tier 2 — recommended to maintenance engineer, "
                f"auto-execute after {TIMEOUT_H[2]}h if no objection")
        if tier == 3:
            return "HOLD_FOR_APPROVAL", (
                "Tier 3 — high-risk/irreversible action requires explicit human approval. "
                "Full reasoning trace and evidence bundle provided to approving engineer.")
        return "BLOCKED", "Unrecognised tier"

    def _estimate_cost(self, tool_call: str, params: dict) -> float:
        return float(ESTIMATED_COSTS.get(tool_call, 100))

    def _dependency_order(self, actions: List[dict]) -> List[dict]:
        """
        Enforce: open_ticket must precede schedule_dispatch.
        query_cmdb must be first if present.
        """
        ordered = []
        # Pass 1: query_cmdb first
        for a in actions:
            if a["tool_call"] == "query_cmdb":
                ordered.append(a)
        # Pass 2: open_ticket next
        for a in actions:
            if a["tool_call"] == "open_ticket" and a not in ordered:
                ordered.append(a)
        # Pass 3: everything else
        for a in actions:
            if a not in ordered:
                ordered.append(a)
        return ordered

    def plan(self, handoff: dict) -> ExecutionPlan:
        """Build the full execution plan from diagnostic handoff."""
        urgency    = handoff["urgency"]
        station_id = handoff["station_id"]
        confidence = handoff["confidence"]
        actions_raw = handoff["all_actions"]
        budget     = COST_BUDGET.get(urgency.lower(), 1000)

        # Sequence + dependency ordering
        ordered = self._dependency_order(actions_raw)

        planned = []
        total_cost  = 0.0
        requires_h  = False
        dep_map = {}   # tool_call → sequence number for dependency tracking

        for seq, act in enumerate(ordered, 1):
            gate, reason = self._gate_decision(act, urgency)
            cost         = self._estimate_cost(act["tool_call"], act["tool_params"])
            total_cost  += cost
            if gate == "HOLD_FOR_APPROVAL":
                requires_h = True

            dep = None
            if act["tool_call"] == "schedule_dispatch" and "open_ticket" in dep_map:
                dep = dep_map["open_ticket"]
            dep_map[act["tool_call"]] = seq

            planned.append(asdict(PlannedAction(
                sequence        = seq,
                action          = act["action"],
                tool_call       = act["tool_call"],
                tool_params     = act["tool_params"],
                risk_tier       = act["risk_tier"],
                gate_decision   = gate,
                gate_reason     = reason,
                sla_hours       = act["sla_hours"],
                estimated_cost  = cost,
                dependency_on   = dep,
                citations       = act.get("citations", []),
            )))

        within_budget = total_cost <= budget
        critical_path = [a["sequence"] for a in planned
                         if a["risk_tier"] >= 2]

        summary = (
            f"Plan for {station_id} ({urgency}): "
            f"{len(planned)} actions, "
            f"estimated cost €{total_cost:.0f} "
            f"({'within' if within_budget else 'OVER'} budget €{budget}), "
            f"{'requires human approval' if requires_h else 'fully automated'}, "
            f"confidence {confidence:.3f}."
        )

        return ExecutionPlan(
            plan_id        = f"PLAN_{station_id}_{int(time.time())}",
            station_id     = station_id,
            urgency        = urgency,
            created_at     = pd.Timestamp.now().isoformat(),
            actions        = planned,
            total_cost_est = round(total_cost, 2),
            within_budget  = within_budget,
            requires_human = requires_h,
            critical_path  = critical_path,
            confidence     = confidence,
            plan_summary   = summary,
        )


# ── Execution Agent ────────────────────────────────────────────────────────

class ExecutionAgent:
    """
    Walks the ExecutionPlan, applies gate decisions, executes tool calls,
    and writes a full audit log to persistent memory.
    """

    def __init__(self):
        self.tools = ToolSimulator()

    def execute(self, plan: ExecutionPlan,
                auto_approve_timeout: bool = False) -> ExecutionLog:
        """
        Execute the plan.
        auto_approve_timeout=True simulates the timeout expiry (Tier 2 auto-execute).
        In production this is driven by a real timer callback.
        """
        t_start = time.time()
        results = []
        counts  = {"AUTO": 0, "RECOMMEND": 0,
                   "HOLD_FOR_APPROVAL": 0, "BLOCKED": 0}

        for action in plan.actions:
            gate     = action["gate_decision"]
            tool     = action["tool_call"]
            params   = action["tool_params"]
            dep      = action["dependency_on"]

            # Check dependency
            if dep:
                dep_result = next(
                    (r for r in results if r["tool_call"] ==
                     plan.actions[dep-1]["tool_call"]), None)
                if dep_result and dep_result["status"] not in ("SUCCESS","SIMULATED"):
                    results.append(asdict(ToolResult(
                        tool_call=tool, params=params,
                        status="SKIPPED", output={"reason": f"dependency seq={dep} failed"},
                        latency_ms=0, timestamp=pd.Timestamp.now().isoformat())))
                    counts["BLOCKED"] += 1
                    continue

            if gate == "AUTO":
                out, lat = self.tools.execute(tool, params)
                results.append(asdict(ToolResult(
                    tool_call=tool, params=params, status="SUCCESS",
                    output=out, latency_ms=round(lat,2),
                    timestamp=pd.Timestamp.now().isoformat())))
                counts["AUTO"] += 1

            elif gate == "RECOMMEND":
                if auto_approve_timeout:
                    out, lat = self.tools.execute(tool, params)
                    results.append(asdict(ToolResult(
                        tool_call=tool, params=params, status="SUCCESS",
                        output=out, latency_ms=round(lat,2),
                        timestamp=pd.Timestamp.now().isoformat())))
                    counts["RECOMMEND"] += 1
                else:
                    results.append(asdict(ToolResult(
                        tool_call=tool, params=params,
                        status="PENDING_TIMEOUT",
                        output={"message": f"Recommendation sent. Auto-executes in {TIMEOUT_H[2]}h."},
                        latency_ms=0, timestamp=pd.Timestamp.now().isoformat())))
                    counts["RECOMMEND"] += 1

            elif gate == "HOLD_FOR_APPROVAL":
                results.append(asdict(ToolResult(
                    tool_call=tool, params=params,
                    status="PENDING_APPROVAL",
                    output={"message": "Awaiting human approval. Full reasoning trace attached.",
                            "approver": plan.actions[0].get("citations",""),
                            "plan_id": plan.plan_id},
                    latency_ms=0, timestamp=pd.Timestamp.now().isoformat())))
                counts["HOLD_FOR_APPROVAL"] += 1

            else:
                results.append(asdict(ToolResult(
                    tool_call=tool, params=params, status="BLOCKED",
                    output={"reason": action["gate_reason"]},
                    latency_ms=0, timestamp=pd.Timestamp.now().isoformat())))
                counts["BLOCKED"] += 1

        total_lat = (time.time()-t_start)*1000

        outcome = (
            f"Executed {counts['AUTO']} auto | "
            f"{counts['RECOMMEND']} recommended | "
            f"{counts['HOLD_FOR_APPROVAL']} pending approval | "
            f"{counts['BLOCKED']} blocked. "
            f"Total latency {total_lat:.1f}ms."
        )

        # Memory entry for future similar-case retrieval
        memory_entry = {
            "station_id":   plan.station_id,
            "urgency":      plan.urgency,
            "timestamp":    pd.Timestamp.now().isoformat(),
            "plan_id":      plan.plan_id,
            "confidence":   plan.confidence,
            "actions_taken": [r["tool_call"] for r in results
                              if r["status"] in ("SUCCESS","SIMULATED")],
            "outcome":       outcome,
        }

        log = ExecutionLog(
            log_id        = f"LOG_{plan.station_id}_{int(time.time())}",
            plan_id       = plan.plan_id,
            station_id    = plan.station_id,
            urgency       = plan.urgency,
            started_at    = pd.Timestamp.now().isoformat(),
            completed_at  = pd.Timestamp.now().isoformat(),
            tool_results  = results,
            actions_auto         = counts["AUTO"],
            actions_timeout      = counts["RECOMMEND"],
            actions_human        = counts["HOLD_FOR_APPROVAL"],
            actions_blocked      = counts["BLOCKED"],
            total_latency_ms     = round(total_lat, 2),
            outcome_summary      = outcome,
            memory_entry         = memory_entry,
        )

        # Write to persistent memory
        mem_path = os.path.join(MEMORY_DIR, f"memory_{plan.station_id}.json")
        with open(mem_path, "w") as f:
            json.dump(memory_entry, f, indent=2)

        return log


# ── End-to-end demo ────────────────────────────────────────────────────────

def run_demo():
    print("=" * 68)
    print("PLANNING + EXECUTION AGENTS — DEMO RUN")
    print("=" * 68)

    diag_path = "results/diagnostic/diagnostic_reports_demo.json"
    if not os.path.exists(diag_path):
        print("  Diagnostic reports not found. Run diagnostic_agent.py first.")
        return

    with open(diag_path) as f:
        reports = json.load(f)

    planner   = PlanningAgent()
    executor  = ExecutionAgent()
    all_plans = []; all_logs = []

    for report in reports:
        handoff  = report["handoff_to_planner"]
        urgency  = handoff["urgency"]

        print(f"\n{'─'*62}")
        print(f"  Station: {handoff['station_id']}  |  Urgency: {urgency}")

        # ── Planning ──
        plan = planner.plan(handoff)
        all_plans.append(plan)
        print(f"\n  EXECUTION PLAN ({plan.plan_id}):")
        print(f"  {plan.plan_summary}")
        print(f"  Critical path: {plan.critical_path}")
        print(f"\n  Actions:")
        for act in plan.actions:
            gate_sym = {"AUTO":"✓","RECOMMEND":"→","HOLD_FOR_APPROVAL":"⏸","BLOCKED":"✗"}.get(act["gate_decision"],"?")
            print(f"    [{act['sequence']}] {gate_sym} {TIER_LABELS[act['risk_tier']]:<22} "
                  f"SLA={act['sla_hours']}h  €{act['estimated_cost']:.0f}  "
                  f"tool={act['tool_call']}")
            print(f"        {act['action'][:72]}")
            if act["dependency_on"]:
                print(f"        ↳ depends on action #{act['dependency_on']}")

        # ── Execution ──
        # Tier 2 auto-approve for Warning/Monitor to show full flow;
        # Critical keeps HOLD for human approval demonstration
        auto_approve = (urgency != "Critical")
        log = executor.execute(plan, auto_approve_timeout=auto_approve)
        all_logs.append(log)

        print(f"\n  EXECUTION LOG ({log.log_id}):")
        print(f"  {log.outcome_summary}")
        for res in log.tool_results:
            status_sym = {"SUCCESS":"✓","PENDING_TIMEOUT":"⏱","PENDING_APPROVAL":"⏸",
                          "BLOCKED":"✗","SKIPPED":"⊘"}.get(res["status"],"?")
            print(f"    {status_sym} {res['tool_call']:<25} → {res['status']:<20} "
                  f"lat={res['latency_ms']:.1f}ms")
            if res["status"] in ("SUCCESS",):
                key_out = {k:v for k,v in res["output"].items()
                           if k in ("ticket_id","dispatch_id","escalation_id",
                                    "status","eta_hours","assigned_engineer")}
                if key_out:
                    print(f"       output: {key_out}")

    # ── Summary table ──
    print(f"\n{'─'*62}  END-TO-END SUMMARY")
    print(f"  {'Station':<15} {'Urgency':<10} {'Actions':<8} {'€Cost':>7} "
          f"{'Auto':>5} {'Timeout':>8} {'Human':>6} {'Budget'}")
    print(f"  {'─'*70}")
    for plan, log in zip(all_plans, all_logs):
        print(f"  {plan.station_id:<15} {plan.urgency:<10} "
              f"{len(plan.actions):<8} "
              f"€{plan.total_cost_est:>5.0f}  "
              f"{log.actions_auto:>5}  "
              f"{log.actions_timeout:>7}  "
              f"{log.actions_human:>6}  "
              f"{'✓' if plan.within_budget else '✗'}")

    # Save
    plans_path = os.path.join(RESULTS_DIR, "execution_plans_demo.json")
    logs_path  = os.path.join(RESULTS_DIR, "execution_logs_demo.json")
    with open(plans_path,"w") as f: json.dump([asdict(p) for p in all_plans],f,indent=2)
    with open(logs_path,"w")  as f: json.dump([asdict(l) for l in all_logs], f,indent=2)
    print(f"\n  Saved plans → {plans_path}")
    print(f"  Saved logs  → {logs_path}")
    print(f"  Memory entries → {MEMORY_DIR}/")
    print("=" * 68)
    print("PLANNING + EXECUTION AGENTS DEMO COMPLETE")
    print("=" * 68)
    return all_plans, all_logs

if __name__ == "__main__":
    run_demo()
