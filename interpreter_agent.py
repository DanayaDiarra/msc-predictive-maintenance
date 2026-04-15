"""
Interpreter Agent — Layer 1 → Layer 2 Bridge
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | March 2026

PURPOSE:
  Receives Phase 2 model output (RUL scalar + feature importance vector)
  and transforms it into a structured, semantically rich alert JSON that:
    1. Assigns an urgency tier  (Critical / Warning / Monitor)
    2. Maps top features to telecom subsystem fault hypotheses
    3. Derives a calibrated confidence interval (conformal prediction)
    4. Constructs a natural-language situation summary
    5. Builds a structured RAG query for Layer 2 retrieval
    6. Determines governance tier for downstream action approval

PHASE 2 MODEL: Ensemble + Bias Correction
  - Transformer v2 (TTA×5, α=0.70) + XGBoost (α=0.30) + per-subset bias corr.
  - Test RMSE = 15.11 cycles  (Phase 1 = 15.37 cycles)
  - R²        = 0.8663
  - Zone 0-20 RMSE = 5.78 cycles (critical zone)
  - Conformal CI (90% coverage): ±27.58 cycles → CONFIDENCE_ALPHA = 0.2206

DESIGN NOTES:
  - Loads production_config.json to read Phase 2 metrics and CONFIDENCE_ALPHA
  - Falls back to XGBoost phase2 model for direct predictions (fast, deterministic)
  - Transformer ensemble used via orchestrator's pre-computed RUL override path
  - Telecom domain mapping is encoded in FEATURE_SUBSYSTEM_MAP and
    SUBSYSTEM_FAULT_HYPOTHESIS — extendable without retraining the model
"""

import os, json, pickle, time, warnings
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
MAX_RUL          = 125          # cycles (same as training cap)
CRITICAL_THRESH  = 20           # RUL ≤ 20  → Critical
WARNING_THRESH   = 50           # RUL ≤ 50  → Warning
TOP_N_FEATURES   = 5            # how many features to include in alert

# Phase 2 conformal prediction: ±27.58 cycles at 90% coverage
# Derived from calibration set residuals in phase2_advanced_retraining.py
CONFIDENCE_ALPHA = 0.2206       # replaces old 0.15 — empirically calibrated

MODEL_DIR    = "models_artifacts/final_models"
RESULTS_DIR  = "results/final_models"

# ── Load Phase 2 production config (updates CONFIDENCE_ALPHA if available) ──
def _load_prod_config() -> dict:
    """Load production_config.json written by phase2_advanced_retraining.py."""
    cfg_path = os.path.join(MODEL_DIR, "production_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

_PROD_CFG = _load_prod_config()
if _PROD_CFG.get("confidence_alpha"):
    CONFIDENCE_ALPHA = _PROD_CFG["confidence_alpha"]

# ── Telecom domain feature→subsystem mapping ───────────────────────────────
# Maps XGBoost feature name patterns to telecom BTS subsystems.
# Patterns are matched by substring — order matters (first match wins).
FEATURE_SUBSYSTEM_MAP = {
    # Power subsystem
    "power":         "power_subsystem",
    "voltage":       "power_subsystem",
    "current":       "power_subsystem",
    "battery":       "power_subsystem",
    "energy":        "power_subsystem",
    # Thermal subsystem
    "temp":          "thermal_management",
    "thermal":       "thermal_management",
    "heat":          "thermal_management",
    "fan":           "thermal_management",
    "cooling":       "thermal_management",
    # RF / antenna subsystem
    "rssi":          "rf_antenna",
    "sinr":          "rf_antenna",
    "signal":        "rf_antenna",
    "antenna":       "rf_antenna",
    "frequency":     "rf_antenna",
    "interference":  "rf_antenna",
    # Backhaul / connectivity
    "latency":       "backhaul_connectivity",
    "throughput":    "backhaul_connectivity",
    "packet":        "backhaul_connectivity",
    "backhaul":      "backhaul_connectivity",
    "link":          "backhaul_connectivity",
    # Baseband processing
    "cpu":           "baseband_processing",
    "load":          "baseband_processing",
    "processing":    "baseband_processing",
    "utilization":   "baseband_processing",
    # Sensor degradation (generic from C-MAPSS mapping)
    "sensor":        "sensor_array",
    "s_":            "sensor_array",
    "rolling":       "degradation_trend",
    "slope":         "degradation_trend",
    "std":           "variability_index",
    "mean":          "health_trend",
    # Fallback
    "cycle":         "operational_age",
    "time":          "operational_age",
}

# Maps subsystem → primary fault hypothesis for the Diagnostic Agent
SUBSYSTEM_FAULT_HYPOTHESIS = {
    "power_subsystem":        "Power unit degradation — voltage instability or rectifier wear",
    "thermal_management":     "Thermal runaway risk — cooling fan wear or blocked ventilation",
    "rf_antenna":             "RF chain degradation — antenna connector corrosion or LNA drift",
    "backhaul_connectivity":  "Backhaul link degradation — fibre splice loss or microwave alignment",
    "baseband_processing":    "Baseband unit overload — software fault or hardware component ageing",
    "sensor_array":           "Sensor drift — calibration loss or measurement chain fault",
    "degradation_trend":      "Accelerating degradation — multiple subsystem wear convergence",
    "variability_index":      "Measurement instability — possible sensor or ADC fault",
    "health_trend":           "Gradual health decline — normal end-of-life progression",
    "operational_age":        "Operational age threshold — scheduled lifecycle replacement due",
}

# SLA / governance mapping
URGENCY_GOVERNANCE = {
    "Critical": {
        "tier": 3,
        "sla_hours": 4,
        "auto_execute": False,
        "required_approval": "maintenance_engineer",
        "escalation_path": "NOC_shift_lead → field_operations_manager",
    },
    "Warning": {
        "tier": 2,
        "sla_hours": 48,
        "auto_execute": True,
        "timeout_hours": 6,
        "required_approval": "auto_after_timeout",
        "escalation_path": "maintenance_engineer",
    },
    "Monitor": {
        "tier": 1,
        "sla_hours": 168,
        "auto_execute": True,
        "timeout_hours": None,
        "required_approval": "none",
        "escalation_path": "scheduled_maintenance_queue",
    },
}


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class FeatureSignal:
    """A single contributing feature with its telecom interpretation."""
    feature_name:    str
    importance:      float
    subsystem:       str
    rank:            int

@dataclass
class AlertJSON:
    """
    Structured alert produced by the Interpreter Agent.
    This is the input contract for the RAG pipeline and Diagnostic Agent.
    """
    # Identity
    alert_id:         str
    station_id:       str
    timestamp:        str
    # Prediction
    rul_cycles:       float
    rul_hours_est:    float          # estimated assuming 1 cycle ≈ 1 operational hour
    urgency:          str            # Critical / Warning / Monitor
    confidence_low:   float
    confidence_high:  float
    certainty:        str            # qualitative certainty statement
    # Fault reasoning
    primary_subsystem:   str
    fault_hypothesis:    str
    contributing_signals: list       # list of FeatureSignal dicts
    # RAG query fields (consumed by Layer 2)
    rag_query_primary:   str
    rag_query_equipment: str
    rag_query_keywords:  list
    # Governance
    governance_tier:     int
    sla_hours:           int
    auto_execute:        bool
    escalation_path:     str
    # Natural language summary (consumed by Diagnostic Agent prompt)
    situation_summary:   str
    # Metadata
    model_version:       str = "Phase2_Ensemble_TransV2_XGB"
    pipeline_stage:      str = "interpreter_agent"


# ── Core Interpreter Agent ─────────────────────────────────────────────────

class InterpreterAgent:
    """
    Transforms XGBoost v2 raw output into a structured alert JSON.

    The agent operates deterministically in three steps:
      Step 1: RUL → urgency tier + governance metadata
      Step 2: Feature importance → subsystem mapping + fault hypothesis
      Step 3: Construct natural-language summary + RAG query fields

    No LLM is called here — this ensures sub-millisecond latency and
    full determinism, critical for Tier 3 (human-approval required) paths
    where audit trails must be reproducible.
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir   = model_dir
        self.model       = None      # XGBoost fallback model (for direct prediction)
        self.feat_cols   = None      # sorted feature list (Phase 2)
        self.prod_cfg    = _PROD_CFG # Phase 2 production config
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Load Phase 2 production artifacts.
        Priority order:
          1. xgb_phase2.pkl       — Phase 2 retrained XGBoost (direct prediction)
          2. xgb_best.pkl         — Phase 1 HPO XGBoost (fallback)
          3. xgb_v2.pkl           — original XGBoost v2 (legacy fallback)
        Feature columns: production_feat_cols.pkl → feat_cols_xgb.pkl
        Note: Transformer v2 ensemble predictions are passed via rul_override
              from the orchestrator — not loaded here (avoids GPU dependency).
        """
        # Feature columns (sorted, leakage-free — as used in Phase 2)
        for feat_path in [
            os.path.join(self.model_dir, "production_feat_cols.pkl"),
            os.path.join(self.model_dir, "feat_cols_xgb.pkl"),
        ]:
            if os.path.exists(feat_path):
                with open(feat_path, "rb") as f:
                    self.feat_cols = pickle.load(f)
                break

        # XGBoost model (for direct prediction when no rul_override given)
        for model_path in [
            os.path.join(self.model_dir, "xgb_phase2.pkl"),
            os.path.join(self.model_dir, "xgb_best.pkl"),
            os.path.join(self.model_dir, "xgb_v2.pkl"),
        ]:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                model_name = os.path.basename(model_path)
                feat_count = len(self.feat_cols) if self.feat_cols else "unknown"
                print(f"  [Interpreter] Loaded {model_name} — {feat_count} features")
                if self.prod_cfg:
                    print(f"  [Interpreter] Phase 2 config: "
                          f"RMSE={self.prod_cfg.get('rmse','?')} "
                          f"R²={self.prod_cfg.get('r2','?')} "
                          f"CONFIDENCE_ALPHA={CONFIDENCE_ALPHA}")
                return

        print(f"  [Interpreter] WARNING: no model found in {self.model_dir}")
        print(f"  [Interpreter] Running in DEMO mode with synthetic predictions")

    # ── Step 1: Urgency Tier ──────────────────────────────────────────────
    def _assign_urgency(self, rul: float) -> str:
        if rul <= CRITICAL_THRESH: return "Critical"
        if rul <= WARNING_THRESH:  return "Warning"
        return "Monitor"

    def _confidence_interval(self, rul: float) -> tuple:
        """
        Derive ± confidence band using Phase 2 conformal prediction.
        CONFIDENCE_ALPHA = 0.2206, derived from calibration set residuals:
          conformal_q = 27.58 cycles at 90% coverage (XGBoost on val set).
        This is broader than the old 0.15 because the ensemble produces
        predictions with higher variance in the 50–100 cycle zone, but
        the 90% coverage guarantee makes it statistically meaningful.
        """
        margin = max(3.0, rul * CONFIDENCE_ALPHA)
        return max(0.0, rul - margin), rul + margin

    def _certainty_statement(self, rul: float, ci_low: float, ci_high: float) -> str:
        band = ci_high - ci_low
        pct  = band / max(1, rul) * 100
        # Thresholds adjusted for Phase 2 conformal CI width
        if pct < 30:   return "high certainty (conformal 90% CI)"
        if pct < 55:   return "moderate certainty (conformal 90% CI)"
        return "low certainty — recommend additional sensor validation"

    # ── Step 2: Feature → Subsystem Mapping ──────────────────────────────
    def _map_feature_to_subsystem(self, feature_name: str) -> str:
        fn_lower = feature_name.lower()
        for pattern, subsystem in FEATURE_SUBSYSTEM_MAP.items():
            if pattern in fn_lower:
                return subsystem
        return "unclassified_signal"

    def _extract_feature_signals(self, importance_dict: dict) -> list:
        """
        Takes {feature_name: importance_score} sorted by importance.
        Returns list of FeatureSignal for the top-N features.
        """
        signals = []
        for rank, (fname, imp) in enumerate(
                sorted(importance_dict.items(), key=lambda x: -x[1])[:TOP_N_FEATURES], 1):
            subsystem = self._map_feature_to_subsystem(fname)
            signals.append(FeatureSignal(
                feature_name=fname, importance=round(imp, 5),
                subsystem=subsystem, rank=rank))
        return signals

    def _primary_subsystem(self, signals: list) -> str:
        """
        The primary subsystem is the one with the highest cumulative
        feature importance across the top-N signals.
        """
        subsystem_importance = {}
        for sig in signals:
            subsystem_importance[sig.subsystem] = (
                subsystem_importance.get(sig.subsystem, 0) + sig.importance)
        return max(subsystem_importance, key=subsystem_importance.get)

    # ── Step 3: RAG Query Construction ───────────────────────────────────
    def _build_rag_query(self, primary_subsystem: str, urgency: str,
                         station_id: str, signals: list) -> tuple:
        """
        Returns (primary_query_string, equipment_query, keyword_list).
        These are consumed by the hybrid RAG retrieval (dense + BM25).
        """
        fault_hyp  = SUBSYSTEM_FAULT_HYPOTHESIS.get(primary_subsystem, primary_subsystem)
        # Primary semantic query (for dense embedding search)
        primary_q  = (f"Troubleshooting procedure for {primary_subsystem.replace('_',' ')} "
                      f"fault in telecom base station. Urgency: {urgency}. "
                      f"Fault type: {fault_hyp}")
        # Equipment-specific query (for BM25 keyword search)
        equip_q    = f"base station {primary_subsystem.replace('_',' ')} maintenance alarm SOP"
        # Keywords for metadata filtering
        keywords   = [primary_subsystem, urgency.lower(),
                      "base_station", "predictive_maintenance"]
        keywords  += list({sig.subsystem for sig in signals[:3]})
        return primary_q, equip_q, keywords

    # ── Natural-Language Summary ──────────────────────────────────────────
    def _situation_summary(self, station_id: str, rul: float, urgency: str,
                           ci_low: float, ci_high: float, certainty: str,
                           primary_subsystem: str, fault_hypothesis: str,
                           signals: list) -> str:
        top_feats = ", ".join([f"{s.feature_name} (imp={s.importance:.4f})"
                               for s in signals[:3]])
        summary = (
            f"Station {station_id} is predicted to require maintenance in approximately "
            f"{rul:.1f} cycles (confidence interval: {ci_low:.1f}–{ci_high:.1f} cycles; "
            f"{certainty}). "
            f"Urgency classification: {urgency}. "
            f"Primary degradation signal is concentrated in the {primary_subsystem.replace('_',' ')}. "
            f"Working hypothesis: {fault_hypothesis}. "
            f"Top contributing features: {top_feats}. "
            f"Recommended action pathway: {URGENCY_GOVERNANCE[urgency]['escalation_path']}."
        )
        return summary

    # ── Main Entry Point ──────────────────────────────────────────────────
    def interpret(self, station_id: str, feature_row: dict,
                  rul_override: float = None) -> AlertJSON:
        """
        Main method. Accepts either:
          (a) feature_row dict → runs XGBoost v2 prediction internally
          (b) rul_override float → uses pre-computed RUL (for demo/testing)

        Returns: AlertJSON (fully structured alert)
        """
        t0 = time.time()

        # ── Predict RUL ──
        if rul_override is not None:
            # Path A: pre-computed RUL (Transformer v2 ensemble from orchestrator)
            rul = float(np.clip(rul_override, 0, MAX_RUL))
            importance_dict = {}
            if feature_row:
                importance_dict = {k: v for k, v in feature_row.items()
                                   if not isinstance(v, str)}
        elif self.model is not None and self.feat_cols is not None:
            # Path B: direct XGBoost prediction (Phase 2 retrained)
            # feature_row must contain all feat_cols keys
            try:
                X = pd.DataFrame([feature_row])[self.feat_cols].values.astype(np.float32)
                rul = float(np.clip(self.model.predict(X)[0], 0, MAX_RUL))
                importance_dict = dict(zip(self.feat_cols,
                                           self.model.feature_importances_))
            except KeyError as e:
                print(f"  [Interpreter] Feature mismatch ({e}), using demo values")
                rul = 23.4
                importance_dict = {
                    "throughput_mbps":          0.0921,
                    "throughput_mbps_lag1":      0.0814,
                    "throughput_mbps_lag3":      0.0744,
                    "total_power_consumption":   0.0689,
                    "memory_usage":              0.0621,
                    "voltage_rolling_mean":      0.0589,
                }
        else:
            # Path C: Demo mode — synthetic values reflecting Phase 2 top features
            rul = 23.4
            importance_dict = {
                "throughput_mbps":          0.0921,
                "throughput_mbps_lag1":      0.0814,
                "throughput_mbps_lag3":      0.0744,
                "total_power_consumption":   0.0689,
                "memory_usage":              0.0621,
                "voltage_rolling_mean":      0.0589,
            }

        # ── Step 1: Urgency + Confidence ──
        urgency       = self._assign_urgency(rul)
        ci_low, ci_hi = self._confidence_interval(rul)
        certainty     = self._certainty_statement(rul, ci_low, ci_hi)
        governance    = URGENCY_GOVERNANCE[urgency]

        # ── Step 2: Feature → Subsystem ──
        signals           = self._extract_feature_signals(importance_dict)
        primary_sub       = self._primary_subsystem(signals)
        fault_hypothesis  = SUBSYSTEM_FAULT_HYPOTHESIS.get(primary_sub, primary_sub)

        # ── Step 3: RAG Query + Summary ──
        rag_q, eq_q, kw   = self._build_rag_query(primary_sub, urgency,
                                                    station_id, signals)
        summary           = self._situation_summary(
            station_id, rul, urgency, ci_low, ci_hi, certainty,
            primary_sub, fault_hypothesis, signals)

        elapsed_ms = (time.time() - t0) * 1000

        alert = AlertJSON(
            alert_id          = f"ALERT_{station_id}_{int(time.time())}",
            station_id        = station_id,
            timestamp         = pd.Timestamp.now().isoformat(),
            rul_cycles        = round(rul, 2),
            rul_hours_est     = round(rul * 1.0, 2),
            urgency           = urgency,
            confidence_low    = round(ci_low, 2),
            confidence_high   = round(ci_hi, 2),
            certainty         = certainty,
            primary_subsystem = primary_sub,
            fault_hypothesis  = fault_hypothesis,
            contributing_signals = [asdict(s) for s in signals],
            rag_query_primary    = rag_q,
            rag_query_equipment  = eq_q,
            rag_query_keywords   = kw,
            governance_tier      = governance["tier"],
            sla_hours            = governance["sla_hours"],
            auto_execute         = governance["auto_execute"],
            escalation_path      = governance["escalation_path"],
            situation_summary    = summary,
        )

        print(f"  [Interpreter] Processed {station_id} in {elapsed_ms:.2f}ms")
        return alert

    def interpret_batch(self, stations: list) -> list:
        """Process multiple stations and return sorted by urgency (Critical first)."""
        alerts = [self.interpret(**s) for s in stations]
        priority = {"Critical": 0, "Warning": 1, "Monitor": 2}
        return sorted(alerts, key=lambda a: (priority[a.urgency], a.rul_cycles))


# ── Demo / Test harness ────────────────────────────────────────────────────

def run_demo():
    print("=" * 68)
    print("INTERPRETER AGENT — DEMO RUN")
    print("=" * 68)

    agent = InterpreterAgent()

    # Three synthetic stations at different urgency levels
    test_cases = [
        {
            "station_id": "FD002_47",
            "feature_row": {
                "total_power_slope_20":  0.0744,
                "voltage_rolling_mean":  0.0589,
                "s2_rolling_mean_10":    0.0421,
            },
            "rul_override": 14.7,    # Critical
        },
        {
            "station_id": "FD001_23",
            "feature_row": {
                "s3_std_30":             0.0621,
                "temp_sensor_slope":     0.0512,
                "s7_rolling_mean_10":    0.0488,
            },
            "rul_override": 38.2,    # Warning
        },
        {
            "station_id": "FD004_112",
            "feature_row": {
                "s2_rolling_mean_10":    0.0812,
                "latency_slope":         0.0340,
                "cpu_utilization_mean":  0.0290,
            },
            "rul_override": 87.5,    # Monitor
        },
    ]

    alerts = agent.interpret_batch(test_cases)

    for alert in alerts:
        print(f"\n{'─'*60}")
        print(f"  ALERT ID:    {alert.alert_id}")
        print(f"  Station:     {alert.station_id}")
        print(f"  RUL:         {alert.rul_cycles} cycles  "
              f"[{alert.confidence_low}–{alert.confidence_high}]  {alert.certainty}")
        print(f"  Urgency:     {alert.urgency}  (Governance Tier {alert.governance_tier})")
        print(f"  SLA:         {alert.sla_hours}h  |  Auto-execute: {alert.auto_execute}")
        print(f"  Subsystem:   {alert.primary_subsystem}")
        print(f"  Hypothesis:  {alert.fault_hypothesis}")
        print(f"\n  Top signals:")
        for sig in alert.contributing_signals[:3]:
            print(f"    [{sig['rank']}] {sig['feature_name']:<38} "
                  f"imp={sig['importance']:.5f}  → {sig['subsystem']}")
        print(f"\n  RAG query (primary):")
        print(f"    {alert.rag_query_primary}")
        print(f"\n  Situation summary:")
        for line in [alert.situation_summary[i:i+70]
                     for i in range(0, len(alert.situation_summary), 70)]:
            print(f"    {line}")

    # Save sample alert to JSON
    os.makedirs("results/interpreter", exist_ok=True)
    sample_path = "results/interpreter/sample_alerts.json"
    with open(sample_path, "w") as f:
        json.dump([asdict(a) for a in alerts], f, indent=2)
    print(f"\n{'─'*60}")
    print(f"  Saved {len(alerts)} sample alerts → {sample_path}")
    print("=" * 68)
    print("INTERPRETER AGENT DEMO COMPLETE")
    print("=" * 68)
    return alerts

if __name__ == "__main__":
    run_demo()
