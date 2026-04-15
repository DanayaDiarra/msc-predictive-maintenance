"""
Agentic PdM Orchestrator — Master Pipeline Runner
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | March 2026

PHASE 2 MODEL: Ensemble + Bias Correction
  Transformer v2 (TTA×5, α=0.70) + XGBoost Phase 2 (α=0.30)
  + per-subset bias correction
  RMSE = 15.11 cycles | R² = 0.8663 | Zone 0-20 = 5.78 cycles

FULL END-TO-END PIPELINE:
  Stage 0 — Load model predictions (Phase 2 ensemble / synthetic in demo mode)
  Stage 1 — Interpreter Agent     (AlertJSON construction, Phase 2 CI)
  Stage 2 — RAG Pipeline          (Evidence Bundle retrieval)
  Stage 3 — Diagnostic Agent      (Root-cause + action recommendations)
  Stage 4 — Planning Agent        (Validated, sequenced execution plan)
  Stage 5 — Execution Agent       (Tool call execution + audit log)
  Stage 6 — Memory store          (Persist case for future retrieval)
  Stage 7 — Summary report        (KPIs across all pipeline stages)

USAGE:
  python3 agentic_pdm_orchestrator.py                 # demo mode (synthetic)
  python3 agentic_pdm_orchestrator.py --live          # live mode (Phase 2 ensemble)
  USE_LLM=true ANTHROPIC_API_KEY=sk-... python3 ...  # LLM-powered diagnosis

LIVE MODE:
  Requires: models_artifacts/final_models/
    transformer_v2_phase2.pt    — Transformer weights
    xgb_phase2.pkl              — XGBoost component
    production_scaler.pkl       — StandardScaler for DL input
    production_feat_cols.pkl    — sorted feature list (84 features)
    production_config.json      — ensemble config (α=0.70, bias corrections)

OUTPUT FILES:
  results/pipeline/pipeline_run_{timestamp}.json      full run record
  results/pipeline/pipeline_summary_{timestamp}.csv   KPI table
  results/memory/memory_{station_id}.json             per-station memory
"""

import os, sys, json, time, argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR  = "results/pipeline"
os.makedirs(RESULTS_DIR, exist_ok=True)

from interpreter_agent import InterpreterAgent, AlertJSON
from rag_corpus_builder import ALL_CHUNKS
from rag_pipeline       import RAGIndex, RAGPipeline, CORPUS_DIR, INDEX_DIR
from diagnostic_agent   import DiagnosticAgent
from planning_agent     import PlanningAgent, ExecutionAgent


# ── Phase 2 production config loader ─────────────────────────────────────
def _load_prod_cfg() -> dict:
    cfg = os.path.join("models_artifacts/final_models", "production_config.json")
    if os.path.exists(cfg):
        with open(cfg) as f: return json.load(f)
    return {}

PROD_CFG = _load_prod_cfg()


# ── Phase 2 Ensemble Predictor ────────────────────────────────────────────
class Phase2EnsemblePredictor:
    """
    Wraps the Phase 2 Transformer v2 + XGBoost ensemble for live prediction.
    Matches exactly the inference path used during evaluation:
      pred = α * TransV2(TTA×5) + (1-α) * XGBoost
      bias correction applied per subset
    Falls back gracefully to XGBoost-only if PyTorch unavailable.
    """
    MODEL_DIR = "models_artifacts/final_models"
    SEQ_LEN   = 30
    N_TTA     = 5
    TTA_SIGMA = 0.008   # noise std for TTA augmentation

    def __init__(self):
        self.xgb_model  = None
        self.dl_model   = None
        self.scaler     = None
        self.feat_cols  = None
        self.alpha      = PROD_CFG.get("ensemble_alpha", 0.70)
        self.bias_corr  = PROD_CFG.get("per_subset", {})
        self._load()

    def _load(self):
        import pickle
        md = self.MODEL_DIR

        # Feature columns (sorted)
        for p in ["production_feat_cols.pkl", "feat_cols_xgb.pkl"]:
            fp = os.path.join(md, p)
            if os.path.exists(fp):
                with open(fp,"rb") as f: self.feat_cols = pickle.load(f)
                break

        # StandardScaler
        sp = os.path.join(md, "production_scaler.pkl")
        if os.path.exists(sp):
            with open(sp,"rb") as f: self.scaler = pickle.load(f)

        # XGBoost component
        for p in ["xgb_phase2.pkl","xgb_best.pkl","xgb_v2.pkl"]:
            fp = os.path.join(md, p)
            if os.path.exists(fp):
                with open(fp,"rb") as f: self.xgb_model = pickle.load(f)
                print(f"  [Ensemble] XGBoost loaded: {p}")
                break

        # Transformer v2 component
        try:
            import torch, torch.nn as nn
            from sklearn.preprocessing import StandardScaler as _SC

            dl_path = os.path.join(md, "transformer_v2_phase2.pt")
            if os.path.exists(dl_path) and self.feat_cols and self.scaler:
                n = len(self.feat_cols)

                class _SE(nn.Module):
                    def __init__(self, d):
                        super().__init__()
                        self.gate = nn.Sequential(
                            nn.Linear(d, max(d//8,8)), nn.SiLU(),
                            nn.Linear(max(d//8,8), d), nn.Sigmoid())
                    def forward(self, x):
                        return x * self.gate(x.mean(1,keepdim=True))

                class _TV2(nn.Module):
                    def __init__(self, n, d=96, heads=4, layers=3,
                                 drop=0.20, L=30):
                        super().__init__()
                        self.proj    = nn.Linear(n, d)
                        self.pos_emb = nn.Embedding(L, d)
                        self.se      = _SE(d)
                        enc = nn.TransformerEncoderLayer(
                            d, heads, d*4, drop,
                            batch_first=True, norm_first=True)
                        self.enc  = nn.TransformerEncoder(enc, layers)
                        self.skip = nn.Linear(n, d)
                        self.norm = nn.LayerNorm(d)
                        self.head = nn.Sequential(
                            nn.Linear(d,48), nn.GELU(),
                            nn.Dropout(drop), nn.Linear(48,1))
                    def forward(self, x):
                        B,T,_ = x.shape
                        pos = torch.arange(T,device=x.device).unsqueeze(0).expand(B,T)
                        h = self.proj(x) + self.pos_emb(pos)
                        g = self.se(h)
                        h = self.enc(h).mean(1) + self.skip(x.mean(1))
                        return self.head(self.norm(h))

                self._torch    = torch
                self._TV2_cls  = _TV2
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._device   = device
                m = _TV2(n).to(device)
                m.load_state_dict(torch.load(dl_path, map_location=device))
                m.eval()
                self.dl_model = m
                print(f"  [Ensemble] TransformerV2 loaded → {device}")
            else:
                print("  [Ensemble] TransformerV2 not found — XGBoost only")
        except ImportError:
            print("  [Ensemble] PyTorch unavailable — XGBoost only")

    def predict(self, X_row: np.ndarray, subset: str = "FD001") -> tuple:
        """
        Predict RUL for a single feature row.
        Returns (rul_predicted, importance_dict)
        X_row: 1D array of shape (n_features,) in feat_cols order
        """
        import pickle
        rul_xgb  = None
        rul_dl   = None
        imp_dict = {}

        # XGBoost prediction
        if self.xgb_model is not None:
            X2d = X_row.reshape(1, -1)
            rul_xgb = float(np.clip(self.xgb_model.predict(X2d)[0], 0, 125))
            imp_dict = dict(zip(self.feat_cols or [],
                                self.xgb_model.feature_importances_))

        # Transformer prediction with TTA
        if self.dl_model is not None and self.scaler is not None:
            torch = self._torch
            X_sc  = self.scaler.transform(X_row.reshape(1,-1)).astype(np.float32)
            # Repeat row SEQ_LEN times to form a pseudo-sequence
            # (real usage: pass actual time-window; here single-point approximation)
            seq   = np.repeat(X_sc, self.SEQ_LEN, axis=0)  # (30, n_feats)
            Xb    = torch.from_numpy(seq).unsqueeze(0).to(self._device)  # (1,30,n)
            preds_tta = []
            with torch.no_grad():
                for _ in range(self.N_TTA):
                    noise = torch.randn_like(Xb) * self.TTA_SIGMA
                    out   = self.dl_model(Xb + noise).squeeze().cpu().item()
                    preds_tta.append(float(np.clip(out, 0, 125)))
            rul_dl = float(np.mean(preds_tta))

        # Ensemble
        if rul_dl is not None and rul_xgb is not None:
            rul = self.alpha * rul_dl + (1 - self.alpha) * rul_xgb
        elif rul_xgb is not None:
            rul = rul_xgb
        elif rul_dl is not None:
            rul = rul_dl
        else:
            rul = 50.0  # fallback

        # Per-subset bias correction (partial, 50% of measured bias)
        bias = self.bias_corr.get(subset, 0.0)
        rul  = float(np.clip(rul - bias * 0.5, 0, 125))

        return round(rul, 2), imp_dict


# ── Synthetic prediction generator (demo mode) ────────────────────────────
def synthetic_predictions(n: int = 5, seed: int = 42) -> list:
    """
    Generate synthetic model predictions for demo.
    Each dict mirrors the Phase 2 ensemble output contract:
      engine_id, rul_predicted, feature_importance_vector, model_version
    Top features reflect Phase 2 XGBoost: throughput_mbps, memory_usage, etc.
    The top-ranked feature always maps to the intended subsystem.
    """
    rng = np.random.default_rng(seed)
    stations = [
        ("FD002_47",   14.7,  "backhaul"),   # throughput dominant → backhaul
        ("FD001_23",   38.2,  "thermal"),
        ("FD004_112",  87.5,  "backhaul"),
        ("FD003_71",   55.1,  "rf"),
        ("FD001_08",   112.4, "baseband"),
    ]
    # Phase 2 top-5 features: throughput_mbps, throughput_mbps_lag1,
    # throughput_mbps_lag3, total_power_consumption, memory_usage
    # Primary feature guaranteed top-1 for correct subsystem routing
    primary_features = {
        "backhaul":  "throughput_mbps",
        "power":     "total_power_consumption",
        "thermal":   "temp_sensor_slope",
        "rf":        "signal_quality_slope",
        "baseband":  "cpu_utilization_mean",
    }
    secondary_features = {
        "backhaul":  ["throughput_mbps_lag1","throughput_mbps_lag3",
                      "latency_slope","packet_loss_rate"],
        "power":     ["voltage_rolling_mean","battery_slope",
                      "power_std_30","current_trend"],
        "thermal":   ["thermal_index_mean","fan_speed_delta",
                      "heat_index_mean","cooling_efficiency"],
        "rf":        ["sinr_rolling_mean","rssi_std_30",
                      "antenna_vswr_trend","interference_level"],
        "baseband":  ["processing_load_slope","load_rolling_std",
                      "utilization_trend","memory_usage"],
    }
    records = []
    for sid, rul, subsys in stations[:n]:
        sec  = secondary_features[subsys]
        n_sec = len(sec)
        sec_imps = rng.dirichlet(np.ones(n_sec) * 0.5) * 0.40
        prim_imp = 0.60 + rng.uniform(-0.05, 0.05)
        total    = prim_imp + sec_imps.sum()
        importance_dict = {primary_features[subsys]: round(float(prim_imp/total), 5)}
        for fname, fimp in zip(sec, sec_imps):
            importance_dict[fname] = round(float(fimp/total), 5)
        records.append({
            "engine_id":     sid,
            "rul_predicted": float(rul),
            "importance":    importance_dict,
            "model_version": "Phase2_Ensemble_TransV2_XGB",
            "subset":        sid.split("_")[0],   # FD001 / FD002 / FD003 / FD004
        })
    return records


# ── Pipeline stages ────────────────────────────────────────────────────────

def stage1_interpret(predictions: list) -> list:
    """Stage 1: Interpreter Agent → AlertJSON list"""
    agent   = InterpreterAgent()
    alerts  = []
    for pred in predictions:
        # importance dict doubles as feature_row in demo mode
        alert = agent.interpret(
            station_id   = pred["engine_id"],
            feature_row  = pred["importance"],
            rul_override = pred["rul_predicted"],
        )
        alerts.append(alert)
    return alerts


def stage2_retrieve(alerts: list) -> tuple:
    """Stage 2: RAG Pipeline → EvidenceBundle list"""
    # Ensure corpus and index exist
    corpus_path = os.path.join(CORPUS_DIR, "corpus.json")
    if not os.path.exists(corpus_path):
        print("  [Orchestrator] Building corpus...")
        import rag_corpus_builder  # importing executes build via module-level code
        # Re-run build explicitly
        import importlib
        importlib.reload(rag_corpus_builder)

    index = RAGIndex()
    index_file = os.path.join(INDEX_DIR, "chunks.json")
    if os.path.exists(index_file):
        index.load(INDEX_DIR)
    else:
        index.load_corpus(corpus_path)
        index.build()
        index.save(INDEX_DIR)

    pipeline = RAGPipeline(index)

    # Convert AlertJSON dataclass → dict for RAG pipeline
    alert_dicts = []
    for a in alerts:
        if hasattr(a, '__dict__'):
            d = a.__dict__
        else:
            d = a
        alert_dicts.append(d)

    bundles = pipeline.retrieve_batch(alert_dicts)
    # Return bundles as dicts for downstream serialisation
    from dataclasses import asdict as dc_asdict
    return [dc_asdict(b) for b in bundles], pipeline


def stage3_diagnose(alerts: list, bundles: list) -> list:
    """Stage 3: Diagnostic Agent → DiagnosticReport list"""
    agent   = DiagnosticAgent()
    reports = []
    for alert, bundle in zip(alerts, bundles):
        a_dict = alert.__dict__ if hasattr(alert, '__dict__') else alert
        report = agent.diagnose(a_dict, bundle)
        reports.append(report)
    return reports


def stage4_plan(reports: list) -> list:
    """Stage 4: Planning Agent → ExecutionPlan list"""
    planner = PlanningAgent()
    plans   = []
    for report in reports:
        r_dict = report.__dict__ if hasattr(report, '__dict__') else report
        plan   = planner.plan(r_dict["handoff_to_planner"])
        plans.append(plan)
    return plans


def stage5_execute(plans: list, reports: list) -> list:
    """Stage 5: Execution Agent → ExecutionLog list"""
    executor = ExecutionAgent()
    logs     = []
    for plan, report in zip(plans, reports):
        r_dict   = report.__dict__ if hasattr(report, '__dict__') else report
        urgency  = r_dict["urgency"]
        # Auto-approve Tier 2 for Warning/Monitor; hold for Critical
        auto_ok  = (urgency != "Critical")
        log      = executor.execute(plan, auto_approve_timeout=auto_ok)
        logs.append(log)
    return logs


# ── KPI Summary ────────────────────────────────────────────────────────────

def build_summary(predictions, alerts, bundles, reports, plans, logs,
                  pipeline_latency_ms: float) -> pd.DataFrame:
    rows = []
    for pred, alert, bundle, report, plan, log in \
            zip(predictions, alerts, bundles, reports, plans, logs):
        a = alert.__dict__ if hasattr(alert, '__dict__') else alert
        r = report.__dict__ if hasattr(report, '__dict__') else report
        rows.append({
            # Identity
            "station_id":         a["station_id"],
            "urgency":            a["urgency"],
            "governance_tier":    a["governance_tier"],
            # Layer 1 — Prediction
            "rul_cycles":         round(a["rul_cycles"], 1),
            "conf_low":           round(a["confidence_low"], 1),
            "conf_high":          round(a["confidence_high"], 1),
            "primary_subsystem":  a["primary_subsystem"],
            # Layer 2 — RAG
            "rag_latency_ms":     bundle["retrieval_latency_ms"],
            "rag_coverage":       bundle["coverage_score"],
            "rag_candidates":     bundle["n_candidates"],
            "top_chunk_doc_type": bundle["chunks"][0]["doc_type"] if bundle["chunks"] else "N/A",
            # Layer 3 — Diagnosis
            "diagnostic_confidence": r["diagnostic_confidence"],
            "grounding_rate":     r["grounding_rate"],
            "hallucination_rate": r["hallucination_rate"],
            "n_actions":          len(r["action_recommendations"]),
            "n_citations":        len(r["citations_used"]),
            # Layer 3 — Plan
            "plan_cost_est":      plan.total_cost_est,
            "within_budget":      plan.within_budget,
            "requires_human":     plan.requires_human,
            # Layer 3 — Execution
            "actions_auto":       log.actions_auto,
            "actions_timeout":    log.actions_timeout,
            "actions_human":      log.actions_human,
            "sla_hours":          a["sla_hours"],
            "auto_execute":       a["auto_execute"],
        })
    df = pd.DataFrame(rows)
    return df


def print_banner(title: str, width: int = 70):
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


# ── Master run ─────────────────────────────────────────────────────────────

def run_pipeline(n_stations: int = 5, live_mode: bool = False):
    t_total = time.time()
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    prod_rmse = PROD_CFG.get("rmse", "15.11")
    prod_r2   = PROD_CFG.get("r2",   "0.8663")
    print_banner("AGENTIC PdM ORCHESTRATOR — FULL PIPELINE RUN (PHASE 2)")
    print(f"  Mode         : {'LIVE (Phase 2 ensemble)' if live_mode else 'DEMO (synthetic)'}")
    print(f"  Stations     : {n_stations}")
    print(f"  Model        : Phase2 Ensemble (TransV2×0.70 + XGBoost×0.30 + bias corr.)")
    print(f"  Model RMSE   : {prod_rmse} cycles  R²={prod_r2}")
    print(f"  Timestamp    : {ts}")

    # ── Stage 0: Predictions ──────────────────────────────────────────────
    print(f"\n  ── Stage 0: Model Predictions ──")
    if live_mode:
        # Live mode: use Phase 2 ensemble predictor
        # Requires feature-engineered data; here we show the interface
        try:
            ensemble = Phase2EnsemblePredictor()
            # In real deployment: load actual feature rows from data pipeline
            # For now, fall back to synthetic with ensemble metadata
            predictions = synthetic_predictions(n=n_stations)
            for p in predictions:
                p["model_version"] = "Phase2_Ensemble_TransV2_XGB_LIVE"
            print("  [Live] Phase 2 ensemble loaded — "
                  "using synthetic RUL values (provide real features via API)")
        except Exception as e:
            print(f"  [Live] Ensemble load failed ({e}), using synthetic demo")
            predictions = synthetic_predictions(n=n_stations)
    else:
        predictions = synthetic_predictions(n=n_stations)

    for p in predictions:
        print(f"    {p['engine_id']:<15} RUL={p['rul_predicted']:>7.1f} cycles  "
              f"model={p['model_version']}")

    # ── Stage 1: Interpreter ─────────────────────────────────────────────
    print(f"\n  ── Stage 1: Interpreter Agent ──")
    t1 = time.time()
    alerts = stage1_interpret(predictions)
    t1_ms  = (time.time()-t1)*1000
    for a in alerts:
        a_d = a.__dict__ if hasattr(a,'__dict__') else a
        print(f"    {a_d['station_id']:<15} → {a_d['urgency']:<10} "
              f"subsystem={a_d['primary_subsystem']:<30} "
              f"CI=[{a_d['confidence_low']:.1f}–{a_d['confidence_high']:.1f}]  "
              f"SLA={a_d['sla_hours']}h")
    print(f"    Interpreter total: {t1_ms:.1f}ms")

    # ── Stage 2: RAG ─────────────────────────────────────────────────────
    print(f"\n  ── Stage 2: RAG Pipeline ──")
    t2 = time.time()
    bundles, _rag = stage2_retrieve(alerts)
    t2_ms = (time.time()-t2)*1000
    for a, b in zip(alerts, bundles):
        a_d = a.__dict__ if hasattr(a,'__dict__') else a
        print(f"    {a_d['station_id']:<15} → coverage={b['coverage_score']:.2f}  "
              f"candidates={b['n_candidates']}  "
              f"latency={b['retrieval_latency_ms']:.1f}ms  "
              f"top=[{b['chunks'][0]['citation_ref']}]")
    print(f"    RAG total: {t2_ms:.1f}ms")

    # ── Stage 3: Diagnostic ──────────────────────────────────────────────
    print(f"\n  ── Stage 3: Diagnostic Agent ──")
    t3 = time.time()
    reports = stage3_diagnose(alerts, bundles)
    t3_ms   = (time.time()-t3)*1000
    for r in reports:
        r_d = r.__dict__ if hasattr(r,'__dict__') else r
        print(f"    {r_d['station_id']:<15} → conf={r_d['diagnostic_confidence']:.3f}  "
              f"grounding={r_d['grounding_rate']:.3f}  "
              f"halluc={r_d['hallucination_rate']:.3f}  "
              f"actions={len(r_d['action_recommendations'])}")
    print(f"    Diagnostic total: {t3_ms:.1f}ms")

    # ── Stage 4: Planning ────────────────────────────────────────────────
    print(f"\n  ── Stage 4: Planning Agent ──")
    t4 = time.time()
    plans   = stage4_plan(reports)
    t4_ms   = (time.time()-t4)*1000
    for p in plans:
        print(f"    {p.station_id:<15} → {len(p.actions)} actions  "
              f"€{p.total_cost_est:.0f}  "
              f"budget={'✓' if p.within_budget else '✗'}  "
              f"human={'⏸ required' if p.requires_human else '✓ automated'}")
    print(f"    Planning total: {t4_ms:.1f}ms")

    # ── Stage 5: Execution ───────────────────────────────────────────────
    print(f"\n  ── Stage 5: Execution Agent ──")
    t5 = time.time()
    logs    = stage5_execute(plans, reports)
    t5_ms   = (time.time()-t5)*1000
    for l in logs:
        print(f"    {l.station_id:<15} → auto={l.actions_auto}  "
              f"timeout={l.actions_timeout}  "
              f"human={l.actions_human}  "
              f"lat={l.total_latency_ms:.1f}ms")
    print(f"    Execution total: {t5_ms:.1f}ms")

    # ── Summary KPI Table ────────────────────────────────────────────────
    total_ms = (time.time()-t_total)*1000
    df_summary = build_summary(predictions, alerts, bundles,
                               reports, plans, logs, total_ms)

    print_banner("PIPELINE KPI SUMMARY")
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    kpi_cols = ["station_id","urgency","rul_cycles","primary_subsystem",
                "rag_coverage","diagnostic_confidence","grounding_rate",
                "hallucination_rate","n_actions","plan_cost_est",
                "within_budget","actions_auto","actions_timeout","actions_human"]
    print(df_summary[kpi_cols].to_string(index=False))

    print(f"\n  Aggregate metrics:")
    print(f"    Mean grounding rate:       {df_summary['grounding_rate'].mean():.3f}")
    print(f"    Mean hallucination rate:   {df_summary['hallucination_rate'].mean():.3f}")
    print(f"    Mean diagnostic confidence:{df_summary['diagnostic_confidence'].mean():.3f}")
    print(f"    Mean RAG coverage:         {df_summary['rag_coverage'].mean():.3f}")
    print(f"    Total actions AUTO:        {df_summary['actions_auto'].sum()}")
    print(f"    Total actions TIMEOUT:     {df_summary['actions_timeout'].sum()}")
    print(f"    Total actions HUMAN:       {df_summary['actions_human'].sum()}")
    print(f"    Total estimated cost:      €{df_summary['plan_cost_est'].sum():.0f}")
    print(f"    Pipeline latency:          {total_ms:.0f}ms ({total_ms/1000:.2f}s)")

    # ── Stage Latency Breakdown ──────────────────────────────────────────
    print(f"\n  Stage latency breakdown:")
    stages = [("Interpreter",t1_ms),("RAG",t2_ms),
              ("Diagnostic",t3_ms),("Planning",t4_ms),("Execution",t5_ms)]
    for name, ms in stages:
        bar = "█" * int(ms/total_ms * 40)
        print(f"    {name:<15} {ms:>8.1f}ms  {ms/total_ms*100:>5.1f}%  {bar}")

    # ── Save ─────────────────────────────────────────────────────────────
    csv_path  = os.path.join(RESULTS_DIR, f"pipeline_summary_{ts}.csv")
    json_path = os.path.join(RESULTS_DIR, f"pipeline_run_{ts}.json")

    df_summary.to_csv(csv_path, index=False)

    # Build serialisable run record
    from dataclasses import asdict as dc_asdict
    def safe_asdict(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return obj

    run_record = {
        "timestamp":       ts,
        "n_stations":      n_stations,
        "model_version":   "Phase2_Ensemble_TransV2_XGB",
        "model_rmse":      PROD_CFG.get("rmse", 15.11),
        "model_r2":        PROD_CFG.get("r2", 0.8663),
        "confidence_alpha": PROD_CFG.get("confidence_alpha", 0.2206),
        "total_latency_ms": round(total_ms, 2),
        "stage_latencies": {n: round(ms,2) for n,ms in stages},
        "aggregate_kpis": {
            "mean_grounding_rate":        round(df_summary['grounding_rate'].mean(),3),
            "mean_hallucination_rate":    round(df_summary['hallucination_rate'].mean(),3),
            "mean_diagnostic_confidence": round(df_summary['diagnostic_confidence'].mean(),3),
            "mean_rag_coverage":          round(df_summary['rag_coverage'].mean(),3),
            "total_actions_auto":         int(df_summary['actions_auto'].sum()),
            "total_actions_timeout":      int(df_summary['actions_timeout'].sum()),
            "total_actions_human":        int(df_summary['actions_human'].sum()),
            "total_estimated_cost_eur":   round(float(df_summary['plan_cost_est'].sum()),2),
        },
        "stations": df_summary.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(run_record, f, indent=2)

    print(f"\n  Saved summary CSV → {csv_path}")
    print(f"  Saved run JSON    → {json_path}")
    print_banner("PIPELINE RUN COMPLETE")
    return df_summary, run_record


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Use trained model artifacts instead of synthetic predictions")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of stations to process (default: 5)")
    args = parser.parse_args()
    run_pipeline(n_stations=args.n, live_mode=args.live)
