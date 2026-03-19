"""
00_setup_and_run.py — Colab / Jupyter / Local Setup Cell
═══════════════════════════════════════════════════════════
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | March 2026
Model:  XGBoost v2 Final (15k trees · lr=0.02 · exp(α=3) sample weights · GPU-trained)

PASTE THIS AS THE FIRST CELL IN YOUR NOTEBOOK.
It handles:
  1. Locating all pipeline scripts regardless of working directory
  2. Adding the correct folder to sys.path
  3. Building the corpus + RAG index if not already present
  4. Running the full 5-stage pipeline
  5. Printing the KPI summary table

HOW TO USE IN GOOGLE COLAB:
  Step 1 — Upload all .py files to Colab (or mount Google Drive)
  Step 2 — Run this cell
  Step 3 — That's it

HOW TO USE LOCALLY:
  cd <folder-containing-all-.py-files>
  python 00_setup_and_run.py

REQUIRED FILES IN SAME FOLDER:
  interpreter_agent.py
  rag_corpus_builder.py
  rag_pipeline.py
  diagnostic_agent.py
  planning_agent.py
  agentic_pdm_orchestrator.py   (optional — this file replaces it)
"""

# ── Step 1: Path setup ──────────────────────────────────────────────────────
import sys, os
from pathlib import Path

# Detect script location vs notebook working directory
_THIS_FILE = Path(__file__).resolve() if "__file__" in dir() else Path.cwd()
_SCRIPT_DIR = _THIS_FILE.parent if _THIS_FILE.is_file() else _THIS_FILE

# Also check common Colab upload locations
_CANDIDATE_DIRS = [
    _SCRIPT_DIR,
    Path.cwd(),
    Path("/content"),                          # Colab default upload
    Path("/content/drive/MyDrive"),            # Colab Google Drive root
    Path.home() / "agentic_pdm",
]

def _find_module_dir(required_module="interpreter_agent.py"):
    for d in _CANDIDATE_DIRS:
        if (d / required_module).exists():
            return d
    return None

MODULE_DIR = _find_module_dir()

if MODULE_DIR is None:
    print("=" * 60)
    print("  ERROR: Cannot find pipeline scripts.")
    print("  Please ensure all .py files are in one of:")
    for d in _CANDIDATE_DIRS:
        print(f"    {d}")
    print("  Or set MODULE_DIR manually below.")
    print("=" * 60)
    # Manual override — set this to your folder if needed:
    MODULE_DIR = Path.cwd()
else:
    print(f"  [Setup] Module directory: {MODULE_DIR}")

# Add to sys.path if not already present
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
    print(f"  [Setup] Added {MODULE_DIR} to sys.path")

# Change working directory so relative data paths resolve correctly
os.chdir(MODULE_DIR)
print(f"  [Setup] Working directory set to: {os.getcwd()}")

# ── Step 2: Verify all modules importable ──────────────────────────────────
REQUIRED_MODULES = [
    "interpreter_agent",
    "rag_corpus_builder",
    "rag_pipeline",
    "diagnostic_agent",
    "planning_agent",
]

print("\n  [Setup] Checking module imports...")
all_ok = True
for mod in REQUIRED_MODULES:
    try:
        __import__(mod)
        print(f"    ✓  {mod}")
    except ImportError as e:
        print(f"    ✗  {mod}  — {e}")
        all_ok = False

if not all_ok:
    print("\n  Some modules failed to import.")
    print("  Make sure all .py files are in the same folder.")
    raise SystemExit(1)

print("\n  [Setup] All modules OK.\n")

# ── Step 3: Build corpus + index if needed ─────────────────────────────────
from rag_corpus_builder import CORPUS_DIR, ALL_CHUNKS
from rag_pipeline import RAGIndex, RAGPipeline, INDEX_DIR
import json

corpus_path = os.path.join(CORPUS_DIR, "corpus.json")
index_path  = os.path.join(INDEX_DIR,  "chunks.json")

if not os.path.exists(corpus_path):
    print("  [Setup] Building telecom knowledge corpus...")
    from rag_corpus_builder import save_corpus
    corpus_path = save_corpus(CORPUS_DIR)
    # Reload ALL_CHUNKS count after save
    from rag_corpus_builder import ALL_CHUNKS as _chunks
    print(f"  [Setup] Corpus saved ({len(_chunks)} chunks) → {corpus_path}")

if not os.path.exists(index_path):
    print("  [Setup] Building RAG index (33-chunk corpus)...")
    index = RAGIndex()
    index.load_corpus(corpus_path)
    index.build()
    index.save(INDEX_DIR)
    print(f"  [Setup] Index saved → {INDEX_DIR}/")
else:
    print(f"  [Setup] RAG index already exists → {index_path}")

# ── Step 4: Run full pipeline ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import time
from dataclasses import asdict as dc_asdict

from interpreter_agent import InterpreterAgent
from rag_pipeline       import RAGIndex, RAGPipeline
from diagnostic_agent   import DiagnosticAgent
from planning_agent     import PlanningAgent, ExecutionAgent

RESULTS_DIR = "results/pipeline"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Synthetic predictions (5 stations across all subsystem types) ──────────
def make_predictions():
    """
    5 stations with guaranteed subsystem-matched top feature.
    Mirrors XGBoost v2 predict() output contract.
    """
    return [
        {
            "engine_id": "FD002_47",  "rul_predicted": 14.7,   "model_version": "xgb_v2_final",  # Final Improved — 15k trees, exp weights, GPU
            "importance": {"voltage_rolling_mean": 0.074, "total_power_slope_20": 0.059,
                           "battery_slope": 0.042, "power_std_30": 0.031}
        },
        {
            "engine_id": "FD001_23",  "rul_predicted": 38.2,   "model_version": "xgb_v2_final",  # Final Improved — 15k trees, exp weights, GPU
            "importance": {"temp_sensor_slope": 0.087, "thermal_index_mean": 0.051,
                           "fan_speed_delta": 0.044, "heat_index_mean": 0.029}
        },
        {
            "engine_id": "FD004_112", "rul_predicted": 87.5,   "model_version": "xgb_v2_final",  # Final Improved — 15k trees, exp weights, GPU
            "importance": {"latency_slope": 0.068, "packet_loss_rate": 0.041,
                           "link_util_mean": 0.038, "throughput_rolling_mean": 0.022}
        },
        {
            "engine_id": "FD003_71",  "rul_predicted": 55.1,   "model_version": "xgb_v2_final",  # Final Improved — 15k trees, exp weights, GPU
            "importance": {"rssi_std_30": 0.081, "sinr_rolling_mean": 0.062,
                           "signal_quality_slope": 0.045, "antenna_vswr_trend": 0.027}
        },
        {
            "engine_id": "FD001_08",  "rul_predicted": 112.4,  "model_version": "xgb_v2_final",  # Final Improved — 15k trees, exp weights, GPU
            "importance": {"cpu_utilization_mean": 0.077, "processing_load_slope": 0.055,
                           "utilization_trend": 0.039, "load_rolling_std": 0.028}
        },
    ]

def print_banner(msg):
    w = 68
    print(f"\n{'═'*w}\n  {msg}\n{'═'*w}")

# ────────────────────────────────────────────────────────────────────────────
print_banner("AGENTIC PdM PIPELINE — FULL RUN (5 STATIONS)")
t_total = time.time()
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

predictions = make_predictions()

# Stage 1 — Interpreter
print("\n  ── Stage 1: Interpreter Agent ──")
t1 = time.time()
interpreter = InterpreterAgent()
alerts = []
for p in predictions:
    a = interpreter.interpret(
        station_id   = p["engine_id"],
        feature_row  = p["importance"],
        rul_override = p["rul_predicted"],
    )
    alerts.append(a)
    d = a.__dict__
    print(f"    {d['station_id']:<15} RUL={d['rul_cycles']:>7.1f}  "
          f"{d['urgency']:<10} subsystem={d['primary_subsystem']:<28} SLA={d['sla_hours']}h")
t1_ms = (time.time()-t1)*1000

# Stage 2 — RAG
print(f"\n  ── Stage 2: RAG Pipeline ──")
t2 = time.time()
rag_index = RAGIndex()
rag_index.load(INDEX_DIR)
rag_pipe  = RAGPipeline(rag_index)
bundles_raw = rag_pipe.retrieve_batch([a.__dict__ for a in alerts])
bundles = [dc_asdict(b) for b in bundles_raw]
t2_ms = (time.time()-t2)*1000
for a, b in zip(alerts, bundles):
    d = a.__dict__
    print(f"    {d['station_id']:<15} coverage={b['coverage_score']:.2f}  "
          f"candidates={b['n_candidates']}  lat={b['retrieval_latency_ms']:.1f}ms  "
          f"top={b['chunks'][0]['citation_ref']}")

# Stage 3 — Diagnostic
print(f"\n  ── Stage 3: Diagnostic Agent ──")
t3 = time.time()
diag_agent = DiagnosticAgent()
reports = []
for a, b in zip(alerts, bundles):
    r = diag_agent.diagnose(a.__dict__, b)
    reports.append(r)
    d = a.__dict__
    print(f"    {d['station_id']:<15} conf={r.diagnostic_confidence:.3f}  "
          f"grounding={r.grounding_rate:.3f}  "
          f"halluc={r.hallucination_rate:.3f}  "
          f"actions={len(r.action_recommendations)}")
t3_ms = (time.time()-t3)*1000

# Stage 4 — Planning
print(f"\n  ── Stage 4: Planning Agent ──")
t4 = time.time()
planner = PlanningAgent()
plans = []
for r in reports:
    p = planner.plan(r.handoff_to_planner)
    plans.append(p)
    print(f"    {p.station_id:<15} {len(p.actions)} actions  "
          f"€{p.total_cost_est:.0f}  "
          f"budget={'✓' if p.within_budget else '✗ OVER'}  "
          f"human={'⏸' if p.requires_human else '✓ auto'}")
t4_ms = (time.time()-t4)*1000

# Stage 5 — Execution
print(f"\n  ── Stage 5: Execution Agent ──")
t5 = time.time()
executor = ExecutionAgent()
logs = []
for plan, report in zip(plans, reports):
    auto_ok = (report.urgency != "Critical")
    log = executor.execute(plan, auto_approve_timeout=auto_ok)
    logs.append(log)
    print(f"    {log.station_id:<15} auto={log.actions_auto}  "
          f"timeout={log.actions_timeout}  "
          f"human={log.actions_human}  "
          f"lat={log.total_latency_ms:.1f}ms")
t5_ms = (time.time()-t5)*1000

total_ms = (time.time()-t_total)*1000

# ── KPI Summary table ──────────────────────────────────────────────────────
rows = []
for pred, alert, bundle, report, plan, log in zip(
        predictions, alerts, bundles, reports, plans, logs):
    d = alert.__dict__
    rows.append({
        "Station":       d["station_id"],
        "Urgency":       d["urgency"],
        "RUL (cycles)":  round(d["rul_cycles"],1),
        "Subsystem":     d["primary_subsystem"],
        "RAG Coverage":  bundle["coverage_score"],
        "Diag Conf":     round(report.diagnostic_confidence,3),
        "Grounding":     round(report.grounding_rate,3),
        "Hallucination": round(report.hallucination_rate,3),
        "# Actions":     len(report.action_recommendations),
        "Est Cost (€)":  plan.total_cost_est,
        "Budget OK":     plan.within_budget,
        "Auto":          log.actions_auto,
        "Timeout":       log.actions_timeout,
        "Human ✓":       log.actions_human,
    })

df = pd.DataFrame(rows)

print_banner("PIPELINE KPI SUMMARY")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:.3f}".format)
print(df.to_string(index=False))

print(f"\n  ── Aggregate Metrics ──")
print(f"    Mean grounding rate:        {df['Grounding'].mean():.3f}")
print(f"    Mean hallucination rate:    {df['Hallucination'].mean():.3f}")
print(f"    Mean diagnostic confidence: {df['Diag Conf'].mean():.3f}")
print(f"    Mean RAG coverage:          {df['RAG Coverage'].mean():.3f}")
print(f"    Total actions AUTO:         {df['Auto'].sum()}")
print(f"    Total actions TIMEOUT:      {df['Timeout'].sum()}")
print(f"    Total actions HUMAN ✓:      {df['Human ✓'].sum()}")
print(f"    Total estimated cost:       €{df['Est Cost (€)'].sum():.0f}")

print(f"\n  ── Stage Latency ──")
stages = [("Interpreter", t1_ms), ("RAG", t2_ms),
          ("Diagnostic", t3_ms),  ("Planning", t4_ms), ("Execution", t5_ms)]
for name, ms in stages:
    bar = "█" * max(1, int(ms/total_ms * 40))
    print(f"    {name:<15} {ms:>8.1f}ms  {ms/total_ms*100:>5.1f}%  {bar}")
print(f"    {'TOTAL':<15} {total_ms:>8.1f}ms")

# ── Save results ───────────────────────────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, f"pipeline_summary_{ts}.csv")
df.to_csv(csv_path, index=False)

run_record = {
    "timestamp":        ts,
    "total_latency_ms": round(total_ms, 2),
    "stage_latencies":  {n: round(ms,2) for n,ms in stages},
    "aggregate_kpis": {
        "mean_grounding_rate":        round(float(df["Grounding"].mean()),3),
        "mean_hallucination_rate":    round(float(df["Hallucination"].mean()),3),
        "mean_diagnostic_confidence": round(float(df["Diag Conf"].mean()),3),
        "mean_rag_coverage":          round(float(df["RAG Coverage"].mean()),3),
        "total_auto":                 int(df["Auto"].sum()),
        "total_timeout":              int(df["Timeout"].sum()),
        "total_human":                int(df["Human ✓"].sum()),
        "total_cost_eur":             float(df["Est Cost (€)"].sum()),
    },
    "stations": df.to_dict(orient="records"),
}
json_path = os.path.join(RESULTS_DIR, f"pipeline_run_{ts}.json")
with open(json_path, "w") as f:
    json.dump(run_record, f, indent=2, default=str)

print(f"\n  Saved CSV  → {csv_path}")
print(f"  Saved JSON → {json_path}")
print_banner("PIPELINE RUN COMPLETE ✓")
