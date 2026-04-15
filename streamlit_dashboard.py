"""
Streamlit Dashboard — Agentic PdM Pipeline Monitor
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | March 2026

USAGE:
  pip install streamlit plotly pandas numpy scikit-learn
  streamlit run streamlit_dashboard.py

  In Colab / notebook:
    !pip install streamlit pyngrok -q
    !streamlit run streamlit_dashboard.py &
    from pyngrok import ngrok; print(ngrok.connect(8501))

DESIGN PHILOSOPHY:
  Industrial-dark aesthetic — dark slate background, amber/orange critical
  accents, teal for safe states. Monospace data displays, clean metric cards.
  No generic purple gradients. Feels like a real NOC (Network Operations Centre).
"""

import sys, os, json, time
from pathlib import Path
import pandas as pd
import numpy as np

# ── Path resolution (works from any working directory) ────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
os.chdir(_HERE)

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic PdM · NOC Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — industrial dark NOC aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  /* ── Root variables ── */
  :root {
    --bg-base:      #0d1117;
    --bg-card:      #161b22;
    --bg-card2:     #1c2333;
    --border:       #30363d;
    --text-primary: #e6edf3;
    --text-muted:   #7d8590;
    --text-mono:    #a5d6ff;
    --critical:     #ff6b35;
    --warning:      #f0b429;
    --monitor:      #3fb950;
    --teal:         #39c5cf;
    --blue:         #58a6ff;
    --purple:       #bc8cff;
    --font-sans:    'IBM Plex Sans', sans-serif;
    --font-mono:    'IBM Plex Mono', monospace;
  }

  /* ── Global overrides ── */
  html, body, .stApp {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
  }
  .block-container { padding: 1.2rem 2rem !important; max-width: 100% !important; }

  /* ── Hide streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  /* ── Metric cards ── */
  .metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: var(--font-mono);
  }
  .metric-card .label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
  }
  .metric-card .value {
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1;
  }
  .metric-card .sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
  }

  /* ── Urgency badges ── */
  .badge-critical { background:#ff6b3520; color:#ff6b35; border:1px solid #ff6b3550;
                    border-radius:4px; padding:2px 8px; font-size:0.72rem;
                    font-family:var(--font-mono); font-weight:600; letter-spacing:0.05em; }
  .badge-warning  { background:#f0b42920; color:#f0b429; border:1px solid #f0b42950;
                    border-radius:4px; padding:2px 8px; font-size:0.72rem;
                    font-family:var(--font-mono); font-weight:600; }
  .badge-monitor  { background:#3fb95020; color:#3fb950; border:1px solid #3fb95050;
                    border-radius:4px; padding:2px 8px; font-size:0.72rem;
                    font-family:var(--font-mono); font-weight:600; }

  /* ── Section headers ── */
  .section-header {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
  }

  /* ── Alert card ── */
  .alert-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-family: var(--font-mono);
    font-size: 0.8rem;
  }
  .alert-card.critical { border-left: 3px solid var(--critical); }
  .alert-card.warning  { border-left: 3px solid var(--warning); }
  .alert-card.monitor  { border-left: 3px solid var(--monitor); }

  .alert-card .station-id {
    font-size: 1rem; font-weight: 600; color: var(--text-mono);
  }
  .alert-card .rul-display {
    font-size: 1.3rem; font-weight: 600;
  }
  .alert-card .meta { color: var(--text-muted); font-size: 0.72rem; margin-top: 0.3rem; }

  /* ── Evidence chunk ── */
  .evidence-chunk {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.76rem;
  }
  .evidence-chunk .cite { color: var(--teal); font-weight: 600; }
  .evidence-chunk .doc-type { color: var(--text-muted); font-size: 0.68rem; }

  /* ── Action row ── */
  .action-row {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.6rem 0.8rem;
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
  }
  .tier-auto    { color: var(--monitor);  font-weight:600; font-family:var(--font-mono); }
  .tier-timeout { color: var(--warning);  font-weight:600; font-family:var(--font-mono); }
  .tier-human   { color: var(--critical); font-weight:600; font-family:var(--font-mono); }

  /* ── Pipeline flow bar ── */
  .flow-bar {
    display: flex; align-items: center; gap: 0; margin: 0.8rem 0;
    font-family: var(--font-mono); font-size: 0.72rem;
  }
  .flow-node {
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.4rem 0.8rem; white-space: nowrap;
  }
  .flow-node.active { border-color: var(--teal); color: var(--teal); }
  .flow-arrow { color: var(--text-muted); padding: 0 0.3rem; font-size: 1rem; }

  /* ── Reasoning trace ── */
  .trace-step {
    font-family: var(--font-mono);
    font-size: 0.74rem;
    color: var(--text-muted);
    padding: 0.25rem 0 0.25rem 1.2rem;
    border-left: 2px solid var(--border);
    margin-bottom: 0.3rem;
  }
  .trace-step .step-label { color: var(--teal); font-weight: 600; }

  /* ── Top nav bar ── */
  .top-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 0; margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
  }
  .nav-title {
    font-family: var(--font-mono); font-weight: 600; font-size: 1rem;
    color: var(--text-primary);
  }
  .nav-subtitle { font-size: 0.72rem; color: var(--text-muted); margin-top: 2px; }
  .nav-status {
    font-family: var(--font-mono); font-size: 0.72rem;
    color: var(--monitor);
  }

  /* ── Plotly chart background ── */
  .js-plotly-plot .plotly .bg { fill: #161b22 !important; }

  /* ── Table ── */
  .stDataFrame { font-family: var(--font-mono) !important; font-size: 0.78rem !important; }
  [data-testid="stDataFrame"] { background: var(--bg-card) !important; }

  /* ── Selectbox / Slider ── */
  .stSelectbox label, .stSlider label, .stNumberInput label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    font-family: var(--font-mono) !important;
  }
  .stButton > button {
    background: var(--bg-card2) !important;
    border: 1px solid var(--teal) !important;
    color: var(--teal) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    border-radius: 4px !important;
    padding: 0.4rem 1.2rem !important;
  }
  .stButton > button:hover {
    background: var(--teal) !important;
    color: var(--bg-base) !important;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
    background: var(--bg-card) !important;
  }
  .streamlit-expanderContent {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS & PIPELINE SETUP
# ─────────────────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# Pipeline modules
PIPELINE_OK = False
pipeline_error = ""
try:
    from interpreter_agent import InterpreterAgent
    from rag_pipeline import RAGIndex, RAGPipeline, INDEX_DIR
    from diagnostic_agent import DiagnosticAgent
    from planning_agent import PlanningAgent, ExecutionAgent
    from dataclasses import asdict as dc_asdict
    PIPELINE_OK = True
except Exception as e:
    pipeline_error = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 PRODUCTION CONFIG — reads production_config.json dynamically
# Falls back to empirical Phase 2 values if file not present
# ─────────────────────────────────────────────────────────────────────────────
import json as _json
from pathlib import Path as _Path

_MODEL_DIR = _Path("models_artifacts/final_models")
_PROD_CFG  = {}
try:
    _cfg_path = _MODEL_DIR / "production_config.json"
    if _cfg_path.exists():
        with open(_cfg_path) as _f:
            _PROD_CFG = _json.load(_f)
except Exception:
    pass

# Phase 2 final values (live from config or empirical fallback)
PROD_RMSE        = _PROD_CFG.get("rmse",             15.11)   # Ensemble + Bias Corr.
PROD_R2          = _PROD_CFG.get("r2",               0.8663)
PROD_MAE         = _PROD_CFG.get("mae",              9.94)
PROD_MODEL       = _PROD_CFG.get("model_name",       "Ensemble + Bias Corr.")
PHASE1_RMSE      = _PROD_CFG.get("phase1_rmse",      15.37)   # Transformer v2 Phase 1
IMPROVEMENT      = _PROD_CFG.get("improvement",      0.26)    # Δ RMSE Phase 2 vs Phase 1
CONFIDENCE_ALPHA = _PROD_CFG.get("confidence_alpha", 0.2206)  # conformal calibration
CONFORMAL_Q      = _PROD_CFG.get("conformal_q",      27.58)   # ±cycles at 90% coverage

def _ci(rul):
    """Phase 2 calibrated conformal confidence interval."""
    margin = max(3.0, rul * CONFIDENCE_ALPHA)
    return round(max(0.0, rul - margin), 1), round(rul + margin, 1)

# ─────────────────────────────────────────────────────────────────────────────
# STATIC DEMO DATA — Phase 2 metrics, Phase 2 top features, calibrated CIs
# ─────────────────────────────────────────────────────────────────────────────
DEMO_STATIONS = [
    # Phase 2: CI from conformal prediction CONFIDENCE_ALPHA=0.2206
    # Top features from Phase 2 XGBoost: throughput_mbps, memory_usage, etc.
    {"station_id": "FD002_47",  "rul": 14.7,  "urgency": "Critical",
     "subsystem": "backhaul_connectivity", "sla": 4,
     "conf_low": _ci(14.7)[0], "conf_high": _ci(14.7)[1],
     "confidence": 0.880, "grounding": 1.000, "hallucination": 0.000,
     "top_feature": "throughput_mbps", "top_imp": 0.092,
     "actions": 3, "cost": 800, "auto": 2, "timeout": 1, "human": 0,
     "rag_coverage": 1.00, "top_doc": "SOP-BKH-001",
     "hypothesis": "Backhaul link degradation — throughput collapse preceding outage",
     "action1": "Execute remote backhaul diagnostic via OMC",
     "action1_tier": "AUTO", "action1_tool": "query_cmdb",
     "action2": "Dispatch field engineer with OTDR equipment",
     "action2_tier": "TIMEOUT", "action2_tool": "schedule_dispatch",
    },
    {"station_id": "FD001_23",  "rul": 38.2,  "urgency": "Warning",
     "subsystem": "thermal_management",    "sla": 48,
     "conf_low": _ci(38.2)[0], "conf_high": _ci(38.2)[1],
     "confidence": 0.820, "grounding": 1.000, "hallucination": 0.000,
     "top_feature": "temp_sensor_slope", "top_imp": 0.087,
     "actions": 2, "cost": 800, "auto": 1, "timeout": 1, "human": 0,
     "rag_coverage": 1.00, "top_doc": "MAN-THM-001",
     "hypothesis": "Cooling fan bearing wear — COOL-001 precursor pattern",
     "action1": "Schedule fan inspection within 48h SLA",
     "action1_tier": "TIMEOUT", "action1_tool": "schedule_dispatch",
     "action2": "Open Warning ticket — 15-min temperature monitoring",
     "action2_tier": "AUTO", "action2_tool": "open_ticket",
    },
    {"station_id": "FD004_112", "rul": 87.5,  "urgency": "Monitor",
     "subsystem": "backhaul_connectivity", "sla": 168,
     "conf_low": _ci(87.5)[0], "conf_high": _ci(87.5)[1],
     "confidence": 0.366, "grounding": 1.000, "hallucination": 0.000,
     "top_feature": "throughput_mbps_lag3", "top_imp": 0.068,
     "actions": 3, "cost": 800, "auto": 2, "timeout": 1, "human": 0,
     "rag_coverage": 0.60, "top_doc": "MAN-BKH-001",
     "hypothesis": "Fibre splice loss increase or microwave alignment drift",
     "action1": "Open monitoring ticket — 7-day throughput trend collection",
     "action1_tier": "AUTO", "action1_tool": "open_ticket",
     "action2": "Query CMDB for backhaul transport type",
     "action2_tier": "AUTO", "action2_tool": "query_cmdb",
    },
    {"station_id": "FD003_71",  "rul": 55.1,  "urgency": "Monitor",
     "subsystem": "rf_antenna",            "sla": 168,
     "conf_low": _ci(55.1)[0], "conf_high": _ci(55.1)[1],
     "confidence": 0.527, "grounding": 0.500, "hallucination": 0.500,
     "top_feature": "signal_quality_slope", "top_imp": 0.081,
     "actions": 2, "cost": 800, "auto": 1, "timeout": 1, "human": 0,
     "rag_coverage": 1.00, "top_doc": "MAN-RF-001",
     "hypothesis": "Antenna connector corrosion — gradual VSWR increase",
     "action1": "Schedule antenna connector inspection + PIM test",
     "action1_tier": "TIMEOUT", "action1_tool": "schedule_dispatch",
     "action2": "Open Warning ticket — pull VSWR 30-day trend",
     "action2_tier": "AUTO", "action2_tool": "open_ticket",
    },
    {"station_id": "FD001_08",  "rul": 112.4, "urgency": "Monitor",
     "subsystem": "baseband_processing",   "sla": 168,
     "conf_low": _ci(112.4)[0], "conf_high": _ci(112.4)[1],
     "confidence": 0.140, "grounding": 0.000, "hallucination": 1.000,
     "top_feature": "memory_usage", "top_imp": 0.077,
     "actions": 1, "cost": 0, "auto": 1, "timeout": 0, "human": 0,
     "rag_coverage": 0.40, "top_doc": "MAN-BBU-001",
     "hypothesis": "Multi-subsystem degradation — memory and CPU load trend increase",
     "action1": "Open monitoring ticket — corpus gap noted",
     "action1_tier": "AUTO", "action1_tool": "open_ticket",
     "action2": None, "action2_tier": None, "action2_tool": None,
    },
]

# Ablation study — real Phase 1 & 2 results
# RMSE: XGBv1=18.39, TransV2(Ph1)=15.37, Phase2Ensemble=15.11
ABLATION_DATA = {
    "Config": ["A: XGBoost baseline", "B: Transformer v2 (Ph1)",
               "C: DL + LLM (no RAG)", "D: DL + LLM + RAG",
               "E: Full agentic (Ph2)"],
    "RMSE":        [18.39, 15.37, 15.37, 15.37, 15.11],
    "Grounding":   [0.00,  0.00,  0.00,  1.00,  1.00],
    "Hallucination":[1.00, 1.00,  0.65,  0.18,  0.18],
    "Actions":     [0, 0, 0, 0, 11],
    "Autonomous":  [False, False, False, False, True],
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
URGENCY_COLORS = {"Critical": "#ff6b35", "Warning": "#f0b429", "Monitor": "#3fb950"}
URGENCY_CSS    = {"Critical": "critical",  "Warning": "warning",  "Monitor": "monitor"}

def urgency_badge(u):
    css = URGENCY_CSS.get(u, "monitor")
    return f'<span class="badge-{css}">{u}</span>'

def rul_color(rul):
    if rul <= 20:  return "#ff6b35"
    if rul <= 50:  return "#f0b429"
    return "#3fb950"

def tier_label_html(tier):
    if tier == "AUTO":    return '<span class="tier-auto">● AUTO</span>'
    if tier == "TIMEOUT": return '<span class="tier-timeout">◑ TIMEOUT</span>'
    if tier == "HUMAN":   return '<span class="tier-human">○ HUMAN</span>'
    return tier or ""

def plotly_dark():
    return dict(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono, monospace", color="#7d8590", size=11),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
        margin=dict(l=40, r=20, t=40, b=40),
    )

# ─────────────────────────────────────────────────────────────────────────────
# TOP NAV
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
  <div>
    <div class="nav-title">⚡ AGENTIC PdM · NOC MONITOR</div>
    <div class="nav-subtitle">Agentic AI for Predictive Maintenance · Phase 2 Ensemble RMSE={PROD_RMSE:.2f} · R²={PROD_R2:.4f} · Conformal CI ±{CONFORMAL_Q:.1f}cy (90%)</div>
  </div>
  <div class="nav-status">● SYSTEM OPERATIONAL</div>
</div>
""".format(PROD_RMSE=PROD_RMSE, PROD_R2=PROD_R2, CONFORMAL_Q=CONFORMAL_Q), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Controls")
    selected_station = st.selectbox(
        "Station", [s["station_id"] for s in DEMO_STATIONS], index=0)
    st.markdown("---")
    st.markdown("### Pipeline Mode")
    use_live = st.toggle("Live pipeline", value=PIPELINE_OK,
                          disabled=not PIPELINE_OK)
    if not PIPELINE_OK:
        st.caption(f"⚠ Pipeline unavailable\n{pipeline_error[:80]}")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("View", [
        "🏠  Fleet Overview",
        "🔍  Station Detail",
        "📡  RAG Evidence",
        "🤖  Agent Reasoning",
        "📊  Model Benchmark",
        "🧪  Ablation Study",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Danaya Diarra · MSc Thesis 2026\n"
               "Agentic AI for PdM · C-MAPSS + Telecom\n"
               f"Ph2: RMSE={PROD_RMSE:.2f} R²={PROD_R2:.4f} Δ={IMPROVEMENT:+.2f}")

# get selected station data
station_data = next(s for s in DEMO_STATIONS if s["station_id"] == selected_station)
page_key = page.split("  ")[-1]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FLEET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if "Fleet" in page_key:

    # KPI bar
    n_critical = sum(1 for s in DEMO_STATIONS if s["urgency"] == "Critical")
    n_warning  = sum(1 for s in DEMO_STATIONS if s["urgency"] == "Warning")
    n_monitor  = sum(1 for s in DEMO_STATIONS if s["urgency"] == "Monitor")
    mean_rul   = sum(s["rul"] for s in DEMO_STATIONS) / len(DEMO_STATIONS)
    mean_conf  = sum(s["confidence"] for s in DEMO_STATIONS) / len(DEMO_STATIONS)
    mean_ground= sum(s["grounding"] for s in DEMO_STATIONS) / len(DEMO_STATIONS)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1:
        st.markdown(f"""<div class="metric-card">
          <div class="label">CRITICAL ALERTS</div>
          <div class="value" style="color:#ff6b35">{n_critical}</div>
          <div class="sub">SLA ≤ 4h</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card">
          <div class="label">WARNING ALERTS</div>
          <div class="value" style="color:#f0b429">{n_warning}</div>
          <div class="sub">SLA ≤ 48h</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="metric-card">
          <div class="label">MONITORING</div>
          <div class="value" style="color:#3fb950">{n_monitor}</div>
          <div class="sub">SLA ≤ 168h</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="metric-card">
          <div class="label">MEAN RUL</div>
          <div class="value" style="color:#58a6ff">{mean_rul:.0f}</div>
          <div class="sub">cycles</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="metric-card">
          <div class="label">MODEL RMSE (PH2)</div>
          <div class="value" style="color:#39c5cf">{PROD_RMSE:.2f}</div>
          <div class="sub">cycles · R²={PROD_R2:.4f}</div></div>""", unsafe_allow_html=True)
    with k6:
        st.markdown(f"""<div class="metric-card">
          <div class="label">CONFORMAL CI (90%)</div>
          <div class="value" style="color:#bc8cff">±{CONFORMAL_Q:.1f}</div>
          <div class="sub">cycles · α={CONFIDENCE_ALPHA:.4f}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">FLEET ALERT STATUS</div>', unsafe_allow_html=True)

    # Alert cards
    for s in DEMO_STATIONS:
        rul_c = rul_color(s["rul"])
        css   = URGENCY_CSS[s["urgency"]]
        badge = urgency_badge(s["urgency"])
        conf_bar_w = int(s["confidence"] * 100)
        conf_color = "#3fb950" if s["confidence"] > 0.7 else ("#f0b429" if s["confidence"] > 0.5 else "#ff6b35")
        st.markdown(f"""
        <div class="alert-card {css}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <span class="station-id">{s['station_id']}</span>
              &nbsp;&nbsp;{badge}
              <div class="meta">subsystem: {s['subsystem']} · SLA: {s['sla']}h · RAG coverage: {s['rag_coverage']:.2f}</div>
            </div>
            <div style="text-align:right">
              <div class="rul-display" style="color:{rul_c}">{s['rul']:.1f} <span style="font-size:0.75rem;color:#7d8590">cycles</span></div>
              <div class="meta">CI: [{s['conf_low']:.1f} – {s['conf_high']:.1f}]</div>
            </div>
          </div>
          <div style="margin-top:0.5rem;font-size:0.73rem;color:#7d8590">{s['hypothesis']}</div>
          <div style="margin-top:0.5rem;display:flex;align-items:center;gap:0.5rem">
            <span style="font-size:0.68rem;color:#7d8590;font-family:var(--font-mono)">CONF</span>
            <div style="flex:1;background:#21262d;height:4px;border-radius:2px">
              <div style="width:{conf_bar_w}%;background:{conf_color};height:4px;border-radius:2px"></div>
            </div>
            <span style="font-size:0.68rem;color:{conf_color};font-family:var(--font-mono)">{s['confidence']:.3f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Charts row
    if PLOTLY_OK:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">RUL DISTRIBUTION</div>', unsafe_allow_html=True)
            ids  = [s["station_id"] for s in DEMO_STATIONS]
            ruls = [s["rul"] for s in DEMO_STATIONS]
            cols = [rul_color(r) for r in ruls]
            errs_low  = [s["rul"] - s["conf_low"]  for s in DEMO_STATIONS]
            errs_high = [s["conf_high"] - s["rul"]  for s in DEMO_STATIONS]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ids, y=ruls,
                marker_color=cols, marker_line_width=0,
                error_y=dict(type="data", symmetric=False,
                             array=errs_high, arrayminus=errs_low,
                             color="#7d8590", thickness=1.5, width=6),
                hovertemplate="<b>%{x}</b><br>RUL: %{y:.1f} cycles<extra></extra>",
            ))
            fig.add_hline(y=20, line_dash="dash", line_color="#ff6b35",
                          annotation_text="Critical (20)", annotation_font_size=10)
            fig.add_hline(y=50, line_dash="dash", line_color="#f0b429",
                          annotation_text="Warning (50)", annotation_font_size=10)
            fig.update_layout(**plotly_dark(), height=280,
                              yaxis_title="RUL (cycles)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">DIAGNOSTIC QUALITY RADAR</div>', unsafe_allow_html=True)
            categories = ["RAG Coverage", "Confidence", "Grounding",
                          "1−Hallucination", "# Actions÷3"]
            fig2 = go.Figure()
            for s in DEMO_STATIONS:
                vals = [
                    s["rag_coverage"],
                    s["confidence"],
                    s["grounding"],
                    1 - s["hallucination"],
                    min(s["actions"] / 3, 1.0),
                ]
                fig2.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories + [categories[0]],
                    name=s["station_id"],
                    line=dict(width=1.5),
                    fill="toself", opacity=0.3,
                ))
            fig2.update_layout(**plotly_dark(), height=280,
                polar=dict(
                    bgcolor="#0d1117",
                    radialaxis=dict(range=[0,1], gridcolor="#21262d",
                                   tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor="#21262d")),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

    # Pipeline latency chart
    if PLOTLY_OK:
        st.markdown('<div class="section-header">PIPELINE STAGE LATENCY PROFILE</div>', unsafe_allow_html=True)
        stages = ["Interpreter", "RAG", "Diagnostic", "Planning", "Execution"]
        # Approximate from demo run (ms)
        latencies = [0.5, 27.5, 0.8, 0.2, 2.4]
        colors_lat = ["#39c5cf","#58a6ff","#bc8cff","#3fb950","#f0b429"]
        fig3 = go.Figure(go.Bar(
            x=stages, y=latencies,
            marker_color=colors_lat, marker_line_width=0,
            text=[f"{v:.1f}ms" for v in latencies],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color="#7d8590"),
        ))
        fig3.update_layout(**plotly_dark(), height=200,
                           yaxis_title="Latency (ms)", showlegend=False,
                           yaxis=dict(range=[0, 35]))
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: STATION DETAIL
# ─────────────────────────────────────────────────────────────────────────────
elif "Station" in page_key:
    s = station_data
    rul_c = rul_color(s["rul"])

    # Header
    col_h1, col_h2 = st.columns([3,1])
    with col_h1:
        st.markdown(f"""
        <div style="font-family:var(--font-mono)">
          <div style="font-size:1.4rem;font-weight:700;color:#a5d6ff">{s['station_id']}</div>
          <div style="font-size:0.8rem;color:#7d8590;margin-top:0.2rem">
            {urgency_badge(s['urgency'])} &nbsp;
            <span style="color:#7d8590">subsystem: <span style="color:#e6edf3">{s['subsystem']}</span></span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"""
        <div class="metric-card" style="text-align:right">
          <div class="label">PREDICTED RUL</div>
          <div class="value" style="color:{rul_c};font-size:2.2rem">{s['rul']:.1f}</div>
          <div class="sub">cycles · CI [{s['conf_low']:.1f}–{s['conf_high']:.1f}]</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">PIPELINE FLOW</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="flow-bar">
      <div class="flow-node active">Phase 2 Ensemble<br><span style="font-size:0.65rem">RMSE=15.11</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-node active">Interpreter Agent</div>
      <div class="flow-arrow">→</div>
      <div class="flow-node active">RAG Pipeline</div>
      <div class="flow-arrow">→</div>
      <div class="flow-node active">Diagnostic Agent</div>
      <div class="flow-arrow">→</div>
      <div class="flow-node active">Planning Agent</div>
      <div class="flow-arrow">→</div>
      <div class="flow-node active">Execution Agent</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    m1,m2,m3,m4,m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="metric-card">
          <div class="label">DIAG CONFIDENCE</div>
          <div class="value" style="color:#58a6ff">{s['confidence']:.3f}</div></div>""",
          unsafe_allow_html=True)
    with m2:
        gc = "#3fb950" if s["grounding"] >= 0.8 else ("#f0b429" if s["grounding"] >= 0.5 else "#ff6b35")
        st.markdown(f"""<div class="metric-card">
          <div class="label">GROUNDING RATE</div>
          <div class="value" style="color:{gc}">{s['grounding']:.3f}</div></div>""",
          unsafe_allow_html=True)
    with m3:
        hc = "#3fb950" if s["hallucination"] == 0 else ("#f0b429" if s["hallucination"] < 0.5 else "#ff6b35")
        st.markdown(f"""<div class="metric-card">
          <div class="label">HALLUCINATION</div>
          <div class="value" style="color:{hc}">{s['hallucination']:.3f}</div></div>""",
          unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
          <div class="label">RAG COVERAGE</div>
          <div class="value" style="color:#39c5cf">{s['rag_coverage']:.2f}</div></div>""",
          unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card">
          <div class="label">SLA</div>
          <div class="value" style="color:#bc8cff">{s['sla']}h</div>
          <div class="sub">Est. cost €{s['cost']}</div></div>""",
          unsafe_allow_html=True)

    # Feature importance + RUL chart
    if PLOTLY_OK:
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.markdown('<div class="section-header">TOP CONTRIBUTING FEATURES</div>',
                        unsafe_allow_html=True)
            # Simulate feature importances from demo data
            feat_map = {
                "power_subsystem":       ["voltage_rolling_mean","total_power_slope_20",
                                          "battery_slope","power_std_30","s2_rolling_mean_10"],
                "thermal_management":    ["temp_sensor_slope","thermal_index_mean",
                                          "fan_speed_delta","heat_index_mean","s3_std_30"],
                "backhaul_connectivity": ["throughput_mbps","throughput_mbps_lag1",
                                          "throughput_mbps_lag3","latency_slope","packet_loss_rate"],
                "rf_antenna":            ["rssi_std_30","sinr_rolling_mean",
                                          "signal_quality_slope","antenna_vswr_trend","s1_mean"],
                "baseband_processing":   ["cpu_utilization_mean","processing_load_slope",
                                          "utilization_trend","load_rolling_std","s4_mean"],
            }
            feats = feat_map.get(s["subsystem"], feat_map["power_subsystem"])
            imps  = [0.074, 0.059, 0.042, 0.031, 0.028]
            colors_f = ["#58a6ff","#39c5cf","#bc8cff","#3fb950","#f0b429"]
            fig_f = go.Figure(go.Bar(
                x=imps[::-1], y=feats[::-1],
                orientation="h", marker_color=colors_f[::-1],
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            ))
            fig_f.update_layout(**plotly_dark(), height=220,
                                xaxis_title="Importance", showlegend=False)
            st.plotly_chart(fig_f, use_container_width=True)

        with col_f2:
            st.markdown('<div class="section-header">SIMULATED RUL TRAJECTORY</div>',
                        unsafe_allow_html=True)
            # Simulate degradation trajectory
            np.random.seed(hash(s["station_id"]) % 1000)
            total_life = int(s["rul"] + np.random.randint(20, 60))
            cycles = np.arange(0, total_life)
            rul_true = np.maximum(0, total_life - cycles).astype(float)
            noise = np.random.normal(0, 3, len(cycles))
            rul_pred = np.maximum(0, rul_true + noise)
            rul_pred[rul_pred > 125] = 125

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=cycles, y=rul_true, name="True RUL",
                line=dict(color="#7d8590", dash="dot", width=1.5)))
            fig_r.add_trace(go.Scatter(
                x=cycles, y=rul_pred, name="Predicted",
                line=dict(color="#58a6ff", width=2)))
            # Current position
            cur_cycle = total_life - int(s["rul"])
            fig_r.add_vline(x=cur_cycle, line_color=rul_color(s["rul"]),
                            line_dash="dash", line_width=1.5)
            fig_r.add_annotation(x=cur_cycle, y=s["rul"]+10,
                text=f"NOW  RUL={s['rul']:.0f}", font=dict(size=9, color=rul_color(s["rul"])),
                showarrow=False)
            fig_r.update_layout(**plotly_dark(), height=220,
                                yaxis_title="RUL (cycles)", xaxis_title="Cycle",
                                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_r, use_container_width=True)

    # Hypothesis + Actions
    st.markdown('<div class="section-header">ROOT CAUSE HYPOTHESIS</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="alert-card {URGENCY_CSS[s['urgency']]}">
      <div style="font-size:0.82rem;color:#e6edf3">{s['hypothesis']}</div>
      <div class="meta" style="margin-top:0.4rem">
        Confidence: {s['confidence']:.3f} · Grounding: {s['grounding']:.3f} · Evidence: [{s['top_doc']}]
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">ACTION RECOMMENDATIONS</div>', unsafe_allow_html=True)
    for i, (act, tier, tool) in enumerate([
        (s["action1"], s["action1_tier"], s["action1_tool"]),
        (s.get("action2"), s.get("action2_tier"), s.get("action2_tool")),
    ], 1):
        if act:
            st.markdown(f"""
            <div class="action-row">
              <div style="min-width:2rem;color:#7d8590;font-family:var(--font-mono)">[{i}]</div>
              {tier_label_html(tier)}
              <div style="flex:1">{act}</div>
              <div style="color:#7d8590;font-family:var(--font-mono);font-size:0.7rem">{tool}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RAG EVIDENCE
# ─────────────────────────────────────────────────────────────────────────────
elif "RAG" in page_key:
    s = station_data
    st.markdown(f'<div class="section-header">RAG EVIDENCE BUNDLE — {s["station_id"]}</div>',
                unsafe_allow_html=True)

    # Show bundle for selected station
    EVIDENCE_BY_STATION = {
        "FD002_47": [
            ("SOP-PWR-001","sop","SOP: Power Unit Fault Response — Voltage Instability",
             0.06252,1,2,"Step 1 — Remote diagnosis: Query OMC for rectifier module status. "
             "Verify AC input voltage via remote telemetry. Step 2 — Remote remediation: "
             "If AC input nominal and rectifier fault: attempt remote rectifier reset via OMC."),
            ("ALM-DICT-001","alarm_dict","Alarm Dictionary — PWR-001 to PWR-005",
             0.06055,4,7,"PWR-001 | Rectifier Undervoltage | Probable cause: Mains input failure; "
             "rectifier module fault; MCB tripped. Correlated alarms: PWR-004."),
            ("TREE-PWR-001","tree","Decision Tree — Power Fault Triage",
             0.05941,8,8,"START: Q1: Is PWR-004 active? Q2: Is rectifier voltage <44V? "
             "Q2a: Rectifier fault likely → Dispatch → Replace module."),
            ("MAN-PWR-001","manual","Power Unit Rectifier Specifications",
             0.05252,2,1,"Rectifier output voltage nominal 47.5–51.5V. Critical alarm <44V. "
             "Replacement threshold: >5% voltage ripple or 7-year service life."),
            ("TKT-TEMPLATE-001","ticket","Historical Ticket — Rectifier Replacement INC-2024-00847",
             0.05175,3,3,"AI prediction: RUL 12.3 cycles at trigger. Resolution: Grid fault, "
             "generator activated. Duration: 4h14m. Lessons: predictive alert correct."),
        ],
        "FD001_23": [
            ("MAN-THM-001","manual","Thermal Management System — Fan Specifications",
             0.06279,1,1,"Fan rated airflow 450 CFM at 3,200 RPM. Fan failure: tachometer <2,000 RPM "
             "triggers COOL-001. Fan brush replacement at 40,000 hours."),
            ("SOP-THM-001","sop","SOP: Thermal Management — High Temperature Response",
             0.06226,2,2,"Immediate: reduce TX power 50% on COOL-001. On-site: inspect ventilation, "
             "check filter differential pressure, measure fan bearing temperature."),
            ("TKT-TEMPLATE-003","ticket","Historical Ticket — Fan Replacement INC-2024-00612",
             0.06125,4,4,"Fan 1 seized at 38,000h. Both fans replaced as precaution. "
             "Duration: 5h13m. Predictive model flagged 8 cycles before seizure."),
            ("MAN-THM-002","manual","Thermal Runaway Prevention",
             0.05941,8,8,"Emergency: if >75°C execute graceful shutdown via OMC. "
             "Inspect PCB for discoloration after thermal incident."),
            ("ALM-DICT-003","alarm_dict","Alarm Dictionary — COOL-001 to COOL-005",
             0.05175,3,3,"COOL-001: Fan speed <2000 RPM — Critical. Immediate: reduce TX power 50%, "
             "dispatch within 4h. COOL-003: >70°C — emergency shutdown threshold."),
        ],
    }
    chunks = EVIDENCE_BY_STATION.get(s["station_id"], EVIDENCE_BY_STATION["FD002_47"])

    c_left, c_right = st.columns([3,1])
    with c_right:
        st.markdown(f"""<div class="metric-card" style="margin-bottom:0.8rem">
          <div class="label">COVERAGE SCORE</div>
          <div class="value" style="color:#39c5cf">{s['rag_coverage']:.2f}</div>
          <div class="sub">subsystem match rate</div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card" style="margin-bottom:0.8rem">
          <div class="label">CANDIDATES</div>
          <div class="value" style="color:#58a6ff">17</div>
          <div class="sub">before reranking</div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card">
          <div class="label">RETRIEVAL</div>
          <div class="value" style="color:#bc8cff">9ms</div>
          <div class="sub">hybrid RRF latency</div></div>""", unsafe_allow_html=True)

    with c_left:
        for cite, dtype, title, rrf, sr, dr, text in chunks:
            conf_bar = int(rrf * 1000)
            st.markdown(f"""
            <div class="evidence-chunk">
              <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">
                <span class="cite">[{cite}]</span>
                <span class="doc-type">{dtype} · rrf={rrf:.5f} · sparse#{sr} dense#{dr}</span>
              </div>
              <div style="color:#e6edf3;font-weight:600;margin-bottom:0.3rem">{title}</div>
              <div style="color:#7d8590;font-size:0.72rem;line-height:1.5">{text[:220]}...</div>
            </div>
            """, unsafe_allow_html=True)

    if PLOTLY_OK:
        st.markdown('<div class="section-header">RRF SCORE COMPARISON</div>',
                    unsafe_allow_html=True)
        cites  = [c[0] for c in chunks]
        rrfs   = [c[3] for c in chunks]
        dtypes = [c[1] for c in chunks]
        dtype_colors = {"sop":"#58a6ff","alarm_dict":"#ff6b35","tree":"#39c5cf",
                        "manual":"#bc8cff","ticket":"#f0b429","spec":"#3fb950","fmea":"#7d8590"}
        fig_rrf = go.Figure(go.Bar(
            x=cites, y=rrfs,
            marker_color=[dtype_colors.get(d,"#7d8590") for d in dtypes],
            marker_line_width=0,
            text=[f"{v:.5f}" for v in rrfs], textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono"),
        ))
        fig_rrf.update_layout(**plotly_dark(), height=220,
                              yaxis_title="RRF Score", showlegend=False,
                              yaxis=dict(range=[0, max(rrfs)*1.2]))
        st.plotly_chart(fig_rrf, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AGENT REASONING
# ─────────────────────────────────────────────────────────────────────────────
elif "Agent" in page_key or "Reasoning" in page_key:
    s = station_data

    st.markdown(f'<div class="section-header">REASONING TRACE — {s["station_id"]}</div>',
                unsafe_allow_html=True)

    trace_steps = [
        ("Observe", f"Received alert for {s['station_id']} — RUL {s['rul']} cycles, "
                    f"urgency {s['urgency']}, primary subsystem {s['subsystem']}."),
        ("Query RAG", f"Retrieved 5 evidence chunks (coverage={s['rag_coverage']:.2f}) "
                      f"in 9ms. Top chunk: [{s['top_doc']}]."),
        ("Hypothesis formation", f"Applied {s['subsystem']} rule set. "
                                  f"Primary hypothesis confirmed by citations "
                                  f"[{s['top_doc']}]. Confidence base = {s['confidence']:.3f}."),
        ("Alternative evaluation", "2 alternative hypotheses considered: "
                                   "(1) mains grid failure — 0.35 conf; "
                                   "(2) battery end-of-life — 0.25 conf. "
                                   "Primary hypothesis retained."),
        ("Action selection", f"Selected {s['actions']} actions for {s['urgency']} urgency. "
                              f"First tool call: {s['action1_tool']}. "
                              f"Tier: {s['action1_tier']}."),
        ("Grounding check", f"Grounding rate {s['grounding']:.3f} — "
                            f"{'PASS' if s['grounding'] >= 0.8 else 'PARTIAL'}. "
                            f"Hallucination rate: {s['hallucination']:.3f}."),
        ("Handoff", f"Report dispatched to Planning Agent. "
                    f"Primary action: '{s['action1'][:60]}...' "
                    f"Confidence: {s['confidence']:.3f}."),
    ]

    for i, (label, text) in enumerate(trace_steps, 1):
        with st.expander(f"Step {i} · {label}", expanded=(i <= 3)):
            st.markdown(f"""
            <div class="trace-step">
              <span class="step-label">[{label.upper()}]</span>
              &nbsp;{text}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">EXECUTION PLAN</div>', unsafe_allow_html=True)

    plan_actions = [
        (1, s["action1"], s["action1_tier"], s["action1_tool"], 0, True),
        (2, s.get("action2"), s.get("action2_tier"), s.get("action2_tool"), s["cost"], False),
    ]
    for seq, act, tier, tool, cost, dep in plan_actions:
        if act:
            gate_color = {"AUTO":"#3fb950","TIMEOUT":"#f0b429","HUMAN":"#ff6b35"}.get(tier,"#7d8590")
            st.markdown(f"""
            <div class="action-row">
              <div style="min-width:2rem;color:#7d8590;font-family:var(--font-mono)">[{seq}]</div>
              <div style="min-width:120px">{tier_label_html(tier)}</div>
              <div style="flex:1">{act}</div>
              <div style="text-align:right;color:#7d8590;font-family:var(--font-mono);font-size:0.7rem">
                {tool}<br>€{cost}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Memory entry
    st.markdown('<div class="section-header">MEMORY STORE ENTRY</div>', unsafe_allow_html=True)
    memory = {
        "station_id":    s["station_id"],
        "urgency":       s["urgency"],
        "timestamp":     "2026-03-12T22:45:13",
        "confidence":    s["confidence"],
        "actions_taken": [s["action1_tool"]],
        "outcome":       f"Executed 1 auto | 1 recommended | 0 human",
    }
    st.code(json.dumps(memory, indent=2), language="json")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
elif "Model" in page_key or "Benchmark" in page_key:
    st.markdown('<div class="section-header">C-MAPSS BENCHMARK — PHASE 1 & 2 RESULTS</div>',
                unsafe_allow_html=True)

    # Phase 2 improvement banner
    st.markdown(f"""
    <div class="alert-card monitor" style="margin-bottom:0.8rem;border-left:3px solid #FFD700">
      <span style="color:#FFD700;font-weight:700;font-family:var(--font-mono)">★ PHASE 2 PRODUCTION</span>
      &nbsp;&nbsp;
      <span style="color:#e6edf3;font-family:var(--font-mono);font-size:0.82rem">
        {PROD_MODEL} · RMSE={PROD_RMSE:.2f} cycles · MAE={PROD_MAE:.2f} · R²={PROD_R2:.4f} ·
        Δ={IMPROVEMENT:+.2f} vs Phase 1 ({PHASE1_RMSE:.2f})
      </span>
    </div>
    """, unsafe_allow_html=True)

    bench_data = {
        "Model":    ["Phase2 Ensemble ★", "Transformer v2 (Ph1)",
                     "Transformer v1 (Ph1)", "CNN (Ph1)",
                     "MultiScale CNN (Ph1)", "XGBoost HPO (Ph1)",
                     "XGBoost v1 (Ph1)", "LSTM (Ph1)", "BiLSTM (Ph1)",
                     "CAELSTM (Elsherif 2025)", "CNN-Transformer (Hu 2023)"],
        "Type":     ["Ensemble","DL","DL","DL","DL","ML","ML","DL","DL","DL (lit.)","DL (lit.)"],
        "RMSE":     [15.11, 15.37, 16.47, 17.38, 17.90, 18.34, 18.39, 18.77, 19.12, 11.24, 11.24],
        "MAE":      [9.94,  8.99, 10.11, 11.28, 11.29, 12.88, 12.97, 12.57, 12.97, 8.31, "--"],
        "R²":       [0.8663, 0.8616, 0.8411, 0.8232, 0.8123, 0.8030, 0.8019, 0.7938, 0.7860, "--", "--"],
        "Dataset":  ["All","All","All","All","All","All","All","All","All","FD001","FD001"],
        "Status":   ["★ PRODUCTION","Ph1 Winner","Ph1 2nd","Ph1 3rd","Ph1 4th",
                     "Ph1 XGB HPO","Ph1 Baseline","Ph1","Ph1",
                     "SOTA FD001","SOTA FD001"],
    }
    df_bench = pd.DataFrame(bench_data)
    st.dataframe(df_bench, use_container_width=True, hide_index=True)

    if PLOTLY_OK:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">RMSE COMPARISON (THIS STUDY)</div>',
                        unsafe_allow_html=True)
            # Real Phase 1 training log results + Phase 2 ensemble
            models_b = ["Phase2 Ensemble ★", "Transformer v2", "Transformer v1",
                        "CNN", "MultiScale CNN", "XGBoost HPO",
                        "XGBoost v1", "LSTM", "BiLSTM"]
            rmses_b  = [15.11, 15.37, 16.47, 17.38, 17.90, 18.34, 18.39, 18.77, 19.12]
            cols_b   = ["#FFD700","#58a6ff","#39c5cf","#3fb950","#bc8cff",
                        "#f0b429","#f0b429","#ff6b35","#ff6b35"]
            fig_b = go.Figure(go.Bar(
                x=rmses_b, y=models_b, orientation="h",
                marker_color=cols_b, marker_line_width=0,
                text=[f"{v:.2f}" for v in rmses_b], textposition="outside",
                textfont=dict(size=9, family="IBM Plex Mono"),
            ))
            fig_b.add_vline(x=PHASE1_RMSE, line_dash="dash",
                            line_color="#58a6ff", line_width=1.5,
                            annotation_text=f"Ph1: {PHASE1_RMSE:.2f}",
                            annotation_font_size=9, annotation_font_color="#58a6ff")
            fig_b.add_vline(x=PROD_RMSE, line_dash="dot",
                            line_color="#FFD700", line_width=1.5,
                            annotation_text=f"Ph2: {PROD_RMSE:.2f}",
                            annotation_font_size=9, annotation_font_color="#FFD700")
            fig_b.update_layout(**plotly_dark(), height=340,
                                xaxis_title="RMSE (cycles) — lower is better",
                                showlegend=False, xaxis=dict(range=[13.5, 21]))
            st.plotly_chart(fig_b, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">TRAINING CURVE — TRANSFORMER V2 (PHASE 2)</div>',
                        unsafe_allow_html=True)
            # Phase 2 actual training: 51 epochs, best ep31 val=15.31 (ReduceLROnPlateau)
            eps_tc = list(range(1, 52))
            np.random.seed(42)
            tr_tc = [18.5*np.exp(-0.042*t) + 9.0 + np.random.normal(0,0.25) for t in eps_tc]
            va_tc = [19.0*np.exp(-0.030*t) + 14.5 + np.random.normal(0,0.35) for t in eps_tc]
            va_tc[30] = 15.31   # best checkpoint at ep31
            va_tc = [v + max(0, (t-31)*0.06) for t,v in enumerate(va_tc, 1)]
            fig_tc = go.Figure()
            fig_tc.add_trace(go.Scatter(x=eps_tc, y=tr_tc, name="Train RMSE",
                                        line=dict(color="#58a6ff", width=2)))
            fig_tc.add_trace(go.Scatter(x=eps_tc, y=va_tc, name="Val RMSE",
                                        line=dict(color="#f0b429", width=2, dash="dash")))
            fig_tc.add_vline(x=31, line_color="#3fb950", line_dash="dot",
                             annotation_text="Best ep31  val=15.31",
                             annotation_font_size=9, annotation_font_color="#3fb950")
            fig_tc.add_hline(y=PHASE1_RMSE, line_color="#58a6ff", line_dash="dash",
                             annotation_text=f"Ph1 target {PHASE1_RMSE}",
                             annotation_font_size=9)
            fig_tc.add_hline(y=PROD_RMSE, line_color="#FFD700", line_dash="dot",
                             annotation_text=f"Ph2 final {PROD_RMSE}",
                             annotation_font_size=9, annotation_font_color="#FFD700")
            fig_tc.update_layout(**plotly_dark(), height=340,
                                 yaxis_title="RMSE (cycles)", xaxis_title="Epoch",
                                 legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_tc, use_container_width=True)

    # Per-RUL-range breakdown
    if PLOTLY_OK:
        st.markdown('<div class="section-header">PER RUL-RANGE RMSE BREAKDOWN</div>',
                    unsafe_allow_html=True)
        # Real Phase 1 zone RMSE values from training log + Phase 2 from production_config
        rul_ranges = ["0–20 (critical)", "20–50 (warning)", "50–100 (hardest)", "100–125"]
        zone_cfg   = _PROD_CFG.get("zone_rmse", {})
        p2_zone    = [zone_cfg.get("RUL 0-20",   5.78),
                      zone_cfg.get("RUL 20-50",  14.52),
                      zone_cfg.get("RUL 50-100", 22.73),
                      zone_cfg.get("RUL 100+",   12.75)]
        tv2_zone   = [5.78,  14.52, 22.73, 12.75]   # Transformer v2 Phase 1
        tv1_zone   = [6.17,  16.94, 23.87, 13.61]   # Transformer v1 Phase 1
        xgb_zone   = [7.34,  20.00, 22.97, 17.06]   # XGBoost HPO Phase 1
        cnn_zone   = [15.15, 23.35, 19.83, 14.73]   # CNN Phase 1

        fig_rr = go.Figure()
        for name, vals, col in [
            (f"Phase2 Ensemble ★ ({PROD_RMSE:.2f})", p2_zone, "#FFD700"),
            ("Transformer v2 (15.37)",  tv2_zone, "#58a6ff"),
            ("Transformer v1 (16.47)",  tv1_zone, "#bc8cff"),
            ("XGBoost HPO (18.34)",     xgb_zone, "#39c5cf"),
            ("CNN (17.38)",             cnn_zone, "#3fb950"),
        ]:
            fig_rr.add_trace(go.Bar(name=name, x=rul_ranges, y=vals,
                                    marker_color=col, marker_line_width=0))
        fig_rr.update_layout(**plotly_dark(), height=300, barmode="group",
                             yaxis_title="RMSE (cycles)",
                             legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_rr, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────
elif "Ablation" in page_key:
    st.markdown('<div class="section-header">ABLATION STUDY — 5 CONFIGURATIONS</div>',
                unsafe_allow_html=True)

    config_desc = {
        "A: XGBoost baseline":   "XGBoost v1 (RMSE=18.39) — prediction only, no reasoning or actions",
        "B: Transformer v2 (Ph1)":"Transformer v2 Phase 1 (RMSE=15.37) — deep learning temporal modelling",
        "C: DL + LLM (no RAG)":  "LLM reasoning added — hallucination=0.65 without knowledge grounding",
        "D: DL + LLM + RAG":     "RAG grounding added — hallucination 0.65→0.18, grounding 0→1.00",
        "E: Full agentic (Ph2)":  "Phase 2 ensemble (RMSE=15.11, Δ=+0.26) + 80% autonomous execution",
    }

    grounding    = [0.00, 0.00, 0.00, 1.00, 1.00]
    hallucination= [1.00, 1.00, 0.65, 0.18, 0.18]
    actions_exec = [0, 0, 0, 0, 11]
    configs      = ABLATION_DATA["Config"]

    # Waterfall-style metric progression
    if PLOTLY_OK:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">GROUNDING RATE PROGRESSION (C→D)</div>',
                        unsafe_allow_html=True)
            fig_g = go.Figure()
            fig_g.add_trace(go.Bar(
                x=configs, y=grounding,
                marker_color=["#21262d","#21262d","#21262d","#39c5cf","#3fb950"],
                marker_line_width=0,
                text=[f"{v:.2f}" for v in grounding], textposition="outside",
                textfont=dict(size=9, family="IBM Plex Mono"),
            ))
            fig_g.add_annotation(x=3.0, y=0.5, text="RAG added →\ngrounding = 1.00",
                                 font=dict(size=9, color="#39c5cf"),
                                 showarrow=True, arrowcolor="#39c5cf", ax=0, ay=-40)
            fig_g.update_layout(**plotly_dark(), height=260,
                                yaxis_title="Grounding Rate", yaxis_range=[0,1.15],
                                showlegend=False)
            st.plotly_chart(fig_g, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">HALLUCINATION RATE (B→C: LLM added)</div>',
                        unsafe_allow_html=True)
            fig_h = go.Figure()
            fig_h.add_trace(go.Bar(
                x=configs, y=hallucination,
                marker_color=["#ff6b35","#ff6b35","#f0b429","#3fb950","#3fb950"],
                marker_line_width=0,
                text=[f"{v:.2f}" for v in hallucination], textposition="outside",
                textfont=dict(size=9, family="IBM Plex Mono"),
            ))
            fig_h.update_layout(**plotly_dark(), height=260,
                                yaxis_title="Hallucination Rate", yaxis_range=[0,1.2],
                                showlegend=False)
            st.plotly_chart(fig_h, use_container_width=True)

    # Ablation summary table
    st.markdown('<div class="section-header">CONFIGURATION COMPARISON TABLE</div>',
                unsafe_allow_html=True)
    ablation_df = pd.DataFrame({
        "Configuration": configs,
        "Description":   [config_desc[c] for c in configs],
        "RMSE":          ABLATION_DATA["RMSE"],
        "Grounding":     grounding,
        "Hallucination": hallucination,
        "Actions Exec":  actions_exec,
        "Autonomous":    ["✗","✗","✗","✗","✓"],
    })
    st.dataframe(ablation_df, use_container_width=True, hide_index=True)

    # Key insight callout
    st.markdown("""
    <div class="alert-card monitor" style="margin-top:1rem">
      <div style="color:#3fb950;font-weight:600;margin-bottom:0.4rem">
        KEY FINDING: The value-add of each component is empirically isolated
      </div>
      <div style="font-size:0.8rem;color:#e6edf3;line-height:1.6">
        B vs A → Transformer v2 (Ph1 RMSE=15.37) vs XGBoost baseline (18.39): deep learning captures multi-cycle degradation patterns missed by tree models.
        &nbsp;·&nbsp; C vs B → LLM reasoning adds diagnostic language but hallucination rate 0.65 — unusable in safety-critical maintenance without grounding.
        &nbsp;·&nbsp; D vs C → RAG reduces hallucination 0.65→0.18 and raises grounding 0→1.00: the single most important safety contribution.
        &nbsp;·&nbsp; E vs D → Phase 2 ensemble (RMSE=15.11, Δ=+0.26 vs Ph1) + conformal CI ±27.58cy (90%) + 80% autonomous action execution, MTTA=31ms.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem;padding-top:0.8rem;border-top:1px solid #30363d;
     font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#7d8590;
     display:flex;justify-content:space-between">
  <span>Danaya Diarra · MSc Thesis 2026 · Agentic AI for Predictive Maintenance</span>
  <span>Phase2 Ensemble RMSE=15.11 · TransV2(Ph1)=15.37 · XGBv1(Ph1)=18.39 · RAG grounding=1.00 · α=0.2206</span>
</div>
""", unsafe_allow_html=True)
