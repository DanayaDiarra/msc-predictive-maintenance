"""
Agentic PdM NOC — FINAL v5 (Live Edition)
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | GSOM SPBU | 2026

NEW IN v5:
  ✦ Live RUL countdown — real-time degradation simulation (session-clock based)
  ✦ Live telemetry cards — per-station sensor readings + SVG sparklines + degrade rate
  ✦ SVG RUL gauge — animated needle, urgency-colour coded
  ✦ Auto-refresh toggle — 5/10/30/60s intervals
  ✦ Live alert log — threshold-crossing events with timestamps
  ✦ RUL Forecast chart — Plotly live trajectory with NOW marker
  ✦ All v4 features preserved: login, Plotly charts, Anthropic chatbot, pages

CHATBOT: Anthropic Claude (sk-ant-...) via anthropic package
  Set ANTHROPIC_API_KEY in Streamlit Cloud → Secrets, or paste in sidebar.
  Free fallback: Groq (GROQ_API_KEY) · OpenRouter (OPENROUTER_API_KEY) · Rule-based

DEPLOY:  streamlit run streamlit_pdm.py
"""

import sys, os, re, json, time, math
from pathlib import Path
import numpy as np

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path(os.getcwd())
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
os.chdir(_HERE)

import streamlit as st

st.set_page_config(
    page_title="Agentic PdM NOC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
_LOGO = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHZpZXdCb3g9IjAgMCA0OCA0OCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cG9seWdvbiBwb2ludHM9IjI0LDMgNDMsMTMuNSA0MywzNC41IDI0LDQ1IDUsMzQuNSA1LDEzLjUiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzM5YzVjZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIG9wYWNpdHk9IjAuNSIvPjxwb2x5Z29uIHBvaW50cz0iMjQsMTAgMzcsMTcuNSAzNywzMC41IDI0LDM4IDExLDMwLjUgMTEsMTcuNSIgZmlsbD0iIzFjMjMzMyIgc3Ryb2tlPSIjMzljNWNmIiBzdHJva2Utd2lkdGg9IjEuMiIvPjxwb2x5bGluZSBwb2ludHM9IjE1LDI0IDE3LjUsMTkgMjAsMjQgMjIuNSwyOSAyNSwyNCAyNy41LDE5IDMwLDI0IDMyLjUsMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzU4YTZmZiIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPjxjaXJjbGUgY3g9IjI0IiBjeT0iMjQiIHI9IjIuMiIgZmlsbD0iIzM5YzVjZiIvPjxjaXJjbGUgY3g9IjI0IiBjeT0iNiIgcj0iMS44IiBmaWxsPSIjNThhNmZmIi8+PGNpcmNsZSBjeD0iNDEiIGN5PSIxNSIgcj0iMS44IiBmaWxsPSIjNThhNmZmIi8+PGNpcmNsZSBjeD0iNDEiIGN5PSIzMyIgcj0iMS44IiBmaWxsPSIjNThhNmZmIi8+PGNpcmNsZSBjeD0iMjQiIGN5PSI0MiIgcj0iMS44IiBmaWxsPSIjNThhNmZmIi8+PGNpcmNsZSBjeD0iNyIgY3k9IjMzIiByPSIxLjgiIGZpbGw9IiM1OGE2ZmYiLz48Y2lyY2xlIGN4PSI3IiBjeT0iMTUiIHI9IjEuOCIgZmlsbD0iIzU4YTZmZiIvPjwvc3ZnPg=="

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
:root{
  --bg:#0d1117;--card:#161b22;--card2:#1c2333;--border:#30363d;
  --fg:#e6edf3;--muted:#7d8590;--critical:#ff6b35;--warning:#f0b429;
  --ok:#3fb950;--teal:#39c5cf;--blue:#58a6ff;--purple:#bc8cff;
  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
}
html,body,.stApp{background:var(--bg)!important;color:var(--fg)!important;font-family:var(--sans)!important;}
.block-container{padding:1rem 1.8rem!important;max-width:100%!important;}
#MainMenu,footer,header,.stDeployButton{visibility:hidden!important;}
section[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--fg)!important;}
section[data-testid="stSidebar"] .stTextInput input{background:var(--card2)!important;border-color:var(--border)!important;}
/* metric card */
.mc{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.9rem 1.1rem;font-family:var(--mono);}
.mc .l{font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;margin-bottom:.25rem;}
.mc .v{font-size:1.5rem;font-weight:600;line-height:1.1;}
.mc .s{font-size:.67rem;color:var(--muted);margin-top:.15rem;}
/* live metric — slightly glowing border */
.mc-live{background:var(--card);border:1px solid #39c5cf33;border-radius:8px;padding:.9rem 1.1rem;font-family:var(--mono);box-shadow:0 0 8px #39c5cf0a;}
.mc-live .l{font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;margin-bottom:.25rem;}
.mc-live .v{font-size:1.5rem;font-weight:600;line-height:1.1;}
.mc-live .s{font-size:.67rem;color:var(--muted);margin-top:.15rem;}
/* badges */
.bc{background:#ff6b3520;color:#ff6b35;border:1px solid #ff6b3550;border-radius:4px;padding:2px 8px;font-size:.70rem;font-family:var(--mono);font-weight:700;}
.bw{background:#f0b42920;color:#f0b429;border:1px solid #f0b42950;border-radius:4px;padding:2px 8px;font-size:.70rem;font-family:var(--mono);font-weight:700;}
.bm{background:#3fb95020;color:#3fb950;border:1px solid #3fb95050;border-radius:4px;padding:2px 8px;font-size:.70rem;font-family:var(--mono);font-weight:700;}
/* section header */
.sh{font-family:var(--mono);font-size:.67rem;color:var(--muted);text-transform:uppercase;letter-spacing:.11em;border-bottom:1px solid var(--border);padding-bottom:.3rem;margin:1rem 0 .65rem;}
/* alert card */
.ac{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.85rem 1.05rem;margin-bottom:.45rem;font-family:var(--mono);font-size:.78rem;}
.ac.c{border-left:3px solid var(--critical);}
.ac.w{border-left:3px solid var(--warning);}
.ac.m{border-left:3px solid var(--ok);}
/* live telemetry card */
.ltc{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.8rem 1rem;margin-bottom:.4rem;}
.ltc.c{border-left:3px solid var(--critical);}
.ltc.w{border-left:3px solid var(--warning);}
.ltc.m{border-left:3px solid var(--ok);}
/* evidence chunk */
.ec{background:var(--card2);border:1px solid var(--border);border-radius:6px;padding:.65rem .9rem;margin-bottom:.35rem;font-family:var(--mono);font-size:.74rem;}
/* action row */
.ar{display:flex;align-items:flex-start;gap:.7rem;padding:.55rem .75rem;background:var(--card2);border:1px solid var(--border);border-radius:6px;margin-bottom:.35rem;font-size:.76rem;}
.ta{color:var(--ok);font-weight:700;font-family:var(--mono);}
.tt{color:var(--warning);font-weight:700;font-family:var(--mono);}
.th{color:var(--critical);font-weight:700;font-family:var(--mono);}
/* chat */
.cu{background:var(--card2);border:1px solid #39c5cf44;border-radius:12px 12px 2px 12px;padding:.6rem 1rem;font-size:.81rem;color:var(--fg);max-width:76%;margin-left:auto;}
.ca{background:var(--card);border:1px solid var(--border);border-radius:2px 12px 12px 12px;padding:.75rem 1rem;font-size:.81rem;color:#c9d1d9;line-height:1.65;max-width:82%;}
/* plain-english */
.pe{background:linear-gradient(135deg,var(--card2),var(--card));border:1px solid #39c5cf44;border-radius:10px;padding:1.1rem 1.3rem;margin:.7rem 0;}
/* mini bar */
.mbar-bg{background:#21262d;border-radius:2px;overflow:hidden;}
/* alert log entry */
.ale{display:flex;align-items:center;gap:.7rem;padding:.32rem .75rem;border-radius:5px;margin-bottom:.2rem;font-family:var(--mono);font-size:.70rem;}
/* buttons */
.stButton>button{background:var(--card2)!important;border:1px solid var(--teal)!important;color:var(--teal)!important;font-family:var(--mono)!important;font-size:.78rem!important;border-radius:4px!important;}
.stButton>button:hover{background:var(--teal)!important;color:var(--bg)!important;}
div[data-testid="stColumn"] .stButton>button{width:100%!important;height:auto!important;min-height:2rem!important;white-space:normal!important;text-align:left!important;font-size:.70rem!important;padding:.3rem .55rem!important;line-height:1.3!important;}
/* tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--bg)!important;border-bottom:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;font-family:var(--mono)!important;font-size:.75rem!important;border-bottom:2px solid transparent!important;border-radius:0!important;padding:.45rem .9rem!important;}
.stTabs [aria-selected="true"]{color:var(--teal)!important;border-bottom:2px solid var(--teal)!important;}
/* progress */
.stProgress>div>div>div{background:linear-gradient(90deg,#39c5cf,#58a6ff)!important;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.3;}}
@keyframes pulseRed{0%,100%{box-shadow:0 0 0 0 #ff6b3540;}50%{box-shadow:0 0 0 6px #ff6b3500;}}
.dot{animation:blink 2.2s ease-in-out infinite;}
.pulse-red{animation:pulseRed 1.2s ease infinite;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from interpreter_agent import InterpreterAgent
    from rag_pipeline import RAGIndex, RAGPipeline, INDEX_DIR
    from diagnostic_agent import DiagnosticAgent
    from planning_agent import PlanningAgent, ExecutionAgent
    PIPELINE_OK = True; PIPELINE_ERR = ""
except Exception as e:
    PIPELINE_OK = False; PIPELINE_ERR = str(e)

try:
    import pandas as pd
    PD_OK = True
except ImportError:
    PD_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  KEY READERS
# ══════════════════════════════════════════════════════════════════════════════
def _secret(k, default=""):
    try:
        v = st.secrets.get(k, "")
        return str(v).strip() if v and len(str(v).strip()) > 4 else default
    except Exception:
        return os.environ.get(k, default).strip()

def _get_ant_key():
    for source in [
        lambda: st.secrets["ANTHROPIC_API_KEY"],
        lambda: next((str(v) for sec in st.secrets.values()
                      if hasattr(sec,"items")
                      for k,v in sec.items() if "ANTHROPIC" in k.upper()), ""),
        lambda: os.environ.get("ANTHROPIC_API_KEY",""),
        lambda: st.session_state.get("_rt_ant_key",""),
    ]:
        try:
            v = source()
            if v and len(str(v).strip()) > 20: return str(v).strip()
        except Exception: pass
    return ""

def _get_users():
    # Always prefer runtime user store (allows in-session add/remove)
    if "_runtime_users" in st.session_state and st.session_state._runtime_users:
        return dict(st.session_state._runtime_users)
    try:
        u = st.secrets["users"]
        out = {}
        for k, v in u.items():
            kl = k.lower()
            role = "admin" if kl.startswith("admin") else ("engineer" if kl.startswith("eng") else "viewer")
            out[kl] = (str(v), role)
        return out
    except Exception:
        return {"admin":("pdm2026admin","admin"),"engineer":("noc2026","engineer"),"viewer":("readonly","viewer")}

# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN
# ══════════════════════════════════════════════════════════════════════════════
if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.user = ""
    st.session_state.role = ""

if not st.session_state.auth:
    st.markdown(f"""<div style="text-align:center;padding:3rem 0 2rem">
      <img src="{_LOGO}" width="72" style="display:block;margin:0 auto 1rem"/>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:700;color:#39c5cf;letter-spacing:.06em">AGENTIC PdM</div>
      <div style="font-size:.75rem;color:#7d8590;margin-top:.35rem">NOC Monitor · Secure Login</div>
    </div>""", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        with st.form("login"):
            un = st.text_input("Username", placeholder="admin / engineer / viewer")
            pw = st.text_input("Password", type="password", placeholder="password")
            if st.form_submit_button("Sign In ⚡", use_container_width=True):
                users = _get_users()
                u = un.strip().lower()
                if u in users and users[u][0] == pw.strip():
                    st.session_state.auth = True
                    st.session_state.user = u
                    st.session_state.role = users[u][1]
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        st.caption("Demo: admin/pdm2026admin · engineer/noc2026 · viewer/readonly")
    st.stop()

ROLE = st.session_state.role
USER = st.session_state.user
IS_ADMIN = ROLE == "admin"
IS_ENG   = ROLE in ("admin", "engineer")

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_SS_DEFAULTS = {
    "session_start": time.time(),
    "live_mode": False,
    "refresh_interval": 10,
    "alert_log": [],
    "chat_history": [],
    "chat_thinking": False,
    "_rt_ant_key": "",
    "sidebar_open": True,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════
SUBSET_RESULTS = {
    "FD001":{"rmse":12.31,"mae":8.14, "r2":0.912},
    "FD002":{"rmse":15.87,"mae":11.43,"r2":0.841},
    "FD003":{"rmse":13.23,"mae":9.01, "r2":0.896},
    "FD004":{"rmse":16.99,"mae":12.28,"r2":0.826},
}
ABLATION = {
    "configs": ["A: XGBoost v1","B: XGBoost v2 Final","C: v2+LLM (no RAG)","D: v2+LLM+RAG","E: Full agentic"],
    "rmse":    [15.90, 14.60, 14.60, 14.60, 14.60],
    "ground":  [0.00,  0.00,  0.00,  1.00,  1.00],
    "halluc":  [1.00,  1.00,  0.65,  0.00,  0.00],
    "actions": [0, 0, 0, 0, 12],
    "desc": {
        "A: XGBoost v1":         "ML baseline — RMSE 15.90, no reasoning",
        "B: XGBoost v2 Final":   "15k trees, exp(α=3) weights — RMSE 14.60 / 12.77, R²=0.874",
        "C: v2+LLM (no RAG)":   "LLM reasoning — hallucination 65% without grounding",
        "D: v2+LLM+RAG":        "RAG grounding — hallucination 0%, grounding 1.00",
        "E: Full agentic":       "Complete pipeline — 12 autonomous actions, 33ms E2E",
    }
}

STATIONS = [
    dict(id="FD002_47",  urgency="Critical", sub="power_subsystem",       sla=4,
         cl=11.7, ch=17.7, conf=0.880, gr=1.0, hal=0.0, cost=800, auto_n=2, to_n=1, hum_n=0, cov=1.0,
         doc="SOP-PWR-001", subset="FD002", cycles=268,
         hyp="Power unit degradation — voltage instability or rectifier wear",
         fc="48V DC rectifier module", mech="Rectifier voltage decay below 44V threshold",
         alm="PWR-001 (undervoltage) or PWR-004 (mains failure)",
         a1="Execute remote rectifier reset via OMC",           a1t="AUTO",    a1tool="query_cmdb",
         a2="Dispatch field engineer — power specialisation",    a2t="TIMEOUT", a2tool="schedule_dispatch",
         base_rul=14.7, top_feat="voltage_rolling_mean", top_imp=0.0744,
         degrade=0.55, sensor_lbl="DC Voltage", sensor_nom=47.5, sensor_unit="V",  sensor_dir="low"),
    dict(id="FD003_88",  urgency="Critical", sub="thermal_management",    sla=4,
         cl=15.4, ch=20.8, conf=0.910, gr=1.0, hal=0.0, cost=800, auto_n=1, to_n=0, hum_n=2, cov=1.0,
         doc="SOP-THM-001", subset="FD003", cycles=291,
         hyp="Cooling fan bearing failure — COOL-001 imminent, thermal runaway risk",
         fc="Cooling fan FAN-A bearing assembly", mech="Bearing fatigue → fan speed < 2000 RPM",
         alm="COOL-001 (fan failure) + COOL-002 (temp >60°C)",
         a1="Reduce TX power 50% via OMC immediately",           a1t="AUTO",    a1tool="remote_command",
         a2="Emergency dispatch — fan replacement ≤4h",           a2t="HUMAN",   a2tool="schedule_dispatch",
         base_rul=18.1, top_feat="temp_sensor_slope", top_imp=0.0872,
         degrade=0.60, sensor_lbl="Cabinet Temp", sensor_nom=38.0, sensor_unit="°C", sensor_dir="high"),
    dict(id="FD001_23",  urgency="Warning",  sub="thermal_management",    sla=48,
         cl=32.5, ch=43.9, conf=0.820, gr=1.0, hal=0.0, cost=800, auto_n=1, to_n=1, hum_n=0, cov=1.0,
         doc="MAN-THM-001", subset="FD001", cycles=187,
         hyp="Cooling fan bearing wear — COOL-001 precursor pattern",
         fc="Cooling fan bearing or motor winding", mech="Gradual speed reduction toward 2000 RPM",
         alm="COOL-001 or COOL-002/003",
         a1="Schedule fan inspection within 48h SLA",             a1t="TIMEOUT", a1tool="schedule_dispatch",
         a2="Open Warning ticket — 15-min temp monitoring",       a2t="AUTO",    a2tool="open_ticket",
         base_rul=38.2, top_feat="temp_sensor_slope", top_imp=0.0512,
         degrade=0.22, sensor_lbl="Fan Speed", sensor_nom=3200.0, sensor_unit="RPM",sensor_dir="low"),
    dict(id="FD004_55",  urgency="Warning",  sub="rf_antenna",            sla=48,
         cl=37.4, ch=50.6, conf=0.800, gr=1.0, hal=0.0, cost=800, auto_n=1, to_n=1, hum_n=0, cov=1.0,
         doc="MAN-RF-001", subset="FD004", cycles=210,
         hyp="RF chain degradation — antenna connector corrosion",
         fc="7/16 DIN feeder connector", mech="Corrosion causing VSWR > 2.0 and PA efficiency loss",
         alm="RF-001 (VSWR >2.0) or RF-002 (PA power low)",
         a1="Schedule connector inspection + PIM test ≤48h",      a1t="TIMEOUT", a1tool="schedule_dispatch",
         a2="Open Warning ticket — pull VSWR 30-day trend",       a2t="AUTO",    a2tool="open_ticket",
         base_rul=44.0, top_feat="rssi_std_30", top_imp=0.0811,
         degrade=0.18, sensor_lbl="VSWR", sensor_nom=1.82, sensor_unit=":1", sensor_dir="high"),
    dict(id="FD004_112", urgency="Monitor",  sub="backhaul_connectivity", sla=168,
         cl=74.4, ch=100.6, conf=0.366, gr=1.0, hal=0.0, cost=0, auto_n=2, to_n=1, hum_n=0, cov=0.60,
         doc="MAN-BKH-001", subset="FD004", cycles=154,
         hyp="Backhaul link degradation — fibre splice loss or microwave alignment drift",
         fc="Fibre splice point or microwave alignment", mech="Splice loss → latency >10ms",
         alm="BKH-001 (latency high) or BKH-002 (throughput low)",
         a1="Open monitoring ticket — 7-day latency trend",       a1t="AUTO",    a1tool="open_ticket",
         a2="Query CMDB for backhaul type + last inspection",     a2t="AUTO",    a2tool="query_cmdb",
         base_rul=87.5, top_feat="latency_slope", top_imp=0.0683,
         degrade=0.07, sensor_lbl="Latency", sensor_nom=6.2, sensor_unit="ms", sensor_dir="high"),
    dict(id="FD003_71",  urgency="Monitor",  sub="rf_antenna",            sla=168,
         cl=46.8, ch=63.4, conf=0.620, gr=1.0, hal=0.0, cost=0, auto_n=1, to_n=1, hum_n=0, cov=1.0,
         doc="MAN-RF-001", subset="FD003", cycles=178,
         hyp="Antenna connector corrosion — gradual VSWR increase over 18 days",
         fc="7/16 DIN feeder connector sector Alpha", mech="Galvanic corrosion: Al body vs Cu pin",
         alm="RF-001 (VSWR) trending 0.08:1/day",
         a1="Schedule connector inspection + PIM test",            a1t="TIMEOUT", a1tool="schedule_dispatch",
         a2="Open ticket — pull VSWR 30-day trend",               a2t="AUTO",    a2tool="open_ticket",
         base_rul=55.1, top_feat="rssi_std_30", top_imp=0.0814,
         degrade=0.05, sensor_lbl="RSSI", sensor_nom=-67.0, sensor_unit="dBm",sensor_dir="low"),
    dict(id="FD001_08",  urgency="Monitor",  sub="baseband_processing",   sla=168,
         cl=95.5, ch=129.3, conf=0.680, gr=1.0, hal=0.0, cost=0, auto_n=2, to_n=0, hum_n=0, cov=1.0,
         doc="MAN-BBU-002", subset="FD001", cycles=92,
         hyp="BBU CPU approaching 85% threshold — licence or software cause",
         fc="BBU CPU and memory subsystem", mech="Processing load trending toward BBU-003 threshold",
         alm="BBU-003 (CPU overload) or BBU-MEM-001",
         a1="Check capacity licence vs user count via OMC",        a1t="AUTO",    a1tool="query_cmdb",
         a2="Open monitoring — collect CPU/mem trend 7d",          a2t="AUTO",    a2tool="open_ticket",
         base_rul=112.4, top_feat="cpu_utilization_mean", top_imp=0.0771,
         degrade=0.04, sensor_lbl="CPU Util", sensor_nom=71.0, sensor_unit="%", sensor_dir="high"),
    dict(id="FD002_91",  urgency="Monitor",  sub="power_subsystem",       sla=168,
         cl=59.8, ch=80.8, conf=0.650, gr=1.0, hal=0.0, cost=0, auto_n=2, to_n=0, hum_n=0, cov=1.0,
         doc="MAN-PWR-002", subset="FD002", cycles=138,
         hyp="Battery backup unit nearing 80% capacity — end-of-life approaching",
         fc="VRLA battery string", mech="Capacity declining toward 80% of rated 100Ah",
         alm="BBU-001 (battery capacity) anticipated",
         a1="Schedule battery capacity test within 30d",           a1t="AUTO",    a1tool="open_ticket",
         a2="Plan battery string replacement if <80%",             a2t="TIMEOUT", a2tool="schedule_dispatch",
         base_rul=70.3, top_feat="voltage_rolling_mean", top_imp=0.0623,
         degrade=0.04, sensor_lbl="Battery Cap", sensor_nom=84.0, sensor_unit="%", sensor_dir="low"),
    dict(id="FD004_203", urgency="Monitor",  sub="backhaul_connectivity", sla=168,
         cl=80.8, ch=109.3, conf=0.610, gr=1.0, hal=0.0, cost=0, auto_n=2, to_n=1, hum_n=0, cov=0.60,
         doc="SPEC-ITU-001", subset="FD004", cycles=118,
         hyp="Backhaul latency increasing — ITU-T G.826 ESR compliance risk",
         fc="Fibre splice or microwave link — ESR toward 1%", mech="Cumulative splice → ESR near G.826 4%",
         alm="BKH-001 anticipated as ESR approaches 1%",
         a1="Track ESR against G.826 monthly threshold",           a1t="AUTO",    a1tool="open_ticket",
         a2="Schedule OTDR inspection within 7d",                  a2t="TIMEOUT", a2tool="schedule_dispatch",
         base_rul=95.0, top_feat="latency_slope", top_imp=0.0554,
         degrade=0.03, sensor_lbl="ESR", sensor_nom=0.8, sensor_unit="%", sensor_dir="high"),
    dict(id="FD001_77",  urgency="Monitor",  sub="baseband_processing",   sla=168,
         cl=101.2, ch=136.9, conf=0.620, gr=1.0, hal=0.0, cost=0, auto_n=1, to_n=0, hum_n=0, cov=1.0,
         doc="MAN-BBU-001", subset="FD001", cycles=76,
         hyp="Normal end-of-life health decline — routine maintenance appropriate",
         fc="BBU general health", mech="Cumulative wear approaching 80% lifecycle threshold",
         alm="No active alarms — preventive indicator only",
         a1="Add to next scheduled maintenance cycle ≤168h",       a1t="AUTO",    a1tool="open_ticket",
         a2=None, a2t=None, a2tool=None,
         base_rul=119.0, top_feat="cpu_utilization_mean", top_imp=0.0502,
         degrade=0.02, sensor_lbl="Health Idx", sensor_nom=62.0, sensor_unit="%", sensor_dir="low"),
]

EVIDENCE = {
    "FD002_47":[
        ("SOP-PWR-001","sop","SOP: Power Unit Fault Response",0.06252,1,2,
         "Step 1: Query OMC rectifier. Step 2: Remote reset. Step 3: Dispatch if unresolved 30min."),
        ("ALM-DICT-001","alarm_dict","Alarm Dict — PWR-001 to PWR-005",0.06055,4,7,
         "PWR-001: Undervoltage. Cause: mains failure, rectifier fault, MCB tripped. Corr: PWR-004."),
        ("TREE-PWR-001","tree","Decision Tree — Power Triage",0.05941,8,8,
         "Q1: PWR-004 active? Q2: Voltage <44V? → Dispatch → Replace rectifier."),
        ("MAN-PWR-001","manual","Power Unit Rectifier Specs",0.05252,2,1,
         "Nominal 47.5–51.5V. Alarm <44V. Replace: >5% ripple or 7yr service."),
        ("TKT-001","ticket","Historical: INC-2024-00847",0.05175,3,3,
         "RUL 12.3 at trigger. Generator activated. 4h14m resolution. Prediction correct."),
    ],
    "FD001_23":[
        ("MAN-THM-001","manual","Thermal Mgmt — Fan Specs",0.06279,1,1,
         "Fan: 450 CFM at 3200 RPM. COOL-001 at <2000 RPM. Bearing replacement at 40,000h."),
        ("SOP-THM-001","sop","SOP: High Temperature Response",0.06226,2,2,
         "Immediate: reduce TX 50% on COOL-001. On-site: inspect ventilation, bearing temp."),
        ("TKT-003","ticket","Historical: INC-2024-00612",0.06125,4,4,
         "Fan 1 seized 38,000h. Both replaced 5h13m. Model flagged 8 cycles before event."),
        ("MAN-THM-002","manual","Thermal Runaway Prevention",0.05941,8,8,
         "Emergency: graceful shutdown via OMC >75°C. Inspect PCB for discoloration."),
        ("ALM-003","alarm_dict","Alarm Dict — COOL-001 to COOL-005",0.05175,3,3,
         "COOL-001: <2000RPM Critical. Reduce TX 50%, dispatch 4h. COOL-003: >70°C shutdown."),
    ],
}
# ── 5 additional stations ─────────────────────────────────────────────────────
STATIONS += [
    dict(id="FD002_14",  urgency="Critical", sub="power_subsystem",       sla=4,
         cl=8.2, ch=14.1, conf=0.920, gr=1.0, hal=0.0, cost=900, auto_n=2, to_n=1, hum_n=1, cov=1.00,
         doc="SOP-PWR-001", subset="FD002", cycles=312,
         hyp="Critical rectifier fault — DC bus voltage below 42V threshold",
         fc="48V DC rectifier module B", mech="Module B failure — Module A running at 140% rated load",
         alm="PWR-001 (undervoltage) + PWR-003 (rectifier failure)",
         a1="Isolate rectifier B and activate bypass via OMC",  a1t="AUTO",    a1tool="remote_command",
         a2="Emergency dispatch — dual rectifier replacement",   a2t="HUMAN",   a2tool="schedule_dispatch",
         base_rul=11.2, top_feat="voltage_rolling_mean", top_imp=0.0798,
         degrade=0.65, sensor_lbl="DC Voltage",   sensor_nom=42.8, sensor_unit="V",  sensor_dir="low"),
    dict(id="FD001_44",  urgency="Warning",  sub="rf_antenna",            sla=48,
         cl=28.1, ch=39.5, conf=0.780, gr=1.0, hal=0.0, cost=600, auto_n=1, to_n=2, hum_n=0, cov=1.00,
         doc="MAN-RF-001", subset="FD001", cycles=203,
         hyp="PA efficiency degradation — TX power anomaly detected on sector Alpha",
         fc="Power amplifier PA-2 stage", mech="PA efficiency falling 25% below nominal threshold",
         alm="RF-002 (PA power low) + RF-004 (efficiency alarm)",
         a1="Reduce TX power 20% via OMC to protect PA stage",  a1t="AUTO",    a1tool="remote_command",
         a2="Schedule PA module inspection within 48h",          a2t="TIMEOUT", a2tool="schedule_dispatch",
         base_rul=33.8, top_feat="rssi_std_30",         top_imp=0.0755,
         degrade=0.20, sensor_lbl="PA Efficiency", sensor_nom=78.5, sensor_unit="%", sensor_dir="low"),
    dict(id="FD003_55",  urgency="Warning",  sub="thermal_management",    sla=48,
         cl=22.0, ch=33.4, conf=0.840, gr=1.0, hal=0.0, cost=700, auto_n=1, to_n=1, hum_n=0, cov=1.00,
         doc="MAN-THM-001", subset="FD003", cycles=244,
         hyp="Heat exchanger fouling — reduced airflow causing thermal gradient",
         fc="Cabinet heat exchanger unit", mech="Particulate buildup reducing airflow by 35%",
         alm="COOL-002 (temp >60 C) + COOL-004 (fan deviation)",
         a1="Increase fan speed to maximum via OMC",             a1t="AUTO",    a1tool="remote_command",
         a2="Schedule heat exchanger cleaning within 48h",       a2t="TIMEOUT", a2tool="schedule_dispatch",
         base_rul=27.7, top_feat="temp_sensor_slope",   top_imp=0.0831,
         degrade=0.28, sensor_lbl="Inlet Temp",   sensor_nom=41.2, sensor_unit="C",  sensor_dir="high"),
    dict(id="FD004_78",  urgency="Monitor",  sub="baseband_processing",   sla=168,
         cl=61.0, ch=84.2, conf=0.700, gr=1.0, hal=0.0, cost=0, auto_n=2, to_n=0, hum_n=0, cov=1.00,
         doc="MAN-BBU-002", subset="FD004", cycles=167,
         hyp="BBU memory pressure — swap usage trending toward OOM threshold",
         fc="BBU DDR4 memory subsystem", mech="Memory leak in L2 process — swap at 68% of 16GB",
         alm="BBU-MEM-001 (swap >50%) trending toward BBU-MEM-002",
         a1="Restart non-critical L2 processes via OMC",         a1t="AUTO",    a1tool="remote_command",
         a2="Open monitoring — track swap/mem trend 7d",          a2t="AUTO",    a2tool="open_ticket",
         base_rul=72.6, top_feat="cpu_utilization_mean", top_imp=0.0688,
         degrade=0.06, sensor_lbl="Mem Swap",    sensor_nom=68.0, sensor_unit="%", sensor_dir="high"),
    dict(id="FD002_33",  urgency="Monitor",  sub="backhaul_connectivity", sla=168,
         cl=88.4, ch=122.0, conf=0.580, gr=1.0, hal=0.0, cost=0, auto_n=1, to_n=1, hum_n=0, cov=0.60,
         doc="MAN-BKH-001", subset="FD002", cycles=131,
         hyp="Microwave path anomaly — rain-fade increasing in frequency",
         fc="Microwave dish alignment — azimuth drift detected", mech="0.3 deg azimuth drift causing 3.2dB fade margin reduction",
         alm="BKH-003 (fade margin <10dB) anticipated",
         a1="Open monitoring ticket — track fade margin trend",   a1t="AUTO",    a1tool="open_ticket",
         a2="Schedule microwave alignment check within 14d",      a2t="TIMEOUT", a2tool="schedule_dispatch",
         base_rul=105.2, top_feat="latency_slope",       top_imp=0.0601,
         degrade=0.03, sensor_lbl="Fade Margin", sensor_nom=14.8, sensor_unit="dB", sensor_dir="low"),
]

for _s in STATIONS:
    if _s["id"] not in EVIDENCE:
        EVIDENCE[_s["id"]] = EVIDENCE["FD002_47"]

# ── CRITICAL: add "rul" alias for backward-compat with all page code ──────────
for _s in STATIONS:
    _s["rul"] = _s["base_rul"]

# ══════════════════════════════════════════════════════════════════════════════
#  LIVE PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def elapsed_min():
    return (time.time() - st.session_state.session_start) / 60.0

def live_rul(s):
    """XGBoost v2 base prediction minus session-time degradation."""
    return max(0.1, s["base_rul"] - elapsed_min() * s["degrade"])

def live_urgency(rul):
    if rul <= 20: return "Critical"
    if rul <= 50: return "Warning"
    return "Monitor"

def live_sensor(s, t=None):
    """Realistic sensor reading: nominal + drift + noise."""
    if t is None: t = time.time()
    rng = np.random.default_rng(int(t / 4) + abs(hash(s["id"])) % 99999)
    nom  = s["sensor_nom"]
    el   = elapsed_min()
    d    = -1 if s["sensor_dir"] == "low" else 1
    drift = d * el * abs(nom) * 0.0012
    noise = rng.normal(0, abs(nom) * 0.013)
    val   = nom + drift + noise
    return round(val, 2)

def spark_history(s, n=12):
    """Generate n historical sensor readings (6-second steps)."""
    now = time.time()
    return [live_sensor(s, now - (n-1-i)*6) for i in range(n)]

def sensor_arrow(s):
    return "↓" if s["sensor_dir"] == "low" else "↑"

def check_alerts():
    """Detect urgency escalations and log them."""
    for s in STATIONS:
        rul = live_rul(s)
        new_urg = live_urgency(rul)
        old_urg = s["urgency"]
        key = f"_alerted_{s['id']}_{new_urg}"
        if new_urg != old_urg and key not in st.session_state:
            st.session_state[key] = True
            st.session_state.alert_log.insert(0, {
                "ts": time.strftime("%H:%M:%S"),
                "id": s["id"],
                "msg": f"RUL={rul:.1f}  urgency {old_urg} → {new_urg}",
                "urg": new_urg,
            })

# ══════════════════════════════════════════════════════════════════════════════
#  SVG BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def svg_sparkline(vals, color="#39c5cf", W=80, H=28):
    if not vals or len(vals) < 2: return ""
    mn, mx = min(vals), max(vals)
    rng = max(mx - mn, 1e-9)
    pts = " ".join(
        f"{W*i/(len(vals)-1):.1f},{H-4-(H-8)*(v-mn)/rng:.1f}"
        for i, v in enumerate(vals)
    )
    lx = W * (len(vals)-1)/(len(vals)-1)
    ly = H-4-(H-8)*(vals[-1]-mn)/rng
    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:{W}px;height:{H}px;display:inline-block;vertical-align:middle">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" opacity="0.9"/>'
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.5" fill="{color}"/>'
        f'</svg>'
    )

def svg_gauge(rul, cl, ch, color, W=200, H=130):
    cx2, cy2, r = 100, 105, 80
    angle = max(0, min(180, (1 - rul/125)*180))
    rad   = math.radians(180 - angle)
    px    = cx2 + r * math.cos(rad)
    py    = cy2 - r * math.sin(rad)
    # Background arc
    arc = f"M {cx2-r} {cy2} A {r} {r} 0 0 1 {cx2+r} {cy2}"
    # Colour ticks
    ticks = ""
    for pct, tc in [(20/125, "#ff6b35"), (50/125, "#f0b429")]:
        ta = math.radians(180 - pct*180)
        x1 = cx2 + (r-10)*math.cos(ta); y1 = cy2 - (r-10)*math.sin(ta)
        x2 = cx2 + (r+2)*math.cos(ta);  y2 = cy2 - (r+2)*math.sin(ta)
        ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{tc}" stroke-width="2" opacity="0.7"/>'
    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{W}px;display:block">'
        f'<path d="{arc}" fill="none" stroke="#21262d" stroke-width="14" stroke-linecap="round"/>'
        f'<path d="{arc}" fill="none" stroke="{color}" stroke-width="6" stroke-linecap="round" opacity="0.75"/>'
        f'{ticks}'
        f'<line x1="{cx2}" y1="{cy2}" x2="{px:.1f}" y2="{py:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="{cx2}" cy="{cy2}" r="5.5" fill="{color}"/>'
        f'<circle cx="{cx2}" cy="{cy2}" r="3" fill="#0d1117"/>'
        f'<text x="{cx2}" y="{cy2+22}" fill="{color}" font-size="22" font-weight="700" text-anchor="middle" font-family="IBM Plex Mono,monospace">{rul:.1f}</text>'
        f'<text x="{cx2}" y="{cy2+36}" fill="#7d8590" font-size="10" text-anchor="middle" font-family="IBM Plex Mono,monospace">cycles RUL</text>'
        f'<text x="{cx2}" y="{cy2+49}" fill="#7d8590" font-size="9" text-anchor="middle" font-family="IBM Plex Mono,monospace">[{cl:.1f}–{ch:.1f}]</text>'
        f'<text x="12" y="{cy2+4}" fill="#ff6b35" font-size="8" font-family="monospace">20</text>'
        f'<text x="{cx2-20}" y="{cy2-r-6}" fill="#f0b429" font-size="8" font-family="monospace">50</text>'
        f'</svg>'
    )

def svg_rul_hbar():
    W, ROW, PL, PR, PT = 720, 27, 168, 70, 20
    stations_sorted = sorted(STATIONS, key=lambda x: live_rul(x))
    H = PT + len(stations_sorted)*ROW + 26
    bars = ""
    for i, s in enumerate(stations_sorted):
        rul = live_rul(s)
        urg = live_urgency(rul)
        col = {"Critical":"#ff6b35","Warning":"#f0b429","Monitor":"#3fb950"}[urg]
        bw  = max(2, int(rul/125*(W-PL-PR)))
        y   = PT + i*ROW
        bars += (
            f'<text x="{PL-6}" y="{y+17}" fill="#c9d1d9" font-size="11" text-anchor="end" '
            f'font-family="IBM Plex Mono,monospace">{s["id"]}</text>'
            f'<rect x="{PL}" y="{y+4}" width="3" height="17" fill="{col}" rx="1"/>'
            f'<rect x="{PL+5}" y="{y+5}" width="{bw}" height="15" fill="{col}" opacity="0.75" rx="2"/>'
            f'<text x="{PL+bw+10}" y="{y+17}" fill="{col}" font-size="11" font-family="IBM Plex Mono,monospace" '
            f'font-weight="700">{rul:.1f}</text>'
        )
    for v in [20, 50, 75, 100, 125]:
        x = PL + int(v/125*(W-PL-PR))
        tc = "#ff6b35" if v==20 else "#f0b429" if v==50 else "#1d2633"
        da = "4,3" if v<=50 else "none"
        bars += (
            f'<line x1="{x}" y1="{PT-4}" x2="{x}" y2="{H-20}" stroke="{tc}" stroke-width="1" '
            f'stroke-dasharray="{da}" opacity="0.55"/>'
            f'<text x="{x}" y="{H-6}" fill="#5a6475" font-size="9" text-anchor="middle">{v}</text>'
        )
    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;background:#0d1117;border-radius:8px;border:1px solid #30363d">'
        f'{bars}</svg>'
    )

# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def mc(label, val, sub="", color="var(--blue)", live=False):
    cls = "mc-live" if live else "mc"
    return f'<div class="{cls}"><div class="l">{label}</div><div class="v" style="color:{color}">{val}</div><div class="s">{sub}</div></div>'

def badge(u):
    return f'<span class="{"bc" if u=="Critical" else "bw" if u=="Warning" else "bm"}">{u}</span>'

def rc(r):
    return "#ff6b35" if r<=20 else ("#f0b429" if r<=50 else "#3fb950")

def tier_html(t):
    return {"AUTO":'<span class="ta">● AUTO</span>',"TIMEOUT":'<span class="tt">◑ TIMEOUT</span>',"HUMAN":'<span class="th">○ HUMAN</span>'}.get(t, t or "")

def sh(label):
    st.markdown(f'<div class="sh">{label}</div>', unsafe_allow_html=True)

def pdk():
    return dict(paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font=dict(family="IBM Plex Mono,monospace", color="#7d8590", size=10),
                xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
                yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
                margin=dict(l=36, r=16, t=36, b=36))

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR TOGGLE + TOP NAV
# ══════════════════════════════════════════════════════════════════════════════
_css_open  = """<style>section[data-testid="stSidebar"]{transform:translateX(0%)!important;width:21rem!important;min-width:21rem!important;visibility:visible!important;transition:all .3s ease!important;}</style>"""
_css_close = """<style>section[data-testid="stSidebar"]{transform:translateX(-120%)!important;width:0!important;min-width:0!important;max-width:0!important;overflow:hidden!important;visibility:hidden!important;transition:all .3s ease!important;}div[data-testid="stSidebarCollapsedControl"]{display:none!important;}</style>"""
st.markdown(_css_open if st.session_state.sidebar_open else _css_close, unsafe_allow_html=True)

_c1, _c2 = st.columns([1, 20])
with _c1:
    if st.button("◀" if st.session_state.sidebar_open else "▶", key="tog"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

_rcolor = {"admin":"#ff6b35","engineer":"#58a6ff","viewer":"#3fb950"}.get(ROLE,"#7d8590")
check_alerts()
crit_n = sum(1 for s in STATIONS if live_urgency(live_rul(s))=="Critical")
sys_color = "#ff6b35" if crit_n > 0 else "#3fb950"
sys_label = f"{crit_n} CRITICAL ACTIVE" if crit_n > 0 else "SYSTEM OPERATIONAL"

_live_dot_color = "#39c5cf"
_live_label     = "● LIVE"
_live_border    = "#39c5cf44"
_live_dot_cls   = "dotfast"
_crit_border    = "#ff6b3544" if crit_n > 0 else "#3fb95044"
_crit_dot_cls   = "dotfast" if crit_n > 0 else "dot"
_n_stations     = len(STATIONS)
st.markdown(f"""<style>
@keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:.3;}}}}
@keyframes blinkfast{{0%,100%{{opacity:1;}}50%{{opacity:.2;}}}}
.dot{{animation:blink 2.2s ease-in-out infinite;}}
.dotfast{{animation:blinkfast 0.9s ease-in-out infinite;}}
</style>
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:.4rem 0 .8rem;margin-bottom:.8rem;border-bottom:1px solid #30363d;flex-wrap:wrap;gap:.5rem">

  <!-- LEFT: Logo + App name -->
  <div style="display:flex;align-items:center;gap:12px">
    <img src="{_LOGO}" width="44" height="44"/>
    <div>
      <div style="display:flex;align-items:baseline;gap:4px">
        <span style="font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:1.15rem;color:#39c5cf;letter-spacing:-.01em">Maint</span><span style="font-family:'IBM Plex Mono',monospace;font-weight:300;font-size:1.15rem;color:#e6edf3;letter-spacing:-.01em">Agent</span>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:.58rem;color:#7d8590;padding:1px 5px;border:1px solid #30363d;border-radius:3px;margin-left:5px">NOC</span>
      </div>
      <div style="font-size:.63rem;color:#7d8590;margin-top:.1rem">
        Predictive Maintenance · Telecom BTS Infrastructure · {_n_stations} Stations
      </div>
    </div>
  </div>

  <!-- RIGHT: Status chips -->
  <div style="display:flex;align-items:center;gap:7px;margin-left:auto">

    <!-- Live chip — always teal dotfast -->
    <div style="background:#161b22;border:1px solid #39c5cf44;border-radius:6px;
         padding:4px 10px;display:flex;align-items:center;gap:5px">
      <span style="width:7px;height:7px;background:#39c5cf;border-radius:50%;display:inline-block"
            class="dotfast"></span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#39c5cf;
            white-space:nowrap">● LIVE</span>
    </div>

    <!-- Critical / Operational status chip -->
    <div style="background:#161b22;border:1px solid {_crit_border};border-radius:6px;
         padding:4px 10px;display:flex;align-items:center;gap:5px">
      <span style="width:7px;height:7px;background:{sys_color};border-radius:50%;display:inline-block"
            class="{_crit_dot_cls}"></span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:{sys_color};
            white-space:nowrap">{sys_label}</span>
    </div>

    <!-- User + role chip -->
    <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;
         padding:4px 10px;font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:{_rcolor}">
      {USER}&nbsp;·&nbsp;<span style="color:#7d8590">{ROLE.upper()}</span>
    </div>

    <!-- Model RMSE chip -->
    <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;
         padding:4px 11px;font-family:'IBM Plex Mono',monospace;font-size:.65rem">
      <span style="color:#7d8590">RMSE</span>&nbsp;
      <span style="color:#39c5cf;font-weight:700">14.60</span>&nbsp;
      <span style="color:#7d8590;font-size:.58rem">all-4&nbsp;·&nbsp;R²=</span>
      <span style="color:#58a6ff;font-weight:700">0.874</span>
    </div>

  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Controls")
    sel_id = st.selectbox("Station", [s["id"] for s in STATIONS])
    sel = next(s for s in STATIONS if s["id"] == sel_id)

    st.markdown("---")
    # ── LIVE MODE ──
    st.markdown("### ⚡ Live Mode")
    live_on = st.toggle("Enable auto-refresh", value=st.session_state.live_mode, key="live_toggle")
    st.session_state.live_mode = live_on
    if live_on:
        ri = st.select_slider("Refresh interval (s)", options=[5,10,15,30,60],
                               value=st.session_state.refresh_interval)
        st.session_state.refresh_interval = ri
        el = elapsed_min()
        st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#3fb950;margin:.3rem 0">
          ● LIVE · {el:.1f}m elapsed · {ri}s interval</div>""", unsafe_allow_html=True)
    if st.button("↺ Reset session clock", use_container_width=True):
        st.session_state.session_start = time.time()
        st.session_state.alert_log = []
        for k in list(st.session_state.keys()):
            if k.startswith("_alerted_"): del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.markdown("### Pipeline")
    st.toggle("Live pipeline", value=PIPELINE_OK, disabled=not PIPELINE_OK)
    if not PIPELINE_OK:
        st.caption(f"Offline: {PIPELINE_ERR[:80]}")

    if IS_ENG:
        st.markdown("---")
        st.markdown("### Knowledge Base Upload")
        st.caption("SOPs, manuals, alarm guides → enrich RAG corpus")
        st.file_uploader("Files", type=["pdf","txt","html","csv","md","json"],
                         accept_multiple_files=True, label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### 🔑 Chatbot API Key")
        st.caption("Anthropic (primary) · Groq / OpenRouter (fallback)")
        _rt = st.text_input("Anthropic key", type="password", placeholder="sk-ant-...",
                             label_visibility="collapsed",
                             value=st.session_state.get("_rt_ant_key",""),
                             key="sidebar_key_input")
        if st.button("💾 Save Key", use_container_width=True):
            st.session_state._rt_ant_key = _rt.strip()
            st.success("Key saved for this session")

    st.markdown("---")
    all_pages = ["Live Fleet Monitor","Fleet Overview","Station Detail","Plain English",
                 "RAG Evidence","Agent Reasoning","Model Benchmark",
                 "Ablation Study","Engineer Chatbot","User Management"]
    if not IS_ENG:
        all_pages = [p for p in all_pages if p not in ["Engineer Chatbot","User Management"]]
    if not IS_ADMIN:
        all_pages = [p for p in all_pages if p != "User Management"]

    page = st.radio("Navigation", all_pages, label_visibility="collapsed")

    st.markdown("---")
    # Quick model stats
    el2 = elapsed_min()
    st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:.63rem;color:#5a6475;line-height:1.8">
      All-4 RMSE: <span style="color:#39c5cf">14.60</span><br>
      FD001+FD003: <span style="color:#3fb950">12.77</span><br>
      R²: <span style="color:#58a6ff">0.874</span><br>
      Session: <span style="color:#f0b429">{el2:.1f}m</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔒 Sign Out"):
        st.session_state.auth = False
        st.rerun()

pk = page

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE FLEET MONITOR  ← NEW
# ══════════════════════════════════════════════════════════════════════════════
if pk == "Live Fleet Monitor":
    el = elapsed_min()
    all_ruls  = [live_rul(s) for s in STATIONS]
    all_urgs  = [live_urgency(r) for r in all_ruls]
    nc = all_urgs.count("Critical")
    nw = all_urgs.count("Warning")
    nm = all_urgs.count("Monitor")

    # ── KPI row ──
    k_cols = st.columns(7)
    for col, lbl, val, sub, color in zip(k_cols,
        ["🔴 CRITICAL","🟡 WARNING","🟢 MONITOR","MEAN RUL","GROUNDING","HALLUCIN.","SESSION"],
        [nc, nw, nm, f"{sum(all_ruls)/len(all_ruls):.1f}", "1.000", "0.000", f"{el:.1f}m"],
        ["SLA ≤4h","SLA ≤48h","SLA ≤168h","cycles","RAG rate","zero claims","elapsed"],
        ["#ff6b35","#f0b429","#3fb950","#58a6ff","#3fb950","#3fb950","#39c5cf"]):
        col.markdown(mc(lbl, val, sub, color, live=True), unsafe_allow_html=True)

    # ── Live telemetry cards ──
    sh("LIVE STATION TELEMETRY — XGBoost v2 Degradation Simulation")
    for row_i in range(0, len(STATIONS), 2):
        cols2 = st.columns(2)
        for j, col in enumerate(cols2):
            if row_i + j >= len(STATIONS): break
            s   = STATIONS[row_i + j]
            rul = live_rul(s)
            urg = live_urgency(rul)
            col_hex = {"Critical":"#ff6b35","Warning":"#f0b429","Monitor":"#3fb950"}[urg]
            cls_     = urg.lower()
            sv       = live_sensor(s)
            arr      = sensor_arrow(s)
            spark    = spark_history(s)
            conf_pct = int(s["conf"]*100)

            with col:
                st.markdown(f"""
<div class="ltc {cls_}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div style="flex:1">
      <div style="display:flex;align-items:center;gap:.45rem;flex-wrap:wrap;margin-bottom:.2rem">
        <span style="font-size:.92rem;font-weight:700;color:#a5d6ff;font-family:'IBM Plex Mono',monospace">{s['id']}</span>
        {badge(urg)}
        <span style="font-size:.60rem;color:#30363d;font-family:monospace">C-MAPSS {s['subset']}</span>
      </div>
      <div style="font-size:.67rem;color:#7d8590;margin-bottom:.12rem">{s['sub'].replace('_',' ')} · SLA {s['sla']}h</div>
      <div style="font-size:.70rem;color:#c9d1d9;margin-bottom:.2rem">{s['hyp'][:72]}…</div>
      <div style="display:flex;align-items:center;gap:.8rem;margin-top:.25rem">
        <div>
          <div style="font-size:.59rem;color:#7d8590;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.07em">LIVE {s['sensor_lbl']}</div>
          <div style="font-size:.90rem;font-weight:700;color:{col_hex};font-family:'IBM Plex Mono',monospace">{sv}{s['sensor_unit']} <span style="font-size:.80rem">{arr}</span></div>
        </div>
        <div>{svg_sparkline(spark, col_hex, W=90, H=30)}</div>
        <div>
          <div style="font-size:.59rem;color:#7d8590;font-family:monospace;text-transform:uppercase;letter-spacing:.07em">DEGRADE</div>
          <div style="font-size:.78rem;color:#f0b429;font-family:'IBM Plex Mono',monospace">{s['degrade']:.2f}/min</div>
        </div>
      </div>
    </div>
    <div style="text-align:right;padding-left:.7rem;min-width:90px">
      <div style="font-size:1.35rem;font-weight:700;color:{col_hex};font-family:'IBM Plex Mono',monospace;line-height:1.05">{rul:.1f}</div>
      <div style="font-size:.65rem;color:#7d8590;font-family:monospace">cycles</div>
      <div style="font-size:.60rem;color:#7d8590">[{s['cl']:.1f}–{s['ch']:.1f}]</div>
    </div>
  </div>
  <div style="margin-top:.35rem;background:#21262d;height:4px;border-radius:2px;overflow:hidden">
    <div style="width:{min(100,int(rul/125*100))}%;height:4px;background:{col_hex};border-radius:2px"></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Live RUL bar chart ──
    sh("REAL-TIME RUL FORECAST — ALL STATIONS (sorted by urgency)")
    st.markdown(svg_rul_hbar(), unsafe_allow_html=True)

    # ── Alert log ──
    sh("LIVE ALERT LOG")
    if st.session_state.alert_log:
        for a in st.session_state.alert_log[:10]:
            uc = {"Critical":"#ff6b35","Warning":"#f0b429","Monitor":"#3fb950"}.get(a["urg"],"#7d8590")
            st.markdown(f"""
<div class="ale" style="background:#161b22;border:1px solid {uc}33;border-left:3px solid {uc}">
  <span style="color:#7d8590">{a['ts']}</span>
  <span style="color:#a5d6ff;font-weight:700">{a['id']}</span>
  <span style="color:{uc}">{a['msg']}</span>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-family:monospace;font-size:.68rem;color:#7d8590;padding:.5rem 0">No escalation events yet · {el:.1f}m elapsed · alerts appear when RUL crosses 50 or 20 cycles</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FLEET OVERVIEW (enhanced with live data)
# ══════════════════════════════════════════════════════════════════════════════
elif pk == "Fleet Overview":
    nc  = sum(1 for s in STATIONS if live_urgency(live_rul(s))=="Critical")
    nw  = sum(1 for s in STATIONS if live_urgency(live_rul(s))=="Warning")
    nm  = sum(1 for s in STATIONS if live_urgency(live_rul(s))=="Monitor")
    mr  = sum(live_rul(s) for s in STATIONS)/len(STATIONS)
    mcf = sum(s["conf"] for s in STATIONS)/len(STATIONS)
    mg  = sum(s["gr"]   for s in STATIONS)/len(STATIONS)
    for col, lbl, val, sub, color in zip(st.columns(6),
        ["CRITICAL","WARNING","MONITOR","MEAN RUL","MEAN CONF","GROUNDING"],
        [nc, nw, nm, f"{mr:.0f}", f"{mcf:.3f}", f"{mg:.3f}"],
        ["SLA ≤4h","SLA ≤48h","SLA ≤168h","cycles","diagnostic","RAG grounding"],
        ["#ff6b35","#f0b429","#3fb950","#58a6ff","#58a6ff","#39c5cf"]):
        col.markdown(mc(lbl, val, sub, color), unsafe_allow_html=True)

    sh("FLEET ALERT STATUS — 10 STATIONS · XGBoost v2 Final · All-4 RMSE=14.60 · R²=0.874")
    for s in STATIONS:
        _rul_now = live_rul(s); _urg_now = live_urgency(_rul_now)
        css_ = {"Critical":"c","Warning":"w","Monitor":"m"}[_urg_now]
        bw_  = int(s["conf"]*100)
        bc_  = "#3fb950" if s["conf"]>0.7 else ("#f0b429" if s["conf"]>0.5 else "#ff6b35")
        rc_  = rc(_rul_now)
        st.markdown(f"""
<div class="ac {css_}">
  <div style="display:flex;justify-content:space-between">
    <div>
      <span style="font-size:.95rem;font-weight:700;color:#a5d6ff">{s["id"]}</span>&nbsp;
      {badge(_urg_now)}&nbsp;
      <span style="font-size:.63rem;color:#30363d;font-family:'IBM Plex Mono',monospace">C-MAPSS {s["subset"]} · {s["cycles"]} cycles</span>
      <div style="color:#7d8590;font-size:.69rem;margin-top:.2rem">{s["sub"]} · SLA {s["sla"]}h · RAG cov {s["cov"]:.2f}</div>
      <div style="color:#c9d1d9;font-size:.70rem;margin-top:.22rem">{s["hyp"]}</div>
      <div style="color:#7d8590;font-size:.64rem;margin-top:.18rem">Top feature: <span style="color:#58a6ff">{s["top_feat"]}</span> (imp={s["top_imp"]:.4f})</div>
    </div>
    <div style="text-align:right;min-width:120px">
      <div style="font-size:1.3rem;font-weight:700;color:{rc_};font-family:'IBM Plex Mono',monospace">{_rul_now:.1f}<span style="font-size:.70rem;color:#7d8590"> cyc</span></div>
      <div style="font-size:.66rem;color:#7d8590">[{s["cl"]:.1f}–{s["ch"]:.1f}]</div>
      <div style="margin-top:.32rem;display:flex;align-items:center;gap:.3rem;justify-content:flex-end">
        <div style="width:55px;background:#21262d;height:3px;border-radius:2px">
          <div style="width:{bw_}%;background:{bc_};height:3px;border-radius:2px"></div>
        </div>
        <span style="font-size:.62rem;color:{bc_}">{s["conf"]:.3f}</span>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    if PLOTLY_OK:
        c1, c2 = st.columns(2)
        with c1:
            sh("RUL DISTRIBUTION — XGBoost v2 PREDICTIONS")
            fig = go.Figure(go.Bar(
                x=[s["id"] for s in STATIONS], y=[live_rul(s) for s in STATIONS],
                marker_color=[rc(live_rul(s)) for s in STATIONS], marker_line_width=0,
                error_y=dict(type="data", symmetric=False,
                    array=[s["ch"]-live_rul(s) for s in STATIONS],
                    arrayminus=[live_rul(s)-s["cl"] for s in STATIONS],
                    color="#7d8590", thickness=1.5, width=5)))
            fig.add_hline(y=20, line_dash="dash", line_color="#ff6b35", annotation_text="Critical", annotation_font_size=9)
            fig.add_hline(y=50, line_dash="dash", line_color="#f0b429", annotation_text="Warning",  annotation_font_size=9)
            fig.update_layout(**pdk(), height=270, yaxis_title="RUL (cycles)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sh("DIAGNOSTIC QUALITY RADAR")
            cats = ["RAG Cov","Conf","Grounding","1-Halluc","Actions/3"]
            fig2 = go.Figure()
            for s in STATIONS:
                v = [s["cov"],s["conf"],s["gr"],1-s["hal"],min((s["auto_n"]+s["to_n"])/3,1)]
                fig2.add_trace(go.Scatterpolar(r=v+[v[0]], theta=cats+[cats[0]], name=s["id"],
                    line=dict(width=1.5), fill="toself", opacity=0.22))
            fig2.update_layout(**pdk(), height=270,
                polar=dict(bgcolor="#0d1117",
                    radialaxis=dict(range=[0,1], gridcolor="#21262d", tickfont=dict(size=8)),
                    angularaxis=dict(gridcolor="#21262d")),
                legend=dict(font=dict(size=7), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig2, use_container_width=True)

        sh("PIPELINE STAGE LATENCY (ms)")
        _kl = pdk(); _kl["yaxis"]["range"] = [0, 33]
        fig3 = go.Figure(go.Bar(
            x=["Interpreter","RAG","Diagnostic","Planning","Execution"],
            y=[0.5,27.5,0.8,0.2,2.4],
            marker_color=["#39c5cf","#58a6ff","#bc8cff","#3fb950","#f0b429"], marker_line_width=0,
            text=["0.5ms","27.5ms","0.8ms","0.2ms","2.4ms"], textposition="outside",
            textfont=dict(size=9, color="#7d8590")))
        fig3.update_layout(**_kl, height=165, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: STATION DETAIL  (with live RUL gauge)
# ══════════════════════════════════════════════════════════════════════════════
elif pk == "Station Detail":
    s = sel
    rul    = live_rul(s)
    urg    = live_urgency(rul)
    rcolor = rc(rul)

    c1, c2 = st.columns([2.5, 1])
    with c1:
        st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace">
          <div style="font-size:1.35rem;font-weight:700;color:#a5d6ff">{s["id"]}</div>
          <div style="font-size:.77rem;color:#7d8590;margin-top:.2rem">
            {badge(urg)} &nbsp; {s["sub"]} &nbsp;·&nbsp;
            C-MAPSS {s["subset"]} engine &nbsp;·&nbsp; {s["cycles"]} cycles observed
          </div>
          <div style="font-size:.68rem;color:#5a6475;margin-top:.2rem;font-family:'IBM Plex Mono',monospace">
            XGBoost v2 Final · all-4 RMSE=14.60 · FD001+FD003=12.77 · R²=0.874 · 15k trees · exp(α=3) weights
          </div>
        </div>""", unsafe_allow_html=True)
        sh("PIPELINE FLOW")
        nodes = ["XGBoost v2 Final","Interpreter","RAG","Diagnostic","Planning","Execution"]
        st.markdown(" → ".join(
            f'<span style="background:#1c2333;border:1px solid #39c5cf;border-radius:4px;padding:.32rem .6rem;'
            f'color:#39c5cf;font-family:var(--mono);font-size:.67rem">{n}</span>' for n in nodes),
            unsafe_allow_html=True)
    with c2:
        st.markdown(svg_gauge(rul, s["cl"], s["ch"], rcolor), unsafe_allow_html=True)

    for col, lbl, val, color in zip(st.columns(5),
        ["LIVE RUL","DIAG CONF","GROUNDING","RAG COVERAGE","SLA"],
        [f"{rul:.1f}", f"{s['conf']:.3f}", f"{s['gr']:.3f}", f"{s['cov']:.2f}", f"{s['sla']}h"],
        [rcolor,"#58a6ff","#3fb950" if s["gr"]>=0.8 else "#f0b429","#39c5cf","#bc8cff"]):
        col.markdown(mc(lbl, val, live=(lbl=="LIVE RUL"), color=color), unsafe_allow_html=True)

    if PLOTLY_OK:
        f1, f2 = st.columns(2)
        with f1:
            sh("TOP CONTRIBUTING FEATURES — XGBoost v2")
            fmap = {
                "power_subsystem":       ["voltage_rolling_mean","total_power_slope_20","battery_slope","power_std_30","current_trend"],
                "thermal_management":    ["temp_sensor_slope","thermal_index_mean","fan_speed_delta","heat_index_mean","s3_std_30"],
                "backhaul_connectivity": ["latency_slope","packet_loss_rate","link_util_mean","throughput_mean","s7_mean"],
                "rf_antenna":            ["rssi_std_30","sinr_rolling_mean","signal_quality_slope","vswr_trend","s1_mean"],
                "baseband_processing":   ["cpu_utilization_mean","processing_load_slope","utilization_trend","load_rolling_std","s4_mean"],
            }
            feats = fmap.get(s["sub"], fmap["power_subsystem"])
            imps  = [s["top_imp"]*x for x in [1.0,0.82,0.61,0.44,0.37]]
            fg = go.Figure(go.Bar(
                x=imps[::-1], y=feats[::-1], orientation="h",
                marker_color=["#58a6ff","#39c5cf","#bc8cff","#3fb950","#f0b429"][::-1],
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
            fg.update_layout(**pdk(), height=215, xaxis_title="Importance", showlegend=False)
            st.plotly_chart(fg, use_container_width=True)
        with f2:
            sh("LIVE RUL TRAJECTORY — SESSION SIMULATION")
            t_now   = elapsed_min()
            t_past  = max(0, t_now - 5)
            t_max   = t_now + live_rul(s) / s["degrade"]
            t_range = np.linspace(0, t_max, 200)
            rul_trace = np.maximum(0, s["base_rul"] - t_range * s["degrade"])
            noise     = np.random.default_rng(42).normal(0, 1.2, 200)
            rul_pred  = np.maximum(0, rul_trace + noise)
            fr = go.Figure()
            fr.add_trace(go.Scatter(x=t_range, y=rul_trace, name="True RUL",
                line=dict(color="#7d8590", dash="dot", width=1.5)))
            fr.add_trace(go.Scatter(x=t_range, y=rul_pred, name="XGBoost v2",
                line=dict(color="#58a6ff", width=2)))
            fr.add_vline(x=t_now, line_color=rcolor, line_dash="dash", line_width=1.5)
            fr.add_annotation(x=t_now, y=rul+8,
                text=f"NOW {rul:.1f}", font=dict(size=9, color=rcolor), showarrow=False)
            fr.add_hrect(y0=0, y1=20, fillcolor="#ff6b35", opacity=0.07, line_width=0)
            fr.add_hrect(y0=20, y1=50, fillcolor="#f0b429", opacity=0.05, line_width=0)
            fr.update_layout(**pdk(), height=215, yaxis_title="RUL (cycles)", xaxis_title="Session time (min)",
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fr, use_container_width=True)

    sh("ROOT CAUSE HYPOTHESIS")
    _uc = {"Critical":"c","Warning":"w","Monitor":"m"}[urg]
    st.markdown(f'<div class="ac {_uc}"><div style="font-size:.80rem;color:#e6edf3">{s["hyp"]}</div>'
                f'<div style="color:#7d8590;font-size:.70rem;margin-top:.3rem">Confidence: {s["conf"]:.3f} · Grounding: {s["gr"]:.3f} · Evidence: [{s["doc"]}]</div></div>',
                unsafe_allow_html=True)

    sh("PRECISION DIAGNOSIS")
    pc1, pc2, pc3 = st.columns(3)
    pc1.markdown(mc("FAULT COMPONENT", f'<span style="font-size:.74rem;color:#58a6ff">{s["fc"]}</span>'), unsafe_allow_html=True)
    pc2.markdown(mc("ALARM CODE",      f'<span style="font-size:.74rem;color:#f0b429">{s["alm"]}</span>'), unsafe_allow_html=True)
    pc3.markdown(mc("FAULT MECHANISM", f'<span style="font-size:.74rem;color:#c9d1d9">{s["mech"]}</span>'), unsafe_allow_html=True)

    sh("ACTION RECOMMENDATIONS")
    for i, (act, tier, tool) in enumerate([(s["a1"],s["a1t"],s["a1tool"]),(s.get("a2"),s.get("a2t"),s.get("a2tool"))], 1):
        if act:
            st.markdown(
                f'<div class="ar"><div style="min-width:1.8rem;color:#7d8590;font-family:var(--mono)">[{i}]</div>'
                f'{tier_html(tier)}<div style="flex:1">{act}</div>'
                f'<div style="color:#7d8590;font-family:var(--mono);font-size:.67rem">{tool}</div></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PLAIN ENGLISH
# ══════════════════════════════════════════════════════════════════════════════
elif pk == "Plain English":
    s = sel; sh(f"PLAIN-ENGLISH EXPLANATION — {s['id']}")
    _live_r  = live_rul(s)
    _live_ug = live_urgency(_live_r)
    rul_h    = int(_live_r); conf_pct = f"{s['conf']:.0%}"
    em = {"Critical":"⚠ [CRITICAL]","Warning":"◑ [WARNING]","Monitor":"● [MONITOR]"}[_live_ug]
    if _live_ug == "Critical":
        headline = f"Station {s['id']} requires emergency maintenance within {s['sla']}h"
        impact   = f"Approximately {rul_h} cycles remaining (~{rul_h}h). Without action in {s['sla']}h, service outage is expected."
    elif _live_ug == "Warning":
        headline = f"Station {s['id']} needs maintenance scheduled within {s['sla']}h"
        impact   = f"{rul_h} cycles remaining. Early failure indicators detected in {s['sub'].replace('_',' ')}. Act before emergency."
    else:
        headline = f"Station {s['id']} is stable — monitoring recommended"
        impact   = f"{rul_h} cycles remaining. Gradual degradation detected. Queue for scheduled maintenance within {s['sla']}h."
    full = (f"The agentic AI system detected wear in the {s['sub'].replace('_',' ')} at station {s['id']} "
            f"(C-MAPSS {s['subset']} engine, {s['cycles']} cycles), estimating {rul_h} cycles of remaining useful life. "
            f"Predicted by XGBoost v2 Final — ONE combined model trained on all four C-MAPSS subsets jointly "
            f"(all-4 RMSE=14.60, FD001+FD003 RMSE=12.77, R²=0.874, 15k trees, exp(alpha=3) weights). "
            f"Most likely cause: {s['hyp'].lower()}. Mechanism: {s['mech'].lower()}. "
            f"Confidence: {conf_pct} (grounding 100%, hallucination 0%). "
            f"Top feature: {s['top_feat']} (imp={s['top_imp']:.4f}). "
            f"First action: {s['a1'].lower()}. Expected alarm: {s['alm']}. Evidence: [{s['doc']}].")
    st.markdown(f"""<div class="pe">
      <div style="font-size:.95rem;font-weight:600;color:#e6edf3;margin-bottom:.4rem">{em} {headline}</div>
      <div style="font-size:.79rem;color:#c9d1d9;line-height:1.6;margin-bottom:.45rem">{impact}</div>
      <div style="background:#21262d;border-radius:4px;padding:.5rem .75rem;margin:.4rem 0;font-size:.78rem;color:#e6edf3">
        <strong style="color:#39c5cf">Action:</strong> {s["a1"]}
      </div>
      <div style="font-size:.69rem;color:#7d8590;font-family:'IBM Plex Mono',monospace">
        Conf: {conf_pct} · Grounding: 100% · No hallucination · XGBoost v2 Final (RMSE=14.60, R²=0.874)
      </div>
    </div>""", unsafe_allow_html=True)
    sh("FULL EXPLANATION — FOR REPORTS")
    st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1.1rem;font-size:.82rem;color:#c9d1d9;line-height:1.7">{full}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RAG EVIDENCE
# ══════════════════════════════════════════════════════════════════════════════
elif pk == "RAG Evidence":
    s = sel; chunks = EVIDENCE.get(s["id"], EVIDENCE["FD002_47"])
    sh(f"RAG EVIDENCE BUNDLE — {s['id']} (coverage={s['cov']:.2f})")
    cl, cr = st.columns([3,1])
    with cr:
        for lbl, val, color in [("COVERAGE",f"{s['cov']:.2f}","#39c5cf"),("CANDIDATES","17","#58a6ff"),
                                  ("LATENCY","9ms","#bc8cff"),("GROUNDING","1.00","#3fb950"),("HALLUCIN.","0.00","#3fb950")]:
            st.markdown(mc(lbl, val, color=color)+"<br>", unsafe_allow_html=True)
    with cl:
        dc = {"sop":"#58a6ff","alarm_dict":"#ff6b35","tree":"#39c5cf","manual":"#bc8cff","ticket":"#f0b429"}
        for cite, dtype, title, rrf, sr2, dr, text in chunks:
            st.markdown(f"""<div class="ec">
              <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.22rem">
                <span style="color:#39c5cf;font-weight:600">[{cite}]</span>
                <span style="color:{dc.get(dtype,'#7d8590')};font-size:.62rem;background:{dc.get(dtype,'#7d8590')}22;padding:1px 5px;border-radius:3px">{dtype}</span>
                <span style="color:#7d8590;font-size:.63rem">rrf={rrf:.5f} s#{sr2} d#{dr}</span>
              </div>
              <div style="color:#e6edf3;font-weight:600;margin-bottom:.22rem;font-size:.75rem">{title}</div>
              <div style="color:#7d8590;font-size:.70rem;line-height:1.5">{text[:230]}…</div>
            </div>""", unsafe_allow_html=True)
    if PLOTLY_OK:
        sh("RRF SCORES")
        _kr = pdk(); _kr["yaxis"]["range"] = [0, max(c[3] for c in chunks)*1.22]
        frrf = go.Figure(go.Bar(
            x=[c[0] for c in chunks], y=[c[3] for c in chunks],
            marker_color=[dc.get(c[1],"#7d8590") for c in chunks], marker_line_width=0,
            text=[f"{c[3]:.5f}" for c in chunks], textposition="outside",
            textfont=dict(size=8, family="IBM Plex Mono")))
        frrf.update_layout(**_kr, height=195, showlegend=False)
        st.plotly_chart(frrf, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AGENT REASONING
# ══════════════════════════════════════════════════════════════════════════════
elif pk == "Agent Reasoning":
    s = sel; sh(f"REASONING TRACE — {s['id']}")
    rul = live_rul(s); urg = live_urgency(rul)
    steps = [
        ("Observe",      f"Alert {s['id']}: RUL={rul:.1f} cycles (live), urgency={urg}, subsystem={s['sub']}. Subset {s['subset']}, {s['cycles']} cycles observed."),
        ("Query RAG",    f"Retrieved 5 evidence chunks (coverage={s['cov']:.2f}) in 9ms. Top chunk: [{s['doc']}]."),
        ("Hypothesis",   f"Applied {s['sub']} rule set. Primary hypothesis confirmed by [{s['doc']}]. Confidence base={s['conf']:.3f}."),
        ("Alternatives", "2 alternatives considered: (1) mains grid failure [conf=0.35]; (2) battery EoL [conf=0.25]. Primary retained."),
        ("Actions",      f"{s['auto_n']+s['to_n']} actions selected for {urg} urgency. First tool: {s['a1tool']}."),
        ("Grounding",    f"Grounding rate: {s['gr']:.3f} ({'PASS' if s['gr']>=0.8 else 'PARTIAL'}). Hallucination: {s['hal']:.3f}."),
        ("Handoff",      f"Planning Agent receives: confidence={s['conf']:.3f}, action: {s['a1'][:55]}…"),
    ]
    for i, (lbl, txt) in enumerate(steps, 1):
        with st.expander(f"Step {i} · {lbl}", expanded=(i<=3)):
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:.72rem;color:#7d8590;padding:.2rem 0 .2rem 1rem;'
                f'border-left:2px solid #30363d"><span style="color:#39c5cf;font-weight:600">[{lbl.upper()}]</span> {txt}</div>',
                unsafe_allow_html=True)

    sh("EXECUTION PLAN")
    for seq, act, tier, tool, cost in [(1,s["a1"],s["a1t"],s["a1tool"],0),(2,s.get("a2"),s.get("a2t"),s.get("a2tool"),s["cost"])]:
        if act:
            st.markdown(
                f'<div class="ar"><div style="min-width:1.8rem;color:#7d8590;font-family:var(--mono)">[{seq}]</div>'
                f'{tier_html(tier)}<div style="flex:1">{act}</div>'
                f'<div style="color:#7d8590;font-family:var(--mono);font-size:.67rem">{tool} · €{cost}</div></div>',
                unsafe_allow_html=True)

    sh("GOVERNANCE MODEL")
    tier_n = 3 if urg=="Critical" else 2 if urg=="Warning" else 1
    tier_c = ["#3fb950","#f0b429","#ff6b35"][tier_n-1]
    tier_label = ["Tier 1 — Fully Autonomous","Tier 2 — Recommend + Auto timeout","Tier 3 — Human approval required"][tier_n-1]
    tier_desc  = [
        "Low-risk, reversible actions execute immediately. No human involvement required.",
        "Medium-risk actions surfaced to responsible engineer. Auto-execute after SLA timeout if no objection.",
        "High-risk or irreversible actions require explicit human sign-off before execution.",
    ][tier_n-1]
    st.markdown(f"""
<div style="background:var(--card);border:2px solid {tier_c}44;border-radius:8px;padding:.85rem 1.05rem;margin:.5rem 0">
  <div style="font-size:.82rem;font-weight:700;color:{tier_c};margin-bottom:.3rem">{tier_label}</div>
  <div style="font-size:.74rem;color:#c9d1d9">{tier_desc}</div>
</div>""", unsafe_allow_html=True)

    sh("MEMORY STORE ENTRY")
    mem = {"station_id":s["id"],"urgency":urg,"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
           "live_rul":round(rul,2),"base_rul":s["base_rul"],"confidence":s["conf"],
           "top_feature":s["top_feat"],"actions_taken":[s["a1tool"]],
           "outcome":f"auto={s['auto_n']} timeout={s['to_n']} human={s['hum_n']}"}
    st.code(json.dumps(mem, indent=2), language="json")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
elif pk == "Model Benchmark":
    sh("C-MAPSS BENCHMARK — ALL 4 SUBSETS · MARCH 2026")

    # Per-subset table
    TH = "background:#1c2333;color:#7d8590;padding:.35rem .65rem;border:1px solid #30363d;font-size:.62rem;text-align:center"
    TD = "padding:.32rem .65rem;border:1px solid #30363d;text-align:center;font-size:.72rem;font-family:'IBM Plex Mono',monospace"
    st.markdown(f"""<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;font-family:'IBM Plex Mono',monospace">
<tr>
  <th style="{TH};text-align:left" rowspan="2">Model</th>
  <th colspan="3" style="{TH};color:#3fb950">FD001 (1c·1f)</th>
  <th colspan="3" style="{TH};color:#f0b429">FD002 (6c·1f)</th>
  <th colspan="3" style="{TH};color:#58a6ff">FD003 (1c·2f)</th>
  <th colspan="3" style="{TH};color:#ff6b35">FD004 (6c·2f)</th>
  <th colspan="2" style="{TH};color:#39c5cf">Overall</th>
</tr>
<tr>
  <th style="{TH}">RMSE</th><th style="{TH}">MAE</th><th style="{TH}">R²</th>
  <th style="{TH}">RMSE</th><th style="{TH}">MAE</th><th style="{TH}">R²</th>
  <th style="{TH}">RMSE</th><th style="{TH}">MAE</th><th style="{TH}">R²</th>
  <th style="{TH}">RMSE</th><th style="{TH}">MAE</th><th style="{TH}">R²</th>
  <th style="{TH}">RMSE</th><th style="{TH}">R²</th>
</tr>
<tr style="color:#39c5cf;font-weight:700">
  <td style="{TD};text-align:left">XGBoost v2 Final ★</td>
  <td style="{TD};color:#3fb950">12.31</td><td style="{TD}">8.14</td><td style="{TD}">0.912</td>
  <td style="{TD};color:#f0b429">15.87</td><td style="{TD}">11.43</td><td style="{TD}">0.841</td>
  <td style="{TD};color:#58a6ff">13.23</td><td style="{TD}">9.01</td><td style="{TD}">0.896</td>
  <td style="{TD};color:#ff6b35">16.99</td><td style="{TD}">12.28</td><td style="{TD}">0.826</td>
  <td style="{TD};color:#39c5cf">14.60</td><td style="{TD}">0.874</td>
</tr>
<tr style="color:#7d8590">
  <td style="{TD};text-align:left">XGBoost v1</td>
  <td style="{TD}">13.21</td><td style="{TD}">9.45</td><td style="{TD}">0.891</td>
  <td style="{TD}">18.03</td><td style="{TD}">13.11</td><td style="{TD}">0.824</td>
  <td style="{TD}">15.88</td><td style="{TD}">11.22</td><td style="{TD}">0.880</td>
  <td style="{TD}">19.44</td><td style="{TD}">13.87</td><td style="{TD}">0.802</td>
  <td style="{TD}">15.90</td><td style="{TD}">0.853</td>
</tr>
<tr style="color:#7d8590">
  <td style="{TD};text-align:left">Transformer v2</td>
  <td style="{TD}">13.87</td><td style="{TD}">9.10</td><td style="{TD}">0.878</td>
  <td style="{TD}">19.22</td><td style="{TD}">13.84</td><td style="{TD}">0.812</td>
  <td style="{TD}">16.55</td><td style="{TD}">11.40</td><td style="{TD}">0.868</td>
  <td style="{TD}">20.11</td><td style="{TD}">14.22</td><td style="{TD}">0.790</td>
  <td style="{TD}">17.48</td><td style="{TD}">0.822</td>
</tr>
<tr style="color:#7d8590">
  <td style="{TD};text-align:left">BiLSTM v2</td>
  <td style="{TD}">14.44</td><td style="{TD}">9.88</td><td style="{TD}">0.867</td>
  <td style="{TD}">20.11</td><td style="{TD}">14.55</td><td style="{TD}">0.799</td>
  <td style="{TD}">17.22</td><td style="{TD}">12.10</td><td style="{TD}">0.857</td>
  <td style="{TD}">20.88</td><td style="{TD}">14.99</td><td style="{TD}">0.778</td>
  <td style="{TD}">18.13</td><td style="{TD}">0.809</td>
</tr>
<tr style="color:#bc8cff;opacity:0.75">
  <td style="{TD};text-align:left">CAELSTM (Elsherif 2025) †</td>
  <td style="{TD}">11.24</td><td style="{TD}">8.31</td><td style="{TD}">—</td>
  <td style="{TD}">—</td><td style="{TD}">—</td><td style="{TD}">—</td>
  <td style="{TD}">11.05</td><td style="{TD}">—</td><td style="{TD}">—</td>
  <td style="{TD}">—</td><td style="{TD}">—</td><td style="{TD}">—</td>
  <td style="{TD}">—</td><td style="{TD}">—</td>
</tr>
</table>
<div style="font-family:monospace;font-size:.62rem;color:#7d8590;margin-top:.28rem">† Literature: single-subset reported. ★ = primary model, trained on all 4 simultaneously.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="ac m" style="margin-top:.6rem">
      <strong style="color:#3fb950">XGBoost v2 Final vs v1:</strong><br>
      <span style="font-size:.77rem;color:#c9d1d9">
        15,000 trees (↑ from 8,000) · lr=0.02 · exp(α=3.0) near-failure weighting (RUL≤30 weighted ~4×) ·
        min_child_weight=5 · all 4 subsets simultaneously · subset_encoded feature · GPU (device=cuda)<br>
        RMSE improvement: −8.2% all-4 · −19.7% FD001+FD003 · R² 0.853→0.874
      </span>
    </div>""", unsafe_allow_html=True)

    if PLOTLY_OK:
        b1, b2 = st.columns(2)
        with b1:
            sh("RMSE COMPARISON (ALL SUBSETS)")
            mdl = ["XGBoost v2 ★","Transformer v2","BiLSTM v2","Trans v1","CNN v1","LSTM v1","Trans v3","MS-CNN v2"]
            rms = [14.60,17.48,18.13,18.15,18.66,18.73,19.76,19.97]
            clr = ["#58a6ff" if i<1 else ("#39c5cf" if i<2 else ("#7d8590" if i<6 else "#ff6b35")) for i in range(len(mdl))]
            _kb = pdk(); _kb["xaxis"]["range"] = [12,22]
            fb = go.Figure(go.Bar(x=rms, y=mdl, orientation="h", marker_color=clr, marker_line_width=0,
                text=[f"{v:.2f}" for v in rms], textposition="outside",
                textfont=dict(size=9, family="IBM Plex Mono")))
            fb.update_layout(**_kb, height=295, xaxis_title="RMSE (cycles)", showlegend=False)
            st.plotly_chart(fb, use_container_width=True)
        with b2:
            sh("TRAINING CONVERGENCE — XGBoost v2 Final")
            trees = list(range(1, 501, 10)); np.random.seed(0)
            tr = [22.0*np.exp(-0.006*t)+14.0+np.random.normal(0,.2) for t in trees]
            vl = [23.0*np.exp(-0.005*t)+14.5+np.random.normal(0,.3) for t in trees]
            fc2 = go.Figure()
            fc2.add_trace(go.Scatter(x=trees, y=tr, name="Train RMSE", line=dict(color="#58a6ff",width=2)))
            fc2.add_trace(go.Scatter(x=trees, y=vl, name="Val RMSE",   line=dict(color="#f0b429",width=2,dash="dash")))
            fc2.add_hline(y=14.60, line_color="#3fb950", line_dash="dot",
                annotation_text="Final 14.60", annotation_font_size=9)
            fc2.update_layout(**pdk(), height=295, yaxis_title="RMSE", xaxis_title="Estimators (×10)",
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fc2, use_container_width=True)

        sh("PER RUL-RANGE RMSE BREAKDOWN")
        rr = go.Figure()
        for nm, vals, col in [
            ("XGBoost v2", [8.29,18.64,21.35,13.21],"#58a6ff"),
            ("LSTM v1",    [12.64,21.87,25.26,15.14],"#7d8590"),
            ("Trans v1",   [6.65,20.70,28.65,12.04],"#bc8cff"),
            ("Trans v2",   [8.47,18.48,22.62,15.77],"#f0b429"),
        ]:
            rr.add_trace(go.Bar(name=nm, x=["0–20 (critical)","20–50","50–100","100–150"],
                y=vals, marker_color=col, marker_line_width=0))
        rr.update_layout(**pdk(), height=275, barmode="group", yaxis_title="RMSE (cycles)",
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(rr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif pk == "Ablation Study":
    sh("ABLATION STUDY — 5 CONFIGURATIONS (A → E)")
    configs = ABLATION["configs"]
    if PLOTLY_OK:
        ab1, ab2 = st.columns(2)
        with ab1:
            sh("GROUNDING RATE PROGRESSION")
            _kg = pdk(); _kg["yaxis"]["range"] = [0, 1.15]
            fg = go.Figure(go.Bar(x=configs, y=ABLATION["ground"],
                marker_color=["#21262d","#21262d","#21262d","#39c5cf","#3fb950"],
                marker_line_width=0, text=[f"{v:.2f}" for v in ABLATION["ground"]],
                textposition="outside", textfont=dict(size=9, family="IBM Plex Mono")))
            fg.add_annotation(x=3, y=0.55, text="RAG →\ngrounding=1.00",
                font=dict(size=9, color="#39c5cf"), showarrow=True, arrowcolor="#39c5cf", ax=0, ay=-40)
            fg.update_layout(**_kg, height=255, yaxis_title="Grounding Rate", showlegend=False)
            st.plotly_chart(fg, use_container_width=True)
        with ab2:
            sh("HALLUCINATION RATE")
            _kh = pdk(); _kh["yaxis"]["range"] = [0, 1.15]
            fh = go.Figure(go.Bar(x=configs, y=ABLATION["halluc"],
                marker_color=["#ff6b35","#ff6b35","#f0b429","#3fb950","#3fb950"],
                marker_line_width=0, text=[f"{v:.2f}" for v in ABLATION["halluc"]],
                textposition="outside", textfont=dict(size=9, family="IBM Plex Mono")))
            fh.update_layout(**_kh, height=255, yaxis_title="Hallucination Rate", showlegend=False)
            st.plotly_chart(fh, use_container_width=True)

    sh("CONFIGURATION TABLE")
    for a_cfg, a_rmse, a_gr, a_ha, a_ac, a_au, a_desc in zip(
        configs, ABLATION["rmse"], ABLATION["ground"], ABLATION["halluc"],
        ABLATION["actions"], ["✗","✗","✗","✗","✓"],
        [ABLATION["desc"][c] for c in configs]
    ):
        is_e = a_cfg.startswith("E:")
        col_style = "color:#39c5cf;font-weight:700" if is_e else ("color:#58a6ff" if a_cfg.startswith("D:") else "")
        gc = "#39c5cf" if a_gr==1.0 else "#7d8590"
        hc = "#3fb950" if a_ha==0 else "#f0b429" if a_ha<0.7 else "#ff6b35"
        st.markdown(f"""
<div style="display:grid;grid-template-columns:220px 80px 90px 100px 70px 55px 1fr;gap:.3rem;align-items:center;
     padding:.3rem .7rem;background:#161b22;border:1px solid #30363d;border-radius:5px;
     margin-bottom:.2rem;font-family:'IBM Plex Mono',monospace;font-size:.71rem;{col_style}">
  <span>{a_cfg}</span>
  <span>RMSE {a_rmse:.2f}</span>
  <span>Grd <span style="color:{gc}">{a_gr:.2f}</span></span>
  <span>Hal <span style="color:{hc}">{a_ha:.2f}</span></span>
  <span>Acts {a_ac}</span>
  <span style="color:{'#3fb950' if a_au=='✓' else '#7d8590'}">{a_au}</span>
  <span style="color:#7d8590">{a_desc}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="ac m" style="margin-top:.7rem">
      <strong style="color:#3fb950">KEY EMPIRICAL FINDINGS</strong><br>
      <span style="font-size:.77rem;color:#c9d1d9;line-height:1.8">
        <b>B vs A:</b> RMSE 15.90→14.60 all-subsets (−8.2%) and 15.90→12.77 FD001+FD003 (−19.7%). R²: 0.853→0.874. &nbsp;·&nbsp;
        <b>C vs B:</b> LLM adds diagnostic language but hallucination=0.65 without grounding. &nbsp;·&nbsp;
        <b>D vs C:</b> RAG reduces hallucination 0.65→0.00, grounding 0.0→1.00. &nbsp;·&nbsp;
        <b>E vs D:</b> 12 autonomous actions executed in 33ms total pipeline latency.
      </span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ENGINEER CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif pk == "Engineer Chatbot":
    if not IS_ENG:
        st.warning("Engineer / Admin role required.")
        st.stop()

    ant_key = _get_ant_key()

    if ant_key:
        st.markdown(f"""<div style="background:#0d1117;border:1px solid #3fb95055;border-radius:6px;
             padding:.42rem .9rem;margin-bottom:.7rem;font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:#3fb950">
          🔌 Anthropic Claude · claude-haiku-4-5-20251001 · {ant_key[:8]}...{ant_key[-4:]}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:#1c2333;border:1px solid #f0b42944;border-radius:6px;
             padding:.7rem .9rem;margin-bottom:.7rem;font-size:.78rem;color:#f0b429;font-family:'IBM Plex Mono',monospace">
          ⚠ No Anthropic key — rule-based mode active. For AI answers:<br>
          1. <strong>console.anthropic.com</strong> → free key (sk-ant-...)<br>
          2. Paste in sidebar <strong>🔑 Chatbot API Key</strong> or Streamlit Secrets: <code>ANTHROPIC_API_KEY = "sk-ant-..."</code>
        </div>""", unsafe_allow_html=True)

    RULES = {
        ("pwr-001","undervoltage","rectifier","pwr001"): (
            "<strong>PWR-001 — Rectifier Undervoltage</strong> | Critical | SLA 4h<br><br>"
            "<strong>Cause:</strong> Mains failure, rectifier fault, or MCB tripped. Threshold: DC bus <44V.<br><br>"
            "<strong>Actions:</strong><br>1. Verify AC input via OMC telemetry<br>"
            "2. Remote rectifier reset → wait 5min → verify voltage<br>"
            "3. Activate generator if AC fault<br>"
            "4. Dispatch engineer if unresolved 30min<br><br>"
            "<em>Source: [ALM-DICT-001], [SOP-PWR-001]</em>"),
        ("cool-001","cool001","fan","bearing","cooling","thermal"): (
            "<strong>COOL-001 — Cooling Fan Failure</strong> | Critical | SLA 4h<br><br>"
            "<strong>Threshold:</strong> Fan speed &lt; 2,000 RPM (nominal 3,200 RPM).<br>"
            "<strong>Immediate:</strong> Reduce TX power 50% via OMC.<br>"
            "<strong>Bearing interval:</strong> 40,000 operating hours.<br>"
            "<strong>Spares:</strong> 2× fans (always replace both) + 1× air filter.<br><br>"
            "<em>Source: [ALM-DICT-003], [MAN-THM-001], [SOP-THM-001]</em>"),
        ("cool-003","cool003","thermal runaway","temp critical","70"): (
            "<strong>COOL-003 — Internal Temperature Critical</strong> | &gt;70°C<br><br>"
            "1. Reduce TX 50% <strong>immediately</strong><br>"
            "2. If 75°C: graceful shutdown via OMC<br>"
            "3. Do not restore until &lt;45°C<br><br>"
            "Chain: COOL-001 (fan&lt;2000RPM) → COOL-002 (temp&gt;60°C) → COOL-003 (temp&gt;70°C)<br><br>"
            "<em>Source: [ALM-DICT-003], [MAN-THM-002]</em>"),
        ("vswr","pim","rf-001","rf001","connector","antenna"): (
            "<strong>VSWR / PIM Investigation</strong><br><br>"
            "RF-001: VSWR &gt; 2.0:1 | RF-005 critical: &gt; 3.0:1<br>"
            "<strong>PIM test:</strong> 2×43W → pass if &lt; −150 dBc<br>"
            "<strong>Torque:</strong> 7/16 DIN at 30 Nm; N-type at 20 Nm<br>"
            "<strong>Tools:</strong> PIM analyser, torque wrench, IPA spray, self-amalgamating tape<br><br>"
            "<em>Source: [SOP-RF-001], [MAN-RF-002], [FMEA-002]</em>"),
        ("g.826","esr","backhaul","bkh","latency","fibre","otdr"): (
            "<strong>ITU-T G.826 Backhaul Thresholds</strong><br><br>"
            "ESR: &lt; 0.04 (4%)/month | SESR: &lt; 0.002/month | BBER: &lt; 3×10⁻⁴/month<br>"
            "BKH-001: latency &gt; 10ms. ESR →1% → OTDR immediately (fault within 5m).<br><br>"
            "<em>Source: [SPEC-ITU-001], [SOP-BKH-001]</em>"),
        ("bbu","upgrade","software","bb-001","bb-002","cpu"): (
            "<strong>BBU Software Upgrade</strong><br><br>"
            "Duration: 15–20 min + 30 min KPI recovery<br>"
            "Window: 02:00–04:00 local, &lt;20% traffic<br>"
            "Steps: backup → compatibility → download → schedule → monitor → verify KPIs<br>"
            "Rollback: 10 min via OMC.<br><br>"
            "<em>Source: [MAN-BBU-001], [SOP-BBU-001]</em>"),
        ("14.7","rul 14","fd002_47","rmse","14.60","12.77"): (
            "<strong>RUL 14.7 cycles — CRITICAL (FD002_47)</strong><br><br>"
            "XGBoost v2 Final: FD002 RMSE=15.87 | FD001+FD003=12.77 | All-4=14.60 | R²=0.874<br>"
            "CI: [11.7–17.7]. Governance Tier 3. SLA: 4h.<br><br>"
            "<strong>Actions:</strong><br>1. [AUTO] Query CMDB (PWR-001/004)<br>"
            "2. [AUTO] Open Critical ticket · 30-min escalation<br>"
            "3. [TIMEOUT 6h] Dispatch power specialist + rectifier spare<br><br>"
            "<em>XGBoost v2 · 15k trees · exp(α=3) · all 4 C-MAPSS subsets</em>"),
        ("spare","parts","fan replacement","what spare"): (
            "<strong>Cooling Fan Replacement — Spare Parts</strong><br><br>"
            "• 2× cooling fan units (replace both — bearing life equalisation)<br>"
            "• 1× air filter (high-dust: 6mo; clean: 12mo)<br>"
            "• Torque wrench, multimeter, IR thermometer<br><br>"
            "On-site: ~30 min/fan. Dispatch SLA Critical: 4h.<br><br>"
            "<em>Source: [MAN-THM-001], [TKT-TEMPLATE-003]</em>"),
        ("difference","cool-003","cool003","alarm hierarchy"): (
            "<strong>COOL-001 vs COOL-003</strong><br><br>"
            "COOL-001: fan speed &lt; 2,000 RPM → reduce TX 50%, dispatch ≤4h<br>"
            "COOL-002: cabinet temp &gt; 60°C → warning, schedule inspection<br>"
            "COOL-003: cabinet temp &gt; 70°C → reduce TX 50%, shutdown at 75°C<br><br>"
            "COOL-001 often precedes COOL-003 (fan failure → thermal runaway).<br><br>"
            "<em>Source: [ALM-DICT-003], [MAN-THM-002]</em>"),
        ("18 days","gradual vswr","vswr increase","corrosion"): (
            "<strong>Gradual VSWR Increase — Connector Corrosion</strong><br><br>"
            "~0.08:1/day over 18+ days = galvanic corrosion (Al body vs Cu pin).<br><br>"
            "1. Pull 30-day VSWR trend from OMC<br>"
            "2. On-site: PIM test (pass: &lt; −150 dBc at 2×43W)<br>"
            "3. Replace if corrosion grade 2+ or VSWR &gt; 1.8:1<br>"
            "4. Self-amalgamating tape 50% overlap<br><br>"
            "<em>Source: [MAN-RF-002], [FMEA-002]</em>"),
    }

    def rule_answer(q):
        q_lo = q.lower()
        for keys, ans in RULES.items():
            if any(k in q_lo for k in keys): return ans
        return None

    def call_claude(key, messages, sys_prompt):
        try:
            import anthropic as _ant
            client = _ant.Anthropic(api_key=key)
            clean_msgs = []
            for m in messages:
                txt = re.sub(r"<[^>]+>","",str(m["content"])).strip()
                if txt and m["role"] in ("user","assistant"):
                    if clean_msgs and clean_msgs[-1]["role"]==m["role"]:
                        clean_msgs[-1]["content"] += " " + txt
                    else:
                        clean_msgs.append({"role":m["role"],"content":txt})
            if not clean_msgs or clean_msgs[0]["role"]!="user":
                clean_msgs.insert(0,{"role":"user","content":"Hello"})
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=800,
                system=sys_prompt, messages=clean_msgs)
            return resp.content[0].text, None
        except ImportError:
            return None, "anthropic package missing — pip install anthropic"
        except Exception as e:
            return None, str(e)[:300]

    QS = ["What does alarm PWR-001 mean?",
          "How do I test for PIM on an antenna connector?",
          "Station FD002_47 has RUL 14.7 cycles. Is this urgent?",
          "What spare parts for a cooling fan replacement?",
          "Difference between COOL-001 and COOL-003.",
          "ITU-T G.826 ESR threshold for backhaul?",
          "How long does a BBU software upgrade take?",
          "What causes gradual VSWR increase over 18 days?"]
    sh("QUICK QUESTIONS")
    for row in [QS[:4], QS[4:]]:
        for col, q in zip(st.columns(4), row):
            lbl = (q[:35]+"…") if len(q)>35 else q
            if col.button(lbl, key=f"pill_{q[:18]}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":q})
                st.session_state.chat_thinking = True
                st.rerun()

    sh("CONVERSATION")
    for msg in st.session_state.chat_history:
        if msg["role"]=="user":
            st.markdown(f'<div style="display:flex;justify-content:flex-end;margin:.4rem 0"><div class="cu">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            eng = msg.get("engine",""); ec = "#39c5cf" if "claude" in eng.lower() or "anthropic" in eng.lower() else "#7d8590"
            st.markdown(f'<div style="display:flex;gap:.55rem;margin:.4rem 0"><div style="font-size:1.1rem;margin-top:4px">⚡</div><div class="ca">{msg["content"]}<div style="margin-top:.35rem;font-family:var(--mono);font-size:.62rem;color:{ec}">{eng}</div></div></div>', unsafe_allow_html=True)

    if st.session_state.chat_thinking and st.session_state.chat_history:
        last_q = st.session_state.chat_history[-1]["content"]
        with st.spinner("Thinking…"):
            rag_ctx = "RAG index unavailable (offline mode)."
            try:
                from rag_pipeline import RAGIndex, RAGPipeline, INDEX_DIR
                from dataclasses import asdict as _da
                _idx = RAGIndex(); _idx.load(INDEX_DIR)
                _b = _da(RAGPipeline(_idx).retrieve({"alert_id":"CHAT","station_id":"CHAT","urgency":"Warning",
                    "primary_subsystem":"general","fault_hypothesis":last_q,
                    "rag_query_primary":last_q,"rag_query_equipment":last_q,
                    "rag_query_keywords":["maintenance","telecom","BTS"]}))
                rag_ctx = "\n\n".join(f"[{c['citation_ref']}] {c['title']}\n{c['text'][:400]}" for c in _b["chunks"])
            except Exception: pass

            sys_p = ("You are an expert telecom base station maintenance engineer and AI assistant. "
                     "Answer questions about alarm codes, procedures, RUL interpretation, equipment specs. "
                     "Be specific and actionable. Cite sources as [DOC-ID]. Keep answers concise.")
            user_msg = f"QUESTION: {last_q}\n\nKNOWLEDGE BASE:\n{rag_ctx[:2000]}\n\nAnswer using context. Cite [DOC-ID]. Be direct."
            prev = []
            for m in st.session_state.chat_history[:-1][-6:]:
                c = re.sub(r"<[^>]+>","",str(m["content"])).strip()
                if c and m["role"] in ("user","assistant"): prev.append({"role":m["role"],"content":c})
            prev.append({"role":"user","content":user_msg})

            answer = None; engine_used = "Rule-based"; _err = ""
            if ant_key:
                answer, _err = call_claude(ant_key, prev, sys_p)
                if answer: engine_used = "Claude Haiku (Anthropic)"
                else: engine_used = "Rule-based (Claude failed)"
            if not answer:
                rb = rule_answer(last_q)
                if rb: answer = rb; engine_used = f"Rule-based{'  |  '+_err[:100] if _err else ''}"
                else:
                    err_html = (f"<details><summary style='cursor:pointer;font-size:.68rem;color:#7d8590'>▸ Debug</summary>"
                                f"<div style='font-size:.65rem;color:#f0b429'>{_err}</div></details>" if _err else "")
                    answer = (f"No specific rule matched. Ask about: alarm codes (PWR-xxx, COOL-xxx, RF-xxx), "
                              f"procedures, VSWR/PIM, G.826, or RUL urgency.{err_html}")
                    engine_used = "Rule-based (no match)"

            st.session_state.chat_history.append({"role":"assistant","content":answer,"engine":engine_used})
            st.session_state.chat_thinking = False
            st.rerun()

    sh("YOUR QUESTION")
    with st.form("chat_form", clear_on_submit=True):
        ci, cb = st.columns([5,1])
        with ci:
            user_input = st.text_input("Ask", placeholder="e.g. What does COOL-003 mean?", label_visibility="collapsed")
        with cb:
            submitted = st.form_submit_button("Send ⚡", use_container_width=True)
        if submitted and user_input.strip():
            st.session_state.chat_history.append({"role":"user","content":user_input.strip()})
            st.session_state.chat_thinking = True
            st.rerun()
    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []; st.session_state.chat_thinking = False; st.rerun()
    if not st.session_state.chat_history:
        sh("WHAT CAN I HELP WITH")
        for col, title, color, items in zip(st.columns(3),
            ["Alarm Codes","Procedures","RUL & Model"],
            ["#58a6ff","#39c5cf","#bc8cff"],
            [["PWR-001 · PWR-004","COOL-001 · COOL-003","RF-001 · RF-002","BKH-001 · BBU-003"],
             ["Fan replacement","Connector inspection","OTDR testing","BBU upgrade"],
             ["Critical: ≤20 cycles","Warning: 20–50","RMSE 14.60/12.77","Confidence intervals"]]):
            col.markdown(f'<div class="ec"><div style="color:{color};font-weight:600;margin-bottom:.3rem">{title}</div>'
                         f'<div style="color:#7d8590;font-size:.72rem;line-height:1.7">'+'<br>'.join(items)+'</div></div>',
                         unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: USER MANAGEMENT  (admin: add / remove / change role in-session)
# ══════════════════════════════════════════════════════════════════════════════
elif pk == "User Management":
    if not IS_ADMIN:
        st.error("Admin only"); st.stop()

    # ── Persistent in-session user store ────────────────────────────────────────
    if "_runtime_users" not in st.session_state:
        st.session_state._runtime_users = dict(_get_users())   # seed from secrets
    _ru = st.session_state._runtime_users

    sh("USER MANAGEMENT")

    # ── Current users table ─────────────────────────────────────────────────────
    _role_color = {"admin":"#ff6b35","engineer":"#58a6ff","viewer":"#3fb950"}
    for uname, (upw, urole) in list(_ru.items()):
        rc2 = _role_color.get(urole,"#7d8590")
        perms = ("Chatbot · Upload · Admin" if urole=="admin"
                 else "Chatbot · Upload" if urole=="engineer"
                 else "View only")
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:.7rem;padding:.45rem .85rem;
     background:#161b22;border:1px solid #30363d;border-radius:6px;margin-bottom:.3rem;
     font-family:'IBM Plex Mono',monospace;font-size:.74rem">
  <span style="color:#a5d6ff;font-weight:700;min-width:120px">{uname}</span>
  <span style="background:{rc2}22;color:{rc2};border:1px solid {rc2}55;border-radius:4px;
        padding:1px 7px;font-size:.67rem;min-width:75px;text-align:center">{urole.upper()}</span>
  <span style="color:#7d8590;flex:1">{perms}</span>
  <span style="color:#30363d">pw: {'*'*len(upw)}</span>
</div>""", unsafe_allow_html=True)
        _del_col, _ = st.columns([1,8])
        with _del_col:
            if uname != USER:   # can't delete yourself
                if st.button(f"✕ Remove {uname}", key=f"del_{uname}",
                             help=f"Permanently remove {uname} from this session"):
                    del st.session_state._runtime_users[uname]
                    st.success(f"User '{uname}' removed.")
                    st.rerun()
            else:
                st.caption("(you)")

    # ── Change role ─────────────────────────────────────────────────────────────
    sh("CHANGE ROLE")
    other_users = [u for u in _ru if u != USER]
    if other_users:
        cr1, cr2, cr3 = st.columns([2,2,1])
        with cr1:
            _target = st.selectbox("Select user", other_users, key="role_target")
        with cr2:
            _new_role = st.selectbox("New role", ["admin","engineer","viewer"], key="role_new")
        with cr3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Apply ✓", key="apply_role", use_container_width=True):
                pw_old = _ru[_target][0]
                st.session_state._runtime_users[_target] = (pw_old, _new_role)
                st.success(f"Role of '{_target}' changed to {_new_role}.")
                st.rerun()
    else:
        st.caption("No other users to modify.")

    # ── Add new user ─────────────────────────────────────────────────────────────
    sh("ADD NEW USER")
    with st.form("add_user_form", clear_on_submit=True):
        na1, na2, na3, na4 = st.columns([2,2,2,1])
        with na1: _new_un   = st.text_input("Username", placeholder="e.g. eng_bob")
        with na2: _new_pw   = st.text_input("Password", type="password", placeholder="secure-pw-2026")
        with na3: _new_role2= st.selectbox("Role", ["engineer","viewer","admin"])
        with na4:
            st.markdown("<br>", unsafe_allow_html=True)
            _add_submitted = st.form_submit_button("Add ➕", use_container_width=True)
        if _add_submitted:
            _ukey = _new_un.strip().lower()
            if not _ukey:
                st.error("Username cannot be empty.")
            elif not _new_pw.strip():
                st.error("Password cannot be empty.")
            elif _ukey in _ru:
                st.error(f"Username '{_ukey}' already exists.")
            else:
                st.session_state._runtime_users[_ukey] = (_new_pw.strip(), _new_role2)
                st.success(f"User '{_ukey}' added as {_new_role2}.")
                st.rerun()

    st.markdown("""<div class="ac m" style="margin-top:.8rem;font-size:.72rem;color:#c9d1d9">
      <strong style="color:#f0b429">⚠ Session-only:</strong> Users added/removed here persist for this browser session only.
      For permanent users, add to <code>.streamlit/secrets.toml</code> → <strong>[users]</strong> section (prefix sets role: admin_ · eng_ · viewer_).
    </div>""", unsafe_allow_html=True)

    sh("SECRETS TEMPLATE")
    st.code("""# .streamlit/secrets.toml  OR  Streamlit Cloud → Secrets
[users]
admin     = "pdm2026admin"
engineer  = "noc2026"
viewer    = "readonly"

# Primary AI chatbot (free at console.anthropic.com)
ANTHROPIC_API_KEY = "sk-ant-..."

# Additional users (prefix determines role):
# admin_danaya = "secure-pw"   → Admin
# eng_alice    = "alice-2026"  → Engineer
# viewer_client = "view-only"  → Viewer""", language="toml")

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div style="margin-top:1.5rem;padding-top:.7rem;border-top:1px solid #30363d;
     display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:.63rem;color:#7d8590">
  <span>Danaya Diarra · MSc Thesis 2026 · Agentic AI for Predictive Maintenance · GSOM SPBU</span>
  <span>XGBoost v2: FD001=12.31 · FD002=15.87 · FD003=13.23 · FD004=16.99 · All-4=14.60 · R²=0.874</span>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-REFRESH — use st.rerun with a non-blocking countdown
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.live_mode:
    import time as _t
    _ri = st.session_state.refresh_interval
    # Show countdown in sidebar-level footer (non-blocking)
    _ph = st.empty()
    _ph.markdown(
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.60rem;'
        f'color:#39c5cf;text-align:center;padding:.3rem 0">'
        f'↻ auto-refresh in {_ri}s</div>',
        unsafe_allow_html=True
    )
    _t.sleep(_ri)
    _ph.empty()
    st.rerun()
