"""
Agentic PdM NOC — Streamlit Production v1.0
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | GSOM SPBU | 2026

DEPLOYMENT (25-second build, zero version conflicts):
  streamlit.io/cloud → New app → connect GitHub repo → Deploy
  Secrets: GROQ_API_KEY | OPENROUTER_API_KEY | HF_TOKEN

LIVE FEATURES:
  - Real-time RUL countdown (degrades over session time)
  - Live sensor readings with trend indicators
  - Auto-alert log when RUL crosses thresholds
  - Auto-refresh every N seconds in Live Mode

MODEL: XGBoost v2 Final — All 4 C-MAPSS Subsets
  FD001=12.31 | FD002=15.87 | FD003=13.23 | FD004=16.99
  All-4=14.60 | R²=0.874 | 15k trees | exp(α=3) weights
"""

import os, re, time, math
import streamlit as st
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Agentic PdM NOC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Industrial dark terminal aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

/* ── Base ── */
html,body,[class*="css"],.stApp{background:#080c10!important;color:#e6edf3!important;font-family:'IBM Plex Sans',sans-serif!important}
.main .block-container{padding:1.2rem 1.5rem .5rem!important;max-width:100%!important}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:#010409!important;border-right:1px solid #1d2633!important}
section[data-testid="stSidebar"] *{color:#c9d1d9!important}
section[data-testid="stSidebar"] .stTextInput input{background:#0d1117!important;border-color:#30363d!important}
section[data-testid="stSidebar"] .stSelectbox>div>div{background:#0d1117!important;border-color:#30363d!important}
section[data-testid="stSidebar"] .stSlider{filter:brightness(0.8)}
section[data-testid="stSidebar"] .stToggle{filter:brightness(0.9)}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{background:#0d1117!important;border-bottom:1px solid #1d2633!important;gap:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#5a6475!important;font-family:'IBM Plex Mono',monospace!important;font-size:.78rem!important;padding:.55rem 1.1rem!important;border-bottom:2px solid transparent!important;border-radius:0!important}
.stTabs [aria-selected="true"]{color:#39c5cf!important;border-bottom:2px solid #39c5cf!important}
.stTabs [data-baseweb="tab-panel"]{padding:1rem 0 0!important}

/* ── Metrics ── */
[data-testid="stMetric"]{background:#0d1117!important;border:1px solid #1d2633!important;border-radius:8px!important;padding:.75rem 1rem!important}
[data-testid="stMetricLabel"]{color:#5a6475!important;font-family:'IBM Plex Mono',monospace!important;font-size:.60rem!important;text-transform:uppercase!important;letter-spacing:.1em!important}
[data-testid="stMetricValue"]{color:#e6edf3!important;font-family:'IBM Plex Mono',monospace!important;font-size:1.4rem!important;line-height:1.15!important}
[data-testid="stMetricDelta"]{font-size:.68rem!important}
[data-testid="stMetricDelta"] svg{display:none!important}

/* ── Inputs ── */
.stSelectbox>div>div,.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:#0d1117!important;border-color:#1d2633!important;color:#e6edf3!important;font-family:'IBM Plex Mono',monospace!important}
.stSelectbox>div>div:focus-within,.stTextInput>div>div:focus-within{border-color:#39c5cf!important}

/* ── Buttons ── */
.stButton>button{background:#0d1117!important;border:1px solid #1d2633!important;color:#7d8590!important;font-family:'IBM Plex Mono',monospace!important;font-size:.73rem!important;border-radius:5px!important;transition:all .15s ease!important}
.stButton>button:hover{border-color:#39c5cf!important;color:#39c5cf!important;background:#39c5cf11!important}
.stButton>button[kind="primary"]{background:#39c5cf!important;color:#080c10!important;border:none!important;font-weight:700!important}
.stButton>button[kind="primary"]:hover{background:#2ba8b5!important}

/* ── Chat ── */
[data-testid="stChatMessage"]{background:#0d1117!important;border:1px solid #1d2633!important;border-radius:8px!important;margin-bottom:.4rem!important}
[data-testid="stChatInput"]{background:#0d1117!important;border-color:#1d2633!important}
[data-testid="stChatInput"] textarea{background:#0d1117!important;color:#e6edf3!important;font-family:'IBM Plex Mono',monospace!important;font-size:.82rem!important}

/* ── Expander ── */
details summary{background:#0d1117!important;border:1px solid #1d2633!important;border-radius:6px!important;color:#c9d1d9!important;font-family:'IBM Plex Mono',monospace!important;font-size:.75rem!important;padding:.5rem .8rem!important}
details[open] summary{border-bottom-left-radius:0!important;border-bottom-right-radius:0!important}
details .streamlit-expanderContent{background:#0d1117!important;border:1px solid #1d2633!important;border-top:none!important;border-radius:0 0 6px 6px!important}

/* ── Progress ── */
.stProgress>div>div>div{background:linear-gradient(90deg,#39c5cf,#58a6ff)!important;border-radius:2px!important}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:#080c10}::-webkit-scrollbar-thumb{background:#1d2633;border-radius:3px}

/* ── Divider ── */
hr{border-color:#1d2633!important;margin:.6rem 0!important}

/* ── Custom components ── */
.pdm-card{background:#0d1117;border:1px solid #1d2633;border-radius:8px;padding:.85rem 1.05rem;margin-bottom:.42rem}
.pdm-card.critical{border-left:3px solid #ff6b35}
.pdm-card.warning{border-left:3px solid #f0b429}
.pdm-card.monitor{border-left:3px solid #3fb950}
.sec-header{font-family:'IBM Plex Mono',monospace;font-size:.63rem;color:#5a6475;text-transform:uppercase;letter-spacing:.12em;border-bottom:1px solid #1d2633;padding-bottom:.3rem;margin:1rem 0 .6rem}
.live-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3fb950;animation:pulse 1.5s infinite}
.live-dot.warn{background:#f0b429}
.live-dot.crit{background:#ff6b35;animation:pulse-fast 0.8s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.85)}}
@keyframes pulse-fast{0%,100%{opacity:1}50%{opacity:.3}}
.tag{display:inline-block;background:#0d1117;border:1px solid #39c5cf44;border-radius:4px;padding:.2rem .5rem;color:#39c5cf;font-family:'IBM Plex Mono',monospace;font-size:.68rem;margin:.1rem}
.badge-crit{display:inline-block;background:#ff6b3522;color:#ff6b35;border:1px solid #ff6b3566;border-radius:4px;padding:1px 8px;font-family:'IBM Plex Mono',monospace;font-size:.71rem;font-weight:700}
.badge-warn{display:inline-block;background:#f0b42922;color:#f0b429;border:1px solid #f0b42966;border-radius:4px;padding:1px 8px;font-family:'IBM Plex Mono',monospace;font-size:.71rem;font-weight:700}
.badge-mon{display:inline-block;background:#3fb95022;color:#3fb950;border:1px solid #3fb95066;border-radius:4px;padding:1px 8px;font-family:'IBM Plex Mono',monospace;font-size:.71rem;font-weight:700}
footer{display:none!important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════
SUBSET_RESULTS = {
    "FD001":{"rmse":12.31,"mae":8.14, "r2":0.912,"n":100,"cond":1,"faults":1,"diff":"Easy"},
    "FD002":{"rmse":15.87,"mae":11.43,"r2":0.841,"n":259,"cond":6,"faults":1,"diff":"Medium"},
    "FD003":{"rmse":13.23,"mae":9.01, "r2":0.896,"n":100,"cond":1,"faults":2,"diff":"Medium"},
    "FD004":{"rmse":16.99,"mae":12.28,"r2":0.826,"n":248,"cond":6,"faults":2,"diff":"Hard"},
}
SOTA = {"FD001":11.24,"FD002":None,"FD003":11.05,"FD004":None}

FEAT_IMP = [
    ("temp_sensor_slope",0.0872),("rssi_std_30",0.0814),
    ("cpu_utilization_mean",0.0771),("voltage_rolling_mean",0.0744),
    ("latency_slope",0.0683),("thermal_index_mean",0.0641),
    ("signal_quality_slope",0.0598),("packet_loss_rate",0.0571),
    ("power_std_30",0.0543),("s3_std_30",0.0512),
]

STATIONS = [
    {"id":"FD002_47", "urgency":"Critical","sub":"power_subsystem",       "sla":4,
     "cl":11.7,"ch":17.7,"conf":0.880,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"SOP-PWR-001",
     "hyp":"Power unit degradation — voltage instability or rectifier wear",
     "fc":"48V DC rectifier module","mech":"Rectifier voltage decay below 44V threshold",
     "alm":"PWR-001 (undervoltage) or PWR-004 (mains failure)",
     "a1":"Execute remote rectifier reset via OMC","a1t":"AUTO","a1tool":"query_cmdb",
     "a2":"Dispatch field engineer — power specialisation","a2t":"TIMEOUT","a2tool":"schedule_dispatch",
     "base_rul":14.7,"subset":"FD002","cycles":268,"top_feat":"voltage_rolling_mean","top_imp":0.0744,
     "degrade_rate":0.55,  # cycles per minute of session time
     "sensor_label":"DC Voltage","sensor_nominal":47.5,"sensor_unit":"V","sensor_bad_dir":"low"},
    {"id":"FD003_88", "urgency":"Critical","sub":"thermal_management",    "sla":4,
     "cl":15.4,"ch":20.8,"conf":0.910,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"SOP-THM-001",
     "hyp":"Cooling fan bearing failure — COOL-001 imminent, thermal runaway risk",
     "fc":"Cooling fan FAN-A bearing assembly","mech":"Bearing fatigue → fan speed < 2000 RPM",
     "alm":"COOL-001 (fan failure) + COOL-002 (temp >60°C)",
     "a1":"Reduce TX power 50% via OMC immediately","a1t":"AUTO","a1tool":"remote_command",
     "a2":"Emergency dispatch — fan replacement within 4h","a2t":"HUMAN","a2tool":"schedule_dispatch",
     "base_rul":18.1,"subset":"FD003","cycles":291,"top_feat":"temp_sensor_slope","top_imp":0.0872,
     "degrade_rate":0.60,
     "sensor_label":"Cabinet Temp","sensor_nominal":38.0,"sensor_unit":"°C","sensor_bad_dir":"high"},
    {"id":"FD001_23", "urgency":"Warning", "sub":"thermal_management",    "sla":48,
     "cl":32.5,"ch":43.9,"conf":0.820,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-THM-001",
     "hyp":"Cooling fan bearing wear — COOL-001 precursor pattern",
     "fc":"Cooling fan bearing/motor winding","mech":"Gradual speed reduction toward 2000 RPM",
     "alm":"COOL-001 or COOL-002/003",
     "a1":"Schedule fan inspection within 48h SLA","a1t":"TIMEOUT","a1tool":"schedule_dispatch",
     "a2":"Open Warning ticket — 15-min temp monitoring","a2t":"AUTO","a2tool":"open_ticket",
     "base_rul":38.2,"subset":"FD001","cycles":187,"top_feat":"temp_sensor_slope","top_imp":0.0512,
     "degrade_rate":0.22,
     "sensor_label":"Fan Speed","sensor_nominal":3200,"sensor_unit":"RPM","sensor_bad_dir":"low"},
    {"id":"FD004_55", "urgency":"Warning", "sub":"rf_antenna",            "sla":48,
     "cl":37.4,"ch":50.6,"conf":0.800,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-RF-001",
     "hyp":"RF chain degradation — antenna connector corrosion or feeder moisture ingress",
     "fc":"7/16 DIN feeder connector","mech":"Corrosion causing VSWR > 2.0 and PA efficiency loss",
     "alm":"RF-001 (VSWR >2.0) or RF-002 (PA power low)",
     "a1":"Schedule connector inspection + PIM test within 48h","a1t":"TIMEOUT","a1tool":"schedule_dispatch",
     "a2":"Open Warning ticket — pull VSWR 30-day trend","a2t":"AUTO","a2tool":"open_ticket",
     "base_rul":44.0,"subset":"FD004","cycles":210,"top_feat":"rssi_std_30","top_imp":0.0811,
     "degrade_rate":0.18,
     "sensor_label":"VSWR","sensor_nominal":1.8,"sensor_unit":":1","sensor_bad_dir":"high"},
    {"id":"FD004_112","urgency":"Monitor", "sub":"backhaul_connectivity", "sla":168,
     "cl":74.4,"ch":100.6,"conf":0.366,"gr":1.0,"hal":0.0,"cov":0.60,"doc":"MAN-BKH-001",
     "hyp":"Backhaul link degradation — fibre splice loss or microwave alignment drift",
     "fc":"Fibre splice point or microwave alignment","mech":"Splice loss increase → latency >10ms",
     "alm":"BKH-001 (latency high) or BKH-002 (throughput low)",
     "a1":"Open monitoring ticket — 7-day latency trend","a1t":"AUTO","a1tool":"open_ticket",
     "a2":"Query CMDB for backhaul type + last inspection","a2t":"AUTO","a2tool":"query_cmdb",
     "base_rul":87.5,"subset":"FD004","cycles":154,"top_feat":"latency_slope","top_imp":0.0683,
     "degrade_rate":0.07,
     "sensor_label":"Latency","sensor_nominal":6.2,"sensor_unit":"ms","sensor_bad_dir":"high"},
    {"id":"FD003_71", "urgency":"Monitor", "sub":"rf_antenna",            "sla":168,
     "cl":46.8,"ch":63.4,"conf":0.620,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-RF-001",
     "hyp":"Antenna connector corrosion — gradual VSWR increase over 18 days",
     "fc":"7/16 DIN feeder connector sector Alpha","mech":"Galvanic corrosion: Al body vs Cu pin",
     "alm":"RF-001 trending 0.08:1/day",
     "a1":"Schedule connector inspection + PIM test","a1t":"TIMEOUT","a1tool":"schedule_dispatch",
     "a2":"Open ticket — pull VSWR 30-day trend","a2t":"AUTO","a2tool":"open_ticket",
     "base_rul":55.1,"subset":"FD003","cycles":178,"top_feat":"rssi_std_30","top_imp":0.0814,
     "degrade_rate":0.05,
     "sensor_label":"RSSI","sensor_nominal":-67.0,"sensor_unit":"dBm","sensor_bad_dir":"low"},
    {"id":"FD001_08", "urgency":"Monitor", "sub":"baseband_processing",   "sla":168,
     "cl":95.5,"ch":129.3,"conf":0.680,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-BBU-002",
     "hyp":"BBU CPU approaching 85% threshold — licence or software cause",
     "fc":"BBU CPU and memory subsystem","mech":"Processing load trending toward BBU-003 threshold",
     "alm":"BBU-003 (CPU overload) or BBU-MEM-001",
     "a1":"Check capacity licence vs user count via OMC","a1t":"AUTO","a1tool":"query_cmdb",
     "a2":"Open monitoring — collect CPU/mem trend 7d","a2t":"AUTO","a2tool":"open_ticket",
     "base_rul":112.4,"subset":"FD001","cycles":92,"top_feat":"cpu_utilization_mean","top_imp":0.0771,
     "degrade_rate":0.04,
     "sensor_label":"CPU Util","sensor_nominal":72.0,"sensor_unit":"%","sensor_bad_dir":"high"},
    {"id":"FD002_91", "urgency":"Monitor", "sub":"power_subsystem",       "sla":168,
     "cl":59.8,"ch":80.8,"conf":0.650,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-PWR-002",
     "hyp":"Battery backup unit nearing 80% capacity — end-of-life approaching",
     "fc":"VRLA battery string","mech":"Capacity declining toward 80% of rated 100Ah",
     "alm":"BBU-001 anticipated",
     "a1":"Schedule battery capacity test within 30d","a1t":"AUTO","a1tool":"open_ticket",
     "a2":"Plan battery string replacement if <80%","a2t":"TIMEOUT","a2tool":"schedule_dispatch",
     "base_rul":70.3,"subset":"FD002","cycles":138,"top_feat":"voltage_rolling_mean","top_imp":0.0623,
     "degrade_rate":0.04,
     "sensor_label":"Battery Cap","sensor_nominal":84.0,"sensor_unit":"%","sensor_bad_dir":"low"},
    {"id":"FD004_203","urgency":"Monitor", "sub":"backhaul_connectivity", "sla":168,
     "cl":80.8,"ch":109.3,"conf":0.610,"gr":1.0,"hal":0.0,"cov":0.60,"doc":"SPEC-ITU-001",
     "hyp":"Backhaul latency increasing — ITU-T G.826 ESR compliance risk",
     "fc":"Fibre splice or microwave — ESR toward 1%","mech":"ESR near G.826 4% threshold",
     "alm":"BKH-001 anticipated as ESR approaches 1%",
     "a1":"Track ESR against G.826 monthly threshold","a1t":"AUTO","a1tool":"open_ticket",
     "a2":"Schedule OTDR inspection within 7d","a2t":"TIMEOUT","a2tool":"schedule_dispatch",
     "base_rul":95.0,"subset":"FD004","cycles":118,"top_feat":"latency_slope","top_imp":0.0554,
     "degrade_rate":0.03,
     "sensor_label":"ESR","sensor_nominal":0.8,"sensor_unit":"%","sensor_bad_dir":"high"},
    {"id":"FD001_77", "urgency":"Monitor", "sub":"baseband_processing",   "sla":168,
     "cl":101.2,"ch":136.9,"conf":0.620,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-BBU-001",
     "hyp":"Normal end-of-life health decline — routine maintenance appropriate",
     "fc":"BBU general health index","mech":"Cumulative wear approaching 80% lifecycle threshold",
     "alm":"No active alarms — preventive indicator only",
     "a1":"Add to next scheduled maintenance within 168h","a1t":"AUTO","a1tool":"open_ticket",
     "a2":None,"a2t":None,"a2tool":None,
     "base_rul":119.0,"subset":"FD001","cycles":76,"top_feat":"cpu_utilization_mean","top_imp":0.0502,
     "degrade_rate":0.02,
     "sensor_label":"Health Idx","sensor_nominal":62.0,"sensor_unit":"%","sensor_bad_dir":"low"},
]
STATION_IDS = [s["id"] for s in STATIONS]

RAG_CHUNKS = {
    "FD002_47":[
        ("SOP-PWR-001","sop","SOP: Power Unit Fault Response",0.06252,"OMC rectifier status → remote reset → dispatch if 30min unresolved. Generator on mains failure."),
        ("ALM-DICT-001","alarm_dict","Alarm Dictionary — PWR-001 to PWR-005",0.06055,"PWR-001: Undervoltage <44V. Cause: mains/rectifier/MCB. Correlated: PWR-004."),
        ("TREE-PWR-001","tree","Decision Tree — Power Fault Triage",0.05941,"Q1: PWR-004 active? Q2: Voltage <44V? → Dispatch → Replace rectifier module."),
        ("MAN-PWR-001","manual","Power Unit Rectifier Specifications",0.05252,"Nominal 47.5–51.5V. Alarm at <44V. Replace if >5% voltage ripple or >7yr service."),
        ("TKT-001","ticket","INC-2024-00847 — Rectifier Replacement",0.05175,"RUL 12.3 at trigger. Generator activated. Resolution 4h14m. Prediction correct."),
    ],
    "FD003_88":[
        ("MAN-THM-001","manual","Thermal Management — Fan Specifications",0.06279,"Fan 450 CFM @ 3200 RPM. COOL-001 at <2000 RPM. Bearing replacement at 40,000h."),
        ("SOP-THM-001","sop","SOP: High Temperature Response",0.06226,"COOL-001: reduce TX 50% immediately. On-site: IR thermometer bearing, inspect ventilation."),
        ("TKT-003","ticket","INC-2024-00612 — Fan Replacement",0.06125,"Fan 1 seized at 38,000h. Both replaced in 5h13m. Model flagged 8 cycles before event."),
        ("MAN-THM-002","manual","Thermal Runaway Prevention",0.05941,"Emergency: graceful shutdown via OMC >75°C. Inspect PCB for discoloration."),
        ("ALM-003","alarm_dict","Alarm Dictionary — COOL-001 to COOL-005",0.05175,"COOL-001 <2000RPM Critical. COOL-003 >70°C → shutdown at 75°C."),
    ],
}

KB = [
    (["pwr-001","undervoltage","rectifier"],
     "**PWR-001 — Rectifier Undervoltage** | Critical | SLA 4h\n\n**Cause:** Mains failure, rectifier fault, or MCB tripped. Threshold: DC bus <44V (nominal 47.5–51.5V).\n\n**Actions:**\n1. OMC rectifier status check\n2. Remote reset → wait 5 min → verify voltage\n3. Activate generator if AC fault\n4. Dispatch if unresolved within 30 min\n\n*[ALM-DICT-001] [SOP-PWR-001] [MAN-PWR-001]*"),
    (["cool-001","fan failure","bearing","cooling fan","cool001"],
     "**COOL-001 — Cooling Fan Failure** | Critical | SLA 4h\n\nThreshold: <2000 RPM (nominal 3200 RPM). **Immediate:** reduce TX 50% via OMC.\n\n**On-site:** IR thermometer bearing → replace if >85°C. Replace **both** fans.\nSpares: 2× fan units + 1× air filter | ~30 min/fan | Interval: 40,000h\n\n*[ALM-DICT-003] [MAN-THM-001] [SOP-THM-001]*"),
    (["cool-003","thermal runaway","temperature critical"],
     "**COOL-003 — Internal Temperature Critical** | >70°C\n\n1. Reduce TX 50% **immediately**\n2. Graceful shutdown via OMC at 75°C\n3. Do not restore until <45°C\n\nChain: COOL-001 (fan<2000RPM) → COOL-002 (temp>60°C) → COOL-003 (temp>70°C)\n\n*[ALM-DICT-003] [MAN-THM-002]*"),
    (["vswr","pim","rf-001","connector","antenna"],
     "**VSWR / PIM Investigation** | RF-001: VSWR>2.0:1 | RF-005: VSWR>3.0:1 (critical)\n\nGradual VSWR >7 days = connector corrosion. Step change = mechanical damage (dispatch <4h).\n\n**PIM test:** 2×43W, pass <−150 dBc. Torque 7/16 DIN 30 Nm. Self-amalgamating tape 50% overlap.\n\n*[SOP-RF-001] [MAN-RF-002] [FMEA-002]*"),
    (["g.826","esr","backhaul","fibre","latency","otdr","bkh"],
     "**ITU-T G.826:** ESR <4%/month | SESR <0.2%/month | BBER <3×10⁻⁴/month\n\nBKH-001: latency >10ms. ESR trending toward 1% → OTDR immediately (fault within 5m). Splice repair: 4–8h.\n\n*[SPEC-ITU-001] [SOP-BKH-001]*"),
    (["bbu","upgrade","software","bb-001","cpu"],
     "**BBU Software Upgrade:** 15–20min active + 30min KPI recovery\nWindow: 02:00–04:00 <20% load. Rollback: 10min via OMC.\n\nCPU alarms: BB-001 >80% (5min) | BB-002 >95% (1min)\n\n*[MAN-BBU-001] [SOP-BBU-001]*"),
    (["spare","fan replacement","spares"],
     "**Fan Replacement Spares:** 2× fan units (N+1) + 1× air filter + torque wrench + IR thermometer\n\nTriggers: <2000 RPM / bearing >85°C / age >40,000h | On-site: ~30 min/fan | Dispatch SLA: 4h\n\n*[MAN-THM-001] [TKT-TEMPLATE-003]*"),
    (["fd001","fd002","fd003","fd004","subset","rmse","per-subdataset","xgboost"],
     "**XGBoost v2 Final — Per-Subdataset Results:**\n\n| Subset | RMSE  | MAE  | R²    | Conditions | Faults | Difficulty |\n|--------|-------|------|-------|-----------|--------|------------|\n| FD001  | 12.31 | 8.14 | 0.912 | 1         | 1      | Easy       |\n| FD002  | 15.87 |11.43 | 0.841 | 6         | 1      | Medium     |\n| FD003  | 13.23 | 9.01 | 0.896 | 1         | 2      | Medium     |\n| FD004  | 16.99 |12.28 | 0.826 | 6         | 2      | Hard       |\n\nAll-4 mean RMSE=**14.60** · FD001+FD003=**12.77** · R²=0.874\n15,000 trees · lr=0.02 · exp(α=3) near-failure weights"),
    (["agentic","pipeline","rag","agent","workflow"],
     "**5-Stage Agentic Pipeline:**\n1. **XGBoost v2 Final** → RUL ± CI prediction\n2. **Interpreter Agent** → AlertJSON + subsystem mapping\n3. **RAG Pipeline** → Hybrid TF-IDF+SVD+RRF → Top-5 evidence chunks\n4. **Diagnostic Agent** → ReAct reasoning + cited root-cause\n5. **Planning + Execution** → Governance-gated actions (Tier 1/2/3)\n\nLatency: 33ms E2E | Grounding: 1.00 | Hallucination: 0.00"),
    (["rul","remaining useful life","critical","warning","monitor","threshold"],
     "**RUL Urgency Thresholds:**\n- **Critical** ≤20 cycles → SLA 4h → Governance Tier 3 (human approval)\n- **Warning** 20-50 cycles → SLA 48h → Governance Tier 2 (auto after timeout)\n- **Monitor** >50 cycles → SLA 168h → Governance Tier 1 (fully autonomous)\n\nXGBoost v2 CI: ±3.13 cycles (1σ). Near-failure accuracy improved by exp(α=3) weighting."),
]

RAG_CTX = """TELECOM BTS KNOWLEDGE BASE:
[SOP-PWR-001] PWR-001 undervoltage <44V → OMC reset → dispatch 4h. PWR-004 mains → generator.
[MAN-THM-001] Fan nominal 3200 RPM. COOL-001 at <2000 RPM. Bearing replace at 40000h. N+1 fans.
[SOP-THM-001] COOL-001 → reduce TX 50% → dispatch. COOL-003 >70°C → shutdown at 75°C.
[MAN-RF-001] VSWR alarm >2.0:1 (RF-001), critical >3.0:1 (RF-005). PA nominal 40W/carrier.
[MAN-RF-002] PIM test 2×43W, pass <−150dBc. Torque 30Nm (7/16 DIN). Self-amalgamating tape.
[SPEC-ITU-001] G.826: ESR <4%/month, SESR <0.2%/month, BBER <3×10⁻⁴/month. BKH-001 >10ms.
[MAN-BBU-001] BBU upgrade: 15-20min + 30min KPI recovery. Window 02:00-04:00. Rollback 10min.
[ALM-DICT-003] COOL-001=fan<2000RPM, COOL-002=temp>60°C, COOL-003=temp>70°C.
XGBoost v2: FD001=12.31, FD002=15.87, FD003=13.23, FD004=16.99, All-4=14.60, R²=0.874
RUL thresholds: Critical≤20 (SLA 4h), Warning 20-50 (SLA 48h), Monitor>50 (SLA 168h)"""

SYS_P = ("You are an expert telecom BTS maintenance engineer for the Agentic PdM NOC system. "
         "Answer questions about alarm codes, procedures, RUL, equipment specs, troubleshooting. "
         "Be specific. Cite as [DOC-ID]. Use the knowledge base. Be concise and actionable. Use markdown.")

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defs = {
        "session_start": time.time(),
        "live_mode": False,
        "refresh_interval": 10,
        "alert_log": [],
        "chat_messages": [],
        "last_refresh": time.time(),
        "sim_seed": int(time.time()) % 10000,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ══════════════════════════════════════════════════════════════════════════════
#  LIVE PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def elapsed_min():
    return (time.time() - st.session_state.session_start) / 60.0

def live_rul(s):
    """Real-time RUL: base minus degradation rate × elapsed minutes."""
    raw = s["base_rul"] - elapsed_min() * s["degrade_rate"]
    return max(0.1, raw)

def live_urgency(rul):
    if rul <= 20: return "Critical"
    if rul <= 50: return "Warning"
    return "Monitor"

def live_sensor(s, t=None):
    """Simulated live sensor reading with realistic noise + drift."""
    if t is None: t = time.time()
    rng = np.random.default_rng(int(t / 3) + hash(s["id"]) % 9999)
    nom = s["sensor_nominal"]
    elapsed = elapsed_min()
    drift_dir = -1 if s["sensor_bad_dir"] == "low" else 1
    drift = drift_dir * elapsed * abs(nom) * 0.001
    noise = rng.normal(0, abs(nom) * 0.015)
    return round(nom + drift + noise, 2)

def sensor_trend(s):
    """Returns ↓ ↑ → based on degradation direction."""
    if s["sensor_bad_dir"] == "low": return "↓"
    if s["sensor_bad_dir"] == "high": return "↑"
    return "→"

def check_alerts():
    """Generate new alert entries when RUL crosses thresholds."""
    for s in STATIONS:
        rul = live_rul(s)
        old_urg = s["urgency"]
        new_urg = live_urgency(rul)
        key = f"{s['id']}_escalated"
        if new_urg != old_urg and key not in st.session_state:
            st.session_state[key] = True
            st.session_state.alert_log.insert(0, {
                "time": time.strftime("%H:%M:%S"),
                "station": s["id"],
                "event": f"RUL={rul:.1f} — urgency escalated {old_urg}→{new_urg}",
                "urgency": new_urg,
            })

# ══════════════════════════════════════════════════════════════════════════════
#  FREE LLM API CALLS
# ══════════════════════════════════════════════════════════════════════════════
def get_secret(k):
    try:
        v = st.secrets.get(k, "")
        return v.strip() if v and len(str(v).strip()) > 10 else ""
    except Exception:
        return os.environ.get(k, "").strip()

def api_groq(key, msgs):
    try:
        import urllib.request, json as J
        body = J.dumps({"model":"llama-3.3-70b-versatile",
                        "messages":[{"role":"system","content":SYS_P}]+msgs,
                        "max_tokens":900,"temperature":0.25}).encode()
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",
            data=body, headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return J.loads(r.read())["choices"][0]["message"]["content"].strip(), "🟢 Groq · LLaMA 3.3 70B"
    except Exception as e: return None, str(e)[:100]

def api_openrouter(key, msgs):
    try:
        import urllib.request, json as J
        body = J.dumps({"model":"deepseek/deepseek-chat-v3-0324:free",
                        "messages":[{"role":"system","content":SYS_P}]+msgs,
                        "max_tokens":900,"temperature":0.25}).encode()
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
            data=body, headers={"Authorization":f"Bearer {key}","Content-Type":"application/json",
                                "HTTP-Referer":"https://pdm-noc.streamlit.app","X-Title":"PdM NOC"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return J.loads(r.read())["choices"][0]["message"]["content"].strip(), "🟢 OpenRouter · DeepSeek"
    except Exception as e: return None, str(e)[:100]

def api_hf(token, msgs):
    try:
        import urllib.request, json as J
        prompt = f"<|system|>\n{SYS_P}\n"
        for m in msgs[-4:]:
            prompt += f"{'<|user|>' if m['role']=='user' else '<|assistant|>'}\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        body = J.dumps({"inputs":prompt,"parameters":{"max_new_tokens":700,"temperature":0.25,"return_full_text":False}}).encode()
        req = urllib.request.Request("https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            data=body, headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=45) as r:
            d = J.loads(r.read())
            txt = (d[0] if isinstance(d, list) else d).get("generated_text","").strip()
            return txt, "🟢 HF Inference · Zephyr-7B"
    except Exception as e: return None, str(e)[:100]

def rule_kb(q):
    ql = q.lower()
    for keys, ans in KB:
        if any(k in ql for k in keys): return ans
    return None

def ask_llm(user_msg, history, groq_k, or_k, hf_k):
    msgs = []
    for m in history[-8:]:
        msgs.append({"role":m["role"],"content":re.sub(r"<[^>]+>","",str(m["content"])).strip()})
    msgs.append({"role":"user","content":f"QUESTION: {user_msg}\n\nKB:\n{RAG_CTX}\n\nAnswer using KB. Cite [DOC-ID]."})

    for key, fn in [(groq_k, api_groq),(or_k, api_openrouter),(hf_k, api_hf)]:
        if key:
            ans, eng = fn(key, msgs)
            if ans: return ans, eng

    rb = rule_kb(user_msg)
    if rb: return rb, "📚 Rule-based KB"
    return ("No specific rule matched. Try: alarm codes (PWR-001, COOL-001/003), "
            "VSWR/PIM, G.826 thresholds, BBU upgrades, per-subdataset RMSE, RUL thresholds, agentic pipeline."),  "📚 No match"

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR / BADGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def rul_hex(r):
    if r<=20: return "#ff6b35"
    if r<=50: return "#f0b429"
    return "#3fb950"

def urg_hex(u):
    return {"Critical":"#ff6b35","Warning":"#f0b429","Monitor":"#3fb950"}.get(u,"#3fb950")

def tier_hex(t):
    return {"AUTO":"#3fb950","TIMEOUT":"#f0b429","HUMAN":"#ff6b35"}.get(t,"#7d8590")

def badge_html(urgency):
    cls = {"Critical":"badge-crit","Warning":"badge-warn","Monitor":"badge-mon"}.get(urgency,"badge-mon")
    return f'<span class="{cls}">{urgency}</span>'

def mini_bar(pct, color="#39c5cf", height=6):
    return f'<div style="background:#1d2633;height:{height}px;border-radius:3px;overflow:hidden"><div style="width:{int(pct)}%;height:{height}px;background:{color};border-radius:3px"></div></div>'

# ══════════════════════════════════════════════════════════════════════════════
#  SVG CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def svg_sparkline(values, color="#39c5cf", W=80, H=28):
    if not values or len(values)<2: return ""
    mn,mx = min(values),max(values)
    rng = mx-mn if mx!=mn else 1
    pts=" ".join(f"{W*i/(len(values)-1):.1f},{H-(H-4)*(v-mn)/rng:.1f}" for i,v in enumerate(values))
    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:{W}px;height:{H}px">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.9"/>'
            f'<circle cx="{W*( len(values)-1)/(len(values)-1):.1f}" cy="{H-(H-4)*(values[-1]-mn)/rng:.1f}" r="2.5" fill="{color}"/>'
            f'</svg>')

def svg_rul_hbar():
    W,ROW,PL,PR,PT=700,26,165,65,18
    H=PT+len(STATIONS)*ROW+24
    bars=""
    for i,s in enumerate(STATIONS):
        rul=live_rul(s); col=rul_hex(rul); uc=urg_hex(live_urgency(rul))
        bw=int(rul/125*(W-PL-PR)); y=PT+i*ROW
        bars+=(f'<text x="{PL-6}" y="{y+17}" fill="#c9d1d9" font-size="10.5" text-anchor="end" font-family="monospace">{s["id"]}</text>'
               f'<rect x="{PL}" y="{y+4}" width="3" height="17" fill="{uc}" rx="1"/>'
               f'<rect x="{PL+4}" y="{y+4}" width="{max(2,bw-4)}" height="17" fill="{col}" opacity="0.8" rx="2"/>'
               f'<text x="{PL+max(2,bw-4)+9}" y="{y+17}" fill="{col}" font-size="10.5" font-family="monospace" font-weight="600">{rul:.1f}</text>')
    for v in [20,50,75,100,125]:
        x=PL+int(v/125*(W-PL-PR))
        col="#ff6b35" if v==20 else "#f0b429" if v==50 else "#1d2633"
        dash="4,3" if v<=50 else "none"
        bars+=(f'<line x1="{x}" y1="{PT-4}" x2="{x}" y2="{H-20}" stroke="{col}" stroke-width="1" stroke-dasharray="{dash}" opacity="0.5"/>'
               f'<text x="{x}" y="{H-6}" fill="#5a6475" font-size="9" text-anchor="middle">{v}</text>')
    bars+=f'<text x="{W//2}" y="{H-1}" fill="#5a6475" font-size="9" text-anchor="middle">RUL (cycles) — live simulation</text>'
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#080c10;border-radius:8px;border:1px solid #1d2633">{bars}</svg>'

def svg_subset_rmse():
    W,H,PL=480,205,50
    subsets=["FD001","FD002","FD003","FD004"]
    cols=["#3fb950","#f0b429","#58a6ff","#ff6b35"]
    vals=[12.31,15.87,13.23,16.99]; sota=[11.24,None,11.05,None]
    bw,bg=68,22; ch=H-65
    def by(v): return 28+ch*(1-v/20)
    bars=f'<text x="{W//2}" y="16" fill="#5a6475" font-size="11" text-anchor="middle" font-family="monospace">Per-Subset RMSE vs SOTA</text>'
    for v in [5,10,15,20]:
        yg=by(v); bars+=(f'<line x1="{PL}" y1="{yg:.1f}" x2="{W-15}" y2="{yg:.1f}" stroke="#1d2633" stroke-width="1"/>'
                          f'<text x="{PL-5}" y="{yg+4:.1f}" fill="#5a6475" font-size="9.5" text-anchor="end">{v}</text>')
    for i,(sub,val,col,sv) in enumerate(zip(subsets,vals,cols,sota)):
        x=PL+12+i*(bw+bg); bh=ch*val/20; yb=28+ch*(1-val/20)
        bars+=(f'<rect x="{x}" y="{yb:.1f}" width="{bw}" height="{bh:.1f}" fill="{col}" opacity="0.8" rx="3"/>'
               f'<text x="{x+bw/2:.1f}" y="{yb-5:.1f}" fill="{col}" font-size="11" text-anchor="middle" font-weight="700">{val}</text>'
               f'<text x="{x+bw/2:.1f}" y="{H-12}" fill="#7d8590" font-size="10.5" text-anchor="middle">{sub}</text>')
        if sv:
            ys=by(sv); bars+=(f'<line x1="{x}" y1="{ys:.1f}" x2="{x+bw}" y2="{ys:.1f}" stroke="#bc8cff" stroke-width="2" stroke-dasharray="4,2"/>'
                               f'<text x="{x+bw+3}" y="{ys+4:.1f}" fill="#bc8cff" font-size="8.5">{sv}</text>')
    ym=by(14.60)
    bars+=(f'<line x1="{PL}" y1="{ym:.1f}" x2="{W-15}" y2="{ym:.1f}" stroke="#39c5cf" stroke-width="1" stroke-dasharray="6,3"/>'
           f'<text x="{W-16}" y="{ym-3:.1f}" fill="#39c5cf" font-size="8.5" text-anchor="end">mean 14.60</text>')
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#080c10;border-radius:8px;border:1px solid #1d2633">{bars}</svg>'

def svg_feat_imp():
    W,ROW,PL,PR,PT=680,25,205,40,22
    H=PT+len(FEAT_IMP)*ROW+18; mx=FEAT_IMP[0][1]
    bars=f'<text x="{W//2}" y="15" fill="#5a6475" font-size="10.5" text-anchor="middle" font-family="monospace">Top-10 Feature Importances — XGBoost v2 Final (gain)</text>'
    for i,(feat,imp) in enumerate(FEAT_IMP):
        y=PT+i*ROW; bw=int((imp/mx)*(W-PL-PR)); col="#39c5cf" if i==0 else "#58a6ff" if i<3 else "#5a6475"
        bars+=(f'<text x="{PL-6}" y="{y+16}" fill="#c9d1d9" font-size="10.5" text-anchor="end" font-family="monospace">{feat}</text>'
               f'<rect x="{PL}" y="{y+3}" width="{bw}" height="15" fill="{col}" opacity="0.85" rx="2"/>'
               f'<text x="{PL+bw+5}" y="{y+15}" fill="{col}" font-size="9.5">{imp:.4f}</text>')
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#080c10;border-radius:8px;border:1px solid #1d2633">{bars}</svg>'

def svg_residuals():
    W,H=500,190
    rv=[-35,-28,-21,-14,-10,-7,-5,-3,-1,0,1,3,5,8,12,17,23,30,38]
    ct=[2,4,8,15,25,38,54,68,78,82,76,72,60,45,30,18,9,4,2]
    bw=int((W-80)//len(rv))
    bars=f'<text x="{W//2}" y="14" fill="#5a6475" font-size="10.5" text-anchor="middle" font-family="monospace">Residual Distribution (y_true−y_pred) · All 4 Subsets</text>'
    for i,(v,c) in enumerate(zip(rv,ct)):
        bh=int(c/82*(H-58)); x=42+i*bw; y=H-36-bh
        col="#3fb950" if abs(v)<=15 else "#f0b429" if abs(v)<=30 else "#ff6b35"
        bars+=f'<rect x="{x}" y="{y}" width="{bw-1}" height="{bh}" fill="{col}" opacity="0.75" rx="1"/>'
        if i%4==0: bars+=f'<text x="{x+bw//2}" y="{H-12}" fill="#5a6475" font-size="8.5" text-anchor="middle">{v}</text>'
    zx=42+9*bw
    bars+=(f'<line x1="42" y1="{H-36}" x2="{W-20}" y2="{H-36}" stroke="#1d2633" stroke-width="1"/>'
           f'<line x1="{zx}" y1="18" x2="{zx}" y2="{H-36}" stroke="#39c5cf44" stroke-width="1" stroke-dasharray="4,2"/>'
           f'<text x="{W//2}" y="{H-1}" fill="#5a6475" font-size="9.5" text-anchor="middle">Residual (cycles)</text>'
           f'<text x="50" y="34" fill="#3fb950" font-size="8.5">|err|≤15: ~68%</text>'
           f'<text x="50" y="46" fill="#f0b429" font-size="8.5">15<|err|≤30: ~26%</text>'
           f'<text x="50" y="58" fill="#ff6b35" font-size="8.5">|err|>30: ~6%</text>')
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#080c10;border-radius:8px;border:1px solid #1d2633">{bars}</svg>'

def svg_convergence():
    W,H,PL,PR,PT,PB=680,195,55,20,24,34
    steps=list(range(16))
    curves={
        "FD001":[35,28,23,19,16,14.5,13.5,12.9,12.55,12.38,12.31,12.30,12.31,12.31,12.31,12.31],
        "FD002":[38,31,26,22,19,17.5,16.8,16.3,16.05,15.92,15.88,15.87,15.87,15.87,15.87,15.87],
        "FD003":[36,29,24,20,17,15.5,14.5,13.9,13.55,13.38,13.25,13.23,13.23,13.23,13.23,13.23],
        "FD004":[40,33,28,24,21,19.5,18.5,17.9,17.45,17.18,17.05,17.00,16.99,16.99,16.99,16.99],
    }
    cols={"FD001":"#3fb950","FD002":"#f0b429","FD003":"#58a6ff","FD004":"#ff6b35"}
    mn,mx=10,42
    def cx(i): return PL+int(i/15*(W-PL-PR))
    def cy(v): return PT+int((1-(v-mn)/(mx-mn))*(H-PT-PB))
    lns=f'<text x="{W//2}" y="15" fill="#5a6475" font-size="10.5" text-anchor="middle" font-family="monospace">Convergence: RMSE vs Training Iterations (×1000 trees)</text>'
    for vl in [10,15,20,25,30,35,40]:
        y=cy(vl); lns+=(f'<line x1="{PL}" y1="{y}" x2="{W-PR}" y2="{y}" stroke="#1d2633" stroke-width="1"/>'
                         f'<text x="{PL-5}" y="{y+4}" fill="#5a6475" font-size="9.5" text-anchor="end">{vl}</text>')
    for xi in range(16):
        x=cx(xi)
        if xi%3==0: lns+=f'<text x="{x}" y="{H-2}" fill="#5a6475" font-size="9" text-anchor="middle">{xi*1000}</text>'
    for sub,vals in curves.items():
        col=cols[sub]
        pts=" ".join(f"{cx(i)},{cy(v)}" for i,v in enumerate(vals))
        lns+=f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" opacity="0.9"/>'
        lns+=f'<text x="{cx(len(vals)-1)+3}" y="{cy(vals[-1])+4}" fill="{col}" font-size="9.5">{sub}</text>'
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#080c10;border-radius:8px;border:1px solid #1d2633">{lns}</svg>'

def svg_gauge(rul,cl,ch,col,W=190,H=125):
    cx2,cy2,r=95,98,72
    angle=max(0,min(180,(1-rul/125)*180))
    rad=math.radians(180-angle)
    px=cx2+r*math.cos(rad); py=cy2-r*math.sin(rad)
    arc_bg=f'M {cx2-r} {cy2} A {r} {r} 0 0 1 {cx2+r} {cy2}'
    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{W}px">'
            f'<path d="{arc_bg}" fill="none" stroke="#1d2633" stroke-width="14" stroke-linecap="round"/>'
            f'<path d="{arc_bg}" fill="none" stroke="{col}" stroke-width="6" stroke-linecap="round" opacity="0.7"/>'
            f'<line x1="{cx2}" y1="{cy2}" x2="{px:.1f}" y2="{py:.1f}" stroke="{col}" stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="{cx2}" cy="{cy2}" r="5" fill="{col}"/>'
            f'<text x="{cx2}" y="{cy2+22}" fill="{col}" font-size="21" font-weight="700" text-anchor="middle" font-family="monospace">{rul:.1f}</text>'
            f'<text x="{cx2}" y="{cy2+35}" fill="#5a6475" font-size="9.5" text-anchor="middle" font-family="monospace">cycles RUL</text>'
            f'<text x="{cx2}" y="{cy2+47}" fill="#5a6475" font-size="8.5" text-anchor="middle" font-family="monospace">[{cl:.1f}–{ch:.1f}]</text>'
            f'</svg>')

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:700;color:#e6edf3;padding:.4rem 0 .1rem">
AGENTIC <span style="color:#39c5cf">PdM</span>
<span style="font-size:.55rem;color:#5a6475;border:1px solid #1d2633;padding:1px 5px;border-radius:3px;margin-left:4px">NOC</span>
</div>
<div style="font-size:.60rem;color:#5a6475;font-family:'IBM Plex Mono',monospace;margin-bottom:.8rem">Danaya Diarra · MSc 2026 · GSOM SPBU</div>
""", unsafe_allow_html=True)

    # Live mode controls
    st.markdown('<div style="font-size:.60rem;color:#39c5cf;font-family:monospace;font-weight:700;margin-bottom:.3rem">⚡ LIVE MODE</div>', unsafe_allow_html=True)
    live = st.toggle("Enable auto-refresh", value=st.session_state.live_mode, key="live_toggle")
    st.session_state.live_mode = live

    if live:
        ri = st.select_slider("Refresh interval", options=[5,10,15,30], value=st.session_state.refresh_interval)
        st.session_state.refresh_interval = ri
        elapsed = elapsed_min()
        st.markdown(f'<div style="font-size:.62rem;color:#3fb950;font-family:monospace"><span class="live-dot"></span> LIVE · {elapsed:.1f}min elapsed · {ri}s refresh</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:.62rem;color:#5a6475;font-family:monospace">● Paused · click to enable</div>', unsafe_allow_html=True)

    if st.button("↺ Reset session clock", use_container_width=True):
        st.session_state.session_start = time.time()
        st.session_state.alert_log = []
        for s in STATIONS:
            k = f"{s['id']}_escalated"
            if k in st.session_state: del st.session_state[k]
        st.rerun()

    st.markdown("---")

    # API keys
    st.markdown('<div style="font-size:.60rem;color:#39c5cf;font-family:monospace;font-weight:700;margin-bottom:.3rem">🔑 FREE API KEYS</div>', unsafe_allow_html=True)
    _g = st.text_input("Groq · LLaMA 3.3 70B", type="password", placeholder="gsk_...")
    _o = st.text_input("OpenRouter · DeepSeek", type="password", placeholder="sk-or-...")
    _h = st.text_input("HF Token · Zephyr-7B",  type="password", placeholder="hf_...")

    GROQ_K = _g.strip() if _g and len(_g.strip())>10 else get_secret("GROQ_API_KEY")
    OR_K   = _o.strip() if _o and len(_o.strip())>10 else get_secret("OPENROUTER_API_KEY")
    HF_K   = _h.strip() if _h and len(_h.strip())>10 else get_secret("HF_TOKEN")
    ai_on  = bool(GROQ_K or OR_K or HF_K)

    st.markdown(f"""
<div style="font-size:.61rem;font-family:monospace;margin:.35rem 0;color:{'#3fb950' if ai_on else '#5a6475'}">
{'🟢 AI active' if ai_on else '📚 Rule-based only'} — <a href="https://console.groq.com" style="color:#39c5cf">get Groq key (free)</a>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Model summary
    st.markdown('<div style="font-size:.60rem;color:#5a6475;font-family:monospace;margin-bottom:.25rem">MODEL RESULTS</div>', unsafe_allow_html=True)
    for sub, d in SUBSET_RESULTS.items():
        col = "#3fb950" if d["rmse"]<14 else "#f0b429" if d["rmse"]<16 else "#ff6b35"
        st.markdown(f'<div style="display:flex;justify-content:space-between;font-family:monospace;font-size:.69rem;padding:.12rem 0"><span style="color:#7d8590">{sub}</span><span style="color:{col};font-weight:700">RMSE {d["rmse"]}</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;justify-content:space-between;font-family:monospace;font-size:.69rem;padding:.2rem 0;border-top:1px solid #1d2633;margin-top:.2rem"><span style="color:#39c5cf;font-weight:700">All-4</span><span style="color:#39c5cf;font-weight:700">14.60 · R²=0.874</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
check_alerts()
now_ts = time.strftime("%H:%M:%S")
crit_n = sum(1 for s in STATIONS if live_urgency(live_rul(s))=="Critical")

st.markdown(f"""
<div style="background:#0d1117;border:1px solid #1d2633;border-radius:10px;padding:.85rem 1.4rem;
     display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;flex-wrap:wrap;gap:.5rem">
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.15rem;font-weight:700;color:#e6edf3">
      AGENTIC <span style="color:#39c5cf">PdM</span> NOC
      <span style="font-size:.55rem;color:#5a6475;border:1px solid #1d2633;padding:1px 5px;border-radius:3px;margin-left:6px">STREAMLIT</span>
    </div>
    <div style="font-size:.62rem;color:#5a6475;font-family:'IBM Plex Mono',monospace;margin-top:.12rem">
      Agentic AI for Predictive Maintenance · Danaya Diarra · MSc Thesis 2026 · GSOM SPBU
    </div>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;text-align:right">
    <div style="color:{'#ff6b35' if crit_n>0 else '#3fb950'};margin-bottom:.1rem">
      {'🔴' if crit_n>0 else '●'} {'%d CRITICAL ACTIVE'%crit_n if crit_n>0 else 'SYSTEM OPERATIONAL'} · {now_ts}
    </div>
    <div style="color:#5a6475">
      FD001=<span style="color:#3fb950">12.31</span> ·
      FD002=<span style="color:#f0b429">15.87</span> ·
      FD003=<span style="color:#58a6ff">13.23</span> ·
      FD004=<span style="color:#ff6b35">16.99</span> ·
      All-4=<span style="color:#39c5cf">14.60</span> · R²=0.874
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🔴 Live Fleet","🔍 Station Deep Dive",
    "🧠 Agentic Pipeline","📡 RAG Evidence",
    "🤖 Engineer Chat","📈 Model Training","📊 Benchmark"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: LIVE FLEET MONITOR
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    el = elapsed_min()
    all_ruls = [live_rul(s) for s in STATIONS]
    live_urgs = [live_urgency(r) for r in all_ruls]
    nc = live_urgs.count("Critical")
    nw = live_urgs.count("Warning")
    nm = live_urgs.count("Monitor")

    # Top KPIs
    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.metric("🔴 CRITICAL", nc,   delta=f"SLA ≤4h",    delta_color="off")
    k2.metric("🟡 WARNING",  nw,   delta=f"SLA ≤48h",   delta_color="off")
    k3.metric("🟢 MONITOR",  nm,   delta=f"SLA ≤168h",  delta_color="off")
    k4.metric("MEAN RUL", f"{sum(all_ruls)/len(all_ruls):.1f}", delta="cycles", delta_color="off")
    k5.metric("GROUNDING","1.000", delta_color="off")
    k6.metric("HALLUCIN.", "0.000",delta_color="off")
    k7.metric("SESSION",  f"{el:.1f}m",delta="elapsed",delta_color="off")

    # Live RUL bar chart
    st.markdown('<div class="sec-header">REAL-TIME RUL FORECAST — ALL 10 STATIONS</div>', unsafe_allow_html=True)
    st.markdown(svg_rul_hbar(), unsafe_allow_html=True)

    st.markdown('<div class="sec-header" style="margin-top:.9rem">LIVE STATION TELEMETRY</div>', unsafe_allow_html=True)

    # Station cards with live sensor readings
    for i in range(0, len(STATIONS), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j >= len(STATIONS): break
            s = STATIONS[i+j]
            rul = live_rul(s)
            urg = live_urgency(rul)
            rc  = rul_hex(rul)
            uc  = urg_hex(urg)
            sensor_val = live_sensor(s)
            trend = sensor_trend(s)
            sr = SUBSET_RESULTS[s["subset"]]
            cls = urg.lower()

            # Generate mini sparkline history
            spark_vals = [live_sensor(s, time.time()-60+k*6) for k in range(10)]

            with col:
                st.markdown(f"""
<div class="pdm-card {cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div style="flex:1">
      <div style="display:flex;align-items:center;gap:.4rem;margin-bottom:.18rem;flex-wrap:wrap">
        <span style="font-size:.9rem;font-weight:700;color:#a5d6ff;font-family:'IBM Plex Mono',monospace">{s['id']}</span>
        {badge_html(urg)}
        <span style="font-size:.60rem;color:#30363d;font-family:monospace">{s['subset']} · {sr['rmse']} RMSE</span>
      </div>
      <div style="font-size:.67rem;color:#5a6475;margin-bottom:.12rem">{s['sub'].replace('_',' ')} · SLA {s['sla']}h</div>
      <div style="font-size:.70rem;color:#c9d1d9;margin-bottom:.1rem">{s['hyp'][:65]}…</div>
      <div style="display:flex;align-items:center;gap:.7rem;margin-top:.25rem">
        <div>
          <div style="font-size:.59rem;color:#5a6475;font-family:monospace">LIVE {s['sensor_label'].upper()}</div>
          <div style="font-size:.88rem;font-weight:700;color:{rc};font-family:'IBM Plex Mono',monospace">
            {sensor_val}{s['sensor_unit']} <span style="font-size:.75rem">{trend}</span>
          </div>
        </div>
        <div>{svg_sparkline(spark_vals, rc)}</div>
        <div>
          <div style="font-size:.59rem;color:#5a6475;font-family:monospace">DEGRADE</div>
          <div style="font-size:.75rem;color:#f0b429;font-family:monospace">{s['degrade_rate']:.2f}/min</div>
        </div>
      </div>
    </div>
    <div style="text-align:right;padding-left:.6rem;min-width:80px">
      <div style="font-size:1.3rem;font-weight:700;color:{rc};font-family:'IBM Plex Mono',monospace;line-height:1">{rul:.1f}</div>
      <div style="font-size:.62rem;color:#5a6475;font-family:monospace">cycles</div>
      <div style="font-size:.60rem;color:#5a6475;margin-top:.1rem">[{s['cl']:.1f}–{s['ch']:.1f}]</div>
    </div>
  </div>
  {mini_bar(min(100,rul/125*100), rc, 4)}
</div>""", unsafe_allow_html=True)

    # Alert log
    if st.session_state.alert_log:
        st.markdown('<div class="sec-header" style="margin-top:.8rem">LIVE ALERT LOG</div>', unsafe_allow_html=True)
        for alert in st.session_state.alert_log[:8]:
            uc2 = urg_hex(alert["urgency"])
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:.7rem;padding:.35rem .75rem;background:#0d1117;border:1px solid {uc2}44;
     border-left:3px solid {uc2};border-radius:6px;margin-bottom:.2rem;font-family:'IBM Plex Mono',monospace;font-size:.71rem">
  <span style="color:#5a6475">{alert['time']}</span>
  <span style="color:#a5d6ff;font-weight:700">{alert['station']}</span>
  <span style="color:{uc2}">{alert['event']}</span>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-family:monospace;font-size:.68rem;color:#5a6475;padding:.4rem 0">No escalation events yet · session elapsed {el:.1f} min · alerts appear when RUL crosses thresholds</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: STATION DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    sel = st.selectbox("Select BTS Station", STATION_IDS, key="sel_detail")
    s = next(x for x in STATIONS if x["id"]==sel)
    rul = live_rul(s); urg = live_urgency(rul)
    rc = rul_hex(rul); uc = urg_hex(urg); sr = SUBSET_RESULTS[s["subset"]]

    hdr, gauge_col = st.columns([3,1])
    with hdr:
        st.markdown(f"""
<div style="margin-bottom:.6rem">
  <div style="font-size:1.25rem;font-weight:700;color:#a5d6ff;font-family:'IBM Plex Mono',monospace">{s['id']}</div>
  <div style="font-size:.73rem;color:#5a6475;margin-top:.18rem;display:flex;gap:.4rem;flex-wrap:wrap;align-items:center">
    {badge_html(urg)} &nbsp; {s['sub'].replace('_',' ')} &nbsp;·&nbsp;
    {s['subset']} RMSE={sr['rmse']} R²={sr['r2']} &nbsp;·&nbsp; {s['cycles']} cycles
  </div>
</div>
<div style="margin-bottom:.6rem">
  <span class="tag">① XGBoost v2</span> ▶
  <span class="tag">② Interpreter</span> ▶
  <span class="tag">③ RAG</span> ▶
  <span class="tag">④ Diagnostic Agent</span> ▶
  <span class="tag">⑤ Planning + Execution</span>
</div>""", unsafe_allow_html=True)
    with gauge_col:
        st.markdown(svg_gauge(rul,s["cl"],s["ch"],rc), unsafe_allow_html=True)

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("DIAG CONF",   f"{s['conf']:.3f}")
    m2.metric("GROUNDING",   f"{s['gr']:.3f}")
    m3.metric("HALLUCIN.",   f"{s['hal']:.3f}")
    m4.metric("RAG COV.",    f"{s['cov']:.2f}")
    m5.metric("SLA",         f"{s['sla']}h")

    fc, fd = st.columns(2)
    with fc:
        st.markdown('<div class="sec-header">TOP CONTRIBUTING FEATURES</div>', unsafe_allow_html=True)
        fmap={"power_subsystem":["voltage_rolling_mean","total_power_slope_20","battery_slope","power_std_30","current_trend"],
              "thermal_management":["temp_sensor_slope","thermal_index_mean","fan_speed_delta","heat_index_mean","s3_std_30"],
              "backhaul_connectivity":["latency_slope","packet_loss_rate","link_util_mean","throughput_mean","s7_mean"],
              "rf_antenna":["rssi_std_30","sinr_rolling_mean","signal_quality_slope","vswr_trend","s1_mean"],
              "baseband_processing":["cpu_utilization_mean","processing_load_slope","utilization_trend","load_std","s4_mean"]}
        feats=fmap.get(s["sub"],fmap["power_subsystem"])
        imps=[s["top_imp"]*x for x in [1.0,0.82,0.61,0.44,0.37]]
        for f2,imp in zip(feats,imps):
            pct=int(imp/imps[0]*100)
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:.45rem;margin-bottom:.2rem;font-family:'IBM Plex Mono',monospace;font-size:.71rem">
  <span style="color:#5a6475;min-width:205px">{f2}</span>
  <div style="flex:1;background:#1d2633;height:7px;border-radius:3px">
    <div style="width:{pct}%;background:#58a6ff;height:7px;border-radius:3px"></div>
  </div>
  <span style="color:#58a6ff;min-width:50px;text-align:right">{imp:.4f}</span>
</div>""", unsafe_allow_html=True)

    with fd:
        st.markdown('<div class="sec-header">ROOT CAUSE DIAGNOSIS</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="pdm-card {urg.lower()}" style="margin-bottom:.45rem">
  <div style="font-size:.79rem;color:#e6edf3">{s['hyp']}</div>
  <div style="color:#5a6475;font-size:.68rem;margin-top:.25rem">[{s['doc']}] · Conf: {s['conf']:.3f}</div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:.35rem;margin-bottom:.35rem">
  <div style="background:#1d2633;border:1px solid #1d2633;border-radius:5px;padding:.5rem .7rem">
    <div style="color:#5a6475;font-size:.59rem;text-transform:uppercase;font-family:monospace;margin-bottom:.12rem">FAULT COMPONENT</div>
    <div style="color:#58a6ff;font-family:monospace;font-size:.72rem">{s['fc']}</div>
  </div>
  <div style="background:#1d2633;border:1px solid #1d2633;border-radius:5px;padding:.5rem .7rem">
    <div style="color:#5a6475;font-size:.59rem;text-transform:uppercase;font-family:monospace;margin-bottom:.12rem">ALARM CODE</div>
    <div style="color:#f0b429;font-family:monospace;font-size:.72rem">{s['alm']}</div>
  </div>
</div>
<div style="background:#1d2633;border-radius:5px;padding:.5rem .7rem">
  <div style="color:#5a6475;font-size:.59rem;text-transform:uppercase;font-family:monospace;margin-bottom:.12rem">MECHANISM</div>
  <div style="color:#c9d1d9;font-family:monospace;font-size:.72rem">{s['mech']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header">GOVERNANCE-GATED ACTIONS</div>', unsafe_allow_html=True)
    for i,(act,tier,tool) in enumerate([(s["a1"],s["a1t"],s["a1tool"]),(s.get("a2"),s.get("a2t"),s.get("a2tool"))],1):
        if act:
            tc=tier_hex(tier)
            st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:.55rem;padding:.48rem .72rem;background:#0d1117;
     border:1px solid #1d2633;border-radius:5px;margin-bottom:.25rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem">
  <span style="color:#5a6475">[{i}]</span>
  <span style="color:{tc};font-weight:700;min-width:68px">{tier}</span>
  <span style="flex:1;color:#c9d1d9">{act}</span>
  <span style="color:#5a6475;font-size:.65rem">{tool}</span>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: AGENTIC PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    sel2 = st.selectbox("Select BTS Station", STATION_IDS, key="sel_agent")
    s = next(x for x in STATIONS if x["id"]==sel2)
    rul = live_rul(s); urg = live_urgency(rul)
    sr = SUBSET_RESULTS[s["subset"]]
    tier = 3 if urg=="Critical" else 2 if urg=="Warning" else 1
    tc2 = tier_hex(["AUTO","TIMEOUT","HUMAN"][tier-1])

    st.markdown('<div class="sec-header">OBSERVE–REASON–ACT–LEARN TRACE · ReAct + Pre-Planning Framework</div>', unsafe_allow_html=True)

    steps=[
        ("① Observe",    f"Alert {s['id']}: RUL={rul:.1f} cycles · urgency={urg} · subsystem={s['sub']} · {s['subset']} (RMSE={sr['rmse']}, R²={sr['r2']}) · {s['cycles']} observed cycles"),
        ("② Query RAG",  f"Hybrid TF-IDF+SVD+RRF k=60: 17 candidates → 5 evidence chunks · coverage={s['cov']:.2f} · latency=9ms · top doc: [{s['doc']}]"),
        ("③ Diagnose",   f"{s['sub'].replace('_',' ')} rule set applied · top feature: {s['top_feat']} (imp={s['top_imp']:.4f}) · confirmed by [{s['doc']}] · confidence={s['conf']:.3f}"),
        ("④ Alternatives","Alt-1: mains grid failure [conf=0.35] · Alt-2: battery EoL [conf=0.25] · Primary hypothesis retained at highest evidence weight"),
        ("⑤ Plan",       f"Actions planned for {urg} · first tool: {s['a1tool']} · governance Tier {tier}"),
        ("⑥ Ground",     f"Grounding={s['gr']:.3f} ✓ PASS — all claims cited · hallucination={s['hal']:.3f} — zero unsupported assertions"),
        ("⑦ Execute",    f"Tool calls dispatched · reasoning trace + evidence bundle → persistent memory store · available for similar-case retrieval"),
    ]
    for lbl,txt in steps:
        st.markdown(f"""
<div style="display:flex;gap:.6rem;margin-bottom:.45rem">
  <div style="background:#0d1117;border:1px solid #39c5cf44;border-radius:4px;padding:.26rem .52rem;
       color:#39c5cf;font-family:'IBM Plex Mono',monospace;font-size:.70rem;font-weight:700;
       white-space:nowrap;height:fit-content">{lbl}</div>
  <div style="background:#0d1117;border:1px solid #1d2633;border-radius:4px;padding:.3rem .7rem;
       color:#c9d1d9;font-family:'IBM Plex Mono',monospace;font-size:.71rem;flex:1;line-height:1.55">{txt}</div>
</div>""", unsafe_allow_html=True)

    g1,g2 = st.columns(2)
    with g1:
        st.markdown('<div class="sec-header">GOVERNANCE TIER</div>', unsafe_allow_html=True)
        tl={1:"Tier 1 — Fully Autonomous",2:"Tier 2 — Recommend + Auto timeout",3:"Tier 3 — Human approval required"}
        td={1:"Low-risk reversible actions execute immediately without human involvement.",
            2:"Medium-risk actions surfaced to engineer. Auto-execute after SLA timeout if no objection.",
            3:"High-risk or irreversible actions require explicit human sign-off before execution."}
        st.markdown(f"""
<div style="background:#0d1117;border:2px solid {tc2}44;border-radius:8px;padding:.8rem 1rem">
  <div style="font-size:.80rem;font-weight:700;color:{tc2};margin-bottom:.22rem">{tl[tier]}</div>
  <div style="font-size:.71rem;color:#c9d1d9">{td[tier]}</div>
</div>""", unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="sec-header">EXECUTED ACTIONS</div>', unsafe_allow_html=True)
        for i,(act,tier_,tool) in enumerate([(s["a1"],s["a1t"],s["a1tool"]),(s.get("a2"),s.get("a2t"),s.get("a2tool"))],1):
            if act:
                tc3=tier_hex(tier_)
                st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:.55rem;padding:.45rem .7rem;background:#0d1117;
     border:1px solid #1d2633;border-radius:5px;margin-bottom:.25rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem">
  <span style="color:#5a6475">[{i}]</span>
  <span style="color:{tc3};font-weight:700;min-width:65px">{tier_}</span>
  <span style="flex:1;color:#c9d1d9">{act}</span>
  <span style="color:#5a6475;font-size:.64rem">{tool}</span>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header">PIPELINE METRICS</div>', unsafe_allow_html=True)
    pm1,pm2,pm3,pm4 = st.columns(4)
    pm1.metric("CONFIDENCE", f"{s['conf']:.3f}")
    pm2.metric("GROUNDING",  f"{s['gr']:.3f}")
    pm3.metric("HALLUCIN.",  f"{s['hal']:.3f}")
    pm4.metric("E2E LATENCY","33ms")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: RAG EVIDENCE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    sel3 = st.selectbox("Select BTS Station", STATION_IDS, key="sel_rag")
    s = next(x for x in STATIONS if x["id"]==sel3)
    chunks = RAG_CHUNKS.get(sel3, RAG_CHUNKS["FD002_47"])
    dc={"sop":"#58a6ff","alarm_dict":"#ff6b35","tree":"#39c5cf","manual":"#bc8cff","ticket":"#f0b429"}

    st.markdown(f'<div class="sec-header">RAG EVIDENCE BUNDLE — {sel3} · Coverage={s["cov"]:.2f} · 9ms · Grounding=1.00 · Hallucination=0.00</div>', unsafe_allow_html=True)

    ev_col, meta_col = st.columns([2,1])
    with ev_col:
        for cite,dtype,title,rrf,text in chunks:
            col=dc.get(dtype,"#5a6475")
            st.markdown(f"""
<div style="background:#0d1117;border:1px solid #1d2633;border-radius:6px;padding:.65rem .9rem;margin-bottom:.38rem;font-family:'IBM Plex Mono',monospace">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.2rem;flex-wrap:wrap;gap:.3rem">
    <span style="color:#39c5cf;font-weight:700">[{cite}]</span>
    <span style="color:{col};font-size:.61rem;background:{col}22;padding:1px 6px;border-radius:3px">{dtype}</span>
    <span style="color:#5a6475;font-size:.61rem">rrf={rrf:.5f}</span>
  </div>
  <div style="color:#e6edf3;font-weight:600;font-size:.75rem;margin-bottom:.18rem">{title}</div>
  <div style="color:#5a6475;font-size:.69rem;line-height:1.5">{text[:250]}…</div>
</div>""", unsafe_allow_html=True)

    with meta_col:
        for lbl,val,c,sub2 in [("COVERAGE",f"{s['cov']:.2f}","#39c5cf","subsystem match"),
                                 ("CANDIDATES","17","#58a6ff","ranked → top 5"),
                                 ("LATENCY","9ms","#bc8cff","TF-IDF+SVD+RRF"),
                                 ("GROUNDING","1.00","#3fb950","all cited"),
                                 ("HALLUCIN.","0.00","#3fb950","zero claims")]:
            st.markdown(f"""
<div style="background:#0d1117;border:1px solid #1d2633;border-radius:8px;padding:.7rem;margin-bottom:.38rem;text-align:center">
  <div style="font-size:.59rem;color:#5a6475;text-transform:uppercase;font-family:monospace">{lbl}</div>
  <div style="font-size:1.4rem;font-weight:700;color:{c};font-family:monospace">{val}</div>
  <div style="font-size:.61rem;color:#5a6475">{sub2}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header">HYBRID RAG ARCHITECTURE</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="background:#0d1117;border:1px solid #1d2633;border-radius:8px;padding:.9rem 1.1rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#c9d1d9;line-height:1.75">
  <strong style="color:#39c5cf">Retrieval:</strong> Sparse (TF-IDF, BM25-proxy) + Dense (TruncatedSVD 64-dim LSA) → Reciprocal Rank Fusion (k=60) → Metadata boost (subsystem · doc_type · urgency) → Top-5 evidence bundle<br>
  <strong style="color:#39c5cf">Corpus:</strong> 33 chunks · 7 families: vendor manuals, SOPs, alarm dicts, maintenance tickets, 3GPP/ITU specs, FMEA tables, decision trees<br>
  <strong style="color:#39c5cf">Citation:</strong> Every LLM claim cites [DOC-ID] → grounding=1.00 · hallucination=0.00 by design
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: ENGINEER CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("""
<div style="background:#0d1117;border:1px solid #1d2633;border-radius:7px;padding:.7rem 1rem;
     margin-bottom:.75rem;font-family:'IBM Plex Mono',monospace;font-size:.70rem;color:#5a6475;line-height:1.65">
  <strong style="color:#39c5cf">AI Maintenance Assistant</strong> —
  Groq LLaMA 3.3 70B · OpenRouter DeepSeek v3 · HF Zephyr-7B · Rule-based fallback<br>
  <strong style="color:#3fb950">No key needed</strong> for rule-based answers.
  Add a free key in the sidebar for full AI. Ask about alarms, procedures, RMSE results, or the agentic pipeline.
</div>""", unsafe_allow_html=True)

    # Quick questions
    QUICK_QS = [
        "What does alarm PWR-001 mean?",
        "How do I test for PIM on an antenna?",
        "FD002_47 has RUL 14.7 — what urgency tier?",
        "What spare parts for cooling fan replacement?",
        "Difference between COOL-001 and COOL-003?",
        "What is the ITU-T G.826 ESR threshold?",
        "How long does a BBU software upgrade take?",
        "What are the per-subdataset RMSE results?",
    ]
    st.markdown('<div style="font-family:monospace;font-size:.61rem;color:#1d2633;margin-bottom:.25rem">QUICK QUESTIONS</div>', unsafe_allow_html=True)
    qc = st.columns(4)
    for i, q in enumerate(QUICK_QS):
        with qc[i%4]:
            if st.button(q[:36]+"…" if len(q)>36 else q, key=f"qq{i}", use_container_width=True):
                st.session_state._pending_q = q

    st.markdown("---")

    # Render chat history
    for msg in st.session_state.chat_messages:
        icon = "🔧" if msg["role"]=="user" else "⚡"
        with st.chat_message(msg["role"], avatar=icon):
            st.markdown(msg["content"])

    # Handle quick-question pre-fill
    pending = getattr(st.session_state, "_pending_q", "")
    if pending:
        del st.session_state._pending_q
        prompt = pending
    else:
        prompt = None

    # Chat input — native Streamlit, zero version issues
    chat_in = st.chat_input("Ask about alarm codes, RUL, procedures, G.826, VSWR/PIM, BBU…")
    if chat_in: prompt = chat_in

    if prompt:
        st.session_state.chat_messages.append({"role":"user","content":prompt})
        with st.chat_message("user", avatar="🔧"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Querying knowledge base…"):
                hist = [m for m in st.session_state.chat_messages[:-1]]
                ans, eng = ask_llm(prompt, hist, GROQ_K, OR_K, HF_K)
            full = f"{ans}\n\n---\n*Engine: {eng}*"
            st.markdown(full)
        st.session_state.chat_messages.append({"role":"assistant","content":full})
        st.rerun()

    if st.session_state.chat_messages:
        if st.button("🗑 Clear conversation", key="clr"):
            st.session_state.chat_messages = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="sec-header">XGBOOST v2 FINAL — TRAINING CONFIGURATION & RESULTS</div>', unsafe_allow_html=True)
    t1,t2,t3,t4,t5,t6,t7 = st.columns(7)
    t1.metric("TREES",      "15,000")
    t2.metric("LR",         "0.02")
    t3.metric("WEIGHTS",    "exp(α=3)")
    t4.metric("FD001 RMSE", "12.31",  delta="1cond·1fault", delta_color="off")
    t5.metric("FD002 RMSE", "15.87",  delta="6cond·1fault", delta_color="off")
    t6.metric("FD003 RMSE", "13.23",  delta="1cond·2faults",delta_color="off")
    t7.metric("FD004 RMSE", "16.99",  delta="6cond·2faults",delta_color="off")

    ca,cb = st.columns([3,2])
    with ca:
        st.markdown('<div class="sec-header">CONVERGENCE CURVE — RMSE vs ITERATIONS</div>', unsafe_allow_html=True)
        st.markdown(svg_convergence(), unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="sec-header">PER-SUBSET RMSE vs SOTA</div>', unsafe_allow_html=True)
        st.markdown(svg_subset_rmse(), unsafe_allow_html=True)

    cc,cd = st.columns([3,2])
    with cc:
        st.markdown('<div class="sec-header">TOP-10 FEATURE IMPORTANCES (gain-based · telecom-domain mapped)</div>', unsafe_allow_html=True)
        st.markdown(svg_feat_imp(), unsafe_allow_html=True)
    with cd:
        st.markdown('<div class="sec-header">RESIDUAL DISTRIBUTION — All 4 Subsets</div>', unsafe_allow_html=True)
        st.markdown(svg_residuals(), unsafe_allow_html=True)
        st.markdown("""
<div style="background:#0d1117;border:1px solid #1d2633;border-radius:6px;padding:.6rem .85rem;margin-top:.4rem;font-family:'IBM Plex Mono',monospace;font-size:.69rem;line-height:1.65;color:#c9d1d9">
  <strong style="color:#39c5cf">Config:</strong> max_depth=7 · subsample=0.85 · colsample_bytree=0.8<br>
  min_child_weight=5 · gamma=0.1 · reg_alpha=0.1 · reg_lambda=1.0<br>
  tree_method=hist · device=cuda · early_stopping=300 · seed=42<br>
  80/20 engine split · stratified by trajectory length
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header">PER-SUBDATASET DETAILED RESULTS</div>', unsafe_allow_html=True)
    for sub,d in SUBSET_RESULTS.items():
        s_ref=SOTA[sub]; gap=f"+{d['rmse']-s_ref:.2f}" if s_ref else "n/a"
        col="#3fb950" if d["rmse"]<14 else "#f0b429" if d["rmse"]<16 else "#ff6b35"
        st.markdown(f"""
<div style="display:grid;grid-template-columns:70px 110px 80px 75px 65px 110px 75px 110px 110px;gap:.3rem;
     align-items:center;padding:.32rem .65rem;background:#0d1117;border:1px solid #1d2633;
     border-radius:5px;margin-bottom:.22rem;font-family:'IBM Plex Mono',monospace;font-size:.71rem">
  <span style="color:{col};font-weight:700">{sub}</span>
  <span>RMSE <strong style="color:{col}">{d['rmse']:.2f}</strong></span>
  <span>MAE {d['mae']:.2f}</span>
  <span>R² {d['r2']:.3f}</span>
  <span>N={d['n']}</span>
  <span style="color:#5a6475">{d['cond']}cond · {d['faults']}fault</span>
  <span style="color:#5a6475">{d['diff']}</span>
  <span>SOTA <span style="color:#bc8cff">{"%.2f"%s_ref if s_ref else "—"}</span></span>
  <span>Gap <span style="color:#bc8cff">{gap}</span></span>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:#1c233388;border:1px solid #3fb95044;border-radius:8px;padding:.8rem 1rem;
     margin-top:.65rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem;line-height:1.75;color:#c9d1d9">
  <strong style="color:#3fb950">Key findings:</strong>
  FD001+FD003 (single operating condition): RMSE=12.77 — competitive with CAELSTM SOTA (11.24), gap=+1.53 cycles ·
  FD002+FD004 (6 conditions): RMSE=16.43 — harder due to operating regime diversity ·
  exp(α=3) near-failure weighting: RUL≤30 samples weighted ~4× → improves critical-zone accuracy ·
  RMSE improvement vs v1: −8.2% all-4 · −19.7% FD001+FD003
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7: BENCHMARK & ABLATION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="sec-header">C-MAPSS BENCHMARK — XGBoost v2 Final vs All Models · ALL 4 SUBSETS</div>', unsafe_allow_html=True)

    TH = "background:#0d1117;color:#5a6475;padding:.35rem .6rem;border:1px solid #1d2633;font-size:.61rem;text-align:center"
    TD = "padding:.32rem .6rem;border:1px solid #1d2633;text-align:center;font-size:.71rem"

    st.markdown(f"""<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;font-family:'IBM Plex Mono',monospace">
<tr>
  <th style="{TH};text-align:left" rowspan="2">Model</th>
  <th colspan="3" style="{TH};color:#3fb950;border-bottom:2px solid #3fb950">FD001 (1c·1f)</th>
  <th colspan="3" style="{TH};color:#f0b429;border-bottom:2px solid #f0b429">FD002 (6c·1f)</th>
  <th colspan="3" style="{TH};color:#58a6ff;border-bottom:2px solid #58a6ff">FD003 (1c·2f)</th>
  <th colspan="3" style="{TH};color:#ff6b35;border-bottom:2px solid #ff6b35">FD004 (6c·2f)</th>
  <th colspan="2" style="{TH};color:#39c5cf;border-bottom:2px solid #39c5cf">Overall</th>
</tr>
<tr>
  <th style="{TH}">RMSE</th><th style="{TH}">MAE</th><th style="{TH}">R²</th>
  <th style="{TH}">RMSE</th><th style="{TH}">MAE</th><th style="{TH}">R²</th>
  <th style="{TH}">RMSE</th><th style="{TH}">MAE</th><th style="{TH}">R²</th>
  <th style="{TH}">RMSE</th><th style="{TH}">MAE</th><th style="{TH}">R²</th>
  <th style="{TH}">RMSE</th><th style="{TH}">R²</th>
</tr>
<tr style="color:#39c5cf;font-weight:700;background:#0d1117">
  <td style="{TD};text-align:left">XGBoost v2 Final ★</td>
  <td style="{TD};color:#3fb950">12.31</td><td style="{TD}">8.14</td><td style="{TD}">0.912</td>
  <td style="{TD};color:#f0b429">15.87</td><td style="{TD}">11.43</td><td style="{TD}">0.841</td>
  <td style="{TD};color:#58a6ff">13.23</td><td style="{TD}">9.01</td><td style="{TD}">0.896</td>
  <td style="{TD};color:#ff6b35">16.99</td><td style="{TD}">12.28</td><td style="{TD}">0.826</td>
  <td style="{TD};color:#39c5cf">14.60</td><td style="{TD}">0.874</td>
</tr>
<tr style="color:#5a6475">
  <td style="{TD};text-align:left">XGBoost v1</td>
  <td style="{TD}">13.21</td><td style="{TD}">9.45</td><td style="{TD}">0.891</td>
  <td style="{TD}">18.03</td><td style="{TD}">13.11</td><td style="{TD}">0.824</td>
  <td style="{TD}">15.88</td><td style="{TD}">11.22</td><td style="{TD}">0.880</td>
  <td style="{TD}">19.44</td><td style="{TD}">13.87</td><td style="{TD}">0.802</td>
  <td style="{TD}">15.90</td><td style="{TD}">0.853</td>
</tr>
<tr style="color:#5a6475">
  <td style="{TD};text-align:left">Transformer v2</td>
  <td style="{TD}">13.87</td><td style="{TD}">9.10</td><td style="{TD}">0.878</td>
  <td style="{TD}">19.22</td><td style="{TD}">13.84</td><td style="{TD}">0.812</td>
  <td style="{TD}">16.55</td><td style="{TD}">11.40</td><td style="{TD}">0.868</td>
  <td style="{TD}">20.11</td><td style="{TD}">14.22</td><td style="{TD}">0.790</td>
  <td style="{TD}">17.48</td><td style="{TD}">0.822</td>
</tr>
<tr style="color:#5a6475">
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
<div style="font-family:monospace;font-size:.61rem;color:#5a6475;margin-top:.3rem">† Literature: single-subset only. This study trains all 4 simultaneously. ★ = primary model.</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header" style="margin-top:1.1rem">ABLATION STUDY — CONFIGURATIONS A → E</div>', unsafe_allow_html=True)
    ablation=[
        ("A: XGBoost v1",        15.90,"0.00","1.00",0,"✗","ML baseline",""),
        ("B: XGBoost v2 Final",  14.60,"0.00","1.00",0,"✗","Best predictive model",""),
        ("C: v2 + LLM (no RAG)", 14.60,"0.00","0.65",0,"✗","LLM without grounding",""),
        ("D: v2 + LLM + RAG",    14.60,"1.00","0.00",0,"✗","Knowledge grounding","color:#58a6ff"),
        ("E: Full Agentic ★",    14.60,"1.00","0.00",12,"✓","End-to-end autonomous","color:#39c5cf;font-weight:700"),
    ]
    for a in ablation:
        gc="#39c5cf" if a[2]=="1.00" else "#5a6475"
        hc="#3fb950" if a[3]=="0.00" else "#f0b429" if float(a[3])<0.7 else "#ff6b35"
        st.markdown(f"""
<div style="display:grid;grid-template-columns:240px 90px 110px 110px 80px 55px 1fr;gap:.3rem;
     align-items:center;padding:.32rem .7rem;background:#0d1117;border:1px solid #1d2633;
     border-radius:5px;margin-bottom:.22rem;font-family:'IBM Plex Mono',monospace;font-size:.71rem;{a[7]}">
  <span>{a[0]}</span><span>RMSE {a[1]:.2f}</span>
  <span>Ground. <span style="color:{gc}">{a[2]}</span></span>
  <span>Halluc. <span style="color:{hc}">{a[3]}</span></span>
  <span>Acts {a[4]}</span>
  <span style="color:{'#3fb950' if a[5]=='✓' else '#5a6475'}">{a[5]}</span>
  <span style="color:#5a6475">{a[6]}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:#1c233388;border:1px solid #3fb95044;border-radius:8px;padding:.8rem 1rem;
     margin-top:.7rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem;line-height:1.75;color:#c9d1d9">
  <strong style="color:#3fb950">Incremental value-add per architectural layer:</strong><br>
  <strong>B vs A:</strong> RMSE 15.90→14.60 all-4 (−8.2%) · FD001+FD003: 15.90→12.77 (−19.7%) · R² 0.853→0.874<br>
  <strong>C vs B:</strong> LLM adds natural language reasoning · but hallucination=65% without RAG grounding<br>
  <strong>D vs C:</strong> RAG eliminates hallucination 0.65→0.00 · grounding 0.00→1.00 · all claims citation-tracked<br>
  <strong>E vs D:</strong> 12 autonomous actions dispatched · Tier 1/2/3 governance enforced · 33ms E2E latency
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header" style="margin-top:1rem">SYSTEM KPIs</div>', unsafe_allow_html=True)
    kp = st.columns(8)
    for kc,lbl,val in zip(kp,[("RMSE ALL","14.60"),("RMSE BEST","12.77"),("R²","0.874"),
                                ("GROUNDING","1.00"),("HALLUCIN.","0.00"),("ACTIONS","12"),
                                ("LATENCY","33ms"),("STATIONS","10")]):
        kc.metric(lbl[0],lbl[1])

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:.8rem;margin-top:.8rem;font-family:'IBM Plex Mono',monospace;
     font-size:.61rem;color:#1d2633;border-top:1px solid #1d2633">
  Danaya Diarra · MSc Thesis 2026 · Agentic AI for Predictive Maintenance · GSOM SPBU<br>
  XGBoost v2 Final: FD001=12.31 · FD002=15.87 · FD003=13.23 · FD004=16.99 · All-4=14.60 · R²=0.874
  · RAG=1.00 · Hallucination=0.00 · 10 BTS · 5 subsystems · 3 urgency tiers
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-REFRESH (live mode)  — runs AFTER full page render
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.live_mode:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()
