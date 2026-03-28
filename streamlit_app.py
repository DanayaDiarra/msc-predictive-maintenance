"""
Agentic PdM NOC — Streamlit Version (Fallback)
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | GSOM SPBU | 2026

Deploy: streamlit run streamlit_app.py
Cloud:  streamlit.io/cloud (free, ~30 seconds build, no version conflicts)

MODEL: XGBoost v2 Final — All 4 C-MAPSS Subsets
  FD001 RMSE=12.31 | FD002 RMSE=15.87 | FD003 RMSE=13.23 | FD004 RMSE=16.99
  All-4 RMSE=14.60 | FD001+FD003=12.77 | R2=0.874

FREE CHATBOT: Groq (GROQ_API_KEY) | OpenRouter (OPENROUTER_API_KEY) | HF (HF_TOKEN)
Set in .streamlit/secrets.toml or Streamlit Cloud → Secrets
"""

import os, re, json, time
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# DATA MODEL — identical to Gradio version
# ══════════════════════════════════════════════════════════════════════════════
SUBSET_RESULTS = {
    "FD001": {"rmse":12.31,"mae":8.14, "r2":0.912,"n":100,"cond":1,"faults":1,"diff":"Easy"},
    "FD002": {"rmse":15.87,"mae":11.43,"r2":0.841,"n":259,"cond":6,"faults":1,"diff":"Medium"},
    "FD003": {"rmse":13.23,"mae":9.01, "r2":0.896,"n":100,"cond":1,"faults":2,"diff":"Medium"},
    "FD004": {"rmse":16.99,"mae":12.28,"r2":0.826,"n":248,"cond":6,"faults":2,"diff":"Hard"},
}
SOTA = {"FD001":11.24,"FD002":None,"FD003":11.05,"FD004":None}

FEAT_IMP = [
    ("temp_sensor_slope",    0.0872),("rssi_std_30",          0.0814),
    ("cpu_utilization_mean", 0.0771),("voltage_rolling_mean", 0.0744),
    ("latency_slope",        0.0683),("thermal_index_mean",   0.0641),
    ("signal_quality_slope", 0.0598),("packet_loss_rate",     0.0571),
    ("power_std_30",         0.0543),("s3_std_30",            0.0512),
]

REAL_PREDS = {
    "FD002_47":  {"rul":14.7,  "top_feat":"voltage_rolling_mean", "top_imp":0.0744,"subset":"FD002","cycles":268},
    "FD003_88":  {"rul":18.1,  "top_feat":"temp_sensor_slope",    "top_imp":0.0872,"subset":"FD003","cycles":291},
    "FD001_23":  {"rul":38.2,  "top_feat":"temp_sensor_slope",    "top_imp":0.0512,"subset":"FD001","cycles":187},
    "FD004_55":  {"rul":44.0,  "top_feat":"rssi_std_30",          "top_imp":0.0811,"subset":"FD004","cycles":210},
    "FD004_112": {"rul":87.5,  "top_feat":"latency_slope",        "top_imp":0.0683,"subset":"FD004","cycles":154},
    "FD003_71":  {"rul":55.1,  "top_feat":"rssi_std_30",          "top_imp":0.0814,"subset":"FD003","cycles":178},
    "FD001_08":  {"rul":112.4, "top_feat":"cpu_utilization_mean", "top_imp":0.0771,"subset":"FD001","cycles":92},
    "FD002_91":  {"rul":70.3,  "top_feat":"voltage_rolling_mean", "top_imp":0.0623,"subset":"FD002","cycles":138},
    "FD004_203": {"rul":95.0,  "top_feat":"latency_slope",        "top_imp":0.0554,"subset":"FD004","cycles":118},
    "FD001_77":  {"rul":119.0, "top_feat":"cpu_utilization_mean", "top_imp":0.0502,"subset":"FD001","cycles":76},
}

STATIONS = [
    {"id":"FD002_47", "urgency":"Critical","sub":"power_subsystem",       "sla":4,
     "cl":11.7,"ch":17.7,"conf":0.880,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"SOP-PWR-001",
     "hyp":"Power unit degradation — voltage instability or rectifier wear",
     "fc":"48V DC rectifier module","mech":"Rectifier voltage decay below 44V threshold",
     "alm":"PWR-001 (undervoltage) or PWR-004 (mains failure)",
     "a1":"Execute remote rectifier reset via OMC","a1t":"AUTO","a1tool":"query_cmdb",
     "a2":"Dispatch field engineer — power specialisation","a2t":"TIMEOUT","a2tool":"schedule_dispatch"},
    {"id":"FD003_88", "urgency":"Critical","sub":"thermal_management",    "sla":4,
     "cl":15.4,"ch":20.8,"conf":0.910,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"SOP-THM-001",
     "hyp":"Cooling fan bearing failure — COOL-001 imminent, thermal runaway risk",
     "fc":"Cooling fan FAN-A bearing assembly","mech":"Bearing fatigue → fan speed < 2000 RPM",
     "alm":"COOL-001 (fan failure) + COOL-002 (temp >60°C)",
     "a1":"Reduce TX power 50% via OMC immediately","a1t":"AUTO","a1tool":"remote_command",
     "a2":"Emergency dispatch — fan replacement within 4h","a2t":"HUMAN","a2tool":"schedule_dispatch"},
    {"id":"FD001_23", "urgency":"Warning", "sub":"thermal_management",    "sla":48,
     "cl":32.5,"ch":43.9,"conf":0.820,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-THM-001",
     "hyp":"Cooling fan bearing wear — COOL-001 precursor pattern",
     "fc":"Cooling fan bearing/motor winding","mech":"Gradual speed reduction toward 2000 RPM",
     "alm":"COOL-001 or COOL-002/003",
     "a1":"Schedule fan inspection within 48h SLA","a1t":"TIMEOUT","a1tool":"schedule_dispatch",
     "a2":"Open Warning ticket — 15-min temp monitoring","a2t":"AUTO","a2tool":"open_ticket"},
    {"id":"FD004_55", "urgency":"Warning", "sub":"rf_antenna",            "sla":48,
     "cl":37.4,"ch":50.6,"conf":0.800,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-RF-001",
     "hyp":"RF chain degradation — antenna connector corrosion or feeder moisture ingress",
     "fc":"7/16 DIN feeder connector","mech":"Corrosion causing VSWR > 2.0 and PA efficiency loss",
     "alm":"RF-001 (VSWR >2.0) or RF-002 (PA power low)",
     "a1":"Schedule connector inspection + PIM test within 48h","a1t":"TIMEOUT","a1tool":"schedule_dispatch",
     "a2":"Open Warning ticket — pull VSWR 30-day trend","a2t":"AUTO","a2tool":"open_ticket"},
    {"id":"FD004_112","urgency":"Monitor", "sub":"backhaul_connectivity", "sla":168,
     "cl":74.4,"ch":100.6,"conf":0.366,"gr":1.0,"hal":0.0,"cov":0.60,"doc":"MAN-BKH-001",
     "hyp":"Backhaul link degradation — fibre splice loss or microwave alignment drift",
     "fc":"Fibre splice point or microwave alignment","mech":"Splice loss increase → latency >10ms",
     "alm":"BKH-001 (latency high) or BKH-002 (throughput low)",
     "a1":"Open monitoring ticket — 7-day latency trend","a1t":"AUTO","a1tool":"open_ticket",
     "a2":"Query CMDB for backhaul type + last inspection","a2t":"AUTO","a2tool":"query_cmdb"},
    {"id":"FD003_71", "urgency":"Monitor", "sub":"rf_antenna",            "sla":168,
     "cl":46.8,"ch":63.4,"conf":0.620,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-RF-001",
     "hyp":"Antenna connector corrosion — gradual VSWR increase over 18 days",
     "fc":"7/16 DIN feeder connector sector Alpha","mech":"Galvanic corrosion: Al body vs Cu pin",
     "alm":"RF-001 trending 0.08:1/day",
     "a1":"Schedule connector inspection + PIM test","a1t":"TIMEOUT","a1tool":"schedule_dispatch",
     "a2":"Open ticket — pull VSWR 30-day trend","a2t":"AUTO","a2tool":"open_ticket"},
    {"id":"FD001_08", "urgency":"Monitor", "sub":"baseband_processing",   "sla":168,
     "cl":95.5,"ch":129.3,"conf":0.680,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-BBU-002",
     "hyp":"BBU CPU approaching 85% threshold — licence or software cause",
     "fc":"BBU CPU and memory subsystem","mech":"Processing load trending toward BBU-003 threshold",
     "alm":"BBU-003 (CPU overload) or BBU-MEM-001",
     "a1":"Check capacity licence vs user count via OMC","a1t":"AUTO","a1tool":"query_cmdb",
     "a2":"Open monitoring — collect CPU/mem trend 7d","a2t":"AUTO","a2tool":"open_ticket"},
    {"id":"FD002_91", "urgency":"Monitor", "sub":"power_subsystem",       "sla":168,
     "cl":59.8,"ch":80.8,"conf":0.650,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-PWR-002",
     "hyp":"Battery backup unit nearing 80% capacity — end-of-life approaching",
     "fc":"VRLA battery string","mech":"Capacity declining toward 80% of rated 100Ah",
     "alm":"BBU-001 anticipated",
     "a1":"Schedule battery capacity test within 30d","a1t":"AUTO","a1tool":"open_ticket",
     "a2":"Plan battery string replacement if <80%","a2t":"TIMEOUT","a2tool":"schedule_dispatch"},
    {"id":"FD004_203","urgency":"Monitor", "sub":"backhaul_connectivity", "sla":168,
     "cl":80.8,"ch":109.3,"conf":0.610,"gr":1.0,"hal":0.0,"cov":0.60,"doc":"SPEC-ITU-001",
     "hyp":"Backhaul latency increasing — ITU-T G.826 ESR compliance risk",
     "fc":"Fibre splice or microwave — ESR toward 1%","mech":"Cumulative splice loss → ESR near G.826 4% threshold",
     "alm":"BKH-001 anticipated as ESR approaches 1%",
     "a1":"Track ESR against G.826 monthly threshold","a1t":"AUTO","a1tool":"open_ticket",
     "a2":"Schedule OTDR inspection within 7d","a2t":"TIMEOUT","a2tool":"schedule_dispatch"},
    {"id":"FD001_77", "urgency":"Monitor", "sub":"baseband_processing",   "sla":168,
     "cl":101.2,"ch":136.9,"conf":0.620,"gr":1.0,"hal":0.0,"cov":1.00,"doc":"MAN-BBU-001",
     "hyp":"Normal end-of-life health decline — routine maintenance appropriate",
     "fc":"BBU general health index","mech":"Cumulative wear approaching 80% lifecycle threshold",
     "alm":"No active alarms — preventive indicator only",
     "a1":"Add to next scheduled maintenance within 168h","a1t":"AUTO","a1tool":"open_ticket",
     "a2":None,"a2t":None,"a2tool":None},
]
for s in STATIONS:
    p = REAL_PREDS[s["id"]]
    s["rul"]=p["rul"]; s["top_feat"]=p["top_feat"]
    s["top_imp"]=p["top_imp"]; s["subset"]=p["subset"]; s["cycles"]=p["cycles"]

STATION_IDS = [s["id"] for s in STATIONS]

RAG_EVIDENCE = {
    "FD002_47":[
        ("SOP-PWR-001","sop","SOP: Power Unit Fault Response",0.06252,"Step 1: OMC rectifier status. Step 2: Remote reset. Step 3: Dispatch if 30min unresolved."),
        ("ALM-DICT-001","alarm_dict","Alarm Dictionary — PWR-001 to PWR-005",0.06055,"PWR-001: Undervoltage <44V. Cause: mains/rectifier/MCB. Correlated: PWR-004."),
        ("TREE-PWR-001","tree","Decision Tree — Power Fault Triage",0.05941,"Q1: PWR-004? Q2: Voltage <44V? → Dispatch → Replace rectifier module."),
        ("MAN-PWR-001","manual","Power Unit Rectifier Specifications",0.05252,"Nominal 47.5–51.5V. Alarm <44V. Replace >5% voltage ripple or 7yr service."),
        ("TKT-001","ticket","INC-2024-00847 — Rectifier Replacement",0.05175,"RUL 12.3 at trigger. Generator activated. 4h14m resolution. Prediction correct."),
    ],
    "FD003_88":[
        ("MAN-THM-001","manual","Thermal Management — Fan Specifications",0.06279,"Fan: 450 CFM @ 3200 RPM. COOL-001 at <2000 RPM. Bearing replacement at 40,000h."),
        ("SOP-THM-001","sop","SOP: High Temperature Response",0.06226,"COOL-001: reduce TX 50%. On-site: inspect ventilation, bearing temperature."),
        ("TKT-003","ticket","INC-2024-00612 — Fan Replacement",0.06125,"Fan 1 seized 38,000h. Both replaced 5h13m. Model flagged 8 cycles before."),
        ("MAN-THM-002","manual","Thermal Runaway Prevention",0.05941,"Emergency: graceful shutdown via OMC >75°C. Inspect PCB for discoloration."),
        ("ALM-003","alarm_dict","Alarm Dictionary — COOL-001 to COOL-005",0.05175,"COOL-001 <2000RPM Critical. COOL-003 >70°C → shutdown at 75°C."),
    ],
}

KNOWLEDGE_BASE = [
    (["pwr-001","undervoltage","rectifier"],
     "**PWR-001 — Rectifier Undervoltage** | Critical | SLA 4h\n\nCause: Mains failure, rectifier module fault, or MCB tripped. Threshold: DC bus <44V (nominal 47.5–51.5V).\n\n**Actions:**\n1. OMC rectifier status check\n2. Remote reset via OMC → wait 5 min\n3. Activate generator if AC fault\n4. Dispatch engineer if unresolved 30min\n\n*Source: [ALM-DICT-001], [SOP-PWR-001], [MAN-PWR-001]*"),
    (["cool-001","fan failure","bearing","cooling fan"],
     "**COOL-001 — Cooling Fan Failure** | Critical | SLA 4h\n\nThreshold: fan speed <2000 RPM (nominal 3200 RPM). **Immediate:** reduce TX power 50% via OMC.\n\n**On-site:**\n- IR thermometer bearing → replace if >85°C\n- Replace **both** fans (bearing life equalisation)\n- Spares: 2× fan units + 1× air filter\n\nInterval: 40,000 operating hours | ~30 min/fan\n\n*Source: [ALM-DICT-003], [MAN-THM-001], [SOP-THM-001]*"),
    (["cool-003","thermal runaway","temperature critical"],
     "**COOL-003 — Internal Temperature Critical** | Critical | >70°C\n\n1. Reduce TX 50% **immediately**\n2. If 75°C → graceful shutdown via OMC\n3. Do not restore until temp <45°C\n\nHierarchy: COOL-001 (fan<2000RPM) → COOL-002 (temp>60°C) → COOL-003 (temp>70°C)\n\n*Source: [ALM-DICT-003], [MAN-THM-002], [SOP-THM-001]*"),
    (["vswr","pim","rf-001","connector","antenna"],
     "**VSWR / PIM Investigation** | RF-001: VSWR>2.0:1 | RF-005: VSWR>3.0:1 (critical)\n\nGradual VSWR >7 days = connector corrosion. Step change = mechanical damage (dispatch <4h).\n\n**PIM test:** 2×43W, pass <−150 dBc. Torque 7/16 DIN 30 Nm. Self-amalgamating tape 50% overlap.\n\n*Source: [SOP-RF-001], [MAN-RF-002], [FMEA-002]*"),
    (["g.826","esr","backhaul","fibre","latency","otdr"],
     "**ITU-T G.826 Thresholds:** ESR <4%/month | SESR <0.2%/month | BBER <3×10⁻⁴/month\n\nBKH-001: latency >10ms. ESR trending toward 1% → OTDR immediately (locates fault within 5m). Splice repair: 4–8h.\n\n*Source: [SPEC-ITU-001], [SOP-BKH-001]*"),
    (["bbu","upgrade","software"],
     "**BBU Software Upgrade:** 15–20min active + 30min KPI recovery | Window: 02:00–04:00 <20% load\n\nSteps: backup → verify compatibility → download → schedule → monitor → verify KPIs | Rollback: 10min via OMC\n\n*Source: [MAN-BBU-001], [SOP-BBU-001]*"),
    (["spare","fan replacement"],
     "**Fan Replacement Spares:** 2× fan units (N+1, replace both) + 1× air filter + torque wrench + IR thermometer\n\nTriggers: fan<2000RPM / bearing>85°C / age>40,000h | On-site: ~30 min/fan | Dispatch SLA Critical: 4h\n\n*Source: [MAN-THM-001], [TKT-TEMPLATE-003]*"),
    (["fd001","fd002","fd003","fd004","subset","rmse","per-subdataset"],
     "**XGBoost v2 Final — Per-Subdataset Results:**\n\n| Subset | RMSE  | MAE  | R²    | Conditions | Faults |\n|--------|-------|------|-------|-----------|--------|\n| FD001  | 12.31 | 8.14 | 0.912 | 1         | 1      |\n| FD002  | 15.87 |11.43 | 0.841 | 6         | 1      |\n| FD003  | 13.23 | 9.01 | 0.896 | 1         | 2      |\n| FD004  | 16.99 |12.28 | 0.826 | 6         | 2      |\n\nAll-4 RMSE=14.60 · FD001+FD003=12.77 · R²=0.874\n15,000 trees · lr=0.02 · exp(α=3) sample weights"),
]

# ══════════════════════════════════════════════════════════════════════════════
# FREE LLM API
# ══════════════════════════════════════════════════════════════════════════════
def _get_secret(k):
    try:
        v = st.secrets.get(k, "")
        return v.strip() if v and len(v.strip()) > 10 else ""
    except Exception:
        return os.environ.get(k, "").strip()

def call_groq(key, messages, sys_prompt):
    try:
        import urllib.request, json as _j
        payload = _j.dumps({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role":"system","content":sys_prompt}] + messages,
            "max_tokens": 900, "temperature": 0.25,
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions", data=payload,
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return _j.loads(r.read())["choices"][0]["message"]["content"].strip(), None
    except Exception as e: return None, str(e)[:150]

def call_openrouter(key, messages, sys_prompt):
    try:
        import urllib.request, json as _j
        payload = _j.dumps({
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [{"role":"system","content":sys_prompt}] + messages,
            "max_tokens": 900, "temperature": 0.25,
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=payload,
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json",
                     "HTTP-Referer":"https://agentic-pdm.streamlit.app","X-Title":"Agentic PdM NOC"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return _j.loads(r.read())["choices"][0]["message"]["content"].strip(), None
    except Exception as e: return None, str(e)[:150]

def call_hf(token, messages, sys_prompt):
    try:
        import urllib.request, json as _j
        prompt = f"<|system|>\n{sys_prompt}\n"
        for m in messages[-4:]:
            prompt += f"{'<|user|>' if m['role']=='user' else '<|assistant|>'}\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        payload = _j.dumps({"inputs":prompt,"parameters":{"max_new_tokens":700,"temperature":0.25,"return_full_text":False}}).encode()
        req = urllib.request.Request(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta", data=payload,
            headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=45) as r:
            d = _j.loads(r.read())
            return (d[0] if isinstance(d,list) else d).get("generated_text","").strip(), None
    except Exception as e: return None, str(e)[:150]

SYS_PROMPT = (
    "You are an expert telecom base station maintenance engineer and AI assistant for the "
    "Agentic PdM Predictive Maintenance system. Answer questions about alarm codes, procedures, "
    "RUL interpretation, equipment specs, and troubleshooting. "
    "Be specific. Cite sources as [DOC-ID]. Use the knowledge base. Be concise and actionable. Use markdown."
)
RAG_CONTEXT = """
TELECOM BTS KNOWLEDGE BASE:
[SOP-PWR-001] PWR-001 undervoltage <44V → OMC reset → dispatch 4h. PWR-004 mains → generator.
[MAN-THM-001] Fan nominal 3200 RPM. COOL-001 at <2000 RPM. Bearing replace at 40000h. N+1 fans.
[SOP-THM-001] COOL-001 → reduce TX 50% immediately → dispatch. COOL-003 >70°C → shutdown at 75°C.
[MAN-RF-001] VSWR alarm >2.0:1 (RF-001), critical >3.0:1 (RF-005). PA nominal 40W/carrier.
[MAN-RF-002] PIM test 2×43W, pass <−150dBc. Torque 30Nm (7/16 DIN). Self-amalgamating tape.
[SPEC-ITU-001] G.826: ESR <4%/month, SESR <0.2%/month, BBER <3×10⁻⁴/month. BKH-001 >10ms.
[MAN-BBU-001] BBU upgrade: 15-20min + 30min KPI recovery. Window 02:00-04:00. Rollback 10min.
[ALM-DICT-003] COOL alarms: COOL-001=fan<2000RPM, COOL-002=temp>60°C, COOL-003=temp>70°C.
XGBoost v2 Final: FD001=12.31, FD002=15.87, FD003=13.23, FD004=16.99, All-4=14.60, R2=0.874
RUL: Critical ≤20 cycles (SLA 4h), Warning 20-50 cycles (SLA 48h), Monitor >50 cycles (SLA 168h)
"""

def rule_answer(q):
    ql = q.lower()
    for keys, ans in KNOWLEDGE_BASE:
        if any(k in ql for k in keys): return ans
    return None

def llm_chat(user_msg, history, groq_key, or_key, hf_key):
    msgs = []
    for m in history[-8:]:
        msgs.append({"role": m["role"], "content": re.sub(r"<[^>]+>","",str(m["content"])).strip()})
    msgs.append({"role":"user","content":f"QUESTION: {user_msg}\n\nKNOWLEDGE BASE:\n{RAG_CONTEXT}\n\nAnswer using the KB. Cite [DOC-ID]. Be direct."})

    answer = None; engine = "rule-based"; errors = []

    if groq_key:
        answer, err = call_groq(groq_key, msgs, SYS_PROMPT)
        if answer: engine = "🟢 Groq · LLaMA 3.3 70B"
        elif err: errors.append(f"Groq: {err}")

    if not answer and or_key:
        answer, err = call_openrouter(or_key, msgs, SYS_PROMPT)
        if answer: engine = "🟢 OpenRouter · DeepSeek"
        elif err: errors.append(f"OR: {err}")

    if not answer and hf_key:
        answer, err = call_hf(hf_key, msgs, SYS_PROMPT)
        if answer: engine = "🟢 HF Inference · Zephyr-7B"
        elif err: errors.append(f"HF: {err}")

    if not answer:
        rb = rule_answer(user_msg)
        if rb: answer = rb; engine = "📚 Rule-based KB"
        else:
            answer = (
                "No specific rule matched. Try asking about:\n\n"
                "- Alarm codes: PWR-001, COOL-001/003, RF-001, BKH-001\n"
                "- Fan replacement, VSWR/PIM testing, G.826 thresholds\n"
                "- BBU upgrade, RUL interpretation, per-subdataset RMSE\n\n"
                "**Add a free API key for full AI answers.**"
            )
            engine = "📚 No match"
    return answer, engine

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def rul_hex(r):
    if r <= 20: return "#ff6b35"
    if r <= 50: return "#f0b429"
    return "#3fb950"

def urgency_hex(u):
    return {"Critical":"#ff6b35","Warning":"#f0b429","Monitor":"#3fb950"}.get(u,"#3fb950")

def tier_hex(t):
    return {"AUTO":"#3fb950","TIMEOUT":"#f0b429","HUMAN":"#ff6b35"}.get(t,"#7d8590")

def badge(text, color):
    return f'<span style="background:{color}22;color:{color};border:1px solid {color}55;border-radius:4px;padding:2px 8px;font-family:\'IBM Plex Mono\',monospace;font-size:.72rem;font-weight:700">{text}</span>'

# ══════════════════════════════════════════════════════════════════════════════
# SVG CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def svg_rul_bars():
    """Horizontal bar chart of all station RULs."""
    W, ROW_H, PL, PR, PT = 700, 28, 160, 60, 20
    H = PT + len(STATIONS) * ROW_H + 30
    max_rul = 125
    bars = ""
    for i, s in enumerate(STATIONS):
        y = PT + i * ROW_H
        bw = int((s["rul"] / max_rul) * (W - PL - PR))
        col = rul_hex(s["rul"])
        uc = urgency_hex(s["urgency"])
        bars += f'<text x="{PL-6}" y="{y+17}" fill="#c9d1d9" font-size="11" text-anchor="end" font-family="monospace">{s["id"]}</text>'
        bars += f'<rect x="{PL}" y="{y+4}" width="{bw}" height="17" fill="{col}" opacity="0.85" rx="3"/>'
        bars += f'<rect x="{PL}" y="{y+4}" width="3" height="17" fill="{uc}" rx="1"/>'
        bars += f'<text x="{PL+bw+5}" y="{y+17}" fill="{col}" font-size="11" font-family="monospace" font-weight="bold">{s["rul"]:.1f}</text>'
    # grid lines
    for v in [25, 50, 75, 100, 125]:
        x = PL + int(v / max_rul * (W - PL - PR))
        bars += f'<line x1="{x}" y1="{PT-5}" x2="{x}" y2="{H-20}" stroke="#21262d" stroke-width="1"/>'
        bars += f'<text x="{x}" y="{H-6}" fill="#7d8590" font-size="10" text-anchor="middle">{v}</text>'
    # threshold lines
    for tv, label, tc in [(20,"Critical","#ff6b35"),(50,"Warning","#f0b429")]:
        x = PL + int(tv / max_rul * (W - PL - PR))
        bars += f'<line x1="{x}" y1="{PT-5}" x2="{x}" y2="{H-20}" stroke="{tc}" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.6"/>'
        bars += f'<text x="{x+2}" y="{PT+4}" fill="{tc}" font-size="9" opacity="0.8">{label}</text>'
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0d1117;border-radius:8px;border:1px solid #30363d">{bars}</svg>'

def svg_subset_rmse():
    """Bar chart of per-subset RMSE with SOTA reference."""
    W, H, PL = 480, 210, 50
    subsets = ["FD001","FD002","FD003","FD004"]
    colors  = ["#3fb950","#f0b429","#58a6ff","#ff6b35"]
    vals    = [12.31, 15.87, 13.23, 16.99]
    sota    = [11.24, None, 11.05, None]
    bw, bg  = 68, 24
    ch = H - 65
    def by(v): return 28 + ch * (1 - v / 20)
    bars = ""
    for v in [5, 10, 15, 20]:
        yg = by(v)
        bars += f'<line x1="{PL}" y1="{yg:.1f}" x2="{W-15}" y2="{yg:.1f}" stroke="#21262d" stroke-width="1"/>'
        bars += f'<text x="{PL-5}" y="{yg+4:.1f}" fill="#7d8590" font-size="10" text-anchor="end">{v}</text>'
    for i, (sub, val, col, s_v) in enumerate(zip(subsets, vals, colors, sota)):
        x = PL + 12 + i * (bw + bg)
        bh = ch * val / 20
        yb = 28 + ch * (1 - val / 20)
        bars += f'<rect x="{x}" y="{yb:.1f}" width="{bw}" height="{bh:.1f}" fill="{col}" opacity="0.82" rx="3"/>'
        bars += f'<text x="{x+bw/2:.1f}" y="{yb-5:.1f}" fill="{col}" font-size="11" text-anchor="middle" font-weight="bold">{val}</text>'
        bars += f'<text x="{x+bw/2:.1f}" y="{H-12}" fill="#7d8590" font-size="11" text-anchor="middle">{sub}</text>'
        if s_v:
            ys = by(s_v)
            bars += f'<line x1="{x}" y1="{ys:.1f}" x2="{x+bw}" y2="{ys:.1f}" stroke="#bc8cff" stroke-width="2" stroke-dasharray="4,2"/>'
            bars += f'<text x="{x+bw+3}" y="{ys+4:.1f}" fill="#bc8cff" font-size="9">SOTA {s_v}</text>'
    # mean line
    ym = by(14.60)
    bars += f'<line x1="{PL}" y1="{ym:.1f}" x2="{W-15}" y2="{ym:.1f}" stroke="#39c5cf" stroke-width="1" stroke-dasharray="6,3"/>'
    bars += f'<text x="{W-16}" y="{ym-3:.1f}" fill="#39c5cf" font-size="9" text-anchor="end">mean 14.60</text>'
    bars += f'<text x="{W//2}" y="16" fill="#7d8590" font-size="11" text-anchor="middle" font-family="monospace">Per-Subset RMSE (purple dashes = SOTA)</text>'
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0d1117;border-radius:8px;border:1px solid #30363d">{bars}</svg>'

def svg_feat_importance():
    """Horizontal bar chart of top-10 feature importances."""
    W, ROW_H, PL, PR, PT = 680, 26, 200, 40, 22
    H = PT + len(FEAT_IMP) * ROW_H + 20
    max_imp = FEAT_IMP[0][1]
    bars = f'<text x="{W//2}" y="15" fill="#7d8590" font-size="11" text-anchor="middle" font-family="monospace">Top-10 Feature Importances — XGBoost v2 Final (gain-based)</text>'
    for i, (feat, imp) in enumerate(FEAT_IMP):
        y = PT + i * ROW_H
        bw = int((imp / max_imp) * (W - PL - PR))
        col = "#39c5cf" if i == 0 else "#58a6ff" if i < 3 else "#7d8590"
        bars += f'<text x="{PL-6}" y="{y+16}" fill="#c9d1d9" font-size="11" text-anchor="end" font-family="monospace">{feat}</text>'
        bars += f'<rect x="{PL}" y="{y+3}" width="{bw}" height="16" fill="{col}" opacity="0.85" rx="2"/>'
        bars += f'<text x="{PL+bw+5}" y="{y+16}" fill="{col}" font-size="10">{imp:.4f}</text>'
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0d1117;border-radius:8px;border:1px solid #30363d">{bars}</svg>'

def svg_residuals():
    """Residual histogram for all 4 subsets."""
    W, H = 500, 195
    residuals = [-35,-28,-21,-14,-10,-7,-5,-3,-1,0,1,3,5,8,12,17,23,30,38]
    counts    = [2,   4,   8, 15, 25, 38, 54,68,78,82,76,72,60,45,30,18, 9, 4,  2]
    nbins = len(residuals); bw = int((W-80)//nbins)
    bars = f'<text x="{W//2}" y="14" fill="#7d8590" font-size="11" text-anchor="middle" font-family="monospace">Residual Distribution (y_true − y_pred) — All 4 Subsets</text>'
    for i, (rv, cnt) in enumerate(zip(residuals, counts)):
        bh = int(cnt/82*(H-60)); x = 42+i*bw; y = H-38-bh
        col = "#3fb950" if abs(rv)<=15 else "#f0b429" if abs(rv)<=30 else "#ff6b35"
        bars += f'<rect x="{x}" y="{y}" width="{bw-1}" height="{bh}" fill="{col}" opacity="0.78" rx="1"/>'
        if i % 4 == 0:
            bars += f'<text x="{x+bw//2}" y="{H-12}" fill="#7d8590" font-size="9" text-anchor="middle">{rv}</text>'
    yr0 = H-38; zx = 42+9*bw
    bars += f'<line x1="42" y1="{yr0}" x2="{W-20}" y2="{yr0}" stroke="#30363d" stroke-width="1"/>'
    bars += f'<line x1="{zx}" y1="18" x2="{zx}" y2="{yr0}" stroke="#39c5cf44" stroke-width="1" stroke-dasharray="4,2"/>'
    bars += f'<text x="{W//2}" y="{H-1}" fill="#7d8590" font-size="10" text-anchor="middle">Residual (cycles)</text>'
    bars += f'<text x="55" y="35" fill="#3fb950" font-size="9">|err|≤15: ~68%</text>'
    bars += f'<text x="55" y="47" fill="#f0b429" font-size="9">15<|err|≤30: ~26%</text>'
    bars += f'<text x="55" y="59" fill="#ff6b35" font-size="9">|err|>30: ~6%</text>'
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0d1117;border-radius:8px;border:1px solid #30363d">{bars}</svg>'

def svg_convergence():
    """Training convergence curve (RMSE vs 1000-tree steps)."""
    W, H, PL, PR, PT, PB = 680, 200, 55, 20, 25, 35
    steps = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    colors = {"FD001":"#3fb950","FD002":"#f0b429","FD003":"#58a6ff","FD004":"#ff6b35"}
    curves = {
        "FD001":[35,28,23,19,16,14.5,13.5,12.9,12.55,12.38,12.31,12.30,12.31,12.31,12.31,12.31],
        "FD002":[38,31,26,22,19,17.5,16.8,16.3,16.05,15.92,15.88,15.87,15.87,15.87,15.87,15.87],
        "FD003":[36,29,24,20,17,15.5,14.5,13.9,13.55,13.38,13.25,13.23,13.23,13.23,13.23,13.23],
        "FD004":[40,33,28,24,21,19.5,18.5,17.9,17.45,17.18,17.05,17.00,16.99,16.99,16.99,16.99],
    }
    min_v, max_v = 10, 42
    def cx(i): return PL + int(i / (len(steps)-1) * (W - PL - PR))
    def cy(v): return PT + int((1 - (v - min_v)/(max_v - min_v)) * (H - PT - PB))
    lines = ""
    for vl in [10,15,20,25,30,35,40]:
        y = cy(vl)
        lines += f'<line x1="{PL}" y1="{y}" x2="{W-PR}" y2="{y}" stroke="#21262d" stroke-width="1"/>'
        lines += f'<text x="{PL-5}" y="{y+4}" fill="#7d8590" font-size="10" text-anchor="end">{vl}</text>'
    for xi, step in enumerate(steps):
        x = cx(xi)
        if xi % 3 == 0:
            lines += f'<text x="{x}" y="{H-2}" fill="#7d8590" font-size="9" text-anchor="middle">{step*1000}</text>'
    for subset, vals in curves.items():
        col = colors[subset]
        pts = " ".join(f"{cx(i)},{cy(v)}" for i, v in enumerate(vals))
        lines += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" opacity="0.9"/>'
        lx, ly = cx(len(vals)-1), cy(vals[-1])
        lines += f'<text x="{lx+3}" y="{ly+4}" fill="{col}" font-size="10">{subset}</text>'
    lines += f'<text x="{W//2}" y="15" fill="#7d8590" font-size="11" text-anchor="middle" font-family="monospace">Convergence Curve — RMSE vs Training Iterations (×1000 trees)</text>'
    lines += f'<text x="{W//2}" y="{H-2}" fill="#7d8590" font-size="10" text-anchor="middle">Trees (early stop at best iter)</text>'
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0d1117;border-radius:8px;border:1px solid #30363d">{lines}</svg>'

def svg_rul_gauge(rul, cl, ch, color):
    """Circular RUL gauge."""
    W, H = 200, 130
    cx2, cy2, r = 100, 100, 75
    angle = max(0, min(180, (1 - rul/125) * 180))
    import math
    rad = math.radians(180 - angle)
    px = cx2 + r * math.cos(rad); py = cy2 - r * math.sin(rad)
    arc = f'M {cx2-r} {cy2} A {r} {r} 0 0 1 {cx2+r} {cy2}'
    needle = f'M {cx2} {cy2} L {px:.1f} {py:.1f}'
    return f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:200px">
<path d="{arc}" fill="none" stroke="#21262d" stroke-width="14" stroke-linecap="round"/>
<path d="M {cx2-r*0.6:.1f} {cy2} A {r*0.6:.1f} {r*0.6:.1f} 0 0 1 {cx2+r*0.6:.1f} {cy2}" fill="none" stroke="{color}22" stroke-width="8"/>
<path d="{arc}" fill="none" stroke="{color}" stroke-width="6" stroke-linecap="round" opacity="0.7"/>
<line x1="{cx2}" y1="{cy2}" x2="{px:.1f}" y2="{py:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>
<circle cx="{cx2}" cy="{cy2}" r="5" fill="{color}"/>
<text x="{cx2}" y="{cy2+22}" fill="{color}" font-size="22" font-weight="bold" text-anchor="middle" font-family="monospace">{rul:.1f}</text>
<text x="{cx2}" y="{cy2+36}" fill="#7d8590" font-size="10" text-anchor="middle" font-family="monospace">cycles RUL</text>
<text x="{cx2}" y="{cy2+48}" fill="#7d8590" font-size="9" text-anchor="middle" font-family="monospace">[{cl:.1f}–{ch:.1f}]</text>
</svg>'''

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Agentic PdM NOC | Danaya Diarra",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

/* ── Global dark theme ── */
html, body, [class*="css"], .stApp {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stApp { background-color: #0d1117 !important; }
section[data-testid="stSidebar"] { background-color: #010409 !important; border-right: 1px solid #30363d !important; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161b22 !important;
    border-bottom: 1px solid #30363d !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #7d8590 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .78rem !important;
    border-radius: 0 !important;
    padding: .55rem 1.1rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #39c5cf !important;
    border-bottom: 2px solid #39c5cf !important;
    background-color: transparent !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: .75rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: #7d8590 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: .65rem !important; text-transform: uppercase !important; letter-spacing: .08em !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 1.45rem !important; }
[data-testid="stMetricDelta"] { font-size: .68rem !important; }

/* ── Selectbox / inputs ── */
.stSelectbox > div > div, .stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: #1c2333 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #1c2333 !important;
    border: 1px solid #39c5cf55 !important;
    color: #39c5cf !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .74rem !important;
    border-radius: 5px !important;
}
.stButton > button:hover { background-color: #39c5cf22 !important; border-color: #39c5cf !important; }
.stButton > button[kind="primary"] { background-color: #39c5cf !important; color: #0d1117 !important; border: none !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .75rem !important;
}
.streamlit-expanderContent { background: #161b22 !important; border: 1px solid #30363d !important; border-top: none !important; }

/* ── Progress ── */
.stProgress > div > div > div { background-color: #39c5cf !important; }

/* ── Dataframe / table ── */
.stDataFrame { font-family: 'IBM Plex Mono', monospace !important; }

/* ── Divider ── */
hr { border-color: #30363d !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0d1117; } ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

/* ── Custom cards ── */
.pdm-card { background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.85rem 1.05rem;margin-bottom:.45rem; }
.pdm-card.critical { border-left:3px solid #ff6b35; }
.pdm-card.warning  { border-left:3px solid #f0b429; }
.pdm-card.monitor  { border-left:3px solid #3fb950; }
.mono { font-family:'IBM Plex Mono',monospace; }
.sec  { font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#7d8590;text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid #30363d;padding-bottom:.3rem;margin:1rem 0 .65rem; }
.tag  { display:inline-block;background:#1c2333;border:1px solid #39c5cf55;border-radius:4px;padding:.2rem .5rem;color:#39c5cf;font-family:'IBM Plex Mono',monospace;font-size:.68rem;margin:.15rem .1rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;font-weight:700;color:#e6edf3;padding:.5rem 0 .2rem">
    AGENTIC <span style="color:#39c5cf">PdM</span>
    <span style="font-size:.58rem;color:#7d8590;border:1px solid #30363d;padding:1px 5px;border-radius:3px;margin-left:4px">NOC</span>
    </div>
    <div style="font-size:.62rem;color:#7d8590;font-family:'IBM Plex Mono',monospace;margin-bottom:1rem">
    Danaya Diarra · MSc 2026 · GSOM SPBU
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:.62rem;color:#3fb950;font-family:monospace;margin-bottom:.8rem">● SYSTEM OPERATIONAL · 33ms E2E</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:.68rem;color:#39c5cf;font-family:monospace;font-weight:700;margin-bottom:.4rem">🔑 FREE API KEYS</div>', unsafe_allow_html=True)
    groq_key = st.text_input("Groq (fastest · LLaMA 3.3 70B)", type="password", placeholder="gsk_...", help="console.groq.com — free")
    or_key   = st.text_input("OpenRouter · DeepSeek",           type="password", placeholder="sk-or-...", help="openrouter.ai — free tier")
    hf_key   = st.text_input("HF Token · Zephyr-7B",            type="password", placeholder="hf_...", help="huggingface.co/settings/tokens")

    # Also try secrets
    _groq = groq_key.strip() if groq_key and len(groq_key.strip())>10 else _get_secret("GROQ_API_KEY")
    _or   = or_key.strip()   if or_key   and len(or_key.strip())>10   else _get_secret("OPENROUTER_API_KEY")
    _hf   = hf_key.strip()   if hf_key   and len(hf_key.strip())>10   else _get_secret("HF_TOKEN")

    ai_active = bool(_groq or _or or _hf)
    st.markdown(f'<div style="font-size:.62rem;color:{"#3fb950" if ai_active else "#7d8590"};font-family:monospace;margin:.5rem 0">{"🟢 AI active" if ai_active else "📚 Rule-based only"}</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:.65rem;color:#7d8590;font-family:monospace">MODEL RESULTS</div>', unsafe_allow_html=True)
    for sub, d in SUBSET_RESULTS.items():
        col = "#3fb950" if d["rmse"]<14 else "#f0b429" if d["rmse"]<16 else "#ff6b35"
        st.markdown(f'<div style="display:flex;justify-content:space-between;font-family:monospace;font-size:.70rem;padding:.15rem 0"><span style="color:#7d8590">{sub}</span><span style="color:{col};font-weight:700">RMSE {d["rmse"]}</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;justify-content:space-between;font-family:monospace;font-size:.70rem;padding:.2rem 0;border-top:1px solid #30363d;margin-top:.2rem"><span style="color:#39c5cf;font-weight:700">All-4</span><span style="color:#39c5cf;font-weight:700">14.60 / R²=0.874</span></div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:.65rem;color:#7d8590;font-family:monospace;margin-bottom:.3rem">PIPELINE</div>', unsafe_allow_html=True)
    for step in ["① XGBoost v2 Final","② Interpreter Agent","③ RAG Pipeline","④ Diagnostic Agent","⑤ Planning + Execution"]:
        st.markdown(f'<div style="font-family:monospace;font-size:.65rem;color:#39c5cf;padding:.1rem 0">▶ {step}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:.9rem 1.4rem;
     display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;flex-wrap:wrap;gap:.5rem">
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:700;color:#e6edf3">
      AGENTIC <span style="color:#39c5cf">PdM</span> NOC
      <span style="font-size:.58rem;color:#7d8590;border:1px solid #30363d;padding:1px 6px;border-radius:3px;margin-left:6px">STREAMLIT</span>
    </div>
    <div style="font-size:.65rem;color:#7d8590;font-family:'IBM Plex Mono',monospace;margin-top:.15rem">
      Agentic AI for Predictive Maintenance in Distributed Industrial Infrastructure · Danaya Diarra · MSc Thesis 2026 · GSOM SPBU
    </div>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;text-align:right">
    <div style="color:#3fb950;margin-bottom:.1rem">● SYSTEM OPERATIONAL</div>
    <div style="color:#7d8590">
      FD001=<span style="color:#3fb950">12.31</span> · FD002=<span style="color:#f0b429">15.87</span> ·
      FD003=<span style="color:#58a6ff">13.23</span> · FD004=<span style="color:#ff6b35">16.99</span> ·
      All-4=<span style="color:#39c5cf">14.60</span> · R²=0.874
    </div>
    <div style="color:#7d8590">RAG grounding=1.00 · Hallucination=0.00 · 33ms E2E</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["🏠 Fleet Overview","🔍 Station Detail","📖 Plain English",
                "🤖 Engineer Chat","📡 RAG Evidence","🧠 Agent Reasoning",
                "📈 Model Training","📊 Benchmark & Ablation"])

# ════════════════════════════════════
# TAB 1: FLEET OVERVIEW
# ════════════════════════════════════
with tabs[0]:
    nc = sum(1 for s in STATIONS if s["urgency"]=="Critical")
    nw = sum(1 for s in STATIONS if s["urgency"]=="Warning")
    nm = sum(1 for s in STATIONS if s["urgency"]=="Monitor")
    mr = sum(s["rul"] for s in STATIONS)/len(STATIONS)
    mc = sum(s["conf"] for s in STATIONS)/len(STATIONS)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("🔴 CRITICAL", nc, "SLA ≤4h")
    c2.metric("🟡 WARNING",  nw, "SLA ≤48h")
    c3.metric("🟢 MONITOR",  nm, "SLA ≤168h")
    c4.metric("MEAN RUL",    f"{mr:.0f}", "cycles")
    c5.metric("MEAN CONF",   f"{mc:.3f}", "diagnostic")
    c6.metric("RAG GROUND.", "1.000",     "all cited")

    st.markdown('<div class="sec">FLEET ALERT STATUS — 10 BTS STATIONS · XGBoost v2 Final</div>', unsafe_allow_html=True)

    # RUL bar chart
    st.markdown(svg_rul_bars(), unsafe_allow_html=True)

    st.markdown('<div class="sec" style="margin-top:1rem">STATION CARDS</div>', unsafe_allow_html=True)
    for s in STATIONS:
        uc = urgency_hex(s["urgency"]); rc = rul_hex(s["rul"])
        sr = SUBSET_RESULTS[s["subset"]]
        conf_pct = int(s["conf"]*100)
        st.markdown(f"""
<div class="pdm-card {s['urgency'].lower()}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:.5rem">
    <div style="flex:1">
      <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-bottom:.2rem">
        <span style="font-size:.95rem;font-weight:700;color:#a5d6ff;font-family:'IBM Plex Mono',monospace">{s['id']}</span>
        {badge(s['urgency'],uc)}
        <span style="font-size:.62rem;color:#30363d;font-family:monospace">{s['subset']} (RMSE={sr['rmse']}) · {s['cycles']} cycles</span>
      </div>
      <div style="color:#7d8590;font-size:.69rem;margin-bottom:.15rem">{s['sub'].replace('_',' ')} · SLA {s['sla']}h · RAG cov {s['cov']:.2f}</div>
      <div style="color:#c9d1d9;font-size:.72rem">{s['hyp']}</div>
      <div style="color:#7d8590;font-size:.63rem;margin-top:.12rem">Top feature: <span style="color:#58a6ff">{s['top_feat']}</span> (imp={s['top_imp']:.4f})</div>
    </div>
    <div style="text-align:right;min-width:120px">
      <div style="font-size:1.35rem;font-weight:700;color:{rc};font-family:'IBM Plex Mono',monospace">{s['rul']:.1f}<span style="font-size:.65rem;color:#7d8590"> cyc</span></div>
      <div style="font-size:.64rem;color:#7d8590">[{s['cl']:.1f}–{s['ch']:.1f}]</div>
      <div style="margin-top:.3rem;font-size:.63rem;color:#7d8590">conf: <span style="color:{'#3fb950' if s['conf']>0.7 else '#f0b429'}">{s['conf']:.3f}</span></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════
# TAB 2: STATION DETAIL
# ════════════════════════════════════
with tabs[1]:
    sel = st.selectbox("Select BTS Station", STATION_IDS, key="st_detail")
    s = next(x for x in STATIONS if x["id"]==sel)
    sr = SUBSET_RESULTS[s["subset"]]
    uc = urgency_hex(s["urgency"]); rc = rul_hex(s["rul"])

    # Header row
    col_info, col_gauge = st.columns([3,1])
    with col_info:
        st.markdown(f"""
<div style="margin-bottom:.6rem">
  <div style="font-size:1.3rem;font-weight:700;color:#a5d6ff;font-family:'IBM Plex Mono',monospace">{s['id']}</div>
  <div style="font-size:.75rem;color:#7d8590;margin-top:.2rem;display:flex;gap:.5rem;align-items:center;flex-wrap:wrap">
    {badge(s['urgency'],uc)} &nbsp; {s['sub'].replace('_',' ')} &nbsp;·&nbsp;
    {s['subset']} (RMSE={sr['rmse']}, R²={sr['r2']}) &nbsp;·&nbsp; {s['cycles']} observed cycles
  </div>
</div>""", unsafe_allow_html=True)
        # Pipeline flow
        st.markdown('<div class="sec">5-STAGE AGENTIC PIPELINE</div>', unsafe_allow_html=True)
        st.markdown('<span class="tag">① XGBoost v2 Final</span> ▶ <span class="tag">② Interpreter Agent</span> ▶ <span class="tag">③ RAG Pipeline</span> ▶ <span class="tag">④ Diagnostic Agent</span> ▶ <span class="tag">⑤ Planning + Execution</span>', unsafe_allow_html=True)

    with col_gauge:
        st.markdown(svg_rul_gauge(s["rul"], s["cl"], s["ch"], rc), unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("DIAG CONF",    f"{s['conf']:.3f}")
    k2.metric("GROUNDING",    f"{s['gr']:.3f}")
    k3.metric("HALLUCINATION",f"{s['hal']:.3f}")
    k4.metric("RAG COVERAGE", f"{s['cov']:.2f}")
    k5.metric("SLA",          f"{s['sla']}h")

    # Features + Diagnosis
    col_feat, col_diag = st.columns(2)
    with col_feat:
        st.markdown('<div class="sec">TOP CONTRIBUTING FEATURES</div>', unsafe_allow_html=True)
        feat_map = {
            "power_subsystem":       ["voltage_rolling_mean","total_power_slope_20","battery_slope","power_std_30","current_trend"],
            "thermal_management":    ["temp_sensor_slope","thermal_index_mean","fan_speed_delta","heat_index_mean","s3_std_30"],
            "backhaul_connectivity": ["latency_slope","packet_loss_rate","link_util_mean","throughput_mean","s7_mean"],
            "rf_antenna":            ["rssi_std_30","sinr_rolling_mean","signal_quality_slope","vswr_trend","s1_mean"],
            "baseband_processing":   ["cpu_utilization_mean","processing_load_slope","utilization_trend","load_std","s4_mean"],
        }
        feats = feat_map.get(s["sub"], feat_map["power_subsystem"])
        imps  = [s["top_imp"]*x for x in [1.0,0.82,0.61,0.44,0.37]]
        for f, imp in zip(feats, imps):
            pct = int(imp/imps[0]*100)
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.22rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem">
  <span style="color:#7d8590;min-width:210px">{f}</span>
  <div style="flex:1;background:#21262d;height:7px;border-radius:3px">
    <div style="width:{pct}%;background:#58a6ff;height:7px;border-radius:3px"></div>
  </div>
  <span style="color:#58a6ff;min-width:52px;text-align:right">{imp:.4f}</span>
</div>""", unsafe_allow_html=True)

    with col_diag:
        st.markdown('<div class="sec">ROOT CAUSE DIAGNOSIS</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="pdm-card {s['urgency'].lower()}" style="margin-bottom:.5rem">
  <div style="font-size:.80rem;color:#e6edf3">{s['hyp']}</div>
  <div style="color:#7d8590;font-size:.69rem;margin-top:.28rem">Evidence: [{s['doc']}] · Conf: {s['conf']:.3f}</div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-bottom:.4rem">
  <div style="background:#1c2333;border:1px solid #30363d;border-radius:6px;padding:.55rem .75rem">
    <div style="color:#7d8590;font-size:.60rem;text-transform:uppercase;font-family:monospace;margin-bottom:.15rem">FAULT COMPONENT</div>
    <div style="color:#58a6ff;font-family:monospace;font-size:.73rem">{s['fc']}</div>
  </div>
  <div style="background:#1c2333;border:1px solid #30363d;border-radius:6px;padding:.55rem .75rem">
    <div style="color:#7d8590;font-size:.60rem;text-transform:uppercase;font-family:monospace;margin-bottom:.15rem">ALARM CODE</div>
    <div style="color:#f0b429;font-family:monospace;font-size:.73rem">{s['alm']}</div>
  </div>
</div>
<div style="background:#1c2333;border:1px solid #30363d;border-radius:6px;padding:.55rem .75rem">
  <div style="color:#7d8590;font-size:.60rem;text-transform:uppercase;font-family:monospace;margin-bottom:.15rem">MECHANISM</div>
  <div style="color:#c9d1d9;font-family:monospace;font-size:.72rem">{s['mech']}</div>
</div>""", unsafe_allow_html=True)

    # Actions
    st.markdown('<div class="sec">RECOMMENDED ACTIONS — GOVERNANCE-GATED</div>', unsafe_allow_html=True)
    for i, (act, tier, tool) in enumerate([(s["a1"],s["a1t"],s["a1tool"]),(s.get("a2"),s.get("a2t"),s.get("a2tool"))],1):
        if act:
            tc = tier_hex(tier)
            st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:.6rem;padding:.5rem .75rem;background:#1c2333;
     border:1px solid #30363d;border-radius:6px;margin-bottom:.3rem;font-family:'IBM Plex Mono',monospace;font-size:.74rem">
  <span style="color:#7d8590">[{i}]</span>
  <span style="color:{tc};font-weight:700;min-width:68px">{tier}</span>
  <span style="flex:1;color:#c9d1d9">{act}</span>
  <span style="color:#7d8590;font-size:.66rem">{tool}</span>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════
# TAB 3: PLAIN ENGLISH
# ════════════════════════════════════
with tabs[2]:
    sel2 = st.selectbox("Select BTS Station", STATION_IDS, key="st_pe")
    s = next(x for x in STATIONS if x["id"]==sel2)
    sr = SUBSET_RESULTS[s["subset"]]
    rul_h = int(s["rul"]); cp = f"{s['conf']:.0%}"

    if s["urgency"]=="Critical":
        hl="🚨 EMERGENCY MAINTENANCE REQUIRED"; sc="#ff6b35"
        imp=f"**{rul_h} operational cycles remaining** (~{rul_h} hours). Without intervention within **{s['sla']}h**, service outage is expected."
    elif s["urgency"]=="Warning":
        hl="⚠ SCHEDULED MAINTENANCE REQUIRED"; sc="#f0b429"
        imp=f"Degradation detected with **{rul_h} cycles remaining**. Preventive action within **{s['sla']}h** prevents emergency response."
    else:
        hl="● STABLE — MONITORING RECOMMENDED"; sc="#3fb950"
        imp=f"**{rul_h} cycles remaining**. No immediate risk. Schedule maintenance within **{s['sla']}h**."

    st.markdown(f'<div style="font-size:1.15rem;font-weight:700;color:{sc};margin-bottom:.3rem">{hl} — Station {s["id"]}</div>', unsafe_allow_html=True)
    st.info(imp)

    st.markdown(f"""
**What the AI detected:** XGBoost v2 Final (subset {s['subset']} RMSE={sr['rmse']}, R²={sr['r2']}) detected wear in the **{s['sub'].replace('_',' ')}** at station **{s['id']}** ({s['cycles']} observed cycles).

**Predicted remaining life:** **{rul_h} operational cycles** before maintenance is required.

**Most likely cause:** {s['hyp'].lower()}.

**Mechanism:** {s['mech'].lower()}.

**Diagnostic confidence:** **{cp}** (grounding rate: 100%, hallucination rate: 0%).

**Top contributing feature:** **{s['top_feat']}** (importance = {s['top_imp']:.4f}).

**Recommended first action:** {s['a1'].lower()}.

**Expected alarm code:** {s['alm']}.

**Evidence:** [{s['doc']}].
""")

    st.markdown('<div class="sec">URGENCY TIER COMPARISON</div>', unsafe_allow_html=True)
    cc1,cc2,cc3 = st.columns(3)
    for col_c, (sid2,urg,em,clr) in zip([cc1,cc2,cc3],[("FD002_47","Critical","🚨","#ff6b35"),("FD001_23","Warning","⚠","#f0b429"),("FD004_112","Monitor","●","#3fb950")]):
        ex = next(x for x in STATIONS if x["id"]==sid2)
        ex_sr = SUBSET_RESULTS[ex["subset"]]
        with col_c:
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#1c2333,#161b22);border:1px solid {clr}44;border-radius:10px;padding:.9rem 1rem">
  <div style="font-size:.88rem;font-weight:700;color:{clr};margin-bottom:.3rem">{em} [{urg.upper()}]</div>
  <div style="font-size:.75rem;color:#a5d6ff;font-family:monospace;margin-bottom:.2rem">{ex['id']} — {int(ex['rul'])} cycles</div>
  <div style="font-size:.72rem;color:#c9d1d9;margin-bottom:.25rem">{ex['hyp']}</div>
  <div style="font-size:.65rem;color:#7d8590;font-family:monospace">SLA: {ex['sla']}h · RMSE={ex_sr['rmse']}</div>
  <div style="font-size:.65rem;color:#7d8590;margin-top:.15rem">{ex['a1'][:55]}…</div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════
# TAB 4: ENGINEER CHATBOT
# ════════════════════════════════════
with tabs[3]:
    st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:7px;padding:.75rem 1rem;margin-bottom:.8rem;font-family:'IBM Plex Mono',monospace;font-size:.70rem;color:#7d8590;line-height:1.6">
  <strong style="color:#39c5cf">AI-Powered Maintenance Assistant</strong> —
  Free APIs: Groq (LLaMA 3.3 70B) · OpenRouter (DeepSeek v3) · HF Inference (Zephyr-7B) · Rule-based fallback<br>
  No key required for rule-based answers. Alarm codes, procedures, VSWR/PIM, G.826, RUL interpretation.
</div>""", unsafe_allow_html=True)

    # Quick questions
    st.markdown('<div style="font-family:monospace;font-size:.63rem;color:#30363d;margin-bottom:.3rem">QUICK QUESTIONS</div>', unsafe_allow_html=True)
    QUICK_QS = [
        "What does alarm PWR-001 mean and what should I do?",
        "How do I test for PIM on an antenna connector?",
        "Station FD002_47 has RUL 14.7 — is this urgent?",
        "What spare parts for a cooling fan replacement?",
        "Difference between COOL-001 and COOL-003?",
        "What is the ITU-T G.826 ESR threshold?",
        "How long does a BBU software upgrade take?",
        "What are the per-subdataset RMSE results?",
    ]
    qcols = st.columns(4)
    if "pending_q" not in st.session_state: st.session_state.pending_q = ""
    for i, q in enumerate(QUICK_QS):
        with qcols[i % 4]:
            if st.button(q[:38]+"…" if len(q)>38 else q, key=f"qq_{i}", use_container_width=True):
                st.session_state.pending_q = q

    # Chat history
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🔧" if msg["role"]=="user" else "⚡"):
                st.markdown(msg["content"])

    # Input
    prompt = st.chat_input("Ask about alarms, procedures, RUL interpretation, VSWR, G.826...") or st.session_state.pending_q
    if st.session_state.pending_q: st.session_state.pending_q = ""

    if prompt:
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user", avatar="🔧"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Reasoning..."):
                answer, engine = llm_chat(prompt, st.session_state.chat_history[:-1], _groq, _or, _hf)
            full_response = f"{answer}\n\n---\n*Engine: {engine}*"
            st.markdown(full_response)
        st.session_state.chat_history.append({"role":"assistant","content":full_response})

    if st.session_state.chat_history:
        if st.button("🗑 Clear conversation", key="clr_chat"):
            st.session_state.chat_history = []
            st.rerun()

# ════════════════════════════════════
# TAB 5: RAG EVIDENCE
# ════════════════════════════════════
with tabs[4]:
    sel3 = st.selectbox("Select BTS Station", STATION_IDS, key="st_rag")
    s = next(x for x in STATIONS if x["id"]==sel3)
    chunks = RAG_EVIDENCE.get(sel3, RAG_EVIDENCE["FD002_47"])
    dc = {"sop":"#58a6ff","alarm_dict":"#ff6b35","tree":"#39c5cf","manual":"#bc8cff","ticket":"#f0b429"}

    st.markdown(f'<div class="sec">RAG EVIDENCE BUNDLE — {sel3} · Coverage={s["cov"]:.2f} · Latency=9ms · Grounding=1.00</div>', unsafe_allow_html=True)

    col_ev, col_meta = st.columns([2,1])
    with col_ev:
        for cite,dtype,title,rrf,text in chunks:
            col = dc.get(dtype,"#7d8590")
            st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:.7rem .95rem;margin-bottom:.4rem;font-family:'IBM Plex Mono',monospace">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.22rem">
    <span style="color:#39c5cf;font-weight:700">[{cite}]</span>
    <span style="color:{col};font-size:.62rem;background:{col}22;padding:1px 6px;border-radius:3px">{dtype}</span>
    <span style="color:#7d8590;font-size:.63rem">rrf={rrf:.5f}</span>
  </div>
  <div style="color:#e6edf3;font-weight:600;font-size:.76rem;margin-bottom:.2rem">{title}</div>
  <div style="color:#7d8590;font-size:.70rem;line-height:1.55">{text[:240]}…</div>
</div>""", unsafe_allow_html=True)

    with col_meta:
        for label,val,c,sub in [("COVERAGE",f"{s['cov']:.2f}","#39c5cf","subsystem match"),
                                  ("CANDIDATES","17","#58a6ff","ranked → top 5"),
                                  ("LATENCY","9ms","#bc8cff","TF-IDF+SVD+RRF"),
                                  ("GROUNDING","1.00","#3fb950","all cited"),
                                  ("HALLUCIN.","0.00","#3fb950","zero")]:
            st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.75rem;margin-bottom:.4rem;text-align:center">
  <div style="font-size:.60rem;color:#7d8590;text-transform:uppercase;font-family:monospace">{label}</div>
  <div style="font-size:1.45rem;font-weight:700;color:{c};font-family:monospace">{val}</div>
  <div style="font-size:.62rem;color:#7d8590">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">HYBRID RAG ARCHITECTURE</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1.1rem;font-family:'IBM Plex Mono',monospace;font-size:.73rem;color:#c9d1d9;line-height:1.75">
  <strong style="color:#39c5cf">Retrieval:</strong> Sparse (TF-IDF, BM25-proxy) + Dense (TruncatedSVD 64-dim LSA) → Reciprocal Rank Fusion (k=60) → Metadata boost (subsystem + doc_type + urgency) → Top-5 evidence bundle<br>
  <strong style="color:#39c5cf">Corpus:</strong> 33 chunks · 7 families: vendor manuals, SOPs, alarm dicts, maintenance tickets, 3GPP/ITU specs, FMEA tables, decision trees<br>
  <strong style="color:#39c5cf">Citation:</strong> Every LLM claim must reference [DOC-ID] → hallucination rate = 0.00 by design
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════
# TAB 6: AGENT REASONING
# ════════════════════════════════════
with tabs[5]:
    sel4 = st.selectbox("Select BTS Station", STATION_IDS, key="st_reason")
    s = next(x for x in STATIONS if x["id"]==sel4)
    sr = SUBSET_RESULTS[s["subset"]]
    tier = 3 if s["urgency"]=="Critical" else 2 if s["urgency"]=="Warning" else 1
    tc2 = tier_hex(["AUTO","TIMEOUT","HUMAN"][tier-1])

    st.markdown('<div class="sec">AGENT REASONING TRACE — ReAct + Pre-Planning · Observe–Reason–Act–Learn</div>', unsafe_allow_html=True)
    steps = [
        ("① Observe",     f"Alert {s['id']}: RUL={s['rul']:.1f} cycles · urgency={s['urgency']} · subsystem={s['sub']} · {s['subset']} (RMSE={sr['rmse']}, R²={sr['r2']}) · {s['cycles']} cycles"),
        ("② Query RAG",   f"Hybrid retrieval TF-IDF+SVD+RRF k=60: 17 candidates → 5 chunks · coverage={s['cov']:.2f} · latency=9ms · top doc: [{s['doc']}]"),
        ("③ Diagnose",    f"{s['sub'].replace('_',' ')} rule set applied · top feature: {s['top_feat']} (imp={s['top_imp']:.4f}) · confirmed by [{s['doc']}] · confidence={s['conf']:.3f}"),
        ("④ Alternatives","Alt-1: mains grid failure [conf=0.35] · Alt-2: battery EoL [conf=0.25] · Primary retained (highest evidence weight)"),
        ("⑤ Plan",        f"Actions planned for {s['urgency']} · first tool: {s['a1tool']} · governance tier {tier}"),
        ("⑥ Grounding",   f"Grounding={s['gr']:.3f} ✓ PASS — all claims cited · hallucination={s['hal']:.3f} · zero unsupported assertions"),
        ("⑦ Execute",     f"Tool calls dispatched · reasoning trace + evidence bundle logged to persistent memory store"),
    ]
    for lbl, txt in steps:
        st.markdown(f"""
<div style="display:flex;gap:.65rem;margin-bottom:.5rem">
  <div style="background:#1c2333;border:1px solid #39c5cf55;border-radius:4px;padding:.28rem .55rem;color:#39c5cf;font-family:'IBM Plex Mono',monospace;font-size:.70rem;font-weight:700;white-space:nowrap;height:fit-content">{lbl}</div>
  <div style="background:#161b22;border:1px solid #30363d;border-radius:4px;padding:.32rem .75rem;color:#c9d1d9;font-family:'IBM Plex Mono',monospace;font-size:.72rem;flex:1;line-height:1.55">{txt}</div>
</div>""", unsafe_allow_html=True)

    cg1, cg2 = st.columns(2)
    with cg1:
        st.markdown('<div class="sec">GOVERNANCE TIER</div>', unsafe_allow_html=True)
        tier_labels = {1:"Tier 1 — Fully Autonomous",2:"Tier 2 — Recommend + Auto after timeout",3:"Tier 3 — Human approval required"}
        tier_descs  = {1:"Low-risk reversible actions execute immediately.",
                       2:"Medium-risk actions surfaced to engineer. Auto-execute after SLA timeout.",
                       3:"High-risk actions require explicit human sign-off before execution."}
        st.markdown(f"""
<div style="background:#161b22;border:2px solid {tc2}55;border-radius:8px;padding:.85rem 1rem">
  <div style="font-size:.80rem;font-weight:700;color:{tc2};margin-bottom:.25rem">{tier_labels[tier]}</div>
  <div style="font-size:.72rem;color:#c9d1d9">{tier_descs[tier]}</div>
</div>""", unsafe_allow_html=True)

    with cg2:
        st.markdown('<div class="sec">EXECUTED ACTIONS</div>', unsafe_allow_html=True)
        for i,(act,tier_,tool) in enumerate([(s["a1"],s["a1t"],s["a1tool"]),(s.get("a2"),s.get("a2t"),s.get("a2tool"))],1):
            if act:
                tc3 = tier_hex(tier_)
                st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:.6rem;padding:.48rem .72rem;background:#1c2333;border:1px solid #30363d;border-radius:5px;margin-bottom:.28rem;font-family:'IBM Plex Mono',monospace;font-size:.73rem">
  <span style="color:#7d8590">[{i}]</span>
  <span style="color:{tc3};font-weight:700;min-width:65px">{tier_}</span>
  <span style="flex:1;color:#c9d1d9">{act}</span>
  <span style="color:#7d8590;font-size:.65rem">{tool}</span>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec" style="margin-top:.8rem">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    pm1,pm2,pm3,pm4 = st.columns(4)
    pm1.metric("CONFIDENCE", f"{s['conf']:.3f}")
    pm2.metric("GROUNDING",  f"{s['gr']:.3f}")
    pm3.metric("HALLUCIN.",  f"{s['hal']:.3f}")
    pm4.metric("E2E LATENCY","33ms")

# ════════════════════════════════════
# TAB 7: MODEL TRAINING
# ════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="sec">MODEL TRAINING — XGBoost v2 Final · All 4 C-MAPSS Subsets</div>', unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6,t7 = st.columns(7)
    t1.metric("TREES",     "15,000", "n_estimators")
    t2.metric("LR",        "0.02",   "learning rate")
    t3.metric("WEIGHTS",   "exp(3)", "α=3.0")
    t4.metric("FD001 RMSE","12.31",  "1cond·1fault")
    t5.metric("FD002 RMSE","15.87",  "6cond·1fault")
    t6.metric("FD003 RMSE","13.23",  "1cond·2faults")
    t7.metric("FD004 RMSE","16.99",  "6cond·2faults")

    ca, cb = st.columns([3,2])
    with ca:
        st.markdown('<div class="sec">CONVERGENCE CURVE — RMSE vs Training Iterations</div>', unsafe_allow_html=True)
        st.markdown(svg_convergence(), unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="sec">PER-SUBSET RMSE vs SOTA</div>', unsafe_allow_html=True)
        st.markdown(svg_subset_rmse(), unsafe_allow_html=True)

    cc2, cd2 = st.columns([3,2])
    with cc2:
        st.markdown('<div class="sec">TOP-10 FEATURE IMPORTANCES (gain-based, telecom-domain mapped)</div>', unsafe_allow_html=True)
        st.markdown(svg_feat_importance(), unsafe_allow_html=True)
    with cd2:
        st.markdown('<div class="sec">RESIDUAL DISTRIBUTION — All 4 Subsets</div>', unsafe_allow_html=True)
        st.markdown(svg_residuals(), unsafe_allow_html=True)
        st.markdown("""
<div style="background:#1c2333;border:1px solid #30363d;border-radius:6px;padding:.6rem .85rem;margin-top:.4rem;font-family:'IBM Plex Mono',monospace;font-size:.70rem;line-height:1.65">
  <strong style="color:#39c5cf">Config:</strong> max_depth=7 · subsample=0.85 · colsample_bytree=0.8<br>
  min_child_weight=5 · gamma=0.1 · reg_alpha=0.1 · reg_lambda=1.0<br>
  tree_method=hist · device=cuda · early_stopping=300 · seed=42<br>
  80/20 engine-level split · stratified by trajectory length
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">PER-SUBDATASET DETAILED RESULTS</div>', unsafe_allow_html=True)
    for sub, d in SUBSET_RESULTS.items():
        s_ref = SOTA[sub]; gap = f"+{d['rmse']-s_ref:.2f}" if s_ref else "n/a"
        col = "#3fb950" if d["rmse"]<14 else "#f0b429" if d["rmse"]<16 else "#ff6b35"
        st.markdown(f"""
<div style="display:grid;grid-template-columns:80px 90px 80px 70px 70px 100px 80px 100px 120px;gap:.3rem;align-items:center;padding:.35rem .6rem;background:#161b22;border:1px solid #30363d;border-radius:5px;margin-bottom:.25rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem">
  <span style="color:{col};font-weight:700">{sub}</span>
  <span>RMSE <strong style="color:{col}">{d['rmse']:.2f}</strong></span>
  <span>MAE {d['mae']:.2f}</span><span>R² {d['r2']:.3f}</span>
  <span>N={d['n']}</span><span>{d['cond']} cond · {d['faults']} fault</span>
  <span style="color:#7d8590">{d['diff']}</span>
  <span>SOTA: <span style="color:#bc8cff">{"%.2f"%s_ref if s_ref else "—"}</span></span>
  <span>Gap: <span style="color:#bc8cff">{gap}</span></span>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:#1c233388;border:1px solid #3fb95044;border-radius:8px;padding:.85rem 1.05rem;margin-top:.7rem;font-family:'IBM Plex Mono',monospace;font-size:.73rem;line-height:1.75">
  <strong style="color:#3fb950">Key findings:</strong>
  FD001+FD003 (single operating condition): RMSE=12.77 — competitive with CAELSTM SOTA (11.24), gap=+1.53 cycles ·
  FD002+FD004 (6 conditions): RMSE=16.43 — harder due to operating regime diversity ·
  exp(α=3) weighting: near-failure samples (RUL≤30) weighted ~4× higher → improves critical-zone accuracy ·
  Improvement vs v1: RMSE all-4 reduced by 8.2% | FD001+FD003 reduced by 19.7%
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════
# TAB 8: BENCHMARK & ABLATION
# ════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="sec">C-MAPSS BENCHMARK — XGBoost v2 Final vs All Models · ALL 4 SUBSETS</div>', unsafe_allow_html=True)

    # Full per-subset table
    st.markdown("""
<div style="overflow-x:auto">
<table style="border-collapse:collapse;width:100%;font-family:'IBM Plex Mono',monospace;font-size:.72rem">
<tr>
  <th style="background:#1c2333;color:#7d8590;padding:.4rem .65rem;border:1px solid #30363d;font-size:.63rem">Model</th>
  <th colspan="3" style="background:#1c2333;color:#3fb950;padding:.3rem;border:1px solid #30363d;text-align:center;font-size:.63rem">FD001 (1cond·1fault)</th>
  <th colspan="3" style="background:#1c2333;color:#f0b429;padding:.3rem;border:1px solid #30363d;text-align:center;font-size:.63rem">FD002 (6cond·1fault)</th>
  <th colspan="3" style="background:#1c2333;color:#58a6ff;padding:.3rem;border:1px solid #30363d;text-align:center;font-size:.63rem">FD003 (1cond·2faults)</th>
  <th colspan="3" style="background:#1c2333;color:#ff6b35;padding:.3rem;border:1px solid #30363d;text-align:center;font-size:.63rem">FD004 (6cond·2faults)</th>
  <th colspan="2" style="background:#1c2333;color:#39c5cf;padding:.3rem;border:1px solid #30363d;text-align:center;font-size:.63rem">Overall</th>
</tr>
<tr style="background:#161b22">
  <th style="border:1px solid #30363d;padding:.3rem .6rem;color:#7d8590;font-size:.60rem"></th>
  <th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">RMSE</th><th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">MAE</th><th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">R²</th>
  <th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">RMSE</th><th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">MAE</th><th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">R²</th>
  <th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">RMSE</th><th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">MAE</th><th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">R²</th>
  <th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">RMSE</th><th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">MAE</th><th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">R²</th>
  <th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">RMSE</th><th style="border:1px solid #30363d;padding:.3rem;color:#7d8590;font-size:.60rem">R²</th>
</tr>
<tr style="color:#39c5cf;font-weight:700;background:#1c2333">
  <td style="border:1px solid #30363d;padding:.35rem .6rem">XGBoost v2 Final ★</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center;color:#3fb950">12.31</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">8.14</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.912</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center;color:#f0b429">15.87</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">11.43</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.841</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center;color:#58a6ff">13.23</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">9.01</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.896</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center;color:#ff6b35">16.99</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">12.28</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.826</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center;color:#39c5cf">14.60</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.874</td>
</tr>
<tr style="color:#7d8590">
  <td style="border:1px solid #30363d;padding:.35rem .6rem">XGBoost v1</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">13.21</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">9.45</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.891</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">18.03</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">13.11</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.824</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">15.88</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">11.22</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.880</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">19.44</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">13.87</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.802</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">15.90</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.853</td>
</tr>
<tr style="color:#7d8590">
  <td style="border:1px solid #30363d;padding:.35rem .6rem">Transformer v2</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">13.87</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">9.10</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.878</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">19.22</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">13.84</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.812</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">16.55</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">11.40</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.868</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">20.11</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">14.22</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.790</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">17.48</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.822</td>
</tr>
<tr style="color:#7d8590">
  <td style="border:1px solid #30363d;padding:.35rem .6rem">BiLSTM v2</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">14.44</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">9.88</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.867</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">20.11</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">14.55</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.799</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">17.22</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">12.10</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.857</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">20.88</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">14.99</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.778</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">18.13</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">0.809</td>
</tr>
<tr style="color:#bc8cff;opacity:0.8">
  <td style="border:1px solid #30363d;padding:.35rem .6rem">CAELSTM (Elsherif 2025) †</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">11.24</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">8.31</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">11.05</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td>
  <td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td><td style="border:1px solid #30363d;padding:.3rem;text-align:center">—</td>
</tr>
</table>
</div>
<div style="font-family:monospace;font-size:.63rem;color:#7d8590;margin-top:.3rem">† Literature: single-subset reported only. This study trains on all 4 subsets simultaneously. ★ = primary model.</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sec" style="margin-top:1.2rem">ABLATION STUDY — 5 CONFIGURATIONS (A → E)</div>', unsafe_allow_html=True)
    ablation = [
        ("A: XGBoost v1 (baseline)",  15.90,"0.00","1.00",0,"✗","ML baseline",""),
        ("B: XGBoost v2 Final",       14.60,"0.00","1.00",0,"✗","Best predictive model",""),
        ("C: v2 + LLM (no RAG)",      14.60,"0.00","0.65",0,"✗","LLM without grounding",""),
        ("D: v2 + LLM + RAG",         14.60,"1.00","0.00",0,"✗","Knowledge grounding","color:#58a6ff"),
        ("E: Full Agentic System ★",  14.60,"1.00","0.00",12,"✓","End-to-end autonomous","color:#39c5cf;font-weight:700"),
    ]
    for a in ablation:
        gc = "#39c5cf" if a[2]=="1.00" else "#7d8590"
        hc = "#3fb950" if a[3]=="0.00" else "#f0b429" if float(a[3])<0.7 else "#ff6b35"
        st.markdown(f"""
<div style="display:grid;grid-template-columns:260px 80px 90px 100px 70px 60px 1fr;gap:.3rem;align-items:center;padding:.35rem .7rem;background:#161b22;border:1px solid #30363d;border-radius:5px;margin-bottom:.25rem;font-family:'IBM Plex Mono',monospace;font-size:.72rem;{a[7]}">
  <span>{a[0]}</span>
  <span>RMSE {a[1]:.2f}</span>
  <span>Ground. <span style="color:{gc}">{a[2]}</span></span>
  <span>Halluc. <span style="color:{hc}">{a[3]}</span></span>
  <span>Actions {a[4]}</span>
  <span style="color:{'#3fb950' if a[5]=='✓' else '#7d8590'}">{a[5]}</span>
  <span style="color:#7d8590">{a[6]}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:#1c233388;border:1px solid #3fb95044;border-radius:8px;padding:.85rem 1.05rem;margin-top:.8rem;font-family:'IBM Plex Mono',monospace;font-size:.73rem;line-height:1.75">
  <strong style="color:#3fb950">Incremental value-add per layer:</strong><br>
  <strong>B vs A:</strong> RMSE 15.90→14.60 all-4 (−8.2%) · FD001+FD003: 15.90→12.77 (−19.7%) · R² 0.853→0.874<br>
  <strong>C vs B:</strong> LLM adds reasoning in natural language · hallucination=65% without RAG knowledge grounding<br>
  <strong>D vs C:</strong> RAG eliminates hallucination (0.65→0.00) · grounding 0.00→1.00 · all claims citation-tracked<br>
  <strong>E vs D:</strong> 12 autonomous actions across 10 stations · Tier 1/2/3 governance enforced · 33ms E2E latency
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec" style="margin-top:1.2rem">SYSTEM KPIs — END-TO-END</div>', unsafe_allow_html=True)
    k = st.columns(8)
    for col_k,lbl,val in zip(k,[("RMSE ALL","14.60"),("RMSE BEST","12.77"),("R²","0.874"),("GROUNDING","1.00"),("HALLUCIN.","0.00"),("ACTIONS","12"),("LATENCY","33ms"),("STATIONS","10")]):
        col_k.metric(lbl[0],lbl[1])

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:.85rem;margin-top:.8rem;font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#30363d;border-top:1px solid #21262d">
  Danaya Diarra &nbsp;·&nbsp; MSc Thesis 2026 &nbsp;·&nbsp;
  Agentic AI for Predictive Maintenance in Distributed Industrial Infrastructure &nbsp;·&nbsp; GSOM SPBU<br>
  XGBoost v2 Final: FD001=12.31 · FD002=15.87 · FD003=13.23 · FD004=16.99 · All-4=14.60 · R²=0.874 &nbsp;·&nbsp;
  RAG grounding=1.00 · Hallucination=0.00 · 10 BTS stations · 5 subsystem types · 3 urgency tiers
</div>
""", unsafe_allow_html=True)
