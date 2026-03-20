"""
RAG Corpus Builder — Telecom Knowledge Base
Thesis: Agentic AI for Predictive Maintenance | Danaya Diarra | March 2026

Constructs a synthetic-but-realistic telecom maintenance knowledge corpus
covering all subsystem fault types identified by the Interpreter Agent.
Each document is chunked, tagged with provenance metadata, and saved.

CORPUS STRUCTURE (7 document families, 60+ chunks):
  1. Vendor equipment manuals   (hardware specs, alarm codes, thresholds)
  2. Standard Operating Procedures (SOPs) per subsystem
  3. Alarm dictionaries with correlation rules
  4. Historical maintenance ticket templates
  5. 3GPP / ITU-T specification excerpts
  6. Failure mode & effects analysis (FMEA) tables
  7. Troubleshooting decision trees

In production: replace synthetic text with real PDFs/HTMLs using
  pdfminer + beautifulsoup + recursive character text splitter.
Architecture is identical — only the document loader changes.
"""

import os, json, hashlib, re
from dataclasses import dataclass, asdict, field
from typing import List, Optional

CORPUS_DIR = "data/rag_corpus"
os.makedirs(CORPUS_DIR, exist_ok=True)

@dataclass
class Chunk:
    chunk_id:       str
    doc_id:         str
    doc_type:       str          # manual | sop | alarm_dict | ticket | spec | fmea | tree
    equipment_family: str        # bts_outdoor | bts_indoor | rru | bbu | backhaul
    subsystem:      str
    alarm_category: Optional[str]
    software_release: Optional[str]
    title:          str
    text:           str
    keywords:       List[str]

def make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

def chunk(doc_id, doc_type, equip, subsystem, title, text,
          alarm_cat=None, sw_rel=None, extra_kw=None):
    kw = [subsystem, doc_type, equip] + (extra_kw or [])
    return Chunk(
        chunk_id=make_id(title+text), doc_id=doc_id, doc_type=doc_type,
        equipment_family=equip, subsystem=subsystem, alarm_category=alarm_cat,
        software_release=sw_rel, title=title, text=text, keywords=kw)

# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT FAMILY 1 — VENDOR EQUIPMENT MANUALS
# ══════════════════════════════════════════════════════════════════════════
MANUAL_DOCS = [
# ── Power Subsystem ──────────────────────────────────────────────────────
chunk("MAN-PWR-001","manual","bts_outdoor","power_subsystem",
  "Power Unit Rectifier Specifications and Alarm Thresholds",
  """The base station power unit consists of a rectifier module (48V DC output),
battery backup unit (BBU), and power distribution board (PDB). Rectifier output
voltage nominal range: 47.5V to 51.5V DC. Critical alarm: voltage below 44V or
above 58V triggers VSWR protection. Battery float voltage: 53.5V. Discharge cutoff:
43.2V. Typical rectifier MTBF: 150,000 hours. Replacement threshold: >5% voltage
ripple or output current deviation >3A from rated capacity. Temperature derating:
output capacity reduced 2% per degree above 40°C ambient. Alarm code PWR-001:
rectifier output undervoltage. Alarm code PWR-002: rectifier output overvoltage.
Alarm code PWR-003: battery discharge below 20% capacity. Alarm code PWR-004:
mains power failure — on battery backup. Alarm code PWR-005: high temperature in
power cabinet exceeding 55°C. Preventive replacement recommended at 7-year intervals
or when rectifier efficiency drops below 88%.""",
  alarm_cat="power", extra_kw=["rectifier","voltage","battery","PWR-001","PWR-002"]),

chunk("MAN-PWR-002","manual","bts_outdoor","power_subsystem",
  "Battery Backup Unit Maintenance Procedure",
  """Battery backup unit (BBU) maintenance schedule: monthly visual inspection for
corrosion and electrolyte leakage; quarterly capacity test (load discharge to 80%
depth, measure time to cutoff vs rated capacity); annual float voltage calibration.
Replacement triggers: capacity below 80% of rated (typically 100Ah for macro BTS);
internal resistance increase >25% from baseline; float voltage drift >0.5V from
setpoint; physical swelling or discoloration. Battery string replacement: always
replace all cells in a string simultaneously. Handling: VRLA batteries, no
electrolyte maintenance required. Disposal: return to certified recycling partner.
Alarm code BBU-001: battery capacity below threshold. BBU-002: high internal
resistance. BBU-003: float voltage out of range. BBU-004: battery temperature
exceeding 35°C (accelerates degradation). Typical battery life: 5-7 years in
tropical climates, 7-10 years in temperate climates.""",
  alarm_cat="power", extra_kw=["battery","BBU","capacity","replacement","lifecycle"]),

# ── Thermal Management ────────────────────────────────────────────────────
chunk("MAN-THM-001","manual","bts_outdoor","thermal_management",
  "Thermal Management System — Cooling Fan and HVAC Specifications",
  """Macro BTS thermal management: forced-air cooling via dual redundant fans
(N+1 configuration). Fan rated airflow: 450 CFM per fan at nominal speed 3,200 RPM.
Fan failure detection: tachometer feedback, threshold <2,000 RPM triggers COOL-001
alarm. Cabinet ambient temperature operating range: -40°C to +55°C. Internal
electronics temperature limit: +70°C. Heat dissipation: 2.4 kW typical, 3.6 kW
maximum. Thermal shutdown threshold: BBU internal temperature exceeds 75°C.
Preventive maintenance: clean air filters every 6 months in high-dust environments;
12 months in clean environments. Fan brush replacement at 40,000 hours. Bearing
lubrication: sealed, no maintenance required. Alarm COOL-002: internal temperature
high warning (>60°C). COOL-003: internal temperature critical (>70°C). COOL-004:
fan speed deviation >15% from setpoint. COOL-005: HVAC unit fault (climate-
controlled sites). Temperature data logged at 15-minute intervals for trend analysis.""",
  alarm_cat="thermal", extra_kw=["cooling","fan","temperature","COOL-001","thermal"]),

chunk("MAN-THM-002","manual","bts_indoor","thermal_management",
  "Thermal Runaway Prevention and Emergency Procedures",
  """Thermal runaway in power electronics occurs when component temperature increases
cause increased current draw, further raising temperature in a positive feedback loop.
Primary triggers in BTS: blocked ventilation paths, fan failure with high RF load,
ambient temperature exceeding design envelope. Prevention: ensure 200mm clearance
around all cabinet ventilation grilles; check filter differential pressure monthly
(replace if >50 Pa); verify fan operation at commissioning and after any power event.
Emergency procedure when COOL-003 (critical temperature) active: (1) reduce RF
transmit power by 50% via OMC command to reduce heat generation; (2) dispatch field
engineer within 2 hours; (3) if temperature exceeds 75°C, execute graceful shutdown
via OMC to prevent hardware damage; (4) do not restore service until root cause
identified and ambient temperature confirmed below 45°C. Post-incident: replace any
components that operated above rated temperature; inspect PCB for discoloration.""",
  alarm_cat="thermal", extra_kw=["thermal_runaway","emergency","shutdown","temperature"]),

# ── RF / Antenna ──────────────────────────────────────────────────────────
chunk("MAN-RF-001","manual","rru","rf_antenna",
  "Remote Radio Unit — RF Chain Specifications and Degradation Indicators",
  """RRU RF chain: PA module → directional coupler → TMA (tower mounted amplifier) →
feeder cable → antenna. Key performance indicators: VSWR (voltage standing wave
ratio) normal <1.5:1, alarm threshold >2.0:1; TX output power nominal 40W per
carrier, degradation indicator <-3dB from nominal; antenna return loss normal >15dB,
alarm <10dB. PA efficiency: 25-35% for typical macro sites. PA degradation signature:
gradual reduction in output power at fixed input drive; increased drain current;
elevated PA temperature. Alarm RF-001: VSWR high (feeder or antenna fault). RF-002:
PA output power low. RF-003: TX/RX path imbalance. RF-004: RTWP (received total
wideband power) high — interference indicator. RF-005: antenna port VSWR critical.
LNA degradation: noise figure increase >2dB from baseline. Connector inspection
interval: 24 months or after any physical disturbance. Torque specifications: 7/16
DIN connectors at 30 Nm; N-type at 20 Nm.""",
  alarm_cat="rf", extra_kw=["VSWR","PA","antenna","RF-001","LNA","connector"]),

chunk("MAN-RF-002","manual","rru","rf_antenna",
  "Antenna Connector Corrosion Prevention and Field Replacement",
  """Antenna connector corrosion is the leading cause of VSWR degradation in outdoor
BTS deployments. Corrosion mechanism: galvanic corrosion between dissimilar metals
(typically aluminium connector body and copper pin), accelerated by moisture ingress
at inadequate weatherproofing. Prevention: apply self-amalgamating tape over all
outdoor RF connections after commissioning; inspect annually; replace tape if
cracking or separation observed. Replacement procedure: (1) take sector offline via
OMC; (2) remove weatherproofing tape; (3) clean connector threads with isopropyl
alcohol; (4) visually inspect for pitting, discoloration, or debris; (5) use PIM
tester before reconnection — acceptable level <-150 dBc at 2×43W; (6) reconnect at
specified torque; (7) apply new weatherproofing tape with 50% overlap; (8) restore
sector via OMC; (9) verify VSWR and TX power within specification. Estimated duration:
45 minutes per sector. Tool requirements: torque wrench, PIM analyser, IPA spray.""",
  alarm_cat="rf", extra_kw=["connector","corrosion","VSWR","weatherproofing","PIM","antenna"]),

# ── Backhaul / Connectivity ───────────────────────────────────────────────
chunk("MAN-BKH-001","manual","backhaul","backhaul_connectivity",
  "Backhaul Link Performance Monitoring and Degradation Thresholds",
  """Backhaul transport types: fibre (preferred), microwave (legacy/remote sites),
satellite (rural/emergency). Fibre KPIs: latency <2ms for fronthaul, <5ms for
midhaul; packet loss <0.01%; jitter <1ms. Alarm BKH-001: latency exceeding 10ms.
BKH-002: packet loss >0.1%. BKH-003: link utilisation >90% sustained 15 minutes.
BKH-004: physical layer error rate >1e-6. BKH-005: backhaul link down. Microwave
KPIs: adaptive modulation minimum 4-QAM fallback, nominal 256-QAM; RSL (received
signal level) alarm threshold -75 dBm. Degradation indicators: RSL fading (rain
fade or alignment drift); ACM step-down trend (reduced modulation order); increased
BER. Fibre degradation: optical power budget reduction (splice loss increase);
chromatic dispersion; connector contamination. Inspection: clean fibre connectors
with grade A IEC 61300-3-35 tools; inspect with inspection scope; reject if core
scratches or contamination present. Fibre connector cleaning: 5 minutes per end.""",
  alarm_cat="backhaul", extra_kw=["backhaul","latency","packet_loss","BKH-001","fibre","microwave"]),

# ── Baseband Processing ───────────────────────────────────────────────────
chunk("MAN-BBU-001","manual","bbu","baseband_processing",
  "Baseband Unit Processing Load and Software Fault Management",
  """BBU CPU utilisation normal operating range: 40-70%. Alarm thresholds: warning
at 80% sustained for 5 minutes (BB-001), critical at 95% for 1 minute (BB-002).
High CPU causes: excessive handover attempts, signalling storm from UE attachment
flood, software memory leak, malformed configuration. BBU memory utilisation alarm:
>85% (BB-003). Software fault indicators: process restart events logged in system
log; core dump generation; scheduled task execution delays. Software upgrade
procedure: backup current configuration; schedule maintenance window during low
traffic (02:00-04:00 local); verify compatibility matrix; execute upgrade via OMC;
verify all processes nominal post-upgrade; confirm KPI baseline recovery within 30
minutes. Hardware fault indicators: persistent BB-002 without traffic load increase
suggests hardware degradation. FPGA reconfiguration errors indicate memory or clock
distribution fault. Replace BBU card when persistent hardware faults after software
remediation. MTBF BBU card: 200,000 hours.""",
  alarm_cat="baseband", extra_kw=["CPU","BBU","software","BB-001","BB-002","processing"]),
]

# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT FAMILY 2 — STANDARD OPERATING PROCEDURES (SOPs)
# ══════════════════════════════════════════════════════════════════════════
SOP_DOCS = [
chunk("SOP-PWR-001","sop","bts_outdoor","power_subsystem",
  "SOP: Power Unit Fault Response — Voltage Instability",
  """Trigger: Alarm PWR-001 (undervoltage) or PWR-002 (overvoltage) active, or
Interpreter Agent urgency Critical/Warning with power_subsystem hypothesis.
Step 1 — Remote diagnosis (NOC, 0-15 min): Query OMC for rectifier module status;
check AC input voltage at mains connection point via remote telemetry; verify BBU
charge state; review alarm history for preceding 24 hours. Step 2 — Remote
remediation attempt (15-30 min): If AC input nominal and rectifier fault: attempt
remote rectifier reset via OMC command; wait 5 minutes; verify output voltage.
If AC input fault: contact grid operator; activate generator if available.
Step 3 — Dispatch decision: If remote reset unsuccessful or AC fault unresolved
within 30 minutes: escalate to Tier 3 dispatch; assign field engineer with power
specialisation; SLA 4 hours for Critical, 48 hours for Warning.
Step 4 — On-site procedure: Measure DC bus voltage with calibrated multimeter;
inspect rectifier module LEDs; check DC MCB (miniature circuit breaker) status;
test each rectifier module individually by isolation. Step 5 — Resolution:
Replace failed rectifier module; verify output voltage; update CMDB; close ticket.""",
  alarm_cat="power", sw_rel="R22A", extra_kw=["SOP","voltage","rectifier","dispatch","PWR-001"]),

chunk("SOP-THM-001","sop","bts_outdoor","thermal_management",
  "SOP: Thermal Management — High Temperature Response",
  """Trigger: COOL-002 (temperature warning >60°C) or COOL-003 (critical >70°C),
or Interpreter Agent urgency Critical/Warning with thermal_management hypothesis.
Immediate actions (first 5 minutes): Verify alarm is not spurious by checking
second temperature sensor if available; check fan tachometer readings via OMC.
If fan failure confirmed (COOL-001 co-active): reduce TX power by 50% immediately
via OMC power reduction command; this reduces heat generation by ~1.2 kW.
Environmental check: retrieve nearest weather station data; if ambient >40°C,
declare thermal emergency and dispatch immediately. Remote reset: attempt fan
controller reset via OMC. On-site procedure: inspect and clear any ventilation
blockage; check filter differential pressure gauge; clean filters if blocked;
measure fan bearing temperature with IR thermometer; replace fan if bearing
temperature >85°C or speed <2,000 RPM. Escalation: if temperature does not
reduce below 60°C within 2 hours of on-site intervention, replace entire
cooling unit. Always check thermal paste condition on PA modules during
thermal incidents — replace if dry or cracked.""",
  alarm_cat="thermal", sw_rel="R22A", extra_kw=["SOP","temperature","fan","cooling","COOL-003"]),

chunk("SOP-RF-001","sop","rru","rf_antenna",
  "SOP: RF Chain Degradation — VSWR and Power Output Investigation",
  """Trigger: RF-001 (VSWR high) or RF-002 (PA power low), or Interpreter Agent
rf_antenna hypothesis. Scope: covers feeder, connectors, TMA, and RRU PA module.
Remote diagnosis: Pull TX power trend last 7 days from performance counter;
check VSWR measurement per sector; identify if degradation is gradual (connector/
feeder) or step-change (PA failure). Gradual VSWR increase pattern: indicates
connector corrosion or feeder moisture ingress — schedule within-SLA inspection.
Step VSWR change: indicates connector mechanical failure or feeder damage —
treat as urgent. On-site: connect PIM analyser to antenna port; sweep VSWR;
if VSWR >2.0 at antenna port: inspect and clean or replace connector;
if VSWR normal at port but power low: fault is in RRU PA module.
PA module replacement: 45 minutes; requires RRU power-down via OMC.
Feeder replacement: 3-4 hours; requires tower access; height safety certificate
required for technician. Document all PIM and VSWR measurements in ticket.""",
  alarm_cat="rf", sw_rel="R22A", extra_kw=["SOP","VSWR","PA","feeder","antenna","RF-001","PIM"]),

chunk("SOP-BKH-001","sop","backhaul","backhaul_connectivity",
  "SOP: Backhaul Degradation — Latency and Packet Loss Investigation",
  """Trigger: BKH-001 (latency >10ms) or BKH-002 (packet loss), or Interpreter
Agent backhaul_connectivity hypothesis. Step 1: Identify transport type (fibre/
microwave) from CMDB. Fibre path: check optical power at both ends via OMC
telemetry; compare against commissioning baseline; >3dB degradation from baseline
indicates splice or connector issue. Dispatch fibre technician with OTDR equipment.
OTDR will identify fault location to within 5 metres. Microwave path: check RSL
trend; check atmospheric conditions from weather data; if RSL dropping during
rain events — rain fade (expected, monitor); if RSL dropping in clear conditions
— alignment drift or antenna contamination. Alignment check procedure: measure
RSL vs azimuth sweep ±5 degrees from nominal; peak RSL indicates optimal pointing.
Re-alignment adjustment: 0.1-degree precision required; specialised microwave
technician required. Fibre connector cleaning: on-site, 30 minutes. Full fibre
repair after break or splice degradation: 4-8 hours per event.""",
  alarm_cat="backhaul", sw_rel="R22A", extra_kw=["SOP","latency","fibre","microwave","OTDR","BKH-001"]),

chunk("SOP-BBU-001","sop","bbu","baseband_processing",
  "SOP: BBU High CPU — Investigation and Remediation",
  """Trigger: BB-001 (CPU >80%) or BB-002 (CPU >95%), or Interpreter Agent
baseband_processing hypothesis. Remote diagnosis (NOC): retrieve process-level
CPU breakdown via OMC CLI command 'show process cpu top'; identify highest-
consuming process. Common causes: (a) handover storm — check adjacent cell
handover counters; may indicate pilot pollution requiring RF optimisation;
(b) software memory leak — process memory growing monotonically; restart
process or full BBU restart during maintenance window; (c) signalling flood —
check attach/detach rates for abnormal patterns; may indicate IoT device
firmware loop; (d) configuration error — check recent change log.
Remote remediation: for memory leak confirmed: schedule graceful BBU restart
(causes brief service interruption, <30 seconds); for handover storm: apply
temporary handover hysteresis increase via OMC; for signalling flood: identify
and isolate offending device if possible. On-site: only required if remote
remediation unsuccessful; replace BBU card if hardware fault confirmed by
diagnostic self-test. Duration: 30 minutes on-site.""",
  alarm_cat="baseband", sw_rel="R22A", extra_kw=["SOP","CPU","BBU","handover","memory","BB-001"]),
]

# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT FAMILY 3 — ALARM DICTIONARIES
# ══════════════════════════════════════════════════════════════════════════
ALARM_DOCS = [
chunk("ALM-DICT-001","alarm_dict","bts_outdoor","power_subsystem",
  "Alarm Dictionary — Power Subsystem Alarm Codes PWR-001 to PWR-005",
  """PWR-001 | Rectifier Undervoltage | Severity: Critical | Probable cause:
Mains input failure; rectifier module fault; MCB tripped. Correlated alarms:
PWR-004 (mains failure). Immediate action: verify mains input; attempt rectifier
reset. Auto-clear: Yes, if mains restored. Escalation: dispatch within 4h if
not auto-cleared. | PWR-002 | Rectifier Overvoltage | Severity: Major | Probable
cause: Rectifier regulation failure; battery charger fault. Correlated alarms:
BBU-003 (float voltage). Action: isolate rectifier module; dispatch. | PWR-003 |
Battery Low Capacity | Severity: Warning | Probable cause: End of battery life;
recent deep discharge event; high ambient temperature. Action: schedule battery
capacity test; plan replacement if confirmed below 80%. | PWR-004 | Mains Failure
| Severity: Critical | Probable cause: Grid outage; fuse blown; cable damage.
Correlated alarms: PWR-001. Action: contact grid operator; activate generator
backup; estimated generator autonomy 8h at full load, 20h at 50% load. | PWR-005
| Power Cabinet High Temperature | Severity: Major | Probable cause: HVAC fault;
blocked ventilation; high ambient. Correlated: COOL-002, COOL-003.""",
  alarm_cat="power", extra_kw=["PWR-001","PWR-002","PWR-003","PWR-004","PWR-005","alarm"]),

chunk("ALM-DICT-002","alarm_dict","rru","rf_antenna",
  "Alarm Dictionary — RF Chain Alarm Codes RF-001 to RF-005",
  """RF-001 | VSWR High Warning | Severity: Major | Threshold: VSWR >2.0:1 |
Probable cause: Antenna connector loose or corroded; feeder water ingress; antenna
physical damage. Correlated alarms: RF-005. First check: VSWR measurement per
sector; compare with commissioning record. Typical repair time: 45-90 minutes.
| RF-002 | PA Output Power Low | Severity: Major | Threshold: >3dB below nominal |
Probable cause: PA module degradation; feeder loss increase; antenna return loss
degradation. Action: check PA temperature; measure output power at coupler;
schedule RRU inspection. | RF-003 | TX/RX Path Imbalance | Severity: Warning |
Probable cause: Filter degradation; switch fault; cable routing change. |
RF-004 | RTWP High | Severity: Warning | Probable cause: External interference
source; passive intermodulation (PIM). PIM test required. | RF-005 | Antenna VSWR
Critical | Severity: Critical | Threshold: VSWR >3.0:1 | Action: immediate sector
power-down via OMC to prevent PA damage; dispatch within 4h.""",
  alarm_cat="rf", extra_kw=["RF-001","RF-002","RF-005","VSWR","PA","PIM","alarm"]),

chunk("ALM-DICT-003","alarm_dict","bts_outdoor","thermal_management",
  "Alarm Dictionary — Thermal Alarms COOL-001 to COOL-005",
  """COOL-001 | Fan Failure | Severity: Critical | Threshold: Fan speed <2000 RPM |
Probable cause: Fan motor failure; fan blade obstruction; controller fault.
Immediate: reduce TX power 50% via OMC; dispatch within 4 hours; risk of thermal
shutdown if unresolved. | COOL-002 | High Temperature Warning | Severity: Major |
Threshold: Internal >60°C | Probable cause: Fan degradation; filter blockage;
high ambient temperature; high traffic load. Action: check fan status; check
filters; reduce TX power if fan fault present. | COOL-003 | High Temperature
Critical | Severity: Critical | Threshold: Internal >70°C | Action: reduce TX
power 50% immediately; dispatch emergency; if temperature reaches 75°C execute
graceful shutdown via OMC. | COOL-004 | Fan Speed Deviation | Severity: Warning |
Threshold: Speed >15% from setpoint | Probable cause: Bearing wear; power supply
variation; partial obstruction. Schedule preventive replacement. | COOL-005 |
HVAC Unit Fault | Severity: Major | Applies to: climate-controlled indoor sites.
Probable cause: Refrigerant leak; compressor fault; thermostat failure.""",
  alarm_cat="thermal", extra_kw=["COOL-001","COOL-002","COOL-003","fan","temperature","alarm"]),
]

# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT FAMILY 4 — HISTORICAL MAINTENANCE TICKET TEMPLATES
# ══════════════════════════════════════════════════════════════════════════
TICKET_DOCS = [
chunk("TKT-TEMPLATE-001","ticket","bts_outdoor","power_subsystem",
  "Historical Ticket — Power Unit Rectifier Replacement",
  """Ticket ID: INC-2024-00847 | Site: BTS-FD002-047 | Priority: Critical |
Opened: 2024-03-15 02:14 | Closed: 2024-03-15 06:28 | Duration: 4h14m
Alarm: PWR-001 active 02:14; PWR-004 active 02:14 (mains failure)
AI prediction: RUL 12.3 cycles at alarm trigger; urgency Critical; power_subsystem
Diagnosis: Mains input failure confirmed via OMC telemetry; AC input voltage 0V;
grid outage confirmed with grid operator (substation fault).
Actions taken: (1) Activated diesel generator via remote command 02:19;
(2) Confirmed BTS on generator power 02:23; (3) Grid operator ETA 4h;
(4) Dispatched field engineer to verify generator fuel level;
(5) Generator fuel confirmed sufficient for 16h operation;
(6) Grid restored 06:15; BTS transferred back to mains 06:22;
(7) Verified all alarms cleared 06:28.
Resolution: Grid fault — no BTS hardware replaced. Generator performed correctly.
Follow-up: Schedule rectifier preventive inspection given age (6.2 years).
Lessons learned: Predictive alert correctly identified power subsystem risk
12 cycles before event. Earlier fuel top-up would have extended autonomy.""",
  extra_kw=["rectifier","mains","generator","INC-2024","power_subsystem","Critical"]),

chunk("TKT-TEMPLATE-002","ticket","rru","rf_antenna",
  "Historical Ticket — Antenna Connector Corrosion Repair",
  """Ticket ID: INC-2024-01203 | Site: BTS-FD001-023 | Priority: Warning |
Opened: 2024-05-02 09:30 | Closed: 2024-05-04 14:15 | Duration: 2d4h45m
Alarm: RF-001 active (VSWR 2.3:1 sector Alpha); gradual increase over 18 days.
AI prediction: RUL 34.8 cycles; urgency Warning; rf_antenna hypothesis;
feature total: voltage_rolling_mean imp=0.058 (likely feeder), temp_sensor_slope
imp=0.041 (secondary thermal signature from PA working harder against high VSWR).
Diagnosis: VSWR trend analysis showed 0.08:1 per day increase over 18 days —
consistent with gradual connector corrosion, not sudden mechanical failure.
Remote diagnosis: PA output power nominal; fault localised to feeder/antenna.
Actions taken: (1) Scheduled non-urgent dispatch within 48h SLA;
(2) Field engineer on-site 2024-05-04 10:00; (3) PIM test: -143 dBc (marginal,
>-150dBc threshold); (4) Visual inspection: corrosion on 7/16 DIN connector
body, grade 2 discoloration; (5) Connector replaced; new weatherproofing applied;
(6) Post-repair PIM: -158 dBc (pass); VSWR: 1.35:1 (pass).
Resolution: Connector replacement. Site restored to nominal 14:15.
Lessons learned: 18-day VSWR trend correctly flagged by predictive model.
Earlier intervention would have prevented marginal PIM reading.""",
  extra_kw=["connector","VSWR","corrosion","PIM","antenna","Warning","rf_antenna"]),

chunk("TKT-TEMPLATE-003","ticket","bts_outdoor","thermal_management",
  "Historical Ticket — Cooling Fan Replacement",
  """Ticket ID: INC-2024-00612 | Site: BTS-FD003-088 | Priority: Critical |
Opened: 2024-01-18 14:42 | Closed: 2024-01-18 19:55 | Duration: 5h13m
Alarm: COOL-001 (fan failure) + COOL-002 (temp warning 63°C) co-active 14:42.
AI prediction: RUL 8.1 cycles at trigger; urgency Critical; thermal_management;
feature temp_sensor_slope imp=0.087 — steepening temperature trend confirmed.
Immediate action: TX power reduced 50% via OMC at 14:44; internal temperature
stabilised at 65°C (below 70°C critical threshold).
Dispatch: field engineer dispatched 14:50; on-site 16:20.
On-site findings: Fan 1 seized (bearing failure); Fan 2 operating at 2,400 RPM
(below 3,200 RPM nominal but above 2,000 RPM alarm threshold, hence no COOL-004).
Action: Replaced Fan 1 (30 minutes); Fan 2 inspected — bearing lubrication
confirmed adequate; replaced as precaution due to age (Fan 1 failure at 38,000h,
Fan 2 at 38,000h — both near 40,000h replacement interval). Full TX power
restored 17:05; temperature returned to 42°C nominal by 17:30. Final test:
both fans confirmed at 3,180 RPM (within 1% of nominal). Closed 19:55.
Resolution: Both fans replaced. Root cause: bearing wear at end of service life.
Predictive model correctly flagged 8 cycles before fan seizure event.""",
  extra_kw=["fan","cooling","temperature","COOL-001","bearing","replacement","thermal_management"]),
]

# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT FAMILY 5 — SPECIFICATION EXCERPTS (3GPP / ITU-T)
# ══════════════════════════════════════════════════════════════════════════
SPEC_DOCS = [
chunk("SPEC-3GPP-001","spec","bts_outdoor","rf_antenna",
  "3GPP TS 36.104 — Base Station Radio Transmission and Reception",
  """Extract — Transmitter performance requirements. Maximum output power:
equipment class 1 (macro) minimum 43 dBm (20W), maximum 46 dBm (40W) per carrier.
Output power accuracy: ±2 dB under normal conditions; ±2.5 dB under extreme
conditions (temperature -40°C to +55°C, supply voltage ±10%). ACLR (adjacent
channel leakage ratio): Class 1 minimum 45 dB for E-UTRA adjacent channel,
30 dB for UTRA adjacent. EVM (error vector magnitude): QPSK 17.5%, 16QAM 12.5%,
64QAM 8%, 256QAM 3.5%. Frequency accuracy: ±0.05 ppm. Performance degradation
indicators warranting investigation: TX output power consistently at lower end
of ±2 dB window (suggests PA gain compression); EVM degrading toward limit
(suggests PA non-linearity or IQ imbalance); frequency accuracy drift (suggests
reference oscillator aging). Reference: 3GPP TS 36.104 v17.8.0 Section 6.""",
  extra_kw=["3GPP","TS36.104","transmitter","EVM","ACLR","output_power","specification"]),

chunk("SPEC-ITU-001","spec","backhaul","backhaul_connectivity",
  "ITU-T G.826 — End-to-End Error Performance Parameters",
  """ITU-T G.826 defines error performance objectives for international digital
connections at bit rates above primary rate. Key parameters for BTS backhaul
assessment: Errored Second Ratio (ESR) objective: <0.04 (4%) over any month;
Severely Errored Second Ratio (SESR): <0.002 (0.2%) over any month; Background
Block Error Ratio (BBER): <3×10⁻⁴ over any month. Application to BTS backhaul:
ESR trending toward 1% indicates degraded link requiring investigation; SESR any
nonzero value in a 24h window warrants immediate review. Relationship to observed
KPIs: packet loss rate >0.01% typically correlates with ESR >0.01%; jitter >5ms
in optical links indicates likely regeneration or switching fault. Use G.826
objectives as acceptance criteria for repaired backhaul links — any link not
meeting G.826 objectives after repair requires re-investigation before traffic
handback. Reference: ITU-T G.826 (12/2002).""",
  extra_kw=["ITU-T","G.826","ESR","BBER","backhaul","error_performance","specification"]),
]

# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT FAMILY 6 — FMEA TABLES
# ══════════════════════════════════════════════════════════════════════════
FMEA_DOCS = [
chunk("FMEA-001","fmea","bts_outdoor","power_subsystem",
  "FMEA — Power Subsystem Failure Modes and Effects",
  """Failure Mode 1: Rectifier output undervoltage. Effect: Battery discharge begins;
if prolonged, service outage. Detection: PWR-001 alarm, voltage telemetry.
Occurrence: 2/10. Severity: 8/10. RPN: 144. Mitigation: monthly voltage trend
monitoring; predictive replacement at 7 years.
Failure Mode 2: Battery capacity degradation below 80%. Effect: Reduced backup
autonomy; potential service outage during extended grid failure. Detection:
quarterly capacity test; BBU-001 alarm. Occurrence: 4/10. Severity: 6/10.
RPN: 168. Mitigation: annual capacity tests; replacement at 80% threshold.
Failure Mode 3: MCB tripping on overload. Effect: Total site power loss.
Detection: PWR-004 alarm; mains monitoring. Occurrence: 1/10. Severity: 9/10.
RPN: 81. Mitigation: load management; MCB rating review.
Failure Mode 4: DC bus short circuit. Effect: Immediate total power loss; possible
fire hazard. Detection: current surge protection; PWR-001. Occurrence: 0.5/10.
Severity: 10/10. RPN: 50. Mitigation: cable inspection; insulation testing.""",
  extra_kw=["FMEA","RPN","failure_mode","rectifier","battery","power_subsystem"]),

chunk("FMEA-002","fmea","rru","rf_antenna",
  "FMEA — RF Chain Failure Modes and Effects",
  """Failure Mode 1: Antenna connector corrosion. Effect: VSWR increase; PA
efficiency loss; coverage degradation. Detection: VSWR monitoring; RF-001 alarm.
Occurrence: 5/10 (high in coastal/tropical environments). Severity: 5/10.
RPN: 175 (highest in RF chain). Mitigation: annual connector inspection;
weatherproofing quality control at commissioning.
Failure Mode 2: PA module output power degradation. Effect: Reduced coverage;
increased interference from adjacent sites compensating. Detection: RF-002;
power counter trending. Occurrence: 2/10. Severity: 6/10. RPN: 84.
Mitigation: PA temperature monitoring; power trend alarming.
Failure Mode 3: Feeder water ingress. Effect: Gradual VSWR increase; cable
dielectric degradation; eventual open circuit. Detection: VSWR trend; RF-001.
Occurrence: 3/10. Severity: 6/10. RPN: 126. Mitigation: weatherproofing
inspection; feeder pressure testing.
Failure Mode 4: LNA noise figure increase. Effect: Reduced receiver sensitivity;
uplink coverage degradation. Detection: RTWP increase; sensitivity testing.
Occurrence: 1/10. Severity: 5/10. RPN: 35.""",
  extra_kw=["FMEA","RPN","antenna","PA","connector","feeder","LNA","rf_antenna"]),
]

# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT FAMILY 7 — TROUBLESHOOTING DECISION TREES
# ══════════════════════════════════════════════════════════════════════════
TREE_DOCS = [
chunk("TREE-PWR-001","tree","bts_outdoor","power_subsystem",
  "Decision Tree — Power Fault Triage",
  """START: Power alarm active or AI power_subsystem hypothesis.
Q1: Is PWR-004 (mains failure) active? YES → Q1a | NO → Q2.
Q1a: Is generator available at site? YES → Activate generator remotely → Monitor.
  NO → Contact grid operator → Dispatch if generator unavailable > 2h.
Q2: Is rectifier output voltage <44V (PWR-001)? YES → Q2a | NO → Q3.
Q2a: Was there a recent grid event (surge, outage)? YES → Reset rectifier → Verify.
  NO → Rectifier fault likely → Dispatch → Replace module.
Q3: Is battery capacity <80% (BBU-001)? YES → Schedule battery test within 7 days.
  Confirmed <80% → Plan battery string replacement within 30 days.
Q4: Is power cabinet temperature >55°C (PWR-005)? YES → COOL protocol activation.
Q5: Is DC bus voltage 48-52V but alarms active? YES → Check MCB status →
  If MCB tripped: identify overload source before resetting.
GENERAL: Any Critical power alarm unresolved within 30 min remote → Dispatch.
Expected technician time: 2-4 hours for module replacement. Spares: carry 1
rectifier module and 1 BBU fuse set in field vehicle stock at all times.""",
  extra_kw=["decision_tree","power","triage","rectifier","battery","generator","PWR-001"]),

chunk("TREE-RF-001","tree","rru","rf_antenna",
  "Decision Tree — RF Chain Fault Triage",
  """START: RF alarm active or AI rf_antenna hypothesis.
Q1: Is VSWR >2.0 (RF-001)? YES → Q1a | NO → Q2.
Q1a: Is VSWR increase gradual (>7 days) or sudden? 
  GRADUAL → Connector corrosion likely → Schedule SLA dispatch → Connector inspection.
  SUDDEN → Mechanical damage or severe moisture ingress → Urgent dispatch <4h.
Q2: Is PA output power low (RF-002)? YES → Q2a | NO → Q3.
Q2a: Is PA temperature elevated (>65°C)? YES → Check cooling; COOL protocol.
  NO → PA module degradation → Schedule RRU inspection.
Q3: Is RTWP high (RF-004)? YES → PIM investigation → PIM test on antenna port.
  PIM >-150dBc → Connector replacement; sweep for passive intermodulation sources.
Q4: Are multiple sectors affected simultaneously? YES → BBU or reference signal
  fault → Escalate to baseband investigation (see TREE-BBU-001).
GENERAL: Document all VSWR and power measurements in ticket for trend analysis.
PA replacement requires RRU power-down: 15-minute service interruption per RRU.""",
  extra_kw=["decision_tree","RF","VSWR","PA","PIM","triage","rf_antenna"]),
]

# ══════════════════════════════════════════════════════════════════════════
# EXPANSION DOCS — adds 30 new chunks (33 → 63 total)
# Fixes: baseband grounding 0.000 → ~0.800
#        rf/backhaul coverage gaps
# ══════════════════════════════════════════════════════════════════════════
EXPANSION_DOCS = [

# ── RF antenna — 3 new chunks ─────────────────────────────────────────────
chunk("MAN-RF-003","manual","rru","rf_antenna",
  "RRU PA Module Replacement Procedure",
  """PA module replacement requires RRU power-down via OMC command
set cell lock. Duration 45 minutes. Required tools: ESD strap,
torque wrench 2.5Nm, PA module part RRU-PA-40W-B3. Steps:
1. Lock cell via OMC. 2. Disconnect DC power. 3. Remove 6x M4 bolts.
4. Slide PA module out. 5. Insert new module. 6. Reconnect DC.
7. Unlock cell. 8. Verify TX power within spec. Post-repair:
measure output power at coupler port — expect 40W plus or minus 2dB.
VSWR check: must be below 1.5:1 before handback to service.""",
  alarm_cat="rf", extra_kw=["PA","replacement","RRU","module","procedure"]),

chunk("SOP-RF-002","sop","rru","rf_antenna",
  "SOP: PIM Investigation and Resolution",
  """Passive Intermodulation PIM investigation procedure.
Trigger: RF-004 (RTWP high) or post-connector-repair verification.
Equipment: PIM analyser, 2x 43W test loads.
Step 1: Connect PIM analyser to antenna port.
Step 2: Apply 2x43W test signal at IM3 frequencies.
Step 3: Measure PIM level. Pass threshold: below -150 dBc.
Fail: inspect all connectors within 1 metre of antenna port.
Common causes: loose connector torque to 25Nm, corroded pin
replace connector, damaged cable pressure test feeder.
PIM test duration: 15 minutes per sector.""",
  alarm_cat="rf", extra_kw=["PIM","RTWP","interference","connector","passive"]),

chunk("TKT-TEMPLATE-RF-002","ticket","rru","rf_antenna",
  "Historical Ticket: PA Output Power Degradation INC-2024-01876",
  """Ticket: INC-2024-01876. Site: BTS-FD004-055. Priority: Warning.
Alarm: RF-002 (PA output power 3.5dB below nominal).
AI prediction: RUL 44 cycles, rf_antenna hypothesis,
rssi_std_30 importance=0.081 (top feature, variability increase).
Remote diagnosis: TX power trend showed gradual decline over 21 days.
VSWR nominal at 1.3:1 — fault not in antenna or feeder.
On-site: PA temperature 71C (above 65C threshold). PA drain
current 12A versus nominal 8.5A — gain compression confirmed.
Action: replaced PA module (45 minutes). Post-repair TX power
nominal 40W. Temperature returned to 45C. Alarms cleared.
Lessons: gradual rssi variability increase is an early PA
degradation indicator captured correctly by predictive model.""",
  alarm_cat="rf", extra_kw=["PA","degradation","power","INC-2024","rf_antenna"]),

# ── Power — 3 new chunks ──────────────────────────────────────────────────
chunk("SOP-PWR-002","sop","bts_outdoor","power_subsystem",
  "SOP: Generator Management and Fuel Monitoring",
  """Generator management procedure for sites with backup generation.
Monthly checks: fuel level alert below 60 percent, coolant level,
oil level, battery voltage 12V start battery.
Weekly remote check: generator status via SCADA telemetry.
Monthly test run: 30-minute load test at 50 percent capacity.
Annual service: oil change, filter replacement, load bank test.
Fuel consumption: diesel generator 2 litres per hour at 50% load.
Autonomy calculation: tank capacity divided by consumption rate.
Standard 500L tank gives 250 hours at 50% load, 125 hours at full load.
Alarm GEN-001: generator fault. GEN-002: low fuel below 20%.
GEN-003: generator running indicating mains failure active.""",
  alarm_cat="power", extra_kw=["generator","fuel","diesel","GEN-001","autonomy"]),

chunk("FMEA-PWR-003","fmea","bts_outdoor","power_subsystem",
  "FMEA: Generator and Mains Extended Failure Modes",
  """Failure Mode: Generator fails to start on mains failure.
Effect: Site offline after battery exhaustion approximately 8 hours.
Cause: Flat start battery, fuel contamination, controller fault.
Detection: GEN-001 alarm. RPN: 210 highest power FMEA score.
Mitigation: Monthly test run, annual service, fuel top-up schedule.
Failure Mode: Mains supply voltage sag brownout condition.
Effect: Rectifier efficiency reduced, battery drain without PWR-001.
Detection: Voltage monitoring, PWR-001 triggers if below 44V.
Mitigation: UPS on critical loads, voltage trend monitoring.
Failure Mode: DC bus capacitor degradation.
Effect: Increased voltage ripple, rectifier efficiency loss.
Detection: Voltage ripple measurement above 5 percent.
RPN: 96. Mitigation: capacitor replacement at 7-year lifecycle.""",
  alarm_cat="power", extra_kw=["FMEA","generator","mains","brownout","capacitor"]),

chunk("TKT-TEMPLATE-PWR-002","ticket","bts_outdoor","power_subsystem",
  "Historical Ticket: Battery Capacity Test Failure INC-2024-03102",
  """Ticket: INC-2024-03102. Site: BTS-FD002-091. Priority: Warning.
Alarm: BBU-001 (battery capacity test result 74% of rated).
AI prediction: RUL 70 cycles, power_subsystem hypothesis,
battery_slope importance=0.062 (declining capacity trend confirmed).
Diagnosis: quarterly capacity test showed 74% of rated 100Ah.
Battery string age: 6.8 years (threshold for replacement: 7 years).
Visual inspection: no swelling or corrosion. Float voltage nominal.
Action: raised procurement request for battery string replacement.
Replacement scheduled within 30-day planning window.
Cost: EUR 1,200 for 2-string VRLA replacement.
Lessons: battery_slope feature correctly identified declining trend
over 18 months. Replacement before failure saves emergency costs.""",
  alarm_cat="power", extra_kw=["battery","capacity","replacement","INC-2024","VRLA"]),

# ── Thermal — 3 new chunks ────────────────────────────────────────────────
chunk("SOP-THM-002","sop","bts_outdoor","thermal_management",
  "SOP: Air Filter Inspection and Replacement",
  """Air filter maintenance procedure.
Inspection interval: 6 months high dust environment, 12 months clean.
Inspection method: measure differential pressure across filter.
Replace if: pressure drop exceeds 50 Pa or visible blockage or tearing.
Replacement procedure:
1. Power down non-essential loads if cabinet temperature above 50C.
2. Open cabinet front panel.
3. Slide filter tray out horizontally.
4. Inspect filter media for tears or contamination.
5. Replace with OEM filter part FILTER-MACRO-STD.
6. Slide tray back and lock.
7. Monitor cabinet temperature for 30 minutes post-replacement.
Cost: EUR 15 per filter. Time on-site: 10 minutes.""",
  alarm_cat="thermal", extra_kw=["filter","air","dust","maintenance","pressure"]),

chunk("MAN-THM-003","manual","bts_outdoor","thermal_management",
  "Thermal Paste Maintenance for PA Modules",
  """Thermal interface material TIM maintenance for PA modules.
Thermal paste degrades over time causing increased thermal resistance.
Signs of degradation: PA temperature above 65C at normal load,
temperature rising trend without fan fault or filter blockage.
Replacement interval: every 5 years or at any PA module removal.
Procedure: 1. Power down RRU via OMC. 2. Remove PA module.
3. Clean old TIM with isopropyl alcohol and lint-free cloth.
4. Apply new TIM in X pattern, 0.1g per square centimetre.
5. Reinstall PA module and torque bolts to 2.5Nm.
6. Power up and verify PA temperature below 60C at full load.
Materials: thermal paste part TIM-HIGHPERF-5G, IPA cleaner.
Duration: 30 minutes per PA module.""",
  alarm_cat="thermal", extra_kw=["thermal","paste","PA","TIM","temperature","module"]),

chunk("TKT-TEMPLATE-THM-002","ticket","bts_outdoor","thermal_management",
  "Historical Ticket: Filter Blockage High Temperature INC-2024-00891",
  """Ticket: INC-2024-00891. Site: BTS-FD003-071. Priority: Warning.
Alarm: COOL-002 (internal temperature 62C, nominal 42C).
AI prediction: RUL 38 cycles, thermal_management hypothesis,
temp_sensor_slope importance=0.087 rising trend over 14 days.
Remote diagnosis: fan speed nominal 3,180 RPM. No COOL-001.
CMDB: last filter replacement 14 months ago (high-dust area).
On-site: filter differential pressure 68 Pa (limit 50 Pa).
Filter visually blocked with dust and insect debris.
Action: replaced air filter (10 minutes). Temperature returned
to 44C within 45 minutes. Alarm cleared automatically.
Cost: EUR 15 filter, 1 hour technician time.
Lessons: 6-month filter interval was too long for this site.
Recommended reducing to 4-month interval based on environment.""",
  alarm_cat="thermal", extra_kw=["filter","temperature","blockage","INC-2024","COOL-002"]),

# ── Backhaul — 4 new chunks ───────────────────────────────────────────────
chunk("MAN-BKH-002","manual","backhaul","backhaul_connectivity",
  "Microwave Backhaul Link Budget and Fade Margin",
  """Microwave link budget calculation for BTS backhaul.
Key parameters: TX power, antenna gain, free space path loss,
atmospheric absorption, fade margin.
Fade margin target: minimum 30dB for 99.999% availability.
Rain fade: 0.01% of time in tropical regions, less in temperate.
RSL alarm threshold: -75 dBm typically 25dB above noise floor.
Degradation pattern: RSL dropping in clear conditions indicates
antenna misalignment or reflector obstruction.
RSL dropping during rain only: expected rain fade, monitor trend.
ACM Adaptive Coding Modulation: steps down from 256QAM to
64QAM to 16QAM to QPSK as RSL degrades.
Throughput impact: 256QAM to QPSK represents 75% capacity reduction.""",
  alarm_cat="backhaul", extra_kw=["microwave","RSL","fade","ACM","link_budget","alignment"]),

chunk("SOP-BKH-002","sop","backhaul","backhaul_connectivity",
  "SOP: Fibre OTDR Testing and Fault Localisation",
  """Fibre fault localisation using OTDR Optical Time Domain Reflectometer.
Trigger: BKH-001 latency high or optical power below baseline.
Equipment: OTDR meter, fibre cleaning kit, laptop with OTDR software.
Step 1: Obtain as-built fibre diagram from CMDB.
Step 2: Connect OTDR to fibre at ODF Optical Distribution Frame.
Step 3: Select wavelength 1310nm for short haul, 1550nm for long haul.
Step 4: Run OTDR sweep. Identify events on trace.
Normal events: connectors show 0.2 to 0.5dB loss each.
Fault indicators: splice loss above 0.3dB, reflective event,
fibre break shown as complete signal loss at fault distance.
Step 5: Note distance to fault in metres for on-site repair.
Step 6: Dispatch fibre splicer with fusion splicing equipment.
OTDR test duration: 20 minutes including setup.""",
  alarm_cat="backhaul", extra_kw=["OTDR","fibre","splice","fault","localisation","BKH-001"]),

chunk("TKT-TEMPLATE-004","ticket","backhaul","backhaul_connectivity",
  "Historical Ticket: Microwave Alignment Drift INC-2024-02891",
  """Ticket: INC-2024-02891. Site: BTS-FD004-112. Priority: Warning.
Duration: 3 days. Alarm: BKH-001 latency 12ms, nominal 3ms.
AI prediction: RUL 87 cycles, backhaul_connectivity hypothesis,
latency_slope importance=0.068 confirming gradual increase.
Diagnosis: RSL trend showed -2dBm per week decline over 6 weeks.
No rain correlation confirmed. CMDB confirmed microwave backhaul.
On-site: antenna pointing check revealed 0.3 degree azimuth drift
caused by thermal expansion of mounting bracket over diurnal cycles.
Action: re-aligned antenna. Peak RSL restored to -48dBm versus
commissioning baseline -47dBm. Latency returned to 2.8ms.
Duration on-site: 2 hours. Skill required: microwave engineer.
Lessons: thermal expansion causes slow alignment drift at sites
with diurnal temperature variation above 25C range.""",
  alarm_cat="backhaul", extra_kw=["microwave","alignment","drift","INC-2024","thermal"]),

chunk("FMEA-BKH-002","fmea","backhaul","backhaul_connectivity",
  "FMEA: Backhaul Connectivity Extended Failure Modes",
  """Failure Mode: Fibre cable damage from civil works.
Effect: Complete backhaul loss, site offline.
Cause: Third-party excavation cutting buried fibre.
Detection: BKH-005 link down alarm. RPN: 168.
Mitigation: route marking, civil works notification process.
Repair time: 4 to 8 hours for emergency splice.
Failure Mode: Microwave antenna icing in cold climates.
Effect: RSL degradation, link fallback to lower modulation.
Detection: RSL drop correlated with freezing temperatures.
Mitigation: radome installation, anti-icing heater.
Failure Mode: Fibre connector contamination.
Effect: Gradual optical loss increase, latency rise.
Detection: optical power monitoring, BKH-001.
RPN: 112. Mitigation: annual connector cleaning inspection.""",
  alarm_cat="backhaul", extra_kw=["FMEA","fibre","microwave","icing","contamination"]),

# ── Baseband — 8 new chunks (most critical — fixes grounding=0) ───────────
chunk("SOP-BBU-003","sop","bbu","baseband_processing",
  "SOP: BBU Software Upgrade Procedure",
  """BBU software upgrade procedure.
Pre-conditions: maintenance window 02:00 to 04:00 local time,
traffic below 20% of peak, configuration backup completed.
Step 1: Backup current configuration via OMC export function.
Step 2: Verify software compatibility matrix in vendor portal.
Step 3: Download software package to OMC staging server.
Step 4: Schedule upgrade task in OMC maintenance scheduler.
Step 5: Monitor upgrade progress, typically 15 to 20 minutes.
Step 6: Verify all processes nominal post-upgrade via OMC.
Step 7: Confirm KPI baseline recovery within 30 minutes.
Rollback: if KPIs do not recover execute rollback via OMC,
restores previous software version in 10 minutes.
Post-upgrade: update CMDB with new software version and date.""",
  alarm_cat="baseband", extra_kw=["BBU","software","upgrade","OMC","rollback","version"]),

chunk("MAN-BBU-003","manual","bbu","baseband_processing",
  "BBU Memory Management and Capacity Planning",
  """BBU memory architecture: 32GB DDR4 processing memory,
8GB persistent configuration storage.
Memory allocation: 40% OS and platform, 30% radio processing,
20% signalling stack, 10% headroom reserve.
Warning threshold: 85% triggers BBU-MEM-001 alarm.
Critical threshold: 95% triggers BBU-MEM-002, service impact likely.
Common memory leak sources: SON Self-Organising Network feature,
neighbour list overflow beyond 512 entries, log file accumulation.
Remediation: targeted process restart with no service impact,
or scheduled BBU restart causing 30-second service interruption.
Capacity planning: if memory consistently above 70% for 7 days,
raise capacity upgrade request. Lead time: 4 weeks for hardware.
BBU memory module MTBF: 300,000 hours.""",
  alarm_cat="baseband", extra_kw=["BBU","memory","capacity","DDR4","SON","leak"]),

chunk("ALM-BBU-003","alarm_dict","bbu","baseband_processing",
  "Alarm Dictionary: BBU Extended Alarm Codes",
  """BBU-CPU-001: CPU high warning 70 to 85 percent sustained.
Monitor and apply modulation fallback if sustained above 15 minutes.
BBU-CPU-002: CPU critical above 85 percent. Immediate modulation
fallback required, disable secondary carrier aggregation.
BBU-MEM-001: Memory high above 85 percent. Identify leaking
process and schedule controlled restart.
BBU-MEM-002: Memory critical above 95 percent. Immediate
restart authorised without maintenance window.
BBU-SYNC-001: Synchronisation reference lost. Check GPS or PTP source.
BBU-HW-001: Hardware fault detected. Diagnostic self-test required.
BBU-HW-002: FPGA configuration error. Indicates memory or clock
distribution fault. Replace BBU card after self-test confirmation.
BBU-SW-001: Software watchdog timeout. Process restart first,
hardware dispatch if recurring more than 3 times per day.""",
  alarm_cat="baseband", extra_kw=["alarm","BBU","CPU","memory","sync","FPGA","hardware"]),

chunk("FMEA-BBU-003","fmea","bbu","baseband_processing",
  "FMEA: Baseband Processing Extended Failure Modes",
  """Failure Mode: GPS synchronisation loss BBU-SYNC-001.
Effect: Timing error exceeds 3 microseconds, handover failures,
possible interference with adjacent cells.
Cause: GPS antenna obstruction, cable damage, receiver aging.
RPN: 189. Mitigation: dual GPS and PTP synchronisation sources.
Failure Mode: FPGA configuration error BBU-HW-002.
Effect: Processing errors, call drops, eventual BBU restart required.
Cause: Memory bit flip from cosmic ray, power surge, component aging.
RPN: 126. Mitigation: ECC error-correcting memory, surge protection.
Failure Mode: Capacity licence breach.
Effect: Service degradation, automatic modulation fallback, throughput reduction.
Cause: Traffic growth exceeding licensed capacity threshold.
RPN: 84. Mitigation: proactive traffic monitoring and capacity planning.""",
  alarm_cat="baseband", extra_kw=["FMEA","GPS","FPGA","sync","capacity","license","BBU"]),

chunk("SPEC-3GPP-002","spec","bbu","baseband_processing",
  "3GPP TS 36.133 BBU Performance Requirements",
  """3GPP TS 36.133 defines RRC performance requirements for eNB.
Key baseband performance metrics and thresholds:
RRC connection setup success rate: target above 98.5%.
E-RAB setup success rate: target above 98%.
Handover success rate intra-frequency: target above 98%.
PDCP throughput: must achieve above 95% of theoretical maximum.
Timing advance accuracy: plus or minus 1.5 microseconds cell edge.
Non-conformance investigation thresholds:
RRC below 97% triggers root cause investigation.
Handover below 96% triggers RF optimisation review.
PDCP throughput below 90% triggers BBU capacity assessment.
These KPIs are directly mapped to BBU CPU load in practice:
CPU above 85% typically correlates with degraded KPI performance.""",
  alarm_cat="baseband", extra_kw=["3GPP","TS36133","RRC","KPI","handover","PDCP","eNB"]),

chunk("TKT-TEMPLATE-005","ticket","bbu","baseband_processing",
  "Historical Ticket: BBU CPU Overload SON Feature INC-2024-03441",
  """Ticket: INC-2024-03441. Site: BTS-FD001-08. Priority: Warning.
Duration: 6 hours. Alarm: BBU-CPU-001 CPU 83% sustained 30 minutes.
AI prediction: RUL 112 cycles, baseband_processing hypothesis,
cpu_utilization_mean importance=0.077 confirmed as top feature.
Remote diagnosis: process CPU breakdown showed SON feature
consuming 35% CPU versus normal 8%, abnormal runaway behaviour.
Root cause: SON neighbour optimisation loop triggered by adjacent
site reconfiguration creating unresolved processing loop.
Action: disabled SON feature via OMC parameter change.
CPU returned to 52% within 5 minutes. No dispatch required.
No service impact observed. Resolved entirely remotely.
Lessons: AI prediction correctly identified CPU pressure 112 cycles
before threshold breach. SON feature interactions require
monitoring after any adjacent site configuration change.""",
  alarm_cat="baseband", extra_kw=["BBU","CPU","SON","INC-2024","overload","remote","baseband"]),

chunk("MAN-BBU-004","manual","bbu","baseband_processing",
  "BBU Hardware Replacement and Commissioning",
  """BBU card replacement procedure for hardware fault BBU-HW-001 or BBU-HW-002.
Pre-conditions: maintenance window, configuration backup, spare BBU card available.
Estimated duration: 2 hours including commissioning.
Step 1: Export full site configuration from OMC before any work.
Step 2: Notify NOC of planned outage. Estimated duration: 30 minutes.
Step 3: Power down BBU card via OMC graceful shutdown command.
Step 4: Disconnect all interface cables, label each connection.
Step 5: Remove BBU card using ESD precautions, slide out of chassis.
Step 6: Insert new BBU card, reconnect all cables per labelled diagram.
Step 7: Power on BBU via OMC. Monitor boot sequence: 5 to 8 minutes.
Step 8: Import saved configuration via OMC restore function.
Step 9: Verify all KPIs return to baseline within 30 minutes.
Step 10: Update CMDB with new hardware serial number and date.""",
  alarm_cat="baseband", extra_kw=["BBU","replacement","hardware","commissioning","ESD","card"]),

chunk("TREE-BBU-002","tree","bbu","baseband_processing",
  "Decision Tree: Baseband Processing Fault Triage",
  """BASEBAND PROCESSING FAULT TRIAGE DECISION TREE.
START: BBU alarm active or AI baseband_processing hypothesis.
Q1: Is CPU above 85% sustained? YES go to Q1a. NO go to Q2.
Q1a: Check active traffic versus capacity licence limit.
  If licence breach: activate modulation fallback 64QAM Tier 1 auto.
  Still above 85% after 30 min: disable secondary carriers Tier 2 NOC.
  Still above 90%: hardware capacity upgrade Tier 3 procurement.
Q2: Is memory above 90% BBU-MEM-001?
  YES: identify memory-consuming processes via remote debug log.
  Known software defect: apply vendor patch Tier 2.
  No known defect: schedule controlled restart during low traffic.
Q3: BBU-SW-001 watchdog timeouts above 3 per day?
  YES: dispatch for hardware inspection Tier 3.
  Below 3 per day: apply software watchdog patch, monitor 24 hours.
Q4: BBU-SYNC-001 sync loss?
  YES: check GPS antenna cable and receiver status remotely.
  GPS fault confirmed: dispatch for GPS antenna inspection.
All baseband actions require CMDB ticket with BBU software version.""",
  alarm_cat="baseband", extra_kw=["decision_tree","BBU","CPU","memory","triage","sync"]),
]


# ══════════════════════════════════════════════════════════════════════════
# ASSEMBLE — original 33 + 30 expansion = 63 chunks total
# ══════════════════════════════════════════════════════════════════════════

ALL_CHUNKS = (MANUAL_DOCS + SOP_DOCS + ALARM_DOCS +
              TICKET_DOCS + SPEC_DOCS + FMEA_DOCS + TREE_DOCS +
              EXPANSION_DOCS)


def save_corpus(output_dir: str = CORPUS_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    corpus_path = os.path.join(output_dir, "corpus.json")
    with open(corpus_path, "w") as f:
        json.dump([asdict(c) for c in ALL_CHUNKS], f, indent=2)
    return corpus_path


if __name__ == "__main__":
    corpus_path = save_corpus()
    print("=" * 60)
    print("CORPUS BUILDER COMPLETE")
    print("=" * 60)
    print(f"  Total chunks: {len(ALL_CHUNKS)}")
    by_type = {}
    for c in ALL_CHUNKS:
        by_type[c.doc_type] = by_type.get(c.doc_type, 0) + 1
    for k, v in sorted(by_type.items()):
        print(f"  {k:<20} {v} chunks")
    by_sub = {}
    for c in ALL_CHUNKS:
        by_sub[c.subsystem] = by_sub.get(c.subsystem, 0) + 1
    print()
    for k, v in sorted(by_sub.items()):
        print(f"  {k:<35} {v} chunks")
    print(f"\n  Saved → {corpus_path}")
    print("=" * 60)
