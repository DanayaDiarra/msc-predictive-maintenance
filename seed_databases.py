"""
seed_databases.py — OrchestrAI NOC
Creates and seeds three test SQLite databases:
  1. hr_database.db       — engineers, skills, on-call roster
  2. supply_chain.db      — spare parts inventory
  3. station_streams.db   — live telemetry / RUL readings per station

Run once:  python seed_databases.py
"""

import sqlite3
import random
import time
import math
from pathlib import Path
from datetime import datetime, timedelta

DB_DIR = Path(__file__).resolve().parent / "data" / "databases"
DB_DIR.mkdir(parents=True, exist_ok=True)

HR_DB = DB_DIR / "hr_database.db"
SC_DB = DB_DIR / "supply_chain.db"
ST_DB = DB_DIR / "station_streams.db"


# ══════════════════════════════════════════════════════════════════════════════
#  1. HR DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def seed_hr():
    conn = sqlite3.connect(str(HR_DB))
    cur  = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS engineers;
    DROP TABLE IF EXISTS engineer_certifications;
    DROP TABLE IF EXISTS shift_schedule;

    CREATE TABLE engineers (
        employee_id     TEXT PRIMARY KEY,
        full_name       TEXT NOT NULL,
        email           TEXT,
        phone           TEXT,
        location        TEXT,
        country         TEXT,
        skill_tags      TEXT,           -- JSON array e.g. '["power","thermal"]'
        specialisation  TEXT,           -- primary subsystem key
        level           TEXT,           -- Junior | Mid | Senior
        on_call         INTEGER DEFAULT 0,  -- 0/1
        shift           TEXT,           -- Day | Night | Rotating
        active          INTEGER DEFAULT 1,
        hire_date       TEXT,
        department      TEXT,
        manager_id      TEXT,
        dispatches_ytd  INTEGER DEFAULT 0,
        avg_resolution_h REAL DEFAULT 0.0
    );

    CREATE TABLE engineer_certifications (
        cert_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT,
        cert_name   TEXT,
        issued_date TEXT,
        expiry_date TEXT,
        FOREIGN KEY (employee_id) REFERENCES engineers(employee_id)
    );

    CREATE TABLE shift_schedule (
        sched_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT,
        shift_date  TEXT,
        shift_type  TEXT,
        on_call     INTEGER,
        FOREIGN KEY (employee_id) REFERENCES engineers(employee_id)
    );
    """)

    engineers = [
        # (employee_id, full_name, email, phone, location, country, skill_tags, specialisation, level, on_call, shift, hire_date, department, manager_id, dispatches_ytd, avg_resolution_h)
        ("ENG001","Awa Diallo",       "a.diallo@noc.sn",    "+221 77 543 2101","Dakar",        "Senegal",      '["power_subsystem","thermal_management"]',  "power_subsystem",       "Senior",1,"Day",     "2019-03-15","Field Operations","MGR001",47,3.2),
        ("ENG002","Mamadou Koné",     "m.kone@noc.ml",      "+223 65 801 4422","Bamako",       "Mali",         '["thermal_management","power_subsystem"]',  "thermal_management",    "Senior",1,"Day",     "2018-07-22","Field Operations","MGR001",52,2.9),
        ("ENG003","Fatou Sow",        "f.sow@noc.sn",       "+221 76 312 8853","Dakar",        "Senegal",      '["rf_antenna","backhaul_connectivity"]',     "rf_antenna",            "Senior",0,"Night",   "2020-01-10","Field Operations","MGR001",38,3.8),
        ("ENG004","Ibrahim Traoré",   "i.traore@noc.ml",    "+223 79 204 6637","Bamako",       "Mali",         '["backhaul_connectivity","rf_antenna"]',     "backhaul_connectivity", "Senior",1,"Day",     "2017-11-05","Field Operations","MGR002",61,2.6),
        ("ENG005","Aminata Bah",      "a.bah@noc.sn",       "+221 78 901 3364","Conakry",      "Guinea",       '["baseband_processing","rf_antenna"]',       "baseband_processing",   "Senior",0,"Night",   "2021-04-18","Field Operations","MGR002",29,4.1),
        ("ENG006","Oumar Ndiaye",     "o.ndiaye@noc.sn",    "+221 77 654 0915","Saint-Louis",  "Senegal",      '["power_subsystem"]',                       "power_subsystem",       "Mid",  1,"Day",     "2022-09-01","Field Operations","MGR001",18,3.9),
        ("ENG007","Kadiatou Barry",   "k.barry@noc.ml",     "+223 66 412 7780","Bamako",       "Mali",         '["thermal_management"]',                    "thermal_management",    "Mid",  1,"Day",     "2022-02-14","Field Operations","MGR001",22,4.2),
        ("ENG008","Seydou Coulibaly", "s.coulibaly@noc.ml", "+223 70 823 5591","Bobo-Dioulasso","Burkina Faso",'["rf_antenna","thermal_management"]',        "rf_antenna",            "Mid",  0,"Night",   "2023-05-20","Field Operations","MGR002",11,4.5),
        ("ENG009","Mariam Keita",     "m.keita@noc.sn",     "+221 76 234 6102","Thiès",        "Senegal",      '["backhaul_connectivity"]',                 "backhaul_connectivity", "Mid",  1,"Day",     "2022-11-30","Field Operations","MGR002",16,3.7),
        ("ENG010","Boubacar Diop",    "b.diop@noc.sn",      "+221 78 567 3243","Dakar",        "Senegal",      '["baseband_processing","power_subsystem"]',  "baseband_processing",   "Junior",1,"Day",   "2024-01-15","Field Operations","MGR001",8, 5.1),
        ("ENG011","Rokhaya Fall",     "r.fall@noc.sn",      "+221 77 890 1154","Dakar",        "Senegal",      '["power_subsystem"]',                       "power_subsystem",       "Junior",0,"Night",  "2024-03-01","Field Operations","MGR001",5, 5.8),
        ("ENG012","Alpha Baldé",      "a.balde@noc.ml",     "+223 63 345 9865","Bamako",       "Mali",         '["rf_antenna"]',                            "rf_antenna",            "Junior",1,"Day",    "2024-06-10","Field Operations","MGR002",4, 6.2),
        ("MGR001","Cheikh Diarra",    "c.diarra@noc.sn",    "+221 77 100 0001","Dakar",        "Senegal",      '["management","power_subsystem"]',           "management",            "Senior",1,"Day",   "2015-01-01","Management",      None,   0, 0.0),
        ("MGR002","Aïssata Coulibaly","ai.coulibaly@noc.ml","+223 65 200 0002","Bamako",       "Mali",         '["management","backhaul_connectivity"]',     "management",            "Senior",1,"Day",   "2016-06-01","Management",      None,   0, 0.0),
    ]

    cur.executemany("""INSERT OR REPLACE INTO engineers
        (employee_id,full_name,email,phone,location,country,skill_tags,specialisation,
         level,on_call,shift,active,hire_date,department,manager_id,dispatches_ytd,avg_resolution_h)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?,?,?,?,?)""", engineers)

    # Certifications
    certs = [
        ("ENG001","Ericsson Certified Field Engineer","2021-03-10","2026-03-10"),
        ("ENG001","High Voltage Safety","2023-01-15","2025-01-15"),
        ("ENG002","Nokia NetAct Specialist","2020-07-01","2025-07-01"),
        ("ENG003","PIM Measurement Certified","2022-05-20","2027-05-20"),
        ("ENG004","ITU-T G.826 Compliance","2021-11-01","2026-11-01"),
        ("ENG005","BBU Software Upgrade — Huawei","2022-09-15","2027-09-15"),
    ]
    cur.executemany("INSERT INTO engineer_certifications (employee_id,cert_name,issued_date,expiry_date) VALUES (?,?,?,?)", certs)

    # Shift schedule (next 7 days)
    today = datetime.now()
    for eng_id, on_call_days in [
        ("ENG001",[0,1,2,3,4]),("ENG002",[0,1,2,3,4]),("ENG006",[0,2,4]),
        ("ENG007",[1,3,5]),("ENG009",[0,1,4,6]),("ENG010",[0,1,2,3,4,5,6]),
    ]:
        for d in range(7):
            date_str = (today + timedelta(days=d)).strftime("%Y-%m-%d")
            cur.execute("INSERT INTO shift_schedule (employee_id,shift_date,shift_type,on_call) VALUES (?,?,?,?)",
                (eng_id, date_str, "Day", 1 if d in on_call_days else 0))

    conn.commit(); conn.close()
    print(f"✓ HR database seeded: {HR_DB}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. SUPPLY CHAIN DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def seed_supply_chain():
    conn = sqlite3.connect(str(SC_DB))
    cur  = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS spare_parts;
    DROP TABLE IF EXISTS warehouses;
    DROP TABLE IF EXISTS purchase_orders;

    CREATE TABLE warehouses (
        warehouse_id   TEXT PRIMARY KEY,
        name           TEXT,
        location       TEXT,
        country        TEXT,
        latitude       REAL,
        longitude      REAL,
        contact_phone  TEXT
    );

    CREATE TABLE spare_parts (
        part_id              TEXT PRIMARY KEY,
        part_name            TEXT NOT NULL,
        part_category        TEXT,               -- power | thermal | rf | backhaul | baseband
        subsystem_tag        TEXT,               -- maps to station subsystem key
        manufacturer         TEXT,
        model_ref            TEXT,
        quantity_available   INTEGER DEFAULT 0,
        quantity_reserved    INTEGER DEFAULT 0,
        unit_cost_eur        REAL,
        warehouse_id         TEXT,
        warehouse_location   TEXT,
        lead_time_hours      INTEGER,            -- hours to get to field
        reorder_level        INTEGER DEFAULT 2,
        last_replenished     TEXT,
        compatible_models    TEXT,               -- JSON array
        FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
    );

    CREATE TABLE purchase_orders (
        po_id          TEXT PRIMARY KEY,
        part_id        TEXT,
        quantity       INTEGER,
        ordered_date   TEXT,
        expected_date  TEXT,
        status         TEXT,
        supplier       TEXT,
        FOREIGN KEY (part_id) REFERENCES spare_parts(part_id)
    );
    """)

    warehouses = [
        ("WH-DKR-01","Dakar Main Warehouse","Dakar","Senegal",      14.6937,-17.4441,"+221 33 800 1001"),
        ("WH-DKR-02","Dakar West Warehouse","Pikine","Senegal",      14.7167,-17.4677,"+221 33 800 1002"),
        ("WH-BKO-01","Bamako Central","Bamako","Mali",               12.6392,-8.0029, "+223 20 220 1001"),
        ("WH-ABJ-01","Abidjan Hub","Abidjan","Côte d'Ivoire",        5.3599,-4.0083,  "+225 27 200 1001"),
        ("WH-OUG-01","Ouagadougou Store","Ouagadougou","Burkina Faso",12.3641,-1.5333,"+226 25 300 1001"),
    ]
    cur.executemany("INSERT OR REPLACE INTO warehouses VALUES (?,?,?,?,?,?,?)", warehouses)

    parts = [
        # (part_id, part_name, part_category, subsystem_tag, manufacturer, model_ref, qty_avail, qty_reserved, unit_cost_eur, warehouse_id, warehouse_location, lead_time_h, reorder_level, last_replenished, compatible_models)
        ("FAN-450CFM-V2","BTS Cooling Fan 450 CFM","thermal","thermal_management","Delta Electronics","AFB1212VHE",14,2,380.00,"WH-DKR-01","Dakar Main WH",4,3,"2025-11-01",'["Ericsson RBS6601","Nokia AirScale","Huawei BTS3900"]'),
        ("FAN-BEARING-KIT","Fan Bearing Replacement Kit","thermal","thermal_management","NSK","6205-2RS",28,0,45.00,"WH-DKR-01","Dakar Main WH",2,5,"2025-10-15",'["All BTS models"]'),
        ("AIR-FILTER-BTS","Cabinet Air Filter (standard)","thermal","thermal_management","Generic","AF-250-BTS",55,3,22.00,"WH-DKR-02","Dakar West WH",2,10,"2025-12-01",'["All BTS models"]'),
        ("HEAT-EXCHANGER-A","Heat Exchanger Unit — Model A","thermal","thermal_management","Rittal","SK 3276","6",0,890.00,"WH-DKR-01","Dakar Main WH",24,1,"2025-09-01",'["Ericsson RBS6601","Nokia FSMF"]'),
        ("RECT-48V-5A","Rectifier Module 48V / 5A","power","power_subsystem","Eltek","Flatpack2 48V/5A",6,1,620.00,"WH-BKO-01","Bamako Central",8,2,"2025-10-20",'["Ericsson RBS","Nokia BTS","Huawei BTS3900"]'),
        ("RECT-48V-10A","Rectifier Module 48V / 10A","power","power_subsystem","Emerson","NetSure 48V/10A",4,0,980.00,"WH-DKR-01","Dakar Main WH",12,2,"2025-09-15",'["Huawei BTS3900A","ZTE ZXSDR"]'),
        ("BATTERY-VRLA-100","VRLA Battery 48V 100Ah","power","power_subsystem","Enersys","PowerSafe 100",8,2,1250.00,"WH-DKR-02","Dakar West WH",24,2,"2025-08-01",'["All BTS models with DC backup"]'),
        ("BBU-FUSE-SET","BBU Fuse Assortment Set (10pc)","power","power_subsystem","Littelfuse","KLDR Series",55,0,18.00,"WH-DKR-01","Dakar Main WH",1,10,"2025-12-10",'["All models"]'),
        ("MCB-63A-4P","MCB 63A 4-Pole (DC)","power","power_subsystem","Schneider","Acti9 iC60N",12,0,145.00,"WH-DKR-01","Dakar Main WH",4,3,"2025-11-15",'["Ericsson Power Cabinet","Generic DC dist"]'),
        ("DIN-716-M-KIT","7/16 DIN Male Connector Kit x5","rf","rf_antenna","Commscope","716-JMFP",42,5,85.00,"WH-DKR-01","Dakar Main WH",2,8,"2025-11-20",'["All feeder cables"]'),
        ("DIN-716-F-KIT","7/16 DIN Female Connector Kit x5","rf","rf_antenna","Commscope","716-JFFP",38,3,85.00,"WH-DKR-01","Dakar Main WH",2,8,"2025-11-20",'["All feeder cables"]'),
        ("COAX-FOAM-50M","Coaxial Cable LCF 1/2\" 50m","rf","rf_antenna","Andrew","LCF12-50J",10,0,320.00,"WH-DKR-02","Dakar West WH",8,2,"2025-10-01",'["Standard BTS feeder runs"]'),
        ("AMAL-TAPE-5M","Self-Amalgamating Tape 5m","rf","rf_antenna","3M","23-Series",80,0,12.00,"WH-DKR-01","Dakar Main WH",1,15,"2025-12-05",'["All outdoor RF connections"]'),
        ("SFP-1310-SM","SFP+ 1310nm Single-mode 10G","backhaul","backhaul_connectivity","Finisar","FTLX1471D3BCL",14,2,145.00,"WH-BKO-01","Bamako Central",12,3,"2025-10-10",'["Cisco","Ericsson TN","Nokia RTN"]'),
        ("SFP-850-MM","SFP+ 850nm Multi-mode 10G","backhaul","backhaul_connectivity","Finisar","FTLX8574D3BCL",8,0,95.00,"WH-BKO-01","Bamako Central",12,3,"2025-10-10",'["Short-haul backhaul"]'),
        ("MW-ODU-18G","Microwave ODU 18GHz (spare)","backhaul","backhaul_connectivity","Ericsson","MINI-LINK CN 510","3",0,4200.00,"WH-DKR-01","Dakar Main WH",48,1,"2025-07-01",'["Ericsson MINI-LINK"]'),
        ("OTDR-ADAPTER","OTDR Adapter Kit (SC/LC/FC)","backhaul","backhaul_connectivity","Fluke","FiberInspector",5,0,280.00,"WH-DKR-01","Dakar Main WH",4,1,"2025-09-20",'["Test equipment — all fibre"]'),
        ("BBU-DDR4-16G","BBU DDR4 RAM Module 16GB","baseband","baseband_processing","Samsung","M393A2K43CB2",6,1,380.00,"WH-DKR-01","Dakar Main WH",24,1,"2025-08-15",'["Ericsson Baseband 6630","Nokia AirScale BBU"]'),
        ("BBU-BLADE-ERX","eRAN Processing Blade (spare)","baseband","baseband_processing","Ericsson","KRC 161 431/1","2",0,8500.00,"WH-DKR-01","Dakar Main WH",48,1,"2025-06-01",'["Ericsson RBS6601 BBU"]'),
        ("PATCH-CAT6-3M","CAT6 Patch Cable 3m (blue)","baseband","baseband_processing","Belden","REVConnect",120,0,8.00,"WH-DKR-01","Dakar Main WH",1,20,"2025-12-01",'["All BBU connections"]'),
    ]
    cur.executemany("""INSERT OR REPLACE INTO spare_parts
        (part_id,part_name,part_category,subsystem_tag,manufacturer,model_ref,
         quantity_available,quantity_reserved,unit_cost_eur,warehouse_id,warehouse_location,
         lead_time_hours,reorder_level,last_replenished,compatible_models) VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", parts)

    # A few open purchase orders
    pos = [
        ("PO-2025-0142","FAN-450CFM-V2",10,"2025-12-01","2026-01-15","In Transit","Delta Electronics"),
        ("PO-2025-0143","RECT-48V-5A",   5,"2025-12-10","2026-02-01","Ordered",   "Eltek West Africa"),
        ("PO-2025-0144","BATTERY-VRLA-100",4,"2025-11-20","2026-01-10","In Transit","Enersys Dakar"),
    ]
    cur.executemany("INSERT OR REPLACE INTO purchase_orders VALUES (?,?,?,?,?,?,?)", pos)

    conn.commit(); conn.close()
    print(f"✓ Supply Chain database seeded: {SC_DB}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. STATION STREAMS DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def seed_station_streams():
    conn = sqlite3.connect(str(ST_DB))
    cur  = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS station_telemetry;
    DROP TABLE IF EXISTS station_metadata;
    DROP TABLE IF EXISTS rul_predictions;

    CREATE TABLE station_metadata (
        station_id     TEXT PRIMARY KEY,
        subset         TEXT,
        subsystem      TEXT,
        city           TEXT,
        country        TEXT,
        latitude       REAL,
        longitude      REAL,
        installed_date TEXT,
        last_pm_date   TEXT,
        model_type     TEXT,
        sla_hours      INTEGER
    );

    CREATE TABLE station_telemetry (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id     TEXT,
        ts             TEXT,
        kpi_name       TEXT,
        kpi_value      REAL,
        unit           TEXT,
        quality        TEXT DEFAULT 'good'
    );

    CREATE TABLE rul_predictions (
        pred_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id     TEXT,
        ts             TEXT,
        predicted_rul  REAL,
        ci_low         REAL,
        ci_high        REAL,
        confidence     REAL,
        urgency        TEXT,
        model_version  TEXT DEFAULT 'Phase2_EnsemblePlusBC'
    );

    CREATE INDEX idx_tel_station ON station_telemetry (station_id, ts);
    CREATE INDEX idx_rul_station ON rul_predictions    (station_id, ts);
    """)

    stations_meta = [
        ("FD002_47",  "FD002","power_subsystem",       "Dakar",          "Senegal",      14.6937,-17.4441,"2018-03-01","2025-09-15","Ericsson RBS6601",4),
        ("FD003_88",  "FD003","thermal_management",    "Pikine",         "Senegal",      14.7167,-17.4677,"2019-07-12","2025-08-20","Nokia AirScale",4),
        ("FD001_23",  "FD001","thermal_management",    "Ziguinchor",     "Senegal",      12.3647,-15.5568,"2020-01-05","2025-10-01","Huawei BTS3900",48),
        ("FD004_55",  "FD004","rf_antenna",            "Touba",          "Senegal",      15.5536,-14.2692,"2017-11-20","2025-07-30","Ericsson RBS6601",48),
        ("FD004_112", "FD004","backhaul_connectivity", "Bamako",         "Mali",         12.6392,-8.0029, "2019-05-10","2025-09-01","Nokia AirScale",168),
        ("FD003_71",  "FD003","rf_antenna",            "Kayes",          "Mali",         14.7645,-10.9734,"2021-02-28","2025-11-10","Huawei BTS3900",168),
        ("FD001_08",  "FD001","baseband_processing",   "Tambacounda",    "Senegal",      13.4531,-13.3543,"2020-08-15","2025-10-15","Ericsson RBS6601",168),
        ("FD002_91",  "FD002","power_subsystem",       "Ouagadougou",    "Burkina Faso", 12.3641,-1.5333, "2018-12-01","2025-08-05","Nokia AirScale",168),
        ("FD004_203", "FD004","backhaul_connectivity", "Bissau",         "Guinea-Bissau",11.8658,-15.5977,"2022-03-20","2025-09-25","Huawei BTS3900",168),
        ("FD001_77",  "FD001","baseband_processing",   "Conakry",        "Guinea",        9.5370,-13.6773,"2019-06-30","2025-11-20","Ericsson RBS6601",168),
        ("FD002_14",  "FD002","power_subsystem",       "Saint-Louis",    "Senegal",      16.0544,-16.7190,"2017-09-01","2025-09-10","Nokia AirScale",4),
        ("FD001_44",  "FD001","rf_antenna",            "Thiès",          "Senegal",      14.3421,-16.0540,"2020-04-12","2025-10-20","Huawei BTS3900",48),
        ("FD003_55",  "FD003","thermal_management",    "Bobo-Dioulasso", "Burkina Faso", 13.5317,-2.1175, "2021-08-05","2025-11-05","Ericsson RBS6601",48),
        ("FD004_78",  "FD004","baseband_processing",   "Abidjan",        "Côte d'Ivoire", 5.3599,-4.0083, "2022-01-15","2025-12-01","Nokia AirScale",168),
        ("FD002_33",  "FD002","backhaul_connectivity", "Bamako-Nord",    "Mali",         12.6437,-8.0024, "2020-10-10","2025-08-28","Huawei BTS3900",168),
    ]
    cur.executemany("INSERT OR REPLACE INTO station_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?)", stations_meta)

    # Telemetry: generate 48h of readings at 15min intervals for each station
    rng = random.Random(42)
    kpi_profiles = {
        "power_subsystem":       [("dc_voltage_v",47.5,"V","low"),("battery_capacity_pct",84.0,"%","low"),("current_a",12.5,"A","normal")],
        "thermal_management":    [("cabinet_temp_c",38.0,"C","high"),("fan_speed_rpm",3200.0,"RPM","low"),("inlet_temp_c",30.0,"C","high")],
        "rf_antenna":            [("vswr",1.82,":1","high"),("rssi_dbm",-67.0,"dBm","low"),("pa_efficiency_pct",78.5,"%","low")],
        "backhaul_connectivity": [("latency_ms",6.2,"ms","high"),("esr_pct",0.8,"%","high"),("fade_margin_db",14.8,"dB","low")],
        "baseband_processing":   [("cpu_util_pct",71.0,"%","high"),("mem_swap_pct",68.0,"%","high"),("health_index_pct",62.0,"%","low")],
    }
    base_ruls = {
        "FD002_47":14.7,"FD003_88":18.1,"FD001_23":38.2,"FD004_55":44.0,"FD004_112":87.5,
        "FD003_71":55.1,"FD001_08":112.4,"FD002_91":70.3,"FD004_203":95.0,"FD001_77":119.0,
        "FD002_14":11.2,"FD001_44":33.8,"FD003_55":27.7,"FD004_78":72.6,"FD002_33":105.2,
    }
    degrade_rates = {
        "FD002_47":0.55,"FD003_88":0.60,"FD001_23":0.22,"FD004_55":0.18,"FD004_112":0.07,
        "FD003_71":0.05,"FD001_08":0.04,"FD002_91":0.04,"FD004_203":0.03,"FD001_77":0.02,
        "FD002_14":0.65,"FD001_44":0.20,"FD003_55":0.28,"FD004_78":0.06,"FD002_33":0.03,
    }

    now = time.time()
    tel_rows = []
    rul_rows = []

    for s_id, subset, subsystem, *_ in stations_meta:
        profiles = kpi_profiles.get(subsystem, kpi_profiles["power_subsystem"])
        base_rul = base_ruls.get(s_id, 50.0)
        degrade  = degrade_rates.get(s_id, 0.1)

        for step in range(192):  # 48h × 4 readings/h
            ts_epoch = now - (191 - step) * 900  # 15min intervals
            ts_str   = datetime.fromtimestamp(ts_epoch).strftime("%Y-%m-%dT%H:%M:%S")
            hours_ago = (191 - step) * 0.25

            for kpi_name, nominal, unit, fail_dir in profiles:
                d = -1 if fail_dir == "low" else 1 if fail_dir == "high" else 0
                drift = d * hours_ago * abs(nominal) * 0.0008 * degrade / 0.1
                noise = rng.gauss(0, abs(nominal) * 0.015)
                val   = round(nominal + drift + noise, 3)
                tel_rows.append((s_id, ts_str, kpi_name, val, unit, "good"))

            # RUL prediction at each step
            rul_val = max(0.5, base_rul - hours_ago * degrade / 60.0)
            conf    = 0.88 - degrade * 0.1
            ci_low  = max(0.1, rul_val - rul_val * 0.2206)
            ci_high = rul_val + rul_val * 0.2206
            urgency = "Critical" if rul_val <= 20 else "Warning" if rul_val <= 50 else "Monitor"
            rul_rows.append((s_id, ts_str, round(rul_val,2), round(ci_low,2), round(ci_high,2), round(conf,3), urgency))

    cur.executemany("INSERT INTO station_telemetry (station_id,ts,kpi_name,kpi_value,unit,quality) VALUES (?,?,?,?,?,?)", tel_rows)
    cur.executemany("INSERT INTO rul_predictions (station_id,ts,predicted_rul,ci_low,ci_high,confidence,urgency) VALUES (?,?,?,?,?,?,?)", rul_rows)

    conn.commit(); conn.close()
    print(f"✓ Station Streams database seeded: {ST_DB}")
    print(f"  • Telemetry rows: {len(tel_rows)}")
    print(f"  • RUL prediction rows: {len(rul_rows)}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Seeding OrchestrAI test databases…\n")
    seed_hr()
    seed_supply_chain()
    seed_station_streams()
    print(f"\n✅ All databases created in: {DB_DIR}")
    print("\nConnection strings for Settings → 🔗 HR & Supply DB:")
    print(f"  HR DB      → SQLite · path: data/databases/hr_database.db")
    print(f"  Supply DB  → SQLite · path: data/databases/supply_chain.db")
    print(f"  Streams DB → SQLite · path: data/databases/station_streams.db")
