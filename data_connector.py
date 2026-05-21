"""
FluxAgent — data_connector.py
Real-Time BTS Sensor Ingestion Layer
======================================
Danaya Diarra | MSc Thesis 2026 | GSOM SPBU

ARCHITECTURE
────────────
Production BTS stations expose KPIs through vendor management systems:

  Ericsson OSS-RC / ENM ──┐
  Nokia NetAct            ──┤──► NMS Northbound API
  Huawei U2020 / U2000    ──┘         │
                                      ▼
                              DataConnector  (this file)
                              ├── collect()        pull KPIs every 60s
                              ├── push_to_store()  write to InfluxDB + CSV
                              └── get_latest()     read back for dashboard

DEPLOYMENT MODES  (set FLUXAGENT_CONNECTOR_MODE in environment)
───────────────────────────────────────────────────────────────
  simulation  Default. Synthetic degradation, zero infra.
  file        NMS exports CSV every 60s to SENSOR_CSV_DIR.
  rest        Poll vendor REST/NETCONF endpoints directly.
  mqtt        Subscribe to station MQTT broker.

SENSOR → FEATURE MAPPING  (same as 2_feature_engineering_pipeline.py)
───────────────────────────────────────────────────────────────────────
  dc_voltage_v        → voltage_rolling_mean
  cabinet_temp_c      → temp_sensor_slope
  rssi_dbm            → rssi_std_30
  latency_ms          → latency_slope
  cpu_util_pct        → cpu_utilization_mean
"""

import os
import time
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

try:
    import pandas as pd
    PD_OK = True
except ImportError:
    PD_OK = False

# ── Configuration ──────────────────────────────────────────────────────────────
MODE            = os.environ.get("FLUXAGENT_CONNECTOR_MODE", "simulation")
SENSOR_CSV_DIR  = Path(os.environ.get("SENSOR_CSV_DIR",  "data/live_feed"))
INFLUX_URL      = os.environ.get("INFLUX_URL",           "http://localhost:8086")
INFLUX_TOKEN    = os.environ.get("INFLUX_TOKEN",         "")
INFLUX_ORG      = os.environ.get("INFLUX_ORG",           "fluxagent")
INFLUX_BUCKET   = os.environ.get("INFLUX_BUCKET",        "bts_sensors")
NMS_REST_BASE   = os.environ.get("NMS_REST_BASE",        "http://nms.example.com/api/v1")
NMS_API_KEY     = os.environ.get("NMS_API_KEY",          "")
MQTT_BROKER     = os.environ.get("MQTT_BROKER",          "localhost")
MQTT_PORT       = int(os.environ.get("MQTT_PORT",        "1883"))
POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_S",  "60"))
STORE_DIR       = Path("data/live_store")
STORE_DIR.mkdir(parents=True, exist_ok=True)
SENSOR_CSV_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FluxAgent] %(levelname)s  %(message)s")
log = logging.getLogger("FluxAgent.Connector")

# ── Sensor schema: subsystem → KPI columns ────────────────────────────────────
SENSOR_SCHEMA = {
    "power_subsystem":       ["dc_voltage_v","battery_cap_pct","rectifier_current_a","power_draw_w"],
    "thermal_management":    ["cabinet_temp_c","fan_speed_rpm","ambient_temp_c"],
    "rf_antenna":            ["rssi_dbm","vswr_ratio","sinr_db","pa_efficiency_pct"],
    "backhaul_connectivity": ["latency_ms","packet_loss_pct","throughput_mbps","link_util_pct"],
    "baseband_processing":   ["cpu_util_pct","mem_util_pct","active_users"],
}

# ── Thread-safe in-memory store ────────────────────────────────────────────────
_lock    = threading.Lock()
_latest  = {}   # {station_id: reading_dict}
_history = {}   # {station_id: [list of reading_dicts, max 200]}
_MAX_H   = 200

# Import centralized station configuration
from config.stations import STATION_NOMINALS as _NOMINALS


# ══════════════════════════════════════════════════════════════════════════════
#  BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

class SimulationBackend:
    """Synthetic degradation — zero infra required."""
    def __init__(self):
        self._t0 = time.time()

    def collect_all(self):
        elapsed = (time.time() - self._t0) / 60.0
        out = {}
        for sid, meta in _NOMINALS.items():
            rng  = np.random.default_rng(int(time.time() / POLL_INTERVAL_S) + abs(hash(sid)) % 9999)
            sub  = meta["sub"]
            d    = meta["degrade"]
            reading = {"ts": datetime.now(timezone.utc).isoformat()}
            for col in SENSOR_SCHEMA.get(sub, []):
                nom   = meta.get(col, 50.0)
                drift = meta["dir"] * d * elapsed * abs(nom) * 0.002
                noise = rng.normal(0, abs(nom) * 0.015)
                reading[col] = round(nom + drift + noise, 3)
            out[sid] = reading
        return out


class FileBackend:
    """
    Watches SENSOR_CSV_DIR for CSV files dropped by the NMS scheduler.

    CSV format (one row per KPI reading):
        station_id, timestamp, kpi_name, kpi_value
        FD002_47, 2026-04-01T10:00:00Z, dc_voltage_v, 47.02

    Configure your NMS to export PM counters on a 1-minute schedule to
    SENSOR_CSV_DIR.  See your vendor's PM Northbound Export guide.
    """
    def collect_all(self):
        if not PD_OK:
            log.warning("pandas not available — FileBackend unavailable")
            return {}
        out = {}
        files = sorted(SENSOR_CSV_DIR.glob("*.csv"))
        if not files:
            log.warning(f"FileBackend: no CSV files in {SENSOR_CSV_DIR}")
            return {}
        latest_file = files[-1]
        try:
            df = pd.read_csv(latest_file, names=["station_id","ts","kpi","value"])
            for sid, grp in df.groupby("station_id"):
                reading = {"ts": grp["ts"].iloc[-1]}
                for _, row in grp.iterrows():
                    reading[row["kpi"]] = float(row["value"])
                out[sid] = reading
            log.info(f"FileBackend: {len(out)} stations from {latest_file.name}")
        except Exception as e:
            log.error(f"FileBackend: {e}")
        return out


class RestBackend:
    """
    Polls the NMS REST API (Ericsson ENM / Nokia NetAct / Huawei U2020).

    Configure NMS_REST_BASE and NMS_API_KEY.
    Endpoint format is vendor-specific — adapt the URL and response parser below.

    Ericsson ENM:   GET /pm/counters?station={sid}&granularity=MINUTE_15
    Nokia NetAct:   GET /monitoring/kpi?dn=MRBTS-{sid}&period=latest
    Huawei U2020:   POST /performance/queryKPI  body: {"neId": sid, ...}
    """
    def collect_all(self):
        import urllib.request
        out = {}
        for sid in _NOMINALS:
            try:
                url = f"{NMS_REST_BASE}/pm/counters?station={sid}&granularity=MINUTE_15"
                req = urllib.request.Request(
                    url, headers={"Authorization": f"Bearer {NMS_API_KEY}", "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data    = json.loads(resp.read())
                    reading = {"ts": datetime.now(timezone.utc).isoformat()}
                    reading.update({k: float(v) for k, v in data.get("counters", {}).items()})
                    out[sid] = reading
            except Exception as e:
                log.debug(f"RestBackend {sid}: {e}")
        if not out:
            log.warning("RestBackend: no data — falling back to simulation")
            return SimulationBackend().collect_all()
        return out


class MqttBackend:
    """
    Subscribes to station MQTT topics.
    Topic pattern:  fluxagent/bts/{station_id}/{kpi_name}
    Payload:        {"value": 47.02, "ts": "2026-04-01T10:00:00Z"}

    Requires:  pip install paho-mqtt

    Test without hardware:
      mosquitto_pub -h localhost -t "fluxagent/bts/FD002_47/dc_voltage_v" \
                    -m '{"value": 47.02}'
    """
    def __init__(self):
        self._buf = {}
        try:
            import paho.mqtt.client as mqtt
            c = mqtt.Client()
            c.on_message = self._on_msg
            c.connect(MQTT_BROKER, MQTT_PORT, 60)
            c.subscribe("fluxagent/bts/#")
            c.loop_start()
            log.info(f"MqttBackend: connected {MQTT_BROKER}:{MQTT_PORT}")
        except ImportError:
            log.warning("paho-mqtt missing — pip install paho-mqtt")
        except Exception as e:
            log.error(f"MqttBackend: {e}")

    def _on_msg(self, client, userdata, msg):
        try:
            parts = msg.topic.split("/")
            if len(parts) == 4:
                sid, kpi  = parts[2], parts[3]
                payload   = json.loads(msg.payload)
                self._buf.setdefault(sid, {"ts": ""})
                self._buf[sid][kpi] = float(payload.get("value", 0))
                self._buf[sid]["ts"] = payload.get("ts", datetime.now(timezone.utc).isoformat())
        except Exception as e:
            log.debug(f"MqttBackend._on_msg: {e}")

    def collect_all(self):
        return dict(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL: INFLUXDB WRITER
# ══════════════════════════════════════════════════════════════════════════════

class InfluxWriter:
    """Writes to InfluxDB for long-term storage. Optional — CSV store always works."""
    def __init__(self):
        self._ok = False
        try:
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.write_api import SYNCHRONOUS
            self._w    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
                             ).write_api(write_options=SYNCHRONOUS)
            self._P    = Point
            self._ok   = True
            log.info(f"InfluxDB connected: {INFLUX_URL}")
        except Exception:
            pass

    def write(self, sid, reading):
        if not self._ok:
            return
        try:
            for k, v in reading.items():
                if k == "ts" or not isinstance(v, (int, float)):
                    continue
                p = self._P("bts").tag("station", sid).tag("kpi", k).field("v", float(v))
                self._w.write(INFLUX_BUCKET, INFLUX_ORG, p)
        except Exception as e:
            log.debug(f"InfluxWriter: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL CSV STORE  (zero-dependency fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv(sid, reading):
    path = STORE_DIR / f"{sid}.csv"
    row  = {"ts": reading.get("ts",""),
            **{k: v for k, v in reading.items() if k != "ts"}}
    if PD_OK:
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    else:
        with open(path, "a") as f:
            if not path.exists() or path.stat().st_size == 0:
                f.write(",".join(row.keys()) + "\n")
            f.write(",".join(str(v) for v in row.values()) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CONNECTOR — main class
# ══════════════════════════════════════════════════════════════════════════════

class DataConnector:
    """
    Used by FluxAgent dashboard instead of synthetic live_sensor() / spark_history().

    Quickstart in streamlit_pdm.py:
        from data_connector import connector
        if "conn_started" not in st.session_state:
            connector.start()
            connector.collect_once()
            st.session_state.conn_started = True

    Replace live_sensor(s) with:
        reading = connector.get_latest(s["id"])
        kpi = {"power_subsystem":"dc_voltage_v", ...}[s["sub"]]
        return reading.get(kpi, s["sensor_nom"])

    Replace spark_history(s) with:
        kpi = {"power_subsystem":"dc_voltage_v", ...}[s["sub"]]
        return connector.get_history(s["id"], kpi, n=12)
    """

    def __init__(self):
        m = MODE.lower()
        if   m == "file":        self._b = FileBackend()
        elif m == "rest":        self._b = RestBackend()
        elif m == "mqtt":        self._b = MqttBackend()
        else:                    self._b = SimulationBackend(); m = "simulation"
        self._mode     = m
        self._influx   = InfluxWriter()
        self._running  = False
        self._thread   = None
        log.info(f"DataConnector ready — mode={m}")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"Polling every {POLL_INTERVAL_S}s")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            self._tick()
            time.sleep(POLL_INTERVAL_S)

    def _tick(self):
        try:
            data = self._b.collect_all()
            with _lock:
                for sid, r in data.items():
                    _latest[sid] = r
                    _history.setdefault(sid, []).append(r)
                    if len(_history[sid]) > _MAX_H:
                        _history[sid] = _history[sid][-_MAX_H:]
            for sid, r in data.items():
                _write_csv(sid, r)
                self._influx.write(sid, r)
        except Exception as e:
            log.error(f"Tick error: {e}")

    def collect_once(self):
        """Synchronous first load."""
        self._tick()

    def get_latest(self, sid: str) -> dict:
        with _lock:
            return dict(_latest.get(sid, {}))

    def get_history(self, sid: str, kpi: str, n: int = 12) -> list:
        with _lock:
            h = _history.get(sid, [])
        return [r[kpi] for r in h[-n:] if kpi in r]

    @property
    def mode(self):
        return self._mode

    def status(self) -> dict:
        with _lock:
            return {"mode": self._mode, "stations": len(_latest),
                    "poll_s": POLL_INTERVAL_S, "running": self._running}


# ── Singleton ──────────────────────────────────────────────────────────────────
connector = DataConnector()


if __name__ == "__main__":
    print("=" * 56)
    print("FluxAgent DataConnector — self-test")
    print(f"  Mode: {MODE}")
    print("=" * 56)
    c = DataConnector()
    rows = c._b.collect_all()
    for sid, r in list(rows.items())[:3]:
        print(f"\n  {sid}:")
        for k, v in r.items():
            print(f"    {k:<30} {v}")
    print(f"\n  Total: {len(rows)} stations  ✓")
    print("=" * 56)
