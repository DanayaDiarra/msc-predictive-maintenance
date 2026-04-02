"""
data_connector.py — Real-Time Sensor Ingestion for VectorAgent
==============================================================
Bridges physical BTS sensors → VectorAgent NOC pipeline.

SUPPORTED PROTOCOLS:
  • MQTT       — lightweight pub/sub (most modern BTS vendors)
  • REST/HTTP  — polling vendor APIs (Nokia NetAct, Ericsson OSS, Huawei iManager)
  • Kafka      — high-throughput environments (NOC already has a message bus)
  • File/CSV   — CSV drops or Parquet files (lab / offline mode)

OUTPUTS:
  • In-memory SensorBuffer (rolling 125-reading window per station)
  • Parquet files: data/live/{station_id}/{date}.parquet
  • FastAPI REST endpoint: GET /stations/{id}/rul  GET /stations/{id}/sensors
  • Direct XGBoost v2 Final inference via interpreter_agent.py

USAGE:
  # Start as background service (production)
  python data_connector.py --mode mqtt --broker mqtt.yourtelecom.com --port 8883

  # REST polling mode (Nokia NetAct example)
  python data_connector.py --mode rest --api-url https://netact.yourtelecom.com/api/v1

  # File-based (lab / demo with real CSV exports from NMS)
  python data_connector.py --mode file --csv-dir data/raw/live_exports/

  # API server only (dashboard reads from it)
  python data_connector.py --serve --port 8765

INTEGRATION WITH streamlit_pdm.py:
  from data_connector import DataConnectorClient
  client = DataConnectorClient("http://localhost:8765")
  rul, importance = client.get_rul("BTS_DAKAR_001")
  sensors = client.get_sensors("BTS_DAKAR_001")
"""

import os, json, time, logging, argparse, threading
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
from typing import Optional
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("VectorAgent.DataConnector")

# ── Paths ──────────────────────────────────────────────────────────────────
LIVE_DIR    = Path("data/live")
MODEL_DIR   = Path("models_artifacts/final_models")
LIVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Standard sensor schema ──────────────────────────────────────────────────
# All connectors normalize to this before writing to the buffer.
# Keys match the telecom-mapped feature names in 2_feature_engineering_pipeline.py.
SENSOR_SCHEMA = {
    "dc_voltage_v":       {"unit": "V",   "nominal": 48.0,  "alarm_low": 44.0,  "alarm_high": 58.0},
    "cabinet_temp_c":     {"unit": "°C",  "nominal": 35.0,  "alarm_low": None,  "alarm_high": 60.0},
    "fan_speed_rpm":      {"unit": "RPM", "nominal": 3200.0,"alarm_low": 2000.0,"alarm_high": None},
    "vswr_ratio":         {"unit": ":1",  "nominal": 1.3,   "alarm_low": None,  "alarm_high": 2.0},
    "latency_ms":         {"unit": "ms",  "nominal": 5.0,   "alarm_low": None,  "alarm_high": 10.0},
    "cpu_util_pct":       {"unit": "%",   "nominal": 60.0,  "alarm_low": None,  "alarm_high": 85.0},
    "battery_cap_pct":    {"unit": "%",   "nominal": 90.0,  "alarm_low": 80.0,  "alarm_high": None},
    "throughput_mbps":    {"unit": "Mbps","nominal": 800.0, "alarm_low": 200.0, "alarm_high": None},
    "packet_loss_pct":    {"unit": "%",   "nominal": 0.0,   "alarm_low": None,  "alarm_high": 0.1},
    "rssi_dbm":           {"unit": "dBm", "nominal": -65.0, "alarm_low": -80.0, "alarm_high": None},
    "tx_power_dbm":       {"unit": "dBm", "nominal": 43.0,  "alarm_low": 40.0,  "alarm_high": None},
    "ambient_temp_c":     {"unit": "°C",  "nominal": 30.0,  "alarm_low": None,  "alarm_high": 45.0},
}

# ── Sensor reading dataclass ────────────────────────────────────────────────
class SensorReading:
    __slots__ = ("station_id","timestamp","readings","quality","source")

    def __init__(self, station_id: str, timestamp: str, readings: dict,
                 quality: str = "GOOD", source: str = "unknown"):
        self.station_id = station_id
        self.timestamp  = timestamp
        self.readings   = readings          # {sensor_name: float}
        self.quality    = quality           # GOOD | DEGRADED | BAD
        self.source     = source

    def to_dict(self):
        return dict(station_id=self.station_id, timestamp=self.timestamp,
                    readings=self.readings, quality=self.quality, source=self.source)

    @classmethod
    def from_dict(cls, d):
        return cls(d["station_id"], d["timestamp"], d["readings"],
                   d.get("quality","GOOD"), d.get("source","unknown"))


# ══════════════════════════════════════════════════════════════════════════════
#  SENSOR BUFFER — rolling window per station
# ══════════════════════════════════════════════════════════════════════════════
class SensorBuffer:
    """
    Maintains a rolling window of the last MAX_READINGS sensor readings per station.
    Thread-safe. Also persists to Parquet for historical replay.
    """
    MAX_READINGS = 125  # matches C-MAPSS max cycle window

    def __init__(self):
        self._buffers: dict[str, deque] = {}
        self._lock = threading.Lock()

    def push(self, reading: SensorReading):
        sid = reading.station_id
        with self._lock:
            if sid not in self._buffers:
                self._buffers[sid] = deque(maxlen=self.MAX_READINGS)
            self._buffers[sid].append(reading)
        self._persist(reading)

    def get_window(self, station_id: str, n: int = 20) -> list:
        """Return last n readings for a station, oldest first."""
        with self._lock:
            buf = self._buffers.get(station_id, deque())
            return list(buf)[-n:]

    def get_latest(self, station_id: str) -> Optional[SensorReading]:
        with self._lock:
            buf = self._buffers.get(station_id, deque())
            return buf[-1] if buf else None

    def stations(self) -> list:
        with self._lock:
            return list(self._buffers.keys())

    def _persist(self, reading: SensorReading):
        """Write reading to daily Parquet file (append mode)."""
        try:
            import pandas as pd
            today = datetime.now().strftime("%Y-%m-%d")
            path  = LIVE_DIR / reading.station_id
            path.mkdir(parents=True, exist_ok=True)
            fpath = path / f"{today}.parquet"
            row = {"timestamp": reading.timestamp, "station_id": reading.station_id,
                   "quality": reading.quality, "source": reading.source}
            row.update(reading.readings)
            df_new = pd.DataFrame([row])
            if fpath.exists():
                df_old = pd.read_parquet(fpath)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
            df_new.to_parquet(fpath, index=False)
        except Exception as e:
            log.warning(f"Persist failed for {reading.station_id}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ASSEMBLER — converts buffer window into XGBoost v2 feature vector
# ══════════════════════════════════════════════════════════════════════════════
class FeatureAssembler:
    """
    Converts a list of SensorReading objects into the feature vector
    expected by XGBoost v2 Final (same contract as 2_feature_engineering_pipeline.py).

    In production: use realtime_feature_engine.py for the full 80-feature set.
    This provides the minimal viable set for immediate inference.
    """

    SENSOR_TO_FEATURE = {
        "dc_voltage_v":    "voltage",
        "cabinet_temp_c":  "cabinet_temperature",
        "fan_speed_rpm":   "fan_speed",
        "vswr_ratio":      "signal_quality",
        "latency_ms":      "latency_ms",
        "cpu_util_pct":    "cpu_utilization",
        "battery_cap_pct": "battery_voltage",
        "throughput_mbps": "throughput_mbps",
        "packet_loss_pct": "packet_loss",
        "rssi_dbm":        "signal_quality",
    }

    def assemble(self, window: list, station_id: str, cycle_count: int) -> dict:
        """
        window: list of SensorReading, most recent last.
        Returns: dict of feature_name → float value.
        """
        if not window:
            return {}

        # Extract time-ordered sensor series
        series = {}
        for r in window:
            for k, v in r.readings.items():
                feat = self.SENSOR_TO_FEATURE.get(k, k)
                series.setdefault(feat, []).append(float(v))

        features = {}
        features["time_cycle"] = cycle_count

        for feat, vals in series.items():
            arr = np.array(vals)
            if len(arr) < 1:
                continue

            # Current value
            features[feat] = float(arr[-1])

            # Rolling stats (5, 10, 20 window)
            for w in [5, 10, 20]:
                if len(arr) >= w:
                    features[f"{feat}_avg{w}"]   = float(np.mean(arr[-w:]))
                    features[f"{feat}_std{w}"]   = float(np.std(arr[-w:]))
                if len(arr) >= w + 1:
                    features[f"{feat}_slope{w}"] = float(
                        (arr[-1] - arr[-w]) / w if w <= len(arr) else 0.0)

            # Lag features
            for lag in [1, 3, 5]:
                if len(arr) > lag:
                    features[f"{feat}_lag{lag}"] = float(arr[-1 - lag])

        # Interaction features
        v  = features.get("voltage", 48.0)
        tp = features.get("throughput_mbps", 500.0)
        lt = features.get("latency_ms", 5.0)
        ct = features.get("cabinet_temperature", 35.0)

        features["network_quality"]  = tp / (lt + 1.0)
        features["power_efficiency"] = tp / (max(v, 0.001))
        features["mem_x_voltage"]    = features.get("cpu_utilization", 60.0) * v

        return features


# ══════════════════════════════════════════════════════════════════════════════
#  MQTT CONNECTOR
# ══════════════════════════════════════════════════════════════════════════════
class MQTTConnector:
    """
    Subscribes to topic bts/+/telemetry on an MQTT broker.
    Expected payload (JSON):
      {"station_id": "BTS_DAKAR_001", "timestamp": "...", "readings": {...}}

    Install: pip install paho-mqtt
    TLS: set tls_ca_cert path for production (port 8883).
    """
    TOPIC = "bts/+/telemetry"

    def __init__(self, broker: str, port: int, buffer: SensorBuffer,
                 username: str = None, password: str = None, ca_cert: str = None):
        self.broker   = broker
        self.port     = port
        self.buffer   = buffer
        self.username = username
        self.password = password
        self.ca_cert  = ca_cert
        self._client  = None

    def start(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            log.error("paho-mqtt not installed. Run: pip install paho-mqtt")
            return

        self._client = mqtt.Client(client_id="vectoragent-connector")
        if self.username:
            self._client.username_pw_set(self.username, self.password)
        if self.ca_cert:
            import ssl
            self._client.tls_set(self.ca_cert, tls_version=ssl.PROTOCOL_TLS)

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = lambda c, u, rc: log.warning(f"MQTT disconnected rc={rc}")

        log.info(f"Connecting to MQTT broker {self.broker}:{self.port}")
        self._client.connect(self.broker, self.port, keepalive=60)
        self._client.loop_start()

    def stop(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info(f"MQTT connected. Subscribing to {self.TOPIC}")
            client.subscribe(self.TOPIC, qos=1)
        else:
            log.error(f"MQTT connect failed: rc={rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            reading = SensorReading(
                station_id = payload.get("station_id", msg.topic.split("/")[1]),
                timestamp  = payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
                readings   = payload.get("readings", {}),
                quality    = payload.get("quality", "GOOD"),
                source     = "mqtt",
            )
            self.buffer.push(reading)
            log.debug(f"Received: {reading.station_id} sensors={list(reading.readings.keys())}")
        except Exception as e:
            log.warning(f"Failed to parse MQTT message: {e} — payload: {msg.payload[:120]}")


# ══════════════════════════════════════════════════════════════════════════════
#  REST POLLER — for Nokia NetAct, Ericsson OSS, Huawei iManager
# ══════════════════════════════════════════════════════════════════════════════
class RESTPoller:
    """
    Polls a vendor REST API for BTS KPIs.

    Supports two vendor formats:
      - Generic JSON (custom NMS)
      - Nokia NetAct format (normalized automatically)
      - Ericsson OSS format (normalized automatically)

    Install: pip install requests
    """
    POLL_INTERVAL_S = 30

    def __init__(self, api_url: str, buffer: SensorBuffer,
                 api_key: str = None, vendor: str = "generic",
                 station_ids: list = None):
        self.api_url     = api_url.rstrip("/")
        self.buffer      = buffer
        self.api_key     = api_key
        self.vendor      = vendor
        self.station_ids = station_ids or []
        self._running    = False
        self._thread     = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info(f"REST poller started: {self.api_url} vendor={self.vendor}")

    def stop(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            for station_id in self.station_ids:
                try:
                    self._poll_station(station_id)
                except Exception as e:
                    log.warning(f"REST poll failed for {station_id}: {e}")
            time.sleep(self.POLL_INTERVAL_S)

    def _poll_station(self, station_id: str):
        try:
            import requests
        except ImportError:
            log.error("requests not installed. Run: pip install requests")
            return

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.vendor == "nokia_netact":
            url = f"{self.api_url}/network-elements/{station_id}/pm-data"
        elif self.vendor == "ericsson_oss":
            url = f"{self.api_url}/managedObjects/{station_id}/attributes"
        else:
            url = f"{self.api_url}/stations/{station_id}/kpis"

        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        readings = self._normalize(raw, self.vendor)
        if readings:
            reading = SensorReading(
                station_id = station_id,
                timestamp  = datetime.now(timezone.utc).isoformat(),
                readings   = readings,
                quality    = "GOOD",
                source     = f"rest_{self.vendor}",
            )
            self.buffer.push(reading)

    def _normalize(self, raw: dict, vendor: str) -> dict:
        """Map vendor-specific field names to standard schema."""
        if vendor == "nokia_netact":
            return {
                "dc_voltage_v":    raw.get("dcPowerSupplyVoltage"),
                "cabinet_temp_c":  raw.get("cabinetTemperature"),
                "fan_speed_rpm":   raw.get("coolingFanSpeed"),
                "cpu_util_pct":    raw.get("bbuCpuLoad"),
                "latency_ms":      raw.get("backhaulLatency"),
                "throughput_mbps": raw.get("backhaulThroughput"),
                "tx_power_dbm":    raw.get("txPowerDlAvg"),
                "vswr_ratio":      raw.get("antennaVSWR"),
            }
        elif vendor == "ericsson_oss":
            return {
                "dc_voltage_v":    raw.get("pmDcInputVoltage"),
                "cabinet_temp_c":  raw.get("pmCabinetTemp"),
                "fan_speed_rpm":   raw.get("pmFanRpm"),
                "cpu_util_pct":    raw.get("pmCpuLoad"),
                "latency_ms":      raw.get("pmBackhaulDelay"),
                "throughput_mbps": raw.get("pmActiveDlThroughput"),
            }
        else:
            # Generic: assume keys already match standard schema
            return {k: v for k, v in raw.items()
                    if k in SENSOR_SCHEMA and v is not None}


# ══════════════════════════════════════════════════════════════════════════════
#  CSV / PARQUET FILE READER — for lab / offline mode with NMS exports
# ══════════════════════════════════════════════════════════════════════════════
class FileConnector:
    """
    Reads CSV or Parquet files exported from NMS and feeds them into the buffer
    at real-time rate (simulates live ingestion from historical data).

    Useful for: testing, demos, academic validation, backfill.

    Expected CSV columns: station_id, timestamp, dc_voltage_v, cabinet_temp_c, ...
    """
    def __init__(self, csv_dir: str, buffer: SensorBuffer,
                 speed_multiplier: float = 1.0):
        self.csv_dir   = Path(csv_dir)
        self.buffer    = buffer
        self.speed     = speed_multiplier   # >1 = replay faster than real-time
        self._running  = False

    def start(self):
        self._running = True
        t = threading.Thread(target=self._replay_loop, daemon=True)
        t.start()
        log.info(f"File connector started: {self.csv_dir}")

    def stop(self):
        self._running = False

    def _replay_loop(self):
        try:
            import pandas as pd
        except ImportError:
            log.error("pandas not installed.")
            return

        files = sorted(self.csv_dir.glob("*.csv")) + sorted(self.csv_dir.glob("*.parquet"))
        if not files:
            log.warning(f"No CSV/Parquet files found in {self.csv_dir}")
            return

        for fpath in files:
            if not self._running:
                break
            df = pd.read_csv(fpath) if fpath.suffix == ".csv" else pd.read_parquet(fpath)
            df = df.sort_values("timestamp") if "timestamp" in df.columns else df

            sensor_cols = [c for c in df.columns if c in SENSOR_SCHEMA]
            prev_ts = None

            for _, row in df.iterrows():
                if not self._running:
                    break
                readings = {c: float(row[c]) for c in sensor_cols
                            if row[c] is not None and not np.isnan(row[c])}
                if not readings:
                    continue
                ts = str(row.get("timestamp", datetime.now(timezone.utc).isoformat()))
                sid = str(row.get("station_id", "UNKNOWN"))
                reading = SensorReading(sid, ts, readings, quality="GOOD", source="file")
                self.buffer.push(reading)

                # Rate-control: sleep to match original timestamps
                if prev_ts and "timestamp" in row:
                    try:
                        from datetime import datetime as _dt
                        t1 = _dt.fromisoformat(str(prev_ts).replace("Z","+00:00"))
                        t2 = _dt.fromisoformat(ts.replace("Z","+00:00"))
                        delta = (t2 - t1).total_seconds()
                        if 0 < delta < 300:  # cap at 5 min gap
                            time.sleep(delta / max(self.speed, 0.01))
                    except Exception:
                        time.sleep(0.1 / max(self.speed, 0.01))
                prev_ts = ts


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE — wraps XGBoost v2 Final for live inference
# ══════════════════════════════════════════════════════════════════════════════
class LivePredictionEngine:
    """
    Loads XGBoost v2 Final model and runs inference on assembled feature vectors.
    Used by the FastAPI server and by streamlit_pdm.py when PIPELINE_OK=True.
    """
    MAX_RUL = 125

    def __init__(self, model_dir: str = str(MODEL_DIR)):
        self.model      = None
        self.feat_cols  = None
        self.assembler  = FeatureAssembler()
        self._lock      = threading.Lock()
        self._load(model_dir)

    def _load(self, model_dir: str):
        import pickle
        mp = Path(model_dir) / "xgb_v2.pkl"
        fp = Path(model_dir) / "feat_cols_xgb.pkl"
        if mp.exists() and fp.exists():
            with open(mp, "rb") as f: self.model     = pickle.load(f)
            with open(fp, "rb") as f: self.feat_cols = pickle.load(f)
            log.info(f"XGBoost v2 Final loaded: {len(self.feat_cols)} features")
        else:
            log.warning(f"Model not found at {model_dir} — inference unavailable")

    def predict(self, station_id: str, buffer: SensorBuffer,
                cycle_count: int = 100) -> dict:
        """
        Predict RUL for a station from its sensor buffer.
        Returns: {"rul": float, "urgency": str, "importance": dict, "source": str}
        """
        window   = buffer.get_window(station_id, n=25)
        features = self.assembler.assemble(window, station_id, cycle_count)

        if not features or self.model is None:
            return {"rul": None, "urgency": None, "importance": {}, "source": "unavailable"}

        try:
            import pandas as pd
            # Align to model feature columns, fill missing with 0
            X = pd.DataFrame([features]).reindex(columns=self.feat_cols, fill_value=0.0)
            with self._lock:
                rul  = float(np.clip(self.model.predict(X.values.astype(np.float32))[0], 0, self.MAX_RUL))
                imps = dict(zip(self.feat_cols, self.model.feature_importances_))
            urgency = "Critical" if rul <= 20 else ("Warning" if rul <= 50 else "Monitor")
            # Top-5 importance
            top_imp = dict(sorted(imps.items(), key=lambda x: -x[1])[:5])
            return {"rul": round(rul, 2), "urgency": urgency, "importance": top_imp, "source": "xgb_v2_live"}
        except Exception as e:
            log.error(f"Prediction failed for {station_id}: {e}")
            return {"rul": None, "urgency": None, "importance": {}, "source": "error"}


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI SERVER — exposes sensor data + RUL to dashboard and external systems
# ══════════════════════════════════════════════════════════════════════════════
class DataConnectorServer:
    """
    REST API server that exposes VectorAgent data to:
      - streamlit_pdm.py (dashboard reads live RUL)
      - External NOC systems (Prometheus scraping, Grafana)
      - Mobile apps or alert gateways

    Install: pip install fastapi uvicorn
    """
    def __init__(self, buffer: SensorBuffer, engine: LivePredictionEngine, port: int = 8765):
        self.buffer = buffer
        self.engine = engine
        self.port   = port

    def serve(self):
        try:
            from fastapi import FastAPI
            import uvicorn
        except ImportError:
            log.error("fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")
            return

        app = FastAPI(title="VectorAgent Data API", version="1.0.0")

        @app.get("/health")
        def health():
            return {"status": "ok", "stations": len(self.buffer.stations())}

        @app.get("/stations")
        def list_stations():
            return {"stations": self.buffer.stations()}

        @app.get("/stations/{station_id}/rul")
        def get_rul(station_id: str, cycles: int = 100):
            result = self.engine.predict(station_id, self.buffer, cycles)
            result["station_id"] = station_id
            result["timestamp"]  = datetime.now(timezone.utc).isoformat()
            return result

        @app.get("/stations/{station_id}/sensors")
        def get_sensors(station_id: str, n: int = 12):
            window = self.buffer.get_window(station_id, n=n)
            latest = self.buffer.get_latest(station_id)
            return {
                "station_id": station_id,
                "latest":     latest.to_dict() if latest else None,
                "window_n":   len(window),
                "readings":   [r.to_dict() for r in window[-5:]],
            }

        @app.get("/stations/{station_id}/sensors/latest")
        def get_latest_sensor(station_id: str):
            latest = self.buffer.get_latest(station_id)
            if not latest:
                return {"error": "no data", "station_id": station_id}
            return latest.to_dict()

        log.info(f"Starting VectorAgent Data API on port {self.port}")
        uvicorn.run(app, host="0.0.0.0", port=self.port, log_level="warning")


# ══════════════════════════════════════════════════════════════════════════════
#  CLIENT — used by streamlit_pdm.py to read from the connector server
# ══════════════════════════════════════════════════════════════════════════════
class DataConnectorClient:
    """
    Lightweight HTTP client used inside streamlit_pdm.py to replace
    the simulation-based live_rul() function with real inference.

    Usage in streamlit_pdm.py:
        from data_connector import DataConnectorClient
        _client = DataConnectorClient("http://localhost:8765")

        # Replace: rul = live_rul(s)
        rul = _client.get_rul(s["id"]) or live_rul(s)  # fallback to sim if offline
    """
    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def _get(self, path: str) -> Optional[dict]:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=self.timeout) as r:
                return json.loads(r.read())
        except Exception:
            return None

    def get_rul(self, station_id: str) -> Optional[float]:
        data = self._get(f"/stations/{station_id}/rul")
        return data.get("rul") if data else None

    def get_urgency(self, station_id: str) -> Optional[str]:
        data = self._get(f"/stations/{station_id}/rul")
        return data.get("urgency") if data else None

    def get_sensors(self, station_id: str) -> Optional[dict]:
        data = self._get(f"/stations/{station_id}/sensors/latest")
        return data.get("readings") if data else None

    def get_importance(self, station_id: str) -> Optional[dict]:
        data = self._get(f"/stations/{station_id}/rul")
        return data.get("importance") if data else None

    def is_available(self) -> bool:
        return self._get("/health") is not None


# ══════════════════════════════════════════════════════════════════════════════
#  RETRAINING TRIGGER — monitors prediction drift and triggers retraining
# ══════════════════════════════════════════════════════════════════════════════
class RetrainingMonitor:
    """
    Monitors the gap between predicted RUL and actual maintenance events.
    When drift exceeds threshold, writes a trigger file for 3_model_training.py.

    Called automatically when a dispatch ticket is validated in streamlit_pdm.py:
        monitor.record_validation(station_id, predicted_rul, actual_rul_at_failure)
    """
    DRIFT_THRESHOLD = 1.5   # retrain if mean_abs_error > 1.5 × historical RMSE
    HISTORICAL_RMSE = 14.60 # XGBoost v2 Final baseline

    def __init__(self, trigger_dir: str = "data/retrain_triggers"):
        self.records   = []
        self.trigger_dir = Path(trigger_dir)
        self.trigger_dir.mkdir(parents=True, exist_ok=True)

    def record_validation(self, station_id: str, predicted_rul: float, actual_rul: float):
        error = abs(predicted_rul - actual_rul)
        self.records.append({"station_id": station_id, "predicted": predicted_rul,
                              "actual": actual_rul, "error": error,
                              "ts": datetime.now(timezone.utc).isoformat()})
        log.info(f"Validation recorded: {station_id} pred={predicted_rul:.1f} actual={actual_rul:.1f} err={error:.1f}")
        self._check_drift()

    def _check_drift(self):
        if len(self.records) < 10:
            return
        mae = np.mean([r["error"] for r in self.records[-20:]])
        if mae > self.DRIFT_THRESHOLD * self.HISTORICAL_RMSE:
            trigger_path = self.trigger_dir / f"retrain_{int(time.time())}.json"
            trigger = {
                "trigger_reason": "drift",
                "mae": round(mae, 2),
                "threshold": round(self.DRIFT_THRESHOLD * self.HISTORICAL_RMSE, 2),
                "n_records": len(self.records),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "command": "python 3_model_training.py --incremental --reason drift",
            }
            trigger_path.write_text(json.dumps(trigger, indent=2))
            log.warning(f"RETRAIN TRIGGERED: MAE={mae:.2f} > threshold={trigger['threshold']:.2f}. "
                        f"See: {trigger_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="VectorAgent Data Connector")
    parser.add_argument("--mode",    choices=["mqtt","rest","file","demo"], default="demo",
                        help="Data source mode")
    parser.add_argument("--serve",   action="store_true", help="Start FastAPI REST server")
    parser.add_argument("--port",    type=int, default=8765, help="API server port")
    # MQTT options
    parser.add_argument("--broker",  default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-user", default=None)
    parser.add_argument("--mqtt-pass", default=None)
    parser.add_argument("--ca-cert",   default=None)
    # REST options
    parser.add_argument("--api-url",  default="http://localhost:9000")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--vendor",   choices=["generic","nokia_netact","ericsson_oss"], default="generic")
    parser.add_argument("--stations", nargs="*", default=[], help="Station IDs to poll")
    # File options
    parser.add_argument("--csv-dir",  default="data/raw/live_exports/")
    parser.add_argument("--speed",    type=float, default=1.0, help="Replay speed multiplier")
    args = parser.parse_args()

    buffer  = SensorBuffer()
    engine  = LivePredictionEngine()
    monitor = RetrainingMonitor()

    connector = None
    if args.mode == "mqtt":
        connector = MQTTConnector(args.broker, args.mqtt_port, buffer,
                                  args.mqtt_user, args.mqtt_pass, args.ca_cert)
        connector.start()
    elif args.mode == "rest":
        connector = RESTPoller(args.api_url, buffer, args.api_key, args.vendor, args.stations)
        connector.start()
    elif args.mode == "file":
        connector = FileConnector(args.csv_dir, buffer, args.speed)
        connector.start()
    elif args.mode == "demo":
        log.info("Demo mode: generating synthetic sensor readings for 3 stations")
        def _demo_feed():
            stations = ["BTS_DAKAR_001","BTS_DAKAR_002","BTS_DAKAR_003"]
            t = 0
            while True:
                for sid in stations:
                    rng = np.random.default_rng(hash(sid + str(t)) % 99999)
                    readings = {
                        "dc_voltage_v":   round(48.0 - t * 0.002 + rng.normal(0, 0.3), 2),
                        "cabinet_temp_c": round(35.0 + t * 0.003 + rng.normal(0, 0.5), 2),
                        "fan_speed_rpm":  round(3200 - t * 0.5   + rng.normal(0, 20),  1),
                        "cpu_util_pct":   round(60.0 + t * 0.01  + rng.normal(0, 1.0), 1),
                        "latency_ms":     round(5.0  + t * 0.002 + rng.normal(0, 0.1), 2),
                    }
                    buffer.push(SensorReading(
                        station_id=sid,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        readings=readings, quality="GOOD", source="demo"))
                t += 1
                time.sleep(5)
        threading.Thread(target=_demo_feed, daemon=True).start()
        log.info("Demo feed started. Producing readings every 5s.")

    if args.serve or args.mode == "demo":
        server = DataConnectorServer(buffer, engine, args.port)
        log.info(f"VectorAgent Data API starting on http://localhost:{args.port}")
        log.info(f"  GET /health")
        log.info(f"  GET /stations")
        log.info(f"  GET /stations/{{id}}/rul")
        log.info(f"  GET /stations/{{id}}/sensors")
        server.serve()
    else:
        # Run connector without API server — just write to buffer + Parquet
        log.info("Connector running (no API server). Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(10)
                stations = buffer.stations()
                if stations:
                    log.info(f"Buffer: {len(stations)} stations active. Latest: {stations[-1]}")
        except KeyboardInterrupt:
            log.info("Stopping connector.")
            if connector:
                connector.stop()


if __name__ == "__main__":
    main()
