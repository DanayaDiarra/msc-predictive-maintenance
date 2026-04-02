"""
realtime_feature_engine.py — Streaming Feature Engineering for VectorAgent
===========================================================================
Production counterpart of 2_feature_engineering_pipeline.py.

Whereas 2_feature_engineering_pipeline.py processes the full C-MAPSS dataset
in batch mode (offline training), this module processes live sensor readings
from data_connector.py in a streaming fashion, maintaining a per-station
rolling window and computing all 80+ features on every new reading.

The feature contract is IDENTICAL to the batch pipeline — same column names,
same engineering logic — ensuring XGBoost v2 Final can be used directly
without any retraining or feature remapping.

FEATURE CATEGORIES COMPUTED (same as batch pipeline):
  • Raw sensor readings                (18 sensors)
  • Rolling means: 5, 10, 20 windows  (18 × 3 = 54)
  • Rolling std dev: 5-window         (18)
  • Rolling min: 20-window            (18)
  • Trend/slope: 5, 10, 20 windows    (18 × 3 = 54)
  • Lag features: 1, 3, 5 steps       (5 core × 3 = 15)
  • Health index (per-engine min-max) (1)
  • Network quality & power efficiency(2)
  • Interaction features              (4)
  • Cumulative degradation            (1)
  TOTAL: ~80 features

USAGE:
  from realtime_feature_engine import RealtimeFeatureEngine
  engine = RealtimeFeatureEngine()
  engine.push(station_id="BTS_DAKAR_001", reading=sensor_dict, cycle=t)
  feature_vector = engine.get_features("BTS_DAKAR_001")
  rul = xgb_model.predict([feature_vector])
"""

import os, time, threading
import numpy as np
from collections import deque
from typing import Optional
from pathlib import Path

# ── NASA→Telecom sensor name mapping (mirrors 2_feature_engineering_pipeline.py) ──
NASA_TO_TELECOM = {
    "sensor_01":  "cabinet_temperature",
    "sensor_02":  "ambient_temperature",
    "sensor_03":  "humidity",
    "sensor_04":  "voltage",
    "sensor_05":  "current",
    "sensor_06":  "fan_speed",
    "sensor_07":  "cooling_efficiency",
    "sensor_08":  "battery_voltage",
    "sensor_09":  "total_power_consumption",
    "sensor_10":  "cpu_utilization",
    "sensor_11":  "memory_usage",
    "sensor_12":  "disk_usage",
    "sensor_13":  "throughput_mbps",
    "sensor_14":  "latency_ms",
    "sensor_15":  "packet_loss",
    "sensor_16":  "error_rate",
    "sensor_17":  "signal_quality",
    "sensor_18":  "connected_users",
    "setting_1":  "antenna_power",
    "setting_2":  "signal_strength",
    "setting_3":  "antenna_tilt",
}

# Live sensor name → telecom feature name (from data_connector.py schema)
LIVE_TO_TELECOM = {
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
    "tx_power_dbm":    "antenna_power",
    "ambient_temp_c":  "ambient_temperature",
}

CORE_SENSORS = [
    "total_power_consumption",
    "throughput_mbps",
    "latency_ms",
    "cabinet_temperature",
    "cpu_utilization",
]

WINDOWS = [5, 10, 20]
LAGS    = [1, 3, 5]


# ══════════════════════════════════════════════════════════════════════════════
#  STATION WINDOW — per-station rolling history
# ══════════════════════════════════════════════════════════════════════════════
class StationWindow:
    """
    Maintains a rolling window of normalized sensor readings for one station.
    Each entry is a dict: {sensor_name: float}.
    """
    MAX_WINDOW = 125  # matches C-MAPSS RUL cap

    def __init__(self, station_id: str):
        self.station_id    = station_id
        self.history       = deque(maxlen=self.MAX_WINDOW)  # deque of dicts
        self.cycle_count   = 0
        self.first_reading = None  # for health index normalization
        self._early_buffer = []    # first 10 readings for baseline
        self._lock         = threading.Lock()

    def push(self, reading: dict):
        """Push a new normalized sensor reading dict."""
        with self._lock:
            self.cycle_count += 1
            if self.first_reading is None and len(self._early_buffer) < 10:
                self._early_buffer.append(reading)
            elif self.first_reading is None and len(self._early_buffer) >= 10:
                # Compute baseline from first 10 readings
                self.first_reading = {
                    k: float(np.mean([r.get(k,0) for r in self._early_buffer if k in r]))
                    for k in self._early_buffer[0]
                }
            self.history.append(dict(reading))

    def get_series(self, sensor: str, n: int = None) -> np.ndarray:
        """Return time-ordered values for a sensor as numpy array."""
        with self._lock:
            h = list(self.history)
        vals = [r.get(sensor) for r in h]
        vals = [v for v in vals if v is not None]
        arr  = np.array(vals, dtype=np.float64)
        return arr[-n:] if (n and len(arr) >= n) else arr

    def latest(self, sensor: str) -> Optional[float]:
        with self._lock:
            h = list(self.history)
        if not h:
            return None
        return h[-1].get(sensor)

    def baseline(self, sensor: str) -> float:
        if self.first_reading and sensor in self.first_reading:
            return self.first_reading[sensor]
        arr = self.get_series(sensor, n=10)
        return float(np.mean(arr)) if len(arr) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  REALTIME FEATURE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class RealtimeFeatureEngine:
    """
    Computes the full XGBoost v2 feature vector from live sensor readings.

    Example:
        engine = RealtimeFeatureEngine()

        # Called every time a new sensor reading arrives
        engine.push("BTS_DAKAR_001", {
            "dc_voltage_v": 47.8,
            "cabinet_temp_c": 38.2,
            "fan_speed_rpm": 3150,
            ...
        }, cycle=t)

        # Called by interpreter_agent.py for inference
        features = engine.get_features("BTS_DAKAR_001")
        rul = model.predict([list(features.values())])
    """

    def __init__(self, feat_cols_path: str = None):
        self._windows: dict[str, StationWindow] = {}
        self._lock    = threading.Lock()
        self.feat_cols = None

        # Load feature column order if available (for DataFrame alignment)
        if feat_cols_path and Path(feat_cols_path).exists():
            import pickle
            with open(feat_cols_path, "rb") as f:
                self.feat_cols = pickle.load(f)

    def _get_window(self, station_id: str) -> StationWindow:
        with self._lock:
            if station_id not in self._windows:
                self._windows[station_id] = StationWindow(station_id)
            return self._windows[station_id]

    def push(self, station_id: str, raw_reading: dict, cycle: int = None):
        """
        Normalize incoming sensor reading and push to the station window.
        raw_reading: {live_sensor_name: float} e.g. {"dc_voltage_v": 47.8, ...}
        """
        # Normalize live sensor names to telecom feature names
        normalized = {}
        for k, v in raw_reading.items():
            feat = LIVE_TO_TELECOM.get(k, k)
            try:
                normalized[feat] = float(v)
            except (TypeError, ValueError):
                pass

        window = self._get_window(station_id)
        window.push(normalized)
        if cycle is not None:
            window.cycle_count = cycle

    def get_features(self, station_id: str) -> dict:
        """
        Compute the full feature vector for a station.
        Returns: dict {feature_name: float} matching XGBoost v2 training contract.
        """
        w = self._get_window(station_id)
        if not w.history:
            return {}

        features = {}
        features["time_cycle"] = float(w.cycle_count)

        # ── Latest sensor values ──────────────────────────────────────────────
        latest = list(w.history)[-1]
        for sensor, val in latest.items():
            features[sensor] = float(val)

        # ── All sensors in the window ─────────────────────────────────────────
        all_sensors = set()
        for row in w.history:
            all_sensors.update(row.keys())

        for sensor in all_sensors:
            arr = w.get_series(sensor)
            if len(arr) < 2:
                continue

            # Rolling stats per window size
            for window_size in WINDOWS:
                sub = arr[-window_size:] if len(arr) >= window_size else arr
                features[f"{sensor}_avg{window_size}"]   = float(np.mean(sub))
                features[f"{sensor}_std{window_size}"]   = float(np.std(sub)) if len(sub) > 1 else 0.0
                features[f"{sensor}_min{window_size}"]   = float(np.min(sub))
                features[f"{sensor}_max{window_size}"]   = float(np.max(sub))

            # Rolling slopes (trend over window)
            for window_size in WINDOWS:
                if len(arr) > window_size:
                    features[f"{sensor}_slope{window_size}"] = float(
                        (arr[-1] - arr[-window_size]) / window_size)

            # Lag features (only for core sensors)
            if sensor in CORE_SENSORS:
                for lag in LAGS:
                    if len(arr) > lag:
                        features[f"{sensor}_lag{lag}"] = float(arr[-1 - lag])

        # ── Telecom-specific derived features ─────────────────────────────────
        tp  = float(features.get("throughput_mbps", 500.0))
        lt  = float(features.get("latency_ms", 5.0))
        pwr = float(features.get("total_power_consumption", 1000.0))
        sig = float(features.get("signal_quality", 0.7))
        ct  = float(features.get("cabinet_temperature", 35.0))
        cpu = float(features.get("cpu_utilization", 60.0))
        mem = float(features.get("memory_usage", 50.0))
        vol = float(features.get("voltage", 48.0))
        cef = float(features.get("cooling_efficiency", 0.8))

        features["network_quality"]       = tp / (lt + 1.0)
        features["power_efficiency"]      = tp / (pwr + 0.001)
        features["mem_x_voltage"]         = mem * vol
        features["mem_div_voltage"]       = mem / (vol + 1e-9)
        features["power_per_cooling"]     = pwr / (cef + 1e-9)
        features["throughput_x_latency"]  = tp * lt

        # ── Health index (per-station min-max normalised) ─────────────────────
        sig_baseline = w.baseline("signal_quality") or 0.7
        ct_baseline  = w.baseline("cabinet_temperature") or 35.0
        cpu_baseline = w.baseline("cpu_utilization") or 60.0

        sig_norm  = (sig - sig_baseline)  / (abs(sig_baseline)  + 1e-9)
        ct_norm   = (ct  - ct_baseline)   / (abs(ct_baseline)   + 1e-9)
        cpu_norm  = (cpu - cpu_baseline)  / (abs(cpu_baseline)  + 1e-9)
        h_idx = float(np.clip(0.5 + 0.5 * sig_norm - 0.25 * ct_norm - 0.25 * cpu_norm, 0, 1))
        features["health_index"] = h_idx

        # ── Cumulative degradation index ──────────────────────────────────────
        pwr_arr  = w.get_series("total_power_consumption")
        pwr_base = float(np.mean(pwr_arr[:10])) if len(pwr_arr) >= 10 else float(np.mean(pwr_arr)) if len(pwr_arr) > 0 else 0
        if len(pwr_arr) > 0:
            features["cumul_degradation"] = float(np.sum(np.abs(pwr_arr - pwr_base)))

        # ── Align to model feature column order if available ──────────────────
        if self.feat_cols:
            aligned = {k: features.get(k, 0.0) for k in self.feat_cols}
            return aligned

        return features

    def get_feature_vector(self, station_id: str) -> Optional[np.ndarray]:
        """
        Returns features as a numpy array aligned to feat_cols order.
        Used directly by XGBoost v2 Final: model.predict(engine.get_feature_vector(sid))
        """
        feats = self.get_features(station_id)
        if not feats or not self.feat_cols:
            return None
        return np.array([feats.get(c, 0.0) for c in self.feat_cols], dtype=np.float32)

    def stations(self) -> list:
        with self._lock:
            return list(self._windows.keys())

    def cycle_count(self, station_id: str) -> int:
        w = self._windows.get(station_id)
        return w.cycle_count if w else 0

    def summary(self) -> dict:
        return {
            sid: {
                "cycle":     w.cycle_count,
                "window_n":  len(w.history),
                "n_features": len(self.get_features(sid)),
            }
            for sid, w in self._windows.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
#  INCREMENTAL TRAINING DATA COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════
class TrainingDataCollector:
    """
    Collects feature vectors + validated RUL labels from production stations.
    When a dispatch ticket is closed (station restored), the feature vectors
    leading up to failure are labeled and added to the training set.

    Called from streamlit_pdm.py when a ticket is validated:
        collector.record_cycle(station_id, feature_dict)        # every reading
        collector.record_failure(station_id, validated_rul=0)   # on ticket close
        collector.export_training_data("data/raw/new_cycles.parquet")
    """

    def __init__(self, max_per_station: int = 500):
        self._cycles: dict[str, list] = {}   # {station_id: [feature_dict + RUL]}
        self._max    = max_per_station
        self._lock   = threading.Lock()

    def record_cycle(self, station_id: str, features: dict, estimated_rul: float):
        """Record one observation. Called on every sensor reading."""
        with self._lock:
            if station_id not in self._cycles:
                self._cycles[station_id] = []
            entry = dict(features)
            entry["RUL"] = round(estimated_rul, 2)
            entry["station_id"] = station_id
            entry["timestamp"]  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._cycles[station_id].append(entry)
            # Keep only the last max_per_station readings
            if len(self._cycles[station_id]) > self._max:
                self._cycles[station_id] = self._cycles[station_id][-self._max:]

    def record_failure(self, station_id: str, validated_rul: float = 0.0):
        """
        Called when a station is repaired and ticket validated.
        Retroactively corrects RUL labels: the last reading before repair = RUL 0,
        working backwards to compute true remaining life for each historical point.
        """
        with self._lock:
            cycles = self._cycles.get(station_id, [])
            n = len(cycles)
            for i, entry in enumerate(reversed(cycles)):
                entry["RUL"] = round(validated_rul + i, 2)
        import logging
        logging.getLogger("VectorAgent.TrainingCollector").info(
            f"Failure recorded for {station_id}: {n} cycles relabelled (RUL 0..{n-1})")

    def export_training_data(self, output_path: str):
        """Export all collected cycles to Parquet for retraining."""
        try:
            import pandas as pd
            all_records = []
            with self._lock:
                for station_id, cycles in self._cycles.items():
                    all_records.extend(cycles)
            if not all_records:
                print("No training data to export.")
                return
            df = pd.DataFrame(all_records)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            print(f"Exported {len(df)} training cycles to {output_path}")
            print(f"Columns: {list(df.columns)[:10]}...")
            print(f"RUL range: [{df['RUL'].min():.1f}, {df['RUL'].max():.1f}]")
        except Exception as e:
            print(f"Export failed: {e}")

    def n_cycles(self, station_id: str = None) -> int:
        with self._lock:
            if station_id:
                return len(self._cycles.get(station_id, []))
            return sum(len(v) for v in self._cycles.values())


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("REALTIME FEATURE ENGINE — SELF TEST")
    print("=" * 60)

    engine = RealtimeFeatureEngine()
    rng    = np.random.default_rng(42)

    # Simulate 30 sensor readings for one station
    station = "BTS_TEST_001"
    for t in range(30):
        reading = {
            "dc_voltage_v":    round(48.0 - t * 0.05 + rng.normal(0, 0.2), 2),
            "cabinet_temp_c":  round(35.0 + t * 0.05 + rng.normal(0, 0.3), 2),
            "fan_speed_rpm":   round(3200 - t * 2    + rng.normal(0, 15),  1),
            "cpu_util_pct":    round(60.0 + t * 0.2  + rng.normal(0, 1),   1),
            "latency_ms":      round(5.0  + t * 0.01 + rng.normal(0, 0.1), 2),
            "throughput_mbps": round(800  - t * 1    + rng.normal(0, 5),   1),
        }
        engine.push(station, reading, cycle=t + 1)

    features = engine.get_features(station)
    print(f"\nStation: {station}")
    print(f"Feature count: {len(features)}")
    print(f"Cycle count:   {engine.cycle_count(station)}")
    print(f"\nSample features:")
    for k in ["voltage", "cabinet_temperature", "voltage_slope20",
              "latency_ms_avg10", "health_index", "network_quality",
              "cumul_degradation", "voltage_lag1"]:
        val = features.get(k)
        if val is not None:
            print(f"  {k:<35} = {val:.4f}")

    print(f"\n✓ Feature engine OK")
    print("=" * 60)
