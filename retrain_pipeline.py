"""
FluxAgent — retrain_pipeline.py
Continuous Learning & Model Update Pipeline
=============================================
Danaya Diarra | MSc Thesis 2026 | GSOM SPBU

PURPOSE
───────
When new BTS stations are commissioned OR when existing stations accumulate
enough new live cycles, this pipeline:

  1. Loads accumulated sensor readings from data/live_store/*.csv
  2. Runs the same leakage-free feature engineering (from 2_feature_engineering_pipeline.py)
  3. Appends new features to the existing training set
  4. Fine-tunes XGBoost v2 (warm_start) — no full retraining required
  5. Evaluates on a held-out validation window
  6. Promotes new model if RMSE improved, keeps old model if not
  7. Writes new model artifact → picked up by Interpreter Agent on next load

WHEN TO RUN
───────────
  New station commissioned:  run manually after 30 cycles accumulated
  Routine update:            run via cron / Airflow nightly (02:00 local)
  Drift detected:            triggered automatically when drift_detector.py
                             reports PSI > 0.2 on any feature

USAGE
─────
  python retrain_pipeline.py                   # check + retrain if needed
  python retrain_pipeline.py --force           # always retrain
  python retrain_pipeline.py --eval-only       # evaluate without retraining
  python retrain_pipeline.py --station FD005_11 # onboard single new station

REQUIREMENTS
────────────
  pip install xgboost scikit-learn pandas numpy pyarrow
"""

import os
import sys
import time
import json
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import pandas as pd
    PD_OK = True
except ImportError:
    print("ERROR: pandas required — pip install pandas"); sys.exit(1)

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    SKL_OK = True
except ImportError:
    SKL_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FluxAgent-Retrain] %(levelname)s  %(message)s")
log = logging.getLogger("FluxAgent.Retrain")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(".")
LIVE_STORE      = Path("data/live_store")          # connector writes here
FEAT_DIR        = Path("data/features/optimized")  # from 2_feature_engineering_pipeline.py
MODEL_DIR       = Path("models_artifacts/final_models")
RETRAIN_LOG     = Path("data/retrain_log.json")
MIN_CYCLES      = 30           # minimum cycles before a new station is included
VALIDATION_FRAC = 0.15         # fraction held out for validation
RUL_CAP         = 125
RMSE_IMPROVE_THRESHOLD = 0.5   # minimum RMSE improvement (cycles) to promote new model
for d in [LIVE_STORE, FEAT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD & VALIDATE LIVE DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_live_data(station_filter: str = None) -> pd.DataFrame:
    """
    Read all per-station CSV files from data/live_store/.
    Each file has columns: ts, {kpi_col1}, {kpi_col2}, ...
    """
    dfs = []
    csv_files = list(LIVE_STORE.glob("*.csv"))
    if not csv_files:
        log.warning(f"No live data found in {LIVE_STORE}. "
                    f"Run data_connector.py first or set FLUXAGENT_CONNECTOR_MODE=file.")
        return pd.DataFrame()

    for f in csv_files:
        sid = f.stem
        if station_filter and sid != station_filter:
            continue
        try:
            df = pd.read_csv(f)
            df["station_id"] = sid
            if len(df) < MIN_CYCLES:
                log.info(f"  {sid}: only {len(df)} cycles (need {MIN_CYCLES}) — skipping")
                continue
            dfs.append(df)
            log.info(f"  {sid}: {len(df)} cycles loaded")
        except Exception as e:
            log.error(f"  {sid}: failed to load — {e}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING  (same logic as 2_feature_engineering_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

# KPI → feature column name mapping (mirrors the C-MAPSS column rename)
KPI_TO_FEATURE = {
    "dc_voltage_v":       "voltage",
    "battery_cap_pct":    "battery_voltage",
    "rectifier_current_a":"current",
    "power_draw_w":       "total_power_consumption",
    "cabinet_temp_c":     "cabinet_temperature",
    "fan_speed_rpm":      "fan_speed",
    "ambient_temp_c":     "ambient_temperature",
    "rssi_dbm":           "signal_quality",
    "vswr_ratio":         "error_rate",
    "sinr_db":            "signal_strength",
    "pa_efficiency_pct":  "throughput_mbps",
    "latency_ms":         "latency_ms",
    "packet_loss_pct":    "packet_loss",
    "throughput_mbps":    "throughput_mbps",
    "link_util_pct":      "cpu_utilization",
    "cpu_util_pct":       "cpu_utilization",
    "mem_util_pct":       "memory_usage",
    "active_users":       "connected_users",
}

CORE_SENSORS = ["voltage","cabinet_temperature","cpu_utilization","latency_ms","signal_quality"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same leakage-free feature engineering as 2_feature_engineering_pipeline.py
    to live sensor data.
    """
    if df.empty:
        return df

    # Rename KPI columns to feature names
    df = df.rename(columns=KPI_TO_FEATURE)

    # Add cycle index per station (proxy for time_cycle)
    df = df.sort_values("station_id")
    df["time_cycle"] = df.groupby("station_id").cumcount() + 1

    # Compute RUL (will be updated as cycles accumulate)
    max_cycle = df.groupby("station_id")["time_cycle"].transform("max")
    df["RUL"] = (max_cycle - df["time_cycle"]).clip(upper=RUL_CAP)

    # Rolling statistics (windows 5, 10, 20) — same as pipeline
    for sensor in CORE_SENSORS:
        if sensor not in df.columns:
            continue
        grp = df.groupby("station_id")[sensor]
        for w in [5, 10, 20]:
            df[f"{sensor}_avg{w}"] = grp.transform(
                lambda x, win=w: x.rolling(win, min_periods=1).mean())
        df[f"{sensor}_std5"]  = grp.transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
        df[f"{sensor}_min20"] = grp.transform(lambda x: x.rolling(20, min_periods=1).min())

    # Slope features (windows 5, 10, 20)
    for sensor in CORE_SENSORS:
        if sensor not in df.columns:
            continue
        for w in [5, 10, 20]:
            lag = df.groupby("station_id")[sensor].shift(w)
            df[f"{sensor}_slope{w}"] = (df[sensor] - lag) / w

    # Lag features
    for sensor in CORE_SENSORS:
        if sensor not in df.columns:
            continue
        for lag in [1, 3, 5]:
            df[f"{sensor}_lag{lag}"] = df.groupby("station_id")[sensor].shift(lag)

    # Interaction features
    if "memory_usage" in df.columns and "voltage" in df.columns:
        df["mem_x_voltage"] = df["memory_usage"] * df["voltage"]

    if "total_power_consumption" in df.columns and "throughput_mbps" in df.columns:
        df["power_efficiency"] = df["throughput_mbps"] / (df["total_power_consumption"] + 0.001)

    if "throughput_mbps" in df.columns and "latency_ms" in df.columns:
        df["network_quality"]      = df["throughput_mbps"] / (df["latency_ms"] + 1)
        df["throughput_x_latency"] = df["throughput_mbps"] * df["latency_ms"]

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    log.info(f"Feature engineering complete: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — MERGE WITH EXISTING TRAINING SET
# ══════════════════════════════════════════════════════════════════════════════

def merge_with_existing(live_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine live-engineered features with the existing C-MAPSS parquet
    (from 2_feature_engineering_pipeline.py).
    """
    existing_path = FEAT_DIR / "optimized_features_all.parquet"

    if existing_path.exists():
        try:
            existing = pd.read_parquet(existing_path)
            log.info(f"Existing training set: {existing.shape[0]:,} rows")

            # Align columns — only keep columns present in both sets
            common_cols = [c for c in existing.columns if c in live_df.columns]
            merged = pd.concat([existing[common_cols], live_df[common_cols]],
                               ignore_index=True)
            log.info(f"Merged training set: {merged.shape[0]:,} rows")
            return merged
        except Exception as e:
            log.warning(f"Could not load existing features: {e}. Using live data only.")

    log.info("No existing training set found — using live data only.")
    return live_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN / FINE-TUNE XGBOOST v2
# ══════════════════════════════════════════════════════════════════════════════

LEAKAGE_COLS = {"RUL","log_RUL","life_pct","station_id","time_cycle",
                "global_unit","unit","cycle","subset","split","ts"}

XGB_PARAMS = {
    "n_estimators":     15000,
    "learning_rate":    0.02,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "tree_method":      "hist",
    "device":           "cpu",   # change to "cuda" if GPU available
    "early_stopping_rounds": 100,
    "eval_metric":      "rmse",
}


def compute_sample_weights(rul_series: pd.Series, alpha: float = 3.0) -> np.ndarray:
    """Exponential weights: near-failure samples weighted ~4× higher."""
    return np.exp(-alpha * rul_series.values / RUL_CAP)


def train_model(df: pd.DataFrame, force: bool = False) -> dict:
    """
    Fine-tune XGBoost v2 on the merged dataset.
    Returns evaluation metrics dict.
    """
    if not XGB_OK:
        log.error("xgboost not installed — pip install xgboost")
        return {"error": "xgboost missing"}
    if not SKL_OK:
        log.error("scikit-learn not installed — pip install scikit-learn")
        return {"error": "sklearn missing"}

    # Feature selection
    feat_cols = [c for c in df.columns
                 if c not in LEAKAGE_COLS and df[c].dtype != object]
    y = df["RUL"].values

    if len(feat_cols) == 0:
        log.error("No feature columns found after filtering leakage cols")
        return {"error": "no features"}

    X = df[feat_cols].values.astype(np.float32)
    log.info(f"Training on {X.shape[0]:,} rows × {X.shape[1]} features")

    # Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALIDATION_FRAC, random_state=42)
    w_tr = compute_sample_weights(pd.Series(y_tr))

    # Load existing model for warm start (fine-tuning)
    existing_model_path = MODEL_DIR / "xgb_v2.pkl"
    start_model = None
    if existing_model_path.exists() and not force:
        try:
            import pickle
            with open(existing_model_path, "rb") as f:
                start_model = pickle.load(f)
            log.info("Warm-starting from existing xgb_v2.pkl")
        except Exception as e:
            log.warning(f"Could not load existing model: {e}. Training from scratch.")

    model = xgb.XGBRegressor(**{k: v for k, v in XGB_PARAMS.items()
                                 if k != "early_stopping_rounds"})
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        verbose=500,
        early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
        xgb_model=start_model.get_booster() if start_model else None,
    )

    # Evaluate
    y_pred_val = model.predict(X_val)
    rmse_val   = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    mae_val    = float(np.mean(np.abs(y_val - y_pred_val)))

    log.info(f"Validation RMSE: {rmse_val:.2f}  MAE: {mae_val:.2f}")

    return {
        "model":      model,
        "feat_cols":  feat_cols,
        "rmse":       rmse_val,
        "mae":        mae_val,
        "n_train":    len(X_tr),
        "n_val":      len(X_val),
        "n_features": len(feat_cols),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — EVALUATE EXISTING MODEL
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_existing(df: pd.DataFrame) -> float:
    """Evaluate the currently deployed model on the new data."""
    existing_path = MODEL_DIR / "xgb_v2.pkl"
    feat_path     = MODEL_DIR / "feat_cols_xgb.pkl"
    if not existing_path.exists():
        return float("inf")
    try:
        import pickle
        with open(existing_path,  "rb") as f: model     = pickle.load(f)
        with open(feat_path,      "rb") as f: feat_cols = pickle.load(f)
        common = [c for c in feat_cols if c in df.columns]
        X  = df[common].fillna(0).values.astype(np.float32)
        y  = df["RUL"].values
        yp = model.predict(X)
        return float(np.sqrt(mean_squared_error(y, yp)))
    except Exception as e:
        log.warning(f"Could not evaluate existing model: {e}")
        return float("inf")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — PROMOTE OR KEEP MODEL
# ══════════════════════════════════════════════════════════════════════════════

def promote_model(result: dict) -> bool:
    """
    Save new model if it's better than the existing one.
    Keeps a dated backup of the old model before overwriting.
    """
    import pickle
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MODEL_DIR / f"xgb_v2_backup_{ts}.pkl"

    existing_path = MODEL_DIR / "xgb_v2.pkl"
    feat_path     = MODEL_DIR / "feat_cols_xgb.pkl"

    if existing_path.exists():
        shutil.copy(existing_path, backup_path)
        log.info(f"Backed up existing model → {backup_path.name}")

    with open(existing_path, "wb") as f:
        pickle.dump(result["model"], f)
    with open(feat_path, "wb") as f:
        pickle.dump(result["feat_cols"], f)

    log.info(f"New model promoted: RMSE={result['rmse']:.2f}  "
             f"Features={result['n_features']}  Trained on {result['n_train']:,} rows")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — DRIFT DETECTION  (basic PSI check)
# ══════════════════════════════════════════════════════════════════════════════

def check_feature_drift(live_df: pd.DataFrame,
                        reference_path: Path = None,
                        psi_threshold: float = 0.2) -> dict:
    """
    Population Stability Index (PSI) check.
    PSI > 0.2 = significant drift → flag for retraining.

    Reference distribution = existing training set.
    Current distribution   = live_df.
    """
    if reference_path is None:
        reference_path = FEAT_DIR / "optimized_features_all.parquet"
    if not reference_path.exists():
        return {"drift_detected": False, "reason": "no reference distribution"}

    try:
        ref = pd.read_parquet(reference_path)
        drift_cols = {}
        for col in CORE_SENSORS:
            if col not in ref.columns or col not in live_df.columns:
                continue
            ref_hist, bins = np.histogram(ref[col].dropna(), bins=10, density=True)
            live_hist,  _  = np.histogram(live_df[col].dropna(), bins=bins, density=True)
            # PSI formula
            ref_hist  = np.where(ref_hist  == 0, 1e-6, ref_hist)
            live_hist = np.where(live_hist == 0, 1e-6, live_hist)
            psi = np.sum((live_hist - ref_hist) * np.log(live_hist / ref_hist))
            drift_cols[col] = round(float(psi), 4)

        max_psi    = max(drift_cols.values()) if drift_cols else 0
        drifted    = [c for c, p in drift_cols.items() if p > psi_threshold]
        return {
            "drift_detected": max_psi > psi_threshold,
            "max_psi":        max_psi,
            "psi_by_feature": drift_cols,
            "drifted_features": drifted,
        }
    except Exception as e:
        return {"drift_detected": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  RETRAIN LOG
# ══════════════════════════════════════════════════════════════════════════════

def log_retrain_event(event: dict):
    history = []
    if RETRAIN_LOG.exists():
        try:
            history = json.loads(RETRAIN_LOG.read_text())
        except Exception:
            pass
    history.insert(0, {**event, "timestamp": datetime.now().isoformat()})
    RETRAIN_LOG.write_text(json.dumps(history[:50], indent=2))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(force: bool = False, eval_only: bool = False, station: str = None):
    print("=" * 60)
    print("FluxAgent — Continuous Retraining Pipeline")
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Force:   {force}    Eval-only: {eval_only}")
    print("=" * 60)

    # 1. Load live data
    log.info("STEP 1: Loading live sensor data")
    live_raw = load_live_data(station_filter=station)
    if live_raw.empty:
        log.warning("No live data available. Exiting.")
        log_retrain_event({"status": "skipped", "reason": "no live data"})
        return

    # 2. Feature engineering
    log.info("STEP 2: Engineering features")
    live_feat = engineer_features(live_raw)

    # 3. Drift check
    log.info("STEP 3: Checking feature drift (PSI)")
    drift = check_feature_drift(live_feat)
    log.info(f"  Drift detected: {drift['drift_detected']}  "
             f"Max PSI: {drift.get('max_psi', 0):.3f}")
    if drift.get("drifted_features"):
        log.warning(f"  Drifted features: {drift['drifted_features']}")

    if eval_only:
        rmse_existing = evaluate_existing(live_feat)
        log.info(f"EVAL ONLY — existing model RMSE on live data: {rmse_existing:.2f}")
        log_retrain_event({"status": "eval_only", "rmse_existing": rmse_existing, **drift})
        return

    # 4. Merge with C-MAPSS baseline
    log.info("STEP 4: Merging with existing training set")
    merged = merge_with_existing(live_feat)

    # 5. Evaluate existing model on new data
    log.info("STEP 5: Evaluating existing model")
    rmse_existing = evaluate_existing(merged)
    log.info(f"  Existing model RMSE: {rmse_existing:.2f}")

    # 6. Decide whether to retrain
    should_retrain = (
        force
        or drift["drift_detected"]
        or rmse_existing > 16.0   # above acceptable threshold
        or not (MODEL_DIR / "xgb_v2.pkl").exists()
    )

    if not should_retrain:
        log.info("Model is performing well — no retraining needed.")
        log_retrain_event({
            "status": "skipped", "reason": "model ok",
            "rmse_existing": rmse_existing, **drift
        })
        return

    # 7. Train new model
    log.info("STEP 6: Training new XGBoost v2 model")
    t_start = time.time()
    result  = train_model(merged, force=force)
    t_train = time.time() - t_start

    if "error" in result:
        log.error(f"Training failed: {result['error']}")
        return

    log.info(f"  Training time: {t_train:.0f}s  RMSE: {result['rmse']:.2f}")

    # 8. Promote if better
    improvement = rmse_existing - result["rmse"]
    if improvement >= RMSE_IMPROVE_THRESHOLD or force:
        promote_model(result)
        status = "promoted"
        log.info(f"  Model promoted (RMSE improved by {improvement:.2f} cycles)")
    else:
        log.info(f"  Model not promoted (improvement {improvement:.2f} < {RMSE_IMPROVE_THRESHOLD})")
        status = "rejected"

    log_retrain_event({
        "status":        status,
        "rmse_existing": round(rmse_existing, 2),
        "rmse_new":      result["rmse"],
        "improvement":   round(improvement, 2),
        "n_train":       result["n_train"],
        "n_features":    result["n_features"],
        "train_time_s":  round(t_train, 1),
        **drift,
    })

    print("=" * 60)
    print(f"  Result:    {status.upper()}")
    print(f"  Old RMSE:  {rmse_existing:.2f}")
    print(f"  New RMSE:  {result['rmse']:.2f}  (Δ {improvement:+.2f})")
    print(f"  Log:       {RETRAIN_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FluxAgent — Continuous Retraining Pipeline")
    parser.add_argument("--force",     action="store_true", help="Force retrain even if model is ok")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate only, no training")
    parser.add_argument("--station",   type=str, default=None, help="Onboard single new station ID")
    args = parser.parse_args()
    run(force=args.force, eval_only=args.eval_only, station=args.station)
