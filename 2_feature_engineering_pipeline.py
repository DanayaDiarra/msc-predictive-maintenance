#!/usr/bin/env python3
"""
===============================================================================
STANDALONE FEATURE ENGINEERING PIPELINE — LEAKAGE-FREE
===============================================================================
Fixes vs previous version:
  1. LEAKAGE REMOVED: life_pct and log_RUL excluded from saved feature set.
       life_pct = time_cycle / max_cycle  ≈ 1 - RUL/max_cycle  → direct RUL proxy
       log_RUL  = log(RUL + 1)           → direct transformation of target
     Both are saved separately for reference but NOT included in feature_cols.

  2. cumul_degradation retained but flagged: it is a soft-leakage risk because
     it accumulates monotonically with time_cycle. It is kept in because it uses
     only power deviation from early-life baseline (causal, not future-looking),
     but it should be ablated in experiments.

  3. All other fixes from previous version retained:
       - global_unit element-wise concat (not f-string on Series)
       - Piecewise-linear RUL cap at 125 cycles
       - Per-engine min-max health_index
       - Multi-window rolling stats (5, 10, 20 cycles)
       - Multi-window slopes (5, 10, 20 cycles)
       - Sensor interaction features

OUTPUTS:
    data/features/optimized/optimized_features_all.parquet
        → Contains only clean, non-leaky features + RUL target + ID columns.

USAGE:
    python 2_feature_engineering_pipeline_fixed.py
===============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
print("=" * 80)
print("FEATURE ENGINEERING PIPELINE — LEAKAGE-FREE VERSION")
print("=" * 80)

RAW_DIR    = Path("data/raw")
OUTPUT_DIR = Path("data/features/optimized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

NASA_TO_TELECOM = {
    'unit':       'station_id',
    'cycle':      'time_cycle',
    'setting1':   'antenna_power',
    'setting2':   'signal_strength',
    'setting3':   'antenna_tilt',
    'sensor_01':  'cabinet_temperature',
    'sensor_02':  'ambient_temperature',
    'sensor_03':  'humidity',
    'sensor_04':  'voltage',
    'sensor_05':  'current',
    'sensor_06':  'fan_speed',
    'sensor_07':  'cooling_efficiency',
    'sensor_08':  'battery_voltage',
    'sensor_09':  'total_power_consumption',
    'sensor_10':  'cpu_utilization',
    'sensor_11':  'memory_usage',
    'sensor_12':  'disk_usage',
    'sensor_13':  'throughput_mbps',
    'sensor_14':  'latency_ms',
    'sensor_15':  'packet_loss',
    'sensor_16':  'error_rate',
    'sensor_17':  'signal_quality',
    'sensor_18':  'connected_users',
    'sensor_19':  'data_volume',
    'sensor_20':  'operational_hours',
    'sensor_21':  'component_wear',
}

CORE_SENSORS = [
    'total_power_consumption',
    'throughput_mbps',
    'latency_ms',
    'cabinet_temperature',
    'cpu_utilization',
]

RUL_CAP = 125   # Piecewise-linear cap — standard C-MAPSS practice

# Columns that must NEVER appear as model features (leakage / identifiers)
LEAKAGE_COLS = {
    'RUL',          # target
    'log_RUL',      # log transform of target  ← LEAKAGE
    'life_pct',     # time_cycle/max_cycle ≈ 1-RUL/max_cycle  ← LEAKAGE
    'station_id',   # engine identifier (string)
    'time_cycle',   # raw cycle index — leaks lifecycle position directly
    'global_unit',  # internal engine ID
    'unit',         # original NASA unit column (if still present)
    'cycle',        # original NASA cycle column (if still present)
    'subset',       # dataset label
    'split',        # train/val/test split label
}

# ============================================================================
# STEP 1: LOAD ALL RAW DATA
# ============================================================================
print("\nSTEP 1: LOADING ALL SUBSETS")
print("-" * 50)

nasa_columns = (
    ['unit', 'cycle']
    + [f'setting{i}' for i in range(1, 4)]
    + [f'sensor_{i:02d}' for i in range(1, 22)]
)

all_dfs = []

for subset in SUBSETS:
    file_path = RAW_DIR / f'train_{subset}.txt'
    if not file_path.exists():
        print(f"   Warning: {file_path} not found, skipping")
        continue
    print(f"   Loading {subset}...")
    df_sub = pd.read_csv(file_path, sep=r'\s+', header=None, names=nasa_columns)
    df_sub['subset'] = subset
    # Element-wise string concat — NOT f-string on Series (that collapses all engines)
    df_sub['global_unit'] = subset + "_" + df_sub['unit'].astype(str)
    all_dfs.append(df_sub)
    print(f"      -> {len(df_sub):,} rows, {df_sub['unit'].nunique()} engines")

df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTOTAL: {len(df):,} rows")
print(f"   Unique engines (global_unit): {df['global_unit'].nunique()}")
for subset in SUBSETS:
    n_eng = df.loc[df['subset'] == subset, 'global_unit'].nunique()
    print(f"   {subset}: {n_eng} engines")

# ============================================================================
# STEP 2: APPLY TELECOM MAPPING
# ============================================================================
print("\nSTEP 2: APPLYING TELECOM MAPPING")
print("-" * 50)

df = df.rename(columns=NASA_TO_TELECOM)
print("Telecom column mapping applied.")

# ============================================================================
# STEP 3: COMPUTE RUL TARGET
# ============================================================================
print("\nSTEP 3: COMPUTING RUL TARGET")
print("-" * 50)

max_cycle = df.groupby('global_unit')['time_cycle'].transform('max')
df['RUL']  = (max_cycle - df['time_cycle']).clip(upper=RUL_CAP)

raw_max = (max_cycle - df['time_cycle']).max()
capped  = (df['RUL'] == RUL_CAP).sum()
print(f"RUL cap applied: raw max={raw_max} → capped at {RUL_CAP}")
print(f"Rows at cap: {capped:,} ({capped/len(df)*100:.1f}%)")
print(f"RUL range: [{df['RUL'].min()}, {df['RUL'].max()}]")

# ── Derived target columns kept for reference ONLY — excluded from features ──
# life_pct and log_RUL are useful for analysis/plotting but are NOT features.
df['life_pct'] = df['time_cycle'] / max_cycle   # saved for reference only
df['log_RUL']  = np.log1p(df['RUL'])            # saved for reference only
print("\n[NOTE] life_pct and log_RUL saved for analysis but EXCLUDED from features.")

# ============================================================================
# STEP 4: TELECOM METRICS
# ============================================================================
print("\nSTEP 4: CREATING TELECOM METRICS")
print("-" * 50)

df['network_quality']  = df['throughput_mbps'] / (df['latency_ms'] + 1)
df['power_efficiency'] = df['throughput_mbps'] / (df['total_power_consumption'] + 0.001)
print("Created: network_quality, power_efficiency")

# ============================================================================
# STEP 5: LAG FEATURES
# ============================================================================
print("\nSTEP 5: CREATING LAG FEATURES")
print("-" * 50)

for sensor in CORE_SENSORS:
    for lag in [1, 3, 5]:
        df[f"{sensor}_lag{lag}"] = df.groupby('global_unit')[sensor].shift(lag)
    print(f"   {sensor}: lags 1, 3, 5")

# ============================================================================
# STEP 6: ROLLING STATISTICS
# ============================================================================
print("\nSTEP 6: CREATING ROLLING STATISTICS")
print("-" * 50)

for sensor in CORE_SENSORS:
    grp = df.groupby('global_unit')[sensor]
    for window in [5, 10, 20]:
        df[f"{sensor}_avg{window}"] = grp.transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean()
        )
    df[f"{sensor}_std5"]  = grp.transform(
        lambda x: x.rolling(5, min_periods=1).std().fillna(0)
    )
    df[f"{sensor}_min20"] = grp.transform(
        lambda x: x.rolling(20, min_periods=1).min()
    )
    print(f"   {sensor}: avg5/10/20, std5, min20")

# ============================================================================
# STEP 7: TREND / SLOPE FEATURES
# ============================================================================
print("\nSTEP 7: CREATING SLOPE FEATURES")
print("-" * 50)

for sensor in CORE_SENSORS:
    for window in [5, 10, 20]:
        lag = df.groupby('global_unit')[sensor].shift(window)
        df[f"{sensor}_slope{window}"] = (df[sensor] - lag) / window
    print(f"   {sensor}: slopes at 5, 10, 20 cycles")

# ============================================================================
# STEP 8: HEALTH INDEX (per-engine min-max normalised)
# ============================================================================
print("\nSTEP 8: CREATING HEALTH INDEX")
print("-" * 50)

required = ['signal_quality', 'cabinet_temperature', 'cpu_utilization']
if all(col in df.columns for col in required):
    def minmax(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    sig_norm  = df.groupby('global_unit')['signal_quality'].transform(minmax)
    temp_norm = df.groupby('global_unit')['cabinet_temperature'].transform(minmax)
    cpu_norm  = df.groupby('global_unit')['cpu_utilization'].transform(minmax)
    df['health_index'] = (
        0.5 * sig_norm - 0.25 * temp_norm - 0.25 * cpu_norm + 0.5
    ).clip(0, 1)
    print(f"health_index range: [{df['health_index'].min():.4f}, {df['health_index'].max():.4f}]")
    if df['health_index'].std() < 0.01:
        print("  ⚠ health_index has very low variance — sensor mappings may need review.")
else:
    missing = [c for c in required if c not in df.columns]
    print(f"⚠ Skipped — missing: {missing}")

# ============================================================================
# STEP 9: INTERACTION FEATURES
# ============================================================================
print("\nSTEP 9: CREATING INTERACTION FEATURES")
print("-" * 50)

if 'memory_usage' in df.columns and 'voltage' in df.columns:
    df['mem_x_voltage']   = df['memory_usage'] * df['voltage']
    df['mem_div_voltage'] = df['memory_usage'] / (df['voltage'] + 1e-9)
    print("   memory_usage × voltage, memory_usage / voltage")

if 'total_power_consumption' in df.columns and 'cooling_efficiency' in df.columns:
    df['power_per_cooling'] = (
        df['total_power_consumption'] / (df['cooling_efficiency'] + 1e-9)
    )
    print("   total_power_consumption / cooling_efficiency")

if 'throughput_mbps' in df.columns and 'latency_ms' in df.columns:
    df['throughput_x_latency'] = df['throughput_mbps'] * df['latency_ms']
    print("   throughput_mbps × latency_ms")

# ============================================================================
# STEP 10: CUMULATIVE DEGRADATION INDEX
# ============================================================================
print("\nSTEP 10: CUMULATIVE DEGRADATION INDEX")
print("-" * 50)
print("   [NOTE] Causal feature (uses only past observations per engine).")
print("   Soft leakage risk: accumulates monotonically — ablate in experiments.")

if 'total_power_consumption' in df.columns:
    baseline = (
        df.sort_values(['global_unit', 'time_cycle'])
          .groupby('global_unit')['total_power_consumption']
          .transform(lambda x: x.iloc[:10].mean())
    )
    power_dev = (df['total_power_consumption'] - baseline).abs()
    df['cumul_degradation'] = (
        df.sort_values(['global_unit', 'time_cycle'])
          .groupby('global_unit')['power_deviation']
          .transform(lambda x: x.cumsum())
        if 'power_deviation' in df.columns
        else df.assign(power_deviation=power_dev)
               .sort_values(['global_unit', 'time_cycle'])
               .groupby('global_unit')['power_deviation']
               .transform(lambda x: x.cumsum())
    )
    print(f"   cumul_degradation range: [{df['cumul_degradation'].min():.1f}, {df['cumul_degradation'].max():.1f}]")

# ============================================================================
# STEP 11: CLEAN DATA
# ============================================================================
print("\nSTEP 11: CLEANING DATA")
print("-" * 50)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.sort_values(['global_unit', 'time_cycle'])
df = (
    df.groupby('global_unit', group_keys=False)
      .apply(lambda x: x.ffill().bfill())
)
df = df.fillna(0)
print(f"   Nulls after cleaning: {df.isnull().sum().sum():,}")

# ============================================================================
# STEP 12: VERIFY NO LEAKAGE
# ============================================================================
print("\nSTEP 12: LEAKAGE VERIFICATION")
print("-" * 50)

# Determine clean feature columns
all_cols     = list(df.columns)
feature_cols = [
    c for c in all_cols
    if c not in LEAKAGE_COLS
    and df[c].dtype != object
]

# Double-check: flag anything correlated with RUL in name
suspicious = [c for c in feature_cols if any(x in c.lower() for x in ['rul', 'log_r', 'life_pct'])]
if suspicious:
    print(f"  ⚠ WARNING — still-suspicious columns: {suspicious}")
    feature_cols = [c for c in feature_cols if c not in suspicious]
else:
    print("  ✓ No RUL-derived columns in feature set")

# Verify life_pct and log_RUL are excluded
for col in ['life_pct', 'log_RUL']:
    status = "✓ EXCLUDED" if col not in feature_cols else "✗ STILL PRESENT — BUG"
    print(f"  {col}: {status}")

print(f"\n  Clean feature count: {len(feature_cols)}")

# ============================================================================
# STEP 13: EXPORT
# ============================================================================
print("\nSTEP 13: EXPORTING")
print("-" * 50)

# Keep: features + RUL target + essential ID columns + reference cols
keep_cols = (
    ['global_unit', 'station_id', 'time_cycle', 'subset']
    + feature_cols
    + ['RUL']
    # life_pct and log_RUL saved for analysis but clearly labelled
    + [c for c in ['life_pct', 'log_RUL'] if c in df.columns]
)
# Deduplicate while preserving order
seen = set()
keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

output_df   = df[keep_cols].copy()
output_path = OUTPUT_DIR / "optimized_features_all.parquet"
output_df.to_parquet(output_path, index=False)

file_size = output_path.stat().st_size / (1024 * 1024)
print(f"Saved: {output_path}")
print(f"   {file_size:.1f} MB  |  {output_df.shape[0]:,} rows × {output_df.shape[1]} cols")

# ============================================================================
# STEP 14: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(f"\nTotal rows:    {len(output_df):,}")
print(f"Total engines: {output_df['global_unit'].nunique()}")
print(f"Clean features:{len(feature_cols)}")
print(f"\nEngines by subset:")
for subset in SUBSETS:
    mask = output_df['subset'] == subset
    engs = output_df.loc[mask, 'global_unit'].nunique()
    rows = mask.sum()
    print(f"   {subset}: {engs:3d} engines  {rows:7,} rows")

print(f"\nFeature breakdown:")
cats = {
    'Original sensors':  lambda c: not any(x in c for x in ['lag','avg','std','slope','min20','health','network','power_eff','mem_x','mem_div','power_per','throughput_x','cumul','subset_encoded']),
    'Lag features':      lambda c: 'lag'   in c,
    'Rolling stats':     lambda c: any(x in c for x in ['avg','std','min20']),
    'Slope features':    lambda c: 'slope' in c,
    'Interactions':      lambda c: any(x in c for x in ['mem_x','mem_div','power_per','throughput_x']),
    'Cumul degradation': lambda c: 'cumul' in c,
    'Telecom metrics':   lambda c: c in ['network_quality','power_efficiency'],
    'Health index':      lambda c: c == 'health_index',
    'Subset encoded':    lambda c: 'subset_encoded' in c,
}
for label, fn in cats.items():
    n = len([c for c in feature_cols if fn(c)])
    if n > 0:
        print(f"   {label:<22}: {n}")

print(f"\n  ✓ life_pct excluded (leakage)")
print(f"  ✓ log_RUL  excluded (leakage)")
print(f"  ✓ time_cycle excluded (direct lifecycle proxy)")
print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE — LEAKAGE-FREE")
print("=" * 80)