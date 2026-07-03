"""
BESS preprocessing:
          - Read monthly CSV(s)
          - Parse 'Time'
          - Resample from 1s to 1min
          - Summarize & optionally handle NaNs
          - Save per-file outputs
"""
from pathlib import Path
import pandas as pd
import numpy as np

pd.options.display.width = 180
pd.options.display.max_columns = 50

def drop_nan(df):
    """
    Drop any rows containing NaN entries
    """
    df = df.copy()
    df = df.replace(['', 'NaN', '?', 'None'], np.nan)
    df = df.dropna()
    return df


def check_valid_time(df, time_column = "Time"):
    """
    Ensure valid time format and monotonic increasing time intervals
    """
    df = df.copy()
    dt = pd.to_datetime(df[time_column], errors="coerce")

    if dt.isna().any():
        raise ValueError("Invalid datetime values detected")

    if not dt.is_monotonic_increasing:
        df = df.sort_values(by=time_column)

    df[time_column] = dt
    df.set_index(time_column, inplace = True)
    return df


def roll_back(df, rule = "1min"):
    """
    Resample from 1-second to 1-minute.
    By default: numeric columns -> mean.
    For common power columns, we also compute sums (kWh/min) alongside.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Build base resample (mean)
    agg_mean = df[num_cols].resample(rule).mean()

    power_series = None
    power_series = pd.to_numeric(df["P_in_W"], errors="coerce").astype("float64")

    P = power_series.interpolate(limit=3).fillna(0.0)

    P_pos = np.maximum(P, 0.0)   # charge W
    P_neg = np.maximum(-P, 0.0)  # discharge W

    energy_charge_kWh    = P_pos.resample(rule).sum() / 3_600_000.0
    energy_discharge_kWh = P_neg.resample(rule).sum() / 3_600_000.0
    net_energy_kWh       = (P.resample(rule).sum()   / 3_600_000.0)

    agg_mean["energy_discharge_kWh"]  = energy_discharge_kWh
    agg_mean["energy_throughput_kWh"] = energy_charge_kWh + energy_discharge_kWh
    agg_mean["net_energy_kWh"]        = net_energy_kWh

    for c in ["V_in_V", "I_in_A", "T_Bat_in_C", "T_Room_in_C"]:
        if c in df.columns:
            agg_mean[f"{c}_std"] = df[c].resample(rule).std()

    return agg_mean


def process_one_csv(in_path: Path, out_dir: Path, rule: str = "1min") -> Path:
    """
    Pipeline for a single monthly CSV:
      read → clean time → drop columns → numeric cast → resample → features → write CSV
    """
    print(f"[READ] {in_path.name}")
    df = pd.read_csv(in_path)

    # 1) validate/parse time
    time_col = "Time"
    df = check_valid_time(df, time_column=time_col)

    # 2) drop undesired columns (e.g., 'Interpolated') and cast numerics
    df.drop(columns = "Interpolated", inplace = True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) resample 1s → 1min
    df_min = roll_back(df, rule=rule)

    # 4) write output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / in_path.name.replace(".csv", f"_{rule}.csv")
    df_min.to_csv(out_path, index=True)
    print(f"[WRITE] {out_path.name}  shape={df_min.shape}")
    return out_path

if __name__ == "__main__":
    DATA_DIR = Path("data/bess_data")
    OUT_DIR  = Path("data/processed")
    PATTERN  = "201*_*_System_ID_18.csv"
    RULE     = "1min"

    files = sorted(DATA_DIR.glob(PATTERN))

    for f in files:
        try:
            process_one_csv(f, OUT_DIR, rule=RULE)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
