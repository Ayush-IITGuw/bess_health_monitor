"""
Exploratory Data Analysis (BESS) on minute-level files created by preprocess.py
  1) Load minute-level CSVs for System_ID_18 across 2017–2018.
  2) Compute window-level health proxies (30/45 min):
        - ΔT per kWh over window (Gives hint about health of battery)
  3) Generates plots of variaton of temperature, output KWh with time
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR      = Path("data/processed")
OUT_DIR       = Path("outputs/eda")
PATTERN       = "201*_System_ID_18_1min.csv"
WINDOWS       =  "45min"
TSAFE         = 28.0
EDIS_MIN_45M  = 0.01

plt.rcParams["figure.figsize"] = (14, 4)
plt.rcParams["axes.grid"] = True

def load_minute_files(data_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(pattern))
    assert files, f"No files matched: {data_dir}/{pattern}"

    dfs = []
    for f in files:
        print(f"[READ] {f.name}")
        dfi = pd.read_csv(f, parse_dates=True, index_col=0)
        for c in dfi.columns:
            dfi[c] = pd.to_numeric(dfi[c], errors="coerce")
        dfs.append(dfi)

    df = pd.concat(dfs, axis=0).sort_index()

    df = df[~df.index.duplicated(keep="first")]
    print("[INFO] Minute-level shape:", df.shape)
    return df

def battery_health(df: pd.DataFrame,window: str = "45min",edis_min_kwh: float = 0.05) -> pd.DataFrame:
    """
    Compute 45‑minute window health proxies from minute‑level data.

    Required minute-level columns in `df`:
        - T_Bat_in_C
        - T_Room_in_C
        - energy_discharge_kWh   (per-minute discharge energy)

    Returns a DataFrame (index aligned to df.index) with:
        - T_Bat_<window>_mean
        - P_in_W_<window>_mean
        - deltaT_<window>_end_minus_start
        - Edis_<window>_kWh
        - temp_rise_per_kWh_<window>
    """
    d = df.copy()

    # Ensure numeric types
    d["T_Bat_in_C"] = pd.to_numeric(d["T_Bat_in_C"], errors="coerce").astype("float64")
    d["T_Room_in_C"] = pd.to_numeric(d["T_Room_in_C"], errors="coerce").astype("float64")
    d["energy_discharge_kWh"] = pd.to_numeric(d["energy_discharge_kWh"], errors="coerce").astype("float64")

    # Ensure clean, monotonic time index
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()
    if d.index.duplicated().any():
        d = d.groupby(level=0).mean()

    # Output frame
    out = pd.DataFrame(index=d.index)


   #  1) context means over the window 
    out[f"T_Bat_in_C_{window}_mean"] = d["T_Bat_in_C"].resample(window).mean()
    if "P_in_W" in d.columns:
            out[f"P_in_W_{window}_mean"] = pd.to_numeric(d["P_in_W"], errors="coerce").resample(window).mean()

    # 2) Per-minute ΔT and window end‑minus‑start ΔT 
    d["deltaT"] = d["T_Bat_in_C"] - d["T_Room_in_C"]

    # min_periods ~ 1/3 of window (≈15 min) to avoid almost-empty windows
    out[f"deltaT_{window}_end_minus_start"] = (
        d["deltaT"].rolling(window, min_periods=15)
                  .apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    )

    #  3) Discharge energy summed over the window 
    out[f"Edis_{window}_kWh"] = (
        d["energy_discharge_kWh"].rolling(window, min_periods=15).sum()
    )

    #  4) Temperature rise per kWh (masked for low energy) 
    EPS = 1e-9
    valid = out[f"Edis_{window}_kWh"] > float(edis_min_kwh)
    trpk = out[f"deltaT_{window}_end_minus_start"] / (out[f"Edis_{window}_kWh"] + EPS)
    out[f"temp_rise_per_kWh_{window}"] = np.where(valid, trpk, np.nan)

    return out


def plot_each_series(out: pd.DataFrame,out_dir: str | Path = "outputs/eda/series",prefix: str = "system18", dpi: int = 160) -> None:
    """
    Saves one PNG per numeric column in `out`.
    File name: <prefix>_<column>.png
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        s = pd.to_numeric(out[col], errors="coerce")

        fig, ax = plt.subplots(figsize=(14, 4))
        s.plot(ax=ax, lw=1.2, color="tab:blue")
        ax.set_title(col)
        ax.set_ylabel(col)
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_png = out_dir / f"{prefix}_{col}.png"
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        print("[PLOT]", out_png)

def run_eda_pipeline(
    data_dir: str | Path = "data/processed",
    pattern: str = "201*_System_ID_18_1min.csv",
    out_dir: str | Path = "outputs/eda/45min",
    window: str = "45min",
    edis_min_kwh: float = 0.05,
    plot_prefix: str = "sys18_45m"
) -> Path:
    """
    Orchestrates EDA:
      1) load minute-level CSVs
      2) concat + clean time index
      3) compute window-level health features
      4) save features CSV
      5) plot each numeric series vs time

    Returns:
      Path to the saved features CSV.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #  1) Load and concat all minute-level files 
    files = sorted(data_dir.glob(pattern))
    assert files, f"No files matched: {data_dir}/{pattern}"

    dfs = []
    for f in files:
        print(f"[READ] {f.name}")
        dfi = pd.read_csv(f, index_col=0, parse_dates=True, low_memory=False)
        # numeric coercion (robust)
        for c in dfi.columns:
            dfi[c] = pd.to_numeric(dfi[c], errors="coerce")
        dfs.append(dfi)

    df_min = pd.concat(dfs, axis=0)

    # 2) Ensure clean, monotonic DatetimeIndex 
    if not isinstance(df_min.index, pd.DatetimeIndex):
        df_min.index = pd.to_datetime(df_min.index, errors="coerce")
    df_min = df_min[~df_min.index.isna()].sort_index()
    if df_min.index.duplicated().any():
        df_min = df_min.groupby(level=0).mean()  

    print("[INFO] Minute-level shape:", df_min.shape)

    # 3) Compute window-level health features 
    feat = battery_health(
        df_min,
        window=window,
        edis_min_kwh=edis_min_kwh,
    )

    # 4) Save features 
    feat_csv = out_dir / f"window_features_{window}.csv"
    feat.to_csv(feat_csv, index=True)
    print("[WRITE]", feat_csv)

    #  5) Plot every numeric series 
    plot_each_series(
        feat,
        out_dir=out_dir / "series",
        prefix=plot_prefix,
        dpi=160
    )

    return feat_csv

if __name__ == "__main__":
    run_eda_pipeline(
        data_dir="data/processed",
        pattern="201*_System_ID_18_1min.csv",
        out_dir="outputs/eda/45min",
        window="45min",
        edis_min_kwh=0.05,
        plot_prefix="sys18_45m"
    )
