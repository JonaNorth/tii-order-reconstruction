import argparse
import glob
import os
import pandas as pd
import numpy as np
from scipy.signal import welch

"""
Baseline approach (very simple):
- Load each file_i.csv (expects a column named 'acc' OR the first column to be acceleration).
- Compute two features:
  1) RMS of acceleration
  2) Spectral band energy around nominal fault bands (Hz): [231, 3781, 5781, 4408]
- Combine to a single score and rank ascending (early) -> descending (late), assuming damage features increase over time.
- Output submission.csv with a single column 'prediction' of length N, where the row index order is file_1 ... file_N.
NOTE: This is a starter. You **must** adapt to the actual column names/format of the real dataset.
"""

NOMINAL_FS = 93750.0  # Hz (given)
FAULT_BANDS_HZ = [231, 3781, 5781, 4408]  # centers; we'll use narrow +/- 50 Hz windows as a simple start
BAND_HALF_WIDTH = 50.0

def feature_peak_abs(x):
    # Classical peak: highest absolute amplitude
    return float(np.max(np.abs(x)))

def feature_crest_factor(x, robust=False, q=99.9):
    """
    Crest factor = peak / RMS
    robust=True -> brug percentil-peak (fx 99.9%) for at undgå outliers.
    """
    rms = feature_rms(x)
    if rms == 0:
        return float("inf")  # eller en meget stor værdi, hvis du vil undgå inf
    if robust:
        p = float(np.percentile(np.abs(x), q))
        return p / rms
    else:
        p = feature_peak_abs(x)
        return p / rms

def read_acc_series(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    # Try to find acceleration column heuristically
    for col in df.columns:
        lc = col.lower()
        if 'acc' in lc or 'accel' in lc:
            return df[col].values.astype(float)
    # Fallback: take the first numeric column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return df[col].values.astype(float)
    raise ValueError(f"No numeric/acc column found in {csv_path}")

def feature_rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))

def feature_band_energy(x: np.ndarray, fs: float, centers, half_width: float) -> float:
    # Use Welch PSD and integrate around small bands
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 4096))
    energy = 0.0
    for c in centers:
        lo, hi = c - half_width, c + half_width
        mask = (f >= lo) & (f <= hi)
        energy += np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0
    return float(energy)

def main(data_dir: str, out_csv: str):
    # Collect files in the expected order of file_1 ... file_N by filename index
    paths = sorted(glob.glob(os.path.join(data_dir, "file_*.csv")),
                   key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
    if not paths:
        raise SystemExit(f"No files found like {data_dir}/file_*.csv")

    rows = []
    for p in paths:
        acc = read_acc_series(p)
        rms = feature_rms(acc)
        band_e = feature_band_energy(acc, NOMINAL_FS, FAULT_BANDS_HZ, BAND_HALF_WIDTH)

        # Crest factor (robust: brug percentil-peak for at undgå enkelt-spikes)
        crest = feature_crest_factor(acc, robust=True, q=99.9)

        rows.append({
        "path": p,
        "rms": rms,
        "band_energy": band_e,
        "crest": crest
})


    df = pd.DataFrame(rows)

    # Z-score normalisering, så ingen feature dominerer kun pga. skala
    for col in ["rms", "band_energy", "crest"]:
        mu = df[col].mean()
        sd = df[col].std(ddof=0) or 1.0
        df[f"z_{col}"] = (df[col] - mu) / sd
    
    # Vægte: giv mest vægt til RMS, næstmest crest, lidt til band_energy
    W_RMS, W_CREST, W_BAND = 0.6, 0.3, 0.1
    df["score"] = (
        W_RMS   * df["z_rms"] +
        W_CREST * df["z_crest"] +
        W_BAND  * df["z_band_energy"]
    )
    
    # Sortér lav -> høj (tidlig -> sen), byg positionsmap og skriv submission
    df_sorted = df.sort_values("score", ascending=True).reset_index(drop=True)
    pos_map = {row["path"]: (idx + 1) for idx, row in df_sorted.iterrows()}
    
    # paths er allerede i fil-nummer-rækkefølge; lav permutationsvektor
    predictions = [pos_map[p] for p in paths]
    
    # Debug-fil så du kan se alle features og score
    df.to_csv("features_debug.csv", index=False)
    
    # Submission
    out = pd.DataFrame({"prediction": predictions})
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="submission.csv")
    args = ap.parse_args()
    main(args.data_dir, args.out)
