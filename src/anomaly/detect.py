"""
detect.py — anomaly detection on electricity load.

Uses Isolation Forest trained on calendar + load lag features.
An "anomaly" is an hour whose load is unusual given the time of day,
day of week, and recent load history.

Produces:
  models/anomaly_model.pkl
  models/anomaly_features.pkl
  reports/anomaly_report.png

Run standalone:
    python src/anomaly/detect.py
"""

import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "forecasting"))
from features import add_calendar, add_load_lags  # noqa: E402

MODELS  = ROOT / "models";  MODELS.mkdir(exist_ok=True)
REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)
DATA    = ROOT / "data" / "energy_dataset_real.csv"

FEATURE_COLS = [
    "hour", "dayofweek", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "load_mw_lag24", "load_mw_lag168", "load_mw_roll24",
]


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = add_calendar(d)
    d = add_load_lags(d, "load_mw")
    d["load_mw_norm"] = d["load_mw"]   # keep raw value too
    d = d.dropna(subset=FEATURE_COLS + ["load_mw"])
    return d


def load_anomaly_model():
    model = joblib.load(MODELS / "anomaly_model.pkl")
    feats = joblib.load(MODELS / "anomaly_features.pkl")
    return model, feats


def predict_anomalies(df: pd.DataFrame, model, feats):
    """Return (scores, flags) arrays aligned to df.index.
    scores: lower (more negative) = more anomalous.
    flags: boolean array, True = anomaly.
    """
    d = build_anomaly_features(df)
    common = [f for f in feats if f in d.columns]
    X = d[common].reindex(df.index).ffill().fillna(0)
    scores = model.score_samples(X)
    flags  = model.predict(X) == -1
    return scores, flags


def main():
    df = pd.read_csv(DATA, parse_dates=["datetime"]).set_index("datetime")
    d  = build_anomaly_features(df)
    X  = d[FEATURE_COLS]

    # contamination=0.02 → flags ~2% of hours as anomalies
    model = IsolationForest(n_estimators=200, contamination=0.02,
                            random_state=42, n_jobs=-1)
    model.fit(X)
    scores = model.score_samples(X)
    flags  = model.predict(X) == -1

    n_anom = flags.sum()
    print(f"Anomalies detected: {n_anom} / {len(d)}  ({100*n_anom/len(d):.1f}%)")

    joblib.dump(model,        MODELS / "anomaly_model.pkl")
    joblib.dump(FEATURE_COLS, MODELS / "anomaly_features.pkl")

    # plot last 30 days
    last = d.iloc[-720:]
    sc   = scores[-720:]
    fl   = flags[-720:]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    a1.plot(last.index, last["load_mw"], color="#6b7280", lw=1, label="Load")
    a1.scatter(last.index[fl], last["load_mw"].values[fl],
               color="#ef4444", zorder=5, s=40, label="Anomaly")
    a1.set_ylabel("Load (MW)"); a1.legend(); a1.grid(alpha=0.3)
    a1.set_title(f"Load anomalies — last 30 days  ({fl.sum()} flagged)")

    a2.plot(last.index, sc, color="#7c3aed", lw=0.8, label="Anomaly score")
    threshold = np.percentile(scores, 2)
    a2.axhline(threshold, color="#ef4444", ls="--", lw=1, label=f"Threshold ({threshold:.3f})")
    a2.set_ylabel("Score (lower = more anomalous)"); a2.legend(); a2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(REPORTS / "anomaly_report.png", dpi=120)
    print("Saved models/anomaly_model.pkl + reports/anomaly_report.png")


if __name__ == "__main__":
    main()
