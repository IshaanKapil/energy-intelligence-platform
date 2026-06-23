"""
train_load.py — train the load forecasting model.

Produces:
  models/load_model.pkl         point forecast (median)
  models/load_q10.pkl, q90.pkl  quantile models for uncertainty bands
  models/load_metrics.json      MAE / RMSE / MAPE on held-out test
  reports/load_forecast.png     actual vs predicted with 80% interval
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import build_load_features, time_split

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "energy_dataset_real.csv"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def main():
    df = pd.read_csv(DATA, parse_dates=["datetime"]).set_index("datetime")
    X, y, idx = build_load_features(df)
    # 3-way chronological split: train (70%) / calibration (10%) / test (20%)
    n = len(idx)
    c1, c2 = int(n * 0.70), int(n * 0.80)
    Xtr, ytr = X.iloc[:c1], y.iloc[:c1]
    Xcal, ycal = X.iloc[c1:c2], y.iloc[c1:c2]
    Xte, yte = X.iloc[c2:], y.iloc[c2:]
    print(f"train={len(Xtr)}  calib={len(Xcal)}  test={len(Xte)}  features={X.shape[1]}")

    common = dict(n_estimators=600, learning_rate=0.05, num_leaves=64,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

    # ---- point forecast (median) ----
    model = lgb.LGBMRegressor(**common)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    # ---- quantile models for uncertainty (10th & 90th pct -> 80% interval) ----
    # More regularization than the point model: quantile fits overfit easily and
    # collapse the interval, so we use fewer leaves + larger leaf size so the
    # 80% band is honestly calibrated on unseen data.
    qparams = dict(objective="quantile", n_estimators=400, learning_rate=0.05,
                   num_leaves=20, min_child_samples=200, subsample=0.8,
                   colsample_bytree=0.8, random_state=42, verbose=-1)
    q10 = lgb.LGBMRegressor(alpha=0.1, **qparams).fit(Xtr, ytr)
    q90 = lgb.LGBMRegressor(alpha=0.9, **qparams).fit(Xtr, ytr)

    # ---- Conformalized Quantile Regression (CQR) ----
    # Widen the raw quantile band by a data-driven amount Q so that the 80%
    # interval has *guaranteed* marginal coverage on unseen data.
    cal_lo, cal_hi = q10.predict(Xcal), q90.predict(Xcal)
    scores = np.maximum(cal_lo - ycal.values, ycal.values - cal_hi)  # conformity
    Q = float(np.quantile(scores, 0.90))                             # 1 - alpha = 0.8 -> 0.9
    lo, hi = q10.predict(Xte) - Q, q90.predict(Xte) + Q
    lo = np.minimum(lo, pred); hi = np.maximum(hi, pred)

    metrics = {
        "MAE":  round(mean_absolute_error(yte, pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(yte, pred))), 3),
        "MAPE_pct": round(mape(yte.values, pred), 3),
        "interval_coverage_80pct": round(float(np.mean((yte.values >= lo) & (yte.values <= hi)) * 100), 1),
        "n_test": int(len(yte)),
    }
    print("metrics:", metrics)

    joblib.dump(model, MODELS / "load_model.pkl")
    joblib.dump(q10, MODELS / "load_q10.pkl")
    joblib.dump(q90, MODELS / "load_q90.pkl")
    joblib.dump({"Q": Q}, MODELS / "load_conformal.pkl")
    joblib.dump(list(X.columns), MODELS / "load_features.pkl")
    (MODELS / "load_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- plot last 5 days of test ----
    show = slice(-120, None)
    t = yte.index[show]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(t, lo[show], hi[show], color="#93c5fd", alpha=0.5, label="80% interval")
    ax.plot(t, yte.values[show], color="#111827", lw=1.6, label="Actual")
    ax.plot(t, pred[show], color="#2563eb", lw=1.4, ls="--", label="Forecast")
    ax.set_title(f"Load forecast — MAE {metrics['MAE']} MW · MAPE {metrics['MAPE_pct']}%")
    ax.set_ylabel("Load (MW)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(REPORTS / "load_forecast.png", dpi=120)
    print("saved models + reports/load_forecast.png")


if __name__ == "__main__":
    main()
