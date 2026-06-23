"""
train_solar.py — forecast solar generation from weather inputs.

Predicts solar_mw from cloud cover, temperature, time-of-day and season
(NOT from irradiance directly) — mirroring the real task of forecasting
generation from a weather forecast.

Produces: models/solar_model.pkl, models/solar_metrics.json,
          reports/solar_forecast.png
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_solar_features

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "energy_dataset_real.csv"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA, parse_dates=["datetime"]).set_index("datetime")
    X, y, idx = build_solar_features(df)
    cut = int(len(idx) * 0.8)
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=48,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
    model.fit(Xtr, ytr)
    pred = np.clip(model.predict(Xte), 0, None)   # generation can't be negative

    metrics = {
        "MAE": round(mean_absolute_error(yte, pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(yte, pred))), 3),
        "R2": round(r2_score(yte, pred), 4),
        "n_test": int(len(yte)),
    }
    print("solar metrics:", metrics)

    joblib.dump(model, MODELS / "solar_model.pkl")
    joblib.dump(list(X.columns), MODELS / "solar_features.pkl")
    (MODELS / "solar_metrics.json").write_text(json.dumps(metrics, indent=2))

    show = slice(-120, None)
    t = yte.index[show]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, yte.values[show], color="#92400e", lw=1.6, label="Actual")
    ax.plot(t, pred[show], color="#f59e0b", lw=1.4, ls="--", label="Forecast")
    ax.set_title(f"Solar forecast — MAE {metrics['MAE']} MW · R² {metrics['R2']}")
    ax.set_ylabel("Solar (MW)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(REPORTS / "solar_forecast.png", dpi=120)
    print("saved models + reports/solar_forecast.png")


if __name__ == "__main__":
    main()
