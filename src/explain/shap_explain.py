"""
shap_explain.py — explain WHY the load forecast is high or low.

Produces:
  reports/shap_summary.png   global feature importance (which drivers matter)
  reports/shap_waterfall.png local explanation for one specific hour
  reports/shap_top_drivers.json  machine-readable drivers (for the AI copilot)
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "forecasting"))
from features import build_load_features  # noqa: E402

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)


def main():
    model = joblib.load(MODELS / "load_model.pkl")
    df = pd.read_csv(ROOT / "data" / "energy_dataset_real.csv",
                     parse_dates=["datetime"]).set_index("datetime")
    X, y, idx = build_load_features(df)
    Xs = X.iloc[-500:]  # explain recent window

    explainer = shap.TreeExplainer(model)
    sv = explainer(Xs)

    # global importance
    plt.figure()
    shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=12)
    plt.tight_layout(); plt.savefig(REPORTS / "shap_summary.png", dpi=120); plt.close()

    # local explanation for the single highest-load hour in the window
    peak_i = int(np.argmax(y.iloc[-500:].values)) if False else int(np.argmax(Xs["load_mw_lag24"].values))
    plt.figure()
    shap.plots.waterfall(sv[peak_i], max_display=10, show=False)
    plt.tight_layout(); plt.savefig(REPORTS / "shap_waterfall.png", dpi=120, bbox_inches="tight"); plt.close()

    # top drivers as JSON (feed this to the AI copilot for natural-language reasons)
    mean_abs = np.abs(sv.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:6]
    drivers = [{"feature": Xs.columns[i], "mean_abs_shap": round(float(mean_abs[i]), 3)}
               for i in order]
    (REPORTS / "shap_top_drivers.json").write_text(json.dumps(drivers, indent=2))

    print("Top load drivers:")
    for d in drivers:
        print(f"  {d['feature']:<22} {d['mean_abs_shap']}")
    print("saved shap_summary.png, shap_waterfall.png, shap_top_drivers.json")


if __name__ == "__main__":
    main()
