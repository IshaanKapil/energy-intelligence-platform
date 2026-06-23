# Energy Intelligence Platform

AI-powered energy platform: forecasts industrial load & solar generation,
optimizes battery dispatch to cut cost, quantifies forecast uncertainty,
and explains its predictions.

> **Status: ~50% built (the core "spine").** Forecasting, uncertainty,
> optimization, explainability, and the API are working with trained models
> and real metrics. Dashboard, anomaly detection, and AI copilot are next.

---

## ✅ What works right now

| Component | File | Result |
|---|---|---|
| **Load forecasting** (LightGBM, day-ahead) | `src/forecasting/train_load.py` | **MAE 3.02 MW · MAPE 9.1%** |
| **Forecast uncertainty** (Conformalized Quantile Regression) | same | **80% interval → 84% coverage** (calibrated) |
| **Solar forecasting** (LightGBM from weather) | `src/forecasting/train_solar.py` | **R² 0.99** (synthetic; expect lower on real data) |
| **Battery optimizer** (OR-Tools linear program) | `src/optimizer/battery.py` | **~6–7% cost saving** vs grid-only |
| **Explainability** (SHAP) | `src/explain/shap_explain.py` | top drivers: last-week load, day-of-week, temperature |
| **REST API** (FastAPI) | `src/api/main.py` | `/forecast/load`, `/forecast/solar`, `/optimize/battery`, `/explain/drivers` |

See `reports/` for the generated plots.

## 🔜 Remaining ~50%
Dashboard (Next.js/Streamlit) · anomaly detection (Isolation Forest) ·
what-if simulator · carbon optimizer · AI copilot (LangChain) · Docker + deploy.

---

## Quick start

```bash
pip install -r requirements.txt

# run the whole pipeline (data -> models -> shap -> optimizer)
python run_all.py

# start the API
uvicorn src.api.main:app --reload
# open http://127.0.0.1:8000/docs  (interactive Swagger UI)
```

## Project layout

```
energy-copilot/
├── data/                       synthetic dataset + README (real-data swap)
├── src/
│   ├── forecasting/
│   │   ├── features.py         leakage-safe feature engineering
│   │   ├── train_load.py       load model + CQR uncertainty
│   │   └── train_solar.py      solar model
│   ├── optimizer/battery.py    OR-Tools battery dispatch LP
│   ├── explain/shap_explain.py SHAP explainability
│   └── api/main.py             FastAPI backend
├── models/                     saved models + metrics.json
├── reports/                    forecast / dispatch / SHAP plots
├── generate_synthetic_data.py  realistic data generator
├── fetch_real_data.py          pull REAL PJM + Open-Meteo data
├── run_all.py                  one-command pipeline
└── requirements.txt
```

---

## Things worth explaining in interviews

- **No data leakage:** every lag/rolling feature uses a horizon ≥ 24h, so the
  model is an honest *day-ahead* forecaster (`features.py`).
- **Calibrated uncertainty:** raw quantile bands were overconfident (48–62%
  coverage), so I added **Conformalized Quantile Regression** — a held-out
  calibration split widens the band to hit the target 80% coverage.
- **The optimizer is a real LP:** decision variables (grid/charge/discharge/SOC),
  power-balance + SOC-dynamics + rate constraints, objective = minimize cost.
  Solved exactly with OR-Tools GLOP. The dispatch plot shows it charging when
  cheap and discharging at peaks — emergent arbitrage, not hard-coded rules.
- **Honest about synthetic data:** solar R² 0.99 is inflated because synthetic
  solar is near-deterministic from weather; on real Open-Meteo data expect a
  lower R². Swap in real data with `fetch_real_data.py`.
```
```
