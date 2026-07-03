# ⚡ Energy Intelligence Platform

![CI](https://github.com/YOUR-USERNAME/energy-intelligence-platform/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)

A **live**, end-to-end machine-learning platform for grid energy. It ingests
**real, current** electricity demand and weather, forecasts load & solar with
calibrated uncertainty, optimizes battery dispatch for **cost and carbon**,
detects anomalies, explains its predictions with SHAP, and answers natural-language
questions via an LLM copilot.

> **Live data, not a frozen dataset.** Load comes from the **EIA Open Data API**
> (real hourly US grid demand) and weather from **Open-Meteo** — the models are
> retrained on a rolling 1-year window, so every forecast is based on data through
> *yesterday*.

---

## 📊 Results (on live data, held-out test set)

| Model | Metric | Result |
|---|---|---|
| **Load forecast** (LightGBM, day-ahead) | MAPE | **5.95%** |
| **Forecast uncertainty** (Conformalized Quantile Regression) | 80% interval coverage | **79.1%** — calibrated (target 80%) |
| **Solar forecast** (LightGBM from weather) | R² | **0.82** |
| **Battery optimizer** (OR-Tools LP) | cost saving vs grid-only | **~6–7%** |
| **Test set size** | | 1,534 load / 1,572 solar hours |

*Evaluated on real PJM demand. A calibrated 80% band (~80% coverage) is the key
signal that the uncertainty estimate is trustworthy, not decorative.*

---

## 🧩 Features (6-tab dashboard + REST API)

1. **📈 Live Forecast** — day-ahead load & solar with an 80% confidence band, plus a backtest-vs-actual view
2. **🔋 Battery Optimizer** — OR-Tools linear program; **Cost / Balanced / Green** modes trading off $ vs CO₂
3. **🔬 What-if Simulator** — resize battery/solar and see savings change, with a full **ROI engine** (payback, NPV, IRR)
4. **🚨 Anomaly Detection** — Isolation Forest flags unusual load hours (winter cold-snap clusters)
5. **🧠 SHAP Explainability** — global + local explanations of what drives each forecast
6. **🤖 AI Copilot** — LLM (Groq / Llama-3.3-70B) grounded in live platform numbers

---

## 🚀 Quick start

```bash
pip install -r requirements.txt

# 1. Pull ~1 year of live data (free EIA key: https://www.eia.gov/opendata/)
python fetch_real_data.py --eia-key YOUR_EIA_KEY

# 2. Train models + optimizer + SHAP
python run_all.py
python src/anomaly/detect.py

# 3. Launch the dashboard
export GROQ_API_KEY=YOUR_GROQ_KEY      # for the AI Copilot tab
streamlit run dashboard.py             # http://localhost:8501

# (optional) REST API
uvicorn src.api.main:app --reload      # http://localhost:8000/docs
```

## 🧪 Tests

```bash
pytest tests/ -v      # 15 tests: optimizer constraints, ROI math, pricing/carbon
```
CI runs the suite on every push via GitHub Actions.

---

## 🗂️ Project layout

```
energy-copilot/
├── data/                          live dataset (EIA demand + Open-Meteo weather)
├── src/
│   ├── forecasting/
│   │   ├── features.py            leakage-safe feature engineering
│   │   ├── train_load.py          load model + CQR uncertainty
│   │   └── train_solar.py         solar model
│   ├── optimizer/
│   │   ├── battery.py             OR-Tools dispatch LP + TOU price + carbon
│   │   └── roi.py                 payback / NPV / IRR engine
│   ├── anomaly/detect.py          Isolation Forest anomaly detection
│   ├── explain/shap_explain.py    SHAP explainability
│   ├── copilot/chat.py            LLM copilot (Groq)
│   └── api/main.py                FastAPI backend
├── tests/                         pytest suite (15 tests)
├── models/  reports/              saved models, metrics, plots
├── dashboard.py                   6-tab Streamlit app
├── fetch_real_data.py             live EIA + Open-Meteo data pipeline
├── run_all.py                     one-command training pipeline
├── Dockerfile                     containerized (API + dashboard)
└── .github/workflows/ci.yml       CI
```

---

## 💬 Worth explaining in interviews

- **Live data pipeline:** the EIA API is paginated past its 5,000-row cap to pull a
  full year; weather merges Open-Meteo's *archive* API (older than 5 days) with the
  *forecast* API (recent), since neither alone covers the whole span.
- **No data leakage:** every lag/rolling feature uses a horizon ≥ 24h, so the model
  is an honest *day-ahead* forecaster (`features.py`).
- **Calibrated uncertainty:** raw quantile bands were overconfident, so I added
  **Conformalized Quantile Regression** — a held-out calibration split widens the
  band to hit ~80% coverage (measured: 79.1%).
- **The optimizer is a real LP:** decision variables (grid/charge/discharge/SOC),
  power-balance + SOC-dynamics + rate constraints, objective = minimize cost (+carbon).
  Solved exactly with OR-Tools GLOP — arbitrage is *emergent*, not hard-coded.
- **Cost vs carbon:** grid carbon follows a duck curve (cleanest midday, dirtiest in
  the evening peak) *decorrelated* from price, so Green Mode produces a genuinely
  different dispatch than Cost Mode.
- **Honest ROI:** the ROI engine openly shows that energy arbitrage alone rarely pays
  back grid-scale storage — real projects stack demand-charge, capacity, and resilience
  value that aren't modelled here.

## Modeling assumptions (be ready to defend)
- Load is scaled to a **campus-microgrid magnitude** (~16 MW mean) from the real
  regional PJM signal, so the 20 MW solar plant and 40 MWh battery are meaningfully sized.
- **Solar** is derived from irradiance × a hypothetical 20 MW plant (a proxy, not metered generation).
- **TOU price** ($80 peak / $40 off-peak) and **carbon intensity** are realistic proxies,
  not live market feeds.
