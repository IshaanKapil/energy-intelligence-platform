# ⚡ Energy Intelligence Platform

![CI](https://github.com/YOUR-USERNAME/energy-intelligence-platform/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)

Hi! This is my 3rd-year CS project — a machine-learning platform for grid energy.
It pulls **real, current** electricity demand and weather, forecasts load & solar,
optimizes when a battery should charge/discharge, flags anomalies, explains its
predictions, and lets you ask questions in plain English via an LLM.

I started it with a synthetic dataset, then swapped in real live data because I
wanted the forecasts to actually mean something. Along the way I learned a lot
about calibrated uncertainty, linear programming, and gluing real APIs together.

> **It uses live data, not a frozen CSV.** Load comes from the **EIA Open Data API**
> (real hourly US grid demand) and weather from **Open-Meteo**. I retrain on a
> rolling 1-year window, so the forecasts are based on data through *yesterday*.

---

## 📊 Results (on live data, held-out test set)

| Model | Metric | Result |
|---|---|---|
| Load forecast (LightGBM, day-ahead) | MAPE | **5.95%** |
| Forecast uncertainty (Conformalized Quantile Regression) | 80% interval coverage | **79.1%** (target 80%) |
| Solar forecast (LightGBM from weather) | R² | **0.82** |
| Battery optimizer (OR-Tools LP) | cost saving vs grid-only | **~6–7%** |
| Test set size | | 1,534 load / 1,572 solar hours |

The thing I'm most happy about is the **79.1% coverage** — it means when the model
says "80% confident," it's actually right ~80% of the time. Getting the uncertainty
*calibrated* was harder than getting the point forecast accurate.

---

## 🧩 What it does (6-tab dashboard + a REST API)

1. **📈 Live Forecast** — day-ahead load & solar with an 80% confidence band + a backtest-vs-actual view
2. **🔋 Battery Optimizer** — an OR-Tools linear program; **Cost / Balanced / Green** modes trade off money vs CO₂
3. **🔬 What-if Simulator** — resize the battery/solar and watch savings change, plus an **ROI calculator** (payback, NPV, IRR)
4. **🚨 Anomaly Detection** — Isolation Forest flags unusual load hours
5. **🧠 SHAP Explainability** — shows what drives each forecast
6. **🤖 AI Copilot** — an LLM (Groq / Llama-3.3-70B) that answers questions using the platform's live numbers

---

## 🚀 Quick start

```bash
pip install -r requirements.txt

# 1. Pull ~1 year of live data (free EIA key: https://www.eia.gov/opendata/)
python fetch_real_data.py --eia-key YOUR_EIA_KEY

# 2. Train the models + optimizer + SHAP
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
GitHub Actions runs these on every push.

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
│   │   └── roi.py                 payback / NPV / IRR
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

## 🛠️ Things I learned / would explain

- **Live data was fiddly.** The EIA API caps responses at 5,000 rows, so I had to
  paginate to get a full year. And Open-Meteo's forecast API only goes back ~92 days,
  so for older weather I had to fall back to their archive API and stitch the two together.
- **Avoiding data leakage.** Every lag/rolling feature uses a horizon ≥ 24h, so the
  model can't "cheat" by seeing data it wouldn't have at prediction time.
- **Calibrated uncertainty (the part I'm proud of).** My first quantile bands were
  overconfident, so I added **Conformalized Quantile Regression** — a held-out
  calibration split that widens the band until it actually hits ~80% coverage.
- **The optimizer is a real linear program**, not if-else rules. I set up the decision
  variables and constraints and let OR-Tools solve it — the "charge cheap, discharge
  at peak" behaviour *emerges* from the math.
- **Cost vs carbon.** Grid carbon is dirtiest in the evening peak and cleanest midday
  (solar), which is different from when it's most expensive — so "Green Mode" actually
  gives a different schedule than "Cost Mode".

## ⚠️ Honest limitations (I know these aren't perfect)

- **Load is scaled** from the real regional PJM signal down to a ~16 MW campus-microgrid
  size, so the battery/solar are meaningfully sized against it.
- **Solar is a proxy** — derived from irradiance × a hypothetical 20 MW plant, not real
  metered generation.
- **Prices and carbon intensity are realistic proxies**, not live market data.
- The ROI tool honestly shows that battery arbitrage *alone* usually doesn't pay back —
  real projects need extra value streams (demand charges, capacity payments) I didn't model.
