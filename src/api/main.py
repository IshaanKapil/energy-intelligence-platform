"""
main.py — FastAPI backend for the Energy Intelligence Platform.

Run from the project root:
    uvicorn src.api.main:app --reload

Endpoints:
    GET  /health
    GET  /forecast/load     -> next-24h load forecast + 80% interval
    GET  /forecast/solar    -> next-24h solar forecast
    POST /optimize/battery  -> battery dispatch schedule + savings
    GET  /explain/drivers   -> top SHAP drivers of load
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "forecasting"))
sys.path.append(str(ROOT / "src" / "optimizer"))
sys.path.append(str(ROOT / "src" / "anomaly"))
sys.path.append(str(ROOT / "src" / "copilot"))
from features import build_load_features, build_solar_features  # noqa: E402
from battery import optimize_battery, BatteryConfig, tou_price  # noqa: E402
from detect import load_anomaly_model, predict_anomalies         # noqa: E402
from chat import ask_copilot                                     # noqa: E402

DATA = ROOT / "data" / "energy_dataset_real.csv"
MODELS = ROOT / "models"

app = FastAPI(title="Energy Intelligence Platform API", version="0.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# load artifacts once at startup
_load_model = joblib.load(MODELS / "load_model.pkl")
_load_q10 = joblib.load(MODELS / "load_q10.pkl")
_load_q90 = joblib.load(MODELS / "load_q90.pkl")
_load_Q = joblib.load(MODELS / "load_conformal.pkl")["Q"]
_load_feats = joblib.load(MODELS / "load_features.pkl")
_solar_model = joblib.load(MODELS / "solar_model.pkl")
_solar_feats = joblib.load(MODELS / "solar_features.pkl")
_anom_model, _anom_feats = (None, None)
if (MODELS / "anomaly_model.pkl").exists():
    _anom_model, _anom_feats = load_anomaly_model()


def _recent(n_hist=400):
    return pd.read_csv(DATA, parse_dates=["datetime"]).set_index("datetime").iloc[-n_hist:]


@app.get("/health")
def health():
    return {"status": "ok", "models": ["load", "load_quantiles", "solar"]}


@app.get("/forecast/load")
def forecast_load():
    df = _recent()
    X, y, idx = build_load_features(df)
    Xh = X.iloc[-24:][_load_feats]
    median = _load_model.predict(Xh)
    lo = _load_q10.predict(Xh) - _load_Q
    hi = _load_q90.predict(Xh) + _load_Q
    return {"horizon_h": 24,
            "datetime": [t.isoformat() for t in idx[-24:]],
            "forecast_mw": [round(float(v), 2) for v in median],
            "lower_mw": [round(float(min(l, m)), 2) for l, m in zip(lo, median)],
            "upper_mw": [round(float(max(h, m)), 2) for h, m in zip(hi, median)]}


@app.get("/forecast/solar")
def forecast_solar():
    df = _recent()
    X, y, idx = build_solar_features(df)
    Xh = X.iloc[-24:][_solar_feats]
    pred = _solar_model.predict(Xh).clip(min=0)
    return {"horizon_h": 24,
            "datetime": [t.isoformat() for t in idx[-24:]],
            "forecast_mw": [round(float(v), 3) for v in pred]}


class OptimizeRequest(BaseModel):
    capacity_mwh: float = 40.0
    max_power_mw: float = 10.0
    use_forecast: bool = True   # if False, use actuals from the dataset


@app.post("/optimize/battery")
def optimize(req: OptimizeRequest):
    df = _recent()
    last24 = df.iloc[-24:]
    if req.use_forecast:
        lf = forecast_load()["forecast_mw"]
        sf = forecast_solar()["forecast_mw"]
        load, solar = lf, sf
    else:
        load, solar = last24.load_mw.tolist(), last24.solar_mw.tolist()
    price = tou_price(last24.index)
    cfg = BatteryConfig(capacity_mwh=req.capacity_mwh, max_power_mw=req.max_power_mw)
    r = optimize_battery(load, solar, price, cfg)
    return {"status": r.status,
            "datetime": [t.isoformat() for t in last24.index],
            "grid_mw": [round(v, 2) for v in r.grid],
            "charge_mw": [round(v, 2) for v in r.charge],
            "discharge_mw": [round(v, 2) for v in r.discharge],
            "soc_mwh": [round(v, 2) for v in r.soc],
            "baseline_cost": round(r.baseline_cost, 1),
            "optimized_cost": round(r.optimized_cost, 1),
            "savings": round(r.savings, 1),
            "savings_pct": round(r.savings_pct, 2)}


@app.get("/explain/drivers")
def drivers():
    p = ROOT / "reports" / "shap_top_drivers.json"
    return json.loads(p.read_text()) if p.exists() else {"error": "run shap_explain.py first"}


class CopilotRequest(BaseModel):
    question: str
    history: list = []


@app.post("/copilot/chat")
def copilot_chat(req: CopilotRequest):
    answer = ask_copilot(req.question, conversation_history=req.history or None)
    return {"answer": answer}


@app.get("/anomaly/detect")
def anomaly_detect(hours: int = 168):
    if _anom_model is None:
        return {"error": "Anomaly model not trained. Run: python src/anomaly/detect.py"}
    df = _recent(max(hours + 200, 400))
    recent = df.iloc[-hours:]
    scores, flags = predict_anomalies(recent, _anom_model, _anom_feats)
    anomalies = [
        {"datetime": t.isoformat(), "load_mw": round(float(v), 2), "score": round(float(s), 4)}
        for t, v, s, f in zip(recent.index, recent["load_mw"], scores, flags) if f
    ]
    return {"hours_checked": hours, "n_anomalies": len(anomalies), "anomalies": anomalies}
