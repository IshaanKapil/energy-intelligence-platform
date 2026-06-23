"""
dashboard.py — Streamlit dashboard for the Energy Intelligence Platform.

Run:
    streamlit run dashboard.py

Tabs:
  1. Live Forecast   — load + solar forecasts with 80% uncertainty band
  2. Battery Optimizer — dispatch schedule + cost savings
  3. What-if Simulator — slider to change battery/solar params and re-run optimizer
  4. Anomaly Detection — flag unusual load hours
  5. SHAP Explainability — top drivers of load
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src" / "forecasting"))
sys.path.append(str(ROOT / "src" / "optimizer"))
sys.path.append(str(ROOT / "src" / "anomaly"))

from features import build_load_features, build_solar_features  # noqa: E402
from battery import optimize_battery, BatteryConfig, tou_price  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Energy Intelligence Platform",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Intelligence Platform")
st.caption("Real-time load & solar forecasting · Battery optimization · Anomaly detection")

# ---------------------------------------------------------------------------
# Load artifacts (cached)
# ---------------------------------------------------------------------------
MODELS = ROOT / "models"
DATA   = ROOT / "data" / "energy_dataset_real.csv"

@st.cache_resource
def load_models():
    return {
        "load":       joblib.load(MODELS / "load_model.pkl"),
        "q10":        joblib.load(MODELS / "load_q10.pkl"),
        "q90":        joblib.load(MODELS / "load_q90.pkl"),
        "Q":          joblib.load(MODELS / "load_conformal.pkl")["Q"],
        "load_feats": joblib.load(MODELS / "load_features.pkl"),
        "solar":      joblib.load(MODELS / "solar_model.pkl"),
        "solar_feats":joblib.load(MODELS / "solar_features.pkl"),
    }

@st.cache_data(ttl=300)
def load_data(n=800):
    return pd.read_csv(DATA, parse_dates=["datetime"]).set_index("datetime").iloc[-n:]

def check_ready():
    if not DATA.exists():
        st.error("No real data found. Run: `python fetch_real_data.py`")
        st.stop()
    if not (MODELS / "load_model.pkl").exists():
        st.error("Models not trained. Run: `python run_all.py`")
        st.stop()

check_ready()
m   = load_models()
df  = load_data()

# ---------------------------------------------------------------------------
# Shared forecast helper
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_forecasts():
    Xl, yl, idxl = build_load_features(df)
    Xh = Xl.iloc[-24:][m["load_feats"]]
    median = m["load"].predict(Xh)
    lo = m["q10"].predict(Xh) - m["Q"]
    hi = m["q90"].predict(Xh) + m["Q"]
    lo = np.minimum(lo, median); hi = np.maximum(hi, median)
    times = idxl[-24:]

    Xs, ys, idxs = build_solar_features(df)
    Xsh = Xs.iloc[-24:][m["solar_feats"]]
    solar = np.clip(m["solar"].predict(Xsh), 0, None)
    return times, median, lo, hi, solar

# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Live Forecast",
    "🔋 Battery Optimizer",
    "🔬 What-if Simulator",
    "🚨 Anomaly Detection",
    "🧠 SHAP Explainability",
    "🤖 AI Copilot",
])

# ---------------------------------------------------------------------------
# TAB 1 — Live Forecast
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Day-Ahead Forecast (next 24 h)")
    times, median, lo, hi, solar = get_forecasts()

    col1, col2, col3 = st.columns(3)
    col1.metric("Peak Load Forecast", f"{median.max():.0f} MW")
    col2.metric("Min Load Forecast",  f"{median.min():.0f} MW")
    col3.metric("Peak Solar Forecast",f"{solar.max():.1f} MW")

    # Load chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(times) + list(times[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill="toself", fillcolor="rgba(37,99,235,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="80% interval"
    ))
    fig.add_trace(go.Scatter(x=times, y=median, mode="lines",
                             line=dict(color="#2563eb", width=2), name="Load forecast"))
    fig.add_trace(go.Scatter(x=times, y=df["load_mw"].iloc[-24:], mode="lines",
                             line=dict(color="#111827", width=1.5, dash="dot"), name="Actual (last 24h)"))
    fig.update_layout(title="Load Forecast with 80% Confidence Interval",
                      yaxis_title="Load (MW)", height=360, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Solar chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=solar, mode="lines",
                              line=dict(color="#f59e0b", width=2), name="Solar forecast"))
    fig2.update_layout(title="Solar Generation Forecast",
                       yaxis_title="Solar (MW)", height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Battery Optimizer
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Battery Dispatch Optimizer")
    st.caption("Charges when power is cheap (off-peak), discharges at peak hours — minimises cost via Linear Programming.")

    times, median, lo, hi, solar = get_forecasts()
    price = tou_price(times)
    cfg   = BatteryConfig()
    res   = optimize_battery(median.tolist(), solar.tolist(), price, cfg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Cost",    f"${res.baseline_cost:,.0f}")
    c2.metric("Optimized Cost",   f"${res.optimized_cost:,.0f}")
    c3.metric("Savings",          f"${res.savings:,.0f}")
    c4.metric("Savings %",        f"{res.savings_pct:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=median, name="Load",
                             line=dict(color="#111827", width=1.5)))
    fig.add_trace(go.Scatter(x=times, y=solar, name="Solar",
                             line=dict(color="#f59e0b", width=1.5)))
    fig.add_trace(go.Bar(x=times, y=res.charge, name="Charge",
                         marker_color="#16a34a", opacity=0.7))
    fig.add_trace(go.Bar(x=times, y=[-d for d in res.discharge], name="Discharge",
                         marker_color="#dc2626", opacity=0.7))
    fig.update_layout(title="Battery Dispatch Schedule",
                      yaxis_title="MW", barmode="overlay", height=360,
                      margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=res.soc, name="State of Charge",
                              line=dict(color="#7c3aed", width=2), fill="tozeroy",
                              fillcolor="rgba(124,58,237,0.1)"))
    fig2.update_layout(title="Battery State of Charge (MWh)",
                       yaxis_title="MWh", height=250, margin=dict(t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — What-if Simulator
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("What-if Simulator")
    st.caption("Adjust battery or solar parameters and see how savings change.")

    col_s, col_r = st.columns([1, 2])
    with col_s:
        cap   = st.slider("Battery Capacity (MWh)", 10, 200, 40, step=5)
        power = st.slider("Max Charge/Discharge (MW)", 2, 50, 10, step=1)
        solar_scale = st.slider("Solar Plant Scale (×)", 0.5, 4.0, 1.0, step=0.25)
        run_btn = st.button("Run Optimizer", type="primary")

    if run_btn:
        times, median, lo, hi, solar_base = get_forecasts()
        solar_adj = solar_base * solar_scale
        price     = tou_price(times)
        cfg_wi    = BatteryConfig(capacity_mwh=cap, max_power_mw=power)
        res_wi    = optimize_battery(median.tolist(), solar_adj.tolist(), price, cfg_wi)

        with col_r:
            w1, w2, w3 = st.columns(3)
            w1.metric("Savings",     f"${res_wi.savings:,.0f}")
            w2.metric("Savings %",   f"{res_wi.savings_pct:.1f}%")
            w3.metric("Status",      res_wi.status)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=median, name="Load",
                                     line=dict(color="#111827")))
            fig.add_trace(go.Scatter(x=times, y=solar_adj, name=f"Solar ×{solar_scale}",
                                     line=dict(color="#f59e0b")))
            fig.add_trace(go.Bar(x=times, y=res_wi.charge,
                                 name="Charge", marker_color="#16a34a", opacity=0.7))
            fig.add_trace(go.Bar(x=times, y=[-d for d in res_wi.discharge],
                                 name="Discharge", marker_color="#dc2626", opacity=0.7))
            fig.update_layout(title=f"Dispatch — {cap} MWh battery, solar ×{solar_scale}",
                              yaxis_title="MW", barmode="overlay", height=360,
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
    else:
        with col_r:
            st.info("Adjust sliders and click **Run Optimizer** to see results.")

# ---------------------------------------------------------------------------
# TAB 4 — Anomaly Detection
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Load Anomaly Detection")
    st.caption("Isolation Forest flags hours where load is unusually high or low given the context.")

    anomaly_model_path = MODELS / "anomaly_model.pkl"
    if not anomaly_model_path.exists():
        st.warning("Anomaly model not trained yet. Run: `python src/anomaly/detect.py`")
    else:
        from detect import load_anomaly_model, predict_anomalies  # noqa: E402

        anom_model, anom_feats = load_anomaly_model()
        recent = df.iloc[-168:]  # last 7 days
        scores, flags = predict_anomalies(recent, anom_model, anom_feats)

        n_flagged = int(flags.sum())
        st.metric("Anomalies in last 7 days", n_flagged)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent["load_mw"],
                                 mode="lines", name="Load",
                                 line=dict(color="#6b7280", width=1.5)))
        anom_idx = recent.index[flags]
        anom_val = recent["load_mw"][flags]
        fig.add_trace(go.Scatter(x=anom_idx, y=anom_val, mode="markers",
                                 name="Anomaly", marker=dict(color="#ef4444", size=10, symbol="x")))
        fig.update_layout(title="Load — last 7 days (anomalies marked ✕)",
                          yaxis_title="Load (MW)", height=360, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Anomaly scores (lower = more anomalous)"):
            score_df = pd.DataFrame({"datetime": recent.index,
                                     "load_mw": recent["load_mw"].values,
                                     "anomaly_score": scores,
                                     "is_anomaly": flags})
            st.dataframe(score_df[score_df["is_anomaly"]].sort_values("anomaly_score"))

# ---------------------------------------------------------------------------
# TAB 5 — SHAP Explainability
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("SHAP Feature Importance")
    st.caption("Why is the load forecast high or low? SHAP shows which features drive the prediction.")

    shap_json = ROOT / "reports" / "shap_top_drivers.json"
    shap_summary = ROOT / "reports" / "shap_summary.png"
    shap_waterfall = ROOT / "reports" / "shap_waterfall.png"

    if shap_json.exists():
        drivers = json.loads(shap_json.read_text())
        names  = [d["feature"] for d in drivers]
        values = [d["mean_abs_shap"] for d in drivers]

        fig = go.Figure(go.Bar(x=values[::-1], y=names[::-1], orientation="h",
                               marker_color="#2563eb"))
        fig.update_layout(title="Top Load Drivers (mean |SHAP|)",
                          xaxis_title="Mean |SHAP value| (MW)",
                          height=350, margin=dict(t=40, b=20, l=160))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    if shap_summary.exists():
        col1.image(str(shap_summary), caption="Global SHAP Summary", use_container_width=True)
    if shap_waterfall.exists():
        col2.image(str(shap_waterfall), caption="Local Waterfall (peak hour)", use_container_width=True)

    if not shap_json.exists():
        st.warning("SHAP reports missing. Run: `python run_all.py`")

# ---------------------------------------------------------------------------
# TAB 6 — AI Copilot
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("🤖 AI Energy Copilot")
    st.caption("Ask anything about load forecasts, battery savings, anomalies, or SHAP drivers.")

    # Resolve the Groq key from Streamlit secrets (cloud) or env var (local).
    import os
    if not os.environ.get("GROQ_API_KEY"):
        try:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set. Locally: `$env:GROQ_API_KEY=\"gsk_...\"`. "
                 "On Streamlit Cloud: add it under App → Settings → Secrets.")
        st.stop()

    sys.path.append(str(ROOT / "src" / "copilot"))
    from chat import ask_copilot  # noqa: E402

    # session state for chat history
    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = []

    # render existing messages
    for msg in st.session_state.copilot_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # suggested questions
    if not st.session_state.copilot_messages:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        suggestions = [
            "Why is load high today?",
            "How are battery savings calculated?",
            "What are the top drivers of electricity demand?",
        ]
        for col, q in zip(cols, suggestions):
            if col.button(q, use_container_width=True):
                st.session_state._copilot_prefill = q
                st.rerun()

    # handle prefilled suggestion clicks
    prefill = st.session_state.pop("_copilot_prefill", None)

    user_input = st.chat_input("Ask the energy copilot…") or prefill
    if user_input:
        st.session_state.copilot_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # build history for multi-turn (exclude last user message, already appended)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.copilot_messages[:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = ask_copilot(user_input, conversation_history=history or None)
                except Exception as e:
                    reply = f"Error calling AI copilot: {e}"
            st.markdown(reply)

        st.session_state.copilot_messages.append({"role": "assistant", "content": reply})

    if st.session_state.copilot_messages:
        if st.button("Clear chat", use_container_width=False):
            st.session_state.copilot_messages = []
            st.rerun()
