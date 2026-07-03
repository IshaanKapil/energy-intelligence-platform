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
from battery import (optimize_battery, BatteryConfig, tou_price,  # noqa: E402
                     carbon_intensity, CARBON_PRICE_MODES)
from roi import compute_roi  # noqa: E402

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


@st.cache_data(ttl=300)
def backtest_load(hours=168):
    """Predict load + 80% band over the last `hours` (where we have actuals)
    and return arrays for a forecast-vs-actual comparison."""
    Xl, yl, idxl = build_load_features(df)
    Xh = Xl.iloc[-hours:][m["load_feats"]]
    pred = m["load"].predict(Xh)
    lo = np.minimum(m["q10"].predict(Xh) - m["Q"], pred)
    hi = np.maximum(m["q90"].predict(Xh) + m["Q"], pred)
    actual = yl.iloc[-hours:].values
    times = idxl[-hours:]
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    coverage = float(np.mean((actual >= lo) & (actual <= hi)) * 100)
    return times, actual, pred, lo, hi, mape, coverage

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
    # ---- Model Accuracy (proves the forecast actually works) ----
    st.subheader("Model Accuracy")
    lm_path = MODELS / "load_metrics.json"
    sm_path = MODELS / "solar_metrics.json"
    lm = json.loads(lm_path.read_text()) if lm_path.exists() else {}
    sm = json.loads(sm_path.read_text()) if sm_path.exists() else {}

    a1, a2, a3, a4 = st.columns(4)
    if lm:
        a1.metric("Load Forecast Error (MAPE)", f"{lm.get('MAPE_pct', 0):.2f}%",
                  help="Mean absolute % error on held-out test data. Lower is better.")
        cov = lm.get("interval_coverage_80pct", 0)
        a2.metric("80% Interval Coverage", f"{cov:.1f}%",
                  delta=f"{cov - 80:+.1f} vs target",
                  help="Share of actual values that fell inside the 80% band. "
                       "Close to 80% means the uncertainty estimate is calibrated.")
    if sm:
        a3.metric("Solar Forecast R²", f"{sm.get('R2', 0):.2f}",
                  help="Variance in solar output explained by the model (1.0 = perfect).")
        a4.metric("Solar MAE", f"{sm.get('MAE', 0):.2f} MW",
                  help="Mean absolute error on held-out test data.")
    st.caption(f"Evaluated on {lm.get('n_test', '—')} held-out load hours · "
               f"{sm.get('n_test', '—')} solar hours. A calibrated 80% band (≈80% coverage) "
               "is the key signal that the uncertainty estimate is trustworthy.")

    with st.expander("📊 Forecast vs. Actual — backtest over the last 7 days"):
        bt, act, pr, blo, bhi, bmape, bcov = backtest_load(168)
        figbt = go.Figure()
        figbt.add_trace(go.Scatter(
            x=list(bt) + list(bt[::-1]), y=list(bhi) + list(blo[::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="80% interval"))
        figbt.add_trace(go.Scatter(x=bt, y=act, mode="lines", name="Actual",
                                   line=dict(color="#111827", width=1.8)))
        figbt.add_trace(go.Scatter(x=bt, y=pr, mode="lines", name="Forecast",
                                   line=dict(color="#2563eb", width=1.6, dash="dash")))
        figbt.update_layout(title=f"Backtest — MAPE {bmape:.2f}% · band caught {bcov:.0f}% of actuals",
                            yaxis_title="Load (MW)", height=340, margin=dict(t=40, b=20))
        st.plotly_chart(figbt, width='stretch')

    st.divider()
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
    st.plotly_chart(fig, width='stretch')

    # Solar chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=solar, mode="lines",
                              line=dict(color="#f59e0b", width=2), name="Solar forecast"))
    fig2.update_layout(title="Solar Generation Forecast",
                       yaxis_title="Solar (MW)", height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig2, width='stretch')

# ---------------------------------------------------------------------------
# TAB 2 — Battery Optimizer
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Battery Dispatch Optimizer")
    st.caption("Charges when power is cheap (off-peak), discharges at peak hours — minimises cost via Linear Programming.")

    mode = st.radio(
        "Optimisation objective",
        ["cost", "balanced", "green"],
        format_func={"cost": "💵 Cost Mode (minimise $)",
                     "balanced": "⚖️ Balanced (price in carbon)",
                     "green": "🌱 Green Mode (minimise CO₂)"}.get,
        horizontal=True,
        help="The grid is dirtier at peak (gas/coal peakers) and cleaner off-peak "
             "(nuclear/wind). Green Mode weights carbon heavily; Cost Mode ignores it.",
    )

    times, median, lo, hi, solar = get_forecasts()
    price  = tou_price(times)
    carbon = carbon_intensity(times)
    cfg    = BatteryConfig()
    res    = optimize_battery(median.tolist(), solar.tolist(), price, cfg,
                              carbon=carbon, carbon_price=CARBON_PRICE_MODES[mode])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Cost",    f"${res.baseline_cost:,.0f}")
    c2.metric("Optimized Cost",   f"${res.optimized_cost:,.0f}")
    c3.metric("Cost Savings",     f"${res.savings:,.0f}", f"{res.savings_pct:.1f}%")
    c4.metric("CO₂ Avoided",      f"{res.emissions_saved/1000:,.1f} t",
              f"{res.emissions_saved_pct:.1f}%",
              help="Tonnes of CO₂ avoided vs running with no battery, over this 24h window.")

    fig = go.Figure()

    # Shade peak-price windows ($80/MWh) so the charge/discharge logic is obvious.
    peak = [p >= 80 for p in price]
    start = None
    for i, is_pk in enumerate(peak):
        if is_pk and start is None:
            start = times[i]
        if (not is_pk or i == len(peak) - 1) and start is not None:
            end = times[i] if not is_pk else times[i]
            fig.add_vrect(x0=start, x1=end, fillcolor="#fca5a5", opacity=0.18,
                          layer="below", line_width=0)
            start = None

    fig.add_trace(go.Scatter(x=times, y=median, name="Load",
                             line=dict(color="#111827", width=1.5)))
    fig.add_trace(go.Scatter(x=times, y=solar, name="Solar",
                             line=dict(color="#f59e0b", width=1.5)))
    fig.add_trace(go.Bar(x=times, y=res.charge, name="Charge",
                         marker_color="#16a34a", opacity=0.7))
    fig.add_trace(go.Bar(x=times, y=[-d for d in res.discharge], name="Discharge",
                         marker_color="#dc2626", opacity=0.7))
    fig.update_layout(title="Battery Dispatch Schedule  (red bands = peak price $80/MWh)",
                      yaxis_title="MW", barmode="overlay", height=360,
                      margin=dict(t=40, b=20))
    st.plotly_chart(fig, width='stretch')
    st.caption("🟢 Charge during cheap off-peak hours · 🔴 Discharge during the shaded "
               "$80/MWh peak windows — that price gap is exactly what the optimizer monetises.")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=res.soc, name="State of Charge",
                              line=dict(color="#7c3aed", width=2), fill="tozeroy",
                              fillcolor="rgba(124,58,237,0.1)"))
    fig2.update_layout(title="Battery State of Charge (MWh)",
                       yaxis_title="MWh", height=250, margin=dict(t=40, b=20))
    st.plotly_chart(fig2, width='stretch')

# ---------------------------------------------------------------------------
# TAB 3 — What-if Simulator
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("What-if Simulator")
    st.caption("Adjust battery or solar parameters and see how savings change.")

    SOLAR_PLANT_MW = 20.0   # base solar plant size (matches fetch_real_data.py)

    col_s, col_r = st.columns([1, 2])
    with col_s:
        cap   = st.slider("Battery Capacity (MWh)", 10, 200, 40, step=5)
        power = st.slider("Max Charge/Discharge (MW)", 2, 50, 10, step=1)
        solar_scale = st.slider("Solar Plant Scale (×)", 0.5, 4.0, 1.0, step=0.25)
        st.markdown("**Investment assumptions**")
        battery_cost = st.number_input("Battery cost ($/kWh)", 100, 600, 250, step=25)
        solar_cost   = st.number_input("Solar cost ($/W)", 0.5, 3.0, 1.0, step=0.1)
        horizon      = st.slider("Project lifetime (years)", 5, 25, 15, step=1)
        run_btn = st.button("Run Optimizer", type="primary")

    if run_btn:
        times, median, lo, hi, solar_base = get_forecasts()
        solar_adj = solar_base * solar_scale
        price     = tou_price(times)
        carbon    = carbon_intensity(times)
        cfg_wi    = BatteryConfig(capacity_mwh=cap, max_power_mw=power)
        res_wi    = optimize_battery(median.tolist(), solar_adj.tolist(), price, cfg_wi,
                                     carbon=carbon, carbon_price=0.0)

        with col_r:
            w1, w2, w3 = st.columns(3)
            w1.metric("Daily Savings", f"${res_wi.savings:,.0f}")
            w2.metric("Savings %",     f"{res_wi.savings_pct:.1f}%")
            w3.metric("CO₂ Avoided/day", f"{res_wi.emissions_saved/1000:,.1f} t")

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
                              yaxis_title="MW", barmode="overlay", height=320,
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, width='stretch')

        # ---- ROI engine: turn daily savings into an investment case ----
        st.divider()
        st.markdown("### 💰 Investment Return (ROI)")
        # Value the WHOLE system vs a grid-only baseline (no solar, no battery),
        # so solar's large energy offset — not just the battery's arbitrage — is
        # credited against the capex we're paying for both.
        grid_only_cost = sum(median[t] * price[t] for t in range(len(price)))
        system_daily_savings = grid_only_cost - res_wi.optimized_cost
        annual_savings = system_daily_savings * 365
        solar_capex    = SOLAR_PLANT_MW * solar_scale * 1e6 * solar_cost   # $/W * W
        battery_capex  = cap * 1000 * battery_cost                          # $/kWh * kWh
        roi = compute_roi(annual_savings, solar_capex, battery_capex,
                          horizon_years=horizon, discount_rate_pct=8.0)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total CAPEX", f"${roi.capex/1e6:,.2f}M")
        r2.metric("Net Annual Savings", f"${roi.annual_savings:,.0f}")
        pay = "—" if roi.payback_years == float("inf") else f"{roi.payback_years:.1f} yrs"
        r3.metric("Payback Period", pay)
        irr = "—" if roi.irr_pct != roi.irr_pct else f"{roi.irr_pct:.1f}%"  # nan check
        r4.metric("IRR", irr, help="Internal rate of return vs an 8% hurdle rate.")

        if roi.npv > 0:
            st.success(f"**NPV over {horizon} yrs (8% discount): +\\${roi.npv/1e6:,.2f}M — viable.** "
                       "The project clears the 8% hurdle rate.")
        else:
            st.warning(
                f"**NPV over {horizon} yrs (8% discount): \\${roi.npv/1e6:,.2f}M — below the 8% hurdle.** "
                "This is realistic: energy *arbitrage alone* (an \\$80/\\$40 TOU spread) rarely pays back "
                "grid-scale storage. Real projects stack extra value streams — demand-charge reduction, "
                "capacity-market payments, and resilience — which aren't modelled here. "
                "Try a larger solar scale or cheaper battery (\\$/kWh) to find a viable configuration.")
        st.caption("Payback = CAPEX ÷ net annual savings · NPV/IRR discount future savings to today "
                   "at 8% · savings valued vs a grid-only baseline (no solar, no battery).")
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

        # Score the FULL history so lag-24/168 + rolling features are valid.
        anom_model, anom_feats = load_anomaly_model()

        @st.cache_data(ttl=300)
        def detect_full():
            full = pd.read_csv(DATA, parse_dates=["datetime"]).set_index("datetime")
            s, f = predict_anomalies(full, anom_model, anom_feats)
            return full.index, full["load_mw"].values, s, f

        idx_all, load_all, scores, flags = detect_full()
        n_flagged = int(flags.sum())
        span = f"{idx_all[0]:%b %Y} – {idx_all[-1]:%b %Y}"

        c1, c2, c3 = st.columns(3)
        c1.metric("Anomalies detected", f"{n_flagged:,}")
        c2.metric("Of total hours", f"{n_flagged / len(idx_all) * 100:.1f}%")
        c3.metric("Period analysed", span)
        st.caption("Anomalies are hours whose load is unusual *given the time of day, "
                   "day of week, and recent history* — they cluster in winter cold snaps, "
                   "when demand spikes far above the seasonal norm.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx_all, y=load_all, mode="lines", name="Load",
                                 line=dict(color="#6b7280", width=0.8)))
        fig.add_trace(go.Scatter(x=idx_all[flags], y=load_all[flags], mode="markers",
                                 name="Anomaly", marker=dict(color="#ef4444", size=5, symbol="x")))
        fig.update_layout(title="Load anomalies across full history (zoom/drag to explore)",
                          yaxis_title="Load (MW)", height=380, margin=dict(t=40, b=20))
        st.plotly_chart(fig, width='stretch')

        with st.expander("🔍 Most anomalous hours (lowest score = most unusual)"):
            score_df = pd.DataFrame({"datetime": idx_all, "load_mw": load_all,
                                     "anomaly_score": scores, "is_anomaly": flags})
            st.dataframe(score_df[score_df["is_anomaly"]]
                         .sort_values("anomaly_score").head(50), width='stretch')

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
        st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    if shap_summary.exists():
        col1.image(str(shap_summary), caption="Global SHAP Summary", width='stretch')
    if shap_waterfall.exists():
        col2.image(str(shap_waterfall), caption="Local Waterfall (peak hour)", width='stretch')

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

    # Build a live snapshot so the copilot answers with real current numbers.
    @st.cache_data(ttl=300)
    def _copilot_live_stats():
        t, med, lo_, hi_, sol = get_forecasts()
        price = tou_price(t)
        carbon = carbon_intensity(t)
        res = optimize_battery(med.tolist(), sol.tolist(), price, BatteryConfig(),
                               carbon=carbon, carbon_price=CARBON_PRICE_MODES["cost"])
        res_green = optimize_battery(med.tolist(), sol.tolist(), price, BatteryConfig(),
                                     carbon=carbon, carbon_price=CARBON_PRICE_MODES["green"])

        def co2_phrase(r):
            t_ = r.emissions_saved / 1000
            if t_ >= 0:
                return f"{t_:.1f} tonnes avoided ({r.emissions_saved_pct:.1f}% lower)"
            return f"{abs(t_):.1f} tonnes MORE ({abs(r.emissions_saved_pct):.1f}% higher)"

        stats = {
            "Peak load forecast (next 24h)": f"{med.max():.0f} MW",
            "Min load forecast (next 24h)": f"{med.min():.0f} MW",
            "Peak solar forecast (next 24h)": f"{sol.max():.1f} MW",
            "Battery baseline cost (next 24h)": f"${res.baseline_cost:,.0f}",
            "Battery optimized cost, Cost Mode (next 24h)": f"${res.optimized_cost:,.0f}",
            "Battery cost savings, Cost Mode (next 24h)": f"${res.savings:,.0f} ({res.savings_pct:.1f}%)",
            "CO2 vs no-battery, Cost Mode": co2_phrase(res),
            "CO2 vs no-battery, Green Mode": co2_phrase(res_green),
            "Cost vs Green tradeoff note": ("Cost Mode maximises $ savings but can slightly raise "
                "emissions; Green Mode sacrifices most $ savings to cut the most CO2."),
            "Latest data timestamp": str(df.index[-1]),
        }
        lm_p = MODELS / "load_metrics.json"
        if lm_p.exists():
            lm_ = json.loads(lm_p.read_text())
            stats["Load forecast accuracy (MAPE)"] = f"{lm_.get('MAPE_pct', 0):.2f}%"
            stats["80% interval coverage"] = f"{lm_.get('interval_coverage_80pct', 0):.1f}%"
        return stats

    live_stats = _copilot_live_stats()

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
            "What are today's battery savings and how good is the forecast?",
            "What are the top drivers of electricity demand right now?",
            "Why does the battery charge off-peak and discharge at peak?",
        ]
        for col, q in zip(cols, suggestions):
            if col.button(q, width='stretch'):
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
                    reply = ask_copilot(user_input, context={"live_stats": live_stats},
                                        conversation_history=history or None)
                except Exception as e:
                    reply = f"Error calling AI copilot: {e}"
            st.markdown(reply)

        st.session_state.copilot_messages.append({"role": "assistant", "content": reply})

    if st.session_state.copilot_messages:
        if st.button("Clear chat", width='content'):
            st.session_state.copilot_messages = []
            st.rerun()
