"""
dashboard/app.py
================
AI Candlestick Trader — Streamlit Dashboard

Run with:
    streamlit run dashboard.py          (from project root)
    -- or --
    act-dash                            (if installed via pip)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Candlestick Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark trading terminal
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* Header gradient banner */
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid #1e2530;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0; color: #e6edf3; }
.hero p  { font-size: 1rem; color: #8b949e; margin: 4px 0 0; }

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #5c7cfa; }
.metric-val { font-size: 1.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-lbl { font-size: 0.78rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }

/* Signal badge */
.signal-buy  { background: #0d2d1f; border: 1.5px solid #00d09c; color: #00d09c;
               border-radius: 8px; padding: 10px 18px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
.signal-sell { background: #2d0d0d; border: 1.5px solid #ff4d4d; color: #ff4d4d;
               border-radius: 8px; padding: 10px 18px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
.signal-hold { background: #1a1a2e; border: 1.5px solid #8b949e; color: #8b949e;
               border-radius: 8px; padding: 10px 18px; font-weight: 700; font-size: 1.1rem; display: inline-block; }

/* Strikout sidebar */
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2530; }

/* Plotly container */
.js-plotly-plot { border-radius: 10px; }

/* Tabs */
button[data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-size: 0.9rem; }

/* Code blocks */
pre { background: #161b22 !important; border: 1px solid #1e2530; border-radius: 8px; }

/* Divider */
hr { border-color: #1e2530; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Imports (deferred to avoid loading time on page refresh)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _import_pkg():
    from ai_candlestick_trader.data.downloader import download_ohlc, ALL_TICKERS
    from ai_candlestick_trader.data.features   import build_features
    from ai_candlestick_trader.data.patterns   import detect_patterns, pattern_summary
    from ai_candlestick_trader.data.dataset    import OHLCDataset, split_dataset
    from ai_candlestick_trader.models.transformer_model import CandlestickTransformer
    from ai_candlestick_trader.models.lstm_model        import CandlestickLSTM
    from ai_candlestick_trader.evaluation.metrics       import evaluate_predictions, format_metrics_table
    from ai_candlestick_trader.evaluation.backtester    import Backtester
    from ai_candlestick_trader.dashboard.charts import (
        candlestick_chart, equity_curve_chart, metrics_bar_chart,
        pattern_frequency_chart, prediction_scatter,
    )
    return {
        "download_ohlc": download_ohlc,
        "ALL_TICKERS":   ALL_TICKERS,
        "build_features": build_features,
        "detect_patterns": detect_patterns,
        "pattern_summary": pattern_summary,
        "OHLCDataset": OHLCDataset,
        "split_dataset": split_dataset,
        "CandlestickTransformer": CandlestickTransformer,
        "CandlestickLSTM": CandlestickLSTM,
        "evaluate_predictions": evaluate_predictions,
        "format_metrics_table": format_metrics_table,
        "Backtester": Backtester,
        "candlestick_chart": candlestick_chart,
        "equity_curve_chart": equity_curve_chart,
        "metrics_bar_chart": metrics_bar_chart,
        "pattern_frequency_chart": pattern_frequency_chart,
        "prediction_scatter": prediction_scatter,
    }


p = _import_pkg()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    market = st.selectbox("🌍 Market", ["EGX (Egypt)", "Tadawul (Saudi Arabia)", "Global"])

    all_tickers: dict[str, str] = p["ALL_TICKERS"]
    if market == "EGX (Egypt)":
        from ai_candlestick_trader.data.downloader import EGX_TICKERS
        ticker_map = EGX_TICKERS
    elif market == "Tadawul (Saudi Arabia)":
        from ai_candlestick_trader.data.downloader import TADAWUL_TICKERS
        ticker_map = TADAWUL_TICKERS
    else:
        ticker_map = all_tickers

    ticker_name = st.selectbox("📊 Stock", list(ticker_map.keys()))
    ticker_sym  = ticker_map[ticker_name]

    st.caption(f"Yahoo Finance symbol: `{ticker_sym}`")
    st.markdown("---")

    period   = st.selectbox("📅 History Period", ["1y", "2y", "3y", "5y"], index=2)
    interval = st.selectbox("⏱ Interval", ["1d", "1wk"], index=0)
    seq_len  = st.slider("🔁 Sequence Length (bars)", 10, 60, 30)

    st.markdown("---")
    st.markdown("### 🤖 Model")
    model_type  = st.selectbox("Architecture", ["Transformer", "LSTM"])
    signal_thr  = st.slider("Signal Threshold", 0.001, 0.02, 0.005, step=0.001,
                             help="Minimum predicted return to trigger a Buy/Sell signal")

    ckpt_path   = st.text_input("Checkpoint (.pt)", value="checkpoints/best_model.pt")
    run_btn     = st.button("🚀 Load Data & Analyse", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small>AI Candlestick Trader v2.0<br>🇪🇬 EGX · 🇸🇦 Tadawul</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
<div class="hero">
  <h1>📈 AI Candlestick Trader</h1>
  <p>Deep-learning candlestick pattern recognition · EGX & Tadawul coverage · Real-time signals</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "ohlc_df"   not in st.session_state: st.session_state["ohlc_df"]   = None
if "feat_df"   not in st.session_state: st.session_state["feat_df"]   = None
if "pat_df"    not in st.session_state: st.session_state["pat_df"]    = None
if "dataset"   not in st.session_state: st.session_state["dataset"]   = None
if "bt_result" not in st.session_state: st.session_state["bt_result"] = None
if "model"     not in st.session_state: st.session_state["model"]     = None

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"⏬ Downloading {ticker_name} ({ticker_sym}) …"):
        try:
            ohlc_df = p["download_ohlc"](ticker_sym, period=period, interval=interval)
            st.session_state["ohlc_df"] = ohlc_df
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            st.stop()

    with st.spinner("🔬 Computing features & detecting patterns …"):
        pat_df  = p["detect_patterns"](ohlc_df)
        feat_df = p["build_features"](ohlc_df, pattern_flags=pat_df)
        st.session_state["pat_df"]  = pat_df
        st.session_state["feat_df"] = feat_df

    with st.spinner("🏗️ Building dataset …"):
        from ai_candlestick_trader.data.dataset import OHLCDataset
        ds = OHLCDataset(feat_df, ohlc_df["Close"], seq_len=seq_len)
        st.session_state["dataset"] = ds

    # Try loading a pretrained model if checkpoint exists
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(ckpt_path):
        with st.spinner("🔮 Loading model checkpoint …"):
            try:
                n_feat = ds.n_features
                if model_type == "Transformer":
                    model = p["CandlestickTransformer"](n_features=n_feat)
                else:
                    model = p["CandlestickLSTM"](n_features=n_feat)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt.get("model_state_dict", ckpt))
                model.to(device).eval()
                st.session_state["model"] = model

                # Run backtest
                bt = p["Backtester"](model, device=device, threshold=signal_thr)
                from torch.utils.data import DataLoader
                result = bt.run(ds, ohlc_df, batch_size=64)
                st.session_state["bt_result"] = result
                st.success("✅ Model loaded and backtest complete!")
            except Exception as e:
                st.warning(f"⚠️ Could not load checkpoint: {e}. Showing data analysis only.")
    else:
        st.info("ℹ️ No checkpoint found — showing data & pattern analysis. Train a model first.")

# ─────────────────────────────────────────────────────────────────────────────
# Main content (only shown after data is loaded)
# ─────────────────────────────────────────────────────────────────────────────

ohlc_df   = st.session_state.get("ohlc_df")
pat_df    = st.session_state.get("pat_df")
feat_df   = st.session_state.get("feat_df")
bt_result = st.session_state.get("bt_result")

if ohlc_df is None:
    st.markdown(
        """
<div style='text-align:center; padding:60px 0; color:#8b949e;'>
    <div style='font-size:4rem;'>📉</div>
    <h3>Select a stock and click <strong>Load Data & Analyse</strong></h3>
    <p>Supports EGX 30 and Tadawul 30 stocks with 5+ years of daily bars.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────────

latest = ohlc_df.iloc[-1]
prev   = ohlc_df.iloc[-2]
chg    = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
chg_col = "#00d09c" if chg >= 0 else "#ff4d4d"

pat_counts = p["pattern_summary"](pat_df)
most_recent_pats = []
for col in pat_df.columns:
    if pat_df[col].iloc[-5:].any():
        most_recent_pats.append(col.replace("pat_", "").replace("_", " ").title())

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
cards = [
    (kpi1, "Close",       f"{latest['Close']:.2f}",                        "Latest Close"),
    (kpi2, "Change",      f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%",    "1-Day Change",  chg_col),
    (kpi3, "Volume",      f"{int(latest['Volume']):,}",                     "Volume"),
    (kpi4, "Bars",        f"{len(ohlc_df):,}",                              "Total Bars"),
    (kpi5, "Patterns",    f"{sum(pat_counts.values()):,}",                  "Detected Patterns"),
]
for col, *rest in cards:
    color = rest[2] if len(rest) > 2 else "#5c7cfa"
    with col:
        st.markdown(
            f"""<div class='metric-card'>
            <div class='metric-val' style='color:{color}'>{rest[1]}</div>
            <div class='metric-lbl'>{rest[2] if len(rest)>2 else rest[1]}</div>
            </div>""".replace(rest[1], rest[0]),  # <-- swap for actual value
            unsafe_allow_html=True,
        )

# Re-render properly
for col, label, val, lbl, *extra in [
    (kpi1, "Close",    f"{latest['Close']:.2f}",              "Latest Close",   "#e6edf3"),
    (kpi2, "Change",   f"{'▲' if chg>=0 else '▼'}{abs(chg):.2f}%", "1D Change", chg_col),
    (kpi3, "Volume",   f"{int(latest['Volume'])/1e6:.2f}M",   "Volume",         "#e6edf3"),
    (kpi4, "Bars",     f"{len(ohlc_df):,}",                   "Total Bars",     "#e6edf3"),
    (kpi5, "Patterns", f"{sum(pat_counts.values())}",          "Patterns Found", "#f5a623"),
]:
    color = extra[0] if extra else "#e6edf3"
    col.empty()
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-val' style='color:{color}'>{val}</div>"
        f"<div class='metric-lbl'>{lbl}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Signal badge ──────────────────────────────────────────────────────────────

if bt_result is not None:
    last_sig = bt_result.signals.iloc[-1]
    sig_label = {1: "🚀 BUY", -1: "⚠️ SELL", 0: "⏸ HOLD"}.get(int(last_sig), "⏸ HOLD")
    sig_cls   = {1: "signal-buy", -1: "signal-sell", 0: "signal-hold"}.get(int(last_sig), "signal-hold")
    last_pred = bt_result.predictions.iloc[-1]

    sig_col, conf_col, _ = st.columns([2, 2, 6])
    with sig_col:
        st.markdown(f"<div class='{sig_cls}'>{sig_label}</div>", unsafe_allow_html=True)
    with conf_col:
        st.markdown(
            f"<div style='padding:10px 0; color:#8b949e; font-size:0.9rem;'>"
            f"Predicted return: <span style='color:#f5a623; font-weight:600'>"
            f"{last_pred*100:+.3f}%</span></div>",
            unsafe_allow_html=True,
        )

    if most_recent_pats:
        st.markdown(
            f"<small style='color:#8b949e'>Last 5 bars patterns: "
            f"<span style='color:#5c7cfa'>{', '.join(most_recent_pats[:5])}</span></small>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Chart",
    "🔮 Predictions",
    "📋 Patterns",
    "📈 Backtest",
    "🧮 Raw Data",
])

with tab1:
    signals_series = bt_result.signals if bt_result else None
    pred_close     = None
    if bt_result is not None:
        # Convert predicted returns to predicted close prices
        last_closes = ohlc_df["Close"].reindex(bt_result.predictions.index)
        pred_close  = last_closes * (1 + bt_result.predictions)
        pred_close.name = "Predicted Close"

    n_bars = st.slider("Show last N bars", 50, min(1000, len(ohlc_df)), 200, key="n_bars_slider")
    fig = p["candlestick_chart"](
        ohlc_df.iloc[-n_bars:],
        pattern_flags=pat_df.iloc[-n_bars:] if pat_df is not None else None,
        predicted_close=pred_close.iloc[-n_bars:] if pred_close is not None else None,
        signals=signals_series.iloc[-n_bars:] if signals_series is not None else None,
        title=f"{ticker_name} ({ticker_sym})",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if bt_result is None:
        st.info("Train and load a model checkpoint to see predictions.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig2 = p["prediction_scatter"](bt_result.actuals, bt_result.predictions)
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            # Predicted vs actual return time series
            import plotly.graph_objects as go
            fig3 = go.Figure()
            fig3.add_trace(go.Scattergl(
                x=bt_result.actuals.index, y=bt_result.actuals.values * 100,
                mode="lines", name="Actual %", line=dict(color="#00d09c", width=1)
            ))
            fig3.add_trace(go.Scattergl(
                x=bt_result.predictions.index, y=bt_result.predictions.values * 100,
                mode="lines", name="Predicted %", line=dict(color="#f5a623", width=1, dash="dot")
            ))
            fig3.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"), height=380,
                title="Actual vs Predicted Returns (%)",
                hovermode="x unified",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#1e2530"),
                yaxis=dict(gridcolor="#1e2530"),
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### 📐 Metrics")
        from ai_candlestick_trader.evaluation.metrics import format_metrics_table
        metrics_df = format_metrics_table(bt_result.metrics)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with tab3:
    c1, c2 = st.columns([2, 1])
    with c1:
        fig4 = p["pattern_frequency_chart"](pat_counts)
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        st.markdown("#### Recent Patterns (last 20 bars)")
        recent_pats = pat_df.iloc[-20:].copy()
        detected = recent_pats[recent_pats.any(axis=1)]
        for ts, row in detected.iterrows():
            pats = [c.replace("pat_","").replace("_"," ").title() for c in row.index if row[c] == 1]
            if pats:
                st.markdown(
                    f"<small><code>{str(ts)[:10]}</code> → "
                    f"<span style='color:#5c7cfa'>{', '.join(pats)}</span></small>",
                    unsafe_allow_html=True,
                )

with tab4:
    if bt_result is None:
        st.info("Load a model checkpoint to view backtest results.")
    else:
        fig5 = p["equity_curve_chart"](bt_result.equity_curve)
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("#### 📊 Performance Metrics")
        fig6 = p["metrics_bar_chart"](bt_result.metrics)
        st.plotly_chart(fig6, use_container_width=True)

        if bt_result.trades:
            st.markdown(f"#### 🗒️ Trade Log  ({len(bt_result.trades)} trades)")
            trade_rows = [
                {
                    "Date":      t.entry_date.strftime("%Y-%m-%d") if hasattr(t.entry_date, "strftime") else str(t.entry_date),
                    "Direction": "🟢 LONG" if t.direction == 1 else "🔴 SHORT",
                    "Return %":  f"{t.return_pct:+.3f}",
                }
                for t in bt_result.trades[-50:]
            ]
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

with tab5:
    st.markdown("#### OHLC Data (last 100 bars)")
    disp = ohlc_df.iloc[-100:].copy()
    disp.index = disp.index.strftime("%Y-%m-%d")
    st.dataframe(
        disp.style.format({"Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}",
                           "Volume": "{:,.0f}"}),
        use_container_width=True,
    )

    if feat_df is not None:
        st.markdown("#### Feature Matrix (last 50 bars)")
        st.dataframe(feat_df.iloc[-50:].round(4), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<hr/>
<div style='text-align:center; color:#8b949e; font-size:0.78rem; padding: 12px 0;'>
    AI Candlestick Trader v2.0 &nbsp;|&nbsp;
    Built with PyTorch · Streamlit · Plotly &nbsp;|&nbsp;
    🇪🇬 Egypt Stock Exchange &nbsp;·&nbsp; 🇸🇦 Tadawul &nbsp;|&nbsp;
    <em>Not financial advice. For research purposes only.</em>
</div>
""",
    unsafe_allow_html=True,
)
