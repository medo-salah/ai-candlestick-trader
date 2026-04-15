"""
dashboard/charts.py
===================
Plotly chart builders for the Streamlit dashboard.
All functions return plotly Figure objects ready for st.plotly_chart().
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ai_candlestick_trader.data.patterns import BULLISH_PATTERNS, BEARISH_PATTERNS

# ── Color palette (dark trading terminal theme) ────────────────────────────────
BULL_COLOR   = "#00d09c"
BEAR_COLOR   = "#ff4d4d"
PRED_COLOR   = "#f5a623"
ACCENT_COLOR = "#5c7cfa"
BG_COLOR     = "#0d1117"
GRID_COLOR   = "#1e2530"
TEXT_COLOR   = "#e6edf3"


def _base_layout(title: str = "", height: int = 500) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=16)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor =BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
        margin=dict(l=50, r=20, t=50, b=40),
        height=height,
    )


def candlestick_chart(
    ohlc_df:         pd.DataFrame,
    pattern_flags:   Optional[pd.DataFrame] = None,
    predicted_close: Optional[pd.Series]    = None,
    signals:         Optional[pd.Series]    = None,
    title:           str = "Candlestick Chart",
    show_volume:     bool = True,
) -> go.Figure:
    """
    Full interactive candlestick chart with:
    - OHLC candlesticks
    - Volume bars in subplot
    - Pattern annotations (arrows + labels)
    - Predicted price line overlay
    - Buy/Sell signal markers

    Parameters
    ----------
    ohlc_df         : DataFrame with Open, High, Low, Close, Volume.
    pattern_flags   : Binary flag DataFrame from detect_patterns().
    predicted_close : Predicted closing price series (same index as ohlc_df).
    signals         : +1 / 0 / -1 signal series (same index as ohlc_df).
    title           : Chart title.
    show_volume     : Whether to add a volume subplot.
    """
    rows = 2 if show_volume else 1
    row_h = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_h,
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=ohlc_df.index,
            open=ohlc_df["Open"],
            high=ohlc_df["High"],
            low=ohlc_df["Low"],
            close=ohlc_df["Close"],
            increasing_line_color=BULL_COLOR,
            decreasing_line_color=BEAR_COLOR,
            increasing_fillcolor=BULL_COLOR,
            decreasing_fillcolor=BEAR_COLOR,
            name="OHLC",
            line=dict(width=1),
        ),
        row=1, col=1,
    )

    # ── Predicted price line ──────────────────────────────────────────────────
    if predicted_close is not None:
        fig.add_trace(
            go.Scatter(
                x=predicted_close.index,
                y=predicted_close.values,
                mode="lines",
                name="AI Prediction",
                line=dict(color=PRED_COLOR, width=1.5, dash="dot"),
                opacity=0.85,
            ),
            row=1, col=1,
        )

    # ── Pattern annotations ───────────────────────────────────────────────────
    if pattern_flags is not None:
        common_idx = ohlc_df.index.intersection(pattern_flags.index)
        for pat_col in pattern_flags.columns:
            is_bull = pat_col in BULLISH_PATTERNS
            color   = BULL_COLOR if is_bull else BEAR_COLOR
            sign    = 1 if is_bull else -1
            mask    = pattern_flags.loc[common_idx, pat_col] == 1
            dates   = common_idx[mask]
            if len(dates) == 0:
                continue
            prices  = ohlc_df.loc[dates, "Low" if is_bull else "High"]
            offset  = sign * prices.abs() * 0.008
            label   = pat_col.replace("pat_", "").replace("_", " ").title()
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=(prices + offset).values,
                    mode="markers+text",
                    marker=dict(
                        symbol="triangle-up" if is_bull else "triangle-down",
                        color=color,
                        size=10,
                        opacity=0.85,
                    ),
                    text=label,
                    textposition="top center" if is_bull else "bottom center",
                    textfont=dict(size=9, color=color),
                    name=label,
                    showlegend=False,
                    hovertemplate=f"<b>{label}</b><br>%{{x}}<extra></extra>",
                ),
                row=1, col=1,
            )

    # ── Buy / Sell signals ────────────────────────────────────────────────────
    if signals is not None:
        common_sig = ohlc_df.index.intersection(signals.index)
        buys  = common_sig[signals.loc[common_sig] == 1]
        sells = common_sig[signals.loc[common_sig] == -1]

        if len(buys):
            fig.add_trace(
                go.Scatter(
                    x=buys,
                    y=ohlc_df.loc[buys, "Low"] * 0.992,
                    mode="markers",
                    marker=dict(symbol="triangle-up", color=BULL_COLOR, size=14),
                    name="BUY Signal",
                    hovertemplate="<b>BUY</b> %{x}<extra></extra>",
                ),
                row=1, col=1,
            )
        if len(sells):
            fig.add_trace(
                go.Scatter(
                    x=sells,
                    y=ohlc_df.loc[sells, "High"] * 1.008,
                    mode="markers",
                    marker=dict(symbol="triangle-down", color=BEAR_COLOR, size=14),
                    name="SELL Signal",
                    hovertemplate="<b>SELL</b> %{x}<extra></extra>",
                ),
                row=1, col=1,
            )

    # ── Volume bars ───────────────────────────────────────────────────────────
    if show_volume and "Volume" in ohlc_df.columns:
        vol_colors = [
            BULL_COLOR if c >= o else BEAR_COLOR
            for c, o in zip(ohlc_df["Close"], ohlc_df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=ohlc_df.index,
                y=ohlc_df["Volume"],
                marker_color=vol_colors,
                name="Volume",
                opacity=0.6,
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor=GRID_COLOR)

    layout = _base_layout(title=title, height=600)
    fig.update_layout(
        **layout,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, showgrid=True)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, showgrid=True)

    return fig


def equity_curve_chart(equity: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """Filled area equity curve chart."""
    pct     = (equity - 1) * 100
    color   = BULL_COLOR if pct.iloc[-1] >= 0 else BEAR_COLOR
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=pct.values,
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            line=dict(color=color, width=2),
            name="Portfolio Return %",
        )
    )
    fig.add_hline(y=0, line=dict(color=TEXT_COLOR, width=0.5, dash="dot"))
    fig.update_layout(**_base_layout(title=title, height=350))
    return fig


def metrics_bar_chart(metrics: dict[str, float]) -> go.Figure:
    """Horizontal bar chart for key trading metrics."""
    keys = ["directional_accuracy", "win_rate_pct", "sharpe_ratio", "calmar_ratio"]
    labels = {
        "directional_accuracy": "Directional Accuracy",
        "win_rate_pct":         "Win Rate %",
        "sharpe_ratio":         "Sharpe Ratio",
        "calmar_ratio":         "Calmar Ratio",
    }
    vals   = [metrics.get(k, 0) for k in keys]
    colors = [BULL_COLOR if v > 0 else BEAR_COLOR for v in vals]
    fig    = go.Figure(
        go.Bar(
            y=[labels[k] for k in keys], x=vals,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(**_base_layout(title="Performance Metrics", height=280))
    return fig


def pattern_frequency_chart(pattern_counts: dict[str, int]) -> go.Figure:
    """Bar chart of pattern occurrences."""
    sorted_p = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    names    = [p[0].replace("pat_", "").replace("_", " ").title() for p in sorted_p]
    counts   = [p[1] for p in sorted_p]
    colors   = [
        BULL_COLOR if any(b in p[0] for b in ["bull", "hammer", "morning", "pierce", "soldier", "dragon"])
        else BEAR_COLOR
        for p in sorted_p
    ]
    fig = go.Figure(go.Bar(x=names, y=counts, marker_color=colors, opacity=0.85))
    fig.update_layout(**_base_layout(title="Pattern Frequency", height=320))
    fig.update_xaxes(tickangle=-35)
    return fig


def prediction_scatter(
    actuals: pd.Series,
    predictions: pd.Series,
    title: str = "Actual vs Predicted Returns",
) -> go.Figure:
    """Scatter plot of true vs predicted returns."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actuals.values, y=predictions.values,
            mode="markers",
            marker=dict(color=ACCENT_COLOR, size=5, opacity=0.6),
            name="Predictions",
        )
    )
    # perfect prediction line
    lim = max(abs(actuals).max(), abs(predictions).max()) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[-lim, lim], y=[-lim, lim],
            mode="lines",
            line=dict(color=TEXT_COLOR, dash="dot", width=1),
            name="Perfect",
        )
    )
    fig.update_layout(**_base_layout(title=title, height=380))
    return fig
