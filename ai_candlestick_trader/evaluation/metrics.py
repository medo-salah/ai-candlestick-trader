"""
evaluation/metrics.py
=====================
Regression and trading-specific evaluation metrics.

Metrics
-------
- MSE / RMSE / MAE        — standard regression quality
- MAPE                    — scale-free error
- Directional Accuracy    — did we get the direction right?
- Pearson R               — correlation between predicted & actual returns
- Sharpe Ratio            — annualised Sharpe of *predicted* signal returns
- Calmar Ratio            — annualised return / max-drawdown
- Max Drawdown            — worst peak-to-trough equity loss
- Win Rate                — fraction of trades with positive P&L
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, list, pd.Series]


def _to_arr(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Regression quality
# ─────────────────────────────────────────────────────────────────────────────

def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t, p = _to_arr(y_true), _to_arr(y_pred)
    return float(np.mean((t - p) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t, p = _to_arr(y_true), _to_arr(y_pred)
    return float(np.mean(np.abs(t - p)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-8) -> float:
    t, p = _to_arr(y_true), _to_arr(y_pred)
    return float(np.mean(np.abs((t - p) / (np.abs(t) + eps))) * 100)


def pearson_r(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t, p = _to_arr(y_true), _to_arr(y_pred)
    if t.std() < 1e-12 or p.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(t, p)[0, 1])


def directional_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of bars where predicted and actual direction match."""
    t, p = _to_arr(y_true), _to_arr(y_pred)
    return float(np.mean(np.sign(t) == np.sign(p)))


# ─────────────────────────────────────────────────────────────────────────────
# Trading performance
# ─────────────────────────────────────────────────────────────────────────────

def _equity_curve(returns: np.ndarray) -> np.ndarray:
    """Cumulative (1+r) product equity curve from a returns array."""
    return np.cumprod(1 + returns)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the equity curve."""
    eq  = _equity_curve(_to_arr(returns))
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / (peak + 1e-12)
    return float(dd.min())


def sharpe_ratio(returns: np.ndarray, annualise: int = 252, rf: float = 0.0) -> float:
    """Annualised Sharpe Ratio."""
    r = _to_arr(returns)
    excess = r - rf / annualise
    if excess.std() < 1e-12:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(annualise))


def calmar_ratio(returns: np.ndarray, annualise: int = 252) -> float:
    """Annualised return / |max drawdown|."""
    r   = _to_arr(returns)
    ann = r.mean() * annualise
    mdd = abs(max_drawdown(r))
    return float(ann / mdd) if mdd > 1e-12 else 0.0


def win_rate(returns: np.ndarray) -> float:
    r = _to_arr(returns)
    return float(np.mean(r > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Bundled evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    y_true:       ArrayLike,
    y_pred:       ArrayLike,
    trade_returns: ArrayLike | None = None,
    annualise:    int = 252,
) -> dict[str, float]:
    """
    Compute all metrics and return them as a dict.

    Parameters
    ----------
    y_true        : True next-bar returns (ground truth).
    y_pred        : Model-predicted returns.
    trade_returns : Actual P&L returns when using the model signal as a trigger.
                    If None, simulated from sign(y_pred) * y_true (long/short).
    annualise     : Trading days per year for Sharpe/Calmar (252 for stocks).

    Returns
    -------
    dict[str, float] with all metric values.
    """
    t, p = _to_arr(y_true), _to_arr(y_pred)

    if trade_returns is None:
        # simple long/short simulation: go long when pred > 0, short when pred < 0
        trade_returns = np.sign(p) * t

    tr = _to_arr(trade_returns)

    return {
        "mse":                  mse(t, p),
        "rmse":                 rmse(t, p),
        "mae":                  mae(t, p),
        "mape_pct":             mape(t, p),
        "pearson_r":            pearson_r(t, p),
        "directional_accuracy": directional_accuracy(t, p),
        "sharpe_ratio":         sharpe_ratio(tr, annualise=annualise),
        "calmar_ratio":         calmar_ratio(tr, annualise=annualise),
        "max_drawdown_pct":     max_drawdown(tr) * 100,
        "win_rate_pct":         win_rate(tr) * 100,
        "total_return_pct":     float((_equity_curve(tr)[-1] - 1) * 100),
    }


def format_metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    """Pretty DataFrame for display in Streamlit or console."""
    labels = {
        "mse":                  "MSE",
        "rmse":                 "RMSE",
        "mae":                  "MAE",
        "mape_pct":             "MAPE (%)",
        "pearson_r":            "Pearson R",
        "directional_accuracy": "Directional Accuracy",
        "sharpe_ratio":         "Sharpe Ratio",
        "calmar_ratio":         "Calmar Ratio",
        "max_drawdown_pct":     "Max Drawdown (%)",
        "win_rate_pct":         "Win Rate (%)",
        "total_return_pct":     "Total Return (%)",
    }
    rows = [
        {"Metric": labels.get(k, k), "Value": f"{v:.4f}"}
        for k, v in metrics.items()
        if k in labels
    ]
    return pd.DataFrame(rows)
