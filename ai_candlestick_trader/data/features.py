"""
data/features.py
================
Converts raw OHLC + Volume DataFrame into a rich feature matrix ready for
the neural network.  All features are normalised per-window so the model is
scale-invariant across different stocks and price levels.

Feature groups
--------------
1. **Candlestick geometry** – body ratio, upper/lower shadow ratios, doji flag
2. **Price returns**        – 1-bar, 3-bar, 5-bar log-returns
3. **Trend indicators**     – SMA-5, SMA-20, SMA-50 distance from close
4. **Momentum**             – RSI(14), Williams %R(14)
5. **Volatility**           – ATR(14), Bollinger-Band width
6. **Volume**               – volume z-score, OBV (on-balance volume)
7. **MACD**                 – macd line, signal line, histogram
8. **Pattern scores**       – from patterns.py (0/1 flags)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [df["High"] - df["Low"],
         (df["High"] - prev_close).abs(),
         (df["Low"]  - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _true_range(df).ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=n, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=n, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hh = df["High"].rolling(n, min_periods=1).max()
    ll = df["Low"].rolling(n,  min_periods=1).min()
    return -100 * (hh - df["Close"]) / (hh - ll + 1e-9)


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal     = _ema(macd_line, sig)
    histogram  = macd_line - signal
    return macd_line, signal, histogram


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, pattern_flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build a feature DataFrame from raw OHLC data.

    Parameters
    ----------
    df            : Raw OHLC DataFrame with columns [Open, High, Low, Close, Volume].
    pattern_flags : Optional binary-flag DataFrame from ``detect_patterns()``
                    (same index as *df*).  If supplied, pattern columns are appended.

    Returns
    -------
    pd.DataFrame  : Feature matrix (one row per bar).  NaN rows from warm-up
                    periods are **not** dropped here — the Dataset class handles that.
    """
    feat = pd.DataFrame(index=df.index)

    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # ── 1. Candlestick geometry ───────────────────────────────────────────────
    body          = c - o
    candle_range  = (h - l).replace(0, np.nan)
    feat["body_ratio"]     = body / candle_range              # positive = bullish
    feat["upper_shadow"]   = (h - pd.concat([c, o], axis=1).max(axis=1)) / candle_range
    feat["lower_shadow"]   = (pd.concat([c, o], axis=1).min(axis=1) - l) / candle_range
    feat["body_abs"]       = body.abs() / candle_range        # body strength
    feat["is_doji"]        = (feat["body_abs"] < 0.1).astype(float)

    # ── 2. Log-returns ────────────────────────────────────────────────────────
    log_c = np.log(c + 1e-9)
    feat["ret_1"]  = log_c.diff(1)
    feat["ret_3"]  = log_c.diff(3)
    feat["ret_5"]  = log_c.diff(5)
    feat["ret_10"] = log_c.diff(10)

    # ── 3. Trend (SMA distance from close, normalised by ATR) ─────────────────
    atr_series = _atr(df, 14)
    for w in [5, 20, 50]:
        feat[f"sma{w}_dist"] = (c - _sma(c, w)) / (atr_series + 1e-9)

    # ── 4. Momentum ───────────────────────────────────────────────────────────
    feat["rsi14"]       = _rsi(c, 14) / 100.0          # normalise to [0,1]
    feat["williams_r"]  = _williams_r(df, 14) / -100.0# normalise to [0,1]

    # ── 5. Volatility ─────────────────────────────────────────────────────────
    feat["atr14"]       = atr_series / (c + 1e-9)       # ATR relative to price
    bb_mid              = _sma(c, 20)
    bb_std              = c.rolling(20, min_periods=1).std()
    feat["bb_width"]    = (2 * bb_std) / (bb_mid + 1e-9)
    feat["bb_pct"]      = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # ── 6. Volume ─────────────────────────────────────────────────────────────
    vol_mean            = v.rolling(20, min_periods=1).mean()
    vol_std             = v.rolling(20, min_periods=1).std().replace(0, np.nan)
    feat["vol_zscore"]  = (v - vol_mean) / (vol_std + 1e-9)
    obv                 = _obv(df)
    obv_norm            = (obv - obv.rolling(50, min_periods=1).mean())
    obv_std             = obv.rolling(50, min_periods=1).std().replace(0, np.nan)
    feat["obv_zscore"]  = obv_norm / (obv_std + 1e-9)

    # ── 7. MACD ───────────────────────────────────────────────────────────────
    macd_line, sig_line, hist = _macd(c)
    feat["macd"]        = macd_line / (c + 1e-9)
    feat["macd_sig"]    = sig_line  / (c + 1e-9)
    feat["macd_hist"]   = hist      / (c + 1e-9)

    # ── 8. Candlestick pattern flags (optional) ───────────────────────────────
    if pattern_flags is not None:
        for col in pattern_flags.columns:
            feat[col] = pattern_flags[col].reindex(feat.index).fillna(0)

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feat


def get_feature_names(include_patterns: bool = True) -> list[str]:
    """Return feature column names (useful for UI labels)."""
    base = [
        "body_ratio", "upper_shadow", "lower_shadow", "body_abs", "is_doji",
        "ret_1", "ret_3", "ret_5", "ret_10",
        "sma5_dist", "sma20_dist", "sma50_dist",
        "rsi14", "williams_r",
        "atr14", "bb_width", "bb_pct",
        "vol_zscore", "obv_zscore",
        "macd", "macd_sig", "macd_hist",
    ]
    return base
