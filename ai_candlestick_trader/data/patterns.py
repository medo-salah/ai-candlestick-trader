"""
data/patterns.py
================
Pure-Python detection of 15 classic Japanese candlestick patterns.
No TA-Lib required.

Each detector returns a boolean/binary Series aligned to the input DataFrame
index.  ``detect_patterns()`` bundles all of them into a single DataFrame.

Patterns implemented
--------------------
Bullish: Hammer, Bullish Engulfing, Morning Star, Piercing Line,
         Bullish Harami, Three White Soldiers, Bullish Marubozu,
         Dragonfly Doji

Bearish: Shooting Star, Bearish Engulfing, Evening Star, Dark Cloud Cover,
         Bearish Harami, Three Black Crows, Bearish Marubozu,
         Gravestone Doji

Neutral: Doji, Spinning Top
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _body(df: pd.DataFrame) -> pd.Series:
    return (df["Close"] - df["Open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    return (df["High"] - df["Low"]).replace(0, np.nan)


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["High"] - pd.concat([df["Close"], df["Open"]], axis=1).max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return pd.concat([df["Close"], df["Open"]], axis=1).min(axis=1) - df["Low"]


def _is_bullish(df: pd.DataFrame) -> pd.Series:
    return df["Close"] > df["Open"]


def _is_bearish(df: pd.DataFrame) -> pd.Series:
    return df["Close"] < df["Open"]


def _body_ratio(df: pd.DataFrame) -> pd.Series:
    return _body(df) / _range(df)


# ─────────────────────────────────────────────────────────────────────────────
# Individual pattern detectors
# ─────────────────────────────────────────────────────────────────────────────

def _hammer(df: pd.DataFrame) -> pd.Series:
    """Long lower shadow ≥ 2× body, small upper shadow, bullish signal."""
    ls   = _lower_shadow(df)
    us   = _upper_shadow(df)
    body = _body(df)
    return ((ls >= 2 * body) & (us <= 0.3 * body) & (body > 0)).astype(int)


def _shooting_star(df: pd.DataFrame) -> pd.Series:
    """Long upper shadow ≥ 2× body, small lower shadow, bearish signal."""
    ls   = _lower_shadow(df)
    us   = _upper_shadow(df)
    body = _body(df)
    return ((us >= 2 * body) & (ls <= 0.3 * body) & (body > 0)).astype(int)


def _doji(df: pd.DataFrame) -> pd.Series:
    """Body < 10 % of range."""
    return (_body_ratio(df) < 0.10).astype(int)


def _spinning_top(df: pd.DataFrame) -> pd.Series:
    """Body 10–30 % of range with meaningful shadows."""
    br = _body_ratio(df)
    return ((br >= 0.10) & (br <= 0.30)).astype(int)


def _dragonfly_doji(df: pd.DataFrame) -> pd.Series:
    """Doji + long lower shadow, almost no upper shadow."""
    doji_flag = _doji(df)
    ls   = _lower_shadow(df)
    us   = _upper_shadow(df)
    rng  = _range(df)
    return (doji_flag & (ls >= 0.6 * rng) & (us <= 0.05 * rng)).astype(int)


def _gravestone_doji(df: pd.DataFrame) -> pd.Series:
    """Doji + long upper shadow, almost no lower shadow."""
    doji_flag = _doji(df)
    ls   = _lower_shadow(df)
    us   = _upper_shadow(df)
    rng  = _range(df)
    return (doji_flag & (us >= 0.6 * rng) & (ls <= 0.05 * rng)).astype(int)


def _bullish_marubozu(df: pd.DataFrame) -> pd.Series:
    """Bullish candle, body > 90 % of range."""
    return (_is_bullish(df) & (_body_ratio(df) > 0.90)).astype(int)


def _bearish_marubozu(df: pd.DataFrame) -> pd.Series:
    """Bearish candle, body > 90 % of range."""
    return (_is_bearish(df) & (_body_ratio(df) > 0.90)).astype(int)


# ── Two-bar patterns ──────────────────────────────────────────────────────────

def _bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_bear  = _is_bearish(df).shift(1)
    curr_bull  = _is_bullish(df)
    engulfs    = (df["Open"] < df["Close"].shift(1)) & (df["Close"] > df["Open"].shift(1))
    return (prev_bear & curr_bull & engulfs).fillna(False).astype(int)


def _bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_bull  = _is_bullish(df).shift(1)
    curr_bear  = _is_bearish(df)
    engulfs    = (df["Open"] > df["Close"].shift(1)) & (df["Close"] < df["Open"].shift(1))
    return (prev_bull & curr_bear & engulfs).fillna(False).astype(int)


def _bullish_harami(df: pd.DataFrame) -> pd.Series:
    prev_bear  = _is_bearish(df).shift(1)
    curr_bull  = _is_bullish(df)
    inside     = (df["Open"] > df["Close"].shift(1)) & (df["Close"] < df["Open"].shift(1))
    return (prev_bear & curr_bull & inside).fillna(False).astype(int)


def _bearish_harami(df: pd.DataFrame) -> pd.Series:
    prev_bull  = _is_bullish(df).shift(1)
    curr_bear  = _is_bearish(df)
    inside     = (df["Open"] < df["Close"].shift(1)) & (df["Close"] > df["Open"].shift(1))
    return (prev_bull & curr_bear & inside).fillna(False).astype(int)


def _piercing_line(df: pd.DataFrame) -> pd.Series:
    prev_bear  = _is_bearish(df).shift(1)
    curr_bull  = _is_bullish(df)
    mid_prev   = (df["Open"].shift(1) + df["Close"].shift(1)) / 2
    gap_down   = df["Open"] < df["Low"].shift(1)
    close_mid  = df["Close"] > mid_prev
    return (prev_bear & curr_bull & gap_down & close_mid).fillna(False).astype(int)


def _dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    prev_bull  = _is_bullish(df).shift(1)
    curr_bear  = _is_bearish(df)
    mid_prev   = (df["Open"].shift(1) + df["Close"].shift(1)) / 2
    gap_up     = df["Open"] > df["High"].shift(1)
    close_mid  = df["Close"] < mid_prev
    return (prev_bull & curr_bear & gap_up & close_mid).fillna(False).astype(int)


# ── Three-bar patterns ────────────────────────────────────────────────────────

def _morning_star(df: pd.DataFrame) -> pd.Series:
    bar1_bear = _is_bearish(df).shift(2)             # large bearish
    star      = _body_ratio(df).shift(1) < 0.30      # small body (star)
    bar3_bull = _is_bullish(df)                       # large bullish
    recov     = df["Close"] > (df["Open"].shift(2) + df["Close"].shift(2)) / 2
    return (bar1_bear & star & bar3_bull & recov).fillna(False).astype(int)


def _evening_star(df: pd.DataFrame) -> pd.Series:
    bar1_bull = _is_bullish(df).shift(2)
    star      = _body_ratio(df).shift(1) < 0.30
    bar3_bear = _is_bearish(df)
    recov     = df["Close"] < (df["Open"].shift(2) + df["Close"].shift(2)) / 2
    return (bar1_bull & star & bar3_bear & recov).fillna(False).astype(int)


def _three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    b1 = _is_bullish(df).shift(2) & (_body_ratio(df).shift(2) > 0.6)
    b2 = _is_bullish(df).shift(1) & (_body_ratio(df).shift(1) > 0.6)
    b3 = _is_bullish(df)          & (_body_ratio(df)          > 0.6)
    c1 = df["Close"].shift(1) > df["Close"].shift(2)
    c2 = df["Close"]          > df["Close"].shift(1)
    return (b1 & b2 & b3 & c1 & c2).fillna(False).astype(int)


def _three_black_crows(df: pd.DataFrame) -> pd.Series:
    b1 = _is_bearish(df).shift(2) & (_body_ratio(df).shift(2) > 0.6)
    b2 = _is_bearish(df).shift(1) & (_body_ratio(df).shift(1) > 0.6)
    b3 = _is_bearish(df)          & (_body_ratio(df)          > 0.6)
    c1 = df["Close"].shift(1) < df["Close"].shift(2)
    c2 = df["Close"]          < df["Close"].shift(1)
    return (b1 & b2 & b3 & c1 & c2).fillna(False).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

PATTERN_FUNCS: dict[str, callable] = {
    # Bullish
    "pat_hammer":            _hammer,
    "pat_bullish_engulf":    _bullish_engulfing,
    "pat_bullish_harami":    _bullish_harami,
    "pat_piercing_line":     _piercing_line,
    "pat_morning_star":      _morning_star,
    "pat_three_white_sol":   _three_white_soldiers,
    "pat_bullish_marubozu":  _bullish_marubozu,
    "pat_dragonfly_doji":    _dragonfly_doji,
    # Bearish
    "pat_shooting_star":     _shooting_star,
    "pat_bearish_engulf":    _bearish_engulfing,
    "pat_bearish_harami":    _bearish_harami,
    "pat_dark_cloud":        _dark_cloud_cover,
    "pat_evening_star":      _evening_star,
    "pat_three_black_crows": _three_black_crows,
    "pat_bearish_marubozu":  _bearish_marubozu,
    "pat_gravestone_doji":   _gravestone_doji,
    # Neutral
    "pat_doji":              _doji,
    "pat_spinning_top":      _spinning_top,
}

BULLISH_PATTERNS = [k for k in PATTERN_FUNCS if "bullish" in k or k in
                    ("pat_hammer", "pat_piercing_line", "pat_morning_star",
                     "pat_three_white_sol", "pat_dragonfly_doji")]

BEARISH_PATTERNS = [k for k in PATTERN_FUNCS if "bearish" in k or k in
                    ("pat_shooting_star", "pat_dark_cloud", "pat_evening_star",
                     "pat_three_black_crows", "pat_gravestone_doji")]


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect all 18 candlestick patterns in *df* and return a binary flag
    DataFrame with one column per pattern.

    Parameters
    ----------
    df : OHLC DataFrame (columns: Open, High, Low, Close, Volume).

    Returns
    -------
    pd.DataFrame : Boolean/int columns, same index as *df*.
    """
    result: dict[str, pd.Series] = {}
    for name, fn in PATTERN_FUNCS.items():
        try:
            result[name] = fn(df)
        except Exception:
            result[name] = pd.Series(0, index=df.index)
    return pd.DataFrame(result, index=df.index)


def pattern_summary(flags: pd.DataFrame) -> dict[str, int]:
    """Return total occurrence count per pattern for the given flag DataFrame."""
    return {col: int(flags[col].sum()) for col in flags.columns}
