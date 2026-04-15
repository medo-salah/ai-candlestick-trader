"""
tests/test_patterns.py
======================
Unit tests for individual candlestick pattern detectors.
"""

import numpy as np
import pandas as pd
import pytest

from ai_candlestick_trader.data.patterns import (
    detect_patterns,
    _hammer, _shooting_star, _doji,
    _bullish_engulfing, _bearish_engulfing,
    _bullish_marubozu, _bearish_marubozu,
    _dragonfly_doji, _gravestone_doji,
    BULLISH_PATTERNS, BEARISH_PATTERNS,
)


def _make_ohlc(**kwargs) -> pd.DataFrame:
    """Build a 1-row OHLC DataFrame from keyword args."""
    return pd.DataFrame({
        "Open":   [kwargs.get("o", 10.0)],
        "High":   [kwargs.get("h", 12.0)],
        "Low":    [kwargs.get("l", 8.0)],
        "Close":  [kwargs.get("c", 10.5)],
        "Volume": [1_000_000.0],
    })


def _make_two_bar(
    o1, h1, l1, c1,
    o2, h2, l2, c2,
) -> pd.DataFrame:
    """Build a 2-bar OHLC DataFrame."""
    return pd.DataFrame({
        "Open":   [o1, o2],
        "High":   [h1, h2],
        "Low":    [l1, l2],
        "Close":  [c1, c2],
        "Volume": [1e6, 1e6],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Single-bar patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestHammer:
    def test_hammer_detected(self):
        # o=11, c=11.2, h=11.25, l=9  → body=0.2, lower shadow=2.0 (>= 2x 0.2), upper=0.05
        df = _make_ohlc(o=11.0, c=11.2, h=11.25, l=9.0)
        assert _hammer(df).iloc[0] == 1

    def test_shooting_star_not_hammer(self):
        # Shooting star: open ≈ close near bottom, very long upper shadow
        df = _make_ohlc(o=10.0, c=10.2, h=14.0, l=9.8)
        assert _hammer(df).iloc[0] == 0


class TestShootingStar:
    def test_shooting_star_detected(self):
        df = _make_ohlc(o=11.8, c=11.0, h=15.0, l=10.9)
        assert _shooting_star(df).iloc[0] == 1


class TestDoji:
    def test_doji_detected(self):
        # Open ≈ Close, range is large
        df = _make_ohlc(o=10.0, c=10.05, h=12.0, l=8.0)
        assert _doji(df).iloc[0] == 1

    def test_no_doji_large_body(self):
        df = _make_ohlc(o=10.0, c=11.5, h=12.0, l=9.5)
        assert _doji(df).iloc[0] == 0


class TestMarubozu:
    def test_bullish_marubozu(self):
        # near-zero shadows, bullish body fills almost all range
        df = _make_ohlc(o=10.0, c=12.0, h=12.02, l=9.98)
        assert _bullish_marubozu(df).iloc[0] == 1

    def test_bearish_marubozu(self):
        df = _make_ohlc(o=12.0, c=10.0, h=12.02, l=9.98)
        assert _bearish_marubozu(df).iloc[0] == 1


class TestDragonflyGravestone:
    def test_dragonfly(self):
        # Doji with long lower shadow
        df = _make_ohlc(o=12.0, c=12.01, h=12.05, l=8.0)
        assert _dragonfly_doji(df).iloc[0] == 1

    def test_gravestone(self):
        # Doji with long upper shadow
        df = _make_ohlc(o=12.0, c=12.01, h=16.0, l=11.95)
        assert _gravestone_doji(df).iloc[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Two-bar patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestEngulfing:
    def test_bullish_engulfing_detected(self):
        # Bar1: bearish, Bar2: bullish that engulfs bar1
        df = _make_two_bar(
            12, 12.5, 10.5, 11.0,   # bearish bar
            10.8, 13.0, 10.7, 12.8, # bullish engulf
        )
        res = _bullish_engulfing(df)
        assert res.iloc[1] == 1

    def test_bearish_engulfing_detected(self):
        df = _make_two_bar(
            10, 11.5, 9.5, 11.0,    # bullish bar
            11.2, 11.5, 8.5,  9.0,  # bearish engulf
        )
        res = _bearish_engulfing(df)
        assert res.iloc[1] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Full detect_patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectPatterns:
    def test_returns_18_columns(self):
        np.random.seed(42)
        n  = 50
        c  = 100 + np.cumsum(np.random.randn(n))
        df = pd.DataFrame({
            "Open":   c - 0.3,
            "Close":  c,
            "High":   c + 0.5,
            "Low":    c - 0.5,
            "Volume": np.ones(n) * 1e6,
        })
        pats = detect_patterns(df)
        assert pats.shape[1] == 18

    def test_bullish_bearish_lists_non_empty(self):
        assert len(BULLISH_PATTERNS) > 0
        assert len(BEARISH_PATTERNS) > 0

    def test_pattern_values_binary(self):
        np.random.seed(7)
        n  = 100
        c  = 50 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "Open":   c - np.abs(np.random.randn(n)) * 0.3,
            "Close":  c,
            "High":   c + np.abs(np.random.randn(n)) * 0.5,
            "Low":    c - np.abs(np.random.randn(n)) * 0.5,
            "Volume": np.ones(n) * 1e6,
        })
        pats = detect_patterns(df)
        unique = set(pats.values.ravel().tolist())
        assert unique.issubset({0, 1})
