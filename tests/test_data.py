"""
tests/test_data.py
==================
Unit tests for the data pipeline components.
"""

import numpy as np
import pandas as pd
import pytest

from ai_candlestick_trader.data.features import build_features
from ai_candlestick_trader.data.patterns import detect_patterns, pattern_summary
from ai_candlestick_trader.data.dataset  import OHLCDataset, split_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_ohlc() -> pd.DataFrame:
    """100-bar synthetic OHLC DataFrame."""
    np.random.seed(0)
    n       = 200
    close   = 100 + np.cumsum(np.random.randn(n) * 0.5)
    idx     = pd.date_range("2023-01-01", periods=n, freq="D")
    noise   = np.abs(np.random.randn(n)) * 0.3
    return pd.DataFrame({
        "Open":   close - noise,
        "Close":  close,
        "High":   close + np.abs(np.random.randn(n)) * 0.7,
        "Low":    close - np.abs(np.random.randn(n)) * 0.7,
        "Volume": np.random.randint(100_000, 5_000_000, n).astype(float),
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Feature tests
# ─────────────────────────────────────────────────────────────────────────────

def test_build_features_shape(synthetic_ohlc):
    feat = build_features(synthetic_ohlc)
    assert feat.shape[0] == len(synthetic_ohlc), "Row count must match OHLC"
    assert feat.shape[1] >= 20, "Should have at least 20 base features"


def test_build_features_no_inf(synthetic_ohlc):
    feat = build_features(synthetic_ohlc)
    assert not np.isinf(feat.values).any(), "Features must not contain Inf"


def test_build_features_with_patterns(synthetic_ohlc):
    pats = detect_patterns(synthetic_ohlc)
    feat = build_features(synthetic_ohlc, pattern_flags=pats)
    assert feat.shape[1] > 20, "Pattern features should be appended"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern tests
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_patterns_shape(synthetic_ohlc):
    pats = detect_patterns(synthetic_ohlc)
    assert pats.shape[0] == len(synthetic_ohlc)
    assert pats.shape[1] == 18, "Should have 18 pattern columns"


def test_detect_patterns_binary(synthetic_ohlc):
    pats = detect_patterns(synthetic_ohlc)
    vals = pats.values
    assert set(np.unique(vals)).issubset({0, 1}), "All values should be 0 or 1"


def test_pattern_summary(synthetic_ohlc):
    pats    = detect_patterns(synthetic_ohlc)
    summary = pattern_summary(pats)
    assert isinstance(summary, dict)
    assert all(v >= 0 for v in summary.values())


# ─────────────────────────────────────────────────────────────────────────────
# Dataset tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ohlc_dataset_len(synthetic_ohlc):
    feat = build_features(synthetic_ohlc)
    ds   = OHLCDataset(feat, synthetic_ohlc["Close"], seq_len=20)
    assert len(ds) > 0


def test_ohlc_dataset_item_shape(synthetic_ohlc):
    feat = build_features(synthetic_ohlc)
    ds   = OHLCDataset(feat, synthetic_ohlc["Close"], seq_len=20)
    x, y = ds[0]
    assert x.shape[0] == 20, "Sequence length must match"
    assert x.ndim == 2
    assert y.ndim == 0, "Target is a scalar"


def test_split_dataset_sizes(synthetic_ohlc):
    feat = build_features(synthetic_ohlc)
    ds   = OHLCDataset(feat, synthetic_ohlc["Close"], seq_len=10)
    tr, val, te = split_dataset(ds, train_ratio=0.70, val_ratio=0.15)
    total = len(tr) + len(val) + len(te)
    assert total == len(ds), "Split must be lossless"
    assert len(tr) > len(val), "Train must be larger than val"


def test_features_normalize_stable(synthetic_ohlc):
    """Feature values should be within a reasonable range after scaling."""
    import torch
    from torch.utils.data import DataLoader

    feat = build_features(synthetic_ohlc)
    ds   = OHLCDataset(feat, synthetic_ohlc["Close"], seq_len=10, scale=True)
    loader = DataLoader(ds, batch_size=16)
    xs = []
    for x, _ in loader:
        xs.append(x)
    all_x = torch.cat(xs)
    assert all_x.abs().max().item() < 50, "Scaled features should be within ±50"
