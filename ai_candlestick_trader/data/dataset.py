"""
data/dataset.py
===============
PyTorch Dataset that converts the feature matrix + OHLC prices into
sliding-window sequences for sequence-to-point regression.

Target
------
The model predicts the *next-bar close* (t+1) relative change:
    target = (Close[t+1] - Close[t]) / Close[t]   ∈ ℝ

This is a unit-less, scale-invariant regression target.

Usage example
-------------
>>> from ai_candlestick_trader.data.downloader import download_ohlc
>>> from ai_candlestick_trader.data.features  import build_features
>>> from ai_candlestick_trader.data.patterns  import detect_patterns
>>> from ai_candlestick_trader.data.dataset   import OHLCDataset
>>>
>>> df   = download_ohlc("COMI.CA", period="3y")
>>> pats = detect_patterns(df)
>>> feat = build_features(df, pattern_flags=pats)
>>> ds   = OHLCDataset(feat, df["Close"], seq_len=30)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class OHLCDataset(Dataset):
    """
    Sliding-window time-series dataset for candlestick regression.

    Parameters
    ----------
    features   : pd.DataFrame — feature matrix (output of ``build_features``).
    close      : pd.Series   — raw close prices (same index as *features*).
    seq_len    : int         — look-back window length (bars).
    target_horizon : int     — bars ahead to predict (default 1 = next bar).
    scale      : bool        — apply StandardScaler to features (recommended).
    scaler     : fitted StandardScaler or None (fitted if *scale* is True and
                              this is None).
    """

    def __init__(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        seq_len: int = 30,
        target_horizon: int = 1,
        scale: bool = True,
        scaler: StandardScaler | None = None,
    ) -> None:
        self.seq_len         = seq_len
        self.target_horizon  = target_horizon

        # ── Align, drop NaN rows ──────────────────────────────────────────────
        combined = features.copy()
        combined["_close"] = close
        combined.dropna(inplace=True)
        combined.sort_index(inplace=True)

        close_vals = combined.pop("_close").values.astype(np.float32)
        feat_vals  = combined.values.astype(np.float32)

        # ── Scale ─────────────────────────────────────────────────────────────
        if scale:
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(feat_vals)
            feat_vals = scaler.transform(feat_vals).astype(np.float32)
        self.scaler = scaler

        self.X      = feat_vals    # (T, F)
        self.close  = close_vals   # (T,)
        self.index  = combined.index

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        # last valid i: i + seq_len + target_horizon - 1 < len
        return max(0, len(self.X) - self.seq_len - self.target_horizon + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq  = self.X[idx : idx + self.seq_len]                   # (seq_len, F)
        c_now  = self.close[idx + self.seq_len - 1]
        c_next = self.close[idx + self.seq_len - 1 + self.target_horizon]
        # relative return as target
        target = (c_next - c_now) / (c_now + 1e-9)
        return (
            torch.tensor(x_seq,  dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Timestamp of the *last bar* in each window (useful for plotting)."""
        return self.index[self.seq_len - 1 : len(self.X) - self.target_horizon]


def split_dataset(
    ds: OHLCDataset,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> Tuple["OHLCDataset", "OHLCDataset", "OHLCDataset"]:
    """
    Time-aware split — NO random shuffling (avoids data leakage).

    Returns (train_ds, val_ds, test_ds) as sub-views sharing the scaler.
    """
    n       = len(ds)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    from torch.utils.data import Subset
    train_ds = Subset(ds, range(0, n_train))
    val_ds   = Subset(ds, range(n_train, n_train + n_val))
    test_ds  = Subset(ds, range(n_train + n_val, n_train + n_val + n_test))
    return train_ds, val_ds, test_ds
