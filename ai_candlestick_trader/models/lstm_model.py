"""
models/lstm_model.py
====================
Stacked Bidirectional LSTM with attention mechanism for candlestick
price-movement regression.

Architecture
------------
Input  (batch, seq_len, n_features)
  │
  ▼
Linear projection  →  hidden_dim
  │
  ▼
Stacked Bi-LSTM  (num_layers, dropout between layers)
  │
  ▼
Multi-head Self-Attention  (over time axis)
  │
  ▼
LayerNorm + Residual
  │
  ▼
MLP regression head  →  scalar output  (next-bar return)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ScaledDotAttention(nn.Module):
    """Single-head scaled dot-product attention over time steps."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale   # (B, T, T)
        attn   = torch.softmax(scores, dim=-1)
        return torch.bmm(attn, V)                                # (B, T, H)


class CandlestickLSTM(nn.Module):
    """
    Bidirectional LSTM + attention for candlestick regression.

    Parameters
    ----------
    n_features  : Number of input features per time step.
    hidden_dim  : LSTM hidden dimension (and projection size).
    num_layers  : Number of stacked LSTM layers.
    dropout     : Dropout probability (inter-layer and final MLP).
    seq_len     : Input sequence length (only needed for shape assertions).
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout:    float = 0.3,
        seq_len:    int = 30,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Bi-LSTM → hidden_dim via linear (fuse forward + backward)
        self.fuse   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm1  = nn.LayerNorm(hidden_dim)

        self.attn   = _ScaledDotAttention(hidden_dim)
        self.norm2  = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, F) float tensor

        Returns
        -------
        (B,) float tensor — predicted next-bar relative return.
        """
        h = self.input_proj(x)               # (B, T, H)

        lstm_out, _ = self.lstm(h)           # (B, T, 2H)
        fused = self.fuse(lstm_out)          # (B, T, H)
        fused = self.norm1(fused + h)        # residual

        attended = self.attn(fused)          # (B, T, H)
        attended = self.norm2(attended + fused)

        ctx = attended[:, -1, :]             # last time step  (B, H)
        return self.head(ctx).squeeze(-1)    # (B,)


def build_lstm(n_features: int, cfg: dict | None = None) -> CandlestickLSTM:
    """Factory using a flat config dict (from Optuna or YAML)."""
    cfg = cfg or {}
    return CandlestickLSTM(
        n_features=n_features,
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout", 0.3),
        seq_len=cfg.get("seq_len", 30),
    )
