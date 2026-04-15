"""
models/transformer_model.py
============================
Temporal Fusion-style Transformer for candlestick regression.

Architecture
------------
Input  (batch, seq_len, n_features)
  │
  ▼
Linear input embedding  →  d_model
  │
  ▼
Sinusoidal positional encoding
  │
  ▼
N × TransformerEncoder layers  (multi-head self-attention + FFN + residuals)
  │
  ▼
[CLS] token global representation
  │
  ▼
MLP regression head  →  scalar  (next-bar relative return)

Why Transformer > plain LSTM?
------------------------------
- Attends to *all* past bars simultaneously (global receptive field)
- Captures long-range dependencies (weekly patterns with daily data)
- Easily parallelised on GPU
- Superior performance for sequence regression tasks
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class _SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CandlestickTransformer(nn.Module):
    """
    Transformer encoder for candlestick time-series regression.

    Parameters
    ----------
    n_features  : Number of input features per time step.
    d_model     : Transformer model dimension (embedding size).
    nhead       : Number of attention heads (must divide d_model evenly).
    num_layers  : Number of TransformerEncoder layers.
    dim_ff      : Feed-forward inner dimension.
    dropout     : Dropout probability.
    max_seq_len : Maximum sequence length (for positional encoding buffer).
    """

    def __init__(
        self,
        n_features:  int,
        d_model:     int   = 128,
        nhead:       int   = 8,
        num_layers:  int   = 4,
        dim_ff:      int   = 512,
        dropout:     float = 0.2,
        max_seq_len: int   = 512,
    ) -> None:
        super().__init__()

        # ── Ensure d_model divisible by nhead ─────────────────────────────────
        if d_model % nhead != 0:
            nhead = max(h for h in [8, 4, 2, 1] if d_model % h == 0)

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable [CLS] token prepended to each sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = _SinusoidalPE(d_model, max_len=max_seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
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
        B = x.size(0)
        h = self.input_proj(x)                           # (B, T, d_model)

        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_model)
        h   = torch.cat([cls, h], dim=1)                 # (B, T+1, d_model)
        h   = self.pos_enc(h)

        enc = self.encoder(h)                            # (B, T+1, d_model)
        cls_repr = enc[:, 0]                             # (B, d_model) — [CLS]

        return self.head(cls_repr).squeeze(-1)           # (B,)


def build_transformer(n_features: int, cfg: dict | None = None) -> CandlestickTransformer:
    """Factory using a flat config dict (from Optuna or YAML)."""
    cfg = cfg or {}
    return CandlestickTransformer(
        n_features=n_features,
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("num_layers", 4),
        dim_ff=cfg.get("dim_ff", 512),
        dropout=cfg.get("dropout", 0.2),
    )
