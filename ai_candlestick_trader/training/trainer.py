"""
training/trainer.py
===================
Production training loop with:
- Combined Huber + MSE loss (Huber is robust to outliers)
- CosineAnnealing LR scheduler with linear warm-up
- EarlyStopping + ModelCheckpoint callbacks
- Gradient clipping
- Epoch-level progress reporting
- Multi-seed training support (for ensemble)

Usage
-----
>>> from ai_candlestick_trader.training.trainer import Trainer
>>> trainer = Trainer(model, train_loader, val_loader, cfg)
>>> history = trainer.fit(epochs=100)
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ai_candlestick_trader.training.callbacks import (
    EarlyStopping,
    LRMonitor,
    ModelCheckpoint,
)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """Alpha * Huber + (1-Alpha) * MSE.  Best of both worlds."""

    def __init__(self, alpha: float = 0.5, delta: float = 1.0) -> None:
        super().__init__()
        self.alpha   = alpha
        self.huber   = nn.HuberLoss(delta=delta, reduction="mean")
        self.mse     = nn.MSELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.huber(pred, target) + (1 - self.alpha) * self.mse(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# Warm-up + CosineAnnealing scheduler
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warm-up then cosine annealing to 0."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the full training + validation loop.

    Parameters
    ----------
    model        : nn.Module (LSTM or Transformer).
    train_loader : DataLoader for training set.
    val_loader   : DataLoader for validation set.
    cfg          : Config dict with keys:
                    lr, weight_decay, grad_clip, alpha_loss, delta_huber,
                    warmup_epochs, patience, save_dir, checkpoint_name, seed.
    device       : ``"cuda"`` or ``"cpu"``.
    """

    DEFAULT_CFG: dict = {
        "lr":               3e-4,
        "weight_decay":     1e-4,
        "grad_clip":        1.0,
        "alpha_loss":       0.5,
        "delta_huber":      1.0,
        "warmup_epochs":    5,
        "patience":         15,
        "save_dir":         "checkpoints",
        "checkpoint_name":  "best_model",
        "seed":             42,
    }

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          Optional[dict] = None,
        device:       Optional[str] = None,
    ) -> None:
        self.cfg = {**self.DEFAULT_CFG, **(cfg or {})}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(self.cfg["seed"])

        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.criterion = CombinedLoss(
            alpha=self.cfg["alpha_loss"],
            delta=self.cfg["delta_huber"],
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )

        self._early_stop = EarlyStopping(patience=self.cfg["patience"], mode="min")
        self._ckpt       = ModelCheckpoint(
            save_dir=self.cfg["save_dir"],
            filename=self.cfg["checkpoint_name"],
            mode="min",
        )
        self._lr_mon = LRMonitor()

        # scheduler set up after knowing total steps
        self._scheduler = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_scheduler(self, epochs: int) -> None:
        total  = epochs * len(self.train_loader)
        warmup = self.cfg["warmup_epochs"] * len(self.train_loader)
        self._scheduler = _warmup_cosine_schedule(self.optimizer, warmup, total)

    def _train_epoch(self) -> float:
        self.model.train()
        running, n = 0.0, 0
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.optimizer.step()
            if self._scheduler:
                self._scheduler.step()
            running += loss.item() * X.size(0)
            n       += X.size(0)
        return running / n if n else 0.0

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        """Returns (combined_loss, mse_only) for the validation set."""
        self.model.eval()
        loss_sum, mse_sum, n = 0.0, 0.0, 0
        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            pred      = self.model(X)
            loss_sum += self.criterion(pred, y).item() * X.size(0)
            mse_sum  += ((pred - y) ** 2).mean().item() * X.size(0)
            n        += X.size(0)
        return (loss_sum / n if n else 0.0), (mse_sum / n if n else 0.0)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, epochs: int = 100) -> dict[str, list]:
        """
        Run the full training loop.

        Returns
        -------
        dict with keys ``train_loss``, ``val_loss``, ``val_mse``, ``lr``.
        """
        self._init_scheduler(epochs)
        history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_mse": [], "lr": []}

        print(f"\n{'─'*60}")
        print(f"  Training on {self.device.upper()}  │  {epochs} epochs")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'─'*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            tr_loss            = self._train_epoch()
            val_loss, val_mse  = self._val_epoch()
            lrs                = self._lr_mon(self.optimizer)
            elapsed            = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_mse"].append(val_mse)
            history["lr"].append(lrs[0])

            print(
                f"  Epoch {epoch:4d}/{epochs}"
                f"  │  train_loss={tr_loss:.5f}"
                f"  │  val_loss={val_loss:.5f}"
                f"  │  val_mse={val_mse:.6f}"
                f"  │  lr={lrs[0]:.2e}"
                f"  │  {elapsed:.1f}s"
            )

            # Checkpoint
            self._ckpt(val_mse, {
                "epoch":             epoch,
                "model_state_dict":  self.model.state_dict(),
                "optimizer_state":   self.optimizer.state_dict(),
                "val_mse":           val_mse,
                "cfg":               self.cfg,
            })

            # Early stopping
            if self._early_stop(val_mse):
                print(f"\n  ⏹ Early stopping at epoch {epoch}  (best val_mse={self._early_stop.best:.6f})")
                break

        print(f"\n{'─'*60}")
        print(f"  Training complete.  Best val_MSE = {self._ckpt.best:.6f}")
        print(f"  Checkpoint → {self._ckpt.path}")
        print(f"{'─'*60}\n")

        return history

    def load_best(self) -> nn.Module:
        """Load best checkpoint weights back into self.model."""
        ckpt = torch.load(self._ckpt.path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        return self.model
