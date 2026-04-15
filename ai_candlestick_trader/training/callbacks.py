"""
training/callbacks.py
=====================
Training callbacks: EarlyStopping, ModelCheckpoint, LRMonitor.
Designed to be used inside the Trainer loop.
"""

from __future__ import annotations

import os
from typing import Optional


class EarlyStopping:
    """
    Stop training when validation metric stops improving.

    Parameters
    ----------
    patience  : Number of epochs with no improvement before stopping.
    min_delta : Minimum change to count as improvement.
    mode      : ``"min"`` (lower is better, e.g. MSE) or ``"max"``.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-6, mode: str = "min") -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.counter   = 0
        self.stop      = False

    def __call__(self, metric: float) -> bool:
        improved = (
            (metric < self.best - self.min_delta) if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class ModelCheckpoint:
    """
    Save the model whenever the monitored metric improves.

    Parameters
    ----------
    save_dir  : Directory to write checkpoint files.
    filename  : Base filename (without extension).
    mode      : ``"min"`` or ``"max"``.
    verbose   : Print a message each time the model is saved.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        filename: str = "best_model",
        mode:     str = "min",
        verbose:  bool = True,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.path    = os.path.join(save_dir, f"{filename}.pt")
        self.mode    = mode
        self.verbose = verbose
        self.best    = float("inf") if mode == "min" else float("-inf")

    def __call__(self, metric: float, obj) -> bool:
        """
        Parameters
        ----------
        metric : Current epoch metric value.
        obj    : Object to save (``torch.save`` compatible, e.g. state_dict dict).

        Returns True if checkpoint was saved.
        """
        import torch

        improved = (
            (metric < self.best) if self.mode == "min" else (metric > self.best)
        )
        if improved:
            self.best = metric
            torch.save(obj, self.path)
            if self.verbose:
                print(f"  ✔ Checkpoint saved  →  {self.path}  (metric={metric:.6f})")
            return True
        return False


class LRMonitor:
    """Log current learning rate(s) after each epoch."""

    def __call__(self, optimizer) -> list[float]:
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        return lrs
