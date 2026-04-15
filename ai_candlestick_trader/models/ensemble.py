"""
models/ensemble.py
==================
Ensemble wrapper that averages predictions from multiple trained models.

Strategy
--------
- Train K=5 models on *different random seeds* (same architecture or mixed)
- Average their scalar predictions  → strong variance reduction
- Track individual model MSE to compute inverse-variance weights

Usage
-----
>>> from ai_candlestick_trader.models.ensemble import EnsembleModel
>>> ens = EnsembleModel(models=[model1, model2, model3])
>>> preds = ens.predict(x_batch)           # averaged
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ai_candlestick_trader.exceptions import ModelNotLoadedError


class EnsembleModel(nn.Module):
    """
    Simple weighted-average ensemble of regression models.

    Parameters
    ----------
    models  : List of trained nn.Module instances.
    weights : Optional list of floats (e.g. 1/MSE for each model).
              If None, simple uniform average is used.
    """

    def __init__(
        self,
        models:  List[nn.Module],
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(models)

        if weights is not None:
            assert len(weights) == len(models)
            w = torch.tensor(weights, dtype=torch.float32)
            w = w / w.sum()
        else:
            w = torch.ones(len(models)) / len(models)

        self.register_buffer("weights", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, F) input tensor

        Returns
        -------
        (B,) weighted-average prediction.
        """
        preds = torch.stack(
            [m(x) for m in self.members], dim=1
        )  # (B, K)
        return (preds * self.weights.unsqueeze(0)).sum(dim=1)   # (B,)

    def predict(self, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Convenience method (no_grad + device handling)."""
        self.eval()
        x = x.to(device)
        with torch.no_grad():
            return self.forward(x).cpu()

    @classmethod
    def from_checkpoints(
        cls,
        paths:      List[str],
        model_cls,
        model_kwargs: dict,
        val_mses:   Optional[List[float]] = None,
        device:     str = "cpu",
    ) -> "EnsembleModel":
        """
        Load K checkpoints and build an ensemble.

        Parameters
        ----------
        paths         : List of .pt checkpoint paths.
        model_cls     : Model class (e.g. CandlestickTransformer).
        model_kwargs  : kwargs for model_cls constructor.
        val_mses      : Validation MSEs per model (for inv-variance weighting).
        device        : Device string.
        """
        models = []
        for path in paths:
            m = model_cls(**model_kwargs)
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
            except Exception as e:
                raise ModelNotLoadedError(f"Failed to load checkpoint {path}: {e}")
            state = ckpt.get("model_state_dict", ckpt)
            m.load_state_dict(state)
            m.to(device)
            m.eval()
            models.append(m)

        weights = None
        if val_mses is not None:
            # inverse-variance weighting
            weights = [1.0 / (mse + 1e-9) for mse in val_mses]

        return cls(models=models, weights=weights)
