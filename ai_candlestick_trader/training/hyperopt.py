"""
training/hyperopt.py
====================
Optuna-based hyperparameter search for the Transformer and LSTM models.

The objective function:
  - Builds a model from trial-suggested hyperparams
  - Trains for a max of ``max_epochs`` (with early stopping)
  - Returns the best val_MSE (lower = better)

Usage
-----
>>> from ai_candlestick_trader.training.hyperopt import run_hpo
>>> best_cfg = run_hpo(train_ds, val_ds, n_trials=50, model_type="transformer")
>>> print(best_cfg)
"""

from __future__ import annotations

import logging
from typing import Literal

import optuna
import torch
from torch.utils.data import DataLoader

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _build_model(trial: optuna.Trial, n_features: int, model_type: str):
    if model_type == "transformer":
        from ai_candlestick_trader.models.transformer_model import CandlestickTransformer

        d_model    = trial.suggest_categorical("d_model", [64, 128, 256])
        nhead      = trial.suggest_categorical("nhead", [4, 8])
        num_layers = trial.suggest_int("num_layers", 2, 6)
        dim_ff     = trial.suggest_categorical("dim_ff", [256, 512, 1024])
        dropout    = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        return CandlestickTransformer(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )
    else:
        from ai_candlestick_trader.models.lstm_model import CandlestickLSTM

        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout    = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        return CandlestickLSTM(
            n_features=n_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )


def _objective(
    trial:       optuna.Trial,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    n_features:   int,
    model_type:   str,
    max_epochs:   int,
    device:       str,
) -> float:
    from ai_candlestick_trader.training.trainer import Trainer

    model = _build_model(trial, n_features, model_type)

    cfg = {
        "lr":              trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "alpha_loss":      trial.suggest_float("alpha_loss", 0.2, 0.8, step=0.1),
        "grad_clip":       trial.suggest_float("grad_clip", 0.5, 2.0, step=0.5),
        "warmup_epochs":   trial.suggest_int("warmup_epochs", 2, 8),
        "patience":        10,  # shorter patience during HPO
        "save_dir":        f"checkpoints/trial_{trial.number}",
        "checkpoint_name": "trial_best",
        "seed":            trial.suggest_int("seed", 0, 9999),
    }

    trainer = Trainer(model, train_loader, val_loader, cfg=cfg, device=device)
    history = trainer.fit(epochs=max_epochs)

    best_val_mse = min(history["val_mse"]) if history["val_mse"] else float("inf")
    trial.set_user_attr("best_val_mse", best_val_mse)
    return best_val_mse


def run_hpo(
    train_ds,
    val_ds,
    n_trials:    int = 30,
    model_type:  Literal["transformer", "lstm"] = "transformer",
    batch_size:  int = 64,
    max_epochs:  int = 40,
    n_jobs:      int = 1,
    device:      str | None = None,
    study_name:  str = "candlestick_hpo",
) -> dict:
    """
    Run Optuna HPO and return the best hyperparameter config.

    Parameters
    ----------
    train_ds   : OHLCDataset or Subset for training.
    val_ds     : OHLCDataset or Subset for validation.
    n_trials   : Number of Optuna trials.
    model_type : ``"transformer"`` or ``"lstm"``.
    batch_size : Batch size for loaders.
    max_epochs : Max training epochs per trial.
    n_jobs     : Parallel Optuna workers (usually 1 on single GPU).
    device     : ``"cuda"`` / ``"cpu"`` (auto if None).
    study_name : Optuna study name (for resuming).

    Returns
    -------
    dict : Best config (can pass directly to Trainer).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Get n_features from dataset
    sample_x, _ = train_ds[0]
    n_features   = sample_x.shape[-1]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: _objective(trial, train_loader, val_loader, n_features, model_type, max_epochs, device),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    best = study.best_params
    best["best_val_mse"] = study.best_value
    best["model_type"]   = model_type

    print(f"\n{'━'*50}")
    print(f"  HPO complete!  Best val_MSE = {study.best_value:.6f}")
    for k, v in best.items():
        print(f"    {k}: {v}")
    print(f"{'━'*50}\n")

    return best
