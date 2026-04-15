"""
cli.py
======
Command-line entry points registered in pyproject.toml:

    act-train   →  train a model on any ticker
    act-dash    →  launch the Streamlit dashboard
"""

from __future__ import annotations

import argparse
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
# act-train
# ─────────────────────────────────────────────────────────────────────────────

def train_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="act-train",
        description="Train the AI Candlestick Trader model on a given ticker.",
    )
    parser.add_argument("--ticker",      default="COMI.CA",       help="Yahoo Finance ticker (e.g. COMI.CA, 2222.SR)")
    parser.add_argument("--period",      default="3y",            help="History period (1y, 2y, 3y, 5y)")
    parser.add_argument("--interval",    default="1d",            help="Bar interval (1d, 1h, 5m)")
    parser.add_argument("--seq-len",     type=int,   default=30,  help="Look-back window (bars)")
    parser.add_argument("--epochs",      type=int,   default=100, help="Max training epochs")
    parser.add_argument("--batch-size",  type=int,   default=64,  help="Batch size")
    parser.add_argument("--lr",          type=float, default=3e-4,help="Initial learning rate")
    parser.add_argument("--model",       default="transformer",   choices=["transformer", "lstm"])
    parser.add_argument("--save-dir",    default="checkpoints",   help="Checkpoint directory")
    parser.add_argument("--hpo",         action="store_true",     help="Run Optuna HPO before training")
    parser.add_argument("--hpo-trials",  type=int,   default=30,  help="Number of Optuna trials")
    parser.add_argument("--ensemble-k",  type=int,   default=1,   help="Number of ensemble members (seeds)")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from ai_candlestick_trader.data.downloader import download_ohlc
    from ai_candlestick_trader.data.features   import build_features
    from ai_candlestick_trader.data.patterns   import detect_patterns
    from ai_candlestick_trader.data.dataset    import OHLCDataset, split_dataset
    from ai_candlestick_trader.models.transformer_model import CandlestickTransformer, build_transformer
    from ai_candlestick_trader.models.lstm_model        import CandlestickLSTM, build_lstm
    from ai_candlestick_trader.training.trainer         import Trainer
    from ai_candlestick_trader.evaluation.metrics       import evaluate_predictions, format_metrics_table

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'━'*60}")
    print(f"  AI Candlestick Trainer  ·  device={device.upper()}")
    print(f"  Ticker: {args.ticker}  │  Period: {args.period}  │  Interval: {args.interval}")
    print(f"{'━'*60}\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("📥 Downloading OHLC data …")
    ohlc_df = download_ohlc(args.ticker, period=args.period, interval=args.interval)
    print(f"   → {len(ohlc_df)} bars loaded\n")

    print("🔬 Building features …")
    pat_flags = detect_patterns(ohlc_df)
    feat_df   = build_features(ohlc_df, pattern_flags=pat_flags)

    ds                     = OHLCDataset(feat_df, ohlc_df["Close"], seq_len=args.seq_len)
    train_ds, val_ds, test_ds = split_dataset(ds, train_ratio=0.70, val_ratio=0.15)
    n_features = ds.n_features
    print(f"   → Features: {n_features}  │  Train: {len(train_ds)}  │  Val: {len(val_ds)}  │  Test: {len(test_ds)}\n")

    # ── 2. Optional HPO ───────────────────────────────────────────────────────
    best_cfg: dict = {}
    if args.hpo:
        from ai_candlestick_trader.training.hyperopt import run_hpo
        print("🔎 Running Optuna HPO …")
        best_cfg = run_hpo(
            train_ds=train_ds, val_ds=val_ds,
            n_trials=args.hpo_trials, model_type=args.model,
            batch_size=args.batch_size, max_epochs=40, device=device,
        )

    # ── 3. Training (possibly ensemble) ───────────────────────────────────────
    val_mses = []
    ckpt_paths = []

    for seed in range(args.ensemble_k):
        print(f"🚀 Training member {seed+1}/{args.ensemble_k}  (seed={seed}) …\n")
        model_kwargs = {**best_cfg, "n_features": n_features}
        model_kwargs.pop("best_val_mse", None)
        model_kwargs.pop("model_type", None)

        if args.model == "transformer":
            model = build_transformer(n_features, cfg=model_kwargs)
        else:
            model = build_lstm(n_features, cfg=model_kwargs)

        cfg = {
            "lr":              best_cfg.get("lr", args.lr),
            "weight_decay":    best_cfg.get("weight_decay", 1e-4),
            "alpha_loss":      best_cfg.get("alpha_loss", 0.5),
            "grad_clip":       1.0,
            "warmup_epochs":   best_cfg.get("warmup_epochs", 5),
            "patience":        15,
            "save_dir":        args.save_dir,
            "checkpoint_name": f"best_model_seed{seed}",
            "seed":            seed,
        }
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

        trainer  = Trainer(model, train_loader, val_loader, cfg=cfg, device=device)
        history  = trainer.fit(epochs=args.epochs)
        best_mse = min(history["val_mse"])
        val_mses.append(best_mse)
        ckpt_paths.append(os.path.join(args.save_dir, f"best_model_seed{seed}.pt"))

    # Also save canonical best (seed 0) as best_model.pt
    import shutil
    shutil.copy(ckpt_paths[0], os.path.join(args.save_dir, "best_model.pt"))

    # ── 4. Evaluation on test set ─────────────────────────────────────────────
    print("\n📊 Evaluating on hold-out test set …")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.ensemble_k > 1:
        from ai_candlestick_trader.models.ensemble import EnsembleModel
        if args.model == "transformer":
            mcls   = CandlestickTransformer
            mkwargs = {"n_features": n_features}
        else:
            mcls   = CandlestickLSTM
            mkwargs = {"n_features": n_features}
        model = EnsembleModel.from_checkpoints(ckpt_paths, mcls, mkwargs, val_mses=val_mses, device=device)
    else:
        model = trainer.load_best()

    preds_all, targets_all = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            preds_all.append(model(X.to(device)).cpu().numpy())
            targets_all.append(y.numpy())

    import numpy as np
    preds   = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    metrics = evaluate_predictions(targets, preds)

    print(format_metrics_table(metrics).to_string(index=False))
    print(f"\n✅ Done!  Checkpoints saved to: {os.path.abspath(args.save_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# act-dash
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_cli() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess

    dashboard_path = os.path.join(
        os.path.dirname(__file__), "dashboard", "app.py"
    )
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.headless", "false"]
    print(f"▶ Launching Streamlit dashboard …\n  {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dash":
        dashboard_cli()
    else:
        train_cli()
