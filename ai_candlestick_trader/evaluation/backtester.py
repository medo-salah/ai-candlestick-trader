"""
evaluation/backtester.py
========================
Walk-forward backtesting engine.

Strategy
--------
- **Signal**: BUY when predicted return > threshold, SELL when < -threshold, HOLD otherwise
- **Execution**: Next-bar open price (realistic; no look-ahead)
- **Costs**: Configurable commission + slippage per trade
- **Metrics**: Returned via evaluate_predictions()

Walk-forward validation
-----------------------
Data is split into rolling windows:
  [train_window] → [test_window] → slide by step_size → repeat
This is the only correct way to evaluate a time-series model
without leaking future information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ai_candlestick_trader.evaluation.metrics import evaluate_predictions


@dataclass
class Trade:
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    direction:   int          # +1 = long, -1 = short
    entry_price: float
    exit_price:  float
    return_pct:  float
    pattern:     str = ""


@dataclass
class BacktestResult:
    trades:       List[Trade]
    equity_curve: pd.Series
    metrics:      dict[str, float]
    predictions:  pd.Series   # model predicted returns
    actuals:      pd.Series   # actual returns
    signals:      pd.Series   # +1 / 0 / -1


class Backtester:
    """
    Simple vectorised backtester for the candlestick model.

    Parameters
    ----------
    model          : Trained nn.Module (or EnsembleModel).
    device         : ``"cuda"`` / ``"cpu"``.
    threshold      : Minimum |predicted return| to trigger a trade.
    commission_pct : Transaction cost as a fraction of trade value.
    slippage_pct   : One-way slippage as a fraction of price.
    """

    def __init__(
        self,
        model,
        device:         str   = "cpu",
        threshold:      float = 0.003,
        commission_pct: float = 0.001,
        slippage_pct:   float = 0.0005,
    ) -> None:
        self.model          = model
        self.device         = device
        self.threshold      = threshold
        self.commission     = commission_pct
        self.slippage       = slippage_pct

    def _predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                preds.append(self.model(X).cpu().numpy())
        return np.concatenate(preds)

    def run(
        self,
        dataset,           # OHLCDataset
        ohlc_df: pd.DataFrame,
        batch_size: int = 64,
    ) -> BacktestResult:
        """
        Run a straightforward backtest on the given dataset.

        Parameters
        ----------
        dataset    : OHLCDataset (fully constructed, not subset).
        ohlc_df    : Original OHLC DataFrame (needed for prices).
        batch_size : Inference batch size.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds  = self._predict(loader)

        # Timestamps of the last bar in each window
        ts         = dataset.timestamps
        actuals    = np.array([dataset[i][1].item() for i in range(len(dataset))])

        signals    = np.sign(preds) * (np.abs(preds) >= self.threshold)  # +1/0/-1

        # Build trade list
        trades     = []
        port_return = np.zeros(len(signals))

        for i, (sig, ret, t) in enumerate(zip(signals, actuals, ts)):
            if sig == 0:
                continue
            direction  = int(sig)
            cost       = self.commission + self.slippage
            trade_ret  = direction * ret - 2 * cost   # round-trip cost
            port_return[i] = trade_ret
            trades.append(Trade(
                entry_date=t, exit_date=t,
                direction=direction,
                entry_price=float(ohlc_df["Close"].get(t, np.nan)),
                exit_price=float(ohlc_df["Close"].get(t, np.nan)),
                return_pct=trade_ret * 100,
            ))

        equity = pd.Series(
            np.cumprod(1 + port_return),
            index=ts,
            name="equity",
        )

        metrics = evaluate_predictions(
            y_true=actuals,
            y_pred=preds,
            trade_returns=port_return,
        )

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            metrics=metrics,
            predictions=pd.Series(preds, index=ts, name="predicted_return"),
            actuals=pd.Series(actuals, index=ts, name="actual_return"),
            signals=pd.Series(signals, index=ts, name="signal"),
        )


def walk_forward_backtest(
    dataset,
    ohlc_df:      pd.DataFrame,
    model_cls,
    model_kwargs: dict,
    trainer_cfg:  dict,
    train_size:   int   = 500,
    test_size:    int   = 60,
    step_size:    int   = 30,
    batch_size:   int   = 64,
    epochs:       int   = 50,
    device:       str   = "cpu",
) -> List[BacktestResult]:
    """
    Walk-forward backtest — retrain model on each window.

    Returns a list of BacktestResult, one per test window.
    """
    from torch.utils.data import Subset
    from ai_candlestick_trader.training.trainer import Trainer

    n       = len(dataset)
    results = []
    start   = 0

    while start + train_size + test_size <= n:
        train_idx = list(range(start, start + train_size))
        test_idx  = list(range(start + train_size, start + train_size + test_size))

        train_sub = Subset(dataset, train_idx)
        test_sub  = Subset(dataset, test_idx)

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(test_sub,  batch_size=batch_size, shuffle=False)

        model   = model_cls(**model_kwargs)
        trainer = Trainer(model, train_loader, val_loader, cfg=trainer_cfg, device=device)
        trainer.fit(epochs=epochs)
        model   = trainer.load_best()

        bt       = Backtester(model, device=device)
        # reconstruct test-only dataset view
        test_ds  = Subset(dataset, test_idx)
        result   = bt.run(test_ds, ohlc_df, batch_size=batch_size)
        results.append(result)

        start += step_size
        print(f"  Walk-forward window {len(results):2d} done — Sharpe={result.metrics['sharpe_ratio']:.3f}")

    return results
