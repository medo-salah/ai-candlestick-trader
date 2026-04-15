# AI Candlestick Trader 📈

> **Deep-learning candlestick pattern recognition & price prediction  
> for EGX (Egypt) and Tadawul (Saudi Arabia) stocks.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ What It Does

| Feature | Detail |
|---|---|
| **18 candlestick patterns** | Hammer, Doji, Engulfing, Morning Star, Three White Soldiers, and 13 more — pure Python, no TA-Lib |
| **22 engineered features** | Body/shadow ratios, RSI, MACD, ATR, Bollinger Bands, OBV, log-returns, SMA distances |
| **Transformer + LSTM** | Pre-LN Transformer with CLS token; Bidirectional LSTM with attention |
| **Ensemble** | Average K=5 models (inverse-variance weighted) for massive MSE reduction |
| **Optuna HPO** | MedianPruner-guided search over architecture + optimizer hyperparams |
| **Huber + MSE loss** | Robust to outliers; warm-up + CosineAnnealing LR scheduler |
| **Walk-forward backtest** | No data leakage — retrain on rolling windows |
| **Streamlit dashboard** | Dark trading-terminal UI with Plotly charts, pattern overlays, equity curve |
| **EGX + Tadawul coverage** | Built-in ticker maps for top 10 stocks from each market |

---

## 🚀 Quick Start

### 1. Install

```bash
# Clone
git clone https://github.com/yourname/ai-candlestick-trader.git
cd ai-candlestick-trader

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Install
pip install -r requirements.txt
# -- or for development --
pip install -e ".[dev]"
```

### 2. Train a model

```bash
# Train Transformer on Commercial International Bank (EGX)
python train.py --ticker COMI.CA --period 3y --epochs 100

# Train with Optuna HPO + 3-member ensemble
python train.py --ticker 2222.SR --model transformer --hpo --hpo-trials 30 --ensemble-k 3

# All options
python train.py --help
```

### 3. Launch the Dashboard

```bash
streamlit run dashboard.py
# -- or --
act-dash          # if installed via pip install -e .
```

### 4. Use in code

```python
from ai_candlestick_trader import (
    download_ohlc,
    build_features,
    detect_patterns,
    CandlestickTransformer,
    evaluate_predictions,
)

# Download 3 years of daily bars for Saudi Aramco
df    = download_ohlc("2222.SR", period="3y")

# Detect 18 candlestick patterns
pats  = detect_patterns(df)
print(pats.sum())                     # count per pattern

# Build 22 technical features
feat  = build_features(df, pattern_flags=pats)
print(feat.shape)                     # (n_bars, 40)

# Build dataset and train
from ai_candlestick_trader.data.dataset  import OHLCDataset, split_dataset
from ai_candlestick_trader.training      import Trainer
from torch.utils.data import DataLoader

ds              = OHLCDataset(feat, df["Close"], seq_len=30)
train_ds, val_ds, test_ds = split_dataset(ds)

model    = CandlestickTransformer(n_features=ds.n_features)
trainer  = Trainer(model,
                   DataLoader(train_ds, batch_size=64, shuffle=True),
                   DataLoader(val_ds,   batch_size=64),
                   cfg={"lr": 3e-4, "epochs": 100})
history  = trainer.fit(epochs=100)
```

---

## 📁 Package Structure

```
ai_candlestick_trader_package/
├── pyproject.toml                   ← pip-installable package config
├── requirements.txt
├── train.py                         ← python train.py --ticker COMI.CA
├── dashboard.py                     ← streamlit run dashboard.py
│
├── ai_candlestick_trader/
│   ├── __init__.py                  ← version 2.0.0 + convenience imports
│   ├── cli.py                       ← act-train / act-dash entry points
│   │
│   ├── data/
│   │   ├── downloader.py            ← yfinance + EGX (.CA) / Tadawul (.SR) maps
│   │   ├── features.py              ← 22 OHLC features (RSI, MACD, ATR, …)
│   │   ├── patterns.py              ← 18 pattern detectors (pure Python)
│   │   └── dataset.py               ← PyTorch sliding-window Dataset
│   │
│   ├── models/
│   │   ├── transformer_model.py     ← Pre-LN Transformer + CLS token
│   │   ├── lstm_model.py            ← Bi-LSTM + scaled-dot attention
│   │   └── ensemble.py              ← Inverse-variance weighted ensemble
│   │
│   ├── training/
│   │   ├── trainer.py               ← Huber+MSE, warm-up CosineAnnealing, EarlyStopping
│   │   ├── callbacks.py             ← EarlyStopping, ModelCheckpoint, LRMonitor
│   │   └── hyperopt.py              ← Optuna HPO with MedianPruner
│   │
│   ├── evaluation/
│   │   ├── metrics.py               ← MSE/RMSE/MAE/Sharpe/Calmar/WinRate
│   │   └── backtester.py            ← Walk-forward backtester + trade log
│   │
│   └── dashboard/
│       ├── app.py                   ← Streamlit dark-terminal dashboard
│       └── charts.py                ← Plotly builders (candle, equity, scatter…)
│
├── tests/
│   ├── test_data.py                 ← Feature + pattern + dataset tests
│   ├── test_model.py                ← LSTM / Transformer / Ensemble / Metrics
│   └── test_patterns.py             ← Individual pattern detector tests
│
└── legacy/                          ← Original image-based code (preserved)
    ├── model.py
    ├── dataset.py
    └── …
```

---

## 🎯 Roadmap to MSE → 0

| Phase | Action | Status |
|---|---|---|
| ✅ | OHLC data pipeline (yfinance, EGX, Tadawul) | Done |
| ✅ | 22 features + 18 candlestick patterns | Done |
| ✅ | Transformer + LSTM models with attention | Done |
| ✅ | Huber+MSE loss + warm-up + CosineAnnealing | Done |
| ✅ | EarlyStopping + ModelCheckpoint | Done |
| ✅ | Optuna HPO (MedianPruner) | Done |
| ✅ | K-member Ensemble (inverse-variance weighting) | Done |
| ✅ | Walk-forward backtesting | Done |
| ✅ | Streamlit dashboard + Plotly charts | Done |
| ✅ | Unit tests (pytest) | Done |
| ⏳ | Multi-timeframe features (5m+1H+daily) | Next |
| ⏳ | Arabic news sentiment integration | Next |
| ⏳ | PDF report export (Arabic + English) | Next |
| ⏳ | Telegram bot for daily signals | Next |
| ⏳ | PyPI publish | Next |

---

## 🏃 Run Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## ⚠️ Disclaimer

This software is for **research and educational purposes only**.  
It does not constitute financial advice.  
Past performance is not indicative of future results.  
Always consult a licensed financial advisor before trading.

**المحتوى للأغراض البحثية والتعليمية فقط.  
ليس نصيحة مالية.**

---

## 📄 License

MIT — see [LICENSE](LICENSE).
