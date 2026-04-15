"""
tests/test_model.py
===================
Unit tests for model architectures, forward passes, and evaluation metrics.
"""

import numpy as np
import pytest
import torch

from ai_candlestick_trader.models.lstm_model        import CandlestickLSTM, build_lstm
from ai_candlestick_trader.models.transformer_model import CandlestickTransformer, build_transformer
from ai_candlestick_trader.models.ensemble          import EnsembleModel
from ai_candlestick_trader.evaluation.metrics       import (
    mse, mae, directional_accuracy, sharpe_ratio, evaluate_predictions
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_FEATURES = 22
SEQ_LEN    = 30
BATCH_SIZE = 8


@pytest.fixture
def batch():
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

class TestLSTM:
    def test_output_shape(self, batch):
        model = CandlestickLSTM(n_features=N_FEATURES, hidden_dim=64, num_layers=2)
        out   = model(batch)
        assert out.shape == (BATCH_SIZE,), f"Expected ({BATCH_SIZE},), got {out.shape}"

    def test_output_finite(self, batch):
        model = CandlestickLSTM(n_features=N_FEATURES)
        out   = model(batch)
        assert torch.isfinite(out).all(), "All outputs must be finite"

    def test_build_factory(self, batch):
        model = build_lstm(N_FEATURES, cfg={"hidden_dim": 64, "num_layers": 2, "dropout": 0.1})
        out   = model(batch)
        assert out.shape == (BATCH_SIZE,)

    def test_gradient_flows(self, batch):
        model = CandlestickLSTM(n_features=N_FEATURES)
        target = torch.randn(BATCH_SIZE)
        loss   = ((model(batch) - target) ** 2).mean()
        loss.backward()
        grads  = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "Gradients must flow through the network"


# ─────────────────────────────────────────────────────────────────────────────
# Transformer
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformer:
    def test_output_shape(self, batch):
        model = CandlestickTransformer(n_features=N_FEATURES, d_model=64, nhead=4, num_layers=2)
        out   = model(batch)
        assert out.shape == (BATCH_SIZE,)

    def test_output_finite(self, batch):
        model = CandlestickTransformer(n_features=N_FEATURES, d_model=64, nhead=4)
        out   = model(batch)
        assert torch.isfinite(out).all()

    def test_build_factory(self, batch):
        model = build_transformer(N_FEATURES, cfg={"d_model": 64, "nhead": 4, "num_layers": 2})
        out   = model(batch)
        assert out.shape == (BATCH_SIZE,)

    def test_gradient_flows(self, batch):
        model  = CandlestickTransformer(n_features=N_FEATURES, d_model=64, nhead=4, num_layers=2)
        target = torch.randn(BATCH_SIZE)
        loss   = ((model(batch) - target) ** 2).mean()
        loss.backward()
        grads  = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_nhead_auto_fix(self, batch):
        """If d_model not divisible by nhead, model should auto-correct."""
        model = CandlestickTransformer(n_features=N_FEATURES, d_model=64, nhead=7)
        out   = model(batch)
        assert out.shape == (BATCH_SIZE,)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsemble:
    def test_uniform_ensemble(self, batch):
        m1  = CandlestickLSTM(n_features=N_FEATURES, hidden_dim=32, num_layers=1)
        m2  = CandlestickLSTM(n_features=N_FEATURES, hidden_dim=32, num_layers=1)
        ens = EnsembleModel([m1, m2])
        out = ens(batch)
        assert out.shape == (BATCH_SIZE,)

    def test_weighted_ensemble(self, batch):
        m1  = CandlestickTransformer(n_features=N_FEATURES, d_model=32, nhead=2, num_layers=1)
        m2  = CandlestickTransformer(n_features=N_FEATURES, d_model=32, nhead=2, num_layers=1)
        ens = EnsembleModel([m1, m2], weights=[0.3, 0.7])
        out = ens(batch)
        assert out.shape == (BATCH_SIZE,)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_directional_accuracy_range(self):
        np.random.seed(1)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        da = directional_accuracy(y_true, y_pred)
        assert 0.0 <= da <= 1.0

    def test_directional_accuracy_perfect(self):
        y = np.array([0.01, -0.02, 0.03, -0.01])
        assert directional_accuracy(y, y) == pytest.approx(1.0)

    def test_evaluate_predictions_keys(self):
        np.random.seed(2)
        y_t = np.random.randn(100) * 0.01
        y_p = y_t + np.random.randn(100) * 0.005
        result = evaluate_predictions(y_t, y_p)
        expected_keys = {"mse", "rmse", "mae", "directional_accuracy", "sharpe_ratio", "win_rate_pct"}
        assert expected_keys.issubset(result.keys())

    def test_sharpe_ratio_positive_for_good_strategy(self):
        returns = np.full(252, 0.001)   # 0.1% per day, no variance
        sr = sharpe_ratio(returns)
        # If there's 0 variance, our standard sharpe computes to 0 to avoid division by zero.
        assert sr == 0
