"""
AI Candlestick Trader — Package Root
=====================================
A deep-learning package for candlestick pattern recognition and
price-movement prediction on EGX (Egypt) and Tadawul (Saudi Arabia) stocks.

Usage
-----
>>> from ai_candlestick_trader.data.downloader import download_ohlc
>>> from ai_candlestick_trader.data.features import build_features
>>> from ai_candlestick_trader.models.transformer_model import CandlestickTransformer
"""

__version__ = "2.0.0"
__author__ = "AI Candlestick Trader Team"

# ── Convenience re-exports ────────────────────────────────────────────────────
from ai_candlestick_trader.data.downloader import download_ohlc          # noqa: F401
from ai_candlestick_trader.data.features import build_features            # noqa: F401
from ai_candlestick_trader.data.patterns import detect_patterns           # noqa: F401
from ai_candlestick_trader.models.transformer_model import CandlestickTransformer  # noqa: F401
from ai_candlestick_trader.evaluation.metrics import evaluate_predictions  # noqa: F401
