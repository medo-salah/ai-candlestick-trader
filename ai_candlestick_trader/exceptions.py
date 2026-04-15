"""
exceptions.py
=============
Custom exception classes for the AI Candlestick Trader package.
"""

class AICandlestickTraderError(Exception):
    """Base exception for all errors in the package."""
    pass

class DataDownloadError(AICandlestickTraderError):
    """Raised when market data fails to download via yfinance."""
    pass

class InsufficientDataError(AICandlestickTraderError):
    """Raised when there are not enough rows (bars) to build sequences or features."""
    pass

class FeatureEngineeringError(AICandlestickTraderError):
    """Raised when a feature generation step fails (e.g. division by zero, invalid data)."""
    pass

class ModelInitializationError(AICandlestickTraderError):
    """Raised when a model fails to build due to invalid architecture configurations."""
    pass

class ModelNotLoadedError(AICandlestickTraderError):
    """Raised when inference is requested but the model checkpoint could not be found or loaded."""
    pass

class TrainingError(AICandlestickTraderError):
    """Raised when an issue occurs during the training loop."""
    pass
