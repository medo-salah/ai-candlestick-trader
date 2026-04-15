"""
api.py  (v2 — OHLC-based)
==========================
FastAPI server exposing:

  GET  /ping               — health check
  POST /analyze/ticker     — analyse a ticker symbol (full pipeline)
  POST /analyze/ohlc       — analyse raw JSON OHLC data
  GET  /patterns           — list all supported pattern names
  GET  /tickers            — list built-in EGX / Tadawul tickers

Legacy image endpoint (/analyze) is preserved for backward compat.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ai_candlestick_trader.exceptions import (
    DataDownloadError,
    InsufficientDataError,
    ModelNotLoadedError,
)

app = FastAPI(
    title="AI Candlestick Trader API",
    version="2.0.0",
    description="Deep-learning candlestick analysis for EGX and Tadawul stocks.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH    = os.environ.get("MODEL_PATH",    "checkpoints/best_model.pt")
MODEL_TYPE    = os.environ.get("MODEL_TYPE",    "transformer")
SEQ_LEN       = int(os.environ.get("SEQ_LEN",   "30"))
SIGNAL_THR    = float(os.environ.get("SIGNAL_THR", "0.005"))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

_model        = None
_n_features   = None


# ── Model loading (lazy) ──────────────────────────────────────────────────────

def _load_model():
    global _model, _n_features
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg  = ckpt.get("cfg", {})

    # Try to infer n_features from checkpoint
    state = ckpt.get("model_state_dict", ckpt)
    # First linear weight shape gives (out, in)
    first_key = next(k for k in state if "weight" in k)
    _n_features = state[first_key].shape[1]

    if MODEL_TYPE == "transformer":
        from ai_candlestick_trader.models.transformer_model import CandlestickTransformer
        model = CandlestickTransformer(n_features=_n_features, **{
            k: v for k, v in cfg.items()
            if k in ("d_model", "nhead", "num_layers", "dim_ff", "dropout")
        })
    else:
        from ai_candlestick_trader.models.lstm_model import CandlestickLSTM
        model = CandlestickLSTM(n_features=_n_features, **{
            k: v for k, v in cfg.items()
            if k in ("hidden_dim", "num_layers", "dropout")
        })

    model.load_state_dict(state)
    model.to(DEVICE).eval()
    _model = model
    return _model


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TickerRequest(BaseModel):
    ticker:   str       = "COMI.CA"
    period:   str       = "1y"
    interval: str       = "1d"
    seq_len:  int       = 30
    threshold: float    = 0.005


class OHLCRequest(BaseModel):
    """Send raw OHLC bars as JSON. dates must be ISO strings."""
    dates:    list[str]
    open:     list[float]
    high:     list[float]
    low:      list[float]
    close:    list[float]
    volume:   list[float]
    seq_len:  int   = 30
    threshold: float = 0.005


# ── Helper: run pipeline on a DataFrame ──────────────────────────────────────

def _run_pipeline(ohlc_df: pd.DataFrame, seq_len: int, threshold: float) -> dict:
    from ai_candlestick_trader.data.features  import build_features
    from ai_candlestick_trader.data.patterns  import detect_patterns, pattern_summary
    from ai_candlestick_trader.data.dataset   import OHLCDataset
    from ai_candlestick_trader.evaluation.metrics import evaluate_predictions

    pat_df  = detect_patterns(ohlc_df)
    feat_df = build_features(ohlc_df, pattern_flags=pat_df)
    ds      = OHLCDataset(feat_df, ohlc_df["Close"], seq_len=seq_len)

    # Pattern counts
    pat_counts = pattern_summary(pat_df)
    recent_pats = [
        col.replace("pat_", "").replace("_", " ").title()
        for col in pat_df.columns
        if pat_df[col].iloc[-5:].any()
    ]

    # Inference (if model loaded)
    model = _load_model()
    signal = "HOLD"
    predicted_return_pct = None
    metrics_out: dict = {}

    if model is not None and len(ds) > 0:
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        preds_list, targets_list = [], []
        with torch.no_grad():
            for X, y in loader:
                preds_list.append(model(X.to(DEVICE)).cpu().numpy())
                targets_list.append(y.numpy())

        preds    = np.concatenate(preds_list)
        targets  = np.concatenate(targets_list)
        last_pred = float(preds[-1])
        predicted_return_pct = round(last_pred * 100, 4)

        if last_pred > threshold:
            signal = "BUY"
        elif last_pred < -threshold:
            signal = "SELL"

        metrics_raw = evaluate_predictions(targets, preds)
        metrics_out = {k: round(v, 6) for k, v in metrics_raw.items()}

    latest = ohlc_df.iloc[-1]
    prev   = ohlc_df.iloc[-2] if len(ohlc_df) > 1 else latest

    return {
        "signal":                signal,
        "predicted_return_pct":  predicted_return_pct,
        "latest_close":          float(latest["Close"]),
        "change_pct":            round((latest["Close"] - prev["Close"]) / prev["Close"] * 100, 4),
        "total_bars":            len(ohlc_df),
        "recent_patterns":       recent_pats[:8],
        "pattern_counts":        {k: v for k, v in pat_counts.items() if v > 0},
        "metrics":               metrics_out,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    return {
        "status":       "ok",
        "model_loaded": _load_model() is not None,
        "model_path":   MODEL_PATH,
        "device":       DEVICE,
        "version":      "2.0.0",
    }


@app.get("/tickers")
def list_tickers():
    from ai_candlestick_trader.data.downloader import EGX_TICKERS, TADAWUL_TICKERS
    return {"egx": EGX_TICKERS, "tadawul": TADAWUL_TICKERS}


@app.get("/patterns")
def list_patterns():
    from ai_candlestick_trader.data.patterns import PATTERN_FUNCS, BULLISH_PATTERNS, BEARISH_PATTERNS
    return {
        "all":     list(PATTERN_FUNCS.keys()),
        "bullish": BULLISH_PATTERNS,
        "bearish": BEARISH_PATTERNS,
    }


@app.post("/analyze/ticker")
def analyze_ticker(req: TickerRequest):
    try:
        from ai_candlestick_trader.data.downloader import download_ohlc
        ohlc_df = download_ohlc(req.ticker, period=req.period, interval=req.interval)
    except DataDownloadError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {e}")

    try:
        result = _run_pipeline(ohlc_df, req.seq_len, req.threshold)
        result["ticker"] = req.ticker
        return JSONResponse(result)
    except InsufficientDataError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")


@app.post("/analyze/ohlc")
def analyze_ohlc(req: OHLCRequest):
    if not (len(req.dates) == len(req.open) == len(req.high) == len(req.low) == len(req.close)):
        raise HTTPException(status_code=422, detail="All OHLC arrays must have equal length.")
    if len(req.dates) < req.seq_len + 2:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {req.seq_len + 2} bars, got {len(req.dates)}.",
        )
    try:
        idx     = pd.to_datetime(req.dates)
        ohlc_df = pd.DataFrame({
            "Open":   req.open, "High": req.high,
            "Low":    req.low,  "Close": req.close,
            "Volume": req.volume if req.volume else [1e6] * len(req.dates),
        }, index=idx)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid data: {e}")

    try:
        result = _run_pipeline(ohlc_df, req.seq_len, req.threshold)
        return JSONResponse(result)
    except InsufficientDataError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
