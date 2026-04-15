"""
data/downloader.py
==================
Downloads historical OHLC data for EGX (Egypt), Tadawul (Saudi Arabia),
and any global ticker via yfinance.

EGX tickers on Yahoo Finance use the suffix `.CA`  (e.g. COMI.CA).
Tadawul tickers use `.SR`                          (e.g. 2222.SR for Aramco).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Well-known EGX 30 tickers (Yahoo Finance format) ──────────────────────────
EGX_TICKERS: dict[str, str] = {
    "Commercial International Bank": "COMI.CA",
    "Eastern Company":               "EAST.CA",
    "Egyptian Kuwaiti Holding":      "EKHO.CA",
    "Juhayna Food Industries":       "JUFO.CA",
    "Medinet Nasr Housing":          "MNHD.CA",
    "Cairo Poultry":                 "POUL.CA",
    "Talaat Moustafa Group":         "TMGH.CA",
    "Ezz Steel":                     "ESRS.CA",
    "Egyptian Resorts":              "EGTS.CA",
    "El Sewedy Electric":            "SWDY.CA",
}

# ── Well-known Tadawul / TASI tickers ────────────────────────────────────────
TADAWUL_TICKERS: dict[str, str] = {
    "Saudi Aramco":          "2222.SR",
    "Al Rajhi Bank":         "1120.SR",
    "Saudi National Bank":   "1180.SR",
    "Saudi Telecom":         "7010.SR",
    "SABIC":                 "2010.SR",
    "Riyad Bank":            "1010.SR",
    "Saudi Electricity":     "5110.SR",
    "Maaden":                "1211.SR",
    "Jarir Marketing":       "4190.SR",
    "Extra Stores":          "4extra.SR",
}

ALL_TICKERS = {**EGX_TICKERS, **TADAWUL_TICKERS}


def download_ohlc(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download OHLC data for *ticker* and return a clean DataFrame.

    Parameters
    ----------
    ticker   : Yahoo Finance symbol, e.g. ``"COMI.CA"`` or ``"2222.SR"``.
    period   : yfinance period string (``"1y"``, ``"5y"``, etc.).
               Ignored when *start* is provided.
    interval : Bar interval – ``"1d"``, ``"1h"``, ``"5m"``, etc.
    start    : ISO date string ``"YYYY-MM-DD"`` (overrides *period*).
    end      : ISO date string ``"YYYY-MM-DD"`` (default = today).

    Returns
    -------
    pd.DataFrame with columns ``[Open, High, Low, Close, Volume]``
    and a ``DatetimeIndex``.  Raises ``ValueError`` if data is empty.
    """
    kwargs: dict = dict(ticker=ticker, interval=interval, auto_adjust=True, progress=False)
    if start:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = period

    logger.info("Downloading %s  interval=%s", ticker, interval)
    df = yf.download(**kwargs)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check the symbol (EGX → .CA suffix, Tadawul → .SR suffix)."
        )

    # ── Flatten multi-index columns produced by newer yfinance versions ───────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    logger.info("Downloaded %d rows for %s", len(df), ticker)
    return df


def list_markets() -> dict[str, dict[str, str]]:
    """Return the built-in EGX and Tadawul ticker dictionaries."""
    return {"egx": EGX_TICKERS, "tadawul": TADAWUL_TICKERS}
