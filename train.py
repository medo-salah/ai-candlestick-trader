"""
train.py  —  top-level training entry point (project root)
===========================================================
A simple script wrapper so users can run:

    python train.py --ticker COMI.CA --epochs 100

This just delegates to the CLI module.
"""

import sys
from ai_candlestick_trader.cli import train_cli

if __name__ == "__main__":
    train_cli()
