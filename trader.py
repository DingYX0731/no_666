"""Backward-compatible wrapper for the new trade engine."""

import argparse

from trade.trader_engine import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run no_666 strategy trader")
    parser.add_argument("--once", action="store_true", help="Run one loop only and exit")
    parser.add_argument("--symbols", type=str, default="BTC/USD", help="One pair, comma list, or all")
    args = parser.parse_args()
    run(once=args.once, symbols=args.symbols)
