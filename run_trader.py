import argparse

from trade.trader_engine import run


def main() -> None:
    """CLI wrapper for the live trading loop."""
    parser = argparse.ArgumentParser(description="Run auto trader loop")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USD",
        help="One pair, comma-separated pairs, or 'all'. Example: BTC/USD,ETH/USD",
    )
    parser.add_argument("--poll-seconds", type=int, default=0, help="Override POLL_SECONDS when > 0")
    parser.add_argument("--strategy", type=str, default="ma", help="Strategy name, mapped to configs/strategies/<name>.yaml")
    parser.add_argument("--strategy-config", type=str, default="", help="Optional explicit strategy config yaml path")
    parser.add_argument("--run-name", type=str, default="", help="Optional custom run name for log folder")
    parser.add_argument("--once", action="store_true", help="Run one loop only and exit")
    args = parser.parse_args()
    run(
        once=args.once,
        symbols=args.symbols,
        poll_seconds=args.poll_seconds or None,
        strategy_name=args.strategy,
        strategy_config=args.strategy_config,
        run_name=args.run_name or None,
    )


if __name__ == "__main__":
    main()
