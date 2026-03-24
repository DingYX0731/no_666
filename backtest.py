import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from data.binance_public_data import KLINES_COLUMNS, BinancePublicDataClient
from data.market_dataset import build_market_feature_dataset
from strategy import BaseStrategy, build_strategy
from trade.logging_utils import prepare_backtest_artifact_dir


@dataclass
class BacktestResult:
    """Container for backtest performance summary metrics.

    - ``start_equity``: initial quote cash (before the first bar); not intraday MTM.
    - ``end_equity``: after the last bar, ``quote + base * last_close`` — i.e. marked to
      the **final** price in ``prices``. For buy-and-hold this reflects PnL vs start
      (plus fees on entry); it only equals ``start_equity`` if, after fees, MTM happens
      to match (unusual).
    - ``end_price``: last bar close used for that final equity (same as ``prices[-1]``).
    """
    start_equity: float
    end_equity: float
    return_pct: float
    max_drawdown_pct: float
    trades: int
    end_price: float


def read_prices_from_csv(csv_path: Path, price_col: str = "close") -> List[float]:
    """Load price series from CSV file."""
    prices: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if price_col not in row:
                raise ValueError(f"CSV does not contain column: {price_col}")
            prices.append(float(row[price_col]))
    if not prices:
        raise ValueError("No prices loaded from CSV.")
    return prices


def synthetic_prices(n: int = 500, seed: int = 7) -> List[float]:
    """Generate synthetic random-walk prices for quick dry testing."""
    random.seed(seed)
    prices = [100.0]
    for _ in range(n - 1):
        drift = 0.0003
        shock = random.uniform(-0.01, 0.01)
        prices.append(max(1.0, prices[-1] * (1.0 + drift + shock)))
    return prices


def read_prices_from_binance(
    symbol: str,
    interval: str,
    frequency: str,
    start_date: str,
    end_date: str,
    limit: int,
    market: str = "spot",
    quote_asset: str = "USDT",
    cache_dir: str = "data_cache/binance_public_data",
) -> List[float]:
    """Fetch klines from Binance public data and return sorted close prices."""
    client = BinancePublicDataClient(cache_dir=cache_dir)
    summary = client.fetch_history(
        symbol=symbol,
        dataset="klines",
        interval=interval,
        frequency=frequency,
        start_date=start_date or None,
        end_date=end_date or None,
        limit=limit or None,
        market=market,
        quote_asset=quote_asset,
        extract=True,
        skip_missing=True,
    )
    if not summary.extracted_csv_files:
        raise ValueError("No Binance kline files found for the given parameters.")

    rows = client.iter_csv_rows(summary.extracted_csv_files, columns=KLINES_COLUMNS)
    if not rows:
        raise ValueError("No rows loaded from Binance kline files.")

    rows.sort(key=lambda x: int(x["open_time"]))
    closes = [float(row["close"]) for row in rows]
    if not closes:
        raise ValueError("No close prices extracted from Binance kline rows.")
    return closes


def _write_step_log_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "time_ms",
        "price",
        "quote",
        "base_coin",
        "equity_before",
        "signal",
        "executed",
        "conf_hold",
        "conf_buy",
        "conf_sell",
        "strategy_input",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _plot_backtest(
    prices: Sequence[float],
    buy_steps: list[int],
    sell_steps: list[int],
    out_path: Path,
    time_ms: Sequence[int] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib. pip install matplotlib") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(prices)
    if time_ms is not None and len(time_ms) == n:
        x = [int(t) / 1000.0 for t in time_ms]
        xlab = "Time (s since epoch, ms/1000)"
    else:
        x = list(range(n))
        xlab = "Bar index"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, list(prices), color="steelblue", linewidth=1.0, label="Close")

    if buy_steps:
        bx = [x[i] for i in buy_steps if 0 <= i < n]
        by = [prices[i] for i in buy_steps if 0 <= i < n]
        ax.scatter(bx, by, marker="^", s=80, c="green", zorder=5, label="Buy (executed)")
    if sell_steps:
        sx = [x[i] for i in sell_steps if 0 <= i < n]
        sy = [prices[i] for i in sell_steps if 0 <= i < n]
        ax.scatter(sx, sy, marker="v", s=80, c="red", zorder=5, label="Sell (executed)")

    ax.set_xlabel(xlab)
    ax.set_ylabel("Price")
    ax.set_title("Backtest: price with executed trades")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_backtest(
    prices: Sequence[float],
    strategy: BaseStrategy,
    fee_rate: float = 0.0008,
    initial_quote: float = 10000.0,
    features: np.ndarray | None = None,
    step_log_path: Path | None = None,
    plot_path: Path | None = None,
    time_ms: Sequence[int] | None = None,
) -> BacktestResult:
    """Run backtest with any strategy implementing BaseStrategy.

    When ``features`` is provided (shape [N, F]), it is passed to the
    strategy at each step via ``kwargs['features']`` and ``kwargs['step']``.

    If ``step_log_path`` or ``plot_path`` is set, each step uses
    ``strategy.evaluate_step`` to log inputs and (HOLD/BUY/SELL) confidences.
    """
    want_detail = step_log_path is not None or plot_path is not None
    quote = initial_quote
    base = 0.0
    trades = 0
    equity_curve: List[float] = []
    log_rows: list[dict[str, Any]] = []
    buy_steps: list[int] = []
    sell_steps: list[int] = []

    history: List[float] = []
    n_bars = len(prices)
    for i, price in enumerate(prices):
        if i > 0 and i % 500 == 0:
            print(f"[backtest] progress {i}/{n_bars} bars ...", flush=True)
        history.append(price)
        extra_kwargs: dict = {"quote_free": quote, "last_price": price}
        if features is not None:
            extra_kwargs["features"] = features
            extra_kwargs["step"] = i

        quote_before = quote
        base_before = base
        equity_before = quote_before + base_before * price

        if want_detail:
            signal, diag = strategy.evaluate_step(history, base_before, **extra_kwargs)
        else:
            signal = strategy.generate_signal(history, base_before, **extra_kwargs)
            diag = {}

        executed = False
        if signal == "BUY" and quote > 0:
            raw_qty = quote / price
            qty = raw_qty * (1.0 - fee_rate)
            base += qty
            quote = 0.0
            trades += 1
            executed = True
            buy_steps.append(i)
        elif signal == "SELL" and base > 0:
            raw_quote = base * price
            quote += raw_quote * (1.0 - fee_rate)
            base = 0.0
            trades += 1
            executed = True
            sell_steps.append(i)

        if want_detail:
            t_ms = int(time_ms[i]) if time_ms is not None and i < len(time_ms) else ""
            ch = diag.get("conf_hold", float("nan"))
            cb = diag.get("conf_buy", float("nan"))
            cs = diag.get("conf_sell", float("nan"))
            log_rows.append(
                {
                    "step": i,
                    "time_ms": t_ms,
                    "price": f"{price:.8f}",
                    "quote": f"{quote_before:.6f}",
                    "base_coin": f"{base_before:.8f}",
                    "equity_before": f"{equity_before:.6f}",
                    "signal": signal,
                    "executed": "yes" if executed else "no",
                    "conf_hold": f"{ch:.6f}" if ch == ch else "",
                    "conf_buy": f"{cb:.6f}" if cb == cb else "",
                    "conf_sell": f"{cs:.6f}" if cs == cs else "",
                    "strategy_input": diag.get("input_summary", ""),
                }
            )

        equity = quote + base * price
        equity_curve.append(equity)

    # Final equity uses the last bar's close (same `price` as final loop iteration).
    end_equity = equity_curve[-1]
    end_price = float(prices[-1]) if n_bars > 0 else 0.0
    peak = equity_curve[0]
    max_drawdown = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

    if step_log_path is not None:
        _write_step_log_csv(Path(step_log_path), log_rows)
        print(f"[backtest] Step log written: {step_log_path}", flush=True)
    if plot_path is not None:
        _plot_backtest(prices, buy_steps, sell_steps, Path(plot_path), time_ms=time_ms)
        print(f"[backtest] Chart written: {plot_path}", flush=True)

    return BacktestResult(
        start_equity=initial_quote,
        end_equity=end_equity,
        return_pct=(end_equity - initial_quote) / initial_quote,
        max_drawdown_pct=max_drawdown,
        trades=trades,
        end_price=end_price,
    )


def main() -> None:
    """CLI entry for quick local backtest."""
    parser = argparse.ArgumentParser(description="Backtest runner with pluggable strategy")
    parser.add_argument(
        "--data-source",
        type=str,
        default="synthetic",
        choices=["synthetic", "csv", "binance"],
        help="Backtest data source",
    )
    parser.add_argument("--csv", type=str, default="", help="CSV path with close column")
    parser.add_argument("--price-col", type=str, default="close", help="Price column name")
    parser.add_argument("--symbol", type=str, default="BTC/USD", help="Symbol for binance source")
    parser.add_argument("--interval", type=str, default="1h", help="Kline interval for binance source")
    parser.add_argument("--frequency", type=str, default="daily", choices=["daily", "monthly"])
    parser.add_argument("--market", type=str, default="spot", choices=["spot", "um", "cm"])
    parser.add_argument("--start-date", type=str, default="", help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="", help="YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=0, help="Period count when date range is omitted")
    parser.add_argument("--quote-asset", type=str, default="USDT")
    parser.add_argument("--cache-dir", type=str, default="data_cache/binance_public_data")
    parser.add_argument("--strategy", type=str, default="ma", help="Strategy name")
    parser.add_argument("--strategy-config", type=str, default="", help="Strategy yaml path")
    parser.add_argument(
        "--step-log",
        type=str,
        default="",
        help="CSV path for per-bar log (signal, confidences, inputs). Empty to skip.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="PNG path for price chart with executed buy/sell markers. Empty to skip.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="",
        help="Explicit directory for steps.csv + backtest_chart.png (creates dirs if needed).",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Write per-bar CSV + price chart under logs/backtest/YYYYMMDD/<run>/ (see --artifact-base).",
    )
    parser.add_argument(
        "--artifact-base",
        type=str,
        default="logs/backtest",
        help="Root directory when using --save-artifacts (default: logs/backtest).",
    )
    args = parser.parse_args()

    features = None
    time_ms_list: list[int] | None = None
    if args.data_source == "csv":
        if not args.csv:
            raise ValueError("--csv is required when --data-source=csv")
        prices = read_prices_from_csv(Path(args.csv), args.price_col)
    elif args.data_source == "binance":
        dataset = build_market_feature_dataset(
            symbol=args.symbol,
            interval=args.interval,
            frequency=args.frequency,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
            market=args.market,
            quote_asset=args.quote_asset,
            cache_dir=args.cache_dir,
            use_agg_trades=False,
            use_trades=False,
        )
        prices = dataset.closes.tolist()
        features = dataset.features
        time_ms_list = [int(x) for x in dataset.open_times.tolist()]
    else:
        prices = synthetic_prices(500)

    pair_label = args.symbol.strip().upper()

    artifact_root: Path | None = None
    if args.report_dir.strip():
        artifact_root = Path(args.report_dir.strip())
        artifact_root.mkdir(parents=True, exist_ok=True)
    elif args.save_artifacts:
        artifact_root = prepare_backtest_artifact_dir(
            base_dir=args.artifact_base,
            symbol=pair_label,
            strategy=args.strategy,
        )

    step_log_path: Path | None = None
    plot_path: Path | None = None
    if artifact_root is not None:
        step_log_path = artifact_root / "steps.csv"
        plot_path = artifact_root / "backtest_chart.png"
    if args.step_log.strip():
        step_log_path = Path(args.step_log.strip())
    if args.plot.strip():
        plot_path = Path(args.plot.strip())

    if artifact_root is not None:
        print(f"[backtest] Artifact directory: {artifact_root.resolve()}", flush=True)
    print(f"[backtest] Building strategy: {args.strategy} ...", flush=True)
    strategy = build_strategy(
        strategy_name=args.strategy,
        strategy_config=args.strategy_config,
        pair=pair_label,
    )
    print(f"[backtest] Running {len(prices)} bars ...", flush=True)
    result = run_backtest(
        prices=prices,
        strategy=strategy,
        features=features,
        step_log_path=step_log_path,
        plot_path=plot_path,
        time_ms=time_ms_list,
    )

    print("=== Backtest Result ===")
    print(f"Strategy     : {strategy.name}")
    print(f"Need Prices  : {strategy.required_prices}")
    print(f"Start Equity : {result.start_equity:.2f}  (initial quote cash)")
    print(f"End Price    : {result.end_price:.8f}  (last bar, used for MTM)")
    print(f"End Equity   : {result.end_equity:.2f}  (= quote + base * End Price)")
    print(f"Return       : {result.return_pct:.2%}")
    print(f"Max Drawdown : {result.max_drawdown_pct:.2%}")
    print(f"Trades       : {result.trades}")


if __name__ == "__main__":
    main()
