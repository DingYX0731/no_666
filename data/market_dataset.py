"""Feature engineering and sample-pair construction for Binance datasets.

This module converts raw Binance public archives into aligned model inputs.
Supported raw sources:
- klines (OHLCV + taker fields)
- aggTrades (aggregated trades)  -- streamed and bucketed to avoid OOM
- trades (raw trades)            -- streamed and bucketed to avoid OOM
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from data.binance_public_data import (
    AGG_TRADES_COLUMNS,
    KLINES_COLUMNS,
    TRADES_COLUMNS,
    BinancePublicDataClient,
)


@dataclass
class MarketFeatureDataset:
    """Aligned feature matrix with corresponding timestamps and closes."""

    open_times: np.ndarray  # shape [N]
    closes: np.ndarray  # shape [N]
    features: np.ndarray  # shape [N, F]
    feature_names: list[str]


@dataclass
class SupervisedPairs:
    """Sliding-window sample pairs for supervised models."""

    x: np.ndarray  # shape [M, lookback, F]
    y: np.ndarray  # shape [M, 1]
    target_times: np.ndarray  # shape [M]
    lookback: int
    horizon: int
    label_mode: Literal["binary", "regression"]


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _interval_to_ms(interval: str) -> int:
    s = interval.strip().lower()
    if len(s) < 2:
        raise ValueError(f"Invalid interval: {interval}")
    unit = s[-1]
    num = int(s[:-1])
    unit_ms = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    if unit not in unit_ms:
        raise ValueError(f"Unsupported interval unit '{unit}' in {interval}")
    return num * unit_ms[unit]


def _bucket_start(ts_ms: int, interval_ms: int) -> int:
    return (ts_ms // interval_ms) * interval_ms


def _fetch_kline_rows(
    client: BinancePublicDataClient,
    *,
    symbol: str,
    interval: str,
    frequency: str,
    start_date: str,
    end_date: str,
    limit: int,
    market: str,
    quote_asset: str,
) -> list[dict]:
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
        return []
    return client.iter_csv_rows(summary.extracted_csv_files, columns=KLINES_COLUMNS)


def _stream_aggregate_buckets(
    client: BinancePublicDataClient,
    *,
    symbol: str,
    dataset: str,
    frequency: str,
    start_date: str,
    end_date: str,
    limit: int,
    market: str,
    quote_asset: str,
    interval_ms: int,
) -> dict[int, dict[str, float]]:
    """Stream CSV files and aggregate into time-bucketed statistics.

    Unlike loading all rows into memory, this reads one CSV row at a time
    and accumulates into pre-allocated bucket dicts. Handles datasets with
    hundreds of millions of rows without OOM.
    """
    summary = client.fetch_history(
        symbol=symbol,
        dataset=dataset,
        interval=None,
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
        return {}

    if dataset == "aggTrades":
        columns = AGG_TRADES_COLUMNS
        ts_col_idx = columns.index("timestamp")
        qty_col_idx = columns.index("quantity")
        price_col_idx = columns.index("price")
        maker_col_idx = columns.index("is_buyer_maker")
    else:
        columns = TRADES_COLUMNS
        ts_col_idx = columns.index("time")
        qty_col_idx = columns.index("qty")
        price_col_idx = -1
        quote_col_idx = columns.index("quote_qty")
        maker_col_idx = columns.index("is_buyer_maker")

    buckets: dict[int, dict[str, float]] = {}

    for csv_path in summary.extracted_csv_files:
        with Path(csv_path).open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) <= max(ts_col_idx, qty_col_idx, maker_col_idx):
                    continue
                try:
                    ts = int(float(row[ts_col_idx]))
                except (ValueError, IndexError):
                    continue

                b = _bucket_start(ts, interval_ms)
                item = buckets.get(b)
                if item is None:
                    item = {"count": 0.0, "qty_sum": 0.0, "quote_sum": 0.0, "maker_sum": 0.0}
                    buckets[b] = item

                qty = _safe_float(row[qty_col_idx])
                is_maker = 1.0 if row[maker_col_idx].strip().lower() == "true" else 0.0

                if dataset == "aggTrades":
                    price = _safe_float(row[price_col_idx])
                    quote_val = qty * price
                else:
                    quote_val = _safe_float(row[quote_col_idx])

                item["count"] += 1.0
                item["qty_sum"] += qty
                item["quote_sum"] += quote_val
                item["maker_sum"] += is_maker

    return buckets


# ---------------------------------------------------------------------------
# Kline-only feature names (no aggTrades/trades dependency)
# ---------------------------------------------------------------------------
_KLINE_FEATURE_NAMES = [
    "ret1_log",
    "realized_vol",
    "hl_spread",
    "oc_change",
    "volume_log1p",
    "quote_volume_log1p",
    "num_trades_log1p",
    "taker_buy_base_ratio",
    "taker_buy_quote_ratio",
]

_AGG_FEATURE_NAMES = [
    "agg_count_log1p",
    "agg_qty_log1p",
    "agg_quote_log1p",
    "agg_maker_ratio",
]

_TRADE_FEATURE_NAMES = [
    "trade_count_log1p",
    "trade_qty_log1p",
    "trade_quote_log1p",
    "trade_maker_ratio",
]

_EMPTY_BUCKET = {"count": 0.0, "qty_sum": 0.0, "quote_sum": 0.0, "maker_sum": 0.0}


def build_market_feature_dataset(
    *,
    symbol: str,
    interval: str,
    frequency: str = "daily",
    start_date: str = "",
    end_date: str = "",
    limit: int = 0,
    market: str = "spot",
    quote_asset: str = "USDT",
    cache_dir: str = "data_cache/binance_public_data",
    use_agg_trades: bool = False,
    use_trades: bool = False,
    vol_window: int = 20,
) -> MarketFeatureDataset:
    """Build aligned feature dataset from Binance raw archives.

    When use_agg_trades / use_trades are False (default), only kline-derived
    features are produced (9 dimensions). This is memory-safe and fast.
    Enabling tick-level data uses streaming aggregation to avoid OOM.
    """
    client = BinancePublicDataClient(cache_dir=cache_dir)
    interval_ms = _interval_to_ms(interval)

    kline_rows = _fetch_kline_rows(
        client,
        symbol=symbol,
        interval=interval,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        market=market,
        quote_asset=quote_asset,
    )
    if not kline_rows:
        raise ValueError("No kline rows loaded for feature dataset.")
    kline_rows.sort(key=lambda r: int(r["open_time"]))

    agg_bucket: dict[int, dict[str, float]] = {}
    if use_agg_trades:
        print("[data] Streaming aggTrades into time buckets ...")
        agg_bucket = _stream_aggregate_buckets(
            client,
            symbol=symbol,
            dataset="aggTrades",
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            market=market,
            quote_asset=quote_asset,
            interval_ms=interval_ms,
        )
        print(f"[data] aggTrades: {len(agg_bucket)} buckets aggregated")

    trade_bucket: dict[int, dict[str, float]] = {}
    if use_trades:
        print("[data] Streaming trades into time buckets ...")
        trade_bucket = _stream_aggregate_buckets(
            client,
            symbol=symbol,
            dataset="trades",
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            market=market,
            quote_asset=quote_asset,
            interval_ms=interval_ms,
        )
        print(f"[data] trades: {len(trade_bucket)} buckets aggregated")

    open_times: list[int] = []
    closes: list[float] = []
    base_features: list[list[float]] = []

    prev_closes: list[float] = []
    log_returns: list[float] = []
    for row in kline_rows:
        ot = int(_safe_float(row.get("open_time", "0")))
        op = _safe_float(row.get("open", "0"))
        hi = _safe_float(row.get("high", "0"))
        lo = _safe_float(row.get("low", "0"))
        cl = _safe_float(row.get("close", "0"))
        vol = _safe_float(row.get("volume", "0"))
        qvol = _safe_float(row.get("quote_asset_volume", "0"))
        ntrades = _safe_float(row.get("number_of_trades", "0"))
        taker_base = _safe_float(row.get("taker_buy_base_asset_volume", "0"))
        taker_quote = _safe_float(row.get("taker_buy_quote_asset_volume", "0"))

        prev = prev_closes[-1] if prev_closes else cl
        prev_safe = max(prev, 1e-12)
        ret1 = np.log(max(cl, 1e-12) / prev_safe)
        log_returns.append(float(ret1))
        prev_closes.append(cl)

        left = max(0, len(log_returns) - vol_window)
        realized_vol = float(np.std(log_returns[left:])) if log_returns else 0.0
        hl_spread = (hi - lo) / max(cl, 1e-12)
        oc_change = (cl - op) / max(op, 1e-12)
        taker_base_ratio = taker_base / max(vol, 1e-12)
        taker_quote_ratio = taker_quote / max(qvol, 1e-12)

        feats: list[float] = [
            ret1,
            realized_vol,
            hl_spread,
            oc_change,
            np.log1p(vol),
            np.log1p(qvol),
            np.log1p(ntrades),
            taker_base_ratio,
            taker_quote_ratio,
        ]

        if use_agg_trades:
            bucket = _bucket_start(ot, interval_ms)
            agg = agg_bucket.get(bucket, _EMPTY_BUCKET)
            feats.extend([
                np.log1p(agg["count"]),
                np.log1p(agg["qty_sum"]),
                np.log1p(agg["quote_sum"]),
                agg["maker_sum"] / max(agg["count"], 1e-12),
            ])

        if use_trades:
            bucket = _bucket_start(ot, interval_ms)
            trd = trade_bucket.get(bucket, _EMPTY_BUCKET)
            feats.extend([
                np.log1p(trd["count"]),
                np.log1p(trd["qty_sum"]),
                np.log1p(trd["quote_sum"]),
                trd["maker_sum"] / max(trd["count"], 1e-12),
            ])

        open_times.append(ot)
        closes.append(cl)
        base_features.append(feats)

    feature_names = list(_KLINE_FEATURE_NAMES)
    if use_agg_trades:
        feature_names.extend(_AGG_FEATURE_NAMES)
    if use_trades:
        feature_names.extend(_TRADE_FEATURE_NAMES)

    return MarketFeatureDataset(
        open_times=np.asarray(open_times, dtype=np.int64),
        closes=np.asarray(closes, dtype=np.float64),
        features=np.asarray(base_features, dtype=np.float64),
        feature_names=feature_names,
    )


def build_supervised_pairs(
    dataset: MarketFeatureDataset,
    *,
    lookback: int = 20,
    horizon: int = 1,
    label_mode: Literal["binary", "regression"] = "binary",
) -> SupervisedPairs:
    """Construct many (X, y) pairs from aligned feature timeline."""
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive.")
    n = dataset.features.shape[0]
    if n <= lookback + horizon:
        raise ValueError("Not enough rows for selected lookback/horizon.")

    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    t_list: list[int] = []
    for t in range(lookback, n - horizon):
        window = dataset.features[t - lookback : t]
        close_now = dataset.closes[t]
        close_future = dataset.closes[t + horizon]
        future_ret = float(np.log(max(close_future, 1e-12) / max(close_now, 1e-12)))
        if label_mode == "binary":
            y_val = 1.0 if future_ret > 0 else 0.0
        else:
            y_val = future_ret
        x_list.append(window)
        y_list.append(y_val)
        t_list.append(int(dataset.open_times[t + horizon]))

    x = np.asarray(x_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64).reshape(-1, 1)
    target_times = np.asarray(t_list, dtype=np.int64)
    return SupervisedPairs(
        x=x,
        y=y,
        target_times=target_times,
        lookback=lookback,
        horizon=horizon,
        label_mode=label_mode,
    )


def split_supervised_pairs(
    pairs: SupervisedPairs,
    *,
    train_ratio: float = 0.8,
    method: Literal["chronological", "random"] = "chronological",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split pairs into train/test sets."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1).")
    x, y = pairs.x, pairs.y
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 samples to split.")

    if method == "chronological":
        cut = int(n * train_ratio)
        return x[:cut], y[:cut], x[cut:], y[cut:]

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * train_ratio)
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]
