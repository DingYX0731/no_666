# data

Reusable historical market data access layer.

## Files

- `binance_public_data.py`: canonical import path for Binance archive interface
- `market_dataset.py`: feature engineering + aligned sample pair builder

## Data source

Binance public archive:
- https://github.com/binance/binance-public-data

## Supported raw inputs

- `klines`: OHLCV candles and taker-buy related fields
- `aggTrades`: aggregated trade stream
- `trades`: raw trade stream

## Feature dimensions (default)

`build_market_feature_dataset(...)` aligns all sources to kline interval buckets
and outputs:

- timestamps: `[N]`
- closes: `[N]`
- features: `[N, 17]`

Default 17 features:

1. `ret1_log`
2. `realized_vol`
3. `hl_spread`
4. `oc_change`
5. `volume_log1p`
6. `quote_volume_log1p`
7. `num_trades_log1p`
8. `taker_buy_base_ratio`
9. `taker_buy_quote_ratio`
10. `agg_count_log1p`
11. `agg_qty_log1p`
12. `agg_quote_log1p`
13. `agg_maker_ratio`
14. `trade_count_log1p`
15. `trade_qty_log1p`
16. `trade_quote_log1p`
17. `trade_maker_ratio`

## Processed cache

`build_market_feature_dataset` caches processed datasets under `data_cache/processed/`
when `use_cache=True` (default). Same params (symbol, interval, dates, etc.) reuse
the cache instead of re-fetching and re-processing. Set `use_cache=False` to force
rebuild. Set `verbose=False` to suppress progress logs.

## Alignment rule

- Use each kline `open_time` as the anchor.
- For `aggTrades`/`trades`, bucket records by `floor(timestamp / interval_ms) * interval_ms`.
- Aggregate each bucket into count/size/quote/maker-ratio statistics.
- Missing bucket values are filled with zeros.

## Supervised pair construction

`build_supervised_pairs(...)` converts timeline features into many `(X, y)` pairs:

- `X`: rolling feature window `[lookback, F]`
- `y`: next-`horizon` movement label (binary or regression)

`split_supervised_pairs(...)` supports:

- `chronological`: time-ordered split (recommended for backtest realism)
- `random`: shuffled split (faster validation, less realistic for deployment)
