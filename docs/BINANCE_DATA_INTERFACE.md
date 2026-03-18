# Binance Data Interface

Implementation:

- `data_interface/binance_public_data.py` (low-level)
- `data/binance_public_data.py` (canonical import path)
- `run_fetch_data.py` (top-level CLI)

Data source:
- [Binance Public Data](https://github.com/binance/binance-public-data)

## Supported Dimensions

- `symbol`: `BTC/USD`, `BTCUSDT`, `BTC`
- `dataset`: `klines`, `aggTrades`, `trades`
- `market`: `spot`, `um`, `cm`
- `frequency`: `daily`, `monthly`
- `interval`: required for `klines` (e.g., `1m`, `1h`, `1d`)
- `start-date`, `end-date`, or `limit`

## Example

```bash
python run_fetch_data.py \
  --symbol BTC/USD \
  --dataset klines \
  --interval 1h \
  --frequency daily \
  --start-date 2024-01-01 \
  --end-date 2024-01-03 \
  --preview-rows 3
```

## Return Summary

`FetchSummary` includes:

- `downloaded`
- `cache_hits`
- `skipped_missing`
- `zip_files`
- `extracted_csv_files`
- `missing_urls`

## Integration

- Backtest uses this interface via `--data-source binance`
- MLP training uses this interface before feature generation
