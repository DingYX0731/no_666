# data

Reusable historical market data access layer.

## Files

- `binance_public_data.py`: canonical import path for Binance archive interface

## Data source

Binance public archive:
- https://github.com/binance/binance-public-data

## Typical usage

Use `BinancePublicDataClient.fetch_history(...)` to download/extract archives,
then `iter_csv_rows(...)` to read rows for backtest/model pipelines.
