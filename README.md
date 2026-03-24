# no666

Automated cryptocurrency trading framework with pluggable strategy support, historical data pipelines, and model training infrastructure.

## Features

- Live trading on Roostoo exchange (single-pair, multi-pair, or all tradable pairs)
- Pluggable strategy system with YAML-driven configuration
- Backtesting engine with multiple data sources
- Binance public kline/trade data fetcher with local caching
- MLP and Deep RL (PPO with MLP+LSTM) model training pipelines
- Per-run isolated logging

## Repository Layout

```
no_666/
├── run_ops.py              # Account and order operations
├── run_trader.py           # Live trading loop
├── run_backtest.py         # Backtesting engine
├── run_fetch_data.py       # Historical data downloader
├── run_train_mlp.py        # MLP training entrypoint
├── run_train_drl.py        # DRL (PPO) training entrypoint
├── strategy/               # Strategy base class, implementations, factory
├── ml/                     # Model architectures, trainers, environments, loss
├── ml_demo/                # Training demo scripts
├── trade/                  # Exchange client, trading engine, logging
├── risk/                   # Risk management
├── ops/                    # Operational CLI (balance, orders, ticker)
├── data/                   # Canonical data access layer
├── data_interface/         # Binance public data client
├── configs/
│   ├── strategies/         # Inference/trading configs (per strategy)
│   └── ml/                 # Training configs (per model type)
├── docs/                   # Documentation
├── requirements.txt        # Core dependencies
└── requirements-drl.txt    # Optional: PyTorch + Stable-Baselines3
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set credentials via environment variables:

```bash
export ROOSTOO_API_KEY="..."
export ROOSTOO_API_SECRET="..."
```

Verify connectivity:

```bash
python run_ops.py server-time
python run_ops.py products
```

## Strategies

All strategies extend `strategy.BaseStrategy` and implement `generate_signal()`. Strategy parameters are loaded from `configs/strategies/<name>.yaml`.


| Strategy | Config                        | Description                                                 |
| -------- | ----------------------------- | ----------------------------------------------------------- |
| `ma`     | `configs/strategies/ma.yaml`  | Moving average crossover                                    |
| `mlp`    | `configs/strategies/mlp.yaml` | Trained MLP binary classifier                               |
| `drl`    | `configs/strategies/drl.yaml` | PPO agent with MLP+LSTM extractor (one checkpoint per pair) |


### Adding a New Strategy

1. Create `strategy/<name>_strategy.py` extending `BaseStrategy`
2. Register in `strategy/factory.py`
3. Add `configs/strategies/<name>.yaml`

## Configuration

Training and inference configs are intentionally separated:


| Purpose       | Path                          |
| ------------- | ----------------------------- |
| MLP training  | `configs/ml/mlp_train.yaml`   |
| DRL training  | `configs/ml/drl_train.yaml`   |
| MA inference  | `configs/strategies/ma.yaml`  |
| MLP inference | `configs/strategies/mlp.yaml` |
| DRL inference | `configs/strategies/drl.yaml` |


## Trading

```bash
# Single pair
python run_trader.py --symbols BTC/USD --strategy ma --once

# Multiple pairs
python run_trader.py --symbols BTC/USD,ETH/USD --strategy drl

# All tradable pairs
python run_trader.py --symbols all --strategy ma
```

Logs are written to `logs/trading/YYYYMMDD/<run_name>/trader.log`.

## Backtesting

```bash
# Synthetic data (writes logs/backtest/... when using --save-artifacts)
python run_backtest.py --data-source synthetic --strategy ma --save-artifacts

# Binance: product, kline interval, date range, strategy + config (full examples in docs/BACKTEST.md)
python run_backtest.py \
  --data-source binance \
  --symbol BTC/USD \
  --interval 1h \
  --frequency daily \
  --market spot \
  --quote-asset USDT \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --strategy drl \
  --strategy-config configs/strategies/drl.yaml \
  --save-artifacts

# CSV file
python run_backtest.py \
  --data-source csv \
  --csv data.csv \
  --price-col close \
  --strategy ma \
  --strategy-config configs/strategies/ma.yaml \
  --save-artifacts
```

Results (Start Equity, End Equity, Return, Max Drawdown, Trades) print to stdout. With `**--save-artifacts**`, per-bar `**steps.csv**` and `**backtest_chart.png**` go under `**logs/backtest/YYYYMMDD/<run>/**` (same style as `logs/training/` and `logs/trading/`). See `docs/BACKTEST.md` for columns, overrides (`--report-dir`, `--step-log`, `--plot`), and more examples.

## Model Training

### MLP

```bash
python run_train_mlp.py --config configs/ml/mlp_train.yaml
```

Produces `checkpoints/mlp/*.npz` containing weights and normalization stats.
Training logs: `logs/training/YYYYMMDD/mlp_<HHMMSS>/train.log`.
The training pipeline now uses `data/market_dataset.py` to:

- align `klines`, `aggTrades`, and `trades` to the kline timeframe
- build multi-factor feature vectors (price, volatility, volume, flow, microstructure)
- construct many supervised `(X, y)` sample pairs with configurable `lookback` and `horizon`
- split into train/test sets via chronological or random policy

For detailed feature dimensions and alignment rules, see `data/README.md`.

### DRL (PPO)

Requires additional dependencies:

```bash
pip install -r requirements-drl.txt
```

Train one agent per symbol:

```bash
python run_train_drl.py --config configs/ml/drl_train.yaml
```

Produces per-pair artifacts under `checkpoints/drl/`:

- `<PAIR>_ppo.zip` (SB3 model)
- `<PAIR>_meta.json` (architecture and training metadata)

Training logs: `logs/training/YYYYMMDD/drl_<HHMMSS>/train.log`.

The DRL feature extractor uses a configurable MLP+LSTM architecture. All hyperparameters (LSTM size, layer count, MLP dimensions, PPO settings) are managed in `configs/ml/drl_train.yaml`.

## Historical Data

```bash
python run_fetch_data.py --symbol BTC/USD --dataset klines --interval 1h \
  --frequency daily --start-date 2024-01-01 --end-date 2024-01-15
```

Data source: [Binance Public Data](https://github.com/binance/binance-public-data). Downloaded files are cached locally under `data_cache/`.

## Operations

```bash
python run_ops.py server-time
python run_ops.py products
python run_ops.py ticker --pair BTC/USD
python run_ops.py balance --pair BTC/USD
python run_ops.py orders --pair BTC/USD
python run_ops.py place-order --pair BTC/USD --side BUY --type MARKET --quantity 0.001 --force
```

## License

MIT