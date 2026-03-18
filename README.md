# NO_666 Trade

Standardized quantitative trading repository with:

- live execution on Roostoo
- reusable data interface for research
- backtest, training, and model checkpoint deployment

Only `run_*.py` entrypoints are exposed at repository root.

---

## Root Entrypoints

- `run_ops.py`: operational commands (products, ticker, balance, orders)
- `run_trader.py`: live trading loop
- `run_backtest.py`: backtest runner
- `run_fetch_data.py`: historical data fetch runner
- `run_train_mlp.py`: model training runner

---

## API Credentials (No `.env` Edit Required)

Use shell environment variables directly:

```bash
export ROOSTOO_API_KEY="your_api_key"
export ROOSTOO_API_SECRET="your_api_secret"
python run_ops.py server-time
```

One-line style:

```bash
ROOSTOO_API_KEY="your_api_key" ROOSTOO_API_SECRET="your_api_secret" python run_trader.py --symbols BTC/USD --strategy ma --once
```

`.env` is optional and can be kept as a convenience fallback.

---

## Strategy Architecture

The strategy layer is class-based and yaml-driven:

- base class: `strategy/base.py`
- factory: `strategy/factory.py`
- implementations: `strategy/ma_strategy.py`, `strategy/mlp_strategy.py`
- configs: `configs/strategies/<strategy>.yaml`

Each strategy implements:

- `required_prices`
- `generate_signal(prices, position_coin) -> BUY | SELL | HOLD`

Default strategy configs:

- `configs/strategies/ma.yaml`
- `configs/strategies/mlp.yaml`

Use by strategy name:

```bash
python run_trader.py --symbols BTC/USD --strategy ma
python run_trader.py --symbols BTC/USD --strategy mlp
python run_backtest.py --data-source binance --symbol BTC/USD --interval 1h --start-date 2024-01-01 --end-date 2024-01-15 --strategy mlp
```

Optional custom strategy yaml path:

```bash
python run_trader.py --symbols BTC/USD --strategy mlp --strategy-config configs/strategies/mlp.yaml
```

---

## Model Training and Deployment

Train and save model checkpoint:

```bash
python run_train_mlp.py --symbol BTC/USD --interval 1h --start-date 2024-01-01 --end-date 2024-01-15 --ckpt-path checkpoints/mlp/default.npz
```

Checkpoint contains:

- model architecture metadata
- learned weights
- feature normalization stats

Deploy in live trader:

```bash
python run_trader.py --symbols BTC/USD --strategy mlp
```

The default `mlp` strategy reads checkpoint path from:

- `configs/strategies/mlp.yaml`

---

## Backtest

Synthetic:

```bash
python run_backtest.py --data-source synthetic --strategy ma
```

CSV:

```bash
python run_backtest.py --data-source csv --csv your_data.csv --price-col close --strategy ma
```

Binance + MLP strategy:

```bash
python run_backtest.py --data-source binance --symbol BTC/USD --interval 1h --start-date 2024-01-01 --end-date 2024-01-15 --strategy mlp
```

---

## Documentation

- `docs/DEPLOYMENT_GUIDE.md`
- `docs/QUICK_REFERENCE.md`
- `docs/BINANCE_DATA_INTERFACE.md`
