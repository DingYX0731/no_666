# NO_666 Trade

Standardized quantitative trading repository with:

- live execution on Roostoo
- reusable data interface for research
- backtest, training, and model checkpoint deployment

---

## Root Entrypoints

- `run_ops.py`: operational commands (products, ticker, balance, orders)
- `run_trader.py`: live trading loop
- `run_backtest.py`: backtest runner
- `run_fetch_data.py`: historical data fetch runner
- `run_train_mlp.py`: model training runner
- `run_train_drl.py`: per-product PPO training (Stable-Baselines3)

---

## API Credentials

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
- implementations: `strategy/ma_strategy.py`, `strategy/mlp_strategy.py`, `strategy/drl_strategy.py`
- configs: `configs/strategies/<strategy>.yaml`

Each strategy implements:

- `required_prices`
- `generate_signal(prices, position_coin, **kwargs) -> BUY | SELL | HOLD` (optional `quote_free`, `last_price` for DRL)

Default strategy configs:

- `configs/strategies/ma.yaml`
- `configs/strategies/mlp.yaml`
- `configs/strategies/drl.yaml`

Use by strategy name:

```bash
python run_trader.py --symbols BTC/USD --strategy ma
python run_trader.py --symbols BTC/USD --strategy mlp
python run_trader.py --symbols BTC/USD,ETH/USD --strategy drl
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

## DRL (FinRL-style crypto env + PPO)

FinRL reference clone: `../FinRL` (see `docs/DRL_AND_FINRL.md`).

Install extra deps, then train **one agent per symbol**:

```bash
pip install -r requirements-drl.txt
python run_train_drl.py --symbol BTC/USD --start-date 2024-01-01 --end-date 2024-02-01 --timesteps 50000
```

Trade with the saved checkpoint:

```bash
python run_trader.py --symbols BTC/USD --strategy drl
```

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

Binance + DRL (match `--symbol` to trained pair):

```bash
python run_backtest.py --data-source binance --symbol BTC/USD --interval 1h --start-date 2024-01-01 --end-date 2024-02-01 --strategy drl
```

---

## Documentation

- `docs/DEPLOYMENT_GUIDE.md`
- `docs/QUICK_REFERENCE.md`
- `docs/BINANCE_DATA_INTERFACE.md`
- `docs/DRL_AND_FINRL.md`
