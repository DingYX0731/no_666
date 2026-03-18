# Quick Reference

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ROOSTOO_API_KEY="your_api_key"
export ROOSTOO_API_SECRET="your_api_secret"
```

## Operations

```bash
python run_ops.py server-time
python run_ops.py products
python run_ops.py products --detail
python run_ops.py ticker --pair BTC/USD
python run_ops.py balance --pair BTC/USD
python run_ops.py pending-count
python run_ops.py orders --pair BTC/USD --pending-only
```

## Orders

Simulation:

```bash
python run_ops.py place-order --pair BTC/USD --side BUY --type MARKET --quantity 0.001
python run_ops.py cancel-order --pair BTC/USD
```

Live execution:

```bash
python run_ops.py place-order --pair BTC/USD --side BUY --type MARKET --quantity 0.001 --force
python run_ops.py cancel-order --order-id 12345 --force
```

## Trader

```bash
python run_trader.py --symbols BTC/USD --strategy ma --once
python run_trader.py --symbols BTC/USD,ETH/USD --strategy ma
python run_trader.py --symbols BTC/USD --strategy mlp
python run_trader.py --symbols all --strategy ma
```

## Backtest

```bash
python run_backtest.py --data-source synthetic --strategy ma
python run_backtest.py --data-source csv --csv your_data.csv --price-col close
python run_backtest.py --data-source binance --symbol BTC/USD --interval 1h --frequency daily --start-date 2024-01-01 --end-date 2024-01-05 --strategy mlp
```

## Data + Training

```bash
python run_fetch_data.py --symbol BTC/USD --dataset klines --interval 1h --frequency daily --start-date 2024-01-01 --end-date 2024-01-03 --preview-rows 3
python run_train_mlp.py --symbol BTC/USD --interval 1h --start-date 2024-01-01 --end-date 2024-01-15 --ckpt-path checkpoints/mlp/default.npz
```
