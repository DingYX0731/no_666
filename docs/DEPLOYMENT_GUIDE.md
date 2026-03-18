# Deployment Guide

This guide covers setup, operations, and strategy deployment for `no_666`.

## Setup

```bash
cd /userhome/cs5/u3664760/Ding/no_666
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Use environment variables directly (no `.env` edit required):

```bash
export ROOSTOO_API_KEY="your_api_key"
export ROOSTOO_API_SECRET="your_api_secret"
```

One-line style:

```bash
ROOSTOO_API_KEY="your_api_key" ROOSTOO_API_SECRET="your_api_secret" python run_ops.py server-time
```

## Operations

Use `run_ops.py`:

```bash
python run_ops.py server-time
python run_ops.py products
python run_ops.py ticker --pair BTC/USD
python run_ops.py balance --pair BTC/USD
python run_ops.py orders --pair BTC/USD
```

## Trading

Single-pair:

```bash
python run_trader.py --symbols BTC/USD --once
```

Multi-pair:

```bash
python run_trader.py --symbols BTC/USD,ETH/USD
```

All tradable pairs:

```bash
python run_trader.py --symbols all
```

Strategy config is yaml-driven:

- `configs/strategies/ma.yaml`
- `configs/strategies/mlp.yaml`

Run by strategy name only:

```bash
python run_trader.py --symbols BTC/USD --strategy ma
python run_trader.py --symbols BTC/USD --strategy mlp
```

## Log Files

Each run writes to:

```text
logs/trading/YYYYMMDD/<run_name>/trader.log
```

## Model Deployment

Train and save checkpoint:

```bash
python run_train_mlp.py --symbol BTC/USD --interval 1h --start-date 2024-01-01 --end-date 2024-01-15 --ckpt-path checkpoints/mlp/default.npz
```

Deploy checkpoint in live trading:

```bash
python run_trader.py --symbols BTC/USD --strategy mlp
```
