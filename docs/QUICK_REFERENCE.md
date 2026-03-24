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

Buy and hold (first tick all-in BUY, then HOLD):

```bash
python run_trader.py --symbols BTC/USD,ETH/USD --strategy buy_hold --strategy-config configs/strategies/buy_hold.yaml --poll-seconds 5
```

Rule-based mean reversion (Bollinger + RSI + trend filter; config: `configs/strategies/bb_rsi.yaml`):

```bash
python run_trader.py --symbols BTC/USD,ETH/USD --strategy bb_rsi --poll-seconds 5
DRY_RUN=false python run_trader.py --symbols BTC/USD,ETH/USD --strategy bb_rsi --poll-seconds 5
```

## Backtest

```bash
python run_backtest.py --data-source synthetic --strategy ma --strategy-config configs/strategies/ma.yaml --save-artifacts
python run_backtest.py --data-source csv --csv your_data.csv --price-col close --strategy ma --strategy-config configs/strategies/ma.yaml --save-artifacts
python run_backtest.py \
  --data-source binance \
  --symbol BTC/USD \
  --interval 1h \
  --frequency daily \
  --market spot \
  --quote-asset USDT \
  --start-date 2024-01-01 \
  --end-date 2024-01-05 \
  --strategy mlp \
  --strategy-config configs/strategies/mlp.yaml \
  --save-artifacts
```

`buy_hold` (buy once with all quote, then HOLD; end equity = MTM at last bar):

```bash
python run_backtest.py --data-source synthetic --strategy buy_hold --strategy-config configs/strategies/buy_hold.yaml --save-artifacts
```

`bb_rsi` (Bollinger Band + RSI mean reversion, no ML):

```bash
python run_backtest.py --data-source synthetic --strategy bb_rsi --strategy-config configs/strategies/bb_rsi.yaml --save-artifacts
python run_backtest.py \
  --data-source binance \
  --symbol BTC/USDT \
  --interval 1m \
  --frequency daily \
  --market spot \
  --quote-asset USDT \
  --start-date 2024-03-01 \
  --end-date 2024-03-15 \
  --strategy bb_rsi \
  --strategy-config configs/strategies/bb_rsi.yaml \
  --save-artifacts
```

Artifacts: `logs/backtest/YYYYMMDD/<run>/steps.csv` + `backtest_chart.png`. See `docs/BACKTEST.md`.

## Data + Training

```bash
python run_fetch_data.py --symbol BTC/USD --dataset klines --interval 1h --frequency daily --start-date 2024-01-01 --end-date 2024-01-03 --preview-rows 3
python run_train_mlp.py --config configs/ml/mlp_train.yaml
python run_train_drl.py --config configs/ml/drl_train.yaml
```
