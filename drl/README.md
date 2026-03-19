# drl

Deep reinforcement learning utilities aligned with FinRL-style crypto envs (cash, position, fees, portfolio return).

## Files

- `crypto_env.py` — Gymnasium `CryptoSingleAssetEnv` for one symbol
- `train_sb3.py` — train PPO on Binance klines, save per-pair artifacts
- `io_utils.py` — pair slug + JSON meta next to SB3 zip

## Training

```bash
pip install -r requirements.txt
pip install -r requirements-drl.txt
python run_train_drl.py --symbol BTC/USD --timesteps 50000
```

Outputs:

- `checkpoints/drl/BTC_USD_ppo.zip`
- `checkpoints/drl/BTC_USD_meta.json`

## Live / backtest

Use strategy name `drl` and ensure each traded pair has a matching checkpoint (see `configs/strategies/drl.yaml`).
