# strategy

Strategy layer based on class inheritance.

## Files

- `base.py`: `BaseStrategy` abstract interface
- `factory.py`: build strategy from yaml
- `ma_strategy.py`: moving-average crossover implementation
- `mlp_strategy.py`: checkpoint-based MLP implementation
- `drl_strategy.py`: Stable-Baselines3 PPO, one checkpoint per product (`pair`)
- `ma_crossover.py`: compatibility wrapper
- `mlp_signal.py`: compatibility wrapper

## Strategy contract

Each strategy must implement:

- `required_prices`
- `generate_signal(prices, position_coin) -> "BUY" | "SELL" | "HOLD"`

Parameters are managed by yaml:

- `configs/strategies/ma.yaml`
- `configs/strategies/mlp.yaml`
