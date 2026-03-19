# configs

Configuration files for pluggable modules.

## ML Training Configs

Path convention:

`configs/ml/<model_name>_train.yaml`

Examples:

- `configs/ml/mlp_train.yaml`
- `configs/ml/drl_train.yaml`

`mlp_train.yaml` supports data-pipeline controls such as:

- `data.use_agg_trades`
- `data.use_trades`
- `data.vol_window`
- `features.lookback`
- `features.horizon`
- `features.label_mode`
- `features.split_method`

## Strategy Configs

Path convention:

`configs/strategies/<strategy_name>.yaml`

Examples:

- `configs/strategies/ma.yaml`
- `configs/strategies/mlp.yaml` (inference config)
- `configs/strategies/drl.yaml` (inference config, per-pair PPO checkpoints under `checkpoints/drl/`)
