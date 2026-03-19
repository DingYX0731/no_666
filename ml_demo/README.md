# ml_demo

Model training demos connected to the reusable data interface.

## Files

- `train_single_layer_mlp_demo.py`: single-hidden-layer MLP binary predictor
- `train_drl_agent_demo.py`: PPO DRL trainer with custom MLP+LSTM extractor

## Purpose

Demonstrate end-to-end wiring:
`Binance data interface -> feature engineering -> model training -> evaluation/deployment`

The demo also exports a deployable checkpoint for live strategy use.

Both demo scripts are yaml-driven:

- `configs/ml/mlp_train.yaml`
- `configs/ml/drl_train.yaml`
