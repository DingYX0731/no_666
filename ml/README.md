# ml

Modular ML training stack:

- `model_architecture.py` - single-hidden-layer MLP model
- `loss.py` - binary classification loss/metrics
- `features.py` - supervised feature/label generation
- `trainer.py` - trainer abstraction and reporting
- `drl_env.py` - FinRL-style single-asset Gymnasium env
- `drl_model_architecture.py` - custom MLP+LSTM SB3 feature extractor
- `drl_trainer.py` - PPO trainer pipeline
- `drl_utils.py` - per-pair checkpoint metadata helpers

Checkpoints are saved in `.npz` format and include:

- model architecture parameters
- model weights
- feature normalization statistics

DRL checkpoints are saved under `checkpoints/drl/` as:

- `<PAIR>_ppo.zip`
- `<PAIR>_meta.json`
