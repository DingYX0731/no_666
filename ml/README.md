# ml

Modular ML training stack:

- `model_architecture.py` - single-hidden-layer MLP model
- `loss.py` - binary classification loss/metrics
- `features.py` - supervised feature/label generation
- `trainer.py` - trainer abstraction and reporting

Checkpoints are saved in `.npz` format and include:

- model architecture parameters
- model weights
- feature normalization statistics
