"""
SymbolicMemory - A JAX-based implementation of symbolic memory and transformer models.
"""

__version__ = "0.1.0"

# Import main components for easier access
from .models.transformer import Transformer
from .models.autoencoder import SparseAutoencoder
from .models.attention import Attention
from .models.mlp import MLP

# Import training components
from .training.trainer import Trainer
from .training.sae_trainer import SAETrainer
from .training.loss import compute_loss

__all__ = [
    "Transformer",
    "SparseAutoencoder",
    "Attention",
    "MLP",
    "Trainer",
    "SAETrainer",
    "compute_loss",
] 