"""Train a transformer model on Lorenz system data."""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.transformer import SimpleTransformer, Transformer
from training.trainer import Trainer, TrainingConfig
from examples.fixed_point_lorenz.utils import load_lorenz_data

# Set random key
key = jax.random.PRNGKey(0)

# Load data
train_data, val_data = load_lorenz_data()

# Initialize model
print("\nInitializing model...")
model = Transformer(
    token_dimension=3,  # For tokens 0,1,2
    n_heads= 4,
    d_model= 4 * 32,    # n_heads * 16
    layers=4,
    max_tokens=100
)

# Configure training
config = TrainingConfig(
    batch_size=16,
    block_size=100,
    learning_rate=1e-4,
    weight_decay=1e-3,
    num_steps=10001,
    eval_interval=100,
    save_interval=100,
    plot_attention=False,
    plot_interval=100,
    test_interval=100,
    save_path= "lorenz_model/model.mo"
)

# Initialize trainer
print("\nInitializing trainer...")
trainer = Trainer(model, config, key)

# Train model
print("\nStarting training...")
trainer.train_test(train_data, val_data)

# Evaluate model
print("\nEvaluating model...")
metrics = trainer.evaluate(val_data, num_batches=5)

print("\nEvaluation Results:")
print(f"Accuracy: {metrics:.4f}")

trainer.plot_losses()

