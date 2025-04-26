"""Example of training a transformer on a cyclic sequence task.

This script demonstrates how to use the training infrastructure to train a transformer
model on a simple cyclic sequence prediction task.
"""

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
from data import generate_cyclic_data


# Set random seed for reproducibility
key = jr.key(0)

# Generate dataset
print("Generating dataset...")
data = generate_cyclic_data(token_dimension = 3, holding_time = 6, data_multiplier = 100000)
print(f"Dataset size: {len(data)} tokens")

# Initialize model
print("\nInitializing model...")
model = SimpleTransformer(
    token_dimension=3,  # For tokens 0,1,2
    n_heads= 9,
    d_model= 9 * 8,    # n_heads * 16
    layers=1,
    max_tokens=100
)

# Configure training
config = TrainingConfig(
    batch_size=128,
    block_size=10,
    learning_rate=1e-4,
    weight_decay=1e-3,
    num_steps=10001,
    eval_interval=100,
    save_interval=100,
    plot_attention=False,
    plot_interval=100,
    test_interval=100,
    save_path= "cyclic_model/st_model.mo"
)

# Initialize trainer
print("\nInitializing trainer...")
trainer = Trainer(model, config, key)

# Train model
print("\nStarting training...")
trainer.train_test(data, data[0:54])

# Evaluate model
print("\nEvaluating model...")
metrics = trainer.evaluate(data, num_batches=5)

print("\nEvaluation Results:")
print(f"Accuracy: {metrics:.4f}")

trainer.plot_losses()