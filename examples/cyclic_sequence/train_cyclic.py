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


def generate_cyclic_data(
    token_dimension: int = 3,
    holding_time: int = 6,
    data_multiplier: int = 100000
) -> jax.Array:
    """Generate a cyclic sequence dataset.
    
    Args:
        token_dimension: Number of unique tokens in the sequence
        holding_time: Number of times each token is repeated
        data_multiplier: Number of times to repeat the base sequence
        
    Returns:
        Array containing the cyclic sequence
    """
    # Create base sequence (e.g., 1,2,3,1,2,3... or 1,1,2,2,3,3,1,1,2,2,3,3)
    base_sequence = jnp.repeat(jnp.arange(token_dimension), holding_time)
    
    # Tile the sequence to create a large dataset
    return jnp.tile(base_sequence, data_multiplier)


# Set random seed for reproducibility
key = jr.key(0)

# Generate dataset
print("Generating dataset...")
data = generate_cyclic_data()
print(f"Dataset size: {len(data)} tokens")

# Initialize model
print("\nInitializing model...")
model = Transformer(
    token_dimension=3,  # For tokens 0,1,2
    n_heads= 4,
    d_model= 72,    # n_heads * 8
    layers=4,
    max_tokens=10
)

# Configure training
config = TrainingConfig(
    batch_size=32,
    block_size=10,
    learning_rate=1e-3,
    num_steps=10000,
    eval_interval=100,
    save_interval=1000,
    plot_attention=False,
    plot_interval=100
)

# Initialize trainer
print("\nInitializing trainer...")
trainer = Trainer(model, config, key)

# Train model
print("\nStarting training...")
trainer.train(data)

# Evaluate model
print("\nEvaluating model...")
metrics = trainer.evaluate(data, num_samples=5)

print("\nEvaluation Results:")
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Show example predictions
print("\nExample Predictions:")
for i in range(5):
    print(f"Input:  {metrics['targets'][i]}")
    print(f"Output: {metrics['predictions'][i]}")
    print()

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(trainer.losses)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()