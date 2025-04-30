"""Train a sparse autoencoder on transformer activations using the SAETrainer class."""

import jax
import jax.numpy as jnp
import jax.random as jr
from training.utils import load
from training.sae_trainer import SAETrainer, SAETrainingConfig
from models.transformer import SimpleTransformer
from data import generate_cyclic_data
import matplotlib.pyplot as plt


# Set random key
key = jr.key(0)

# Load trained transformer model
print("Loading transformer model...")
model = load("cyclic_model/st_model.mo", SimpleTransformer)

# Generate training data
print("Generating training data...")
token_dimension = 3
holding_time = 6
data_multiplier = 100000
data = generate_cyclic_data(
    token_dimension=token_dimension,
    holding_time=holding_time,
    data_multiplier=data_multiplier
)

# Split data into train and test
train_size = int(0.9 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Get model dimensions from a sample forward pass
sample_seq = data[0:10]
sample_activations = model.residual(sample_seq, layer=0)
d_model = sample_activations.shape[-1]  # Get the dimension of the activations

# Configure SAE training
numsteps = [1000, 5000]
for layer_level in range(2):
    config = SAETrainingConfig(
        batch_size=32,
        block_size=10,
        learning_rate=1e-3,
        weight_decay=0.01,
        num_steps=numsteps[layer_level],
        eval_interval=100,
        save_interval=500,
        save_path="cyclic_model/sae_layer_" + str(layer_level) + ".mo",
        plot_reconstruction=False,
        plot_interval=200,
        e_factor=2,
        layer_level=layer_level  # Train on first layer
    )

    # Initialize trainer
    print("Initializing SAE trainer...")
    trainer = SAETrainer(
        transformer_model=model,
        config=config,
        key=key
    )

    # Train the SAE
    print("Starting training...")
    trainer.train_test(train_data, test_data)

    # Plot final results
    print("Plotting results...")
    trainer.plot_losses()

    # Test reconstruction on a sample sequence
    print("\nTesting reconstruction on a sample sequence:")
    sample_activations = model.residual(sample_seq, layer=0)
    trainer.plot_reconstruction(sample_activations)

    # Save final model
    print("\nSaving final model...")
    trainer.save_checkpoint(trainer.config.num_steps)

    print("\nTraining complete!")
