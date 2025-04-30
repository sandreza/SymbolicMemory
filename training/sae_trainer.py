"""Training infrastructure for sparse autoencoders on transformer activations."""

from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from .loss import sae_make_step, sae_loss_fn, sae_batch_loss_function
from .utils import save as save_model, load as load_model
from models.autoencoder import AutoEncoder
from models.transformer import SimpleTransformer


@dataclass
class SAETrainingConfig:
    """Configuration for sparse autoencoder training."""
    batch_size: int = 32
    block_size: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_steps: int = 2000
    eval_interval: int = 100
    save_interval: int = 1000
    save_path: Optional[str] = None
    overwrite_saves: bool = True
    plot_reconstruction: bool = False
    plot_interval: int = 100
    load_path: Optional[str] = None
    test_interval: int = 100
    e_factor: float = 2.0  # Expansion factor for autoencoder
    layer_level: int = 0  # Which transformer layer to train on


class SAETrainer:
    """Sparse autoencoder trainer for transformer activations.
    
    This class handles training a sparse autoencoder on the activations
    of a specific layer in a transformer model.
    """
    
    def __init__(
        self,
        transformer_model: SimpleTransformer,
        sae: Optional[AutoEncoder] = None,
        config: Optional[SAETrainingConfig] = None,
        key: jax.random.PRNGKey = jr.key(0)
    ):
        """Initialize the SAE trainer.
        
        Args:
            transformer_model: The trained transformer model to extract activations from
            sae: Optional sparse autoencoder to train (will create new one if None)
            config: Training configuration
            key: Random key for initialization
        """
        if config is None:
            config = SAETrainingConfig()
        self.config = config
        self.key = key
        self.transformer_model = transformer_model
        
        # Initialize or load SAE
        if config.load_path is not None:
            self.sae = self.load_checkpoint(config.load_path)
        elif sae is not None:
            self.sae = sae
        else:
            # Create new SAE with appropriate dimensions
            d_model = transformer_model.W_E.shape[1]  # Get model dimension
            e_model = int(d_model * config.e_factor)  # Expanded dimension
            self.sae = AutoEncoder(d_model=d_model, e_model=e_model, key=key)
        
        # Initialize optimizer
        self.optimizer = optax.chain(
            optax.adamw(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay
            )
        )
        self.opt_state = self.optimizer.init(eqx.filter(self.sae, eqx.is_array))
        
        # Initialize metrics
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []
        self.test_steps: List[int] = []
        
        # Setup save directory if needed
        if self.config.save_path:
            os.makedirs(os.path.dirname(self.config.save_path), exist_ok=True)

    def load_checkpoint(self, checkpoint_path: str) -> AutoEncoder:
        """Load a sparse autoencoder from a checkpoint file."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            sae = load_model(checkpoint_path, AutoEncoder)
            print(f"Successfully loaded SAE from {checkpoint_path}")
            return sae
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {str(e)}")

    def get_batch(
        self,
        data: jax.Array,
        key: jax.random.PRNGKey
    ) -> Tuple[jax.Array, jax.Array]:
        """Get a batch of transformer activations for training.
        
        Args:
            data: Input data to feed through transformer
            key: Random key for sampling
            
        Returns:
            Tuple of (input_activations, output_activations) where each has shape
            (batch_size, block_size, d_model)
        """
        # Generate random starting indices
        ix = jr.randint(
            key,
            (self.config.batch_size,),
            0,
            len(data) - self.config.block_size - 2
        )
        
        # Create input sequences
        x = jnp.stack([data[i:i+self.config.block_size] for i in ix])
        
        # Get transformer activations for the target layer
        activations = jax.vmap(
            lambda seq: self.transformer_model.residual(seq, layer=self.config.layer_level)
        )(x)
        
        # Return activations in their original shape
        return activations, activations  # Autoencoder targets are the same as inputs

    @eqx.filter_jit
    def make_step(
        self,
        sae: AutoEncoder,
        input_data: jax.Array,
        opt_state: optax.OptState
    ) -> Tuple[jax.Array, AutoEncoder, optax.OptState]:
        """Perform a single training step.
        
        Args:
            sae: Current SAE state
            input_data: Input activations of shape (batch_size, block_size, d_model)
            opt_state: Current optimizer state
            
        Returns:
            Tuple of (loss, updated_sae, updated_opt_state)
        """
        # Reshape input data to (batch_size * block_size, d_model)
        batch_size, block_size, d_model = input_data.shape
        input_data = input_data.reshape(batch_size * block_size, d_model)
        
        return sae_make_step(
            sae,
            input_data,
            self.config.e_factor,
            opt_state,
            self.optimizer.update
        )

    def save_checkpoint(self, step: int) -> None:
        """Save a SAE checkpoint."""
        if not self.config.save_path:
            return
            
        base_path = os.path.splitext(self.config.save_path)[0]
        
        if self.config.overwrite_saves:
            save_path = self.config.save_path
        else:
            save_path = f"{base_path}_step_{step}.pt"
        
        # Get hyperparameters
        hyperparams = {
            "d_model": self.sae.W_E.shape[1],
            "e_model": self.sae.W_E.shape[0],
        }
        
        # Save model
        save_model(save_path, hyperparams, self.sae)
        print(f"Saved SAE checkpoint to {save_path}")

    def train_test(
        self,
        train_data: jax.Array,
        test_data: jax.Array,
        num_steps: Optional[int] = None
    ) -> None:
        """Train the SAE with periodic evaluation on test data."""
        num_steps = num_steps or self.config.num_steps
        
        for step in tqdm(range(num_steps)):
            # Training step
            self.key, subkey = jr.split(self.key)
            input_data, _ = self.get_batch(train_data, subkey)
            loss, self.sae, self.opt_state = self.make_step(
                self.sae, input_data, self.opt_state
            )
            self.train_losses.append(loss)
            
            # Evaluation on test set
            if step % self.config.test_interval == 0:
                test_loss = self.evaluate(test_data)
                self.test_losses.append(test_loss)
                self.test_steps.append(step)
                print(f"Step {step}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}")
                
                if self.config.plot_reconstruction and step % self.config.plot_interval == 0:
                    self.plot_reconstruction(input_data[0])
            
            # Regular evaluation and saving
            elif step % self.config.eval_interval == 0:
                print(f"Step {step}, Train Loss: {loss:.4f}")
            
            # Save checkpoint if configured
            if self.config.save_path and step % self.config.save_interval == 0:
                self.save_checkpoint(step)

    def plot_losses(self) -> None:
        """Plot training and test losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.train_losses)), self.train_losses, label='Train Loss')
        if self.test_losses:
            plt.plot(self.test_steps, self.test_losses, label='Test Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Test Losses')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_reconstruction(
        self,
        input_activation: jax.Array,
        save_path: Optional[str] = None
    ) -> None:
        """Plot original and reconstructed activations.
        
        Args:
            input_activation: Input activation of shape (seq_len, d_model)
            save_path: Optional path to save the plot
        """
        # Get reconstruction for each position in the sequence
        reconstructions = jax.vmap(self.sae)(input_activation)
        
        # Plot original and reconstructed activations
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(input_activation)
        plt.title("Original Activations")
        plt.xlabel("Dimension")
        plt.ylabel("Position")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructions)
        plt.title("Reconstructed Activations")
        plt.xlabel("Dimension")
        plt.ylabel("Position")
        plt.colorbar()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def evaluate(
        self,
        data: jax.Array,
        num_batches: int = 5
    ) -> float:
        """Evaluate SAE on multiple batches of data."""
        losses = []
        for _ in range(num_batches):
            self.key, subkey = jr.split(self.key)
            input_data, _ = self.get_batch(data, subkey)
            loss = self.compute_loss(self.sae, input_data)
            losses.append(loss)
        return jnp.mean(jnp.array(losses))

    @eqx.filter_jit
    def compute_loss(
        self,
        sae: AutoEncoder,
        input_data: jax.Array
    ) -> float:
        """Compute loss for a batch of data."""
        # Ensure input_data has the correct shape
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        return sae_batch_loss_function(sae, input_data, self.config.e_factor) 