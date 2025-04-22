"""Training infrastructure for transformer models."""

from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

from .loss import batch_loss_fn, attention_entropy


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    block_size: int = 10
    learning_rate: float = 1e-3
    num_steps: int = 2000
    eval_interval: int = 100
    save_interval: int = 1000
    plot_attention: bool = False
    plot_interval: int = 100


class Trainer:
    """Transformer model trainer.
    
    This class handles the training loop, evaluation, and visualization of
    transformer models.
    """
    
    def __init__(
        self,
        model: eqx.Module,
        config: TrainingConfig,
        key: jax.random.PRNGKey = jr.key(0)
    ):
        """Initialize the trainer.
        
        Args:
            model: Transformer model to train
            config: Training configuration
            key: Random key for initialization
        """
        self.model = model
        self.config = config
        self.key = key
        
        # Initialize optimizer
        self.optimizer = optax.chain(
            optax.clip(1.0),
            optax.adamw(
                learning_rate=config.learning_rate,
                weight_decay=0.01
            )
        )
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        
        # Initialize metrics
        self.losses = []
        self.attention_entropies = []

    def get_batch(
        self,
        data: jax.Array,
        key: jax.random.PRNGKey
    ) -> Tuple[jax.Array, jax.Array]:
        """Get a batch of training data.
        
        Args:
            data: Full dataset array
            key: Random key for sampling
            
        Returns:
            Tuple of (input_data, output_data) where each has shape
            (batch_size, block_size)
        """
        # Generate random starting indices
        ix = jr.randint(
            key,
            (self.config.batch_size,),
            0,
            len(data) - self.config.block_size
        )
        
        # Create input and output sequences
        x = jnp.stack([data[i:i+self.config.block_size] for i in ix])
        y = jnp.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        
        return x, y

    @eqx.filter_jit
    def make_step(
        self,
        model: eqx.Module,
        input_data: jax.Array,
        output_data: jax.Array,
        opt_state: optax.OptState
    ) -> Tuple[jax.Array, eqx.Module, optax.OptState]:
        """Perform a single training step.
        
        Args:
            model: Current model state
            input_data: Input tokens of shape (batch_size, block_size)
            output_data: Target tokens of shape (batch_size, block_size)
            opt_state: Current optimizer state
            
        Returns:
            Tuple of (loss, updated_model, updated_opt_state)
        """
        # Define loss function wrapper for gradient computation
        def loss_fn(model):
            logits = jax.vmap(model)(input_data)
            loss, metrics = batch_loss_fn(logits, output_data)
            return loss
        
        # Compute loss and gradients
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        # Update model
        updates, opt_state = self.optimizer.update(
            grads,
            opt_state,
            params=eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        
        return loss, model, opt_state

    def train(
        self,
        data: jax.Array,
        num_steps: Optional[int] = None
    ) -> None:
        """Train the model.
        
        Args:
            data: Training dataset
            num_steps: Optional number of steps to train for (overrides config)
        """
        num_steps = num_steps or self.config.num_steps
        
        for step in tqdm(range(num_steps)):
            # Get batch
            self.key, subkey = jr.split(self.key)
            input_data, output_data = self.get_batch(data, subkey)
            
            # Training step
            loss, self.model, self.opt_state = self.make_step(
                self.model, input_data, output_data, self.opt_state
            )
            
            # Store metrics
            self.losses.append(loss)
            
            # Evaluation and visualization
            if step % self.config.eval_interval == 0:
                print(f"Step {step}, Loss: {loss:.4f}")
                
                if self.config.plot_attention and step % self.config.plot_interval == 0:
                    self.plot_attention(input_data[0])

    def plot_attention(
        self,
        input_seq: jax.Array,
        layer_idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """Plot attention weights for a given input sequence.
        
        Args:
            input_seq: Input sequence of shape (seq_len,)
            layer_idx: Layer index to visualize
            save_path: Optional path to save the plot
        """
        # Get attention weights
        attention_weights = self.model.attention(input_seq, layer_idx)
        n_heads = attention_weights.shape[0]
        
        # Create figure
        sqrt_n = int(jnp.ceil(jnp.sqrt(n_heads)))
        fig, axes = plt.subplots(sqrt_n, sqrt_n, figsize=(sqrt_n, sqrt_n))
        
        # Plot each attention head
        for i in range(n_heads):
            row = i // sqrt_n
            col = i % sqrt_n
            axes[row, col].imshow(attention_weights[i])
            axes[row, col].set_title(f"Head {i}")
            axes[row, col].axis("off")
        
        # Remove empty subplots
        for i in range(n_heads, sqrt_n * sqrt_n):
            row = i // sqrt_n
            col = i % sqrt_n
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def evaluate(
        self,
        data: jax.Array,
        num_samples: int = 5
    ) -> Dict[str, Any]:
        """Evaluate the model on the dataset.
        
        Args:
            data: Evaluation dataset
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.key, subkey = jr.split(self.key)
        input_data, output_data = self.get_batch(data, subkey)
        
        # Get model predictions for the first num_samples sequences
        input_subset = input_data[:num_samples]
        target_subset = output_data[:num_samples]
        
        # Get logits for each sequence
        logits = jax.vmap(self.model)(input_subset)  # shape: (num_samples, seq_len, vocab_size)
        
        # Get predictions for each position in each sequence
        predictions = jnp.argmax(logits, axis=-1)  # shape: (num_samples, seq_len)
        
        # Compute accuracy
        accuracy = jnp.mean(predictions == target_subset)
        
        # Compute attention entropy for first sequence
        attention_weights = self.model.attention(input_subset[0], layer_idx=0)
        entropy = attention_entropy(attention_weights)
        
        return {
            "accuracy": accuracy,
            "attention_entropy": entropy,
            "predictions": predictions,
            "targets": target_subset,
            "logits": logits
        } 