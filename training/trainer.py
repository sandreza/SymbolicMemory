"""Training infrastructure for transformer models."""

from typing import Tuple, Optional, Dict, Any, Type
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

from .loss import batch_loss_function
from .utils import save as save_model, load as load_model


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    block_size: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_steps: int = 2000
    eval_interval: int = 100
    save_interval: int = 1000
    save_path: Optional[str] = None
    overwrite_saves: bool = True
    plot_attention: bool = False
    plot_interval: int = 100
    load_path: Optional[str] = None


class Trainer:
    """Transformer model trainer.
    
    This class handles the training loop, evaluation, visualization, and
    checkpointing of transformer models.
    """
    
    def __init__(
        self,
        model: Optional[eqx.Module] = None,
        config: Optional[TrainingConfig] = None,
        model_class: Optional[Type[eqx.Module]] = None,
        key: jax.random.PRNGKey = jr.key(0)
    ):
        """Initialize the trainer.
        
        The trainer can be initialized in two ways:
        1. With a model instance and config for training from scratch
        2. With a model_class and config containing load_path to resume training
        
        Args:
            model: Transformer model to train (optional if loading from checkpoint)
            config: Training configuration
            model_class: Class of the model to load (required if loading from checkpoint)
            key: Random key for initialization
        """
        if config is None:
            config = TrainingConfig()
        self.config = config
        self.key = key
        
        # Load model from checkpoint if specified
        if config.load_path is not None:
            if model_class is None:
                raise ValueError("model_class must be provided when loading from checkpoint")
            if model is not None:
                print("Warning: model argument will be ignored when loading from checkpoint")
            self.model = self.load_checkpoint(config.load_path, model_class)
        elif model is not None:
            self.model = model
        else:
            raise ValueError("Either model or (model_class and config.load_path) must be provided")
        
        # Initialize optimizer
        self.optimizer = optax.chain(
            optax.adamw(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay
            )
        )
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Initialize metrics
        self.losses = []
        
        # Setup save directory if needed
        if self.config.save_path:
            os.makedirs(os.path.dirname(self.config.save_path), exist_ok=True)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model_class: Type[eqx.Module]
    ) -> eqx.Module:
        """Load a model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model_class: The model class to instantiate
            
        Returns:
            The loaded model instance
            
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
            ValueError: If the checkpoint file is invalid
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            model = load_model(checkpoint_path, model_class)
            print(f"Successfully loaded model from {checkpoint_path}")
            return model
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {str(e)}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class: Type[eqx.Module],
        config: Optional[TrainingConfig] = None,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> "Trainer":
        """Create a trainer instance by loading a model from a checkpoint.
        
        This is a convenience method that creates a new trainer instance
        with a model loaded from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model_class: The model class to instantiate
            config: Optional training configuration
            key: Random key for initialization
            
        Returns:
            A new Trainer instance with the loaded model
        """
        if config is None:
            config = TrainingConfig()
        config.load_path = checkpoint_path
        return cls(model_class=model_class, config=config, key=key)

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
            len(data) - self.config.block_size - 2
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
            loss = batch_loss_function(model, input_data, output_data)
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

    def save_checkpoint(self, step: int) -> None:
        """Save a model checkpoint.
        
        Args:
            step: Current training step
        """
        if not self.config.save_path:
            return
            
        # Get base filename without extension
        base_path = os.path.splitext(self.config.save_path)[0]
        
        # Create filename based on save mode
        if self.config.overwrite_saves:
            save_path = self.config.save_path
        else:
            save_path = f"{base_path}_step_{step}.pt"
        
        # Get hyperparameters
        hyperparams = {
            "n_heads": self.model.Attentions[0].W_Q.shape[0],  # Number of attention heads
            "d_model": self.model.W_E.shape[1],  # Model dimension
            "token_dimension": self.model.W_E.shape[0],  # Vocabulary size
            "layers": len(self.model.Attentions),  # Number of layers
            "max_tokens": self.model.P_E.num_embeddings  # Maximum sequence length
        }
        
        # Save model
        save_model(save_path, hyperparams, self.model)
        print(f"Saved model checkpoint to {save_path}")

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
            
            # Save checkpoint if configured
            if self.config.save_path and step % self.config.save_interval == 0:
                self.save_checkpoint(step)

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
        attention_weights = self.model.attention(input_seq, layer=layer_idx)
        n_heads = attention_weights.shape[0]
        
        if n_heads == 1:
            # Single attention head case
            plt.figure(figsize=(6, 6))
            plt.imshow(attention_weights[0])
            plt.title("Attention Weights")
            plt.axis("off")
        else:
            # Multiple attention heads case
            sqrt_n = int(jnp.ceil(jnp.sqrt(n_heads)))
            fig, axes = plt.subplots(sqrt_n, sqrt_n, figsize=(2*sqrt_n, 2*sqrt_n))
            
            # Make axes indexable for any number of heads
            if sqrt_n == 1:
                axes = np.array([[axes]])
            elif not hasattr(axes[0], '__len__'):
                axes = np.array([axes])
            
            # Plot each attention head
            for i in range(n_heads):
                row = i // sqrt_n
                col = i % sqrt_n
                ax = axes[row, col]
                ax.imshow(attention_weights[i])
                ax.set_title(f"Head {i}")
                ax.axis("off")
            
            # Remove empty subplots
            if n_heads < sqrt_n * sqrt_n:
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
        attention_weights = self.model.attention(input_subset[0], layer=0)
        
        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "targets": target_subset,
            "logits": logits
        } 