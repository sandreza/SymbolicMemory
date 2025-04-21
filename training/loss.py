"""Loss functions for transformer training."""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any


def cross_entropy_loss(
    logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array = None
) -> jax.Array:
    """Compute cross entropy loss.
    
    Args:
        logits: Model output logits of shape (seq_len, vocab_size)
        targets: Target tokens of shape (seq_len,)
        mask: Optional mask of shape (seq_len,)
        
    Returns:
        Cross entropy loss value
    """
    # Get vocab size
    vocab_size = logits.shape[-1]
    
    # Convert targets to one-hot
    targets_one_hot = jax.nn.one_hot(targets, vocab_size)
    
    # Compute cross entropy
    loss = -jnp.sum(targets_one_hot * jax.nn.log_softmax(logits), axis=-1)
    
    # Apply mask if provided
    if mask is not None:
        loss = loss * mask
    
    return jnp.mean(loss)


def batch_loss_fn(
    logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array = None
) -> Tuple[jax.Array, Dict[str, Any]]:
    """Compute loss for a batch of data.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        targets: Target tokens of shape (batch_size, seq_len)
        mask: Optional mask of shape (batch_size, seq_len)
        
    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Compute loss for each sequence in the batch
    losses = jax.vmap(cross_entropy_loss)(logits, targets, mask)
    
    # Average across batch
    total_loss = jnp.mean(losses)
    
    # Return loss and metrics
    metrics = {
        "loss": total_loss,
        "per_sequence_loss": losses
    }
    
    return total_loss, metrics


def attention_entropy(attention_weights: jax.Array) -> jax.Array:
    """Compute entropy of attention weights.
    
    Args:
        attention_weights: Attention weights of shape (n_heads, seq_len, seq_len)
        
    Returns:
        Average entropy across heads and positions
    """
    # Compute entropy for each head and position
    entropy = -jnp.sum(
        attention_weights * jnp.log(attention_weights + 1e-10),
        axis=-1
    )
    
    # Normalize by log(sequence_length)
    seq_len = attention_weights.shape[-1]
    entropy = entropy / jnp.log(seq_len)
    
    # Average across heads and positions
    return jnp.mean(entropy) 