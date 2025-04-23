"""Loss functions for transformer training."""

import jax
import jax.numpy as jnp
import functools as ft
import equinox as eqx
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


@eqx.filter_jit      
def sae_loss_fn(model, l1_penalty, input):
    output = model(input)
    loss = jnp.mean((output - input) ** 2) + l1_penalty * jnp.mean(jnp.abs(model.hx(input)))
    return loss       

@eqx.filter_jit     
def sae_batch_loss_function(model, l1_penalty, input_data):
    loss_function = ft.partial(sae_loss_fn, model, l1_penalty)
    loss_function = jax.vmap(loss_function)
    return jnp.mean(loss_function(input_data))

@eqx.filter_jit     
def sae_make_step(model, input_data, l1_penalty, opt_state, opt_update):
    loss_function = eqx.filter_value_and_grad(sae_batch_loss_function)
    loss, grads = loss_function(model, l1_penalty, input_data)
    updates, opt_state = opt_update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state