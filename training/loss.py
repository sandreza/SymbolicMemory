"""Loss functions for transformer training."""

import jax
import jax.numpy as jnp
import functools as ft
import equinox as eqx
import optax
from typing import Tuple, Dict, Any


@eqx.filter_jit      
def loss_fn(model, index_seq, labels):
    logits = model(index_seq)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = loss.mean()
    return loss       

@eqx.filter_jit     
def batch_loss_function(model, input_data, output_data):
    loss_function = ft.partial(loss_fn, model)
    loss_function = jax.vmap(loss_function)
    return jnp.mean(loss_function(input_data, output_data))

@eqx.filter_jit     
def make_step(model, input_data, output_data, opt_state, opt_update):
    loss_function = eqx.filter_value_and_grad(batch_loss_function)
    loss, grads = loss_function(model, input_data, output_data)
    updates, opt_state = opt_update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

@eqx.filter_jit 
def get_batch(data, rng_key, batch_size, block_size):
    """
    Extracts a random batch of input and target data
    Args:
        data: An array of all the data's token ID's.
        rng_key: Random number generator key.
        batch_size: Number of parallel batches.
        block_size: Maximum time length for the token sequence.
    Returns:
        Input token ID's and target token ID's.
    """
    ix = jax.random.randint(key=rng_key, shape=(batch_size, ), minval=0, maxval=len(data) - block_size-2)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@eqx.filter_jit      
def sae_loss_fn(model, l1_penalty, input):
    output = model(input)
    loss = jnp.mean((output - input) ** 2) + l1_penalty * jnp.mean(jnp.abs(model.hx(input)))
    return loss       

@eqx.filter_jit     
def sae_batch_loss_function(model, input_data, l1_penalty):
    """Compute loss for a batch of data.
    
    Args:
        model: The autoencoder model
        input_data: Input data of shape (batch_size, d_model)
        l1_penalty: L1 regularization coefficient
        
    Returns:
        Mean loss over the batch
    """
    # Ensure input_data has at least 2 dimensions
    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)
    
    loss_function = ft.partial(sae_loss_fn, model, l1_penalty)
    loss_function = jax.vmap(loss_function)
    return jnp.mean(loss_function(input_data))

@eqx.filter_jit     
def sae_make_step(model, input_data, l1_penalty, opt_state, opt_update):
    loss_function = eqx.filter_value_and_grad(sae_batch_loss_function)
    loss, grads = loss_function(model, input_data, l1_penalty)
    updates, opt_state = opt_update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state