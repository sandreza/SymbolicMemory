"""Utility functions for model operations.

This module provides utilities for saving, loading, and generating predictions
from transformer models.
"""

import json
from typing import Any, Dict, TypeVar, Type, Tuple, Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

# Type variable for the model class
ModelT = TypeVar('ModelT', bound=eqx.Module)


def generate_predictions(
    model: eqx.Module,
    initial_seq: jax.Array,
    max_new_tokens: int,
    block_size: int,
    key: jax.random.PRNGKey,
    temperature: float = 1.0,
    batch_size: int = 1
) -> jax.Array:
    """Generate predictions from a transformer model.
    
    Args:
        model: The transformer model to use for generation
        initial_seq: Initial sequence to condition on, can be 1D or 2D array
        max_new_tokens: Number of new tokens to generate
        block_size: Maximum context length to use for prediction
        key: Random key for sampling
        temperature: Sampling temperature (higher = more random)
        batch_size: Number of sequences to generate in parallel
    
    Returns:
        Array of shape (batch_size, max_seq_len + max_new_tokens) containing
        the original sequence (padded if necessary) followed by generated tokens
    """
    # Convert initial sequence to jax array if needed
    if isinstance(initial_seq, list):
        # Handle list of sequences by stacking them
        max_len = max(len(seq) for seq in initial_seq)
        padded_seqs = []
        for seq in initial_seq:
            seq = jnp.asarray(seq)
            if len(seq.shape) == 0:
                seq = seq.reshape(1)
            if len(seq) < max_len:
                padding = jnp.full((max_len - len(seq),), seq[-1])
                seq = jnp.concatenate([seq, padding])
            padded_seqs.append(seq)
        initial_seq = jnp.stack(padded_seqs)
    else:
        initial_seq = jnp.asarray(initial_seq)
    
    # Handle input shapes
    if initial_seq.ndim == 0:
        initial_seq = initial_seq.reshape(1, 1)
    elif initial_seq.ndim == 1:
        initial_seq = initial_seq.reshape(1, -1)
    elif initial_seq.ndim > 2:
        raise ValueError(f"Initial sequence must be 1D or 2D, got shape {initial_seq.shape}")
    
    # If we have multiple sequences of different lengths, pad to the longest
    if initial_seq.shape[0] > 1:
        max_seq_len = initial_seq.shape[1]
        # Create a mask for valid positions
        seq_lens = jnp.sum(initial_seq != initial_seq[:, -1:], axis=1) + 1
        # Pad shorter sequences with their last token
        for i in range(initial_seq.shape[0]):
            if seq_lens[i] < max_seq_len:
                padding = jnp.full((max_seq_len - seq_lens[i],), initial_seq[i, seq_lens[i]-1])
                initial_seq = initial_seq.at[i, seq_lens[i]:].set(padding)
    
    # Handle batch size
    if batch_size > 1 and initial_seq.shape[0] == 1:
        initial_seq = jnp.repeat(initial_seq, batch_size, axis=0)
    elif batch_size > initial_seq.shape[0]:
        # If we want more batches than provided sequences, repeat the existing ones
        repeats = batch_size // initial_seq.shape[0] + 1
        initial_seq = jnp.repeat(initial_seq, repeats, axis=0)[:batch_size]
    elif batch_size < initial_seq.shape[0]:
        # If we want fewer batches, take the first batch_size sequences
        initial_seq = initial_seq[:batch_size]
    
    # Setup for batched generation
    index_seq = initial_seq
    vocab_size = model.W_E.shape[0]  # Get vocabulary size from embedding matrix
    batched_choice = jax.vmap(jax.random.choice)
    
    # Generate tokens
    for _ in range(max_new_tokens):
        # Crop sequence to block_size
        index_cond = index_seq[:, -block_size:]
        
        # Get model predictions
        logits = jax.vmap(model)(index_cond)
        
        # Focus on last timestep and apply temperature
        logits = logits[:, -1, :] / temperature
        
        # Convert to probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Sample next tokens
        key, *choice_keys = jax.random.split(key, batch_size + 1)
        choice_keys = jnp.array(choice_keys)
        
        # Create vocabulary indices array once
        vocab_indices = jnp.arange(vocab_size).reshape(1, -1)
        vocab_indices = jnp.repeat(vocab_indices, batch_size, axis=0)
        
        # Sample from probability distribution
        next_tokens = batched_choice(
            choice_keys,
            vocab_indices,
            p=probs
        )
        next_tokens = next_tokens.reshape(batch_size, 1)
        
        # Append new tokens
        index_seq = jnp.concatenate([index_seq, next_tokens], axis=1)
    
    return index_seq


def save(
    filename: str,
    hyperparams: Dict[str, Any],
    model: eqx.Module
) -> None:
    """Save a model and its hyperparameters to a file.
    
    The function saves both the model's hyperparameters as JSON and the model's
    parameters using equinox's serialization. The hyperparameters are stored
    as a JSON string in the first line, followed by the serialized model parameters.
    
    Args:
        filename: Path where the model should be saved
        hyperparams: Dictionary of model hyperparameters
        model: The equinox model to save
    """
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(
    filename: str,
    model_name: Type[ModelT]
) -> ModelT:
    """Load a model and its hyperparameters from a file.
    
    The function reconstructs a model by:
    1. Reading and parsing the hyperparameters from the first line
    2. Creating a new model instance with these hyperparameters
    3. Loading the saved parameters into the model
    
    Args:
        filename: Path to the saved model file
        model_name: The model class to instantiate
        
    Returns:
        A model instance with the loaded parameters
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_name(key=jr.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)


def load_hyperparameters(
    filename: str
) -> Dict[str, Any]:
    """Load only the hyperparameters from a saved model file.
    
    This is useful when you need to inspect or modify the hyperparameters
    without loading the full model.
    
    Args:
        filename: Path to the saved model file
        
    Returns:
        Dictionary containing the model's hyperparameters
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        return hyperparams