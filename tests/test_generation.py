"""Tests for model generation functionality.

This module tests different ways of using the generate_predictions function,
including various batch sizes, sequence lengths, and input shapes.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from models.transformer import SimpleTransformer
from models.utils import generate_predictions


@pytest.fixture
def small_model():
    """Create a small transformer model for testing."""
    key = jr.PRNGKey(0)
    model = SimpleTransformer(
        key=key,
        n_heads=2,            # Small number of heads
        d_model=8,            # Small model dimension
        token_dimension=4,    # Small vocabulary
        layers=1,             # Single layer
        max_tokens=16         # Small context size
    )
    return model


def test_single_token_generation(small_model):
    """Test generating from a single token."""
    initial_seq = jnp.array([1])  # Single token
    output = generate_predictions(
        model=small_model,
        initial_seq=initial_seq,
        max_new_tokens=5,
        block_size=8,
        key=jr.PRNGKey(0),
        batch_size=1
    )
    assert output.shape == (1, 6)  # 1 initial + 5 generated
    assert (output[:, 0] == initial_seq).all()


def test_sequence_generation(small_model):
    """Test generating from a sequence of tokens."""
    initial_seq = jnp.array([1, 2, 3])
    output = generate_predictions(
        model=small_model,
        initial_seq=initial_seq,
        max_new_tokens=4,
        block_size=8,
        key=jr.PRNGKey(0),
        batch_size=1
    )
    assert output.shape == (1, 7)  # 3 initial + 4 generated
    assert (output[0, :3] == initial_seq).all()


def test_batched_generation(small_model):
    """Test generating multiple sequences in parallel."""
    initial_seq = jnp.array([1])
    output = generate_predictions(
        model=small_model,
        initial_seq=initial_seq,
        max_new_tokens=3,
        block_size=8,
        key=jr.PRNGKey(0),
        batch_size=4  # Generate 4 different sequences
    )
    assert output.shape == (4, 4)  # 4 sequences, each with 1 initial + 3 generated
    # Check that all sequences start with the same token
    assert (output[:, 0] == initial_seq[0]).all()
    # Check that sequences are different (due to different random sampling)
    assert not (output[0] == output[1]).all()


def test_batched_sequence_generation(small_model):
    """Test generating multiple sequences from a longer initial sequence."""
    initial_seq = jnp.array([1, 2, 3])
    output = generate_predictions(
        model=small_model,
        initial_seq=initial_seq,
        max_new_tokens=2,
        block_size=8,
        key=jr.PRNGKey(0),
        batch_size=3  # Generate 3 different continuations
    )
    assert output.shape == (3, 5)  # 3 sequences, each with 3 initial + 2 generated
    # Check that all sequences start with the same tokens
    assert (output[:, :3] == initial_seq).all()
    # Check that continuations are different
    assert not (output[0, 3:] == output[1, 3:]).all()


def test_long_context_generation(small_model):
    """Test generating with a long context that needs to be cropped."""
    initial_seq = jnp.arange(12) % 4  # Length 12 sequence
    block_size = 8
    output = generate_predictions(
        model=small_model,
        initial_seq=initial_seq,
        max_new_tokens=3,
        block_size=block_size,  # Should only use last 8 tokens
        key=jr.PRNGKey(0),
        batch_size=1
    )
    assert output.shape == (1, 15)  # 12 initial + 3 generated
    assert (output[0, :12] == initial_seq).all()


def test_temperature_effect(small_model):
    """Test that different temperatures produce different distributions."""
    initial_seq = jnp.array([1])
    key = jr.PRNGKey(0)
    
    # Generate with high temperature (more random)
    high_temp = generate_predictions(
        model=small_model,
        initial_seq=initial_seq,
        max_new_tokens=20,
        block_size=8,
        key=key,
        temperature=2.0,
        batch_size=1
    )
    
    # Generate with low temperature (more focused)
    key, _ = jr.split(key)
    low_temp = generate_predictions(
        model=small_model,
        initial_seq=initial_seq,
        max_new_tokens=20,
        block_size=8,
        key=key,
        temperature=0.1,
        batch_size=1
    )
    
    # High temperature should have more token variety
    high_unique = len(jnp.unique(high_temp))
    low_unique = len(jnp.unique(low_temp))
    assert high_unique >= low_unique


def test_input_validation(small_model):
    """Test that invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError):
        # Test with 3D input
        invalid_input = jnp.ones((2, 3, 4))
        generate_predictions(
            model=small_model,
            initial_seq=invalid_input,
            max_new_tokens=5,
            block_size=8,
            key=jr.PRNGKey(0)
        ) 


def test_batched_same_lengths(small_model):
    """Test generating from batched sequences of same lengths."""
    # Create two sequences of same lengths
    seq1 = jnp.array([1, 2, 3, 1])  # Length 4
    seq2 = jnp.array([2, 1, 3, 1])        # Length 2
    initial_seqs = jnp.array([seq1, seq2])
    
    output = generate_predictions(
        model=small_model,
        initial_seq=initial_seqs,
        max_new_tokens=3,
        block_size=8,
        key=jr.PRNGKey(0),
        batch_size=2
    )
    
    # Check output shape (should be batch_size, max_input_len + new_tokens)
    assert output.shape == (2, 7)  # max_input_len=4, new_tokens=3


