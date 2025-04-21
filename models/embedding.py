"""Embedding modules for transformer architectures.

This module implements token and positional embeddings for transformer models,
following the architecture described in "Attention is All You Need" (Vaswani et al., 2017).
"""

from typing import Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class TokenEmbedding(eqx.Module):
    """Token embedding layer for transformer models.
    
    This layer maps discrete token indices to continuous vector representations.
    
    Attributes:
        embedding_matrix: Weight matrix of shape (vocab_size, d_model)
        d_model: Dimension of the embedding space
    """
    
    embedding_matrix: jax.Array
    d_model: int

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the token embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the embedding space
            key: Random key for initialization
        """
        self.d_model = d_model
        self.embedding_matrix = jr.normal(key, (vocab_size, d_model)) / jnp.sqrt(d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Embed input tokens.
        
        Args:
            x: Token indices of shape (seq_len,) or (batch_size, seq_len)
            
        Returns:
            Embedded tokens of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
        """
        return jnp.take(self.embedding_matrix, x, axis=0)


class PositionalEmbedding(eqx.Module):
    """Positional embedding layer for transformer models.
    
    This layer adds positional information to token embeddings using sinusoidal
    positional encodings as described in "Attention is All You Need".
    
    Attributes:
        positional_encodings: Matrix of shape (max_seq_len, d_model) containing
            pre-computed positional encodings
    """
    
    positional_encodings: jax.Array

    def __init__(
        self,
        *,
        max_seq_len: int,
        d_model: int
    ) -> None:
        """Initialize the positional embedding layer.
        
        Args:
            max_seq_len: Maximum sequence length
            d_model: Dimension of the embedding space
        """
        # Create position indices
        position = jnp.arange(max_seq_len)[:, None]
        
        # Create frequency terms
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
        
        # Initialize positional encodings
        pe = jnp.zeros((max_seq_len, d_model))
        
        # Compute sine and cosine terms
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        self.positional_encodings = pe

    def __call__(self, x: jax.Array) -> jax.Array:
        """Add positional encodings to input embeddings.
        
        Args:
            x: Input embeddings of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
            
        Returns:
            Position-augmented embeddings of same shape as input
        """
        seq_len = x.shape[-2]
        return x + self.positional_encodings[:seq_len]


class Embedding(eqx.Module):
    """Combined token and positional embedding layer.
    
    This module combines token and positional embeddings into a single layer
    for convenience.
    
    Attributes:
        token_embedding: Token embedding layer
        positional_embedding: Positional embedding layer
    """
    
    token_embedding: TokenEmbedding
    positional_embedding: PositionalEmbedding

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the combined embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the embedding space
            max_seq_len: Maximum sequence length
            key: Random key for initialization
        """
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            key=key
        )
        self.positional_embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            d_model=d_model
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Embed input tokens and add positional information.
        
        Args:
            x: Token indices of shape (seq_len,) or (batch_size, seq_len)
            
        Returns:
            Position-augmented embeddings of shape (seq_len, d_model) or
            (batch_size, seq_len, d_model)
        """
        # Get token embeddings
        token_embeddings = self.token_embedding(x)
        
        # Add positional encodings
        return self.positional_embedding(token_embeddings) 