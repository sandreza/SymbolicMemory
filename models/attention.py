"""Multi-head attention module for transformer architectures.

This module implements the scaled dot-product attention mechanism with multiple attention heads,
following the architecture described in "Attention is All You Need" (Vaswani et al., 2017).
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class Attention(eqx.Module):
    """Multi-head attention module.
    
    Attributes:
        W_Q: Query weight matrix of shape (n_heads, d_embedding, d_model)
        W_K: Key weight matrix of shape (n_heads, d_embedding, d_model)
        W_V: Value weight matrix of shape (n_heads, d_embedding, d_model)
        W_O: Output weight matrix of shape (d_model, n_heads, d_embedding)
        b_Q: Query bias of shape (n_heads, 1, d_embedding)
        b_K: Key bias of shape (n_heads, 1, d_embedding)
        b_V: Value bias of shape (n_heads, 1, d_embedding)
        b_O: Output bias of shape (1, d_model)
        d_model: Dimension of the model
    """
    
    W_Q: jax.Array
    W_K: jax.Array
    W_V: jax.Array
    W_O: jax.Array
    b_Q: jax.Array
    b_K: jax.Array
    b_V: jax.Array
    b_O: jax.Array
    d_model: int

    def __init__(
        self,
        *,
        n_heads: int = 4,
        d_model: int = 768,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the attention module.
        
        Args:
            n_heads: Number of attention heads
            d_model: Dimension of the model
            key: Random key for initialization
        """
        d_embedding = d_model // n_heads
        self.d_model = d_model
        
        # Split keys for each weight matrix
        keys = jr.split(key, 9)
        i = 1
        
        # Initialize weights with scaled normal distribution
        self.W_Q = jr.normal(keys[i], (n_heads, d_embedding, d_model)) / jnp.sqrt(d_model)
        i += 1
        self.W_K = jr.normal(keys[i], (n_heads, d_embedding, d_model)) / jnp.sqrt(d_model)
        i += 1
        self.W_V = jr.normal(keys[i], (n_heads, d_embedding, d_model)) / jnp.sqrt(d_model)
        i += 1
        self.W_O = jr.normal(keys[i], (d_model, n_heads, d_embedding)) / jnp.sqrt(d_model)
        i += 1
        
        # Initialize biases
        self.b_Q = jr.normal(keys[i], (n_heads, 1, d_embedding)) / jnp.sqrt(d_embedding)
        i += 1
        self.b_K = jr.normal(keys[i], (n_heads, 1, d_embedding)) / jnp.sqrt(d_embedding)
        i += 1
        self.b_V = jr.normal(keys[i], (n_heads, 1, d_embedding)) / jnp.sqrt(d_embedding)
        i += 1
        self.b_O = jr.normal(keys[i], (1, d_model)) / jnp.sqrt(d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Compute attention output for input sequence.
        
        Args:
            x: Input sequence of shape (seq_len, d_model)
            
        Returns:
            Output sequence of shape (seq_len, d_model)
        """
        # Compute query, key, value projections
        Q = jnp.einsum("hed, td -> hte", self.W_Q, x) + self.b_Q
        K = jnp.einsum("hed, td -> hte", self.W_K, x) + self.b_K
        V = jnp.einsum("hed, td -> hte", self.W_V, x) + self.b_V
        
        # Compute attention scores
        QK = jnp.einsum("hte, hse -> hts", Q, K) / jnp.sqrt(self.d_model)
        
        # Apply causal mask
        mask = jnp.triu(0 * QK - jnp.inf, k=1)
        QK = QK + mask
        
        # Compute attention weights
        A = jax.vmap(jax.nn.softmax)(QK)
        
        # Compute output
        Z = jnp.einsum("hse, hts -> the", V, A)
        O = jnp.einsum("dhe, the -> td", self.W_O, Z) + self.b_O
        
        return O

    def attention(self, x: jax.Array) -> jax.Array:
        """Compute attention weights for input sequence.
        
        Args:
            x: Input sequence of shape (seq_len, d_model)
            
        Returns:
            Attention weights of shape (n_heads, seq_len, seq_len)
        """
        Q = jnp.einsum("hed, td -> hte", self.W_Q, x) + self.b_Q
        K = jnp.einsum("hed, td -> hte", self.W_K, x) + self.b_K
        QK = jnp.einsum("hte, hse -> hts", Q, K) / jnp.sqrt(self.d_model)
        mask = jnp.triu(0 * QK - jnp.inf, k=1)
        QK = QK + mask
        A = jax.vmap(jax.nn.softmax)(QK)
        return A

    def value(self, x: jax.Array) -> jax.Array:
        """Compute value projections for input sequence.
        
        Args:
            x: Input sequence of shape (seq_len, d_model)
            
        Returns:
            Value projections of shape (n_heads, seq_len, d_embedding)
        """
        return jnp.einsum("hed, td -> hte", self.W_V, x) + self.b_V

    def z_value(self, x: jax.Array) -> jax.Array:
        """Compute attention-weighted value projections.
        
        Args:
            x: Input sequence of shape (seq_len, d_model)
            
        Returns:
            Attention-weighted values of shape (n_heads, seq_len, d_embedding)
        """
        Q = jnp.einsum("hed, td -> hte", self.W_Q, x) + self.b_Q
        K = jnp.einsum("hed, td -> hte", self.W_K, x) + self.b_K
        V = jnp.einsum("hed, td -> hte", self.W_V, x) + self.b_V
        
        QK = jnp.einsum("hte, hse -> hts", Q, K)
        mask = jnp.triu(0 * QK - jnp.inf, k=1)
        QK = QK + mask
        A = jax.vmap(jax.nn.softmax)(QK)
        Z = jnp.einsum("hse, hts -> the", V, A)
        
        return Z 