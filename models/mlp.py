"""Feed-forward network module for transformer architectures.

This module implements the position-wise feed-forward network used in transformer models,
following the architecture described in "Attention is All You Need" (Vaswani et al., 2017).
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class MLP(eqx.Module):
    """Position-wise feed-forward network.
    
    This module implements a two-layer feed-forward network with a GELU activation
    function, applied independently to each position in the sequence.
    
    Attributes:
        A_E: First layer weight matrix of shape (e_model, d_model)
        b_E: First layer bias of shape (1, e_model)
        A_UE: Second layer weight matrix of shape (d_model, e_model)
        b_UE: Second layer bias of shape (1, d_model)
    """
    
    A_E: jax.Array
    b_E: jax.Array
    A_UE: jax.Array
    b_UE: jax.Array

    def __init__(
        self,
        *,
        d_model: int = 768,
        e_model: int = 3072,  # Default expansion factor of 4
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the feed-forward network.
        
        Args:
            d_model: Input/output dimension
            e_model: Hidden layer dimension (typically 4x d_model)
            key: Random key for initialization
        """
        # Split keys for each weight matrix
        keys = jr.split(key, 5)
        
        # Initialize first layer weights and bias
        self.A_E = jr.normal(keys[1], (e_model, d_model)) / jnp.sqrt(d_model)
        self.b_E = jr.normal(keys[2], (1, e_model)) / jnp.sqrt(e_model)
        
        # Initialize second layer weights and bias
        self.A_UE = jr.normal(keys[3], (d_model, e_model)) / jnp.sqrt(e_model)
        self.b_UE = jr.normal(keys[4], (1, d_model)) / jnp.sqrt(d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the feed-forward network to input sequence.
        
        Args:
            x: Input sequence of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
            
        Returns:
            Output sequence of same shape as input
        """
        # First layer with GELU activation
        hx = jnp.einsum("td, ed -> te", x, self.A_E) + self.b_E
        hx = jax.nn.gelu(hx)
        
        # Second layer
        h = jnp.einsum("te, de -> td", hx, self.A_UE) + self.b_UE
        
        return h


class SimpleMLP(eqx.Module):
    """Simplified feed-forward network with tied weights.
    
    This is a simpler version of the MLP that uses tied weights between the
    encoder and decoder layers, similar to an autoencoder.
    
    Attributes:
        W_E: Weight matrix of shape (e_model, d_model)
    """
    
    W_E: jax.Array

    def __init__(
        self,
        *,
        d_model: int = 768,
        e_model: int = 768,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the simplified feed-forward network.
        
        Args:
            d_model: Input/output dimension
            e_model: Hidden layer dimension
            key: Random key for initialization
        """
        keys = jr.split(key, 2)
        self.W_E = jr.normal(keys[1], (e_model, d_model)) / jnp.sqrt(d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the simplified feed-forward network to input.
        
        Args:
            x: Input of shape (d_model,) or (batch_size, d_model)
            
        Returns:
            Output of same shape as input
        """
        # Encoder layer with ReLU activation
        hx = jnp.einsum("d, ed -> e", x, self.W_E)
        hx = jax.nn.relu(hx)
        
        # Decoder layer (using same weights)
        h = jnp.einsum("e, ed -> d", hx, self.W_E)
        
        return h

    def encode(self, x: jax.Array) -> jax.Array:
        """Encode input to hidden representation.
        
        Args:
            x: Input of shape (d_model,) or (batch_size, d_model)
            
        Returns:
            Hidden representation of shape (e_model,) or (batch_size, e_model)
        """
        hx = jnp.einsum("d, ed -> e", x, self.W_E)
        hx = jax.nn.relu(hx)
        return hx 