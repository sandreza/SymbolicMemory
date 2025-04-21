"""Transformer model implementation.

This module implements the complete transformer architecture, combining attention,
embeddings, and feed-forward networks as described in "Attention is All You Need"
(Vaswani et al., 2017).
"""

from typing import Optional, List, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .attention import Attention
from .embedding import Embedding
from .mlp import MLP


class TransformerLayer(eqx.Module):
    """A single transformer layer.
    
    This module implements one layer of the transformer architecture, consisting of:
    1. Multi-head attention with residual connection and layer normalization
    2. Feed-forward network with residual connection and layer normalization
    
    Attributes:
        attention: Multi-head attention module
        mlp: Feed-forward network module
        ln1: First layer normalization
        ln2: Second layer normalization
    """
    
    attention: Attention
    mlp: MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(
        self,
        *,
        n_heads: int = 4,
        d_model: int = 768,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize a transformer layer.
        
        Args:
            n_heads: Number of attention heads
            d_model: Dimension of the model
            key: Random key for initialization
        """
        keys = jr.split(key, 3)
        
        self.attention = Attention(n_heads=n_heads, d_model=d_model, key=keys[0])
        self.mlp = MLP(d_model=d_model, key=keys[1])
        self.ln1 = eqx.nn.LayerNorm(shape=d_model)
        self.ln2 = eqx.nn.LayerNorm(shape=d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the transformer layer to input sequence.
        
        Args:
            x: Input sequence of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
            
        Returns:
            Output sequence of same shape as input
        """
        # Attention block with residual connection
        attn_out = self.attention(jax.vmap(self.ln1)(x))
        x = x + attn_out
        
        # Feed-forward block with residual connection
        mlp_out = self.mlp(jax.vmap(self.ln2)(x))
        x = x + mlp_out
        
        return x


class Transformer(eqx.Module):
    """Complete transformer model.
    
    This module implements the full transformer architecture, consisting of:
    1. Token and positional embeddings
    2. Stack of transformer layers
    3. Final layer normalization
    
    Attributes:
        embedding: Token and positional embedding module
        layers: List of transformer layers
        ln_final: Final layer normalization
    """
    
    embedding: Embedding
    layers: List[TransformerLayer]
    ln_final: eqx.nn.LayerNorm

    def __init__(
        self,
        *,
        vocab_size: int,
        n_heads: int = 4,
        d_model: int = 768,
        n_layers: int = 6,
        max_seq_len: int = 512,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            n_heads: Number of attention heads
            d_model: Dimension of the model
            n_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            key: Random key for initialization
        """
        keys = jr.split(key, n_layers + 2)
        
        # Initialize embeddings
        self.embedding = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            key=keys[0]
        )
        
        # Initialize transformer layers
        self.layers = [
            TransformerLayer(
                n_heads=n_heads,
                d_model=d_model,
                key=keys[i+1]
            )
            for i in range(n_layers)
        ]
        
        # Initialize final layer normalization
        self.ln_final = eqx.nn.LayerNorm(shape=d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the transformer to input tokens.
        
        Args:
            x: Input tokens of shape (seq_len,) or (batch_size, seq_len)
            
        Returns:
            Output logits of shape (seq_len, vocab_size) or (batch_size, seq_len, vocab_size)
        """
        # Get embeddings
        x = self.embedding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final layer normalization
        x = jax.vmap(self.ln_final)(x)
        
        # Project to vocabulary size
        x = jnp.einsum("td, vd -> tv", x, self.embedding.token_embedding.embedding_matrix)
        
        return x

    def attention(self, x: jax.Array, layer_idx: int = 0) -> jax.Array:
        """Get attention weights for a specific layer.
        
        Args:
            x: Input tokens of shape (seq_len,) or (batch_size, seq_len)
            layer_idx: Index of the layer to get attention weights from
            
        Returns:
            Attention weights of shape (n_heads, seq_len, seq_len)
        """
        # Get embeddings
        x = self.embedding(x)
        
        # Apply layers up to the specified one
        for i in range(layer_idx):
            x = self.layers[i](x)
        
        # Get attention weights from the specified layer
        return self.layers[layer_idx].attention.attention(jax.vmap(self.layers[layer_idx].ln1)(x)) 