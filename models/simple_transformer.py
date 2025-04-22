"""Simple transformer implementation using only attention layers.

This module provides a minimal transformer implementation that uses:
- Multi-head attention only (no layer norm)
- Token and positional embeddings
- Linear output layer
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple

from .attention import Attention
from .embedding import Embedding


class SimpleTransformer(eqx.Module):
    """Simple transformer using only attention layers.
    
    This implementation removes layer normalization and feed-forward networks,
    focusing only on the core attention mechanism.
    
    Attributes:
        token_embedding: Embedding layer for input tokens
        pos_embedding: Positional embedding layer
        attention_layers: List of attention layers
        output_proj: Linear projection to vocabulary size
        max_seq_len: Maximum sequence length
    """
    
    embedding: Embedding
    attention_layers: list
    output_proj: eqx.nn.Linear
    max_seq_len: int
    
    def __init__(
        self,
        vocab_size: int,
        n_layers: int = 4,
        n_heads: int = 9,
        d_model: int = 72,
        max_seq_len: int = 10,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """Initialize the simple transformer.
        
        Args:
            vocab_size: Size of the vocabulary
            n_layers: Number of attention layers
            n_heads: Number of attention heads per layer
            d_model: Model dimension (must be divisible by n_heads)
            max_seq_len: Maximum sequence length
            key: Random key for initialization
        """
        super().__init__()
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Split keys for different components
        keys = jax.random.split(key, 4)
        
        # Initialize embeddings
        self.embedding = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            key=keys[0]
        )
        
        # Initialize attention layers
        self.attention_layers = []
        attn_keys = jax.random.split(keys[2], n_layers)
        for i in range(n_layers):
            self.attention_layers.append(
                Attention(
                    d_model=d_model,
                    n_heads=n_heads,
                    key=attn_keys[i]
                )
            )
        
        # Initialize output projection
        self.output_proj = eqx.nn.Linear(
            d_model,
            vocab_size,
            key=keys[3]
        )
        
        self.max_seq_len = max_seq_len
    
    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None
    ) -> jax.Array:
        """Forward pass through the transformer.
        
        Args:
            x: Input tokens of shape (seq_len,)
            mask: Optional attention mask
            
        Returns:
            Logits of shape (seq_len, vocab_size)
        """
        # Get sequence length
        seq_len = len(x)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum "
                f"length {self.max_seq_len}"
            )
        
        # Embed tokens and positions
        h = self.embedding(x)
        
        # Apply attention layers
        for layer in self.attention_layers:
            h = h + layer(h)
        
        # Project to vocabulary size
        logits = jnp.einsum("td, vd -> tv", h, self.embedding.token_embedding.embedding_matrix)
        
        return logits
    
    def attention(
        self,
        x: jax.Array,
        layer_idx: int = -1
    ) -> jax.Array:
        """Get attention weights for a given input sequence.
        
        Args:
            x: Input tokens of shape (seq_len,)
            layer_idx: Layer index to get attention weights from (-1 for last layer)
            
        Returns:
            Attention weights of shape (n_heads, seq_len, seq_len)
        """
        # Embed tokens and positions
        # Embed tokens and positions
        h = self.embedding(x)
        
        # Apply attention layers

        
        # Get attention weights from specified layer
        if layer_idx < 0:
            layer_idx = len(self.attention_layers) + layer_idx
        
        # Apply attention layers up to the specified layer
        i = 0
        for layer in self.attention_layers:
            if i == layer_idx:
                return layer.attention(h)
            i += 1
            h = h + layer(h)
        
        return layer.attention(h)  # Return last layer weights if layer_idx too large 