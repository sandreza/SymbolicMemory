"""Transformer model implementations.

This module provides several transformer architectures with varying levels of complexity:
- SimpleTransformer: Basic transformer with attention only
- SimpleTransformer2: Variant with separate input/output embeddings
- SimpleTransformerMLP: Adds MLP layers after attention
- Transformer: Full implementation with layer normalization

All implementations follow the core ideas from "Attention is All You Need" (Vaswani et al., 2017)
but offer different architectural choices for experimentation and analysis.
"""

from typing import Optional, List, Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Int

from .attention import Attention
from .mlp import MLP

class SimpleTransformer(eqx.Module):
    """Basic transformer implementation using only attention layers.
    
    This is the simplest variant that uses:
    - Token embeddings (W_E) for both input embedding and output projection
    - Positional embeddings (P_E)
    - Multi-head attention layers without layer normalization or MLPs
    
    Attributes:
        W_E: Token embedding matrix of shape (token_dimension, d_model)
        P_E: Positional embedding layer
        Attentions: List of attention layers
    """
    
    W_E: Float[Array, "token_dim model_dim"]
    P_E: eqx.nn.Embedding
    Attentions: List[Attention]

    def __init__(
        self, 
        *, 
        n_heads: int = 4,
        d_model: int = 768,
        token_dimension: int = 32,
        layers: int = 1,
        max_tokens: int = 100,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the simple transformer.
        
        Args:
            n_heads: Number of attention heads per layer
            d_model: Hidden dimension of the model
            token_dimension: Size of the token vocabulary
            layers: Number of attention layers
            max_tokens: Maximum sequence length for positional embeddings
            key: Random key for initialization
        """
        key, subkey, subkey2 = jr.split(key, 3)
        self.W_E = jr.normal(subkey, shape=(token_dimension, d_model)) / jnp.sqrt(2 * d_model)
        self.P_E = eqx.nn.Embedding(num_embeddings=max_tokens, embedding_size=d_model, key=subkey2)
        self.Attentions = []
        keys = jr.split(subkey, layers + 1)
        for j in range(layers):
            self.Attentions.append(Attention(n_heads=n_heads, d_model=d_model, key=keys[j+1]))

    def __call__(
        self,
        x: Int[Array, "seq_len"]
    ) -> Float[Array, "token_dim seq_len"]:
        """Forward pass through the transformer.
        
        Args:
            x: Input token indices of shape (seq_len,)
            
        Returns:
            Logits over vocabulary of shape (token_dimension, seq_len)
        """
        t = self.W_E[x]  # Token embeddings
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))  # Positional embeddings
        t = t + p  # Combined embeddings
        
        # Apply attention layers sequentially
        for j in range(len(self.Attentions)):
            t = t + self.Attentions[j](t)
            
        # Project back to vocabulary space
        o = jnp.einsum("td,nd->nt", self.W_E, t)
        return o

    def attention(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0
    ) -> Float[Array, "n_heads seq_len seq_len"]:
        """Get attention weights from a specific layer.
        
        Args:
            x: Input token indices
            layer: Index of the attention layer to inspect
            
        Returns:
            Attention weights from the specified layer
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        # Apply layers up to the target layer
        for j in range(layer-1):
            t = t + self.Attentions[j](t)
            
        return self.Attentions[layer].attention(t)

    def residual(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0
    ) -> Float[Array, "seq_len model_dim"]:
        """Get intermediate representations up to a specific layer.
        
        Args:
            x: Input token indices
            layer: Layer index to stop at
            
        Returns:
            Hidden states at the specified layer
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        # Apply layers up to the target layer
        for j in range(layer):
            t = t + self.Attentions[j](t)
            
        return t

    def intervention(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0,
        intervention: Callable[[Float[Array, "seq_len model_dim"]], 
                            Float[Array, "seq_len model_dim"]] = lambda a: a
    ) -> Float[Array, "token_dim seq_len"]:
        """Apply an intervention to hidden states at a specific layer.
        
        This allows analyzing the model's behavior by modifying intermediate
        representations at any layer.
        
        Args:
            x: Input token indices
            layer: Layer at which to apply the intervention
            intervention: Function that modifies hidden states
            
        Returns:
            Model output after applying the intervention
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        # Apply layers up to intervention point
        for j in range(layer):
            t = t + self.Attentions[j](t)
            
        # Apply intervention
        t = intervention(t)
        
        # Continue with remaining layers
        for j in range(layer, len(self.Attentions)):
            t = t + self.Attentions[j](t)
            
        o = jnp.einsum("td,nd->nt", self.W_E, t)
        return o

class SimpleTransformer2(eqx.Module):
    """Transformer variant with separate input and output embeddings.
    
    This implementation differs from SimpleTransformer by using:
    - Separate embedding matrices for input (W_E) and output (W_UE) projections
    - Positional embeddings (P_E)
    - Multi-head attention layers without layer normalization or MLPs
    
    This separation can be beneficial when input and output vocabularies differ
    or when asymmetric embedding spaces are desired.
    
    Attributes:
        W_E: Input embedding matrix of shape (token_dimension, d_model)
        W_UE: Output embedding matrix of shape (token_dimension, d_model)
        P_E: Positional embedding layer
        Attentions: List of attention layers
    """
    
    W_E: Float[Array, "token_dim model_dim"]
    W_UE: Float[Array, "token_dim model_dim"]
    P_E: eqx.nn.Embedding
    Attentions: List[Attention]

    def __init__(
        self,
        *,
        n_heads: int = 4,
        d_model: int = 768,
        token_dimension: int = 32,
        layers: int = 1,
        max_tokens: int = 100,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the transformer with separate embeddings.
        
        Args:
            n_heads: Number of attention heads per layer
            d_model: Hidden dimension of the model
            token_dimension: Size of the token vocabulary
            layers: Number of attention layers
            max_tokens: Maximum sequence length for positional embeddings
            key: Random key for initialization
        """
        key, subkey, subkey2, subkey3 = jr.split(key, 4)
        self.W_E = jr.normal(subkey, shape=(token_dimension, d_model)) / jnp.sqrt(2 * d_model)
        self.W_UE = jr.normal(subkey3, shape=(token_dimension, d_model)) / jnp.sqrt(2 * d_model)
        self.P_E = eqx.nn.Embedding(num_embeddings=max_tokens, embedding_size=d_model, key=subkey2)
        self.Attentions = []
        keys = jr.split(subkey, layers + 1)
        for j in range(layers):
            self.Attentions.append(Attention(n_heads=n_heads, d_model=d_model, key=keys[j+1]))

    def __call__(
        self,
        x: Int[Array, "seq_len"]
    ) -> Float[Array, "token_dim seq_len"]:
        """Forward pass through the transformer.
        
        Args:
            x: Input token indices of shape (seq_len,)
            
        Returns:
            Logits over vocabulary of shape (token_dimension, seq_len)
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        for j in range(len(self.Attentions)):
            t = t + self.Attentions[j](t)
            
        # Use separate output embedding
        o = jnp.einsum("td,nd->nt", self.W_UE, t)
        return o

    def attention(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0
    ) -> Float[Array, "n_heads seq_len seq_len"]:
        """Get attention weights from a specific layer.
        
        Args:
            x: Input token indices
            layer: Index of the attention layer to inspect
            
        Returns:
            Attention weights from the specified layer
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        # Apply layers up to the target layer
        for j in range(layer-1):
            t = t + self.Attentions[j](t)
            
        return self.Attentions[layer].attention(t)

    def residual(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0
    ) -> Float[Array, "seq_len model_dim"]:
        """Get intermediate representations up to a specific layer.
        
        Args:
            x: Input token indices
            layer: Layer index to stop at
            
        Returns:
            Hidden states at the specified layer
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        # Apply layers up to the target layer
        for j in range(layer):
            t = t + self.Attentions[j](t)
            
        return t

    def intervention(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0,
        intervention: Callable[[Float[Array, "seq_len model_dim"]], 
                            Float[Array, "seq_len model_dim"]] = lambda a: a
    ) -> Float[Array, "token_dim seq_len"]:
        """Apply an intervention to hidden states at a specific layer.
        
        This allows analyzing the model's behavior by modifying intermediate
        representations at any layer.
        
        Args:
            x: Input token indices
            layer: Layer at which to apply the intervention
            intervention: Function that modifies hidden states
            
        Returns:
            Model output after applying the intervention
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        # Apply layers up to intervention point
        for j in range(layer):
            t = t + self.Attentions[j](t)
            
        # Apply intervention
        t = intervention(t)
        
        # Continue with remaining layers
        for j in range(layer, len(self.Attentions)):
            t = t + self.Attentions[j](t)
            
        o = jnp.einsum("td,nd->nt", self.W_E, t)
        return o

class SimpleTransformerMLP(eqx.Module):
    """Transformer implementation with attention and feed-forward layers.
    
    This variant extends SimpleTransformer by adding:
    - MLP layers after each attention layer
    - Residual connections around both attention and MLP blocks
    
    This architecture more closely follows the original transformer design
    but still omits layer normalization for simplicity.
    
    Attributes:
        W_E: Token embedding matrix of shape (token_dimension, d_model)
        P_E: Positional embedding layer
        Attentions: List of attention layers
        MLPs: List of feed-forward networks
    """
    
    W_E: Float[Array, "token_dim model_dim"]
    P_E: eqx.nn.Embedding
    Attentions: List[Attention]
    MLPs: List[MLP]

    def __init__(
        self,
        *,
        n_heads: int = 4,
        d_model: int = 768,
        token_dimension: int = 32,
        layers: int = 1,
        max_tokens: int = 100,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the transformer with MLPs.
        
        Args:
            n_heads: Number of attention heads per layer
            d_model: Hidden dimension of the model
            token_dimension: Size of the token vocabulary
            layers: Number of transformer layers
            max_tokens: Maximum sequence length for positional embeddings
            key: Random key for initialization
        """
        key, subkey, subkey2 = jr.split(key, 3)
        self.W_E = jr.normal(subkey, shape=(token_dimension, d_model)) / jnp.sqrt(2 * d_model)
        self.P_E = eqx.nn.Embedding(num_embeddings=max_tokens, embedding_size=d_model, key=subkey2)
        self.Attentions = []
        self.MLPs = []
        keys = jr.split(subkey, layers + 1)
        keys_mlp = jr.split(keys[0], layers + 1)
        for j in range(layers):
            self.Attentions.append(Attention(n_heads=n_heads, d_model=d_model, key=keys[j+1]))
            self.MLPs.append(MLP(d_model=d_model, e_model=4 * d_model, key=keys_mlp[j+1]))

    def __call__(
        self,
        x: Int[Array, "seq_len"]
    ) -> Float[Array, "token_dim seq_len"]:
        """Forward pass through the transformer.
        
        Args:
            x: Input token indices of shape (seq_len,)
            
        Returns:
            Logits over vocabulary of shape (token_dimension, seq_len)
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        for j in range(len(self.Attentions)):
            t = t + self.Attentions[j](t)
            t = t + self.MLPs[j](t)
            
        o = jnp.einsum("td,nd->nt", self.W_E, t)
        return o

    def attention(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0
    ) -> Float[Array, "n_heads seq_len seq_len"]:
        """Get attention weights from a specific layer.
        
        Args:
            x: Input token indices
            layer: Index of the attention layer to inspect
            
        Returns:
            Attention weights from the specified layer
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        for j in range(layer-1):
            t = t + self.Attentions[j](t)
            t = t + self.MLPs[j](t)
            
        return self.Attentions[layer].attention(t)

    def residual(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0
    ) -> Float[Array, "seq_len model_dim"]:
        """Get intermediate representations up to a specific layer.
        
        Args:
            x: Input token indices
            layer: Layer index to stop at
            
        Returns:
            Hidden states at the specified layer
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        for j in range(layer):
            t = t + self.Attentions[j](t)
            t = t + self.MLPs[j](t)
            
        return t

    def intervention(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0,
        intervention: Callable[[Float[Array, "seq_len model_dim"]], 
                            Float[Array, "seq_len model_dim"]] = lambda a: a
    ) -> Float[Array, "token_dim seq_len"]:
        """Apply an intervention to hidden states at a specific layer.
        
        This allows analyzing the model's behavior by modifying intermediate
        representations at any layer.
        
        Args:
            x: Input token indices
            layer: Layer at which to apply the intervention
            intervention: Function that modifies hidden states
            
        Returns:
            Model output after applying the intervention
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        # Apply layers up to intervention point
        for j in range(layer):
            t = t + self.Attentions[j](t)
            t = t + self.MLPs[j](t)
            
        # Apply intervention
        t = intervention(t)
        
        # Continue with remaining layers
        for j in range(layer, len(self.Attentions)):
            t = t + self.Attentions[j](t)
            t = t + self.MLPs[j](t)
            
        o = jnp.einsum("td,nd->nt", self.W_E, t)
        return o

class Transformer(eqx.Module):
    """Full transformer implementation with layer normalization.
    
    This is the complete transformer architecture including:
    - Token and positional embeddings
    - Multi-head attention with layer normalization
    - Feed-forward networks with layer normalization
    - Residual connections around both attention and MLP blocks
    
    This implementation closely follows the original transformer paper,
    making it suitable for more complex sequence modeling tasks.
    
    Attributes:
        W_E: Token embedding matrix of shape (token_dimension, d_model)
        P_E: Positional embedding layer
        Attentions: List of attention layers
        MLPs: List of feed-forward networks
        LNs: List of layer norms before attention
        LN2s: List of layer norms before MLPs
    """
    
    W_E: Float[Array, "token_dim model_dim"]
    P_E: eqx.nn.Embedding
    Attentions: List[Attention]
    MLPs: List[MLP]
    LNs: List[eqx.nn.LayerNorm]
    LN2s: List[eqx.nn.LayerNorm]

    def __init__(
        self,
        *,
        n_heads: int = 4,
        d_model: int = 768,
        token_dimension: int = 32,
        layers: int = 1,
        max_tokens: int = 100,
        key: jax.random.PRNGKey = jr.key(0)
    ) -> None:
        """Initialize the full transformer.
        
        Args:
            n_heads: Number of attention heads per layer
            d_model: Hidden dimension of the model
            token_dimension: Size of the token vocabulary
            layers: Number of transformer layers
            max_tokens: Maximum sequence length for positional embeddings
            key: Random key for initialization
        """
        key, subkey, subkey2 = jr.split(key, 3)
        self.W_E = jr.normal(subkey, shape=(token_dimension, d_model)) / jnp.sqrt(2 * d_model)
        self.P_E = eqx.nn.Embedding(num_embeddings=max_tokens, embedding_size=d_model, key=subkey2)
        self.Attentions = []
        self.MLPs = []
        self.LNs = []
        self.LN2s = []
        keys = jr.split(subkey, layers + 1)
        keys_mlp = jr.split(keys[0], layers + 1)
        for j in range(layers):
            self.Attentions.append(Attention(n_heads=n_heads, d_model=d_model, key=keys[j+1]))
            self.MLPs.append(MLP(d_model=d_model, e_model=4 * d_model, key=keys_mlp[j+1]))
            self.LNs.append(eqx.nn.LayerNorm(shape=d_model))
            self.LN2s.append(eqx.nn.LayerNorm(shape=d_model))

    def __call__(
        self,
        x: Int[Array, "seq_len"]
    ) -> Float[Array, "token_dim seq_len"]:
        """Forward pass through the transformer.
        
        Args:
            x: Input token indices of shape (seq_len,)
            
        Returns:
            Logits over vocabulary of shape (token_dimension, seq_len)
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        for j in range(len(self.Attentions)):
            t = t + self.Attentions[j](jax.vmap(self.LNs[j])(t))
            t = t + self.MLPs[j](jax.vmap(self.LN2s[j])(t))
            
        o = jnp.einsum("td,nd->nt", self.W_E, t)
        return o

    def attention(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0
    ) -> Float[Array, "n_heads seq_len seq_len"]:
        """Get attention weights from a specific layer.
        
        Args:
            x: Input token indices
            layer: Index of the attention layer to inspect
            
        Returns:
            Attention weights from the specified layer
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        for j in range(layer-1):
            t = t + self.Attentions[j](jax.vmap(self.LNs[j])(t))
            t = t + self.MLPs[j](jax.vmap(self.LN2s[j])(t))
            
        return self.Attentions[layer].attention(jax.vmap(self.LNs[layer])(t))

    def residual(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0
    ) -> Float[Array, "seq_len model_dim"]:
        """Get intermediate representations up to a specific layer.
        
        Args:
            x: Input token indices
            layer: Layer index to stop at
            
        Returns:
            Hidden states at the specified layer
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        for j in range(layer):
            t = t + self.Attentions[j](jax.vmap(self.LNs[j])(t))
            t = t + self.MLPs[j](jax.vmap(self.LN2s[j])(t))
            
        return t

    def intervention(
        self,
        x: Int[Array, "seq_len"],
        *,
        layer: int = 0,
        intervention: Callable[[Float[Array, "seq_len model_dim"]],
                           Float[Array, "seq_len model_dim"]] = lambda a: a
    ) -> Float[Array, "token_dim seq_len"]:
        """Apply an intervention to hidden states at a specific layer.
        
        This allows analyzing the model's behavior by modifying intermediate
        representations at any layer.
        
        Args:
            x: Input token indices
            layer: Layer at which to apply the intervention
            intervention: Function that modifies hidden states
            
        Returns:
            Model output after applying the intervention
        """
        t = self.W_E[x]
        p = jax.vmap(self.P_E)(jnp.arange(len(x)))
        t = t + p
        
        for j in range(layer):
            t = t + self.Attentions[j](jax.vmap(self.LNs[j])(t))
            t = t + self.MLPs[j](jax.vmap(self.LN2s[j])(t))
            
        t = intervention(t)
        
        for j in range(layer, len(self.Attentions)):
            t = t + self.Attentions[j](jax.vmap(self.LNs[j])(t))
            t = t + self.MLPs[j](jax.vmap(self.LN2s[j])(t))
            
        o = jnp.einsum("td,nd->nt", self.W_E, t)
        return o