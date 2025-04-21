"""Autoencoder network components.

This module provides the encoder and decoder networks for the autoencoder,
including:
- MLP-based encoder and decoder
- Convolutional encoder and decoder
- Custom activation functions
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple, Callable, List
from jaxtyping import Array, Float, Int


class MLPEncoder(eqx.Module):
    """MLP-based encoder network.
    
    This class implements a multi-layer perceptron encoder with:
    - Customizable number of layers
    - Customizable hidden dimensions
    - Optional dropout
    - Custom activation functions
    
    Attributes:
        layers: List of linear layers
        activation: Activation function
        dropout: Dropout rate
    """
    
    layers: List[eqx.nn.Linear]
    activation: Callable
    dropout: Optional[float]
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: Callable = jax.nn.relu,
        dropout: Optional[float] = None,
        key: jax.random.PRNGKey = None
    ):
        """Initialize the MLP encoder.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
            key: Random key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Create layers
        self.layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            key, subkey = jax.random.split(key)
            self.layers.append(
                eqx.nn.Linear(dims[i], dims[i + 1], key=subkey)
            )
        
        self.activation = activation
        self.dropout = dropout
    
    def __call__(
        self,
        x: Float[Array, "..."],
        key: Optional[jax.random.PRNGKey] = None
    ) -> Float[Array, "..."]:
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor
            key: Optional random key for dropout
            
        Returns:
            Encoded representation
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Don't apply activation to last layer
                x = self.activation(x)
                if self.dropout is not None and key is not None:
                    x = eqx.nn.Dropout(self.dropout)(x, key=key)
        return x


class MLPDecoder(eqx.Module):
    """MLP-based decoder network.
    
    This class implements a multi-layer perceptron decoder with:
    - Customizable number of layers
    - Customizable hidden dimensions
    - Optional dropout
    - Custom activation functions
    
    Attributes:
        layers: List of linear layers
        activation: Activation function
        dropout: Dropout rate
    """
    
    layers: List[eqx.nn.Linear]
    activation: Callable
    dropout: Optional[float]
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Callable = jax.nn.relu,
        dropout: Optional[float] = None,
        key: jax.random.PRNGKey = None
    ):
        """Initialize the MLP decoder.
        
        Args:
            latent_dim: Latent dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function
            dropout: Dropout rate
            key: Random key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Create layers
        self.layers = []
        dims = [latent_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            key, subkey = jax.random.split(key)
            self.layers.append(
                eqx.nn.Linear(dims[i], dims[i + 1], key=subkey)
            )
        
        self.activation = activation
        self.dropout = dropout
    
    def __call__(
        self,
        z: Float[Array, "..."],
        key: Optional[jax.random.PRNGKey] = None
    ) -> Float[Array, "..."]:
        """Forward pass through the decoder.
        
        Args:
            z: Latent representation
            key: Optional random key for dropout
            
        Returns:
            Decoded output
        """
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:  # Don't apply activation to last layer
                z = self.activation(z)
                if self.dropout is not None and key is not None:
                    z = eqx.nn.Dropout(self.dropout)(z, key=key)
        return z 