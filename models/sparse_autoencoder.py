"""Sparse autoencoder module.

This module provides a sparse autoencoder implementation that:
- Uses the base autoencoder class
- Implements sparsity constraints
- Provides visualization utilities
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Callable, Any
from jaxtyping import Array, Float, Int

from .autoencoder import Autoencoder
from .autoencoder_networks import MLPEncoder, MLPDecoder


class SparseAutoencoder(Autoencoder):
    """Sparse autoencoder implementation.
    
    This class extends the base autoencoder with:
    - MLP-based encoder and decoder
    - Sparsity constraints
    - Visualization utilities
    
    Attributes:
        encoder: MLP-based encoder network
        decoder: MLP-based decoder network
        sparsity_penalty: Sparsity penalty coefficient
        sparsity_target: Target sparsity level
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        sparsity_penalty: float = 0.1,
        sparsity_target: float = 0.1,
        activation: Callable = jax.nn.relu,
        dropout: Optional[float] = None,
        key: jax.random.PRNGKey = None
    ):
        """Initialize the sparse autoencoder.
        
        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            hidden_dims: List of hidden layer dimensions
            sparsity_penalty: Sparsity penalty coefficient
            sparsity_target: Target sparsity level
            activation: Activation function
            dropout: Dropout rate
            key: Random key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Split key for encoder and decoder
        key1, key2 = jax.random.split(key)
        
        # Initialize encoder and decoder
        encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims + [latent_dim],
            activation=activation,
            dropout=dropout,
            key=key1
        )
        
        decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims[::-1],
            output_dim=input_dim,
            activation=activation,
            dropout=dropout,
            key=key2
        )
        
        # Initialize base autoencoder
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            sparsity_penalty=sparsity_penalty,
            sparsity_target=sparsity_target
        )
    
    def visualize_activations(
        self,
        x: Array,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Dict[str, plt.Figure]:
        """Visualize the autoencoder activations.
        
        Args:
            x: Input batch
            key: Optional random key for dropout
            
        Returns:
            Dictionary of matplotlib figures
        """
        # Get activations
        z = self.encode(x)
        x_recon = self.decode(z)
        
        # Create figures
        figures = {}
        
        # Plot input vs reconstruction
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(x[0], cmap='viridis')
        ax[0].set_title('Input')
        ax[1].imshow(x_recon[0], cmap='viridis')
        ax[1].set_title('Reconstruction')
        figures['reconstruction'] = fig
        
        # Plot latent activations
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(jnp.abs(z), cmap='viridis', aspect='auto')
        ax.set_title('Latent Activations')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Sample')
        figures['latent_activations'] = fig
        
        # Plot activation statistics
        fig, ax = plt.subplots(figsize=(10, 5))
        mean_activations = jnp.mean(jnp.abs(z), axis=0)
        ax.bar(jnp.arange(len(mean_activations)), mean_activations)
        ax.axhline(y=self.sparsity_target, color='r', linestyle='--')
        ax.set_title('Mean Activation per Latent Dimension')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Mean Activation')
        figures['activation_stats'] = fig
        
        return figures 