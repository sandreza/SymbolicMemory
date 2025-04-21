"""Base autoencoder module.

This module provides the core autoencoder functionality, including:
- Base autoencoder class
- Reconstruction loss computation
- Sparse activation regularization
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple, Callable, Dict, Any
from jaxtyping import Array, Float, Int


class Autoencoder(eqx.Module):
    """Base autoencoder class.
    
    This class provides the core autoencoder functionality, including:
    - Encoding and decoding operations
    - Reconstruction loss computation
    - Sparse activation regularization
    
    Attributes:
        encoder: The encoder network
        decoder: The decoder network
        sparsity_penalty: The sparsity penalty coefficient
        sparsity_target: The target sparsity level
    """
    
    encoder: eqx.Module
    decoder: eqx.Module
    sparsity_penalty: float
    sparsity_target: float
    
    def __init__(
        self,
        encoder: eqx.Module,
        decoder: eqx.Module,
        sparsity_penalty: float = 0.1,
        sparsity_target: float = 0.1
    ):
        """Initialize the autoencoder.
        
        Args:
            encoder: The encoder network
            decoder: The decoder network
            sparsity_penalty: The sparsity penalty coefficient
            sparsity_target: The target sparsity level
        """
        self.encoder = encoder
        self.decoder = decoder
        self.sparsity_penalty = sparsity_penalty
        self.sparsity_target = sparsity_target
    
    def encode(self, x: Array) -> Array:
        """Encode the input.
        
        Args:
            x: The input to encode
            
        Returns:
            The encoded representation
        """
        return self.encoder(x)
    
    def decode(self, z: Array) -> Array:
        """Decode the encoded representation.
        
        Args:
            z: The encoded representation
            
        Returns:
            The decoded output
        """
        return self.decoder(z)
    
    def __call__(self, x: Array) -> Array:
        """Forward pass through the autoencoder.
        
        Args:
            x: The input to encode and decode
            
        Returns:
            The reconstructed output
        """
        z = self.encode(x)
        return self.decode(z)
    
    def compute_loss(
        self,
        x: Array,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[Array, Dict[str, Array]]:
        """Compute the autoencoder loss.
        
        Args:
            x: The input to encode and decode
            key: Optional random key for dropout
            
        Returns:
            The total loss and a dictionary of loss components
        """
        # Encode and decode
        z = self.encode(x)
        x_recon = self.decode(z)
        
        # Compute reconstruction loss
        recon_loss = jnp.mean((x - x_recon) ** 2)
        
        # Compute sparsity loss
        sparsity_loss = self.sparsity_penalty * jnp.mean(
            (jnp.mean(jnp.abs(z), axis=0) - self.sparsity_target) ** 2
        )
        
        # Total loss
        total_loss = recon_loss + sparsity_loss
        
        # Return loss components
        loss_dict = {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss
        }
        
        return total_loss, loss_dict 