"""Base autoencoder module.

This module provides the core autoencoder functionality, including:
- Base autoencoder class
- Reconstruction loss computation
- Sparse activation regularization
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Optional, Tuple, Callable, Dict, Any
from jaxtyping import Array, Float, Int


class SimpleAutoEncoder(eqx.Module):
    W_E: jax.Array
    def __init__(self, *, d_model = 768, e_model  = 768, key = jr.key(0)): 
       keys = jr.split(key, 2)
       self.W_E  = jr.normal(keys[1], shape = (e_model, d_model)) / jnp.sqrt(d_model)
    def __call__(self, x): 
        hx = jnp.einsum("d, ed -> e", x, self.W_E) 
        hx = jax.nn.relu(hx)
        h = jnp.einsum("e, ed -> d", hx, self.W_E)
        return h
    def hx(self, x): 
        hx = jnp.einsum("d, ed -> e", x, self.W_E) 
        hx = jax.nn.relu(hx)
        return hx

class AutoEncoder(eqx.Module):
    W_E: jax.Array
    b_E: jax.Array
    W_UE: jax.Array
    def __init__(self, *, d_model = 768, e_model  = 768, key = jr.key(0)): 
       keys = jr.split(key, 4)
       self.W_E  = jr.normal(keys[1], shape = (e_model, d_model)) / jnp.sqrt(d_model)
       self.W_UE  = jr.normal(keys[2], shape = (e_model, d_model)) / jnp.sqrt(e_model)
       self.b_E = jr.normal(keys[3], shape = (e_model)) / jnp.sqrt(e_model)
    def __call__(self, x): 
        hx = jnp.einsum("d, ed -> e", x, self.W_E) + self.b_E
        hx = jax.nn.relu(hx)
        h = jnp.einsum("e, ed -> d", hx, self.W_UE)
        return h
    def hx(self, x): 
        hx = jnp.einsum("d, ed -> e", x, self.W_E)  + self.b_E
        hx = jax.nn.relu(hx)
        return hx