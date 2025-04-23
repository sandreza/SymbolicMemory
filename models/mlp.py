
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class MLP(eqx.Module):
    A_E: jax.Array
    b_E: jax.Array  
    A_UE: jax.Array
    b_UE: jax.Array
    def __init__(self, *, d_model = 768, e_model  = 768, key = jr.key(0)): 
       keys = jr.split(key, 5)
       self.A_E  = jr.normal(keys[1], shape = (e_model, d_model)) / jnp.sqrt(d_model)
       self.b_E = jr.normal(keys[2], shape = (1, e_model)) / jnp.sqrt(e_model)
       self.A_UE = jr.normal(keys[3], shape = (d_model, e_model)) / jnp.sqrt(e_model)
       self.b_UE = jr.normal(keys[4], shape = (1, d_model)) / jnp.sqrt(d_model)
    def __call__(self, x): 
        hx = jnp.einsum("td, ed -> te", x, self.A_E) + self.b_E
        hx = jax.nn.gelu(hx)
        h = jnp.einsum("te, de -> td", hx, self.A_UE) + self.b_UE
        return h