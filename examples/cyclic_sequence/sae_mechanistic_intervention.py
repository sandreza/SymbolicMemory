"""Perform mechanistic interventions using a trained sparse autoencoder."""

import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jr
from training.utils import load
from models.transformer import SimpleTransformer
from models.autoencoder import AutoEncoder
from data import generate_cyclic_data

# Set random key
key = jr.key(0)

# Load trained models
print("Loading models...")
transformer = load("cyclic_model/st_model.mo", SimpleTransformer)
sae = load("cyclic_model/sae.mo", AutoEncoder)

# Generate test data
print("Generating test data...")
token_dimension = 3
holding_time = 6
data = generate_cyclic_data(
    token_dimension=token_dimension,
    holding_time=holding_time,
    data_multiplier=1000
)

# Test parameters
block_size = 10
layer_level = 0

# Get candidate indices for intervention
print("\nFinding candidate indices for intervention...")
candidate_group = []
for i in range(18):
    x = data[i:block_size+i]
    t = transformer.residual(x, layer=layer_level)
    rv = t[-1]
    candidate_indices = jnp.flip(jnp.argsort(sae.hx(rv)))
    candidate_group.append(candidate_indices)

# Combine candidate indices from different sequences
candidate_indices = candidate_group[0][:20]
for i in range(1, 18):
    candidate_indices = jnp.union1d(candidate_indices, candidate_group[i][:20])

print(f"Found {len(candidate_indices)} candidate indices for intervention")

# Define intervention function
def sae_intervention(sae, t):
    f = jax.vmap(sae.hx)(t)
    g = f.at[-1, candidate_indices[0]].multiply(10.0)
    s_tilde = jnp.einsum("te, ed -> td", g, sae.W_UE)
    s = jax.vmap(sae)(t)
    t = s_tilde - s + t
    return t

intervention = ft.partial(sae_intervention, sae)

# Test intervention on different sequences
print("\nTesting intervention on different sequences:")

# Test 1: Normal sequence
print("\nTest 1: Normal sequence")
x = data[0:block_size]
o = transformer.intervention(x, layer=layer_level, intervention=intervention)
print("Original prediction:", jnp.argmax(transformer(x)[-1]))
print("Intervened prediction:", jnp.argmax(o[-1]))
print("Difference:", o[-1] - transformer(x)[-1])

# Test 2: Forbidden sequence [1, 0]
print("\nTest 2: Forbidden sequence [1, 0]")
x = jnp.array([1, 0])
x = jnp.pad(x, (0, block_size - len(x)))
o = transformer.intervention(x, layer=layer_level, intervention=intervention)
print("Original prediction:", jnp.argmax(transformer(x)[-1]))
print("Intervened prediction:", jnp.argmax(o[-1]))
print("Difference:", o[-1] - transformer(x)[-1])

# Test 3: Random sequence
print("\nTest 3: Random sequence")
key, subkey = jr.split(key)
x = jr.randint(subkey, (block_size,), 0, token_dimension)
o = transformer.intervention(x, layer=layer_level, intervention=intervention)
print("Original prediction:", jnp.argmax(transformer(x)[-1]))
print("Intervened prediction:", jnp.argmax(o[-1]))
print("Difference:", o[-1] - transformer(x)[-1])

# Test 4: Generate sequence with intervention
print("\nTest 4: Generate sequence with intervention")
x = data[0:block_size]
seq_length = 100
xs = jnp.arange(seq_length)

for j in range(seq_length):
    key, subkey = jr.split(key)
    a = jnp.arange(token_dimension)
    y = transformer.intervention(x, layer=layer_level, intervention=intervention)
    p = jax.nn.softmax(y)[-1]
    x_new = jnp.array([jr.choice(subkey, a, p=p)])
    xs = xs.at[j].set(x_new[0])
    x = jnp.concatenate([x, x_new])[-block_size:]

print("Generated sequence:", xs)

# Test 5: Test each candidate index individually
print("\nTest 5: Testing each candidate index individually")
for candidate_index in candidate_indices:
    def sae_intervention(sae, t):
        f = jax.vmap(sae.hx)(t)
        g = f.at[-1, candidate_index].multiply(20.0)
        s_tilde = jnp.einsum("te, ed -> td", g, sae.W_UE)
        s = jax.vmap(sae)(t)
        t = s_tilde - s + t
        return t
    
    intervention = ft.partial(sae_intervention, sae)
    
    print(f"\nTesting candidate index {candidate_index}:")
    x = data[0:block_size]
    seq_length = 200
    xs = jnp.arange(seq_length)
    
    for j in range(seq_length):
        key, subkey = jr.split(key)
        a = jnp.arange(token_dimension)
        y = transformer.intervention(x, layer=layer_level, intervention=intervention)
        p = jax.nn.softmax(y)[-1]
        x_new = jnp.array([jr.choice(subkey, a, p=p)])
        xs = xs.at[j].set(x_new[0])
        x = jnp.concatenate([x, x_new])[-block_size:]
    
    print("Generated sequence:", xs)

