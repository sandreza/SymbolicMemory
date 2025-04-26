from training.utils import load
from models.transformer import SimpleTransformer
from models.utils import generate_predictions
from data import generate_cyclic_data
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt

model = load("cyclic_model/st_model.mo", SimpleTransformer)

td = 3 
ht = 6
data = generate_cyclic_data(token_dimension = td, holding_time = ht, data_multiplier = 4)


istart = 0
block = 10
x = jnp.array(data[istart:istart+block])
p = jax.vmap(jax.nn.softmax)(model(x))
print(p)
print(x)

istart = 5
block = 10
x = jnp.array(data[istart:istart+block])
p = jax.vmap(jax.nn.softmax)(model(x))
print(p)
print(x)

predictions = generate_predictions(
    model,
    initial_seq = jnp.array(data[0]),
    max_new_tokens= 100,
    block_size= 10,
    key = jr.key(0),
    batch_size = 1,
)
# Forbiden sequence: 1, 0
hallucination_1 = generate_predictions(
    model,
    initial_seq = jnp.array([1, 0]),
    max_new_tokens= 100,
    block_size= 10,
    key = jr.key(0),
    batch_size = 1,
)

# Forbiden sequence: 2, 1, 0, 2, 1,0, 2, 1, 0, 2, 1, 0
hallucination_2 = generate_predictions(
    model,
    initial_seq = jnp.array([2, 1, 0, 2, 1,0, 2, 1, 0, 2, 1, 0]),
    max_new_tokens= 100,
    block_size= 10,
    key = jr.key(0),
    batch_size = 1,
)

keys = jr.split(jr.key(0), 4)
x = jr.randint(keys[0], (10,), 0, 3)
hallucination_3 = generate_predictions(
    model,
    initial_seq = x,
    max_new_tokens= 100,
    block_size= 10,
    key = keys[1],
    batch_size = 1,
)

x = jr.randint(keys[2], (10,), 0, 3)
hallucination_4 = generate_predictions(
    model,
    initial_seq = x,
    max_new_tokens= 100,
    block_size= 10,
    key = keys[3],
    batch_size = 1,
)

