from training.utils import load
from models.transformer import SimpleTransformer
from models.utils import generate_predictions
from data import generate_cyclic_data
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt

print("\n=== Loading model and generating test data ===")
model = load("cyclic_model/st_model.mo", SimpleTransformer)

td = 3 
ht = 6
print(f"\nGenerating cyclic data with token_dimension={td}, holding_time={ht}")
data = generate_cyclic_data(token_dimension = td, holding_time = ht, data_multiplier = 4)

print("\n=== Testing model predictions on different windows ===")
print("\nWindow 1 (start=0):")
istart = 0
block = 10
x = jnp.array(data[istart:istart+block])
p = jax.vmap(jax.nn.softmax)(model(x))
print("Input sequence:", x)
print("Predicted probabilities:", p)

print("\nWindow 2 (start=5):")
istart = 5
block = 10
x = jnp.array(data[istart:istart+block])
p = jax.vmap(jax.nn.softmax)(model(x))
print("Input sequence:", x)
print("Predicted probabilities:", p)

print("\n=== Testing model generation on various sequences ===")

print("\nTest 1: Normal sequence generation")
predictions = generate_predictions(
    model,
    initial_seq = jnp.array(data[0]),
    max_new_tokens= 100,
    block_size= 10,
    key = jr.key(0),
    batch_size = 1,
)
print("Initial token:", data[0])
print("Generated sequence:", predictions.flatten())

print("\nTest 2: Forbidden sequence [1, 0]")
hallucination_1 = generate_predictions(
    model,
    initial_seq = jnp.array([1, 0]),
    max_new_tokens= 100,
    block_size= 10,
    key = jr.key(0),
    batch_size = 1,
)
print("Generated sequence:", hallucination_1.flatten())

print("\nTest 3: Repeated forbidden pattern [2,1,0,...]")
hallucination_2 = generate_predictions(
    model,
    initial_seq = jnp.array([2, 1, 0, 2, 1,0, 2, 1, 0, 2, 1, 0]),
    max_new_tokens= 100,
    block_size= 10,
    key = jr.key(0),
    batch_size = 1,
)
print("Generated sequence:", hallucination_2.flatten())

print("\nTest 4: Random sequence 1")
keys = jr.split(jr.key(0), 4)
x = jr.randint(keys[0], (10,), 0, 3)
print("Initial random sequence:", x)
hallucination_3 = generate_predictions(
    model,
    initial_seq = x,
    max_new_tokens= 100,
    block_size= 10,
    key = keys[1],
    batch_size = 1,
)
print("Generated sequence:", hallucination_3.flatten())

print("\nTest 5: Random sequence 2")
x = jr.randint(keys[2], (10,), 0, 3)
print("Initial random sequence:", x)
hallucination_4 = generate_predictions(
    model,
    initial_seq = x,
    max_new_tokens= 100,
    block_size= 10,
    key = keys[3],
    batch_size = 1,
)
print("Generated sequence:", hallucination_4.flatten())

print("\nTest 6: Repeated zeros with single 2")
hallucination_5 = generate_predictions(
    model,
    initial_seq = jnp.array([0, 0, 0, 0, 0, 2, 0, 0, 0, 0, ]),
    max_new_tokens= 100,
    block_size= 10,
    key = jr.key(0),
    batch_size = 1,
)
print("Generated sequence:", hallucination_5.flatten())

print("\nTest 7: Pattern with multiple 2s")
hallucination_6 = generate_predictions(
    model,
    initial_seq = jnp.array([2, 0, 0, 0, 0, 2, 0, 0, 0, 2, ]),
    max_new_tokens= 100,
    block_size= 10,
    key = jr.key(0),
    batch_size = 1,
)
print("Generated sequence:", hallucination_6.flatten())

print("\n=== Testing completed ===")
