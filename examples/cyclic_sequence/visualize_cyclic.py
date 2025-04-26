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
len(model.attention(x))
attention_weights = model.attention(x, layer=0)
n_heads = attention_weights.shape[0]
n_layers = len(model.Attentions)


for layer in range(n_layers):
    for istart in range(0, td * ht * 3):
        x = jnp.array(data[istart:istart+block])
        attention_weights = model.attention(x, layer=layer)
        sqrt_n = int(jnp.ceil(jnp.sqrt(n_heads)))
        fig, axes = plt.subplots(sqrt_n, sqrt_n, figsize=(2*sqrt_n, 2*sqrt_n))
        plt.suptitle(str(x))
        # Plot each attention head
        for i in range(n_heads):
            row = i // sqrt_n
            col = i % sqrt_n
            ax = axes[row, col]
            ax.imshow(attention_weights[i])
            ax.set_title(f"Head {i}")
            ax.axis("off")
        plt.savefig('cyclic_model/' + 'layer_' + str(layer) + '_p_image_' + str(istart) +  '.png')
        plt.close('all')

    plt.close('all')


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



