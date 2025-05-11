from training.utils import load
from models.transformer import SimpleTransformer, Transformer
from models.utils import generate_predictions
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import load_lorenz_data
model = load("lorenz_model/model.mo", Transformer) # load("lorenz_model/st_model.mo", SimpleTransformer)

data, val_data = load_lorenz_data()

istart = 0
block = 100
x = jnp.array(data[istart:istart+block])
len(model.attention(x))
attention_weights = model.attention(x, layer=0)
n_heads = attention_weights.shape[0]
n_layers = len(model.Attentions)


for layer in range(n_layers):
    for istart in range(0, 30):
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
        plt.savefig('lorenz_model/' + 'layer_' + str(layer) + '_p_image_' + str(istart) +  '.png')
        plt.close('all')

    plt.close('all')