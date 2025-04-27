import functools as ft
from models.autoencoder import AutoEncoder
from models.transformer import SimpleTransformer
from training.utils import load
from training.loss import sae_make_step, sae_loss_fn, sae_batch_loss_function
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from data import generate_cyclic_data
e_factor = 2
d_model = 9 * 8
e_model = 9 * 8 * e_factor
sae = AutoEncoder(d_model = d_model, e_model = e_model)

def get_batch(
    data: jax.Array,
    batch_size: int,
    block_size: int,
    key: jax.random.PRNGKey
):
    ix = jr.randint(
        key,
        (batch_size,),
        0,
        len(data) - block_size - 2
    )
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

optimizer = optax.adamw(learning_rate=1e-3, weight_decay = 1e-2) # ,  weight_decay = 1e-2
opt_state = optimizer.init(eqx.filter(sae, eqx.is_inexact_array))

layer_level = 1
model = load("cyclic_model/st_model.mo", SimpleTransformer)
data = generate_cyclic_data(token_dimension = 3, holding_time = 6, data_multiplier = 100000)
t0 = model.residual(data[0:10], layer = 0)
t1 = model.residual(data[0:10], layer = 1)
t = t0
rv = t[-1]
jax.vmap(sae)(t)

batch_size = 10
block_size = 10


key = jr.key(0)
key, subkey = jr.split(key)

losses = []
for step in range(100 * 10):
    key, subkey = jr.split(subkey)
    input_data, output_data = get_batch(data, batch_size, block_size, subkey)
    input_data = model.residual(input_data[0], layer = layer_level)
    loss, sae, opt_state = sae_make_step(sae, input_data, e_factor, opt_state, optimizer.update)
    losses.append(loss)
    if step%10==0:
        print('-------')
        print(loss)
        print(sae_loss_fn(sae, 0.0, rv))
        print('-------')

i = 2
x= data[i:block_size+i]

t = model.residual(x, layer = layer_level)
rv = t[-1]
jnp.argsort(sae.hx(rv))
sae(rv) - rv
jnp.sum(sae.hx(rv) > 1.0)

def sae_intervention(sae, t):
    f = jax.vmap(sae.hx)(t)
    g = f.at[-1, 26].multiply(10.0)
    s_tilde = jnp.einsum("te, ed -> td", g, sae.W_UE)
    s = jax.vmap(sae)(t) 
    t = s_tilde - s + t
    return t

intervention = ft.partial(sae_intervention, sae)

for i in jnp.arange(block_size*2):
    x = data[i:block_size+i]
    o = model.intervention(x, layer = layer_level, intervention =  intervention)
    # print(o[-1])
    print(jnp.argmax(o[-1]))
    print(o[-1] - model(x)[-1] )

i = 0
token_dimension = 3
x = data[i:block_size+i]
key, subkey = jr.split(jr.key(0))
seq_length = 100
xs = jnp.arange(seq_length)
for j in jnp.arange(seq_length):
    if j%10==0:
        print(j)
    key, subkey = jr.split(key)
    a = jnp.arange(token_dimension)
    y = model(x) # model.intervention(x, layer = layer_level, intervention =  intervention) # 
    p = jax.nn.softmax(y)[-1]
    x_new = jnp.array([jax.random.choice(key, a, p=p)])
    xs = xs.at[j].set(x_new[0])
    x = jnp.concatenate([x, x_new])[-block_size:]