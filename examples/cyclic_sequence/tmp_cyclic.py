jax.vmap(jax.nn.softmax)(trainer.model(jnp.array([0, 0, 0, 0])))
jax.vmap(jax.nn.softmax)(trainer.model(jnp.array([1, 1, 0])))
jax.vmap(jax.nn.softmax)(trainer.model(jnp.array([1, 1, 0, 2])))
jax.vmap(jax.nn.softmax)(trainer.model(jnp.array([1, 1, 0, 2, 2])))
jax.vmap(jax.nn.softmax)(trainer.model(jnp.array([1, 1, 0, 2, 2, 2])))
jax.vmap(jax.nn.softmax)(trainer.model(jnp.array([1, 1, 0, 2, 2, 2, 2])))
jax.vmap(jax.nn.softmax)(trainer.model(jnp.array([1, 1, 0, 2, 2, 2, 2, 2])))
jax.vmap(jax.nn.softmax)(trainer.model(jnp.array([1, 1, 0, 2, 2, 2, 2, 2, 0])))
jax.vmap(jax.nn.softmax)(trainer.model(jnp.array([1, 1, 0, 2, 2, 2, 2, 2, 0, 0])))

# Summary 
# 1, 1, 0, 2, 2, 2, 2, 2 
# overconfidently precitions that the next token is 0
# 1, 1, 1, 2, 2, 2, 2, 2 
# correctly predicts the next token as 2
# 1, 1, 2, 2, 2, 2, 2, 2
# correctly predicts the next token as 0