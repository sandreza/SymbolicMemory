import jax
import jax.numpy as jnp

def generate_cyclic_data(
    token_dimension: int = 3,
    holding_time: int = 6,
    data_multiplier: int = 100000
) -> jax.Array:
    """Generate a cyclic sequence dataset.
    
    Args:
        token_dimension: Number of unique tokens in the sequence
        holding_time: Number of times each token is repeated
        data_multiplier: Number of times to repeat the base sequence
        
    Returns:
        Array containing the cyclic sequence
    """
    # Create base sequence (e.g., 1,2,3,1,2,3... or 1,1,2,2,3,3,1,1,2,2,3,3)
    base_sequence = jnp.repeat(jnp.arange(token_dimension), holding_time)
    
    # Tile the sequence to create a large dataset
    return jnp.tile(base_sequence, data_multiplier)