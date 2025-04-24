
def cross_entropy_loss(
    logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array = None
) -> jax.Array:
    """Compute cross entropy loss.
    
    Args:
        logits: Model output logits of shape (seq_len, vocab_size)
        targets: Target tokens of shape (seq_len,)
        mask: Optional mask of shape (seq_len,)
        
    Returns:
        Cross entropy loss value
    """
    # Get vocab size
    vocab_size = logits.shape[-1]
    
    # Convert targets to one-hot
    targets_one_hot = jax.nn.one_hot(targets, vocab_size)
    
    # Compute cross entropy
    loss = -jnp.sum(targets_one_hot * jax.nn.log_softmax(logits), axis=-1)
    
    # Apply mask if provided
    if mask is not None:
        loss = loss * mask
    
    return jnp.mean(loss)


def batch_loss_fn(
    logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array = None
) -> Tuple[jax.Array, Dict[str, Any]]:
    """Compute loss for a batch of data.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        targets: Target tokens of shape (batch_size, seq_len)
        mask: Optional mask of shape (batch_size, seq_len)
        
    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Compute loss for each sequence in the batch
    losses = jax.vmap(cross_entropy_loss)(logits, targets, mask)
    
    # Average across batch
    total_loss = jnp.mean(losses)
    
    # Return loss and metrics
    metrics = {
        "loss": total_loss,
        "per_sequence_loss": losses
    }
    
    return total_loss, metrics
