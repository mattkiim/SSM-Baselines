import jax.numpy as jnp


def ucb_from_ensemble(q_stack: jnp.ndarray, k: float, eps: float = 1e-8):
    """Compute mean + k * std across ensemble dimension (axis 0)."""
    if q_stack.ndim == 3:
        q_stack = q_stack[..., 0]
    mean = jnp.mean(q_stack, axis=0)
    std = jnp.std(q_stack, axis=0)
    return (mean + k * (std + eps))[:, None]

