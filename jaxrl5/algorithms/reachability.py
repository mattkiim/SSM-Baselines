"""Discounted maximum-based reachability backup (RCRL, equation 9)."""

import jax.numpy as jnp


def reachability_target(h, next_qh, not_terminated, discount):
    """For a terminal transition there is no future constraint to bootstrap.

    Time-limit truncations retain bootstrapping, matching continuing-task
    training. For transition constraints, h is the observed h(s, a), not a
    binary cost and not a sum of current and successor margins.
    """
    target = (1.0 - discount) * h + discount * jnp.maximum(h, next_qh)
    return jnp.where(not_terminated, target, h)
