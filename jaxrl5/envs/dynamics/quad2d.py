"""Planar 2D quadrotor dynamics utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np
import jax
import jax.numpy as jnp
from jax.core import Tracer


def _is_jax_value(x) -> bool:
    # True if x is a JAX tracer/array (including DeviceArray)
    return isinstance(x, Tracer) or x.__class__.__module__.startswith("jax")


def step_dynamics(state, action, dt: float, params: Dict):
    """Propagate planar quadrotor dynamics using semi-implicit Euler.

    Args:
        state: Array of shape (6,) = [x, xdot, z, zdot, theta, thetadot].
        action: Array of shape (2,) = [T1, T2] in normalized units.
        dt: Integration timestep.
        params: Dictionary containing dynamics parameters. Expected keys:
            - m: Mass.
            - I: Moment of inertia.
            - g: Gravity constant.
            - thrust_scale: Scale mapping normalized thrust to force.
            - torque_scale: Scale for torque from thrust difference.
            - action_low: Lower bound for normalized thrust (default 0.0).
            - action_high: Upper bound for normalized thrust (default 1.0).

    Returns:
        Next state with shape (6,).
        - If inputs are numpy -> returns np.ndarray (env-friendly)
        - If inputs are JAX -> returns jnp.ndarray (jit/grad-friendly)
    """
    use_jax = _is_jax_value(state) or _is_jax_value(action)
    xp = jnp if use_jax else np

    # Keep previous numeric behavior for env (float64 internal), but JAX path uses float32
    state_dtype = xp.float32 if use_jax else xp.float64
    action_dtype = xp.float32

    action_low = params.get("action_low", 0.0)
    action_high = params.get("action_high", 1.0)

    # IMPORTANT: do NOT call np.asarray on JAX tracers; use xp.asarray
    s = xp.asarray(state, dtype=state_dtype)
    a = xp.asarray(action, dtype=action_dtype)
    clipped_action = xp.clip(a, action_low, action_high)

    # unpack: [x, xdot, z, zdot, theta, thetadot]
    x, xdot, z, zdot, theta, thetadot = s

    m = params.get("m", 1.0)
    I = params.get("I", 0.02)
    g = params.get("g", 9.81)
    thrust_scale = params["thrust_scale"]
    torque_scale = params["torque_scale"]

    F1 = thrust_scale * clipped_action[0]
    F2 = thrust_scale * clipped_action[1]
    u1 = F1 + F2
    tau = torque_scale * (clipped_action[1] - clipped_action[0])

    # Use xp.sin/cos so JAX can trace/grad
    xddot = -(u1 / m) * xp.sin(theta)
    zddot = (u1 / m) * xp.cos(theta) - g
    thetaddot = tau / I

    # Semi-implicit (symplectic) Euler integration: update velocities first, then positions.
    xdot = xdot + xddot * dt
    x = x + xdot * dt
    zdot = zdot + zddot * dt
    z = z + zdot * dt
    thetadot = thetadot + thetaddot * dt
    theta = theta + thetadot * dt

    next_state = xp.stack([x, xdot, z, zdot, theta, thetadot], axis=0).astype(xp.float32)

    # env path expects numpy arrays
    if use_jax:
        return next_state
    return np.asarray(next_state, dtype=np.float32)
