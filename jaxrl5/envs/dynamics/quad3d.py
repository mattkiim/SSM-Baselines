"""3D quadrotor dynamics utilities (JAX + NumPy compatible)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import jax.numpy as jnp
from jax.core import Tracer


def _is_jax_value(x) -> bool:
    return isinstance(x, Tracer) or x.__class__.__module__.startswith("jax")


def step_dynamics_3d(state, action, dt: float, params: Dict):
    """
    3D quadrotor dynamics using semi-implicit Euler.

    State:  [px, py, pz, vx, vy, vz, phi, theta, psi]
    Action: [f, phi_dot, theta_dot, psi_dot]

    Returns next_state with same shape.
    """
    use_jax = _is_jax_value(state) or _is_jax_value(action)
    xp = jnp if use_jax else np

    state_dtype = xp.float32 if use_jax else xp.float64
    action_dtype = xp.float32

    s = xp.asarray(state, dtype=state_dtype)
    a = xp.asarray(action, dtype=action_dtype)

    # unpack state
    px, py, pz, vx, vy, vz, phi, theta, psi = s

    # unpack params
    m = params.get("m", 1.0)
    g = params.get("g", 9.81)

    # controls
    f = a[0]        # thrust
    phi_dot = a[1]
    theta_dot = a[2]
    psi_dot = a[3]

    # trig
    s_phi = xp.sin(phi)
    c_phi = xp.cos(phi)
    s_theta = xp.sin(theta)
    c_theta = xp.cos(theta)

    # --- accelerations (from your attached Quad3D model) ---
    ax = -(f / m) * s_theta
    ay =  (f / m) * c_theta * s_phi
    az = -(f / m) * c_theta * c_phi + g

    # --- semi-implicit Euler ---
    vx = vx + ax * dt
    px = px + vx * dt

    vy = vy + ay * dt
    py = py + vy * dt

    vz = vz + az * dt
    pz = pz + vz * dt

    phi = phi + phi_dot * dt
    theta = theta + theta_dot * dt
    psi = psi + psi_dot * dt

    next_state = xp.stack(
        [px, py, pz, vx, vy, vz, phi, theta, psi],
        axis=0
    ).astype(xp.float32)

    if use_jax:
        return next_state
    return np.asarray(next_state, dtype=np.float32)