from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register


_CUSTOM_ENVS = {
    "QuadrotorTracking2D-v0": "jaxrl5.envs.quadrotor_tracking_2d:make_quadrotor_tracking_2d_env",
    "QuadrotorStabilization2D-v0": "jaxrl5.envs.quadrotor_stabilization_2d:make_quadrotor_stabilization_2d_env",
    "QuadrotorTracking3D-v0": "jaxrl5.envs.quadrotor_tracking_3d:make_quadrotor_tracking_3d_env",
    "QuadrotorStabilization3D-v0": "jaxrl5.envs.quadrotor_stabilization_3d:make_quadrotor_stabilization_3d_env",
}


def ensure_custom_envs_registered() -> None:
    """Register custom environments if not already registered."""
    for env_id, entry_point in _CUSTOM_ENVS.items():
        try:
            gym.spec(env_id)
        except gym.error.Error:
            register(
                id=env_id,
                entry_point=entry_point,
            )
