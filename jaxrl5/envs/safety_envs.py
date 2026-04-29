from __future__ import annotations

import gymnasium as gym
import safety_gymnasium

from jaxrl5.envs.registration import ensure_custom_envs_registered
from jaxrl5.wrappers import (
    AddCostFromInfo,
    SafetyRecordEpisodeStatistics,
    SinglePrecision,
)


def make_safety_env(
    env_name: str,
    seed: int | None = None,
    render_mode: str | None = None,
    **kwargs,
):
    """Factory returning a Gymnasium-compatible Safety-Gymnasium environment."""

    env = safety_gymnasium.make(env_name, render_mode=render_mode, **kwargs)

    # Optionally perform an initial reset to apply the seed immediately.
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()

    return env


def make_env(
    env_name: str,
    seed: int | None = None,
    render_mode: str | None = None,
    **kwargs,
):
    """Create either a Safety-Gymnasium or custom quadrotor environment.

    All environments are wrapped to emit float32 observations/actions and record
    episode statistics. Custom environments are additionally wrapped to expose a
    cost signal derived from their info dict to match the 6-tuple interface.
    """

    if env_name in {"QuadrotorTracking2D-v0", 
                    "QuadrotorTracking3D-v0",
                    "QuadrotorStabilization3D-v0",
                    }:
        ensure_custom_envs_registered()
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
        env = AddCostFromInfo(env)
    else:
        env = make_safety_env(env_name, seed=seed, render_mode=render_mode, **kwargs)

    env = SinglePrecision(env)
    env = SafetyRecordEpisodeStatistics(env, deque_size=1)

    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()

    return env
