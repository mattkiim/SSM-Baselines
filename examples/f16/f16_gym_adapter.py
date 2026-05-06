"""Gymnasium-style adapter for F16StabilizeEnvV6.

Converts the old-gym 6-return step signature into the gymnasium 5-return
signature, and exposes the binary safety cost via info["cost"] so that
jaxrl5's AddCostFromInfo wrapper picks it up.

Design notes:
- F16StabilizeEnvV6.step returns (obs, reward, h_margin, binary_cost, done, info).
  We map this to (obs, reward, terminated, truncated, info_with_cost).
- terminated  := info["terminated"]  (invalid_dynamics OR hard_terminal)
- truncated   := info["timeout"]
- info["cost"] := binary_cost   (= 1{h_margin > 0})
- info["h"]    := h_margin      (continuous SDF, useful for diagnostics)
- The underlying env's reset returns just obs; we return (obs, info_dict).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from f16_stabilize_env_fast import F16StabilizeEnvV6P2 as F16StabilizeEnvV6


class F16StabilizeGymWrapper(gym.Env):
    """Gymnasium-compatible wrapper around F16StabilizeEnvV6."""

    metadata = {"render_modes": []}

    def __init__(self, **env_kwargs):
        super().__init__()
        self._env = F16StabilizeEnvV6(**env_kwargs)

        # Re-expose spaces. The underlying env already uses gym_compat shims
        # but we want pure gymnasium spaces here.
        obs_low = np.full(self._env.obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(self._env.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._env.action_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------ properties
    @property
    def unwrapped(self):
        return self._env

    # ----------------------------------------------------------------- reset
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._env._rng = np.random.RandomState(seed)
        obs = self._env.reset(options=options)

        info = {
            "raw_state": np.asarray(self._env._x, dtype=np.float32).copy(),
        }
        return np.asarray(obs, dtype=np.float32), info

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray):
        obs, reward, h_margin, binary_cost, done, info = self._env.step(action)

        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("timeout", False)) and (not terminated)

        # Inject standardized fields for jaxrl5 RAC pipeline.
        info_out = dict(info)
        info_out["cost"] = float(binary_cost)
        info_out["h"] = float(h_margin)
        info_out["raw_state"] = np.asarray(self._env._x, dtype=np.float32).copy()

        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            info_out,
        )

    def close(self):
        # No resources to release.
        pass


def make_f16_stabilize_v6(**kwargs) -> gym.Env:
    """Factory used by env registration."""
    return F16StabilizeGymWrapper(**kwargs)