from __future__ import annotations

from typing import Any

import gymnasium as gym


class UniversalSeed(gym.Wrapper):
    """Gymnasium-friendly seeding wrapper."""

    def __init__(self, env: gym.Env, seed: int | None = None):
        super().__init__(env)
        self._seed = seed

        if seed is not None:
            if hasattr(self.env.action_space, "seed"):
                self.env.action_space.seed(seed)
            if hasattr(self.env.observation_space, "seed"):
                self.env.observation_space.seed(seed)
            self.env.reset(seed=seed)

    def reset(self, **kwargs: Any):
        if self._seed is not None and "seed" not in kwargs:
            kwargs["seed"] = self._seed
        return self.env.reset(**kwargs)
