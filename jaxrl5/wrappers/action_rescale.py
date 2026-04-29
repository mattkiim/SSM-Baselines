# jaxrl5/wrappers/action_rescale.py
from __future__ import annotations

import gymnasium as gym
import numpy as np


class SymmetricActionWrapper(gym.ActionWrapper):
    """
    Expose a symmetric [-1, 1] action space to the agent, while mapping
    actions back to the original env Box(low, high) before stepping.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "Only supports Box action space."

        self._orig_low = np.array(env.action_space.low, dtype=np.float32)
        self._orig_high = np.array(env.action_space.high, dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action):
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)

        # [-1, 1] -> [0, 1]
        a01 = (a + 1.0) * 0.5

        # [0, 1] -> [orig_low, orig_high] (对一般 Box 也成立)
        return self._orig_low + a01 * (self._orig_high - self._orig_low)
