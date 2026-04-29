from __future__ import annotations

import gymnasium as gym


class AddCostFromInfo(gym.Wrapper):
    """Adapt an environment to emit a cost signal based on info.

    This wrapper assumes the wrapped environment returns a standard Gymnasium
    5-tuple from ``step`` (obs, reward, terminated, truncated, info). It lifts
    the ``info['cost']`` field into the returned tuple to match the 6-tuple
    interface expected by jaxrl5 evaluation and buffers.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = float(info.get("cost", 0.0))
        return obs, reward, cost, terminated, truncated, info

    def reset(self, **kwargs):  # type: ignore[override]
        return self.env.reset(**kwargs)
