from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict

import gymnasium as gym


class SafetyRecordEpisodeStatistics(gym.Wrapper):
    """Record episode statistics when env.step returns reward and cost."""

    def __init__(self, env: gym.Env, deque_size: int = 100):
        super().__init__(env)
        self.return_queue: Deque[float] = deque(maxlen=deque_size)
        self.cost_queue: Deque[float] = deque(maxlen=deque_size)
        self.length_queue: Deque[int] = deque(maxlen=deque_size)
        self._reset_episode_stats()

    def _reset_episode_stats(self) -> None:
        self._episode_return = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._reset_episode_stats()
        return observation, info

    def step(self, action):
        (  # type: ignore[assignment]
            observation,
            reward,
            cost,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        self._episode_return += float(reward)
        self._episode_cost += float(cost)
        self._episode_length += 1

        done = bool(terminated or truncated)
        if done:
            stats = {
                "return": self._episode_return,
                "cost": self._episode_cost,
                "length": self._episode_length,
                # legacy keys
                "r": self._episode_return,
                "c": self._episode_cost,
                "l": self._episode_length,
            }

            self.return_queue.append(self._episode_return)
            self.cost_queue.append(self._episode_cost)
            self.length_queue.append(self._episode_length)

            info = dict(info)
            info["episode"] = stats

            self._reset_episode_stats()

        return observation, reward, cost, terminated, truncated, info
