from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        capacity: int,
        obs_dtype: np.dtype = np.float32,
        act_dtype: np.dtype = np.float32,
    ) -> None:
        self.capacity = int(capacity)
        self.size = 0
        self.ptr = 0

        self.observations = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_observations = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((self.capacity, *act_shape), dtype=act_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.costs = np.zeros((self.capacity,), dtype=np.float32)

        # Gymnasium termination flags
        self.terminated = np.zeros((self.capacity,), dtype=np.bool_)
        self.truncated = np.zeros((self.capacity,), dtype=np.bool_)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

        self.rng = np.random.default_rng()

    def seed(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def insert(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward,
        cost,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(np.asarray(reward, dtype=np.float32))
        self.costs[self.ptr] = float(np.asarray(cost, dtype=np.float32))
        self.next_observations[self.ptr] = next_obs

        self.terminated[self.ptr] = bool(terminated)
        self.truncated[self.ptr] = bool(truncated)
        self.dones[self.ptr] = bool(terminated or truncated)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = self.rng.integers(0, self.size, size=batch_size)

        batch = dict(
            observations=self.observations[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            costs=self.costs[idxs],
            next_observations=self.next_observations[idxs],
            dones=self.dones[idxs],
            terminated=self.terminated[idxs],
            truncated=self.truncated[idxs],
            not_terminated=(~self.terminated[idxs]).astype(np.float32),
        )
        return batch

    def __len__(self) -> int:
        return self.size
