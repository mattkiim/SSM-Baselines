from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import ClipAction, FlattenObservation, RescaleAction

from jaxrl5.wrappers.add_cost_from_info import AddCostFromInfo
from jaxrl5.wrappers.pixels import wrap_pixels
from jaxrl5.wrappers.record_episode_statistics import SafetyRecordEpisodeStatistics
from jaxrl5.wrappers.single_precision import SinglePrecision
from jaxrl5.wrappers.universal_seed import UniversalSeed
from jaxrl5.wrappers.wandb_video import WANDBVideo


def wrap_gym(env: gym.Env, seed: int | None = None, rescale_actions: bool = True) -> gym.Env:
    # env = SinglePrecision(env)
    env = UniversalSeed(env, seed=seed)
    if rescale_actions:
        env = RescaleAction(env, -1.0, 1.0)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = ClipAction(env)

    return env


__all__ = [
    "AddCostFromInfo",
    "SafetyRecordEpisodeStatistics",
    "SinglePrecision",
    "UniversalSeed",
    "WANDBVideo",
    "wrap_gym",
    "wrap_pixels",
]
