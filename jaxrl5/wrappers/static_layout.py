from __future__ import annotations

import gymnasium as gym


class StaticLayoutWrapper(gym.Wrapper):
    """Pin every reset to the same layout seed, removing episode-to-episode randomization.

    Safety-Gymnasium only re-randomizes hazard/goal/robot placement when a seed is
    explicitly passed to reset() -- omitting it just continues the existing RNG
    stream. This wrapper forces every reset() call (regardless of what the caller
    requests) to re-seed with the same fixed value, so the layout is identical
    across all episodes.
    """

    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._static_seed = seed

    def reset(self, **kwargs):
        kwargs["seed"] = self._static_seed
        return self.env.reset(**kwargs)
