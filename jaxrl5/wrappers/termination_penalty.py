from __future__ import annotations

import gymnasium as gym


class TerminationPenaltyWrapper(gym.Wrapper):
    """
    If an episode terminates (terminated=True), add a large negative reward
    to discourage 'early death' as an exploitation to avoid future costs.

    - Does NOT change cost.
    - Keeps the same step() return format as the wrapped env (5-tuple or 6-tuple).
    """

    def __init__(
        self,
        env: gym.Env,
        penalty: float = -200.0,
        apply_on_truncated: bool = False,
        info_key: str = "termination_penalty",
    ):
        super().__init__(env)
        self.penalty = float(penalty)
        self.apply_on_truncated = bool(apply_on_truncated)
        self.info_key = str(info_key)

    def step(self, action):
        out = self.env.step(action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            do_penalty = bool(terminated) or (self.apply_on_truncated and bool(truncated))
            if do_penalty:
                info = dict(info)
                info["reward_before_penalty"] = float(reward)
                info[self.info_key] = float(self.penalty)
                reward = float(reward) + self.penalty
            return obs, reward, terminated, truncated, info

        if len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            do_penalty = bool(terminated) or (self.apply_on_truncated and bool(truncated))
            if do_penalty:
                info = dict(info)
                info["reward_before_penalty"] = float(reward)
                info[self.info_key] = float(self.penalty)
                reward = float(reward) + self.penalty
            return obs, reward, cost, terminated, truncated, info

        raise RuntimeError(f"Unexpected step() return length: {len(out)}")
