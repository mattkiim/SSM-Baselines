from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def _sample(env, action, episode_length):

    next_obs, reward, cost, terminated, truncated, info = env.step(action)

    if episode_length + 1 >= 400:
        truncated = True

    return next_obs, reward, cost, terminated, truncated, info

def evaluate(
    env,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    episodes: int = 10,
    max_episode_steps: int | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Evaluate a policy on a Gymnasium-compatible safety environment."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if max_episode_steps is not None and max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")

    returns, costs, lengths, violation_rates = [], [], [], []
    for episode in range(episodes):
        obs, info = env.reset() if seed is None else env.reset(seed=seed + episode)
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        violation_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, cost, terminated, truncated, info = env.step(action)
            # obs, reward, cost, terminated, truncated, info = _sample(env, action, ep_len)
            ep_ret += float(reward)
            cost_value = float(np.asarray(cost))
            ep_cost += cost_value
            ep_len += 1
            violation_steps += int(cost_value > 0.0)
            if max_episode_steps is not None and ep_len >= max_episode_steps:
                truncated = True

        returns.append(ep_ret)
        costs.append(ep_cost)
        lengths.append(ep_len)
        if ep_len > 0:
            violation_rates.append(violation_steps / ep_len)
        else:
            violation_rates.append(0.0)

    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/cost_mean": float(np.mean(costs)),
        "eval/cost_std": float(np.std(costs)),
        "eval/ep_len_mean": float(np.mean(lengths)),
        "eval/violation_rate_mean": float(np.mean(violation_rates)),
        "eval/violation_rate_std": float(np.std(violation_rates)),
    }
