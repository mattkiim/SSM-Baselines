from __future__ import annotations

from typing import Callable, Dict, Optional

import gymnasium as gym
import numpy as np

from jaxrl5.envs.quadrotor_stabilization_2d import FIXED_PAPER_START


def sample_lowz_hrej_initial_states_stab(
    env: gym.Env,
    n: int = 200,
    seed: int = 0,
    z_high: float = 0.50,
) -> np.ndarray:
    """Sample the lowz050_hrej evaluation suite for Quad2D stabilization."""
    if hasattr(env.unwrapped, "seed"):
        env.unwrapped.seed(seed)
    states = []
    for _ in range(n):
        states.append(env.unwrapped.sample_lowz_hrej_state(z_high=z_high))
    return np.asarray(states, dtype=np.float32)


def make_fixed_vel_small_states(n: int = 200, seed: int = 0) -> np.ndarray:
    """Fixed-start multimodal panel with small velocity perturbations."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n, 6), dtype=np.float32)
    states[:, 0] = FIXED_PAPER_START["init_x"]
    states[:, 2] = FIXED_PAPER_START["init_z"]
    states[:, 4] = FIXED_PAPER_START["init_theta"]
    states[:, 5] = FIXED_PAPER_START["init_omega"]
    states[:, 1] = rng.uniform(-0.10, 0.10, size=n)
    states[:, 3] = rng.uniform(-0.05, 0.05, size=n)
    return states


def reset_to_state(env: gym.Env, state: np.ndarray):
    return env.reset(
        options={
            "init_x": float(state[0]),
            "init_vx": float(state[1]),
            "init_z": float(state[2]),
            "init_vz": float(state[3]),
            "init_theta": float(state[4]),
            "init_omega": float(state[5]),
        }
    )


def evaluate_initial_states(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    initial_states: np.ndarray,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    returns = []
    costs = []
    lengths = []
    successes = []
    collisions = []
    max_goal_streaks = []

    for state in np.asarray(initial_states, dtype=np.float32):
        obs, _ = reset_to_state(env, state)
        ep_ret = 0.0
        ep_cost = 0.0
        steps = 0
        last_info: Dict[str, float] = {}
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, cost, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_cost += float(cost)
            steps += 1
            last_info = info
            if max_steps is not None and steps >= max_steps:
                break
        returns.append(ep_ret)
        costs.append(ep_cost)
        lengths.append(steps)
        successes.append(float(last_info.get("success", 0.0)))
        collisions.append(float(last_info.get("collision", 0.0)))
        max_goal_streaks.append(float(last_info.get("episode_max_goal_streak", 0.0)))

    returns_arr = np.asarray(returns, dtype=np.float32)
    costs_arr = np.asarray(costs, dtype=np.float32)
    lengths_arr = np.asarray(lengths, dtype=np.float32)
    successes_arr = np.asarray(successes, dtype=np.float32)
    collisions_arr = np.asarray(collisions, dtype=np.float32)
    goal_streaks_arr = np.asarray(max_goal_streaks, dtype=np.float32)
    violation_rates = costs_arr / np.maximum(lengths_arr, 1.0)

    return {
        "eval/return_mean": float(returns_arr.mean()),
        "eval/return_std": float(returns_arr.std()),
        "eval/cost_mean": float(costs_arr.mean()),
        "eval/cost_std": float(costs_arr.std()),
        "eval/violation_rate_mean": float(violation_rates.mean()),
        "eval/violation_rate_std": float(violation_rates.std()),
        "eval/ep_len_mean": float(lengths_arr.mean()),
        "eval/ep_len_std": float(lengths_arr.std()),
        "eval/success_rate": float(successes_arr.mean()),
        "eval/collision_rate": float(collisions_arr.mean()),
        "eval/max_goal_streak_mean": float(goal_streaks_arr.mean()),
    }
