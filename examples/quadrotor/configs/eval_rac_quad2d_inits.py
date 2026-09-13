#!/usr/bin/env python3
"""Evaluate a QuadrotorTracking2D-v0 RAC policy from saved initial states."""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jaxrl5.agents.rac.rac_learner import RACLearner
from jaxrl5.envs.registration import ensure_custom_envs_registered
from jaxrl5.envs.quadrotor_tracking_2d import make_quadrotor_tracking_2d_env
from jaxrl5.tools.load_rac import load_rac
from jaxrl5.wrappers import AddCostFromInfo
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


def sdf_quad2d(x: float, z: float) -> float:
    return max(0.5 - z, z - 1.5, abs(x) - 2.0, abs(z) - 3.0)


def create_env(seed: int) -> gym.Env:
    ensure_custom_envs_registered()
    env = make_quadrotor_tracking_2d_env()
    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(env.action_space.high, 1.0):
        env = SymmetricActionWrapper(env)
    env = AddCostFromInfo(env)
    env.reset(seed=seed)
    return env


def load_agent(ckpt_path: str, step: int | None, env: gym.Env, seed: int) -> RACLearner:
    agent, _, _ = load_rac(
        ckpt_path,
        step=step,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
        deterministic=False,
    )
    return agent


def make_eval_policy(agent: RACLearner) -> Callable[[np.ndarray], np.ndarray]:
    eval_agent = agent

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(np.asarray(obs, dtype=np.float32))
        return np.asarray(action, dtype=np.float32)

    return policy


def obs_from_init_state(env: gym.Env, init_state: np.ndarray) -> np.ndarray:
    state = np.asarray(init_state, dtype=np.float32)
    unwrapped = env.unwrapped
    waypoint_idx = unwrapped._nearest_waypoint_idx(float(state[0]), float(state[2]))
    ref = np.asarray(unwrapped._waypoints[waypoint_idx], dtype=np.float32)
    return np.concatenate([state, ref]).astype(np.float32)


def estimate_vh_for_inits(
    agent: RACLearner,
    env: gym.Env,
    init_states: np.ndarray,
    K: int,
    rng: jax.random.PRNGKey,
) -> np.ndarray:
    obs_batch = np.stack([obs_from_init_state(env, s) for s in init_states], axis=0)
    obs_tiled = jnp.asarray(np.repeat(obs_batch, int(K), axis=0))
    dist = agent.actor.apply_fn({"params": agent.actor.params}, obs_tiled)
    actions = dist.sample(seed=rng)
    qh = agent.safety_critic.apply_fn(
        {"params": agent.safety_critic.params},
        obs_tiled,
        actions,
        training=False,
    )
    qh = qh.reshape(obs_batch.shape[0], int(K))
    return np.asarray(jnp.min(qh, axis=1), dtype=np.float32)


def rollout_from_init_state(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    init_state: np.ndarray,
    seed: int,
) -> Dict[str, np.ndarray]:
    obs, info = env.reset(
        seed=seed,
        options={
            "init_x": float(init_state[0]),
            "init_vx": float(init_state[1]),
            "init_z": float(init_state[2]),
            "init_vz": float(init_state[3]),
            "init_theta": float(init_state[4]),
            "init_omega": float(init_state[5]),
        },
    )

    Q = np.asarray(env.unwrapped.Q, dtype=np.float32)
    R = np.asarray(env.unwrapped.R, dtype=np.float32)
    a_ref = np.asarray(env.unwrapped.a_ref, dtype=np.float32)

    states, refs, actions = [np.asarray(obs[:6], dtype=np.float32)], [], []
    rewards, costs = [], []
    sdf_costs, h_alt_list = [], []
    alt_violations, oob_violations = [], []
    weighted_state_errors, unweighted_state_errors = [], []
    weighted_action_errors, unweighted_action_errors = [], []

    # Record initial state position for plotting (matches SSM script's xs/zs lists,
    # which include the starting state before any action is taken).
    xs = [float(obs[0])]
    zs = [float(obs[2])]

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = policy_fn(obs)

        actions.append(action)

        obs, reward, cost, terminated, truncated, info = env.step(action)
        state = np.asarray(obs[:6], dtype=np.float32)
        ref = np.asarray(obs[6:], dtype=np.float32)
        state_err = state - ref
        action_err = action - a_ref

        x = float(state[0])
        z = float(state[2])
        h_alt = max(0.5 - z, z - 1.5)
        h_sdf = sdf_quad2d(x, z)

        weighted_state_errors.append(float(state_err @ Q @ state_err))
        unweighted_state_errors.append(float(np.sum(state_err ** 2)))
        weighted_action_errors.append(float(action_err @ R @ action_err))
        unweighted_action_errors.append(float(np.sum(action_err ** 2)))

        states.append(state)
        refs.append(ref)
        rewards.append(float(reward))
        costs.append(float(cost))
        sdf_costs.append(float(h_sdf > 0.0))
        h_alt_list.append(float(h_alt))
        alt_violations.append(float(h_alt > 0.0))
        oob_violations.append(float((abs(x) > 2.0) or (abs(z) > 3.0)))

        xs.append(x)
        zs.append(z)

    return {
        "states": np.asarray(states, dtype=np.float32),
        "refs": np.asarray(refs, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "sdf_costs": np.asarray(sdf_costs, dtype=np.float32),
        "h_alt": np.asarray(h_alt_list, dtype=np.float32),
        "alt_violations": np.asarray(alt_violations, dtype=np.float32),
        "oob_violations": np.asarray(oob_violations, dtype=np.float32),
        "weighted_state_errors": np.asarray(weighted_state_errors, dtype=np.float32),
        "unweighted_state_errors": np.asarray(unweighted_state_errors, dtype=np.float32),
        "weighted_action_errors": np.asarray(weighted_action_errors, dtype=np.float32),
        "unweighted_action_errors": np.asarray(unweighted_action_errors, dtype=np.float32),
        "x": np.asarray(xs, dtype=np.float32),
        "z": np.asarray(zs, dtype=np.float32),
        "terminated": bool(terminated),
    }


def load_init_states(path: str, key: str, num_episodes: int | None) -> np.ndarray:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"{path} missing key={key!r}. Available keys: {list(data.keys())}")

    states = np.asarray(data[key], dtype=np.float32)

    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(f"Expected init states shape (N, 6), got {states.shape}")

    if num_episodes is not None:
        states = states[: int(num_episodes)]

    return states


def aggregate_ssm_style_metrics(
    rollouts: List[Dict[str, np.ndarray]],
    max_steps: int,
    init_states: np.ndarray,
    init_vh_values: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray | float | int | str]:
    rewards = np.asarray([r["rewards"].sum() for r in rollouts], dtype=np.float32)
    env_total_costs = np.asarray([r["costs"].sum() for r in rollouts], dtype=np.float32)
    total_costs = np.asarray([r["sdf_costs"].sum() for r in rollouts], dtype=np.float32)
    crashes = np.asarray([r["terminated"] for r in rollouts], dtype=np.float32)
    lengths = np.asarray([len(r["rewards"]) for r in rollouts], dtype=np.float32)

    alt_violation_rates = np.asarray(
        [r["alt_violations"].mean() if len(r["alt_violations"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    cost_rates = np.asarray(
        [r["sdf_costs"].mean() if len(r["sdf_costs"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    env_cost_rates = np.asarray(
        [r["costs"].mean() if len(r["costs"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    oob_rates = np.asarray(
        [r["oob_violations"].mean() if len(r["oob_violations"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )

    episode_has_alt_violation = np.asarray(
        [np.any(r["alt_violations"] > 0) for r in rollouts],
        dtype=np.float32,
    )
    episode_has_cost = np.asarray(
        [np.any(r["sdf_costs"] > 0) for r in rollouts],
        dtype=np.float32,
    )
    if init_vh_values is None:
        init_vh_values = np.full((len(rollouts),), np.nan, dtype=np.float32)
    init_vh_values = np.asarray(init_vh_values, dtype=np.float32)
    actual_positive = ~episode_has_cost.astype(bool)
    has_vh = np.isfinite(init_vh_values)
    predicted_positive = has_vh & (init_vh_values < 0.0)
    predicted_negative = has_vh & ~predicted_positive
    tp_mask = predicted_positive & actual_positive
    fp_mask = predicted_positive & ~actual_positive
    tn_mask = predicted_negative & ~actual_positive
    fn_mask = predicted_negative & actual_positive
    cm_n = int(np.sum(has_vh))
    cm_tp = int(np.sum(tp_mask))
    cm_fp = int(np.sum(fp_mask))
    cm_tn = int(np.sum(tn_mask))
    cm_fn = int(np.sum(fn_mask))
    cm_actual_pos = int(np.sum(has_vh & actual_positive))
    cm_actual_neg = int(np.sum(has_vh & ~actual_positive))

    weighted_state_error_means = np.asarray(
        [r["weighted_state_errors"].mean() if len(r["weighted_state_errors"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    unweighted_state_error_means = np.asarray(
        [r["unweighted_state_errors"].mean() if len(r["unweighted_state_errors"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    weighted_action_error_means = np.asarray(
        [r["weighted_action_errors"].mean() if len(r["weighted_action_errors"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    unweighted_action_error_means = np.asarray(
        [r["unweighted_action_errors"].mean() if len(r["unweighted_action_errors"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )

    fixed_T = int(max_steps) * len(rollouts)
    actual_T = int(sum(len(r["alt_violations"]) for r in rollouts))
    total_alt_violations = float(sum(r["alt_violations"].sum() for r in rollouts))
    total_cost_steps = float(sum(r["sdf_costs"].sum() for r in rollouts))
    total_env_cost_steps = float(sum(r["costs"].sum() for r in rollouts))

    return {
        "label": "RAC",
        "num_episodes": len(rollouts),
        "init_states": np.asarray(init_states, dtype=np.float32),
        "returns": rewards,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "return_mean": float(rewards.mean()),
        "return_std": float(rewards.std()),
        "cost_sums": total_costs,
        "cost_sum_mean": float(total_costs.mean()),
        "cost_sum_std": float(total_costs.std()),
        "env_cost_sums": env_total_costs,
        "env_cost_sum_mean": float(env_total_costs.mean()),
        "env_cost_sum_std": float(env_total_costs.std()),
        "violation_rates": env_cost_rates,
        "violation_rate_mean": float(env_cost_rates.mean()),
        "lengths": lengths,
        "length_mean": float(lengths.mean()),
        "alt_violation_rate_mean": float(alt_violation_rates.mean()),
        "alt_violation_rate_std": float(alt_violation_rates.std()),
        "cost_rate_mean": float(cost_rates.mean()),
        "cost_rate_std": float(cost_rates.std()),
        "oob_rate_mean": float(oob_rates.mean()),
        "oob_rate_std": float(oob_rates.std()),
        "alt_viol_rate_actual_T": float(total_alt_violations / actual_T) if actual_T > 0 else 0.0,
        "cost_rate_actual_T": float(total_cost_steps / actual_T) if actual_T > 0 else 0.0,
        "env_cost_rate_actual_T": float(total_env_cost_steps / actual_T) if actual_T > 0 else 0.0,
        "alt_viol_rate_fixed_T": float(total_alt_violations / fixed_T) if fixed_T > 0 else 0.0,
        "cost_rate_fixed_T": float(total_cost_steps / fixed_T) if fixed_T > 0 else 0.0,
        "env_cost_rate_fixed_T": float(total_env_cost_steps / fixed_T) if fixed_T > 0 else 0.0,
        "episode_alt_violation_rate": float(episode_has_alt_violation.mean()),
        "episode_cost_rate": float(episode_has_cost.mean()),
        "crash_rate": float(crashes.mean()),
        "crash_frac": float(crashes.mean()),
        "alt_violation_rates": alt_violation_rates,
        "cost_rates": cost_rates,
        "env_cost_rates": env_cost_rates,
        "oob_rates": oob_rates,
        "episode_has_alt_violation": episode_has_alt_violation,
        "episode_has_cost": episode_has_cost,
        "init_vh_values": init_vh_values,
        "confusion_n": cm_n,
        "confusion_tp": cm_tp,
        "confusion_fp": cm_fp,
        "confusion_tn": cm_tn,
        "confusion_fn": cm_fn,
        "confusion_actual_positive": cm_actual_pos,
        "confusion_actual_negative": cm_actual_neg,
        "confusion_predicted_positive": int(np.sum(predicted_positive)),
        "confusion_predicted_negative": int(np.sum(predicted_negative)),
        "confusion_accuracy": float((cm_tp + cm_tn) / cm_n) if cm_n else float("nan"),
        "confusion_precision": float(cm_tp / (cm_tp + cm_fp)) if (cm_tp + cm_fp) else float("nan"),
        "confusion_recall": float(cm_tp / (cm_tp + cm_fn)) if (cm_tp + cm_fn) else float("nan"),
        "confusion_fpr": float(cm_fp / cm_actual_neg) if cm_actual_neg else float("nan"),
        "confusion_fnr": float(cm_fn / cm_actual_pos) if cm_actual_pos else float("nan"),
        "confusion_actual_positive_mask": actual_positive,
        "confusion_predicted_positive_mask": predicted_positive,
        "confusion_tp_mask": tp_mask,
        "confusion_fp_mask": fp_mask,
        "confusion_tn_mask": tn_mask,
        "confusion_fn_mask": fn_mask,
        "weighted_state_error_means": weighted_state_error_means,
        "weighted_state_error_mean": float(weighted_state_error_means.mean()),
        "weighted_state_error_std": float(weighted_state_error_means.std()),
        "unweighted_state_error_means": unweighted_state_error_means,
        "unweighted_state_error_mean": float(unweighted_state_error_means.mean()),
        "unweighted_state_error_std": float(unweighted_state_error_means.std()),
        "weighted_action_error_means": weighted_action_error_means,
        "weighted_action_error_mean": float(weighted_action_error_means.mean()),
        "weighted_action_error_std": float(weighted_action_error_means.std()),
        "unweighted_action_error_means": unweighted_action_error_means,
        "unweighted_action_error_mean": float(unweighted_action_error_means.mean()),
        "unweighted_action_error_std": float(unweighted_action_error_means.std()),
    }


def plot_rollouts(
    rollouts: List[Dict[str, np.ndarray]],
    out_path: str,
    label: str = "RAC",
    best_n: int = 3,
) -> None:
    """Plot best-N trajectories overlaid on the altitude-band constraint and reference circle.

    Mirrors the SSM eval script's `plot` function.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Altitude-band violation regions (z < 0.5 or z > 1.5) and bounds.
    x_line = np.linspace(-2.2, 2.2, 200)
    ax.fill_between(x_line, -0.2, 0.5, alpha=0.25)
    ax.fill_between(x_line, 1.5, 2.2, alpha=0.25)
    ax.plot(x_line, np.full_like(x_line, 0.5), "k-", lw=2)
    ax.plot(x_line, np.full_like(x_line, 1.5), "k-", lw=2)

    # Reference circle: center=(0, 1), radius=1.
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), 1 + np.sin(theta), "k--", lw=1.5, label="Reference")

    # Sort by (terminated, total_cost, -total_reward) to pick "best" rollouts,
    # matching the SSM script's ranking.
    ranked = sorted(
        rollouts,
        key=lambda r: (
            bool(r["terminated"]),
            float(r["sdf_costs"].sum()),
            -float(r["rewards"].sum()),
        ),
    )

    for j, r in enumerate(ranked[:best_n]):
        x = r["x"]
        z = r["z"]
        ax.plot(x, z, lw=2.5, label=label if j == 0 else None)
        ax.plot(x[0], z[0], "o")
        if r["terminated"]:
            ax.plot(x[-1], z[-1], "x", markersize=10)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.2, 2.2)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="results/evaluations/eval_quad2d_from_inits")

    parser.add_argument("--init_npz", required=True)
    parser.add_argument("--init_key", default="init_states")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--confusion_matrix", action="store_true")
    parser.add_argument("--K", type=int, default=64, help="Number of sampled actions for Vh(init)=min_a Qh(init,a).")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = create_env(args.seed)
    agent = load_agent(args.ckpt_path, args.step, env, args.seed)
    policy_fn = make_eval_policy(agent)

    init_states = load_init_states(args.init_npz, args.init_key, args.num_episodes)
    init_vh = (
        estimate_vh_for_inits(agent, env, init_states, args.K, jax.random.PRNGKey(args.seed))
        if args.confusion_matrix
        else np.full((len(init_states),), np.nan, dtype=np.float32)
    )

    rollouts: List[Dict[str, np.ndarray]] = []

    for i, init_state in enumerate(init_states):
        print(f"rollout {i + 1}/{len(init_states)}")
        rollouts.append(
            rollout_from_init_state(
                env=env,
                policy_fn=policy_fn,
                init_state=init_state,
                seed=args.seed + i,
            )
        )

    metrics = aggregate_ssm_style_metrics(
        rollouts,
        max_steps=int(env.unwrapped.max_episode_steps),
        init_states=init_states,
        init_vh_values=init_vh,
    )
    if args.confusion_matrix:
        print(
            "Confusion(Vh<0 vs rollout safe): "
            f"TP={metrics['confusion_tp']} "
            f"FP={metrics['confusion_fp']} "
            f"TN={metrics['confusion_tn']} "
            f"FN={metrics['confusion_fn']} "
            f"Acc={metrics['confusion_accuracy']:.4f} "
            f"Prec={metrics['confusion_precision']:.4f} "
            f"Recall={metrics['confusion_recall']:.4f}"
        )

    # Save aggregate stats plus per-episode trajectory data for downstream re-plotting.
    save_dict: Dict[str, np.ndarray] = dict(metrics)
    for i, r in enumerate(rollouts):
        save_dict[f"states_{i}"] = r["states"]
        save_dict[f"refs_{i}"] = r["refs"]
        save_dict[f"actions_{i}"] = r["actions"]
        save_dict[f"rewards_{i}"] = r["rewards"]
        save_dict[f"costs_{i}"] = r["costs"]
        save_dict[f"sdf_costs_{i}"] = r["sdf_costs"]
        save_dict[f"h_alt_{i}"] = r["h_alt"]
        save_dict[f"alt_violations_{i}"] = r["alt_violations"]
        save_dict[f"oob_violations_{i}"] = r["oob_violations"]
        save_dict[f"weighted_state_errors_{i}"] = r["weighted_state_errors"]
        save_dict[f"unweighted_state_errors_{i}"] = r["unweighted_state_errors"]
        save_dict[f"weighted_action_errors_{i}"] = r["weighted_action_errors"]
        save_dict[f"unweighted_action_errors_{i}"] = r["unweighted_action_errors"]
        save_dict[f"x_{i}"] = r["x"]
        save_dict[f"z_{i}"] = r["z"]
        save_dict[f"total_reward_{i}"] = np.float32(r["rewards"].sum())
        save_dict[f"total_cost_{i}"] = np.float32(r["sdf_costs"].sum())
        save_dict[f"env_total_cost_{i}"] = np.float32(r["costs"].sum())
        save_dict[f"terminated_{i}"] = np.bool_(r["terminated"])

    np.savez(os.path.join(args.out_dir, "rollouts_from_inits.npz"), **save_dict)

    # Plot best-N trajectories.
    plot_rollouts(
        rollouts,
        out_path=os.path.join(args.out_dir, "trajectories.png"),
        label="RAC",
        best_n=100,
    )

    summary = {
        "init_npz": args.init_npz,
        "init_key": args.init_key,
    }
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            continue
        summary[key] = value

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
