#!/usr/bin/env python3
"""Evaluate a QuadrotorTracking2D-v0 policy.

This script does three things:
1. Optionally samples and rolls out V_h-accepted initial states.
2. Plots the V_h<0 contour on (x, z) slices using the learned safety critic.
3. Prints the Q and R used by the environment and reports both weighted and
   unweighted tracking error, so different reward scalings can be compared fairly.

Supports RAC checkpoints directly. The V_h plotting code is adapted from the
existing plot_vh_region utility.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization
from flax.core import frozen_dict

from jaxrl5.agents.rac.rac_learner import RACLearner
from jaxrl5.envs.registration import ensure_custom_envs_registered
from jaxrl5.envs.quadrotor_tracking_2d import make_quadrotor_tracking_2d_env
from jaxrl5.tools.checkpoints import resolve_checkpoint
from jaxrl5.tools.load_rac import load_rac
from jaxrl5.wrappers import AddCostFromInfo
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


def parse_float_list(text: str) -> List[float]:
    return [float(t.strip()) for t in text.split(",") if t.strip()]


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


def get_qr(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
    unwrapped = env.unwrapped
    Q = np.asarray(unwrapped.Q, dtype=np.float32)
    R = np.asarray(unwrapped.R, dtype=np.float32)
    return Q, R


def make_eval_policy(agent: RACLearner) -> Callable[[np.ndarray], np.ndarray]:
    eval_agent = agent

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(np.asarray(obs, dtype=np.float32))
        return np.asarray(action, dtype=np.float32)

    return policy


# ------------------------- rollout / error evaluation -------------------------

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

    states = []
    refs = []
    actions = []
    rewards = []
    costs = []
    sdf_costs = []
    h_alt_list = []
    alt_violations = []
    oob_violations = []
    weighted_state_errors = []
    unweighted_state_errors = []
    weighted_action_errors = []
    unweighted_action_errors = []

    Q, R = get_qr(env)
    a_ref = np.asarray(env.unwrapped.a_ref, dtype=np.float32)
    xs = [float(obs[0])]
    zs = [float(obs[2])]

    terminated = False
    truncated = False
    while not (terminated or truncated):
        state = np.asarray(obs[:6], dtype=np.float32)
        ref = np.asarray(obs[6:], dtype=np.float32)
        act = policy_fn(obs)

        state_err = state - ref
        action_err = act - a_ref

        weighted_state_errors.append(float(state_err @ Q @ state_err))
        unweighted_state_errors.append(float(np.sum(state_err ** 2)))
        weighted_action_errors.append(float(action_err @ R @ action_err))
        unweighted_action_errors.append(float(np.sum(action_err ** 2)))

        states.append(state)
        refs.append(ref)
        actions.append(act)

        obs, reward, cost, terminated, truncated, info = env.step(act)
        next_state = np.asarray(obs[:6], dtype=np.float32)
        x = float(next_state[0])
        z = float(next_state[2])
        h_alt = max(0.5 - z, z - 1.5)
        h_sdf = sdf_quad2d(x, z)

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
        "x": np.asarray(xs, dtype=np.float32),
        "z": np.asarray(zs, dtype=np.float32),
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
        "terminated": bool(terminated),
        "Q": Q,
        "R": R,
    }


def plot_rollouts(
    rollouts: List[Dict[str, np.ndarray]],
    out_path: str,
    label: str = "RAC",
    best_n: int = 100,
) -> None:
    """Plot best-N trajectories like eval_rac_quad2d_inits.py."""
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(7, 6))

    x_line = np.linspace(-2.2, 2.2, 200)
    ax.fill_between(x_line, -0.2, 0.5, alpha=0.25)
    ax.fill_between(x_line, 1.5, 2.2, alpha=0.25)
    ax.plot(x_line, np.full_like(x_line, 0.5), "k-", lw=2)
    ax.plot(x_line, np.full_like(x_line, 1.5), "k-", lw=2)
    ax.add_patch(Circle((0.0, 1.0), radius=1.0, fill=False, edgecolor="black", linestyle="--", linewidth=1.5))

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


def aggregate_ssm_style_metrics(
    rollouts: List[Dict[str, np.ndarray]],
    max_steps: int,
    label: str,
    init_states: Optional[np.ndarray] = None,
    init_vh_values: Optional[np.ndarray] = None,
    safe_threshold: Optional[float] = None,
) -> Dict[str, np.ndarray | float | int | str]:
    rewards = np.asarray([r["rewards"].sum() for r in rollouts], dtype=np.float32)
    env_total_costs = np.asarray([r["costs"].sum() for r in rollouts], dtype=np.float32)
    total_costs = np.asarray([r["sdf_costs"].sum() for r in rollouts], dtype=np.float32)
    crashes = np.asarray([r["terminated"] for r in rollouts], dtype=np.float32)

    alt_violation_rates = np.asarray(
        [r["alt_violations"].mean() if len(r["alt_violations"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    cost_rates = np.asarray(
        [r["sdf_costs"].mean() if len(r["sdf_costs"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    oob_rates = np.asarray(
        [r["oob_violations"].mean() if len(r["oob_violations"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    unweighted_state_error_means = np.asarray(
        [
            r["unweighted_state_errors"].mean() if len(r["unweighted_state_errors"]) else 0.0
            for r in rollouts
        ],
        dtype=np.float32,
    )
    weighted_state_error_means = np.asarray(
        [
            r["weighted_state_errors"].mean() if len(r["weighted_state_errors"]) else 0.0
            for r in rollouts
        ],
        dtype=np.float32,
    )
    weighted_action_error_means = np.asarray(
        [
            r["weighted_action_errors"].mean() if len(r["weighted_action_errors"]) else 0.0
            for r in rollouts
        ],
        dtype=np.float32,
    )
    unweighted_action_error_means = np.asarray(
        [
            r["unweighted_action_errors"].mean() if len(r["unweighted_action_errors"]) else 0.0
            for r in rollouts
        ],
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

    fixed_T = int(max_steps) * len(rollouts)
    total_alt_violations = float(sum(r["alt_violations"].sum() for r in rollouts))
    total_cost_steps = float(sum(r["sdf_costs"].sum() for r in rollouts))
    actual_T = int(sum(len(r["alt_violations"]) for r in rollouts))

    if init_states is None:
        init_states = np.asarray([r["states"][0] for r in rollouts], dtype=np.float32)
    if init_vh_values is None:
        init_vh_values = np.full((len(rollouts),), np.nan, dtype=np.float32)
    init_vh_values = np.asarray(init_vh_values, dtype=np.float32)
    vh_threshold = float(safe_threshold) if safe_threshold is not None else 0.0
    actual_positive = ~episode_has_cost.astype(bool)
    has_vh = np.isfinite(init_vh_values)
    predicted_positive = has_vh & (init_vh_values < vh_threshold)
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

    return {
        "label": label,
        "num_episodes": len(rollouts),
        "init_states": np.asarray(init_states, dtype=np.float32),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "cost_sum_mean": float(total_costs.mean()),
        "cost_sum_std": float(total_costs.std()),
        "env_cost_sum_mean": float(env_total_costs.mean()),
        "env_cost_sum_std": float(env_total_costs.std()),
        "alt_violation_rate_mean": float(alt_violation_rates.mean()),
        "alt_violation_rate_std": float(alt_violation_rates.std()),
        "cost_rate_mean": float(cost_rates.mean()),
        "cost_rate_std": float(cost_rates.std()),
        "oob_rate_mean": float(oob_rates.mean()),
        "oob_rate_std": float(oob_rates.std()),
        "alt_viol_rate_actual_T": float(total_alt_violations / actual_T) if actual_T > 0 else 0.0,
        "cost_rate_actual_T": float(total_cost_steps / actual_T) if actual_T > 0 else 0.0,
        "alt_viol_rate_fixed_T": float(total_alt_violations / fixed_T) if fixed_T > 0 else 0.0,
        "cost_rate_fixed_T": float(total_cost_steps / fixed_T) if fixed_T > 0 else 0.0,
        "episode_alt_violation_rate": float(episode_has_alt_violation.mean()),
        "episode_cost_rate": float(episode_has_cost.mean()),
        "crash_frac": float(crashes.mean()),
        "alt_violation_rates": alt_violation_rates,
        "cost_rates": cost_rates,
        "oob_rates": oob_rates,
        "episode_has_alt_violation": episode_has_alt_violation,
        "episode_has_cost": episode_has_cost,
        "unweighted_state_error_mean": float(unweighted_state_error_means.mean()),
        "unweighted_state_error_std": float(unweighted_state_error_means.std()),
        "unweighted_state_error_means": unweighted_state_error_means,
        "weighted_state_error_mean": float(weighted_state_error_means.mean()),
        "weighted_state_error_std": float(weighted_state_error_means.std()),
        "weighted_state_error_means": weighted_state_error_means,
        "weighted_action_error_mean": float(weighted_action_error_means.mean()),
        "weighted_action_error_std": float(weighted_action_error_means.std()),
        "weighted_action_error_means": weighted_action_error_means,
        "unweighted_action_error_mean": float(unweighted_action_error_means.mean()),
        "unweighted_action_error_std": float(unweighted_action_error_means.std()),
        "unweighted_action_error_means": unweighted_action_error_means,
        "init_vh_values": init_vh_values,
        "vh_safe_threshold": vh_threshold,
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
    }

# ------------------------------ Vh contour plot -------------------------------

def build_obs_batch(grid_x: np.ndarray, grid_z: np.ndarray, z_dot: float, ref: np.ndarray) -> np.ndarray:
    B = grid_x.size
    obs = np.zeros((B, 12), dtype=np.float32)
    obs[:, 0] = grid_x.reshape(-1)
    obs[:, 2] = grid_z.reshape(-1)
    obs[:, 3] = float(z_dot)
    obs[:, 6:] = ref[None, :]
    return obs


def sample_actions(agent: RACLearner, obs_tiled: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
    dist = agent.actor.apply_fn({"params": agent.actor.params}, obs_tiled)
    return dist.sample(seed=rng)


def eval_qh_batch(agent: RACLearner, obs_tiled: jnp.ndarray, actions_flat: jnp.ndarray, batch_chunk: int) -> jnp.ndarray:
    total = obs_tiled.shape[0]
    if batch_chunk <= 0 or batch_chunk >= total:
        qh = agent.safety_critic.apply_fn(
            {"params": agent.safety_critic.params},
            obs_tiled,
            actions_flat,
            training=False,
        )
        return qh

    outputs = []
    for start in range(0, total, batch_chunk):
        end = min(start + batch_chunk, total)
        qh = agent.safety_critic.apply_fn(
            {"params": agent.safety_critic.params},
            obs_tiled[start:end],
            actions_flat[start:end],
            training=False,
        )
        outputs.append(qh)
    return jnp.concatenate(outputs, axis=0)


def compute_vh_for_slice(agent: RACLearner, obs_batch: np.ndarray, K: int, rng: jax.random.PRNGKey, batch_chunk: int) -> np.ndarray:
    B = obs_batch.shape[0]
    obs_tiled = np.repeat(obs_batch, K, axis=0)
    obs_tiled = jnp.asarray(obs_tiled)
    actions_flat = sample_actions(agent, obs_tiled, rng)
    qh_flat = eval_qh_batch(agent, obs_tiled, actions_flat, batch_chunk)
    qh_flat = qh_flat.reshape(B, K)
    vh_flat = qh_flat.min(axis=1)
    return np.asarray(vh_flat)


def obs_from_init_state(env: gym.Env, init_state: np.ndarray) -> np.ndarray:
    """Build the tracking observation for a proposed 6D initial state."""
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
    batch_chunk: int,
) -> np.ndarray:
    obs_batch = np.stack([obs_from_init_state(env, s) for s in init_states], axis=0)
    return compute_vh_for_slice(agent, obs_batch, K, rng, batch_chunk)


def quad2d_init_bounds() -> Tuple[np.ndarray, np.ndarray]:
    low = np.array([-1.5, -1.0, 0.5, -1.5, -0.2, -0.1], dtype=np.float32)
    high = np.array([1.5, 1.0, 1.5, 1.5, 0.2, 0.1], dtype=np.float32)
    return low, high


def sample_random_inits(num_inits: int, seed: int) -> np.ndarray:
    low, high = quad2d_init_bounds()
    rng_np = np.random.default_rng(seed)
    return rng_np.uniform(low, high, size=(num_inits, low.size)).astype(np.float32)


def sample_vh_rejected_inits(
    agent: RACLearner,
    env: gym.Env,
    num_inits: int,
    seed: int,
    K: int,
    threshold: float,
    batch_chunk: int,
    proposal_batch: int,
    max_tries: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Rejection-sample init states from the env reset box using estimated V_h."""
    low, high = quad2d_init_bounds()
    rng_np = np.random.default_rng(seed)

    accepted: List[np.ndarray] = []
    accepted_vh: List[float] = []
    tries = 0
    key = jax.random.PRNGKey(seed)

    while len(accepted) < num_inits and tries < max_tries:
        remaining_tries = max_tries - tries
        batch_n = min(max(1, int(proposal_batch)), remaining_tries)
        candidates = rng_np.uniform(low, high, size=(batch_n, low.size)).astype(np.float32)
        tries += batch_n

        key, subkey = jax.random.split(key)
        vh = estimate_vh_for_inits(agent, env, candidates, K, subkey, batch_chunk)
        for candidate, candidate_vh in zip(candidates, vh):
            if float(candidate_vh) < threshold:
                accepted.append(candidate)
                accepted_vh.append(float(candidate_vh))
                if len(accepted) >= num_inits:
                    break

    if len(accepted) < num_inits:
        raise RuntimeError(
            f"VH rejection sampling accepted {len(accepted)}/{num_inits} "
            f"states after {tries} tries. Increase --vh_max_tries or relax --vh_threshold."
        )

    return (
        np.asarray(accepted, dtype=np.float32),
        np.asarray(accepted_vh, dtype=np.float32),
        tries,
    )


def get_ref_from_env(env: gym.Env, seed: int) -> np.ndarray:
    obs, _ = env.reset(seed=seed)
    return np.asarray(obs[6:], dtype=np.float32)


def plot_vh_grid(
    X: np.ndarray,
    Z: np.ndarray,
    vh_list: List[np.ndarray],
    z_dot_list: List[float],
    out_path: str,
    traj_xz: Optional[np.ndarray] = None,
) -> None:
    import numpy as _np
    import matplotlib.pyplot as _plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Circle

    cols = len(z_dot_list)
    xlim = (float(_np.min(X)), float(_np.max(X)))
    ylim = (float(_np.min(Z)), float(_np.max(Z)))
    x_range = max(xlim[1] - xlim[0], 1e-6)
    y_range = max(ylim[1] - ylim[0], 1e-6)
    axes_ratio = x_range / y_range

    all_vh = _np.concatenate([v.reshape(-1) for v in vh_list])
    all_vh = all_vh[_np.isfinite(all_vh)]
    max_abs = float(_np.nanmax(_np.abs(all_vh))) if all_vh.size else 1.0
    max_abs = max(max_abs, 1e-6)
    vmin, vmax = -max_abs, max_abs
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    levels = _np.linspace(vmin, vmax, 61)

    base_h = 4.8
    per_ax_w = base_h * axes_ratio
    fig_w = cols * per_ax_w + 1.2

    fig = _plt.figure(figsize=(fig_w, base_h))
    gs = fig.add_gridspec(1, cols + 1, width_ratios=[axes_ratio] * cols + [0.10], wspace=0.22)
    axes = [_plt.subplot(gs[0, i]) for i in range(cols)]
    cax = _plt.subplot(gs[0, cols])

    last_im = None
    for ax, vh, z_dot in zip(axes, vh_list, z_dot_list):
        last_im = ax.contourf(X, Z, vh, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
        ax.contour(X, Z, vh, levels=[0.0], colors=["orange"], linestyles="--", linewidths=2.0)
        ax.axhline(0.5, color="black", linewidth=2.0)
        ax.axhline(1.5, color="black", linewidth=2.0)
        ax.add_patch(Circle((0.0, 1.0), radius=1.0, fill=False, edgecolor="black", linestyle="--", linewidth=2.0))
        if traj_xz is not None and traj_xz.size > 0:
            ax.plot(traj_xz[:, 0], traj_xz[:, 1], color="lime", linewidth=2.5, label="rollout")
            ax.scatter(traj_xz[0, 0], traj_xz[0, 1], color="lime", s=28, marker="o", zorder=5)
            ax.scatter(traj_xz[-1, 0], traj_xz[-1, 1], color="lime", s=36, marker="x", zorder=5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(rf"$\dot{{z}} = {z_dot:g}$")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    fig.colorbar(last_im, cax=cax)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    _plt.close(fig)


# -------------------------------- checkpoint load -----------------------------

def load_agent(ckpt_path: str, step: Optional[int], env: gym.Env, seed: int) -> RACLearner:
    agent, _, _ = load_rac(
        ckpt_path,
        step=step,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
        deterministic=False,
    )
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Quad2D RAC policy and V_h contour.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/evaluations/eval_quad2d")
    parser.add_argument("--grid_n", type=int, default=101)
    parser.add_argument("--x_min", type=float, default=-1.5)
    parser.add_argument("--x_max", type=float, default=1.5)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--z_dot_list", type=str, default="-1,0,1")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--batch_chunk", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--reject_by_current_vh", action="store_true")
    parser.add_argument("--vh_num_inits", type=int, default=None)
    parser.add_argument("--vh_threshold", type=float, default=0.0)
    parser.add_argument("--vh_proposal_batch", type=int, default=256)
    parser.add_argument("--vh_max_tries", type=int, default=100_000)
    parser.add_argument("--confusion_matrix", action="store_true")
    parser.add_argument("--best_n", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = create_env(args.seed)
    agent = load_agent(args.ckpt_path, args.step, env, args.seed)
    policy_fn = make_eval_policy(agent)

    # Print Q & R explicitly.
    Q, R = get_qr(env)
    print("Q =")
    print(Q)
    print("R =")
    print(R)

    num_episodes = int(args.num_episodes)
    reject_by_current_vh = bool(args.reject_by_current_vh)
    if args.vh_num_inits is not None:
        num_episodes = int(args.vh_num_inits)
        reject_by_current_vh = True
        print("--vh_num_inits is deprecated; use --num_episodes with --reject_by_current_vh.")

    max_steps = int(env.unwrapped.max_episode_steps)
    eval_summary_metrics = None
    traj_xz = None
    sampling_summary = None
    if num_episodes > 0:
        if reject_by_current_vh:
            init_states, init_vh, sample_tries = sample_vh_rejected_inits(
                agent=agent,
                env=env,
                num_inits=num_episodes,
                seed=args.seed,
                K=args.K,
                threshold=args.vh_threshold,
                batch_chunk=args.batch_chunk,
                proposal_batch=args.vh_proposal_batch,
                max_tries=args.vh_max_tries,
            )
            print(
                f"VH rejection sampling accepted {len(init_states)} states "
                f"from {sample_tries} proposals "
                f"(rate={len(init_states) / sample_tries:.4f}, threshold={args.vh_threshold:g})"
            )
        else:
            init_states = sample_random_inits(num_episodes, args.seed)
            init_vh = np.full((num_episodes,), np.nan, dtype=np.float32)
            sample_tries = num_episodes
            print(f"Random init sampling selected {len(init_states)} states without VH rejection.")

        if args.confusion_matrix and not np.all(np.isfinite(init_vh)):
            init_vh = estimate_vh_for_inits(
                agent,
                env,
                init_states,
                args.K,
                jax.random.PRNGKey(args.seed),
                args.batch_chunk,
            )

        sampled_rollouts = [
            rollout_from_init_state(env, policy_fn, init_state, seed=args.seed + i)
            for i, init_state in enumerate(init_states)
        ]
        sampled_returns = np.asarray([r["rewards"].sum() for r in sampled_rollouts], dtype=np.float32)
        sampled_cost_sums = np.asarray([r["costs"].sum() for r in sampled_rollouts], dtype=np.float32)
        sampled_violation_rates = np.asarray(
            [r["costs"].mean() if len(r["costs"]) else 0.0 for r in sampled_rollouts],
            dtype=np.float32,
        )
        sampled_lengths = np.asarray([len(r["rewards"]) for r in sampled_rollouts], dtype=np.float32)
        sampled_metrics = aggregate_ssm_style_metrics(
            sampled_rollouts,
            max_steps=max_steps,
            label="RAC",
            init_states=init_states,
            init_vh_values=init_vh,
            safe_threshold=args.vh_threshold,
        )
        if args.confusion_matrix:
            print(
                "Confusion(Vh<0 vs rollout safe): "
                f"TP={sampled_metrics['confusion_tp']} "
                f"FP={sampled_metrics['confusion_fp']} "
                f"TN={sampled_metrics['confusion_tn']} "
                f"FN={sampled_metrics['confusion_fn']} "
                f"Acc={sampled_metrics['confusion_accuracy']:.4f} "
                f"Prec={sampled_metrics['confusion_precision']:.4f} "
                f"Recall={sampled_metrics['confusion_recall']:.4f}"
            )

        sampled_save: Dict[str, np.ndarray] = {
            "init_states": init_states,
            "init_vh": init_vh,
            "returns": sampled_returns,
            "cost_sums": sampled_cost_sums,
            "violation_rates": sampled_violation_rates,
            "lengths": sampled_lengths,
            "reject_by_current_vh": np.bool_(reject_by_current_vh),
            "threshold": np.float32(args.vh_threshold if reject_by_current_vh else np.nan),
            "tries": np.int64(sample_tries),
            "confusion_enabled": np.bool_(args.confusion_matrix),
        }
        sampled_save.update(sampled_metrics)
        for i, r in enumerate(sampled_rollouts):
            sampled_save[f"states_{i}"] = r["states"]
            sampled_save[f"refs_{i}"] = r["refs"]
            sampled_save[f"actions_{i}"] = r["actions"]
            sampled_save[f"x_{i}"] = r["x"]
            sampled_save[f"z_{i}"] = r["z"]
            sampled_save[f"rewards_{i}"] = r["rewards"]
            sampled_save[f"costs_{i}"] = r["costs"]
            sampled_save[f"sdf_costs_{i}"] = r["sdf_costs"]
            sampled_save[f"h_alt_{i}"] = r["h_alt"]
            sampled_save[f"alt_violations_{i}"] = r["alt_violations"]
            sampled_save[f"oob_violations_{i}"] = r["oob_violations"]
            sampled_save[f"unweighted_state_errors_{i}"] = r["unweighted_state_errors"]
            sampled_save[f"total_reward_{i}"] = np.float32(r["rewards"].sum())
            sampled_save[f"total_cost_{i}"] = np.float32(r["sdf_costs"].sum())
            sampled_save[f"terminated_{i}"] = np.bool_(r["terminated"])

        rollout_npz_name = "vh_sampled_rollouts.npz" if reject_by_current_vh else "sampled_rollouts.npz"
        np.savez(os.path.join(args.out_dir, rollout_npz_name), **sampled_save)
        eval_summary_metrics = sampled_metrics
        best_rollout = min(
            sampled_rollouts,
            key=lambda r: (
                bool(r["terminated"]),
                float(r["sdf_costs"].sum()),
                -float(r["rewards"].sum()),
            ),
        )
        traj_xz = best_rollout["states"][:, [0, 2]] if best_rollout["states"].size else None
        plot_rollouts(
            sampled_rollouts,
            out_path=os.path.join(args.out_dir, "trajectories.png"),
            label="RAC",
            best_n=args.best_n,
        )
        sampling_summary = {
            "num_episodes_requested": int(num_episodes),
            "reject_by_current_vh": bool(reject_by_current_vh),
            "sample_tries": int(sample_tries),
            "length_mean": float(sampled_lengths.mean()),
            "env_violation_rate_mean": float(sampled_violation_rates.mean()),
        }
        if reject_by_current_vh:
            sampling_summary.update(
                {
                    "vh_threshold": float(args.vh_threshold),
                    "vh_sample_tries": int(sample_tries),
                    "vh_accept_rate": float(len(init_states) / sample_tries),
                    "vh_init_mean": float(init_vh.mean()),
                    "vh_init_min": float(init_vh.min()),
                    "vh_init_max": float(init_vh.max()),
                }
            )

    # V_h contour.
    ref = get_ref_from_env(env, args.seed)
    z_dot_list = parse_float_list(args.z_dot_list)
    x = np.linspace(args.x_min, args.x_max, args.grid_n, dtype=np.float32)
    z = np.linspace(args.z_min, args.z_max, args.grid_n, dtype=np.float32)
    X, Z = np.meshgrid(x, z, indexing="xy")

    vh_list = []
    for z_dot in z_dot_list:
        obs_batch = build_obs_batch(X, Z, z_dot, ref)
        rng = jax.random.PRNGKey(args.seed)
        vh_flat = compute_vh_for_slice(agent, obs_batch, args.K, rng, args.batch_chunk)
        vh_list.append(vh_flat.reshape(args.grid_n, args.grid_n))

    plot_vh_grid(
        X,
        Z,
        vh_list,
        z_dot_list,
        os.path.join(args.out_dir, "vh_region.png"),
        traj_xz=traj_xz,
    )
    np.savez(
        os.path.join(args.out_dir, "vh_region.npz"),
        X=X,
        Z=Z,
        z_dot_list=np.array(z_dot_list, dtype=np.float32),
        Vh_list=np.stack(vh_list, axis=0),
        ref=ref,
    )

    # Save summary for the main evaluation cohort.
    summary = {
        "Q": Q.tolist(),
        "R": R.tolist(),
    }
    if eval_summary_metrics is not None:
        for key, value in eval_summary_metrics.items():
            if isinstance(value, np.ndarray):
                continue
            summary[key] = value
    if sampling_summary is not None:
        summary.update(sampling_summary)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
