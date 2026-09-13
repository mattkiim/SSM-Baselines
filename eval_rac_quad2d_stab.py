#!/usr/bin/env python3
"""Evaluate a trained RAC policy on Quad2D stabilization initial states.

Examples:
    python eval_rac_quad2d_stab.py \
        --run_dir results/QuadrotorStabilization2D-v0/jaxrl5_quad2d_stab_rac/2026-05-06_warm_anchor_fork20_mirror_qx2 \
        --fixed_starts \
        --out_dir results/evaluations/eval_rac_quad2d_stab

    python eval_rac_quad2d_stab.py \
        --run_dir results/QuadrotorStabilization2D-v0/jaxrl5_quad2d_stab_rac/2026-05-06_warm_anchor_fork20_mirror_qx2 \
        --suite lowz_hrej \
        --num_episodes 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")

import numpy as np


FIXED_STARTS = np.array(
    [
        [0.00, 0.00, -1.08, 0.00, 0.00, 0.00],
        [-0.30, 0.15, -0.95, 0.20, 0.02, 0.00],
        [0.30, -0.15, -0.95, 0.20, -0.02, 0.00],
        [0.00, 0.00, -0.55, 0.10, 0.00, 0.00],
    ],
    dtype=np.float32,
)


DEFAULT_ENV_KWARGS: Dict[str, Any] = {
    "layout_name": "corridor_v2",
    "reset_mode": "simple_under3",
    "obs_feature_mode": "state",
    "q_x": 2.0,
    "q_z": 10.0,
    "reset_simple_bar_frac": 0.24,
    "reset_simple_left_block_frac": 0.08,
    "reset_simple_right_block_frac": 0.08,
    "reset_simple_gap_frac": 0.08,
    "reset_simple_low_uniform_frac": 0.20,
    "reset_simple_near_frac": 0.12,
    "reset_simple_side_frac": 0.0,
    "reset_simple_fork_frac": 0.20,
}


def resolve_checkpoint(run_dir: str, ckpt_path: Optional[str], load_step: Optional[int]) -> str:
    if ckpt_path is not None:
        return ckpt_path

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if load_step is not None:
        path = os.path.join(ckpt_dir, f"ckpt_{load_step}.msgpack")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find checkpoint: {path}")
        return path

    candidates = glob.glob(os.path.join(ckpt_dir, "ckpt_*.msgpack"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any ckpt_*.msgpack under {ckpt_dir}. "
            "Pass --ckpt_path explicitly if your checkpoint lives elsewhere."
        )
    return sorted(candidates, key=_checkpoint_sort_key)[-1]


def _checkpoint_sort_key(path: str) -> tuple[int, str]:
    match = re.search(r"ckpt_(\d+)", os.path.basename(path))
    step = int(match.group(1)) if match else -1
    return step, path


def load_env_kwargs(run_dir: str) -> Dict[str, Any]:
    env_kwargs = dict(DEFAULT_ENV_KWARGS)
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: no config.json found at {config_path}; using warm-anchor defaults.")
        return env_kwargs

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    saved_env_kwargs = raw.get("env_kwargs")
    if isinstance(saved_env_kwargs, dict):
        env_kwargs.update(saved_env_kwargs)
    return env_kwargs


def apply_env_overrides(env_kwargs: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(env_kwargs)
    for name in ("layout_name", "reset_mode", "obs_feature_mode"):
        value = getattr(args, name)
        if value is not None:
            out[name] = value
    if args.q_x is not None:
        out["q_x"] = args.q_x
    if args.q_z is not None:
        out["q_z"] = args.q_z
    return out


def make_eval_env(args: argparse.Namespace, env_kwargs: Dict[str, Any]):
    from jaxrl5.envs import make_env

    return make_env(args.env_name, seed=args.seed, **env_kwargs)


def load_init_states(path: str, key: str, num_episodes: Optional[int]) -> np.ndarray:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"{path} missing key={key!r}. Available: {list(data.keys())}")
    states = np.asarray(data[key], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(f"Expected init states with shape (N, 6), got {states.shape}")
    return states[:num_episodes] if num_episodes is not None else states


def select_initial_states(env, args: argparse.Namespace) -> np.ndarray:
    if args.fixed_starts:
        states = FIXED_STARTS
    elif args.init_npz is not None:
        states = load_init_states(args.init_npz, args.init_key, args.num_episodes)
    elif args.suite == "lowz_hrej":
        from examples.quadrotor.quad2d_stab_eval import sample_lowz_hrej_initial_states_stab

        states = sample_lowz_hrej_initial_states_stab(
            env,
            n=args.num_episodes,
            seed=args.seed,
            z_high=args.lowz_high,
        )
    elif args.suite == "fixed_vel_small":
        from examples.quadrotor.quad2d_stab_eval import make_fixed_vel_small_states

        states = make_fixed_vel_small_states(n=args.num_episodes, seed=args.seed)
    else:
        raise ValueError("Provide --fixed_starts, --init_npz, or --suite.")

    if args.num_episodes is not None:
        states = states[: args.num_episodes]
    return np.asarray(states, dtype=np.float32)


def reset_to_state(env, init_state: np.ndarray):
    return env.reset(
        options={
            "init_x": float(init_state[0]),
            "init_vx": float(init_state[1]),
            "init_z": float(init_state[2]),
            "init_vz": float(init_state[3]),
            "init_theta": float(init_state[4]),
            "init_omega": float(init_state[5]),
        }
    )


def step_env(env, action: np.ndarray):
    out = env.step(action)
    if len(out) == 6:
        obs, reward, cost, terminated, truncated, info = out
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        cost = info.get("cost", 0.0)
    else:
        raise RuntimeError(f"Unexpected env.step return length: {len(out)}")
    return obs, float(reward), float(cost), bool(terminated), bool(truncated), info


def make_rac_policy_fn(agent, deterministic: bool):
    state_holder = {"agent": agent}

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)
        current_agent = state_holder["agent"]
        if deterministic:
            out = current_agent.eval_actions(obs_np)
        else:
            out = current_agent.sample_actions(obs_np)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            action, new_agent = out
            state_holder["agent"] = new_agent
        else:
            action = out

        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 1:
            raise ValueError(f"Expected RAC action shape (act_dim,), got {action.shape}")
        return action

    return policy_fn


def rollout_from_init_state(env, policy_fn, init_state: np.ndarray, max_steps: Optional[int]):
    obs, _ = reset_to_state(env, init_state)

    states, actions = [], []
    rewards, costs, hs = [], [], []
    weighted_state_errors, unweighted_state_errors = [], []
    weighted_action_errors, unweighted_action_errors = [], []
    xs = [float(env.unwrapped.state[0])]
    zs = [float(env.unwrapped.state[2])]
    first_violation_step = -1
    last_info: Dict[str, Any] = {}

    q_diag = np.asarray(env.unwrapped.Q_diag, dtype=np.float32)
    r_diag = np.asarray(env.unwrapped.R_diag, dtype=np.float32)
    x_ref = np.asarray(env.unwrapped.x_ref, dtype=np.float32)
    a_ref = np.asarray(env.unwrapped.a_ref, dtype=np.float32)

    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        state = np.asarray(env.unwrapped.state, dtype=np.float32)
        action = np.asarray(policy_fn(obs), dtype=np.float32)
        if action.shape != env.action_space.shape:
            raise ValueError(
                f"Policy returned action shape {action.shape}, expected {env.action_space.shape}."
            )
        action_raw = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0

        state_err = state - x_ref
        action_err = action_raw - a_ref
        h = float(env.unwrapped.h_phys(state))
        if h > 0.0 and first_violation_step < 0:
            first_violation_step = steps

        weighted_state_errors.append(float(np.sum((state_err**2) * q_diag)))
        unweighted_state_errors.append(float(np.sum(state_err**2)))
        weighted_action_errors.append(float(np.sum((action_err**2) * r_diag)))
        unweighted_action_errors.append(float(np.sum(action_err**2)))

        states.append(state)
        actions.append(action)
        hs.append(h)

        obs, reward, cost, terminated, truncated, info = step_env(env, action)
        rewards.append(reward)
        costs.append(cost)
        last_info = dict(info)
        xs.append(float(env.unwrapped.state[0]))
        zs.append(float(env.unwrapped.state[2]))
        steps += 1

        if first_violation_step < 0 and float(info.get("h_phys", info.get("h", -np.inf))) > 0.0:
            first_violation_step = steps
        if max_steps is not None and steps >= max_steps:
            break

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "h": np.asarray(hs, dtype=np.float32),
        "weighted_state_errors": np.asarray(weighted_state_errors, dtype=np.float32),
        "unweighted_state_errors": np.asarray(unweighted_state_errors, dtype=np.float32),
        "weighted_action_errors": np.asarray(weighted_action_errors, dtype=np.float32),
        "unweighted_action_errors": np.asarray(unweighted_action_errors, dtype=np.float32),
        "x": np.asarray(xs, dtype=np.float32),
        "z": np.asarray(zs, dtype=np.float32),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", terminated)),
        "max_goal_streak": float(last_info.get("episode_max_goal_streak", 0.0)),
        "first_violation_step": np.int64(first_violation_step),
    }


def plot_layout(ax, layout_name: str, plt_module):
    from jaxrl5.envs.quadrotor_stabilization_2d import get_layout

    layout = get_layout(layout_name)
    xmin, xmax = layout.xlim
    zmin, zmax = layout.zlim
    ax.plot([xmin, xmax, xmax, xmin, xmin], [zmin, zmin, zmax, zmax, zmin], "k-", lw=1.5)

    for obstacle in layout.obstacles:
        cx, cz = obstacle.center
        sx, sz = obstacle.size
        rect = plt_module.Rectangle(
            (cx - 0.5 * sx, cz - 0.5 * sz),
            sx,
            sz,
            facecolor="tab:red",
            edgecolor="k",
            alpha=0.25,
            lw=1.2,
        )
        ax.add_patch(rect)

    ax.plot(layout.goal[0], layout.goal[1], "*", color="tab:green", markersize=14, label="Goal")
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(zmin - 0.05, zmax + 0.05)


def plot_rollouts(rollouts: List[Dict[str, np.ndarray]], out_path: str, layout_name: str, best_n: int = 100):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local binary wheels.
        print(f"Warning: matplotlib unavailable; skipping trajectory plot: {exc}")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_layout(ax, layout_name, plt)

    ranked = sorted(
        rollouts,
        key=lambda r: (
            bool(r["collision"]),
            float(r["costs"].sum()),
            -float(r["rewards"].sum()),
        ),
    )
    for j, rollout in enumerate(ranked[:best_n]):
        label = "RAC" if j == 0 else None
        ax.plot(rollout["x"], rollout["z"], lw=1.8, alpha=0.85, label=label)
        ax.plot(rollout["x"][0], rollout["z"][0], "o", ms=4, color="tab:blue")
        if rollout["collision"]:
            ax.plot(rollout["x"][-1], rollout["z"][-1], "x", ms=9, color="tab:red")

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def nanmean_or_nan(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")


def save_outputs(
    args: argparse.Namespace,
    rollouts: List[Dict[str, np.ndarray]],
    init_states: np.ndarray,
    meta: Dict[str, Any],
    env_kwargs: Dict[str, Any],
):
    os.makedirs(args.out_dir, exist_ok=True)

    returns = np.asarray([r["rewards"].sum() for r in rollouts], dtype=np.float32)
    cost_sums = np.asarray([r["costs"].sum() for r in rollouts], dtype=np.float32)
    violation_rates = np.asarray(
        [r["costs"].mean() if len(r["costs"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    lengths = np.asarray([len(r["rewards"]) for r in rollouts], dtype=np.float32)
    collisions = np.asarray([r["collision"] for r in rollouts], dtype=np.float32)
    successes = np.asarray([r["success"] for r in rollouts], dtype=np.float32)
    weighted_state_error_means = np.asarray(
        [mean_or_nan(r["weighted_state_errors"]) for r in rollouts], dtype=np.float32
    )
    unweighted_state_error_means = np.asarray(
        [mean_or_nan(r["unweighted_state_errors"]) for r in rollouts], dtype=np.float32
    )
    weighted_action_error_means = np.asarray(
        [mean_or_nan(r["weighted_action_errors"]) for r in rollouts], dtype=np.float32
    )
    unweighted_action_error_means = np.asarray(
        [mean_or_nan(r["unweighted_action_errors"]) for r in rollouts], dtype=np.float32
    )
    first_violation_steps = np.asarray([r["first_violation_step"] for r in rollouts], dtype=np.int64)

    save_dict: Dict[str, Any] = {
        "init_states": init_states,
        "returns": returns,
        "cost_sums": cost_sums,
        "violation_rates": violation_rates,
        "lengths": lengths,
        "weighted_state_error_means": weighted_state_error_means,
        "unweighted_state_error_means": unweighted_state_error_means,
        "weighted_action_error_means": weighted_action_error_means,
        "unweighted_action_error_means": unweighted_action_error_means,
        "first_violation_steps": first_violation_steps,
        "collision_rate": np.float32(collisions.mean()),
        "success_rate": np.float32(successes.mean()),
        "num_episodes": np.int64(len(rollouts)),
        "label": np.array("RAC"),
    }

    per_rollout = []
    for i, rollout in enumerate(rollouts):
        save_dict[f"x_{i}"] = rollout["x"]
        save_dict[f"z_{i}"] = rollout["z"]
        save_dict[f"h_{i}"] = rollout["h"]
        save_dict[f"total_reward_{i}"] = np.float32(rollout["rewards"].sum())
        save_dict[f"total_cost_{i}"] = np.float32(rollout["costs"].sum())
        save_dict[f"terminated_{i}"] = np.bool_(rollout["terminated"])
        save_dict[f"truncated_{i}"] = np.bool_(rollout["truncated"])
        save_dict[f"collision_{i}"] = np.bool_(rollout["collision"])
        save_dict[f"success_{i}"] = np.bool_(rollout["success"])
        save_dict[f"first_violation_step_{i}"] = np.int64(rollout["first_violation_step"])
        per_rollout.append(
            {
                "episode": i,
                "return": float(rollout["rewards"].sum()),
                "cost_sum": float(rollout["costs"].sum()),
                "violation_rate": float(rollout["costs"].mean()) if len(rollout["costs"]) else 0.0,
                "length": int(len(rollout["rewards"])),
                "terminated": bool(rollout["terminated"]),
                "truncated": bool(rollout["truncated"]),
                "collision": bool(rollout["collision"]),
                "success": bool(rollout["success"]),
                "first_violation_step": int(rollout["first_violation_step"]),
            }
        )

    np.savez(os.path.join(args.out_dir, "rollouts.npz"), **save_dict)
    np.savez(os.path.join(args.out_dir, "rollouts_from_inits.npz"), **save_dict)
    plot_rollouts(
        rollouts,
        os.path.join(args.out_dir, "trajectories.png"),
        layout_name=env_kwargs.get("layout_name", "corridor_v2"),
        best_n=args.best_n,
    )

    summary = {
        "num_episodes": int(len(init_states)),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "cost_sum_mean": float(cost_sums.mean()),
        "cost_sum_std": float(cost_sums.std()),
        "violation_rate_mean": float(violation_rates.mean()),
        "violation_rate_std": float(violation_rates.std()),
        "length_mean": float(lengths.mean()),
        "weighted_state_error_mean": nanmean_or_nan(weighted_state_error_means),
        "unweighted_state_error_mean": nanmean_or_nan(unweighted_state_error_means),
        "weighted_action_error_mean": nanmean_or_nan(weighted_action_error_means),
        "unweighted_action_error_mean": nanmean_or_nan(unweighted_action_error_means),
        "collision_rate": float(collisions.mean()),
        "success_rate": float(successes.mean()),
        "violation_episode_rate": float(np.mean(first_violation_steps >= 0)),
        "run_dir": args.run_dir,
        "ckpt_path": meta["ckpt_path"],
        "load_step": meta.get("step"),
        "init_npz": args.init_npz,
        "init_key": args.init_key,
        "suite": args.suite,
        "deterministic": not args.stochastic,
        "env_name": args.env_name,
        "env_kwargs": env_kwargs,
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "per_rollout.json"), "w", encoding="utf-8") as f:
        json.dump(per_rollout, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--load_step", type=int, default=None)
    parser.add_argument("--env_name", default="QuadrotorStabilization2D-v0")
    parser.add_argument("--init_npz", default=None)
    parser.add_argument("--init_key", default="init_states")
    parser.add_argument("--fixed_starts", action="store_true")
    parser.add_argument("--suite", choices=["lowz_hrej", "fixed_vel_small"], default=None)
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lowz_high", type=float, default=0.50)
    parser.add_argument("--best_n", type=int, default=100)
    parser.add_argument("--out_dir", default="results/evaluations/eval_rac_quad2d_stab")
    parser.add_argument("--layout_name", default=None)
    parser.add_argument("--reset_mode", default=None)
    parser.add_argument("--obs_feature_mode", default=None)
    parser.add_argument("--q_x", type=float, default=None)
    parser.add_argument("--q_z", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from jaxrl5.tools.load_rac import load_rac

    np.random.seed(args.seed)

    args.run_dir = str(Path(args.run_dir))
    ckpt_path = resolve_checkpoint(args.run_dir, args.ckpt_path, args.load_step)
    env_kwargs = apply_env_overrides(load_env_kwargs(args.run_dir), args)
    env = make_eval_env(args, env_kwargs)

    init_states = select_initial_states(env, args)
    agent, _, meta = load_rac(
        ckpt_path,
        step=None,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        deterministic=not args.stochastic,
    )
    policy_fn = make_rac_policy_fn(agent, deterministic=not args.stochastic)

    rollouts = []
    for i, init_state in enumerate(init_states):
        print(f"rollout {i + 1}/{len(init_states)}")
        if hasattr(env.unwrapped, "seed"):
            env.unwrapped.seed(args.seed + i)
        rollouts.append(rollout_from_init_state(env, policy_fn, init_state, args.max_steps))

    save_outputs(args, rollouts, init_states, meta, env_kwargs)
    env.close()


if __name__ == "__main__":
    main()
