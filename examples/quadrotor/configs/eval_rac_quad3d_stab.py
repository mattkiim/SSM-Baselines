#!/usr/bin/env python3
"""Rollout + visualize RAC Quad3D stabilization trajectories.

Examples:
    python examples/quadrotor/configs/eval_rac_quad3d_stab.py rollout \
        --ckpt_path results/QuadrotorStabilization3D-v0/jaxrl5_quad3d_rac \
        --num_episodes 10 --plot_immediate

    python examples/quadrotor/configs/eval_rac_quad3d_stab.py plot \
        --trajs results/evaluations/quad3d_results/rac_quad3d_traj.npz \
        --out rac_quad3d_traj.png

    python examples/quadrotor/configs/eval_rac_quad3d_stab.py rollout \
        --ckpt_path ... --init_pz -0.1 --plot_immediate

    python examples/quadrotor/configs/eval_rac_quad3d_stab.py rollout \
        --ckpt_path ... --init_npz rac_quad3d_traj_inits.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import jax
import jax.numpy as jnp


STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
COLORS = [
    "#0040FF",
    "#E50000",
    "#00BB00",
    "#8A2BE2",
    "#FF8C00",
    "#00CED1",
    "#FF1493",
    "#32CD32",
]
RESULTS_DIR = os.path.join("results", "evaluations", "quad3d_results")

INIT_STATE_KEYS = [
    "init_px",
    "init_py",
    "init_pz",
    "init_vx",
    "init_vy",
    "init_vz",
    "init_phi",
    "init_theta",
    "init_psi",
]


def results_png_path(path: str | None, default_name: str) -> str:
    if path:
        name = os.path.basename(path)
        stem, _ = os.path.splitext(name)
        if not stem:
            stem = default_name
    else:
        stem = default_name
    return os.path.join(RESULTS_DIR, f"{stem}.png")


def create_env(env_id: str, seed: int) -> Any:
    import gymnasium as gym

    from jaxrl5.envs.registration import ensure_custom_envs_registered
    from jaxrl5.wrappers import AddCostFromInfo

    ensure_custom_envs_registered()
    env = gym.make(env_id)
    env = AddCostFromInfo(env)
    env.reset(seed=seed)
    return env


def load_agent(ckpt_path: str, step: int | None, env: Any, seed: int) -> Any:
    from jaxrl5.tools.load_rac import load_rac

    agent, _, _ = load_rac(
        ckpt_path,
        step=step,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
        deterministic=False,
    )
    return agent


def make_eval_policy(agent: Any) -> Callable[[np.ndarray], np.ndarray]:
    eval_agent = agent

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(np.asarray(obs, dtype=np.float32))
        return np.asarray(action, dtype=np.float32)

    return policy


def predict_vh(agent: Any, obs: np.ndarray, K: int, rng: jax.random.PRNGKey) -> float:
    obs = np.asarray(obs, dtype=np.float32)
    obs_tiled = jnp.asarray(np.repeat(obs[None, :], int(K), axis=0))
    dist = agent.actor.apply_fn({"params": agent.actor.params}, obs_tiled)
    actions = dist.sample(seed=rng)
    qh = agent.safety_critic.apply_fn(
        {"params": agent.safety_critic.params},
        obs_tiled,
        actions,
        training=False,
    )
    return float(np.asarray(jnp.min(qh)))


def split_state_ref(obs: np.ndarray, state_dim: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32)
    state = obs[:state_dim]
    if obs.shape[0] == state_dim:
        ref = np.zeros_like(state)
    else:
        ref = obs[state_dim : state_dim * 2]
    return state, ref


def load_init_states(path: str, key: str = "init_states", num_episodes: int | None = None) -> np.ndarray:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"{path} missing key={key!r}. Available keys: {list(data.keys())}")

    states = np.asarray(data[key], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != len(INIT_STATE_KEYS):
        raise ValueError(f"Expected init states shape (N, 9), got {states.shape}")

    if num_episodes is not None:
        states = states[: int(num_episodes)]

    return states


def init_state_to_options(state: np.ndarray) -> Dict[str, float]:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (len(INIT_STATE_KEYS),):
        raise ValueError(f"Expected one Quad3D init state with shape (9,), got {state.shape}")
    return {key: float(state[i]) for i, key in enumerate(INIT_STATE_KEYS)}


def init_options_to_state(init: Dict[str, float]) -> np.ndarray:
    return np.asarray([init[key] for key in INIT_STATE_KEYS], dtype=np.float32)


def reset_with_init(env: Any, seed: int, init_opts: Optional[Dict[str, float]] = None):
    return env.reset(seed=seed, options=init_opts or {})


def env_reference(env: Any, state_dim: int = 9) -> np.ndarray:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "goal"):
        return np.asarray(unwrapped.goal, dtype=np.float32)
    return np.zeros(state_dim, dtype=np.float32)


def env_success_radius(env: Any, default: float = 0.3) -> float:
    return float(getattr(env.unwrapped, "goal_radius", default))


def env_h(env: Any, state: np.ndarray, fallback_safe_z: float = 0.0, fallback_safe_radius: float = 3.0) -> float:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "_h"):
        return float(unwrapped._h(state))
    safe_z = float(getattr(unwrapped, "safe_z", fallback_safe_z))
    safe_radius = float(getattr(unwrapped, "safe_radius", fallback_safe_radius))
    return max(float(state[2] - safe_z), float(np.linalg.norm(state) - safe_radius))


def compute_metrics(
    trajs: List[Dict],
    x_ref: np.ndarray,
    success_radius: float = 0.3,
    tail_steps: int = 50,
    confusion_threshold: float = 0.0,
) -> Dict[str, np.ndarray | float]:
    x_ref = np.asarray(x_ref, dtype=np.float32).reshape(-1)
    if x_ref.shape[0] == 3:
        x_ref_full = np.zeros(9, dtype=np.float32)
        x_ref_full[:3] = x_ref
    else:
        x_ref_full = x_ref.astype(np.float32)

    if not trajs:
        return {
            "violation_rate": float("nan"),
            "violated_trajectory_rate": float("nan"),
            "crashed_trajectory_rate": float("nan"),
            "terminal_state_error_mean": float("nan"),
            "stability_rate": float("nan"),
            "terminal_state_errors": np.asarray([], dtype=np.float32),
            "stability_errors": np.asarray([], dtype=np.float32),
        }

    total_states = 0
    violated_states = 0
    violated_trajs = 0
    crashed_trajs = 0
    terminal_errors = []
    stability_errors = []
    actual_positive = []

    for traj in trajs:
        states = np.asarray(traj["states"], dtype=np.float32)
        h_vals = np.asarray(traj["h_vals"], dtype=np.float32)
        violations = h_vals > 0.0

        total_states += int(h_vals.size)
        violated_states += int(np.sum(violations))
        violated_trajs += int(np.any(violations))
        crashed_trajs += int(bool(traj["terminated"]))

        terminal_errors.append(float(np.sum(np.abs(states[-1] - x_ref_full))))
        tail = states[-min(50, states.shape[0]) :]
        stability_errors.append(float(np.mean(np.sum(np.abs(tail - x_ref_full[None, :]), axis=1))))
        actual_positive.append(bool((not np.any(violations)) and (not bool(traj["terminated"]))))

    pred_vh = np.asarray([float(t.get("pred_vh", np.nan)) for t in trajs], dtype=np.float32)
    has_vh = np.isfinite(pred_vh)
    actual_positive_arr = np.asarray(actual_positive, dtype=bool)
    predicted_positive = has_vh & (pred_vh < float(confusion_threshold))
    predicted_negative = has_vh & ~predicted_positive
    tp_mask = predicted_positive & actual_positive_arr
    fp_mask = predicted_positive & ~actual_positive_arr
    tn_mask = predicted_negative & ~actual_positive_arr
    fn_mask = predicted_negative & actual_positive_arr
    cm_n = int(np.sum(has_vh))
    cm_tp = int(np.sum(tp_mask))
    cm_fp = int(np.sum(fp_mask))
    cm_tn = int(np.sum(tn_mask))
    cm_fn = int(np.sum(fn_mask))
    cm_actual_pos = int(np.sum(has_vh & actual_positive_arr))
    cm_actual_neg = int(np.sum(has_vh & ~actual_positive_arr))

    n_trajs = len(trajs)
    return {
        "violation_rate": float(violated_states / total_states) if total_states else float("nan"),
        "violated_trajectory_rate": float(violated_trajs / n_trajs),
        "crashed_trajectory_rate": float(crashed_trajs / n_trajs),
        "terminal_state_error_mean": float(np.mean(terminal_errors)),
        "stability_rate": float(np.mean(stability_errors)),
        "terminal_state_errors": np.asarray(terminal_errors, dtype=np.float32),
        "stability_errors": np.asarray(stability_errors, dtype=np.float32),
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
        "pred_vh": pred_vh,
        "confusion_threshold": float(confusion_threshold),
        "confusion_actual_positive_mask": actual_positive_arr,
        "confusion_predicted_positive_mask": predicted_positive,
        "confusion_tp_mask": tp_mask,
        "confusion_fp_mask": fp_mask,
        "confusion_tn_mask": tn_mask,
        "confusion_fn_mask": fn_mask,
    }


def print_metrics(metrics: Dict[str, np.ndarray | float], prefix: str = "Metrics") -> None:
    print(
        f"{prefix}: "
        f"ViolationRate={metrics['violation_rate']:.4f} "
        f"ViolatedTrajRate={metrics['violated_trajectory_rate']:.4f} "
        f"CrashedTrajRate={metrics['crashed_trajectory_rate']:.4f} "
        f"TerminalStateErrMean={metrics['terminal_state_error_mean']:.4f} "
        f"StabilityRate={metrics['stability_rate']:.4f}"
    )
    if int(metrics.get("confusion_n", 0)) > 0:
        print(
            f"{prefix} confusion(Vh<0 vs rollout safe): "
            f"TP={int(metrics['confusion_tp'])} "
            f"FP={int(metrics['confusion_fp'])} "
            f"TN={int(metrics['confusion_tn'])} "
            f"FN={int(metrics['confusion_fn'])} "
            f"Acc={metrics['confusion_accuracy']:.4f} "
            f"Prec={metrics['confusion_precision']:.4f} "
            f"Recall={metrics['confusion_recall']:.4f}"
        )


def rollout_agent(
    ckpt_path: str,
    *,
    step: int | None = None,
    env_id: str = "QuadrotorStabilization3D-v0",
    num_episodes: int = 10,
    seed: int = 42,
    init_opts: Optional[Dict[str, float]] = None,
    init_states_all: Optional[np.ndarray] = None,
    state_dim: int = 9,
    confusion_matrix: bool = False,
    K: int = 64,
    confusion_threshold: float = 0.0,
) -> Tuple[List[Dict], np.ndarray, float]:
    env = create_env(env_id, seed)
    agent = load_agent(ckpt_path, step, env, seed)
    policy_fn = make_eval_policy(agent)
    x_ref = env_reference(env, state_dim)
    success_radius = env_success_radius(env)

    if init_states_all is not None and len(init_states_all) < num_episodes:
        raise ValueError(f"Need at least {num_episodes} init states, got {len(init_states_all)}.")

    trajectories = []
    for ep in range(num_episodes):
        ep_init_opts = init_state_to_options(init_states_all[ep]) if init_states_all is not None else init_opts
        obs, info = reset_with_init(env, seed + ep, ep_init_opts)
        pred_vh = (
            predict_vh(agent, obs, K=K, rng=jax.random.PRNGKey(seed + ep))
            if confusion_matrix
            else float("nan")
        )

        state, ref = split_state_ref(obs, state_dim)
        states = [state.copy()]
        refs = [ref.copy()]
        actions = []
        rewards = []
        costs = []
        h_vals = [float(info.get("h", env_h(env, state)))]
        done = False
        terminated = False
        truncated = False
        goal_hit = bool(info.get("goal", False))

        while not done:
            action = policy_fn(obs)
            obs, reward, cost, terminated, truncated, info = env.step(action)
            state, ref = split_state_ref(obs, state_dim)

            actions.append(action)
            rewards.append(float(reward))
            costs.append(float(cost))
            states.append(state.copy())
            refs.append(ref.copy())
            h_vals.append(float(info.get("h", env_h(env, state))))

            goal_hit = goal_hit or bool(info.get("goal", False))
            done = bool(terminated or truncated)

        traj = {
            "states": np.asarray(states, dtype=np.float32),
            "refs": np.asarray(refs, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "costs": np.asarray(costs, dtype=np.float32),
            "h_vals": np.asarray(h_vals, dtype=np.float32),
            "length": int(len(actions)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "goal_hit": bool(goal_hit),
            "pred_vh": float(pred_vh),
        }
        trajectories.append(traj)

        ep_metrics = compute_metrics(
            [traj],
            x_ref,
            success_radius=success_radius,
            confusion_threshold=confusion_threshold,
        )
        print(
            f"  Ep {ep}: "
            f"Viol={ep_metrics['violation_rate']:.3f} "
            f"TermErr={ep_metrics['terminal_state_error_mean']:.3f} "
            f"Stable={ep_metrics['stability_rate']:.3f} "
            f"Crash={bool(terminated)} "
            f"L={traj['length']}"
        )

    return trajectories, x_ref, success_radius


def save_trajectories(
    trajs: List[Dict],
    label: str,
    path: str,
    x_ref: np.ndarray | None = None,
    success_radius: float = 0.3,
    confusion_threshold: float = 0.0,
) -> None:
    if x_ref is None:
        x_ref = np.zeros(9, dtype=np.float32)

    metrics = compute_metrics(
        trajs,
        x_ref,
        success_radius=success_radius,
        confusion_threshold=confusion_threshold,
    )
    init_states = np.asarray([t["states"][0] for t in trajs], dtype=np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    np.savez(
        path,
        label=label,
        x_ref=np.asarray(x_ref, dtype=np.float32),
        success_radius=np.asarray(success_radius, dtype=np.float32),
        n_episodes=len(trajs),
        init_states=init_states,
        **{f"states_{i}": t["states"] for i, t in enumerate(trajs)},
        **{f"refs_{i}": t["refs"] for i, t in enumerate(trajs)},
        **{f"actions_{i}": t["actions"] for i, t in enumerate(trajs)},
        **{f"rewards_{i}": t["rewards"] for i, t in enumerate(trajs)},
        **{f"costs_{i}": t["costs"] for i, t in enumerate(trajs)},
        **{f"h_vals_{i}": t["h_vals"] for i, t in enumerate(trajs)},
        lengths=np.asarray([t["length"] for t in trajs], dtype=np.int32),
        terminated=np.asarray([t["terminated"] for t in trajs], dtype=bool),
        truncated=np.asarray([t["truncated"] for t in trajs], dtype=bool),
        goal_hits=np.asarray([t["goal_hit"] for t in trajs], dtype=bool),
        returns=np.asarray([np.sum(t["rewards"]) for t in trajs], dtype=np.float32),
        cost_sums=np.asarray([np.sum(t["costs"]) for t in trajs], dtype=np.float32),
        pred_vh=np.asarray(metrics["pred_vh"], dtype=np.float32),
        confusion_threshold=np.asarray(metrics["confusion_threshold"], dtype=np.float32),
        confusion_actual_positive_mask=np.asarray(metrics["confusion_actual_positive_mask"], dtype=bool),
        confusion_predicted_positive_mask=np.asarray(metrics["confusion_predicted_positive_mask"], dtype=bool),
        confusion_tp_mask=np.asarray(metrics["confusion_tp_mask"], dtype=bool),
        confusion_fp_mask=np.asarray(metrics["confusion_fp_mask"], dtype=bool),
        confusion_tn_mask=np.asarray(metrics["confusion_tn_mask"], dtype=bool),
        confusion_fn_mask=np.asarray(metrics["confusion_fn_mask"], dtype=bool),
        violation_rate=np.asarray(metrics["violation_rate"], dtype=np.float32),
        violated_trajectory_rate=np.asarray(metrics["violated_trajectory_rate"], dtype=np.float32),
        crashed_trajectory_rate=np.asarray(metrics["crashed_trajectory_rate"], dtype=np.float32),
        terminal_state_error_mean=np.asarray(metrics["terminal_state_error_mean"], dtype=np.float32),
        stability_rate=np.asarray(metrics["stability_rate"], dtype=np.float32),
        terminal_state_errors=metrics["terminal_state_errors"],
        stability_errors=metrics["stability_errors"],
    )
    print(f"Saved: {path}")
    print_metrics(metrics)

    init_path = path.replace(".npz", "_inits.npz")
    np.savez(init_path, init_states=init_states)
    print(f"Saved init states: {init_path}")


def save_summary_json(
    path: str,
    trajs: List[Dict],
    x_ref: np.ndarray,
    success_radius: float = 0.3,
    confusion_threshold: float = 0.0,
) -> None:
    metrics = compute_metrics(
        trajs,
        x_ref,
        success_radius=success_radius,
        confusion_threshold=confusion_threshold,
    )
    summary = {
        "n_eval": len(trajs),
        "violation_rate": float(metrics["violation_rate"]),
        "violated_trajectory_rate": float(metrics["violated_trajectory_rate"]),
        "crashed_trajectory_rate": float(metrics["crashed_trajectory_rate"]),
        "terminal_state_error_mean": float(metrics["terminal_state_error_mean"]),
        "stability_rate": float(metrics["stability_rate"]),
        "return_mean": float(np.mean([np.sum(t["rewards"]) for t in trajs])) if trajs else float("nan"),
        "cost_sum_mean": float(np.mean([np.sum(t["costs"]) for t in trajs])) if trajs else float("nan"),
        "episode_length_mean": float(np.mean([t["length"] for t in trajs])) if trajs else float("nan"),
        "confusion_n": int(metrics["confusion_n"]),
        "confusion_threshold": float(metrics["confusion_threshold"]),
        "confusion_tp": int(metrics["confusion_tp"]),
        "confusion_fp": int(metrics["confusion_fp"]),
        "confusion_tn": int(metrics["confusion_tn"]),
        "confusion_fn": int(metrics["confusion_fn"]),
        "confusion_accuracy": float(metrics["confusion_accuracy"]),
        "confusion_precision": float(metrics["confusion_precision"]),
        "confusion_recall": float(metrics["confusion_recall"]),
        "confusion_fpr": float(metrics["confusion_fpr"]),
        "confusion_fnr": float(metrics["confusion_fnr"]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_trajectories(path: str) -> Tuple[str, List[Dict], np.ndarray, float]:
    data = np.load(path, allow_pickle=True)
    label = str(data["label"]) if "label" in data.files else "RAC"
    n = int(data["n_episodes"]) if "n_episodes" in data.files else len([k for k in data.files if k.startswith("states_")])
    x_ref = np.asarray(data["x_ref"], dtype=np.float32) if "x_ref" in data.files else np.zeros(9, dtype=np.float32)
    success_radius = float(data["success_radius"]) if "success_radius" in data.files else 0.3

    trajs = []
    for i in range(n):
        states = np.asarray(data[f"states_{i}"], dtype=np.float32)
        h_vals = (
            np.asarray(data[f"h_vals_{i}"], dtype=np.float32)
            if f"h_vals_{i}" in data.files
            else np.maximum(states[:, 2], np.linalg.norm(states, axis=1) - 3.0).astype(np.float32)
        )
        trajs.append(
            {
                "states": states,
                "refs": np.asarray(data[f"refs_{i}"], dtype=np.float32) if f"refs_{i}" in data.files else np.zeros_like(states),
                "actions": np.asarray(data[f"actions_{i}"], dtype=np.float32) if f"actions_{i}" in data.files else np.empty((0, 4), dtype=np.float32),
                "rewards": np.asarray(data[f"rewards_{i}"], dtype=np.float32) if f"rewards_{i}" in data.files else np.asarray([], dtype=np.float32),
                "costs": np.asarray(data[f"costs_{i}"], dtype=np.float32) if f"costs_{i}" in data.files else np.asarray([], dtype=np.float32),
                "h_vals": h_vals,
                "length": int(data["lengths"][i]) if "lengths" in data.files else int(states.shape[0] - 1),
                "terminated": bool(data["terminated"][i]) if "terminated" in data.files else False,
                "truncated": bool(data["truncated"][i]) if "truncated" in data.files else False,
                "goal_hit": bool(data["goal_hits"][i]) if "goal_hits" in data.files else False,
            }
        )
    return label, trajs, x_ref, success_radius


def plot_trajectories(
    trajs: List[Dict],
    out_path: str,
    label: str = "RAC",
    best_n: int = 5,
    title: str | None = None,
    x_ref: np.ndarray | None = None,
    success_radius: float = 0.3,
    dt: float = 0.01,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.lines import Line2D

    rcParams.update(STYLE)

    if x_ref is None:
        x_ref = np.zeros(9, dtype=np.float32)
    x_ref = np.asarray(x_ref, dtype=np.float32)
    if x_ref.shape[0] == 3:
        x_ref_full = np.zeros(9, dtype=np.float32)
        x_ref_full[:3] = x_ref
    else:
        x_ref_full = x_ref.astype(np.float32)

    if best_n <= 0:
        raise ValueError(f"best_n must be positive, got {best_n}.")

    sorted_trajs = sorted(
        trajs,
        key=lambda t: (
            bool(t["terminated"]),
            bool(np.any(np.asarray(t["h_vals"]) > 0.0)),
            float(np.sum(np.abs(t["states"][-1] - x_ref_full))),
        ),
    )
    show_trajs = sorted_trajs[:best_n]
    if not show_trajs:
        raise ValueError("No trajectories to plot.")

    print(
        "Plot selection: "
        f"{len(show_trajs)}/{len(trajs)} trajectories shown. "
        "Sorted by crash, violation, then terminal L1 state error."
    )
    print_metrics(compute_metrics(show_trajs, x_ref_full))

    ref_px, ref_py, ref_pz = float(x_ref_full[0]), float(x_ref_full[1]), float(x_ref_full[2])
    alphas = np.linspace(1.0, 0.4, max(best_n, 1))
    max_t = max(t["states"].shape[0] for t in show_trajs) * dt

    fig = plt.figure(figsize=(14, 11))

    ax3d = fig.add_subplot(221, projection="3d")
    gx = np.linspace(-2, 2, 20)
    gy = np.linspace(-2, 2, 20)
    gx_grid, gy_grid = np.meshgrid(gx, gy)
    gz_grid = np.zeros_like(gx_grid)
    ax3d.plot_surface(gx_grid, gy_grid, gz_grid, alpha=0.15, color="red")

    ax3d.plot([-2.0, 2.0], [0.0, 0.0], [0.0, 0.0], color="black", linewidth=1.2, alpha=0.45)
    ax3d.plot([0.0, 0.0], [-2.0, 2.0], [0.0, 0.0], color="black", linewidth=1.2, alpha=0.45)
    ax3d.plot([0.0, 0.0], [0.0, 0.0], [-1.5, 0.5], color="black", linewidth=1.2, alpha=0.45)

    ref_is_origin = np.allclose([ref_px, ref_py, ref_pz], [0.0, 0.0, 0.0], atol=1e-6)
    origin_label = "Origin / x_ref" if ref_is_origin else "Origin (0,0,0)"
    ax3d.scatter([0.0], [0.0], [0.0], color="black", s=95, marker="+", linewidths=2.0, zorder=7, label=origin_label)
    if not ref_is_origin:
        ax3d.scatter([ref_px], [ref_py], [ref_pz], color="green", s=80, marker="*", edgecolors="darkgreen", zorder=6)

    for i, traj in enumerate(show_trajs):
        states = traj["states"]
        color = COLORS[i % len(COLORS)]
        alpha = alphas[min(i, len(alphas) - 1)]
        ax3d.plot(
            states[:, 0],
            states[:, 1],
            states[:, 2],
            color=color,
            linewidth=1.8,
            alpha=alpha,
            label=f"Ep{i} E={np.sum(np.abs(states[-1] - x_ref_full)):.2f} V={int(np.any(np.asarray(traj['h_vals']) > 0.0))}",
        )
        ax3d.scatter([states[0, 0]], [states[0, 1]], [states[0, 2]], color=color, s=40, marker="o", edgecolors="white", zorder=5)
        if traj["terminated"]:
            ax3d.scatter([states[-1, 0]], [states[-1, 1]], [states[-1, 2]], color="red", s=60, marker="X", zorder=5)

    ax3d.set_xlabel("$p_x$")
    ax3d.set_ylabel("$p_y$")
    ax3d.set_zlabel("$p_z$")
    ax3d.set_xlim(-2, 2)
    ax3d.set_ylim(-2, 2)
    ax3d.set_zlim(-1.5, 0.5)
    ax3d.invert_zaxis()
    ax3d.set_title("3D Trajectory", fontsize=13)

    ax_norm = fig.add_subplot(222)
    ax_norm.axhline(success_radius, color="green", linewidth=1.5, linestyle="--", label=f"success_radius={success_radius:g}")
    for i, traj in enumerate(show_trajs):
        states = traj["states"]
        time_axis = np.arange(states.shape[0]) * dt
        norms = np.linalg.norm(states - x_ref_full, axis=1)
        color = COLORS[i % len(COLORS)]
        alpha = alphas[min(i, len(alphas) - 1)]
        ax_norm.plot(time_axis, norms, color=color, linewidth=1.8, alpha=alpha)
        if traj["terminated"]:
            ax_norm.scatter([time_axis[-1]], [norms[-1]], color="red", s=60, marker="X", zorder=5)
    ax_norm.set_xlabel("Time (s)")
    ax_norm.set_ylabel(r"$\| x_t - x_{ref} \|$")
    ax_norm.set_xlim(0, max_t)
    ax_norm.set_ylim(bottom=0)
    ax_norm.set_title("Convergence to Reference", fontsize=13)
    ax_norm.grid(alpha=0.3)

    ax_xy = fig.add_subplot(223)
    ax_xy.axhline(ref_px, color="gray", linewidth=1.0, linestyle=":", alpha=0.6)
    for i, traj in enumerate(show_trajs):
        states = traj["states"]
        time_axis = np.arange(states.shape[0]) * dt
        color = COLORS[i % len(COLORS)]
        alpha = alphas[min(i, len(alphas) - 1)]
        ax_xy.plot(time_axis, states[:, 0], color=color, linewidth=1.6, alpha=alpha, linestyle="-")
        ax_xy.plot(time_axis, states[:, 1], color=color, linewidth=1.6, alpha=alpha, linestyle="--")
        if traj["terminated"]:
            ax_xy.scatter([time_axis[-1]], [states[-1, 0]], color="red", s=60, marker="X", zorder=5)
            ax_xy.scatter([time_axis[-1]], [states[-1, 1]], color="red", s=60, marker="X", zorder=5)
    ax_xy.legend(
        handles=[
            Line2D([0], [0], color="gray", linestyle=":", label=f"ref ({ref_px:.2f},{ref_py:.2f})"),
            Line2D([0], [0], color="black", linestyle="-", label="$p_x$ (solid)"),
            Line2D([0], [0], color="black", linestyle="--", label="$p_y$ (dashed)"),
        ],
        fontsize=9,
        loc="upper right",
    )
    ax_xy.set_xlabel("Time (s)")
    ax_xy.set_ylabel("$p_x, p_y$")
    ax_xy.set_xlim(0, max_t)
    ax_xy.set_title("Lateral Position (xy)", fontsize=13)
    ax_xy.grid(alpha=0.3)

    ax_pz = fig.add_subplot(224)
    t_fill = np.array([0, max_t])
    ax_pz.fill_between(t_fill, 0, 0.5, color="#FFE0E0", alpha=0.4, label="Unsafe (pz>0)")
    ax_pz.axhline(0.0, color="black", linewidth=2.0, label="Ground (pz=0)")
    ax_pz.axhline(0.3, color="red", linewidth=1.5, linestyle=":", label="Termination (pz=0.3)")
    if abs(ref_pz) > 1e-6:
        ax_pz.axhline(ref_pz, color="green", linewidth=1.5, linestyle="--", label=f"pz_ref={ref_pz:.2f}")
    for i, traj in enumerate(show_trajs):
        states = traj["states"]
        time_axis = np.arange(states.shape[0]) * dt
        color = COLORS[i % len(COLORS)]
        alpha = alphas[min(i, len(alphas) - 1)]
        ax_pz.plot(time_axis, states[:, 2], color=color, linewidth=1.8, alpha=alpha)
        if traj["terminated"]:
            ax_pz.scatter([time_axis[-1]], [states[-1, 2]], color="red", s=60, marker="X", zorder=5)
    ax_pz.set_xlabel("Time (s)")
    ax_pz.set_ylabel("$p_z$")
    ax_pz.set_ylim(-1.5, 0.5)
    ax_pz.invert_yaxis()
    ax_pz.legend(fontsize=9, loc="upper right")
    ax_pz.set_title("Altitude Profile", fontsize=14)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    else:
        metrics = compute_metrics(trajs, x_ref_full)
        ref_str = f"x_ref=({ref_px:.1f},{ref_py:.1f},{ref_pz:.1f})"
        fig.suptitle(
            f"{label}  "
            f"Viol={metrics['violation_rate']:.3f}  "
            f"ViolTraj={metrics['violated_trajectory_rate']:.0%}  "
            f"Crash={metrics['crashed_trajectory_rate']:.0%}  "
            f"TermErr={metrics['terminal_state_error_mean']:.2f}  "
            f"Stable={metrics['stability_rate']:.2f}  "
            f"{ref_str}  (n={len(trajs)})",
            fontsize=11,
            y=1.02,
        )

    fig.tight_layout()
    out_path = results_png_path(out_path, "rac_quad3d_traj")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"Saved: {out_path}")


def add_init_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--init_px", type=float, default=None)
    parser.add_argument("--init_py", type=float, default=None)
    parser.add_argument("--init_pz", type=float, default=None)
    parser.add_argument("--init_vx", type=float, default=None)
    parser.add_argument("--init_vy", type=float, default=None)
    parser.add_argument("--init_vz", type=float, default=None)
    parser.add_argument("--init_phi", type=float, default=None)
    parser.add_argument("--init_theta", type=float, default=None)
    parser.add_argument("--init_psi", type=float, default=None)


def parse_init_overrides(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    init_opts = {}
    for key in INIT_STATE_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            init_opts[key] = float(val)
    return init_opts or None


def main() -> None:
    parser = argparse.ArgumentParser(description="RAC Quad3D stabilization rollout + plot")
    subparsers = parser.add_subparsers(dest="command")

    p_roll = subparsers.add_parser("rollout")
    p_roll.add_argument("--ckpt_path", "--ckpt", dest="ckpt_path", required=True)
    p_roll.add_argument("--step", type=int, default=None)
    p_roll.add_argument("--env_id", type=str, default="QuadrotorStabilization3D-v0")
    p_roll.add_argument("--label", default="RAC")
    p_roll.add_argument("--num_episodes", "--n_eval", dest="num_episodes", type=int, default=10)
    p_roll.add_argument("--seed", type=int, default=42)
    p_roll.add_argument("--out", default=os.path.join(RESULTS_DIR, "rac_quad3d_traj.npz"))
    p_roll.add_argument("--plot_immediate", action="store_true")
    p_roll.add_argument("--best_n", type=int, default=5)
    p_roll.add_argument("--state_dim", type=int, default=9)
    p_roll.add_argument("--init_npz", type=str, default=None)
    p_roll.add_argument("--init_key", type=str, default="init_states")
    p_roll.add_argument("--confusion_matrix", action="store_true")
    p_roll.add_argument("--K", type=int, default=64, help="Number of sampled actions for Vh(init)=min_a Qh(init,a).")
    p_roll.add_argument(
        "--confusion_threshold",
        type=float,
        default=0.0,
        help="Predicted safe iff Vh(init) is below this threshold.",
    )
    add_init_args(p_roll)

    p_plot = subparsers.add_parser("plot")
    p_plot.add_argument("--trajs", nargs="+", required=True)
    p_plot.add_argument("--out", default="rac_quad3d_traj.png")
    p_plot.add_argument("--best_n", type=int, default=5)
    p_plot.add_argument("--title", default=None)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    if args.command == "rollout":
        init_opts = parse_init_overrides(args)
        init_states_all = None
        if args.init_npz is not None:
            init_states_all = load_init_states(args.init_npz, args.init_key, args.num_episodes)
            print(f"Loaded {len(init_states_all)} init states from {args.init_npz}[{args.init_key}]")
            if init_opts is not None:
                print("  init_npz provided; ignoring --init_* overrides.")
                init_opts = None

        print(f"Rolling out {args.num_episodes} episodes from {args.ckpt_path} ...")
        trajs, x_ref, success_radius = rollout_agent(
            args.ckpt_path,
            step=args.step,
            env_id=args.env_id,
            num_episodes=args.num_episodes,
            seed=args.seed,
            init_opts=init_opts,
            init_states_all=init_states_all,
            state_dim=args.state_dim,
            confusion_matrix=args.confusion_matrix,
            K=args.K,
            confusion_threshold=args.confusion_threshold,
        )

        npz_path = args.out if args.out.endswith(".npz") else args.out.rsplit(".", 1)[0] + ".npz"
        save_trajectories(
            trajs,
            args.label,
            npz_path,
            x_ref=x_ref,
            success_radius=success_radius,
            confusion_threshold=args.confusion_threshold,
        )
        save_summary_json(
            str(Path(npz_path).with_name("summary.json")),
            trajs,
            x_ref,
            success_radius=success_radius,
            confusion_threshold=args.confusion_threshold,
        )

        if args.plot_immediate or not args.out.endswith(".npz"):
            png_path = args.out if not args.out.endswith(".npz") else args.out.replace(".npz", ".png")
            plot_trajectories(
                trajs,
                png_path,
                label=args.label,
                best_n=args.best_n,
                x_ref=x_ref,
                success_radius=success_radius,
            )

    elif args.command == "plot":
        all_trajs = []
        label = "RAC"
        x_ref = None
        success_radius = None
        for path in args.trajs:
            loaded_label, trajs, loaded_x_ref, loaded_success_radius = load_trajectories(path)
            all_trajs.extend(trajs)
            label = loaded_label
            if x_ref is None:
                x_ref = loaded_x_ref
            elif not np.allclose(x_ref, loaded_x_ref):
                print(
                    f"  WARNING: {path} has x_ref={loaded_x_ref[:3]} which differs "
                    f"from first file's x_ref={x_ref[:3]}; using first."
                )
            if success_radius is None:
                success_radius = loaded_success_radius
            elif not np.isclose(success_radius, loaded_success_radius):
                print(
                    f"  WARNING: {path} has success_radius={loaded_success_radius} which differs "
                    f"from first file's success_radius={success_radius}; using first."
                )
            print(f"  Loaded {len(trajs)} episodes from {path}")

        plot_trajectories(
            all_trajs,
            args.out,
            label=label,
            best_n=args.best_n,
            title=args.title,
            x_ref=x_ref,
            success_radius=success_radius if success_radius is not None else 0.3,
        )


if __name__ == "__main__":
    main()
