#!/usr/bin/env python3
"""Rollout + visualize RAC F16 stabilize-avoid trajectories.

Examples:
    python examples/f16/configs/eval_rac_f16.py rollout \
        --ckpt_path results/F16StabilizeV6-v0/jaxrl5_f16_stabilize_v6_rac/... \
        --num_episodes 100 --rollout_T 512 --plot_immediate

    python examples/f16/configs/eval_rac_f16.py plot \
        --trajs results/evaluations/eval_f16/rollouts.npz --out results/evaluations/eval_f16/rollouts.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import jax.numpy as jnp

F16_DIR = Path(__file__).resolve().parents[1]
if str(F16_DIR) not in sys.path:
    sys.path.insert(0, str(F16_DIR))

IDX_VT, IDX_ALPHA, IDX_BETA = 0, 1, 2
IDX_PHI, IDX_THETA, IDX_PSI = 3, 4, 5
IDX_P, IDX_Q, IDX_R = 6, 7, 8
IDX_PN, IDX_PE, IDX_H = 9, 10, 11
IDX_POW = 12


STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}
COLORS = [
    "#0040FF",
    "#E50000",
    "#00BB00",
    "#8A2BE2",
    "#FF8C00",
    "#00A6A6",
    "#CC0077",
    "#606060",
]
RESULTS_DIR = os.path.join("results/evaluations/eval_f16")
STATE_DIM = 16
EVAL_PERTURBATIONS = [
    (0.0, 500.0),
    (0.4, 500.0),
    (-0.4, 500.0),
    (0.0, 200.0),
    (0.0, 900.0),
    (0.6, 400.0),
    (-0.6, 600.0),
    (0.0, 100.0),
]
INIT_STATE_KEYS = [
    "init_vt",
    "init_alpha",
    "init_beta",
    "init_phi",
    "init_theta",
    "init_psi",
    "init_p",
    "init_q",
    "init_r",
    "init_pn",
    "init_pe",
    "init_h",
    "init_pow",
]
INIT_KEY_TO_IDX = {
    "init_vt": IDX_VT,
    "init_alpha": IDX_ALPHA,
    "init_beta": IDX_BETA,
    "init_phi": IDX_PHI,
    "init_theta": IDX_THETA,
    "init_psi": IDX_PSI,
    "init_p": IDX_P,
    "init_q": IDX_Q,
    "init_r": IDX_R,
    "init_pn": IDX_PN,
    "init_pe": IDX_PE,
    "init_h": IDX_H,
    "init_pow": IDX_POW,
}
TASK_H_LABELS = ("alt", "alpha", "beta", "theta", "pe", "p")


def create_env(seed: int, *, env_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    from f16_gym_adapter import F16StabilizeGymWrapper
    from jaxrl5.wrappers import AddCostFromInfo

    kwargs = dict(env_kwargs or {})
    env = F16StabilizeGymWrapper(seed=seed, **kwargs)
    env = AddCostFromInfo(env)
    env.unwrapped.set_eval_mode()
    env.reset(seed=seed)
    return env


def load_agent(ckpt_path: str, step: int | None, env: Any, seed: int) -> Tuple[Any, Dict[str, Any]]:
    from jaxrl5.tools.load_rac import load_rac

    agent, _, meta = load_rac(
        ckpt_path,
        step=step,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
        deterministic=True,
    )
    return agent, meta


def make_eval_policy(agent: Any) -> Callable[[np.ndarray], np.ndarray]:
    eval_agent = agent

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(np.asarray(obs, dtype=np.float32))
        return np.asarray(action, dtype=np.float32)

    return policy


def predict_safety_value(agent: Any, obs: np.ndarray) -> float:
    """Evaluate RAC safety value V(x) at obs using the deterministic policy action."""
    action, _ = agent.eval_actions(np.asarray(obs, dtype=np.float32))
    obs_b = jnp.asarray(np.asarray(obs, dtype=np.float32)[None, :])
    action_b = jnp.asarray(np.asarray(action, dtype=np.float32)[None, :])
    value = agent.safety_critic.apply_fn(
        {"params": agent.safety_critic.params},
        obs_b,
        action_b,
        False,
    )
    return float(np.asarray(value).reshape(-1)[0])


def state_to_options(state: np.ndarray) -> Dict[str, float]:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] < STATE_DIM:
        raise ValueError(f"Expected an F16 raw state with at least {STATE_DIM} dims, got {state.shape}")
    return {key: float(state[idx]) for key, idx in INIT_KEY_TO_IDX.items()}


def options_to_state(env: Any, init: Dict[str, float]) -> np.ndarray:
    state = np.asarray(env.unwrapped.nominal_state_v5(), dtype=np.float64)
    for key, val in init.items():
        if key in INIT_KEY_TO_IDX:
            state[INIT_KEY_TO_IDX[key]] = float(val)
    return state


def install_rollout_state(env: Any, state: np.ndarray) -> np.ndarray:
    """Install an exact eval state after reset has initialized episode counters."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    env.unwrapped._x = state.copy()

    # Keep episode diagnostics consistent with the externally supplied state.
    env.unwrapped._ep_final_goal_distance = env.unwrapped._task_goal_distance(env.unwrapped._x)
    env.unwrapped._ep_min_altitude = float(env.unwrapped._x[IDX_H])
    env.unwrapped._ep_peak_h = env.unwrapped._task_safety_margin(env.unwrapped._x)
    env.unwrapped._ep_peak_h_components = {
        label: float(value)
        for label, value in zip(TASK_H_LABELS, env.unwrapped._task_h_components(env.unwrapped._x))
    }
    env.unwrapped._ep_max_abs_beta = abs(float(env.unwrapped._x[IDX_BETA]))
    env.unwrapped._ep_max_abs_theta = abs(float(env.unwrapped._x[IDX_THETA]))

    h_raw = env.unwrapped._task_h_components_raw(env.unwrapped._x)
    env.unwrapped._ep_init_goal = float(env.unwrapped._task_goal_bool(env.unwrapped._x))
    env.unwrapped._ep_init_safe = float(env.unwrapped._task_safety_bool(env.unwrapped._x))
    env.unwrapped._ep_init_pe_abs = float(abs(env.unwrapped._x[IDX_PE]))
    env.unwrapped._ep_init_h = float(env.unwrapped._task_safety_margin(env.unwrapped._x))
    env.unwrapped._ep_init_h_alpha = float(h_raw[1])
    env.unwrapped._ep_init_h_beta = float(h_raw[2])
    env.unwrapped._ep_init_alpha = float(env.unwrapped._x[IDX_ALPHA])
    env.unwrapped._ep_init_abs_beta = float(abs(env.unwrapped._x[IDX_BETA]))

    return np.asarray(env.unwrapped._get_obs(env.unwrapped._x), dtype=np.float32)


def load_init_states(
    path: str,
    key: str = "init_states",
    num_episodes: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"{path} missing key={key!r}. Available keys: {list(data.keys())}")

    states = np.asarray(data[key], dtype=np.float64)
    if states.ndim != 2:
        raise ValueError(f"Expected init states shape (N, D), got {states.shape}")
    if states.shape[1] == STATE_DIM + 1:
        print("[info] Detected 17D states; dropping final FREEZE dim for F16 rollout.")
        states = states[:, :STATE_DIM]
    elif states.shape[1] != STATE_DIM:
        raise ValueError(f"Expected 16D or 17D F16 states, got {states.shape}")

    if num_episodes is not None:
        num_episodes = min(int(num_episodes), len(states))
        if seed is None:
            states = states[:num_episodes]
        else:
            rng = np.random.default_rng(seed)
            idxs = rng.choice(len(states), size=num_episodes, replace=False)
            states = states[idxs]
    return states


def sample_eval_init_states(env: Any, num_episodes: int, seed: int, safe_margin_max: float) -> np.ndarray:
    sampler = getattr(env.unwrapped, "sample_x0_eval_diag_v5", None)
    if sampler is not None:
        states = sampler(num=num_episodes, seed=seed, safe_margin_max=safe_margin_max)
        if len(states) >= num_episodes:
            return np.asarray(states[:num_episodes], dtype=np.float64)

    _, _, bb_x0 = env.unwrapped.diagnostic_grid_v5()
    states = bb_x0.reshape(-1, STATE_DIM)
    margins = np.asarray([env.unwrapped._task_safety_margin(x) for x in states], dtype=np.float64)
    states = states[margins < safe_margin_max]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(states), size=min(num_episodes, len(states)), replace=False)
    return np.asarray(states[idxs], dtype=np.float64)


def perturbation_grid_init_states(env: Any, num_episodes: int) -> np.ndarray:
    states = []
    base = np.asarray(env.unwrapped.nominal_state_v5(), dtype=np.float64)
    for i in range(num_episodes):
        delta_theta, h_alt = EVAL_PERTURBATIONS[i % len(EVAL_PERTURBATIONS)]
        state = base.copy()
        state[IDX_THETA] = float(base[IDX_THETA] + delta_theta)
        state[IDX_H] = float(h_alt)
        states.append(state)
    return np.asarray(states, dtype=np.float64)


def rollout_one(
    env: Any,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    seed: int,
    init_state: np.ndarray,
    rollout_T: int,
    value_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict:
    init_state = np.asarray(init_state, dtype=np.float64).reshape(-1)
    env.reset(seed=seed)
    obs = install_rollout_state(env, init_state)
    pred_v = float(value_fn(obs)) if value_fn is not None else float("nan")

    states = [np.asarray(env.unwrapped._x, dtype=np.float32).copy()]
    h_vals = [float(env.unwrapped._task_safety_margin(states[-1]))]
    goal_flags = [bool(env.unwrapped._task_goal_bool(states[-1]))]
    actions = []
    rewards = []
    costs = []
    infos = []
    terminated = False
    truncated = False

    while (not (terminated or truncated)) and len(actions) < rollout_T:
        action = np.clip(policy_fn(obs), env.action_space.low, env.action_space.high)
        obs, reward, cost, terminated, truncated, info = env.step(action)

        raw_state = np.asarray(info.get("raw_state", env.unwrapped._x), dtype=np.float32)
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        costs.append(float(cost))
        states.append(raw_state.copy())
        h_vals.append(float(info.get("h", env.unwrapped._task_safety_margin(raw_state))))
        goal_flags.append(bool(info.get("in_goal", env.unwrapped._task_goal_bool(raw_state))))
        infos.append(dict(info))

    success = any(bool(info.get("success", False)) for info in infos)
    crashed = any(bool(info.get("crashed", False)) for info in infos)
    max_goal_streak = max([int(info.get("max_goal_streak", 0)) for info in infos] or [0])

    return {
        "init": init_state.astype(np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "h_vals": np.asarray(h_vals, dtype=np.float32),
        "goal_flags": np.asarray(goal_flags, dtype=bool),
        "length": int(len(actions)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(success),
        "crashed": bool(crashed),
        "max_goal_streak": int(max_goal_streak),
        "return": float(np.sum(rewards)),
        "cost_sum": float(np.sum(costs)),
        "pred_v": pred_v,
    }


def tail_window(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values)
    return values[-min(n, len(values)) :]


def compute_metrics(trajs: List[Dict], env: Any, tail_steps: int = 50) -> Dict[str, Any]:
    if not trajs:
        return {}

    strict_safe = []
    tail_safe = []
    strict_stabilized_last50 = []
    lengths = []

    for traj in trajs:
        h_vals = np.asarray(traj["h_vals"], dtype=np.float32)
        goal_flags = np.asarray(traj["goal_flags"], dtype=bool)
        length = int(traj["length"])
        crashed = bool(traj.get("crashed", False))

        # Match the SSM v6_EP strict safety metric: init + every rollout state
        # safe under the env h-margin convention, and no crash/termination.
        strict_safe.append(bool(h_vals.size > 0 and np.all(h_vals <= 0.0) and (not crashed)))
        tail_safe.append(
            bool(
                length >= tail_steps
                and h_vals.size >= tail_steps
                and np.all(tail_window(h_vals, tail_steps) <= 0.0)
                and (not crashed)
            )
        )
        strict_stabilized_last50.append(
            bool(
                length >= tail_steps
                and goal_flags.size >= tail_steps
                and np.all(tail_window(goal_flags, tail_steps))
                and (not crashed)
            )
        )
        lengths.append(length)

    safe_arr = np.asarray(strict_safe, dtype=bool)
    tail_safe_arr = np.asarray(tail_safe, dtype=bool)
    stabilized_arr = np.asarray(strict_stabilized_last50, dtype=bool)
    safe_and_stable_arr = safe_arr
    pred_v_arr = np.asarray([float(t.get("pred_v", np.nan)) for t in trajs], dtype=np.float32)
    has_pred_v = np.isfinite(pred_v_arr)
    pred_positive_arr = has_pred_v & (pred_v_arr < 0.0)
    pred_negative_arr = has_pred_v & ~pred_positive_arr
    tp_arr = pred_positive_arr & safe_and_stable_arr
    fp_arr = pred_positive_arr & ~safe_and_stable_arr
    tn_arr = pred_negative_arr & ~safe_and_stable_arr
    fn_arr = pred_negative_arr & safe_and_stable_arr
    cm_total = int(np.sum(has_pred_v))
    tp, fp, tn, fn = int(np.sum(tp_arr)), int(np.sum(fp_arr)), int(np.sum(tn_arr)), int(np.sum(fn_arr))
    actual_pos = int(np.sum(has_pred_v & safe_and_stable_arr))
    actual_neg = int(np.sum(has_pred_v & ~safe_and_stable_arr))

    return {
        "n_eval": len(trajs),
        "safety_rate": float(np.mean(safe_arr)),
        "safety_rate_all": float(np.mean(safe_arr)),
        "tail_safety_rate_last50": float(np.mean(tail_safe_arr)),
        "stabilize_rate": float(np.mean(stabilized_arr)),
        "stabilize_rate_last50": float(np.mean(stabilized_arr)),
        "episode_length_mean": float(np.mean(lengths)),
        "episode_length_std": float(np.std(lengths)),
        "safe_count": int(np.sum(safe_arr)),
        "unsafe_count": int(np.sum(~safe_arr)),
        "safe_all_count": int(np.sum(safe_arr)),
        "unsafe_all_count": int(np.sum(~safe_arr)),
        "tail_safe_count": int(np.sum(tail_safe_arr)),
        "tail_unsafe_count": int(np.sum(~tail_safe_arr)),
        "stabilized_count": int(np.sum(stabilized_arr)),
        "unstabilized_count": int(np.sum(~stabilized_arr)),
        "safe_and_stable_count": int(np.sum(safe_and_stable_arr)),
        "unsafe_or_unstable_count": int(np.sum(~safe_and_stable_arr)),
        "confusion_n": cm_total,
        "confusion_tp": tp,
        "confusion_fp": fp,
        "confusion_tn": tn,
        "confusion_fn": fn,
        "confusion_actual_positive": actual_pos,
        "confusion_actual_negative": actual_neg,
        "confusion_predicted_positive": int(np.sum(pred_positive_arr)),
        "confusion_predicted_negative": int(np.sum(pred_negative_arr)),
        "confusion_accuracy": float((tp + tn) / cm_total) if cm_total else float("nan"),
        "confusion_precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "confusion_recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "confusion_fpr": float(fp / actual_neg) if actual_neg else float("nan"),
        "confusion_fnr": float(fn / actual_pos) if actual_pos else float("nan"),
        "confusion_pred_v_mean": float(np.mean(pred_v_arr[has_pred_v])) if cm_total else float("nan"),
        "safe_mask": safe_arr,
        "safe_all_mask": safe_arr,
        "tail_safe_mask": tail_safe_arr,
        "stabilized_mask": stabilized_arr,
        "safe_and_stable_mask": safe_and_stable_arr,
        "pred_v": pred_v_arr,
        "pred_positive_mask": pred_positive_arr,
        "pred_negative_mask": pred_negative_arr,
        "confusion_tp_mask": tp_arr,
        "confusion_fp_mask": fp_arr,
        "confusion_tn_mask": tn_arr,
        "confusion_fn_mask": fn_arr,
    }


def jsonable_summary(metrics: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    summary = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    summary["checkpoint"] = meta
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2))
    print(
        "F16 eval: "
        f"Safety={summary['safety_rate']:.4f} "
        f"SafetyAll={summary['safety_rate_all']:.4f} "
        f"Stabilize={summary['stabilize_rate']:.4f} "
        f"EpisodeLength={summary['episode_length_mean']:.1f}"
    )
    if int(summary.get("confusion_n", 0)) > 0:
        print(
            "Confusion(V<0 vs safe_and_stable): "
            f"TP={summary['confusion_tp']} "
            f"FP={summary['confusion_fp']} "
            f"TN={summary['confusion_tn']} "
            f"FN={summary['confusion_fn']} "
            f"Acc={summary['confusion_accuracy']:.4f} "
            f"Prec={summary['confusion_precision']:.4f} "
            f"Recall={summary['confusion_recall']:.4f}"
        )


def save_rollouts(path: str, trajs: List[Dict], metrics: Dict[str, Any], meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        n_episodes=np.asarray(len(trajs), dtype=np.int32),
        checkpoint_meta=np.asarray(json.dumps(meta), dtype=object),
        init_states=np.asarray([t["init"] for t in trajs], dtype=np.float32),
        lengths=np.asarray([t["length"] for t in trajs], dtype=np.int32),
        terminated=np.asarray([t["terminated"] for t in trajs], dtype=bool),
        truncated=np.asarray([t["truncated"] for t in trajs], dtype=bool),
        success=np.asarray([t["success"] for t in trajs], dtype=bool),
        crashed=np.asarray([t["crashed"] for t in trajs], dtype=bool),
        returns=np.asarray([t["return"] for t in trajs], dtype=np.float32),
        cost_sums=np.asarray([t["cost_sum"] for t in trajs], dtype=np.float32),
        pred_v=np.asarray(metrics["pred_v"], dtype=np.float32),
        safe_mask=np.asarray(metrics["safe_mask"], dtype=bool),
        safe_all_mask=np.asarray(metrics["safe_all_mask"], dtype=bool),
        safe_and_stable_mask=np.asarray(metrics["safe_and_stable_mask"], dtype=bool),
        stabilized_mask=np.asarray(metrics["stabilized_mask"], dtype=bool),
        pred_positive_mask=np.asarray(metrics["pred_positive_mask"], dtype=bool),
        pred_negative_mask=np.asarray(metrics["pred_negative_mask"], dtype=bool),
        confusion_tp_mask=np.asarray(metrics["confusion_tp_mask"], dtype=bool),
        confusion_fp_mask=np.asarray(metrics["confusion_fp_mask"], dtype=bool),
        confusion_tn_mask=np.asarray(metrics["confusion_tn_mask"], dtype=bool),
        confusion_fn_mask=np.asarray(metrics["confusion_fn_mask"], dtype=bool),
        **{f"states_{i}": t["states"] for i, t in enumerate(trajs)},
        **{f"actions_{i}": t["actions"] for i, t in enumerate(trajs)},
        **{f"rewards_{i}": t["rewards"] for i, t in enumerate(trajs)},
        **{f"costs_{i}": t["costs"] for i, t in enumerate(trajs)},
        **{f"h_vals_{i}": t["h_vals"] for i, t in enumerate(trajs)},
        **{f"goal_flags_{i}": t["goal_flags"] for i, t in enumerate(trajs)},
    )
    np.savez(path.replace(".npz", "_inits.npz"), init_states=np.asarray([t["init"] for t in trajs], dtype=np.float32))


def load_rollouts(path: str) -> Tuple[List[Dict], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    n = int(data["n_episodes"]) if "n_episodes" in data.files else len([k for k in data.files if k.startswith("states_")])
    meta = {}
    if "checkpoint_meta" in data.files:
        try:
            meta = json.loads(str(data["checkpoint_meta"].item()))
        except Exception:
            meta = {}

    trajs = []
    for i in range(n):
        states = np.asarray(data[f"states_{i}"], dtype=np.float32)
        trajs.append(
            {
                "init": np.asarray(data["init_states"][i], dtype=np.float32) if "init_states" in data.files else states[0],
                "states": states,
                "actions": np.asarray(data[f"actions_{i}"], dtype=np.float32) if f"actions_{i}" in data.files else np.empty((0, 4), dtype=np.float32),
                "rewards": np.asarray(data[f"rewards_{i}"], dtype=np.float32) if f"rewards_{i}" in data.files else np.asarray([], dtype=np.float32),
                "costs": np.asarray(data[f"costs_{i}"], dtype=np.float32) if f"costs_{i}" in data.files else np.asarray([], dtype=np.float32),
                "h_vals": np.asarray(data[f"h_vals_{i}"], dtype=np.float32) if f"h_vals_{i}" in data.files else np.asarray([], dtype=np.float32),
                "goal_flags": np.asarray(data[f"goal_flags_{i}"], dtype=bool) if f"goal_flags_{i}" in data.files else np.asarray([], dtype=bool),
                "length": int(data["lengths"][i]) if "lengths" in data.files else max(0, states.shape[0] - 1),
                "terminated": bool(data["terminated"][i]) if "terminated" in data.files else False,
                "truncated": bool(data["truncated"][i]) if "truncated" in data.files else False,
                "success": bool(data["success"][i]) if "success" in data.files else False,
                "crashed": bool(data["crashed"][i]) if "crashed" in data.files else False,
                "return": float(data["returns"][i]) if "returns" in data.files else float("nan"),
                "cost_sum": float(data["cost_sums"][i]) if "cost_sums" in data.files else float("nan"),
            }
        )
    return trajs, meta


def select_plot_trajs(trajs: List[Dict], best_n: int) -> List[Dict]:
    return sorted(
        trajs,
        key=lambda t: (
            bool(t["crashed"] or t["terminated"]),
            bool(np.any(np.asarray(t["h_vals"]) > 0.0)),
            -int(t.get("success", False)),
            abs(float(t["states"][-1, IDX_H]) - 100.0),
        ),
    )[:best_n]


def plot_traj3d(env: Any, trajs: List[Dict], out_path: str, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update(STYLE)
    show_trajs = [t for t in trajs if len(t["states"]) > 0]
    if not show_trajs:
        return

    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(show_trajs), 1)))

    for i, traj in enumerate(show_trajs):
        states = np.asarray(traj["states"], dtype=np.float32)
        color = colors[i % len(colors)]
        label = f"ep {i}"
        if traj["crashed"] or traj["terminated"]:
            label += " crash"
        ax.plot(states[:, IDX_PN], states[:, IDX_PE], states[:, IDX_H], lw=1.2, color=color, label=label)
        ax.scatter(states[0, IDX_PN], states[0, IDX_PE], states[0, IDX_H], color="black", s=12, marker="s")
        ax.scatter(states[-1, IDX_PN], states[-1, IDX_PE], states[-1, IDX_H], color=color, s=18, marker="o")

    all_states = np.concatenate([t["states"] for t in show_trajs], axis=0)
    pn_min, pn_max = float(np.min(all_states[:, IDX_PN])), float(np.max(all_states[:, IDX_PN]))
    if np.isclose(pn_min, pn_max):
        pn_min -= 1.0
        pn_max += 1.0
    pe_min, pe_max = -float(env.unwrapped.safe_pe), float(env.unwrapped.safe_pe)
    h_min, h_max = float(env.unwrapped.safe_h_min), float(env.unwrapped.safe_h_max)

    pp, ee = np.meshgrid([pn_min, pn_max], [pe_min, pe_max])
    ax.plot_surface(pp, ee, np.full_like(pp, h_min), alpha=0.08, color="green")
    ax.plot_surface(pp, ee, np.full_like(pp, h_max), alpha=0.08, color="green")

    pp2, hh2 = np.meshgrid([pn_min, pn_max], [h_min, h_max])
    ax.plot_surface(pp2, np.full_like(pp2, pe_min), hh2, alpha=0.06, color="red")
    ax.plot_surface(pp2, np.full_like(pp2, pe_max), hh2, alpha=0.06, color="red")

    ax.set_xlabel("PN (ft)")
    ax.set_ylabel("PE (ft)")
    ax.set_zlabel("H (ft)")
    ax.set_title(title or "F16 path (PN, PE, H)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_summary(env: Any, trajs: List[Dict], out_path: str, best_n: int = 12, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update(STYLE)
    show_trajs = select_plot_trajs(trajs, best_n)
    if not show_trajs:
        return

    metrics = compute_metrics(trajs, env)
    max_t = max(t["states"].shape[0] for t in show_trajs) * float(getattr(env.unwrapped, "dt", 0.05))
    fig = plt.figure(figsize=(13, 9))

    ax_alt = fig.add_subplot(221)
    ax_alt.axhspan(env.unwrapped.safe_h_min, env.unwrapped.safe_h_max, color="#DDF2DD", alpha=0.45, label="safe alt")
    ax_alt.axhspan(env.unwrapped.goal_h_min, env.unwrapped.goal_h_max, color="#7BC96F", alpha=0.35, label="goal alt")
    for i, traj in enumerate(show_trajs):
        states = np.asarray(traj["states"], dtype=np.float32)
        t = np.arange(states.shape[0]) * float(getattr(env.unwrapped, "dt", 0.05))
        color = COLORS[i % len(COLORS)]
        ax_alt.plot(t, states[:, IDX_H], color=color, lw=1.5, alpha=0.85)
        if traj["crashed"] or traj["terminated"]:
            ax_alt.scatter([t[-1]], [states[-1, IDX_H]], color="red", marker="x", s=50)
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("H (ft)")
    ax_alt.set_xlim(0, max_t)
    ax_alt.set_title("Altitude")
    ax_alt.grid(alpha=0.25)
    ax_alt.legend(loc="best")

    ax_margin = fig.add_subplot(222)
    ax_margin.axhline(0.0, color="black", lw=1.2, linestyle="--", label="safety boundary")
    for i, traj in enumerate(show_trajs):
        h_vals = np.asarray(traj["h_vals"], dtype=np.float32)
        t = np.arange(h_vals.shape[0]) * float(getattr(env.unwrapped, "dt", 0.05))
        color = COLORS[i % len(COLORS)]
        ax_margin.plot(t, h_vals, color=color, lw=1.5, alpha=0.85)
    ax_margin.set_xlabel("Time (s)")
    ax_margin.set_ylabel("h margin")
    ax_margin.set_xlim(0, max_t)
    ax_margin.set_title("Safety margin")
    ax_margin.grid(alpha=0.25)
    ax_margin.legend(loc="best")

    ax_pe = fig.add_subplot(223)
    ax_pe.axhspan(-env.unwrapped.safe_pe, env.unwrapped.safe_pe, color="#DDEBFF", alpha=0.5)
    for i, traj in enumerate(show_trajs):
        states = np.asarray(traj["states"], dtype=np.float32)
        color = COLORS[i % len(COLORS)]
        label = f"ep {i}"
        if traj["crashed"] or traj["terminated"]:
            label += " crash"
        ax_pe.plot(states[:, IDX_PN], states[:, IDX_PE], color=color, lw=1.4, alpha=0.85, label=label)
        ax_pe.scatter(states[0, IDX_PN], states[0, IDX_PE], color="black", s=14, marker="s")
    ax_pe.set_xlabel("PN (ft)")
    ax_pe.set_ylabel("PE (ft)")
    ax_pe.set_title("Lateral corridor")
    ax_pe.grid(alpha=0.25)
    ax_pe.legend(loc="best", fontsize=7)

    ax_pitch = fig.add_subplot(224)
    ax_pitch.axhline(env.unwrapped.safe_theta, color="black", lw=1.0, linestyle="--")
    ax_pitch.axhline(-env.unwrapped.safe_theta, color="black", lw=1.0, linestyle="--")
    for i, traj in enumerate(show_trajs):
        states = np.asarray(traj["states"], dtype=np.float32)
        t = np.arange(states.shape[0]) * float(getattr(env.unwrapped, "dt", 0.05))
        color = COLORS[i % len(COLORS)]
        ax_pitch.plot(t, states[:, IDX_THETA], color=color, lw=1.3, alpha=0.85, linestyle="-")
        ax_pitch.plot(t, states[:, IDX_ALPHA], color=color, lw=1.0, alpha=0.55, linestyle="--")
    ax_pitch.set_xlabel("Time (s)")
    ax_pitch.set_ylabel("rad")
    ax_pitch.set_xlim(0, max_t)
    ax_pitch.set_title("Theta (solid) and alpha (dashed)")
    ax_pitch.grid(alpha=0.25)

    fig.suptitle(
        title
        or (
            f"F16 RAC eval  Safety={metrics['safety_rate']:.2%}  "
            f"Stabilize={metrics['stabilize_rate']:.2%}  "
            f"Len={metrics['episode_length_mean']:.1f}  n={len(trajs)}"
        ),
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region_profile", type=str, default="v6")
    parser.add_argument("--reset_box_mode", type=str, default="ours")
    parser.add_argument("--init_curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init_curriculum_mode", type=str, default="box")
    parser.add_argument("--max_episode_steps", type=int, default=512)
    parser.add_argument("--goal_dwell_steps", type=int, default=50)


def env_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "region_profile": args.region_profile,
        "reset_box_mode": args.reset_box_mode,
        "init_curriculum": bool(args.init_curriculum),
        "init_curriculum_mode": args.init_curriculum_mode,
        "max_episode_steps": int(args.max_episode_steps),
        "goal_dwell_steps": int(args.goal_dwell_steps),
    }


def add_init_args(parser: argparse.ArgumentParser) -> None:
    for key in INIT_STATE_KEYS:
        parser.add_argument(f"--{key}", type=float, default=None)


def parse_init_overrides(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    init_opts = {}
    for key in INIT_STATE_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            init_opts[key] = float(val)
    return init_opts or None


def main() -> None:
    parser = argparse.ArgumentParser(description="RAC F16 stabilize-avoid rollout + plot")
    subparsers = parser.add_subparsers(dest="command")

    p_roll = subparsers.add_parser("rollout")
    p_roll.add_argument("--ckpt_path", "--ckpt", dest="ckpt_path", required=True)
    p_roll.add_argument("--step", type=int, default=None)
    p_roll.add_argument("--label", default="RAC-F16")
    p_roll.add_argument("--num_episodes", "--n_eval", dest="num_episodes", type=int, default=100)
    p_roll.add_argument("--seed", type=int, default=42)
    p_roll.add_argument(
        "--init_seed",
        type=int,
        default=None,
        help="Seed for selecting init states from --init_npz. Defaults to --seed.",
    )
    p_roll.add_argument("--out_dir", type=str, default=RESULTS_DIR)
    p_roll.add_argument("--plot_immediate", action="store_true")
    p_roll.add_argument("--best_n", type=int, default=12)
    p_roll.add_argument("--rollout_T", type=int, default=512)
    p_roll.add_argument(
        "--init_source",
        choices=["diag", "perturbations"],
        default="diag",
        help="diag samples the broad theta-altitude diagnostic grid; perturbations matches train_rac_f16.py eval starts.",
    )
    p_roll.add_argument("--safe_margin_max", type=float, default=-0.2)
    p_roll.add_argument("--init_npz", type=str, default=None)
    p_roll.add_argument("--init_key", type=str, default="states")
    p_roll.add_argument(
        "--confusion_matrix",
        action="store_true",
        help="Compute TP/FP/TN/FN for V(x)<0 versus full-rollout safe-and-stable.",
    )
    add_env_args(p_roll)
    add_init_args(p_roll)

    p_plot = subparsers.add_parser("plot")
    p_plot.add_argument("--trajs", nargs="+", required=True)
    p_plot.add_argument("--out", default=os.path.join(RESULTS_DIR, "rollouts.png"))
    p_plot.add_argument("--best_n", type=int, default=12)
    p_plot.add_argument("--title", default=None)
    add_env_args(p_plot)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    if args.command == "rollout":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        env = create_env(args.seed, env_kwargs=env_kwargs_from_args(args))
        agent, meta = load_agent(args.ckpt_path, args.step, env, args.seed)
        policy_fn = make_eval_policy(agent)
        value_fn = (lambda obs: predict_safety_value(agent, obs)) if args.confusion_matrix else None

        init_opts = parse_init_overrides(args)
        init_seed = args.seed if args.init_seed is None else args.init_seed
        if args.init_npz is not None:
            init_states = load_init_states(args.init_npz, args.init_key, args.num_episodes, seed=init_seed)
            print(f"Loaded {len(init_states)} init states from {args.init_npz}[{args.init_key}]")
            if init_opts is not None:
                print("  init_npz provided; ignoring --init_* overrides.")
        elif init_opts is not None:
            init_state = options_to_state(env, init_opts)
            init_states = np.tile(init_state[None, :], (args.num_episodes, 1))
        elif args.init_source == "perturbations":
            init_states = perturbation_grid_init_states(env, args.num_episodes)
        else:
            init_states = sample_eval_init_states(
                env,
                num_episodes=args.num_episodes,
                seed=args.seed,
                safe_margin_max=args.safe_margin_max,
            )

        if len(init_states) < args.num_episodes:
            print(f"Only found {len(init_states)} eval init states; requested {args.num_episodes}.")

        meta = dict(meta)
        meta["rollout_T"] = int(args.rollout_T)
        meta["init_seed"] = int(init_seed)
        print(f"Rolling out {len(init_states)} F16 episodes from {meta.get('ckpt_path', args.ckpt_path)} ...")
        trajs = [
            rollout_one(
                env,
                policy_fn,
                seed=args.seed + i,
                init_state=init_state,
                rollout_T=args.rollout_T,
                value_fn=value_fn,
            )
            for i, init_state in enumerate(init_states)
        ]

        metrics = compute_metrics(trajs, env)
        summary = jsonable_summary(metrics, meta)
        print_summary(summary)

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(out_dir / "per_rollout.json", "w", encoding="utf-8") as f:
            safe_mask = np.asarray(metrics["safe_mask"], dtype=bool)
            safe_all_mask = np.asarray(metrics["safe_all_mask"], dtype=bool)
            safe_and_stable_mask = np.asarray(metrics["safe_and_stable_mask"], dtype=bool)
            stabilized_mask = np.asarray(metrics["stabilized_mask"], dtype=bool)
            pred_positive_mask = np.asarray(metrics["pred_positive_mask"], dtype=bool)
            pred_v = np.asarray(metrics["pred_v"], dtype=np.float32)
            json.dump(
                [
                    {
                        "idx": i,
                        "length": t["length"],
                        "safe": bool(safe_mask[i]),
                        "safe_all": bool(safe_all_mask[i]),
                        "stabilized": bool(stabilized_mask[i]),
                        "safe_and_stable": bool(safe_and_stable_mask[i]),
                        "pred_v": float(pred_v[i]),
                        "predicted_positive": bool(pred_positive_mask[i]),
                        "terminated": t["terminated"],
                        "truncated": t["truncated"],
                        "init_h": float(t["init"][IDX_H]),
                        "init_theta": float(t["init"][IDX_THETA]),
                    }
                    for i, t in enumerate(trajs)
                ],
                f,
                indent=2,
            )

        rollouts_path = str(out_dir / "rollouts.npz")
        save_rollouts(rollouts_path, trajs, metrics, meta)
        print(f"Saved outputs to: {out_dir}")

        if args.plot_immediate:
            plot_summary(env, trajs, str(out_dir / "rollouts.png"), best_n=args.best_n)
            plot_traj3d(env, select_plot_trajs(trajs, args.best_n), str(out_dir / "rollouts_3d.png"))
            print(f"Saved plots to: {out_dir}")
        env.close()

    elif args.command == "plot":
        env = create_env(args.seed if hasattr(args, "seed") else 0, env_kwargs=env_kwargs_from_args(args))
        all_trajs = []
        for path in args.trajs:
            trajs, _ = load_rollouts(path)
            all_trajs.extend(trajs)
            print(f"Loaded {len(trajs)} episodes from {path}")

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_summary(env, all_trajs, str(out_path), best_n=args.best_n, title=args.title)
        plot_traj3d(env, select_plot_trajs(all_trajs, args.best_n), str(out_path.with_name(out_path.stem + "_3d.png")), title=args.title)
        print(f"Saved: {out_path}")
        env.close()


if __name__ == "__main__":
    main()
