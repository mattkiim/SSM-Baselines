#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from jaxrl5.agents.rac.rac_learner import RACLearner
from jaxrl5.envs.registration import ensure_custom_envs_registered
from jaxrl5.tools.load_rac import load_rac
from jaxrl5.wrappers import AddCostFromInfo
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


def create_env(env_id: str, seed: int) -> gym.Env:
    ensure_custom_envs_registered()
    env = gym.make(env_id)

    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(env.action_space.high, 1.0):
        env = SymmetricActionWrapper(env)

    env = AddCostFromInfo(env)
    env.reset(seed=seed)
    return env


def make_eval_policy(agent: RACLearner) -> Callable[[np.ndarray], np.ndarray]:
    eval_agent = agent

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(np.asarray(obs, dtype=np.float32))
        return np.asarray(action, dtype=np.float32)

    return policy


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


def split_state_ref(obs: np.ndarray, state_dim: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32)

    if state_dim is None:
        if obs.shape[0] % 2 != 0:
            raise ValueError(
                f"Observation dim {obs.shape[0]} is odd; pass --state_dim explicitly."
            )
        state_dim = obs.shape[0] // 2

    state = obs[:state_dim]
    ref = obs[state_dim:state_dim * 2]

    if ref.shape[0] != state.shape[0]:
        raise ValueError(
            f"State/ref mismatch: state dim {state.shape[0]}, ref dim {ref.shape[0]}."
        )

    return state, ref


def crash_predicate(
    state: np.ndarray,
    *,
    z_idx: int,
    x_idx: int,
    y_idx: int | None,
    z_crash: float,
    x_limit: float,
    use_radius: bool,
    z_positive_down: bool,
) -> bool:
    x = float(state[x_idx])
    z = float(state[z_idx])

    if z_positive_down:
        z_bad = z > abs(z_crash)
    else:
        z_bad = z < -abs(z_crash)

    if use_radius:
        if y_idx is None:
            lateral = abs(x)
        else:
            y = float(state[y_idx])
            lateral = float(np.sqrt(x * x + y * y))
        x_bad = lateral > x_limit
    else:
        x_bad = abs(x) > x_limit

    return bool(z_bad or x_bad)


def sample_valid_init(
    rng: np.random.Generator,
    *,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    z_idx: int,
    x_idx: int,
    y_idx: int | None,
    z_crash: float,
    x_limit: float,
    use_radius: bool,
    z_positive_down: bool,
    min_abs_z: float,
    max_tries: int,
) -> Dict[str, float]:
    for _ in range(max_tries):
        init_x = float(rng.uniform(*x_range))
        init_y = float(rng.uniform(*y_range))
        init_z = float(rng.uniform(*z_range))

        # Boyang note: reject near boundary / invalid z.
        if abs(init_z) <= min_abs_z:
            continue

        dummy = np.zeros(9, dtype=np.float32)
        dummy[x_idx] = init_x
        if y_idx is not None:
            dummy[y_idx] = init_y
        dummy[z_idx] = init_z

        if crash_predicate(
            dummy,
            z_idx=z_idx,
            x_idx=x_idx,
            y_idx=y_idx,
            z_crash=z_crash,
            x_limit=x_limit,
            use_radius=use_radius,
            z_positive_down=z_positive_down,
        ):
            continue

        return {
            "init_x": init_x,
            "init_y": init_y,
            "init_z": init_z,
            "init_vx": 0.0,
            "init_vy": 0.0,
            "init_vz": 0.0,
            "init_phi": 0.0,
            "init_theta": 0.0,
            "init_psi": 0.0,
        }

    raise RuntimeError("Failed to sample a valid init. Widen ranges or relax rejection criteria.")


def reset_with_init(env: gym.Env, seed: int, init: Dict[str, float]):
    try:
        return env.reset(seed=seed, options=init)
    except TypeError:
        # Some envs do not support options. Try mutating common init attrs.
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "INIT_STATE"):
            state = np.zeros(9, dtype=np.float32)
            state[0] = init["init_x"]
            state[1] = init["init_y"]
            state[2] = init["init_z"]
            state[3] = init["init_vx"]
            state[4] = init["init_vy"]
            state[5] = init["init_vz"]
            state[6] = init["init_phi"]
            state[7] = init["init_theta"]
            state[8] = init["init_psi"]
            unwrapped.INIT_STATE = state
        return env.reset(seed=seed)


def rollout_one(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int,
    init: Dict[str, float],
    state_dim: int | None,
    z_idx: int,
    x_idx: int,
    y_idx: int | None,
    z_crash: float,
    x_limit: float,
    use_radius: bool,
    z_positive_down: bool,
) -> Dict[str, np.ndarray | float | bool | Dict[str, float]]:
    obs, info = reset_with_init(env, seed, init)

    states, refs, actions, rewards, costs = [], [], [], [], []
    crashed = False

    terminated = False
    truncated = False

    while not (terminated or truncated):
        state, ref = split_state_ref(obs, state_dim)
        action = policy_fn(obs)

        states.append(state)
        refs.append(ref)
        actions.append(action)

        if crash_predicate(
            state,
            z_idx=z_idx,
            x_idx=x_idx,
            y_idx=y_idx,
            z_crash=z_crash,
            x_limit=x_limit,
            use_radius=use_radius,
            z_positive_down=z_positive_down,
        ):
            crashed = True
            break

        obs, reward, cost, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        costs.append(float(cost))

    states = np.asarray(states, dtype=np.float32)
    refs = np.asarray(refs, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    terminal_error = float(np.linalg.norm(states[-1] - refs[-1])) if len(states) else float("nan")
    terminal_pos_error = float(np.linalg.norm(states[-1, :3] - refs[-1, :3])) if len(states) else float("nan")

    return {
        "init": init,
        "states": states,
        "refs": refs,
        "actions": actions,
        "rewards": np.asarray(rewards, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "terminal_error": terminal_error,
        "terminal_pos_error": terminal_pos_error,
        "crashed": crashed,
        "return": float(np.sum(rewards)),
        "cost_sum": float(np.sum(costs)),
        "length": int(len(states)),
    }


def plot_rollouts_xz(rollouts: List[Dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, r in enumerate(rollouts):
        states = r["states"]
        refs = r["refs"]

        if states.size == 0:
            continue

        label = f"traj {i}"
        if r["crashed"]:
            label += " crashed"

        line, = ax.plot(states[:, 0], states[:, 2], linewidth=2, label=label)
        color = line.get_color()
        ax.scatter(states[0, 0], states[0, 2], color=color, marker="o", s=20)
        # ax.scatter(states[-1, 0], states[-1, 2], marker="x", s=36)

        if i == 0 and refs.size:
            ax.plot(refs[:, 0], refs[:, 2], linestyle="--", linewidth=2, label="ref")

    ax.axhline(-0.3, color="black", linewidth=1.5, linestyle="--", label="z crash")
    ax.axvline(-3.5, color="black", linewidth=1.0, linestyle=":")
    ax.axvline(3.5, color="black", linewidth=1.0, linestyle=":")

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Quad3D rollouts: x-z projection")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rollouts_xy(rollouts: List[Dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, r in enumerate(rollouts):
        states = r["states"]
        if states.size == 0:
            continue
        ax.plot(states[:, 0], states[:, 1], linewidth=2, label=f"traj {i}")

    circle = plt.Circle((0.0, 0.0), 3.5, fill=False, linestyle="--", linewidth=1.5)
    ax.add_patch(circle)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Quad3D rollouts: x-y projection")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    refs = rollouts[0]["refs"]
    plt.plot(refs[:, 0], refs[:, 1], '--', label="ref")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rollouts_3d(rollouts: List[Dict], out_path: str) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for i, r in enumerate(rollouts):
        states = r["states"]
        refs = r["refs"]

        if states.size == 0:
            continue

        label = f"traj {i}"
        if r["crashed"]:
            label += " crashed"

        # ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=2, label=label)
        # ax.scatter(states[0, 0], states[0, 1], states[0, 2], marker="o", s=20)

        line, = ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=2, label=label)
        color = line.get_color()
        ax.scatter(states[0, 0], states[0, 1], states[0, 2], marker="o", color=color, s=20)

        if i == 0 and refs.size:
            ax.plot(
                refs[:, 0],
                refs[:, 1],
                refs[:, 2],
                linestyle="--",
                linewidth=2,
                label="ref",
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Quad3D rollouts: 3D trajectory")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1)

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--env_id", type=str, default="QuadrotorTracking3D-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="results/evaluations/eval_quad3d")

    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--state_dim", type=int, default=9)

    p.add_argument("--x_range", type=float, nargs=2, default=[-2.5, 2.5])
    p.add_argument("--y_range", type=float, nargs=2, default=[-2.5, 2.5])
    p.add_argument("--z_range", type=float, nargs=2, default=[1.0, 2.0])
    p.add_argument("--min_abs_z", type=float, default=1e-3)

    p.add_argument("--x_limit", type=float, default=3.5)
    p.add_argument("--z_crash", type=float, default=0.3)
    p.add_argument("--use_radius", action="store_true")
    p.add_argument("--z_positive_down", action="store_true")

    p.add_argument("--max_init_tries", type=int, default=10_000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = create_env(args.env_id, args.seed)
    agent = load_agent(args.ckpt_path, args.step, env, args.seed)
    policy_fn = make_eval_policy(agent)

    rng = np.random.default_rng(args.seed)

    rollouts = []
    rows = []

    for i in range(args.n_eval):
        init = sample_valid_init(
            rng,
            x_range=tuple(args.x_range),
            y_range=tuple(args.y_range),
            z_range=tuple(args.z_range),
            z_idx=2,
            x_idx=0,
            y_idx=1,
            z_crash=args.z_crash,
            x_limit=args.x_limit,
            use_radius=args.use_radius,
            z_positive_down=args.z_positive_down,
            min_abs_z=args.min_abs_z,
            max_tries=args.max_init_tries,
        )

        r = rollout_one(
            env,
            policy_fn,
            seed=args.seed + i,
            init=init,
            state_dim=args.state_dim,
            z_idx=2,
            x_idx=0,
            y_idx=1,
            z_crash=args.z_crash,
            x_limit=args.x_limit,
            use_radius=args.use_radius,
            z_positive_down=args.z_positive_down,
        )

        rollouts.append(r)
        rows.append({
            "idx": i,
            "terminal_error": r["terminal_error"],
            "terminal_pos_error": r["terminal_pos_error"],
            "crashed": bool(r["crashed"]),
            "return": r["return"],
            "cost_sum": r["cost_sum"],
            "length": r["length"],
            **{f"init_{k}": v for k, v in init.items()},
        })

    terminal_errors = np.asarray([r["terminal_error"] for r in rollouts], dtype=np.float32)
    terminal_pos_errors = np.asarray([r["terminal_pos_error"] for r in rollouts], dtype=np.float32)
    crashed = np.asarray([r["crashed"] for r in rollouts], dtype=np.float32)

    summary = {
        "n_eval": args.n_eval,
        "terminal_error_mean": float(np.nanmean(terminal_errors)),
        "terminal_error_std": float(np.nanstd(terminal_errors)),
        "terminal_pos_error_mean": float(np.nanmean(terminal_pos_errors)),
        "terminal_pos_error_std": float(np.nanstd(terminal_pos_errors)),
        "crash_rate": float(np.mean(crashed)),
        "crash_count": int(np.sum(crashed)),
        "crash_rule": {
            "x_limit": args.x_limit,
            "z_crash": args.z_crash,
            "use_radius": args.use_radius,
            "z_positive_down": args.z_positive_down,
        },
    }

    print(json.dumps(summary, indent=2))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "per_rollout.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    np.savez(
        out_dir / "rollouts.npz",
        states=np.array([r["states"] for r in rollouts], dtype=object),
        refs=np.array([r["refs"] for r in rollouts], dtype=object),
        actions=np.array([r["actions"] for r in rollouts], dtype=object),
        crashed=crashed,
        terminal_errors=terminal_errors,
        terminal_pos_errors=terminal_pos_errors,
    )

    # r0 = rollouts[0]
    # states = r0["states"]
    # refs = r0["refs"]

    # print("state_dim:", args.state_dim)
    # print("states shape:", states.shape)
    # print("refs shape:", refs.shape)

    # print("ref first xyz:", refs[0, :3])
    # print("ref last xyz:", refs[-1, :3])
    # print("state first xyz:", states[0, :3])
    # print("state last xyz:", states[-1, :3])

    # print("state z range:", states[:, 2].min(), states[:, 2].max())
    # print("ref z range:", refs[:, 2].min(), refs[:, 2].max())

    # quit()

    plot_rollouts_xz(rollouts[: min(args.n_eval, 20)], str(out_dir / "rollouts_xz.png"))
    plot_rollouts_xy(rollouts[: min(args.n_eval, 20)], str(out_dir / "rollouts_xy.png"))
    plot_rollouts_3d(rollouts[: min(args.n_eval, 20)], str(out_dir / "rollouts_3d.png"))

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
