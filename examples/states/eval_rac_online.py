#!/usr/bin/env python3
"""Rollout + eval for a RAC checkpoint on any Safety-Gymnasium environment.

Loads a checkpoint written by train_rac_online.py (which saves ckpt_*.msgpack
under <run_dir>/checkpoints/ alongside a <run_dir>/config.json) and rolls out
the deterministic policy, reporting return/cost/violation-rate stats. Optionally
renders a video of the first episode and/or plots reward, cost, and the RAC
safety critic's predicted V(s) over time.

Examples:
    python examples/states/eval_rac_online.py \
        --ckpt_path results/SafetyCarGoal1-v0/cargoal1_rac/2026-07-16_seed0000 \
        --env_name SafetyCarGoal1-v0 \
        --num_episodes 20 --render --plot --safety_value
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from jaxrl5.envs import make_safety_env
from jaxrl5.tools.load_rac import load_rac
from jaxrl5.wrappers import StaticLayoutWrapper
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


def predict_safety_value(agent, obs: np.ndarray) -> float:
    """Evaluate RAC safety value V(x) at obs using the deterministic policy action."""
    import jax.numpy as jnp

    action, _ = agent.eval_actions(np.asarray(obs, dtype=np.float32))
    obs_b = jnp.asarray(np.asarray(obs, dtype=np.float32)[None, :])
    action_b = jnp.asarray(np.asarray(action, dtype=np.float32)[None, :])
    value = agent.safety_critic.apply_fn(
        {"params": agent.safety_critic.params}, obs_b, action_b, False
    )
    return float(np.asarray(value).reshape(-1)[0])


def rollout_one(env, obs, policy_fn, value_fn=None, render: bool = False, max_episode_steps=1000) -> Dict[str, Any]:
    frames = [env.render()] if render else []
    rewards: List[float] = []
    costs: List[float] = []
    values = [float(value_fn(obs))] if value_fn is not None else []
    terminated = truncated = False

    while not (terminated or truncated):
        action = np.clip(policy_fn(obs), env.action_space.low, env.action_space.high)
        obs, reward, cost, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        costs.append(float(cost))
        if len(rewards) >= max_episode_steps:
            truncated = True
        if value_fn is not None:
            values.append(float(value_fn(obs)))
        if render:
            frames.append(env.render())

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    costs_arr = np.asarray(costs, dtype=np.float32)
    return {
        "return": float(rewards_arr.sum()),
        "cost_sum": float(costs_arr.sum()),
        "length": int(len(rewards_arr)),
        "violation_rate": float(np.mean(costs_arr > 0.0)) if len(costs_arr) else 0.0,
        "rewards": rewards_arr,
        "costs": costs_arr,
        "values": np.asarray(values, dtype=np.float32),
        "frames": frames,
    }


def save_video(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    import imageio

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, [np.asarray(f, dtype=np.uint8) for f in frames], fps=fps)


def plot_episode(traj: Dict[str, Any], out_path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 3 if traj["values"].size else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.3 * n_panels, 4))

    t = np.arange(len(traj["rewards"]))
    axes[0].plot(t, np.cumsum(traj["rewards"]))
    axes[0].set_title("Cumulative reward")
    axes[0].set_xlabel("step")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, np.cumsum(traj["costs"]), color="crimson")
    axes[1].set_title("Cumulative cost")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.25)

    if traj["values"].size:
        axes[2].axhline(0.0, color="black", lw=1.0, linestyle="--", label="safety boundary")
        axes[2].plot(traj["values"], color="darkorange")
        axes[2].set_title("Predicted safety value V(s)")
        axes[2].set_xlabel("step")
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAC Safety-Gymnasium rollout + eval")
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Checkpoint file, or a run/checkpoints dir to auto-pick the latest ckpt_*.msgpack.",
    )
    parser.add_argument("--step", type=int, default=None, help="Specific checkpoint step to load.")
    parser.add_argument(
        "--env_name", required=True, help="Safety-Gymnasium environment id, e.g. SafetyCarGoal1-v0."
    )
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/evaluations/eval_states")
    parser.add_argument("--render", action="store_true", help="Save an mp4 of the first episode.")
    parser.add_argument(
        "--safety_value",
        action="store_true",
        help="Track the RAC safety critic's predicted V(s) at every step.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Save a reward/cost/value plot for the first episode."
    )
    parser.add_argument(
        "--static_seed",
        type=int,
        default=None,
        help="If set, pin every rollout episode to this exact layout seed (matching "
        "--static_seed used during training) instead of varying the layout per episode.",
    )
    parser.add_argument(
        "--init_states",
        type=str,
        default=None,
        help="Path to an .npz with 'qpos'/'qvel' arrays (as saved by save_init_states.py) "
        "to pin each rollout's MuJoCo state, overriding the seed-based reset. Requires "
        "num_episodes <= number of saved states.",
    )
    args = parser.parse_args()
    if args.num_episodes < 1 or args.max_episode_steps < 1:
        parser.error('Episode count and step limit must be positive')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_mode = "rgb_array" if args.render else None
    env = make_safety_env(args.env_name, seed=args.seed, render_mode=render_mode)
    if args.static_seed is not None:
        env = StaticLayoutWrapper(env, seed=args.static_seed)

    agent, policy_fn, meta = load_rac(
        args.ckpt_path,
        step=args.step,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        deterministic=True,
    )
    value_fn = (lambda obs: predict_safety_value(agent, obs)) if args.safety_value else None
    with open(meta['config_path'], encoding='utf-8') as f:
        saved_config = json.load(f)
    if saved_config.get('flags', {}).get('normalize_actions', False):
        env = SymmetricActionWrapper(env)

    init_states = None
    if args.init_states is not None:
        loaded = np.load(args.init_states)
        init_states = (loaded["qpos"], loaded["qvel"])
        assert args.num_episodes <= len(init_states[0]), (
            f"--num_episodes={args.num_episodes} exceeds the "
            f"{len(init_states[0])} states saved in {args.init_states}"
        )

    trajs = []
    for i in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + i)
        if init_states is not None:
            env.unwrapped.set_state(init_states[0][i], init_states[1][i])
            obs = env.unwrapped._get_obs()
        traj = rollout_one(env, obs, policy_fn, value_fn=value_fn,
                           render=(args.render and i == 0), max_episode_steps=args.max_episode_steps)
        trajs.append(traj)
        print(
            f"episode {i}: return={traj['return']:.2f} cost={traj['cost_sum']:.2f} "
            f"len={traj['length']} viol_rate={traj['violation_rate']:.3f}"
        )

    returns = np.asarray([t["return"] for t in trajs], dtype=np.float32)
    costs = np.asarray([t["cost_sum"] for t in trajs], dtype=np.float32)
    lengths = np.asarray([t["length"] for t in trajs], dtype=np.float32)
    viol_rates = np.asarray([t["violation_rate"] for t in trajs], dtype=np.float32)

    summary = {
        "checkpoint": meta,
        "env_name": args.env_name,
        "n_episodes": len(trajs),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "cost_mean": float(costs.mean()),
        "cost_std": float(costs.std()),
        "episode_length_mean": float(lengths.mean()),
        "violation_rate_mean": float(viol_rates.mean()),
    }
    print(json.dumps(summary, indent=2))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.render and trajs[0]["frames"]:
        video_path = str(out_dir / "episode0.mp4")
        save_video(trajs[0]["frames"], video_path)
        print(f"Saved video to {video_path}")

    if args.plot:
        plot_path = str(out_dir / "episode0.png")
        plot_episode(trajs[0], plot_path, title=f"{args.env_name} RAC eval (ep 0)")
        print(f"Saved plot to {plot_path}")

    env.close()


if __name__ == "__main__":
    main()
