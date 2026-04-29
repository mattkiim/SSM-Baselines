import argparse
import csv
import os
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from jaxrl5.envs import make_env
from jaxrl5.tools import load_agent


def _reference_circle(num_points: int = 360):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = np.cos(angles)
    z = 1.0 + np.sin(angles)
    return x, z


def rollout_episodes(env, policy_fn, episodes: int, deterministic: bool, seed: int, algo: str):
    trajectories: List[List[Dict[str, float]]] = []
    for ep in range(episodes):
        obs, info = env.reset(
            seed=seed + ep,
            options={
                "init_x": 1.0,
                "init_z": 1.0,
                "init_vx": 0.0,
                "init_vz": 0.0,
                "init_theta": 0.0,
                "init_omega": 0.0,
                "init_waypoint_idx": 0,
            },
        )

        print("start x,z =", float(obs[0]), float(obs[2]))
        ep_data: List[Dict[str, float]] = []
        t = 0
        while True:
            action = policy_fn(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            step_out = env.step(action)
            if len(step_out) == 6:
                next_obs, reward, cost, terminated, truncated, info = step_out
            else:
                next_obs, reward, terminated, truncated, info = step_out
                cost = float(info.get("cost", 0.0))
            x, z = float(obs[0]), float(obs[2])
            x_ref, z_ref = float(obs[6]), float(obs[8])
            idx = int(info.get("idx", t % 360))
            ep_data.append(
                {
                    "t": t,
                    "x": x,
                    "z": z,
                    "x_ref": x_ref,
                    "z_ref": z_ref,
                    "reward": float(reward),
                    "cost": float(cost),
                    "h": float(info.get("h", 0.0)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "idx": idx,
                }
            )
            obs = next_obs
            t += 1
            if terminated or truncated:
                break
        trajectories.append(ep_data)
    return trajectories


def plot_trajectories(trajectories: List[List[Dict[str, float]]], out_path: str, algo: str, checkpoint_label: str) -> None:
    plt.figure(figsize=(8, 8))
    plt.axhline(0.5, color="black", linestyle="-")
    plt.axhline(1.5, color="black", linestyle="-")
    circle_x, circle_z = _reference_circle()
    plt.plot(circle_x, circle_z, "k--", label="reference circle")

    colors = plt.cm.tab10.colors
    for i, ep_data in enumerate(trajectories):
        xs = [d["x"] for d in ep_data]
        zs = [d["z"] for d in ep_data]
        status = "term" if ep_data[-1]["terminated"] else ("trunc" if ep_data[-1]["truncated"] else "done")
        plt.plot(xs, zs, color=colors[i % len(colors)], label=f"episode {i} ({status})")

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"{algo.upper()} trajectory @ {checkpoint_label}")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def save_csv(trajectories: List[List[Dict[str, float]]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "episode",
        "t",
        "x",
        "z",
        "x_ref",
        "z_ref",
        "reward",
        "cost",
        "h",
        "terminated",
        "truncated",
        "idx",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep_idx, ep_data in enumerate(trajectories):
            for row in ep_data:
                writer.writerow({"episode": ep_idx, **row})


def main():
    parser = argparse.ArgumentParser(description="Visualize Quadrotor policy trajectories")
    parser.add_argument("--env_name", default="QuadrotorTracking2D-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=None, help="Alias for num_episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (eval)")
    parser.add_argument("--algo", choices=["td3", "td3_lag", "ssm", "cal"], default=None)
    parser.add_argument("--agent", choices=["td3", "td3_lag", "ssm", "cal"], default=None, help="Deprecated alias for --algo")
    parser.add_argument("--ckpt_path", default=None, help="Checkpoint directory or file")
    parser.add_argument("--checkpoint_dir", default=None, help="Deprecated alias for --ckpt_path")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (optional)")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Deprecated alias for --step")
    parser.add_argument("--out_dir", default="results/visualizations")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--ssm_ddpm_temperature", type=float, default=None, help="Override SSM ddpm_temperature during evaluation")
    parser.add_argument("--config_path", default=None, help="Optional path to SSM config.json/variant.json")
    args = parser.parse_args()

    algo = args.algo if args.algo is not None else (args.agent or "td3")
    ckpt_path = args.ckpt_path or args.checkpoint_dir
    if ckpt_path is None:
        parser.error("--ckpt_path (or --checkpoint_dir) is required")
    step = args.step if args.step is not None else args.checkpoint_step
    episodes = args.episodes if args.episodes is not None else args.num_episodes

    env = make_env(args.env_name, seed=args.seed)

    agent, policy_fn, meta = load_agent(
        algo,
        ckpt_path,
        step=step,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        deterministic=args.deterministic,
        ddpm_temperature=args.ssm_ddpm_temperature,
        config_path=args.config_path,
    )
    resolved_ckpt = meta.get("ckpt_resolved_path", ckpt_path)

    trajectories = rollout_episodes(
        env,
        policy_fn,
        episodes=episodes,
        deterministic=args.deterministic,
        seed=args.seed,
        algo=algo,
    )

    ckpt_label = os.path.basename(resolved_ckpt)
    out_png = os.path.join(args.out_dir, f"traj_{algo}_{ckpt_label}.png")
    plot_trajectories(trajectories, out_png, algo, ckpt_label)
    print(f"Saved trajectory plot to {out_png}")

    if args.save_csv:
        out_csv = os.path.join(args.out_dir, f"traj_{algo}_{ckpt_label}.csv")
        save_csv(trajectories, out_csv)
        print(f"Saved trajectory CSV to {out_csv}")

    env.close()


if __name__ == "__main__":
    main()
