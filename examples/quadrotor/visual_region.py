#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Dict, Tuple, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from jaxrl5.envs import make_env
from jaxrl5.tools import load_agent


# -----------------------------
# Helpers
# -----------------------------
def reference_circle(num_points: int = 360):
    ang = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = np.cos(ang)
    z = 1.0 + np.sin(ang)
    return x, z


def step_env(env, action: np.ndarray):
    """Handle both 5-tuple and 6-tuple step APIs."""
    out = env.step(action)
    if len(out) == 6:
        next_obs, reward, cost, terminated, truncated, info = out
    else:
        next_obs, reward, terminated, truncated, info = out
        cost = float(info.get("cost", 0.0))
    return next_obs, float(reward), float(cost), bool(terminated), bool(truncated), info


def reseed_agent_if_possible(agent, seed: int):
    """
    Try to reseed stochastic agents (esp. SSM) so MC rollouts are not identical.
    If the agent doesn't expose rng-like fields, this becomes a no-op.
    """
    try:
        import jax
    except Exception:
        return agent

    if not hasattr(agent, "replace"):
        return agent

    key = jax.random.PRNGKey(seed)
    for name in ("rng", "rng_key", "key"):
        if hasattr(agent, name):
            try:
                return agent.replace(**{name: key})
            except Exception:
                pass
    return agent


def agent_act(agent_local, obs_np: np.ndarray, deterministic: bool):
    """
    Unified action interface:
      - If returns (action, new_agent): update agent_local
      - If returns action only: keep agent_local
    """
    if deterministic and hasattr(agent_local, "eval_actions"):
        out = agent_local.eval_actions(obs_np)
    elif (not deterministic) and hasattr(agent_local, "sample_actions"):
        out = agent_local.sample_actions(obs_np)
    elif hasattr(agent_local, "eval_actions"):
        out = agent_local.eval_actions(obs_np)
    else:
        raise AttributeError("Agent has neither eval_actions nor sample_actions.")

    if isinstance(out, (tuple, list)) and len(out) == 2:
        action, new_agent = out
        return np.asarray(action, dtype=np.float32), new_agent
    return np.asarray(out, dtype=np.float32), agent_local


# -----------------------------
# Rollout + Grid eval
# -----------------------------
def rollout_violation_stats(
    env,
    agent,
    *,
    init_opts: Dict,
    horizon: int,
    deterministic: bool,
    seed: int,
) -> Tuple[bool, float]:
    """
    violated: whether any step violates (cost>0 or h>0)
    max_h: max(info["h"]) over the rollout if available, else nan
    """
    obs, info = env.reset(seed=seed, options=init_opts)

    # If reset already unsafe, early return (saves time and consistent).
    h0 = info.get("h", None)
    if h0 is not None:
        try:
            if float(h0) > 0.0:
                return True, float(h0)
        except Exception:
            pass

    max_h = -np.inf
    agent_local = reseed_agent_if_possible(agent, seed)

    for _ in range(horizon):
        obs_np = np.asarray(obs, dtype=np.float32)
        action, agent_local = agent_act(agent_local, obs_np, deterministic)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, cost, terminated, truncated, info = step_env(env, action)

        h_val = info.get("h", None)
        if h_val is not None:
            try:
                max_h = max(max_h, float(h_val))
            except Exception:
                pass

        if cost > 0.0 or (h_val is not None and float(h_val) > 0.0):
            return True, (float(max_h) if max_h != -np.inf else float("nan"))

        if terminated or truncated:
            break

    if max_h == -np.inf:
        max_h = float("nan")
    return False, float(max_h)


def evaluate_grid(
    env,
    agent,
    *,
    zdot: float,
    x_lin: np.ndarray,
    z_lin: np.ndarray,
    mc: int,
    horizon: int,
    deterministic: bool,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      P: [Nz, Nx] violation probability in [0,1]
      M: [Nz, Nx] mean max_h
    """
    Nx = len(x_lin)
    Nz = len(z_lin)
    P = np.zeros((Nz, Nx), dtype=np.float32)
    M = np.full((Nz, Nx), np.nan, dtype=np.float32)

    common = dict(
        init_vx=0.0,
        init_theta=0.0,
        init_omega=0.0,
        init_waypoint_idx=0,
    )

    for iz, z0 in enumerate(z_lin):
        for ix, x0 in enumerate(x_lin):
            n_viol = 0
            max_h_list = []

            init_opts = dict(common)
            init_opts.update(
                init_x=float(x0),
                init_z=float(z0),
                init_vz=float(zdot),
            )

            for k in range(mc):
                seed = base_seed + 100000 * (iz * Nx + ix) + k
                violated, max_h = rollout_violation_stats(
                    env,
                    agent,
                    init_opts=init_opts,
                    horizon=horizon,
                    deterministic=deterministic,
                    seed=seed,
                )
                n_viol += int(violated)
                max_h_list.append(max_h)

                # if already all violated, break early
                if n_viol == mc:
                    break

            P[iz, ix] = n_viol / float(mc)

            arr = np.asarray(max_h_list, dtype=np.float32)
            if np.isfinite(arr).any():
                M[iz, ix] = np.nanmean(arr)

    return P, M


# -----------------------------
# Plotting
# -----------------------------
def plot_risk_slices(
    results: Dict[float, Dict[str, np.ndarray]],
    *,
    x_lin: np.ndarray,
    z_lin: np.ndarray,
    out_png: str,
    corridor_zmin: float = 0.5,
    corridor_zmax: float = 1.5,
    show_circle: bool = True,
    bin_size: float = 0.1,
):
    """
    Plot p_viol heatmaps (discrete bins):
      - Higher p_viol => deeper purple
      - Discrete bins: 0.0,0.1,...,1.0 by default
    """
    z_list = sorted(results.keys())
    ncols = len(z_list)

    # --- figure layout: last column reserved for colorbar (no overlap) ---
    fig_w = 5.2 * ncols + 0.8  # extra width for colorbar
    fig_h = 4.8
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(
        1, ncols + 1,
        width_ratios=([1.0] * ncols + [0.06]),
        wspace=0.15
    )

    axes = []
    for i in range(ncols):
        if i == 0:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axes[0])
        axes.append(ax)
    cax = fig.add_subplot(gs[0, -1])  # dedicated colorbar axis

    extent = [x_lin.min(), x_lin.max(), z_lin.min(), z_lin.max()]
    circle_x, circle_z = reference_circle()

    # --- discrete bins ---
    boundaries = np.arange(0.0, 1.0 + bin_size, bin_size)
    boundaries[-1] = 1.0  # ensure exact 1.0
    n_bins = len(boundaries) - 1

    # Use reversed viridis: high value => purple, low value => yellow/green
    cmap = plt.cm.get_cmap("viridis", n_bins)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    last_im = None
    for ax, zdot in zip(axes, z_list):
        P = results[zdot]["P"]

        last_im = ax.imshow(
            P,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        # corridor lines
        ax.axhline(corridor_zmin, color="black", linewidth=2.2)
        ax.axhline(corridor_zmax, color="black", linewidth=2.2)

        # reference circle
        if show_circle:
            ax.plot(circle_x, circle_z, "k--", linewidth=1.2)

        # --- typography upgrades ---
        ax.set_title(rf"$\dot z = {zdot:.1f}$", fontsize=16, pad=8)
        ax.set_xlabel("x", fontsize=14)

        if ax is axes[0]:
            ax.set_ylabel("z", fontsize=14)
        else:
            # hide duplicate y tick labels for shared y
            ax.tick_params(labelleft=False)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=12,
            length=5,
            width=1.2,
            direction="out",
        )

    # --- colorbar in its own axis (no overlap) ---
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label(r"$p_{\mathrm{viol}}$", fontsize=14, labelpad=10)
    cbar.set_ticks(boundaries)
    cbar.set_ticklabels([f"{t:.1f}" for t in boundaries])
    cbar.ax.tick_params(labelsize=12, length=4, width=1.1, direction="out")

    # save
    out_dir = os.path.dirname(out_png) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved: {out_png}")


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser("Risk heatmap visualization (zdot slices)")
    p.add_argument("--env_name", default="QuadrotorTracking2D-v0")
    p.add_argument("--algo", choices=["td3", "td3_lag", "ssm"], default="ssm")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="deterministic actions (may collapse probabilities)")
    p.add_argument("--ddpm_temperature", type=float, default=None)
    p.add_argument("--config_path", default=None)

    # grid
    p.add_argument("--x_min", type=float, default=-1.6)
    p.add_argument("--x_max", type=float, default=1.6)
    p.add_argument("--z_min", type=float, default=0.0)
    p.add_argument("--z_max", type=float, default=2.0)
    p.add_argument("--nx", type=int, default=81)
    p.add_argument("--nz", type=int, default=81)

    # slices
    p.add_argument("--zdot_list", type=str, default="-1.0,0.0,1.0")

    # visualization bins (default 0.1 matches mc=10)
    p.add_argument("--bin_size", type=float, default=0.1, help="color bin size for p_viol (default 0.1)")

    p.add_argument("--out_png", type=str, default="results/visualizations/risk_heatmap.png")

    args = p.parse_args()

    # ✅固定：每个点10次rollout，每次40步
    mc = 10
    horizon = 40

    zdot_list = [float(s.strip()) for s in args.zdot_list.split(",") if s.strip()]

    env = make_env(args.env_name, seed=args.seed)

    agent, policy_fn, meta = load_agent(
        args.algo,
        args.ckpt_path,
        step=args.step,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        deterministic=args.deterministic,
        ddpm_temperature=args.ddpm_temperature,
        config_path=args.config_path,
    )

    x_lin = np.linspace(args.x_min, args.x_max, args.nx, dtype=np.float32)
    z_lin = np.linspace(args.z_min, args.z_max, args.nz, dtype=np.float32)

    results = {}
    for zdot in zdot_list:
        print(f"=== evaluating zdot={zdot:.2f} (mc={mc}, horizon={horizon}) ===")
        P, M = evaluate_grid(
            env,
            agent,
            zdot=zdot,
            x_lin=x_lin,
            z_lin=z_lin,
            mc=mc,
            horizon=horizon,
            deterministic=args.deterministic,
            base_seed=args.seed + int((zdot + 10) * 1000),
        )
        results[zdot] = {"P": P, "M": M}

    plot_risk_slices(
        results,
        x_lin=x_lin,
        z_lin=z_lin,
        out_png=args.out_png,
        bin_size=args.bin_size,
    )

    env.close()


if __name__ == "__main__":
    main()
