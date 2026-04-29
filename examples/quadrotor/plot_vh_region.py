#!/usr/bin/env python3
"""Visualize V_h(s) ≈ min_a Q_h(s,a) on (x,z) slices for QuadrotorTracking2D-v0."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization
from flax.core import frozen_dict

from jaxrl5.envs.quadrotor_tracking_2d import make_quadrotor_tracking_2d_env
from jaxrl5.networks import ddpm_sampler
from jaxrl5.tools.checkpoints import resolve_checkpoint
from jaxrl5.tools.load_rac import load_rac
from jaxrl5.tools.load_sac_lag import load_sac_lag
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper
from jaxrl5.agents.safe_matching.safe_matching_learner import (
    SafeScoreMatchingLearner,
)


def parse_float_list(text: str) -> List[float]:
    items = [t.strip() for t in text.split(",") if t.strip()]
    return [float(item) for item in items]


def create_env(env_name: str, seed: int) -> gym.Env:
    if env_name == "QuadrotorTracking2D-v0":
        env = make_quadrotor_tracking_2d_env()
    else:
        env = gym.make(env_name)

    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
        env.action_space.high, 1.0
    ):
        env = SymmetricActionWrapper(env)

    env.reset(seed=seed)
    return env


def _iter_array_leaves_with_paths(tree) -> List[Tuple[str, np.ndarray]]:
    stack = [("", tree)]
    leaves: List[Tuple[str, np.ndarray]] = []
    while stack:
        path, node = stack.pop()
        if isinstance(node, (dict, frozen_dict.FrozenDict)):
            for key, value in node.items():
                new_path = f"{path}/{key}" if path else str(key)
                stack.append((new_path, value))
        elif isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                new_path = f"{path}/{idx}" if path else str(idx)
                stack.append((new_path, value))
        else:
            try:
                arr = np.asarray(node)
            except Exception:
                continue
            if hasattr(arr, "ndim"):
                leaves.append((path, arr))
    return leaves


def _unwrap_params(tree):
    current = tree
    while isinstance(current, (dict, frozen_dict.FrozenDict)) and "params" in current:
        next_tree = current.get("params")
        if isinstance(next_tree, (dict, frozen_dict.FrozenDict)):
            current = next_tree
        else:
            break
    return current


def _infer_mlp_hidden_dims_from_params(params_subtree) -> Tuple[int, ...]:
    params_subtree = _unwrap_params(params_subtree)
    leaves = _iter_array_leaves_with_paths(params_subtree)
    kernel_candidates = [
        (p, a)
        for p, a in leaves
        if a.ndim == 2 and (p.lower().endswith("kernel") or "kernel" in p.lower())
    ]
    if not kernel_candidates:
        kernel_candidates = [(p, a) for p, a in leaves if a.ndim == 2]
    if len(kernel_candidates) < 1:
        return ()

    shapes = [(path, arr.shape) for path, arr in kernel_candidates]
    chains = []
    for path, (in_dim, out_dim) in shapes:
        chain = [(path, (in_dim, out_dim))]
        used = {path}
        current_out = out_dim
        improved = True
        while improved:
            improved = False
            candidates = [
                (p, s)
                for p, s in shapes
                if p not in used and s[0] == current_out
            ]
            if candidates:
                p_sel, s_sel = max(candidates, key=lambda x: x[1][1])
                chain.append((p_sel, s_sel))
                used.add(p_sel)
                current_out = s_sel[1]
                improved = True
        chains.append(chain)
    best_chain = max(chains, key=len)
    chain_shapes = [shape for _, shape in best_chain]
    if len(chain_shapes) <= 1:
        return ()
    return tuple(s[1] for s in chain_shapes[:-1])


def _find_config_path(run_dir: Path) -> Optional[Path]:
    candidates = [
        "config.json",
        "variant.json",
        "flags.json",
        "args.json",
        "params.json",
    ]
    for name in candidates:
        cand = run_dir / name
        if cand.exists():
            return cand
    return None


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_create_kwargs(cfg: dict) -> dict:
    sig = inspect.signature(SafeScoreMatchingLearner.create)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def _load_ssm_with_inferred_dims(
    ckpt_path: str,
    step: Optional[int],
    observation_space: gym.Space,
    action_space: gym.Space,
    seed: int,
) -> Tuple[SafeScoreMatchingLearner, dict]:
    resolved = Path(resolve_checkpoint(ckpt_path, step))
    data = resolved.read_bytes()
    state = serialization.msgpack_restore(data)
    if isinstance(state, SafeScoreMatchingLearner):
        return state, {"ckpt_resolved_path": str(resolved)}

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state)}")

    score_state = state.get("score_model", {})
    safety_state = state.get("safety_critic", {})
    critic_state = state.get("critic_1", {})
    lambda_state = state.get("lambda_net", {})

    actor_hidden_dims = _infer_mlp_hidden_dims_from_params(score_state.get("params", {}))
    safety_hidden_dims = _infer_mlp_hidden_dims_from_params(
        safety_state.get("params", {})
    )
    critic_hidden_dims = _infer_mlp_hidden_dims_from_params(
        critic_state.get("params", {})
    )
    lambda_hidden_dims = _infer_mlp_hidden_dims_from_params(
        lambda_state.get("params", {})
    )

    run_dir = resolved.parent.parent if resolved.parent.name == "checkpoints" else resolved.parent
    cfg = {}
    config_path = _find_config_path(run_dir)
    if config_path is not None:
        cfg = _load_config(config_path)

    create_kwargs = _filter_create_kwargs(cfg)
    if actor_hidden_dims:
        create_kwargs["actor_hidden_dims"] = actor_hidden_dims
    if safety_hidden_dims:
        create_kwargs["safety_hidden_dims"] = safety_hidden_dims
    if critic_hidden_dims:
        create_kwargs["critic_hidden_dims"] = critic_hidden_dims
    if lambda_hidden_dims:
        create_kwargs["lambda_hidden_dims"] = lambda_hidden_dims

    template = SafeScoreMatchingLearner.create(
        seed=seed,
        observation_space=observation_space,
        action_space=action_space,
        **create_kwargs,
    )
    agent = serialization.from_state_dict(template, state)
    meta = {"ckpt_resolved_path": str(resolved)}
    if config_path is not None:
        meta["config_path"] = str(config_path)
    return agent, meta


def get_ref_from_env(
    env: gym.Env, ref_mode: str, ref_waypoint_idx: int | None, seed: int
) -> np.ndarray:
    if ref_mode == "zero":
        return np.zeros(6, dtype=np.float32)

    if ref_mode == "waypoint":
        try:
            waypoints = env.unwrapped._waypoints  # noqa: SLF001
            if ref_waypoint_idx is None:
                ref_waypoint_idx = 0
            ref = np.asarray(waypoints[int(ref_waypoint_idx)], dtype=np.float32)
            return ref
        except Exception:
            ref_mode = "reset"

    obs, _ = env.reset(seed=seed)
    return np.asarray(obs[6:], dtype=np.float32)


def build_obs_batch(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    z_dot: float,
    ref: np.ndarray,
) -> np.ndarray:
    B = grid_x.size
    obs = np.zeros((B, 12), dtype=np.float32)
    obs[:, 0] = grid_x.reshape(-1)
    obs[:, 2] = grid_z.reshape(-1)
    obs[:, 3] = float(z_dot)
    obs[:, 6:] = ref[None, :]
    return obs


def sample_actions_gaussian(
    agent,
    obs_tiled: jnp.ndarray,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    dist = agent.actor.apply_fn({"params": agent.actor.params}, obs_tiled)
    actions = dist.sample(seed=rng)
    return actions


def sample_actions_ddpm(
    agent,
    obs_tiled: jnp.ndarray,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    actions, _ = ddpm_sampler(
        agent.score_model.apply_fn,
        agent.score_model.params,
        agent.T,
        rng,
        agent.act_dim,
        obs_tiled,
        agent.alphas,
        agent.alpha_hats,
        agent.betas,
        agent.ddpm_temperature,
        agent.clip_sampler,
    )
    return actions


def eval_qh_batch(
    algo: str,
    agent,
    obs_tiled: jnp.ndarray,
    actions_flat: jnp.ndarray,
    batch_chunk: int,
) -> jnp.ndarray:
    total = obs_tiled.shape[0]
    if batch_chunk <= 0 or batch_chunk >= total:
        return eval_qh_batch_chunk(algo, agent, obs_tiled, actions_flat)

    outputs = []
    for start in range(0, total, batch_chunk):
        end = min(start + batch_chunk, total)
        outputs.append(
            eval_qh_batch_chunk(algo, agent, obs_tiled[start:end], actions_flat[start:end])
        )
    return jnp.concatenate(outputs, axis=0)


def eval_qh_batch_chunk(
    algo: str,
    agent,
    obs_tiled: jnp.ndarray,
    actions_flat: jnp.ndarray,
) -> jnp.ndarray:
    if algo in {"rac", "ssm"}:
        qh = agent.safety_critic.apply_fn(
            {"params": agent.safety_critic.params},
            obs_tiled,
            actions_flat,
            training=False,
        )
        return qh

    qcs = agent.cost_critic.apply_fn(
        {"params": agent.cost_critic.params},
        obs_tiled,
        actions_flat,
    )
    return qcs.min(axis=0)


def compute_vh_for_slice(
    algo: str,
    agent,
    obs_batch: np.ndarray,
    K: int,
    rng: jax.random.PRNGKey,
    batch_chunk: int,
) -> np.ndarray:
    B = obs_batch.shape[0]
    obs_tiled = np.repeat(obs_batch, K, axis=0)
    obs_tiled = jnp.asarray(obs_tiled)

    if algo == "ssm":
        actions_flat = sample_actions_ddpm(agent, obs_tiled, rng)
    else:
        actions_flat = sample_actions_gaussian(agent, obs_tiled, rng)

    qh_flat = eval_qh_batch(algo, agent, obs_tiled, actions_flat, batch_chunk)
    qh_flat = qh_flat.reshape(B, K)
    vh_flat = qh_flat.min(axis=1)
    return np.asarray(vh_flat)


def plot_vh_grid(
    X: np.ndarray,
    Z: np.ndarray,
    vh_list: List[np.ndarray],
    z_dot_list: List[float],
    threshold: float,
    out_path: str,
) -> None:
    import numpy as _np
    import matplotlib.pyplot as _plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Circle
    import matplotlib.patheffects as pe

    cols = len(z_dot_list)

    # -------------------- 数据范围 & 轴比例（保证圆不变形） --------------------
    xlim = (float(_np.min(X)), float(_np.max(X)))
    ylim = (float(_np.min(Z)), float(_np.max(Z)))
    x_range = max(xlim[1] - xlim[0], 1e-6)
    y_range = max(ylim[1] - ylim[0], 1e-6)
    axes_ratio = x_range / y_range  # 每个子图“理想的”宽高比（宽/高）

    # -------------------- 颜色归一化：发散色图 + 0 居中 --------------------
    all_vh = _np.concatenate([v.reshape(-1) for v in vh_list])
    all_vh = all_vh[_np.isfinite(all_vh)]
    if all_vh.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        max_abs = float(_np.nanmax(_np.abs(all_vh)))
        max_abs = max(max_abs, 1e-6)
        vmin, vmax = -max_abs, max_abs

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    levels = _np.linspace(vmin, vmax, 61)

    # -------------------- 布局：按 axes_ratio 调整子图形状 + 独立 colorbar --------------------
    base_h = 4.8
    per_ax_w = base_h * axes_ratio
    cbar_w = 0.55
    fig_w = cols * per_ax_w + cbar_w + 0.6

    fig = _plt.figure(figsize=(fig_w, base_h))
    # 每个子图的 width_ratios 用 axes_ratio，确保行内子图宽高更符合数据比例
    gs = fig.add_gridspec(
        nrows=1,
        ncols=cols + 1,
        width_ratios=[axes_ratio] * cols + [0.10],
        wspace=0.22,
    )
    axes = [_plt.subplot(gs[0, i]) for i in range(cols)]
    cax = _plt.subplot(gs[0, cols])

    # 字体大小（可按喜好再调）
    title_fs = 22
    label_fs = 18
    tick_fs = 16

    # 阈值等值线颜色：用高对比“亮黄”，并加黑色描边，任何底色都清晰
    boundary_color = "orange"  # 金黄
    boundary_effects = [pe.Stroke(linewidth=2.6, foreground="black"), pe.Normal()]

    last_im = None
    for ax, vh, z_dot in zip(axes, vh_list, z_dot_list):
        # 主图：发散 cmap（正负明显）
        last_im = ax.contourf(
            X, Z, vh,
            levels=levels,
            cmap="RdBu_r",   # 你也可以改成 "seismic"（更“硬”）或 "coolwarm"
            norm=norm,
            extend="both",
        )

        # 阈值等值线（替代原白色虚线）：高对比颜色 + 描边
        cs = ax.contour(
            X, Z, vh,
            levels=[threshold],
            colors=[boundary_color],
            linestyles="--",
            linewidths=2.0,
        )
        for coll in cs.collections:
            coll.set_path_effects(boundary_effects)

        # 两条黑色实线：y(这里是 z)=0.5 和 1.5
        ax.axhline(0.5, color="black", linewidth=2.0)
        ax.axhline(1.5, color="black", linewidth=2.0)

        # 黑色虚线圆：圆心 (0,1)，半径 1（配合 equal aspect 保证不变形）
        circle = Circle(
            (0.0, 1.0),
            radius=1.0,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=2.0,
        )
        ax.add_patch(circle)

        # 关键：保证数据单位等比例 -> 圆不变椭圆；同时 axes box 会按数据范围变成长方形
        ax.set_aspect("equal", adjustable="box")

        # 标题：LaTeX 的 z 上加点
        ax.set_title(rf"$\dot{{z}} = {z_dot:g}$", fontsize=title_fs)

        # 轴标签与刻度
        ax.set_xlabel("x", fontsize=label_fs)
        ax.set_ylabel("z", fontsize=label_fs)
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_yticks([0.5, 1.0, 1.5])
        ax.tick_params(axis="both", labelsize=tick_fs)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # colorbar 放到专用 cax（不会盖在子图上）
    cb = fig.colorbar(last_im, cax=cax)
    cb.ax.tick_params(labelsize=tick_fs)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    _plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V_h(s) region on (x,z) grid.")
    parser.add_argument("--algo", choices=["rac", "ssm", "sac_lag"], required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--env_name", default="QuadrotorTracking2D-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_n", type=int, default=101)
    parser.add_argument("--x_min", type=float, default=-1.5)
    parser.add_argument("--x_max", type=float, default=1.5)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--z_dot_list", type=str, default="-1,0,1")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--out", type=str, default="vh_region.png")
    parser.add_argument("--save_npz", type=str, default="")
    parser.add_argument("--ref_mode", choices=["reset", "waypoint", "zero"], default="reset")
    parser.add_argument("--ref_waypoint_idx", type=int, default=None)
    parser.add_argument("--batch_chunk", type=int, default=0)
    args = parser.parse_args()

    env = create_env(args.env_name, args.seed)
    obs_space = env.observation_space
    act_space = env.action_space

    if args.algo == "rac":
        agent, _, _ = load_rac(
            args.ckpt_path,
            step=args.step,
            observation_space=obs_space,
            action_space=act_space,
            seed=args.seed,
            deterministic=False,
        )
        threshold = 0.0
    elif args.algo == "ssm":
        agent, _ = _load_ssm_with_inferred_dims(
            args.ckpt_path, args.step, obs_space, act_space, args.seed
        )
        threshold = 0.0
    else:
        agent, _, _ = load_sac_lag(
            args.ckpt_path,
            step=args.step,
            observation_space=obs_space,
            action_space=act_space,
            seed=args.seed,
            deterministic=False,
        )
        threshold = float(getattr(agent, "cost_limit", 0.0))

    ref = get_ref_from_env(env, args.ref_mode, args.ref_waypoint_idx, args.seed)

    z_dot_list = parse_float_list(args.z_dot_list)
    x = np.linspace(args.x_min, args.x_max, args.grid_n, dtype=np.float32)
    z = np.linspace(args.z_min, args.z_max, args.grid_n, dtype=np.float32)
    X, Z = np.meshgrid(x, z, indexing="xy")

    B = args.grid_n * args.grid_n
    print(f"Env obs_space={obs_space}, act_space={act_space}")
    print(f"ref_mode={args.ref_mode}, ref={ref}")
    print(f"B={B}, K={args.K}, batch_chunk={args.batch_chunk}")
    print(f"Output: {args.out}")

    vh_list = []
    for z_dot in z_dot_list:
        obs_batch = build_obs_batch(X, Z, z_dot, ref)
        rng = jax.random.PRNGKey(args.seed)
        vh_flat = compute_vh_for_slice(
            args.algo,
            agent,
            obs_batch,
            args.K,
            rng,
            args.batch_chunk,
        )
        vh = vh_flat.reshape(args.grid_n, args.grid_n)
        vh_list.append(vh)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_vh_grid(X, Z, vh_list, z_dot_list, threshold, args.out)

    if args.save_npz:
        np.savez(
            args.save_npz,
            X=X,
            Z=Z,
            z_dot_list=np.array(z_dot_list, dtype=np.float32),
            Vh_list=np.stack(vh_list, axis=0),
            threshold=float(threshold),
            algo=args.algo,
            ckpt_path=args.ckpt_path,
            step=args.step if args.step is not None else -1,
            K=args.K,
            ref_mode=args.ref_mode,
            ref=ref,
        )
        print(f"Saved npz to {args.save_npz}")


if __name__ == "__main__":
    main()
