"""Reusable rollout utilities for QuadrotorTracking2D-v0.

Run a single algorithm:
python -m examples.quadrotor.rollout_utils --algo sac_lag --ckpt_path /ABS/PATH/TO/run_dir --step 2000000 --horizon 200

Run all algorithms (skips missing checkpoints):
python -m examples.quadrotor.rollout_utils \
  --ckpt_ssm /ABS/PATH/TO/ssm_run_dir \
  --ckpt_rac /ABS/PATH/TO/rac_run_dir \
  --ckpt_cal /ABS/PATH/TO/cal_run_dir \
  --ckpt_sac_cbf /ABS/PATH/TO/sac_cbf_run_dir \
  --ckpt_sac_lag /ABS/PATH/TO/sac_lag_run_dir \
  --step 2000000 --horizon 200
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from jaxrl5.envs import make_env
from jaxrl5.tools.load_agent import load_agent
from jaxrl5.wrappers import AddCostFromInfo
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


@dataclass
class RolloutResult:
    algo: str
    ckpt_path: str
    step: Optional[int]
    seed: int
    deterministic: bool
    horizon: int
    init_options: Dict[str, Any]
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: List[Dict[str, Any]]
    episode_return: float
    episode_cost: float
    violation_rate: float
    safe_rate: float
    terminated_any: bool
    truncated_any: bool


def _default_init_options() -> Dict[str, Any]:
    return {
        "init_x": 0.0,
        "init_vx": 0.0,
        "init_z": 1.0,
        "init_vz": 0.0,
        "init_theta": 0.0,
        "init_omega": 0.0,
        "init_waypoint_idx": 0,
    }


def make_quad2d_env(env_name: str = "QuadrotorTracking2D-v0", seed: int = 0) -> gym.Env:
    """Create a QuadrotorTracking2D env with cost-aware 6-tuple step output."""
    try:
        env = make_env(env_name, seed=seed)
    except Exception:
        env = gym.make(env_name)
        env = AddCostFromInfo(env)
        env = SymmetricActionWrapper(env)
        env.reset(seed=seed)

    step_out = env.step(env.action_space.sample())
    if len(step_out) != 6:
        raise RuntimeError("cost wrapper missing: env.step must return 6 values")
    env.reset(seed=seed)
    return env


def load_policy(
    algo: str,
    ckpt_path: str,
    step: Optional[int],
    env: gym.Env,
    seed: int = 0,
    deterministic: bool = True,
) -> Tuple[object, callable, Dict[str, Any]]:
    """Load an agent + policy_fn via jaxrl5.tools.load_agent."""
    algo_norm = algo.strip().lower().replace("-", "_")
    return load_agent(
        algo_norm,
        ckpt_path,
        step=step,
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
        deterministic=deterministic,
    )


def _validate_action(action: np.ndarray, act_dim: int, *, context: str) -> np.ndarray:
    action_np = np.asarray(action, dtype=np.float32)
    if action_np.ndim == 2 and action_np.shape[0] == 1:
        action_np = action_np[0]
    if action_np.shape != (act_dim,):
        raise ValueError(f"{context}: action shape {action_np.shape} != ({act_dim},)")
    if not np.all(np.isfinite(action_np)):
        preview = action_np[: min(5, action_np.size)]
        raise ValueError(f"{context}: non-finite action {preview}")
    return action_np


def rollout_once(
    algo: str,
    ckpt_path: str,
    step: Optional[int] = None,
    env_name: str = "QuadrotorTracking2D-v0",
    seed: int = 0,
    deterministic: bool = True,
    horizon: int = 360,
    init_options: Optional[Dict[str, Any]] = None,
    render: bool = False,
) -> RolloutResult:
    """Roll out a single episode and return structured trajectory data."""
    env = make_quad2d_env(env_name=env_name, seed=seed)
    init_opts = dict(_default_init_options())
    if init_options is not None:
        init_opts.update(init_options)

    agent, policy_fn, meta = load_policy(
        algo=algo,
        ckpt_path=ckpt_path,
        step=step,
        env=env,
        seed=seed,
        deterministic=deterministic,
    )

    obs, _ = env.reset(seed=seed, options=init_opts)
    obs_list = [np.asarray(obs, dtype=np.float32)]
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    costs: List[float] = []
    terminated_flags: List[bool] = []
    truncated_flags: List[bool] = []
    infos: List[Dict[str, Any]] = []

    act_dim = env.action_space.shape[-1]

    for _ in range(horizon):
        action = policy_fn(obs)
        action = _validate_action(
            action,
            act_dim,
            context=f"algo={algo} ckpt={ckpt_path} step={step} obs={np.asarray(obs)[:5]}",
        )
        action = np.clip(action, env.action_space.low, env.action_space.high)

        step_out = env.step(action)
        if len(step_out) != 6:
            raise RuntimeError("cost wrapper missing: env.step must return 6 values")
        next_obs, reward, cost, terminated, truncated, info = step_out

        actions.append(action)
        rewards.append(float(reward))
        costs.append(float(cost))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        infos.append(dict(info))

        obs = next_obs
        obs_list.append(np.asarray(obs, dtype=np.float32))

        if render:
            try:
                env.render()
            except Exception:
                pass

        if terminated or truncated:
            break

    env.close()

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    costs_arr = np.asarray(costs, dtype=np.float32)
    terminated_arr = np.asarray(terminated_flags, dtype=bool)
    truncated_arr = np.asarray(truncated_flags, dtype=bool)

    episode_return = float(rewards_arr.sum())
    episode_cost = float(costs_arr.sum())
    violation_rate = float(np.mean(costs_arr > 0.0)) if costs_arr.size else 0.0
    safe_rate = float(np.mean(costs_arr == 0.0)) if costs_arr.size else 0.0

    return RolloutResult(
        algo=str(meta.get("algo", algo)),
        ckpt_path=ckpt_path,
        step=meta.get("step", step),
        seed=seed,
        deterministic=deterministic,
        horizon=horizon,
        init_options=init_opts,
        observations=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=rewards_arr,
        costs=costs_arr,
        terminated=terminated_arr,
        truncated=truncated_arr,
        infos=infos,
        episode_return=episode_return,
        episode_cost=episode_cost,
        violation_rate=violation_rate,
        safe_rate=safe_rate,
        terminated_any=bool(terminated_arr.any()) if terminated_arr.size else False,
        truncated_any=bool(truncated_arr.any()) if truncated_arr.size else False,
    )


def rollout_many(
    algo: str,
    ckpt_path: str,
    step: Optional[int] = None,
    env_name: str = "QuadrotorTracking2D-v0",
    seed: int = 0,
    deterministic: bool = True,
    horizon: int = 360,
    init_options: Optional[Dict[str, Any]] = None,
    num_rollouts: int = 10,
) -> Dict[str, float]:
    """Roll out multiple episodes and return aggregated statistics."""
    results = []
    for i in range(num_rollouts):
        res = rollout_once(
            algo=algo,
            ckpt_path=ckpt_path,
            step=step,
            env_name=env_name,
            seed=seed + i,
            deterministic=deterministic,
            horizon=horizon,
            init_options=init_options,
        )
        results.append(res)

    returns = np.asarray([r.episode_return for r in results], dtype=np.float32)
    costs = np.asarray([r.episode_cost for r in results], dtype=np.float32)
    violation_rates = np.asarray([r.violation_rate for r in results], dtype=np.float32)
    success_mask = np.asarray(
        [not r.terminated_any and np.all(r.costs == 0.0) for r in results], dtype=np.float32
    )

    return {
        "returns_mean": float(returns.mean()) if returns.size else 0.0,
        "returns_std": float(returns.std()) if returns.size else 0.0,
        "costs_mean": float(costs.mean()) if costs.size else 0.0,
        "costs_std": float(costs.std()) if costs.size else 0.0,
        "violation_rate_mean": float(violation_rates.mean()) if violation_rates.size else 0.0,
        "violation_rate_std": float(violation_rates.std()) if violation_rates.size else 0.0,
        "success_rate": float(success_mask.mean()) if success_mask.size else 0.0,
    }


def _run_single_algo(args: argparse.Namespace) -> bool:
    try:
        result = rollout_once(
            algo=args.algo,
            ckpt_path=args.ckpt_path,
            step=args.step,
            horizon=args.horizon,
            deterministic=args.deterministic,
            seed=args.seed,
        )
        print(
            " "
            .join(
                [
                    f"algo={result.algo}",
                    f"step={result.step}",
                    f"return={result.episode_return:.2f}",
                    f"cost={result.episode_cost:.2f}",
                    f"viol_rate={result.violation_rate:.3f}",
                    f"terminated={result.terminated_any}",
                    f"truncated={result.truncated_any}",
                ]
            )
        )
        return True
    except Exception as exc:
        print(f"algo={args.algo} FAILED: {exc}")
        return False


def _run_all_algos(args: argparse.Namespace) -> bool:
    algo_to_ckpt = {
        "ssm": args.ckpt_ssm,
        "rac": args.ckpt_rac,
        "cal": args.ckpt_cal,
        "sac_cbf": args.ckpt_sac_cbf,
        "sac_lag": args.ckpt_sac_lag,
    }

    all_ok = True
    for algo, ckpt in algo_to_ckpt.items():
        if not ckpt:
            print(f"algo={algo} skipped (missing ckpt path)")
            continue
        try:
            result = rollout_once(
                algo=algo,
                ckpt_path=ckpt,
                step=args.step,
                horizon=args.horizon,
                deterministic=args.deterministic,
                seed=args.seed,
            )
            print(
                " "
                .join(
                    [
                        f"algo={result.algo}",
                        f"step={result.step}",
                        f"return={result.episode_return:.2f}",
                        f"cost={result.episode_cost:.2f}",
                        f"viol_rate={result.violation_rate:.3f}",
                        f"terminated={result.terminated_any}",
                        f"truncated={result.truncated_any}",
                    ]
                )
            )
        except Exception as exc:
            print(f"algo={algo} FAILED: {exc}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout utility for QuadrotorTracking2D-v0")
    parser.add_argument("--algo", default=None, help="Algorithm name (optional)")
    parser.add_argument("--ckpt_path", default=None, help="Checkpoint path (single algo)")
    parser.add_argument("--ckpt_ssm", default=None)
    parser.add_argument("--ckpt_rac", default=None)
    parser.add_argument("--ckpt_cal", default=None)
    parser.add_argument("--ckpt_sac_cbf", default=None)
    parser.add_argument("--ckpt_sac_lag", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.algo:
        if not args.ckpt_path:
            raise SystemExit("--ckpt_path is required when using --algo")
        ok = _run_single_algo(args)
    else:
        ok = _run_all_algos(args)

    print("PASSED" if ok else "FAILED")
