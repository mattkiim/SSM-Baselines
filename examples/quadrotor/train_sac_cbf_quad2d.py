#!/usr/bin/env python
from __future__ import annotations

# NOTE: visualize_policy_trajectory/load_agent currently does not support "sac_cbf".
# The repository already includes jaxrl5/tools/load_sac_cbf.py; it still needs to be
# registered in jaxrl5/tools/load_agent.py (Prompt 6.5).
#
# Quick smoke run example:
# python examples/quadrotor/train_sac_cbf_quad2d.py \
#   --env_name QuadrotorTracking2D-v0 \
#   --seed 0 \
#   --max_steps 2000 \
#   --start_training 200 \
#   --eval_interval 500 \
#   --eval_episodes 2 \
#   --save_interval 500 \
#   --batch_size 256 \
#   --utd_ratio 1 \
#   --wandb False \
#   --config examples/quadrotor/configs/sac_cbf_quad2d_config.py

import datetime
import glob
import json
import os
from typing import Callable, Dict, Optional

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents.sac.sac_cbf_learner import SACCbfLearner
from jaxrl5.data import ReplayBuffer
from jaxrl5.envs import make_env
from jaxrl5.utils import append_history
from jaxrl5.wrappers import WANDBVideo
from jaxrl5.wrappers.termination_penalty import TerminationPenaltyWrapper
from jaxrl5.envs.registration import ensure_custom_envs_registered

FLAGS = flags.FLAGS

EVAL_STARTS = [
    (1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 0.53),
    (0.0, 1.47),
]

_DEBUG_ENV_PRINTED = False

flags.DEFINE_string("project_name", "jaxrl5_quad2d_sac_cbf", "wandb project name.")
flags.DEFINE_string("run_name", "", "wandb run name.")
flags.DEFINE_string("env_name", "QuadrotorTracking2D-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("eval_episodes", 4, "Evaluation episodes.")
flags.DEFINE_integer("log_interval", 400, "Logging interval (steps).")
flags.DEFINE_integer("eval_interval", 5000, "Evaluation interval (steps).")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 200_000, "Number of training steps.")
flags.DEFINE_integer("start_training", 10_000, "Number of steps before learning starts.")
flags.DEFINE_boolean("wandb", False, "Enable wandb logging.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_integer("eval_seed_offset", 12345, "Offset for evaluation environment seed.")
flags.DEFINE_boolean("save_video", False, "Upload videos during evaluation (wandb only).")
flags.DEFINE_integer("save_interval", 5000, "Checkpoint save interval.")
flags.DEFINE_enum("mode", "training", ["training", "testing"], "Run mode.")
flags.DEFINE_string("load_dir", "", "Directory containing checkpoints for testing.")
flags.DEFINE_integer("load_step", None, "Checkpoint step to load for testing.")
flags.DEFINE_string(
    "results_root",
    "results/QuadrotorTracking2D-v0/jaxrl5_quad2d_sac_cbf",
    "Root directory for experiment outputs.",
)
config_flags.DEFINE_config_file(
    "config",
    "examples/quadrotor/configs/sac_cbf_quad2d_config.py",
    "Path to SAC-CBF hyperparameter configuration.",
    lock_config=False,
)


def _maybe_init_wandb():
    if not FLAGS.wandb:
        return

    try:
        import wandb
    except ImportError:
        print("wandb is not installed; disabling wandb logging.")
        FLAGS.wandb = False
        return

    run_name = FLAGS.run_name or None
    try:
        wandb.init(
            project=FLAGS.project_name,
            name=run_name,
            tags=[FLAGS.run_name] if FLAGS.run_name else None,
        )
    except Exception as exc:
        print(f"wandb init failed ({exc}); disabling wandb logging.")
        FLAGS.wandb = False
        return

    wandb.config.update(FLAGS)


def _make_env(env_name: str, seed: int, allow_video: bool = True):
    global _DEBUG_ENV_PRINTED

    ensure_custom_envs_registered()

    env = make_env(env_name, seed=seed)

    # if (not _DEBUG_ENV_PRINTED) and (env_name == "QuadrotorTracking2D-v0"):
    #     print("[debug] env type:", type(env))
    #     print("[debug] wrapped action_space low/high:", env.action_space.low, env.action_space.high)

    #     try:
    #         print(
    #             "[debug] unwrapped action_space low/high:",
    #             env.unwrapped.action_space.low,
    #             env.unwrapped.action_space.high,
    #         )
    #     except Exception as exc:
    #         print("[debug] cannot access unwrapped action_space:", exc)

    #     obs, _ = env.reset(seed=seed)
    #     out = env.step(env.action_space.sample())
    #     print("[debug] step return len:", len(out))
    #     env.reset(seed=seed)

    #     _DEBUG_ENV_PRINTED = True

    if allow_video and FLAGS.wandb and FLAGS.save_video:
        env = WANDBVideo(env)
    return env


def _make_eval_policy(agent: SACCbfLearner) -> Callable[[np.ndarray], np.ndarray]:
    eval_agent = agent

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(np.asarray(obs, dtype=np.float32))
        return np.asarray(action, dtype=np.float32)

    return policy


def evaluate_four_starts(env, policy_fn, seed: int = 0):
    """Replicate paper: 4 runs with static init; report mean/std of return and violation rate."""
    returns = []
    costs = []
    ep_lens = []
    viol_rates = []

    for i, (x0, z0) in enumerate(EVAL_STARTS):
        obs, _ = env.reset(
            seed=seed + i,
            options={
                "init_x": x0,
                "init_z": z0,
                "init_vx": 0.0,
                "init_vz": 0.0,
                "init_theta": 0.0,
                "init_omega": 0.0,
            },
        )

        ep_ret = 0.0
        ep_cost = 0.0
        steps = 0
        viol_sum = 0.0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            act = policy_fn(obs)
            obs, r, c, terminated, truncated, info = env.step(act)
            ep_ret += float(r)
            ep_cost += float(c)
            viol_sum += float(c)
            steps += 1

        returns.append(ep_ret)
        costs.append(ep_cost)
        ep_lens.append(steps)
        viol_rates.append(viol_sum / 360)

    returns = np.asarray(returns, np.float32)
    costs = np.asarray(costs, np.float32)
    ep_lens = np.asarray(ep_lens, np.float32)
    viol_rates = np.asarray(viol_rates, np.float32)

    return {
        "eval/return_mean": float(returns.mean()),
        "eval/return_std": float(returns.std()),
        "eval/cost_mean": float(costs.mean()),
        "eval/cost_std": float(costs.std()),
        "eval/ep_len_mean": float(ep_lens.mean()),
        "eval/ep_len_std": float(ep_lens.std()),
        "eval/violation_rate_mean": float(viol_rates.mean()),
        "eval/violation_rate_std": float(viol_rates.std()),
    }


def _save_config(run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    cfg = {
        "flags": {k: v for k, v in FLAGS.flag_values_dict().items() if k != "config"},
        "config": FLAGS.config.to_dict(),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _format_run_dir(seed: int) -> str:
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    return os.path.join(FLAGS.results_root, f"{date_str}_seed{seed:04d}")


def _save_checkpoint(agent: SACCbfLearner, ckpt_dir: str, step: int) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{step}.msgpack")
    agent.save(path)
    return path


def _load_checkpoint(path: str) -> SACCbfLearner:
    return SACCbfLearner.load(path)


def _find_checkpoint(load_dir: str, load_step: Optional[int]) -> str:
    ckpt_dir = os.path.join(load_dir, "checkpoints")
    if load_step is not None:
        path = os.path.join(ckpt_dir, f"ckpt_{load_step}.msgpack")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = glob.glob(os.path.join(ckpt_dir, "ckpt_*.msgpack"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return sorted(candidates)[-1]


def _run_evaluation(agent: SACCbfLearner, eval_env, step: int, experiment_name: str):
    eval_policy = _make_eval_policy(agent)

    metrics_all = evaluate_four_starts(
        eval_env,
        eval_policy,
        seed=FLAGS.seed + FLAGS.eval_seed_offset,
    )

    if FLAGS.wandb:
        import wandb

        wandb.log(metrics_all, step=step)
    else:
        print(
            f"[step {step}] return={metrics_all['eval/return_mean']:.2f} "
            f"cost={metrics_all['eval/cost_mean']:.2f} "
            f"viol_rate={metrics_all['eval/violation_rate_mean']:.3f} "
            f"len={metrics_all['eval/ep_len_mean']:.1f}"
        )

    append_history(
        step,
        FLAGS.env_name,
        experiment_name,
        FLAGS.seed,
        {
            "eval/return_mean": metrics_all["eval/return_mean"],
            "eval/return_std": metrics_all["eval/return_std"],
            "eval/cost_mean": metrics_all["eval/cost_mean"],
            "eval/cost_std": metrics_all.get("eval/cost_std", float("nan")),
            "eval/violation_rate_mean": metrics_all["eval/violation_rate_mean"],
            "eval/violation_rate_std": metrics_all.get("eval/violation_rate_std", float("nan")),
            "eval/ep_len_mean": metrics_all["eval/ep_len_mean"],
        },
    )
    return metrics_all


def _training_loop(run_dir: str) -> None:
    _maybe_init_wandb()

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    _save_config(run_dir)

    train_env = _make_env(FLAGS.env_name, seed=FLAGS.seed, allow_video=True)
    eval_env = _make_env(
        FLAGS.env_name, seed=FLAGS.seed + FLAGS.eval_seed_offset, allow_video=True
    )

    obs_shape = train_env.observation_space.shape
    act_shape = train_env.action_space.shape

    train_env = TerminationPenaltyWrapper(
        train_env,
        penalty=0.0,
        apply_on_truncated=False,
    )

    replay_buffer = ReplayBuffer(obs_shape, act_shape, capacity=FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent: SACCbfLearner = globals()[model_cls].create(
        FLAGS.seed,
        train_env.observation_space,
        train_env.action_space,
        **kwargs,
    )

    observation, _ = train_env.reset(seed=FLAGS.seed)
    episode_return, episode_cost, episode_length = 0.0, 0.0, 0
    experiment_name = FLAGS.run_name or FLAGS.project_name

    for step in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if step < FLAGS.start_training:
            action = np.asarray(train_env.action_space.sample(), dtype=np.float32)
        else:
            action, agent = agent.sample_actions(np.asarray(observation, dtype=np.float32))
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, train_env.action_space.low, train_env.action_space.high)

        next_obs, reward, cost, terminated, truncated, info = train_env.step(action)
        done = bool(terminated or truncated)

        replay_buffer.insert(
            observation,
            action,
            float(reward),
            float(cost),
            next_obs,
            terminated,
            truncated,
        )

        episode_return += float(reward)
        episode_cost += float(cost)
        episode_length += 1
        observation = next_obs

        if step >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info: Dict[str, float] = {}
            for _ in range(FLAGS.utd_ratio):
                agent, update_info = agent.update(batch)

            if FLAGS.wandb and step % FLAGS.log_interval == 0:
                import wandb

                wandb.log({f"training/{k}": float(v) for k, v in update_info.items()}, step=step)

        if done:
            if FLAGS.wandb:
                import wandb

                wandb.log(
                    {
                        "training/return": episode_return,
                        "training/cost": episode_cost,
                        "training/length": episode_length,
                    },
                    step=step,
                )

            observation, _ = train_env.reset()
            episode_return, episode_cost, episode_length = 0.0, 0.0, 0

        if step % FLAGS.eval_interval == 0:
            _run_evaluation(agent, eval_env, step=step, experiment_name=experiment_name)

        if step % FLAGS.save_interval == 0:
            _save_checkpoint(agent, ckpt_dir, step)

    train_env.close()
    eval_env.close()


def _testing_loop() -> None:
    if not FLAGS.load_dir:
        raise ValueError("--load_dir is required in testing mode.")

    ckpt_path = _find_checkpoint(FLAGS.load_dir, FLAGS.load_step)
    agent = _load_checkpoint(ckpt_path)

    eval_env = _make_env(
        FLAGS.env_name, seed=FLAGS.seed + FLAGS.eval_seed_offset, allow_video=False
    )
    experiment_name = FLAGS.run_name or FLAGS.project_name
    metrics = _run_evaluation(
        agent, eval_env, step=FLAGS.load_step or 0, experiment_name=experiment_name
    )

    result_path = os.path.join(
        FLAGS.load_dir, f"test_results_step{FLAGS.load_step or 'latest'}.json"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    eval_env.close()
    print(f"Saved evaluation results to {result_path}")


def main(_):
    if FLAGS.mode == "testing":
        _testing_loop()
    else:
        run_dir = _format_run_dir(FLAGS.seed)
        _training_loop(run_dir)


if __name__ == "__main__":
    app.run(main)
