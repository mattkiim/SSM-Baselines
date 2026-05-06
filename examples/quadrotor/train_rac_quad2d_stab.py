#!/usr/bin/env python
from __future__ import annotations

import datetime
import glob
import inspect
import json
import os
import sys
from typing import Callable, Dict, Optional

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.quadrotor.quad2d_stab_eval import (
    evaluate_initial_states,
    make_fixed_vel_small_states,
    sample_lowz_hrej_initial_states_stab,
)
from jaxrl5.agents.rac.rac_learner import RACLearner
from jaxrl5.data import ReplayBuffer
from jaxrl5.envs import make_env
from jaxrl5.envs.registration import ensure_custom_envs_registered
from jaxrl5.utils import append_history
from jaxrl5.wrappers import WANDBVideo
from jaxrl5.wrappers.termination_penalty import TerminationPenaltyWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "jaxrl5_quad2d_stab_rac", "wandb project name.")
flags.DEFINE_string("run_name", "", "wandb run name.")
flags.DEFINE_string("env_name", "QuadrotorStabilization2D-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("log_interval", 400, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Evaluation interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 200_000, "Number of training steps.")
flags.DEFINE_integer("start_training", 10_000, "Number of steps before learning starts.")
flags.DEFINE_boolean("wandb", False, "Enable wandb logging.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_integer("eval_seed_offset", 12345, "Offset for evaluation sampling.")
flags.DEFINE_integer("eval_rollouts", 200, "Rollouts per eval suite.")
flags.DEFINE_boolean("save_video", False, "Upload videos during evaluation when wandb is enabled.")
flags.DEFINE_integer("save_interval", 5000, "Checkpoint save interval.")
flags.DEFINE_enum("mode", "training", ["training", "testing"], "Run mode.")
flags.DEFINE_string("load_dir", "", "Directory containing checkpoints for testing.")
flags.DEFINE_integer("load_step", None, "Checkpoint step to load for testing.")
flags.DEFINE_string(
    "results_root",
    "results/QuadrotorStabilization2D-v0/jaxrl5_quad2d_stab_rac",
    "Root directory for experiment outputs.",
)

flags.DEFINE_string("layout_name", "corridor_v2", "Quad2D stabilization layout.")
flags.DEFINE_string("reset_mode", "simple_under3", "Training reset mode.")
flags.DEFINE_string("obs_feature_mode", "state", "Observation feature mode.")
flags.DEFINE_float("q_x", 2.0, "Position x reward weight for warm_anchor_fork20_mirror_qx2.")
flags.DEFINE_float("q_z", 10.0, "Position z reward weight.")
flags.DEFINE_float("simple_bar", 0.24, "simple_under3 bar_under reset fraction.")
flags.DEFINE_float("simple_left", 0.08, "simple_under3 left_block_under reset fraction.")
flags.DEFINE_float("simple_right", 0.08, "simple_under3 right_block_under reset fraction.")
flags.DEFINE_float("simple_gap", 0.08, "simple_under3 gap_entry reset fraction.")
flags.DEFINE_float("simple_low", 0.20, "simple_under3 low_uniform reset fraction.")
flags.DEFINE_float("simple_near", 0.12, "simple_under3 near_goal reset fraction.")
flags.DEFINE_float("simple_fork", 0.20, "simple_under3 fork reset fraction.")

config_flags.DEFINE_config_file(
    "config",
    "examples/quadrotor/configs/rac_quad2d_stab_config.py",
    "Path to RAC hyperparameter configuration.",
    lock_config=False,
)


def _env_kwargs() -> Dict:
    return {
        "layout_name": FLAGS.layout_name,
        "reset_mode": FLAGS.reset_mode,
        "obs_feature_mode": FLAGS.obs_feature_mode,
        "q_x": FLAGS.q_x,
        "q_z": FLAGS.q_z,
        "reset_simple_bar_frac": FLAGS.simple_bar,
        "reset_simple_left_block_frac": FLAGS.simple_left,
        "reset_simple_right_block_frac": FLAGS.simple_right,
        "reset_simple_gap_frac": FLAGS.simple_gap,
        "reset_simple_low_uniform_frac": FLAGS.simple_low,
        "reset_simple_near_frac": FLAGS.simple_near,
        "reset_simple_side_frac": 0.0,
        "reset_simple_fork_frac": FLAGS.simple_fork,
    }


def _maybe_init_wandb() -> None:
    if not FLAGS.wandb:
        return
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; disabling wandb logging.")
        FLAGS.wandb = False
        return
    wandb.init(project=FLAGS.project_name, name=FLAGS.run_name or None)
    wandb.config.update(FLAGS)


def _make_env(seed: int, allow_video: bool = True):
    ensure_custom_envs_registered()
    env = make_env(FLAGS.env_name, seed=seed, **_env_kwargs())
    if allow_video and FLAGS.wandb and FLAGS.save_video:
        env = WANDBVideo(env)
    return env


def _make_eval_policy(agent: RACLearner) -> Callable[[np.ndarray], np.ndarray]:
    eval_agent = agent

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(np.asarray(obs, dtype=np.float32))
        return np.asarray(action, dtype=np.float32)

    return policy


def _save_config(run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    cfg = {
        "flags": {k: v for k, v in FLAGS.flag_values_dict().items() if k != "config"},
        "config": FLAGS.config.to_dict(),
        "env_kwargs": _env_kwargs(),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _format_run_dir(seed: int) -> str:
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    suffix = FLAGS.run_name or f"seed{seed:04d}"
    return os.path.join(FLAGS.results_root, f"{date_str}_{suffix}")


def _save_checkpoint(agent: RACLearner, ckpt_dir: str, step: int) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{step}.msgpack")
    agent.save(path)
    return path


def _load_checkpoint(path: str) -> RACLearner:
    return RACLearner.load(path)


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


def _filter_create_kwargs(cfg: Dict) -> Dict:
    sig = inspect.signature(RACLearner.create)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def _run_evaluation(agent: RACLearner, eval_env, step: int, experiment_name: str) -> Dict[str, float]:
    policy = _make_eval_policy(agent)
    lowz_states = sample_lowz_hrej_initial_states_stab(
        eval_env,
        n=FLAGS.eval_rollouts,
        seed=FLAGS.seed + FLAGS.eval_seed_offset,
        z_high=0.50,
    )
    fixed_states = make_fixed_vel_small_states(
        n=FLAGS.eval_rollouts,
        seed=FLAGS.seed + FLAGS.eval_seed_offset + 1,
    )
    lowz = evaluate_initial_states(eval_env, policy, lowz_states)
    fixed = evaluate_initial_states(eval_env, policy, fixed_states)
    metrics = {f"eval_lowz050_hrej/{k.removeprefix('eval/')}": v for k, v in lowz.items()}
    metrics.update({f"eval_fixed_vel_small/{k.removeprefix('eval/')}": v for k, v in fixed.items()})

    if FLAGS.wandb:
        import wandb

        wandb.log(metrics, step=step)
    else:
        print(
            f"[step {step}] lowz_ret={metrics['eval_lowz050_hrej/return_mean']:.2f} "
            f"lowz_cost={metrics['eval_lowz050_hrej/cost_mean']:.2f} "
            f"fixed_ret={metrics['eval_fixed_vel_small/return_mean']:.2f} "
            f"fixed_cost={metrics['eval_fixed_vel_small/cost_mean']:.2f}"
        )

    append_history(
        step,
        FLAGS.env_name,
        experiment_name,
        FLAGS.seed,
        {
            "eval/return_mean": metrics["eval_lowz050_hrej/return_mean"],
            "eval/cost_mean": metrics["eval_lowz050_hrej/cost_mean"],
            "eval/violation_rate_mean": metrics["eval_lowz050_hrej/violation_rate_mean"],
            "eval/fixed_return_mean": metrics["eval_fixed_vel_small/return_mean"],
            "eval/fixed_cost_mean": metrics["eval_fixed_vel_small/cost_mean"],
        },
    )
    return metrics


def _training_loop(run_dir: str) -> None:
    _maybe_init_wandb()
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    _save_config(run_dir)

    train_env = _make_env(FLAGS.seed, allow_video=True)
    eval_env = _make_env(FLAGS.seed + FLAGS.eval_seed_offset, allow_video=True)
    train_env = TerminationPenaltyWrapper(train_env, penalty=0.0, apply_on_truncated=False)

    replay_buffer = ReplayBuffer(train_env.observation_space.shape, train_env.action_space.shape, capacity=FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    kwargs = _filter_create_kwargs(dict(FLAGS.config))
    model_cls = kwargs.pop("model_cls", "RACLearner")
    agent: RACLearner = globals()[model_cls].create(
        FLAGS.seed,
        train_env.observation_space,
        train_env.action_space,
        **kwargs,
    )

    observation, _ = train_env.reset(seed=FLAGS.seed)
    episode_return, episode_cost, episode_length = 0.0, 0.0, 0
    experiment_name = FLAGS.run_name or FLAGS.project_name
    update_info: Dict[str, float] = {}

    for step in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        if step < FLAGS.start_training:
            action = np.asarray(train_env.action_space.sample(), dtype=np.float32)
        else:
            action, agent = agent.sample_actions(np.asarray(observation, dtype=np.float32))
            action = np.clip(np.asarray(action, dtype=np.float32), train_env.action_space.low, train_env.action_space.high)

        next_obs, reward, cost, terminated, truncated, _info = train_env.step(action)
        replay_buffer.insert(observation, action, float(reward), float(cost), next_obs, terminated, truncated)
        episode_return += float(reward)
        episode_cost += float(cost)
        episode_length += 1
        observation = next_obs

        if step >= FLAGS.start_training:
            for _ in range(FLAGS.utd_ratio):
                batch = replay_buffer.sample(FLAGS.batch_size)
                agent, update_info = agent.update(batch)
            if FLAGS.wandb and step % FLAGS.log_interval == 0:
                import wandb

                wandb.log({f"training/{k}": float(v) for k, v in update_info.items()}, step=step)

        if terminated or truncated:
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
    eval_env = _make_env(FLAGS.seed + FLAGS.eval_seed_offset, allow_video=False)
    metrics = _run_evaluation(agent, eval_env, step=FLAGS.load_step or 0, experiment_name=FLAGS.run_name or FLAGS.project_name)
    result_path = os.path.join(FLAGS.load_dir, f"test_results_step{FLAGS.load_step or 'latest'}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    eval_env.close()
    print(f"Saved evaluation results to {result_path}")


def main(_):
    if FLAGS.mode == "testing":
        _testing_loop()
    else:
        _training_loop(_format_run_dir(FLAGS.seed))


if __name__ == "__main__":
    app.run(main)
