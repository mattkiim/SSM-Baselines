#! /usr/bin/env python
from __future__ import annotations

import json
import os
import csv

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents.rac.rac_learner import RACLearner
from jaxrl5.data import ReplayBuffer
from jaxrl5.envs import make_safety_env
from jaxrl5.evaluation import evaluate
from jaxrl5.tools.load_rac import load_rac
from jaxrl5.utils import append_history, get_run_dir
from jaxrl5.wrappers import SafetyRecordEpisodeStatistics, StaticLayoutWrapper, WANDBVideo
from jaxrl5.wrappers.velocity_constraint import VelocityConstraint
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "jaxrl5_rac_online", "wandb project name.")
flags.DEFINE_string("run_name", "", "wandb run name.")
flags.DEFINE_string(
    "env_name",
    "SafetyPointGoal1-v0",
    "Safety-Gymnasium environment name.",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Evaluation episodes.")
flags.DEFINE_integer("eval_max_episode_steps", 1000, "Maximum steps per evaluation episode.")
flags.DEFINE_integer("eval_seed", None, "If set, evaluate on seeds eval_seed through eval_seed+episodes-1.")
flags.DEFINE_boolean("normalize_actions", False, "Map agent actions from [-1,1] to native environment bounds.")
flags.DEFINE_integer("log_interval", 400, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Evaluation interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of steps before learning starts."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", False, "Enable wandb logging.")
flags.DEFINE_boolean("save_video", False, "Upload videos during evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_integer("epoch_length", 400, "Number of environment steps per epoch.")
flags.DEFINE_integer("save_interval", 50000, "Checkpoint save interval (steps).")
flags.DEFINE_string("resume_checkpoint", None, "Learner checkpoint to continue from; replay starts empty.")
flags.DEFINE_integer("resume_step", 0, "Completed environment steps at the checkpoint.")
flags.DEFINE_integer("resume_warmup_steps", 10000, "Policy transitions to collect before resumed updates.")
flags.DEFINE_integer("replay_capacity", None, "Replay capacity; defaults to max_steps.")
flags.DEFINE_integer(
    "static_seed",
    None,
    "If set, pin every episode reset (train and eval) to this exact "
    "Safety-Gymnasium layout seed, disabling per-episode randomization "
    "of hazard/goal/robot placement.",
)

config_flags.DEFINE_config_file(
    "config",
    "examples/states/configs/rac_config.py",
    "Path to the training hyperparameter configuration.",
    lock_config=False,
)


def _maybe_init_wandb():
    if FLAGS.wandb:
        if FLAGS.run_name:
            wandb.init(
                project=FLAGS.project_name,
                name=FLAGS.run_name,
                tags=[FLAGS.run_name],
            )
        else:
            wandb.init(project=FLAGS.project_name)
        wandb.config.update(FLAGS)


def _make_env(env_name: str, seed: int, allow_video: bool = True):
    env = make_safety_env(env_name, seed=seed)
    if FLAGS.normalize_actions:
        env = SymmetricActionWrapper(env)
    if FLAGS.config.safety_h_mode == 'reachability_transition':
        env = VelocityConstraint(env)
    if FLAGS.static_seed is not None:
        env = StaticLayoutWrapper(env, seed=FLAGS.static_seed)
        env.reset()
    env = SafetyRecordEpisodeStatistics(env, deque_size=1)
    if allow_video and FLAGS.wandb and FLAGS.save_video:
        env = WANDBVideo(env)
    return env


def _make_eval_policy(agent):
    eval_agent = agent

    def policy(obs):
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(obs)
        return np.asarray(action)

    return policy


def _save_config(run_dir, config) -> None:
    os.makedirs(run_dir, exist_ok=True)
    cfg = {
        "flags": {k: v for k, v in FLAGS.flag_values_dict().items() if k != "config"},
        "config": dict(config),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _save_checkpoint(agent: RACLearner, ckpt_dir, step: int) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{step}.msgpack")
    agent.save(path)
    return path


def main(_):
    if FLAGS.eval_episodes < 1 or FLAGS.eval_max_episode_steps < 1:
        raise ValueError("Evaluation episode count and step limit must be positive")
    if FLAGS.resume_checkpoint:
        if not 0 < FLAGS.resume_step < FLAGS.max_steps:
            raise ValueError("resume_step must be positive and below max_steps")
        if FLAGS.resume_warmup_steps < 1:
            raise ValueError("resume_warmup_steps must be positive")
    elif FLAGS.resume_step:
        raise ValueError("resume_step requires resume_checkpoint")
    if FLAGS.replay_capacity is not None and FLAGS.replay_capacity < 1:
        raise ValueError("replay_capacity must be positive")
    _maybe_init_wandb()

    train_env = _make_env(FLAGS.env_name, seed=FLAGS.seed)
    train_env.action_space.seed(FLAGS.seed)
    eval_env = _make_env(FLAGS.env_name, seed=FLAGS.seed + 42)

    obs_shape = train_env.observation_space.shape
    act_shape = train_env.action_space.shape

    replay_buffer = ReplayBuffer(
        obs_shape, act_shape, capacity=FLAGS.replay_capacity or FLAGS.max_steps,
        store_safety_h=FLAGS.config.safety_h_mode == 'reachability_transition',
    )
    replay_buffer.seed(FLAGS.seed)

    config = FLAGS.config.copy_and_resolve_references()
    if not config.get("target_entropy"):
        config.target_entropy = -float(train_env.action_space.shape[-1])

    agent_kwargs = dict(config)
    for key in (
        "model_cls",
        "env_name",
        "seed",
        "max_steps",
        "batch_size",
        "start_training",
        "eval_interval",
        "eval_episodes",
        "log_interval",
        "utd_ratio",
        "epoch_length",
    ):
        agent_kwargs.pop(key, None)

    if FLAGS.resume_checkpoint:
        agent, _, meta = load_rac(
            FLAGS.resume_checkpoint,
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            seed=FLAGS.seed,
        )
        if meta["step"] is not None and meta["step"] != FLAGS.resume_step:
            raise ValueError("resume_step does not match the checkpoint filename")
        if agent.safety_h_mode != FLAGS.config.safety_h_mode:
            raise ValueError('Cannot resume a checkpoint with a different safety formulation')
        # Retain the source learner configuration used by the checkpoint loader.
        with open(meta["config_path"], encoding="utf-8") as f:
            source = json.load(f)
        for key, expected in (("env_name", FLAGS.env_name), ("seed", FLAGS.seed)):
            if source.get("flags", {}).get(key) != expected:
                raise ValueError(f"Checkpoint {key} does not match this run")
        if source.get('flags', {}).get('normalize_actions', False) != FLAGS.normalize_actions:
            raise ValueError('Cannot resume with a different action mapping')
        config = source.get("config", source)
        print(f"Resumed {FLAGS.resume_checkpoint} at step {FLAGS.resume_step}; "
              f"refilling replay for {FLAGS.resume_warmup_steps} policy steps.", flush=True)
    else:
        agent = RACLearner.create(
            FLAGS.seed,
            train_env.observation_space,
            train_env.action_space,
            **agent_kwargs,
        )

    experiment_name = FLAGS.run_name or FLAGS.project_name
    run_dir = get_run_dir(FLAGS.env_name, experiment_name, FLAGS.seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if os.path.exists(os.path.join(run_dir, "config.json")):
        raise ValueError(f"Run already exists: {run_dir}; choose a new run_name")
    _save_config(run_dir, config)

    observation, _ = train_env.reset(seed=FLAGS.seed)
    episode_return, episode_cost, episode_length = 0.0, 0.0, 0
    epoch_reward, epoch_cost = 0.0, 0.0
    latest_cost_mean = None
    latest_actor_metrics = {}
    latest_actor_metrics_step = 0
    initial_learner_updates = int(agent.update_step)

    for step in tqdm.tqdm(
        range(FLAGS.resume_step + 1, FLAGS.max_steps + 1),
        initial=FLAGS.resume_step, total=FLAGS.max_steps,
        smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if not FLAGS.resume_checkpoint and step < FLAGS.start_training:
            action = np.asarray(train_env.action_space.sample(), dtype=np.float32)
        else:
            action, agent = agent.sample_actions(observation)
            action = np.asarray(action, dtype=np.float32)

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
            safety_h=info['safety_h'] if replay_buffer.safety_h is not None else None,
        )

        episode_return += float(reward)
        episode_cost += float(cost)
        episode_length += 1
        epoch_reward += float(reward)
        epoch_cost += float(cost)
        observation = next_obs

        update_start = (
            FLAGS.resume_step + FLAGS.resume_warmup_steps
            if FLAGS.resume_checkpoint else FLAGS.start_training
        )
        if step >= update_start:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch)

            learner_iteration = initial_learner_updates + step - update_start + 1
            if learner_iteration % agent.policy_update_period == 0:
                latest_actor_metrics = {key: update_info[key] for key in (
                    'actor_loss', 'entropy', 'logp_mean', 'temperature_loss',
                    'action_saturation_fraction')}
                latest_actor_metrics_step = step

            if step % FLAGS.log_interval == 0 and latest_actor_metrics:
                # Log actual multipliers: update_info contains zero placeholders
                # on steps where the multiplier optimizer is not scheduled.
                lam = np.asarray(agent._lambda_values(batch['observations']))
                diagnostic = {key: float(value) for key, value in update_info.items()}
                diagnostic.update({key: float(value) for key, value in latest_actor_metrics.items()})
                diagnostic.update(
                    actor_metrics_step=latest_actor_metrics_step,
                    step=step, lambda_mean=float(lam.mean()),
                    lambda_min=float(lam.min()), lambda_max_observed=float(lam.max()),
                    lambda_at_cap_fraction=0. if agent.reference_protocol else float(np.mean(lam >= agent.lambda_max)),
                )
                if not all(np.isfinite(value) for value in diagnostic.values()):
                    raise FloatingPointError(f'Nonfinite training diagnostic at step {step}')
                path = os.path.join(run_dir, 'training_metrics.csv')
                new_file = not os.path.exists(path)
                with open(path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(diagnostic))
                    if new_file:
                        writer.writeheader()
                    writer.writerow(diagnostic)

            if FLAGS.wandb and step % FLAGS.log_interval == 0:
                wandb.log(
                    {f"training/{k}": float(v) for k, v in update_info.items()},
                    step=step,
                )

        if done:
            if FLAGS.wandb:
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

        if step % FLAGS.epoch_length == 0:
            latest_cost_mean = epoch_cost / FLAGS.epoch_length
            if FLAGS.wandb:
                wandb.log(
                    {
                        "epoch/reward_sum": epoch_reward,
                        "epoch/reward_mean": epoch_reward / FLAGS.epoch_length,
                        "epoch/cost_sum": epoch_cost,
                        "epoch/cost_mean": latest_cost_mean,
                        "training/cost_mean": latest_cost_mean,
                    },
                    step=step,
                )
            epoch_reward, epoch_cost = 0.0, 0.0

        if step % FLAGS.eval_interval == 0:
            metrics = evaluate(
                eval_env,
                _make_eval_policy(agent),
                episodes=FLAGS.eval_episodes,
                max_episode_steps=FLAGS.eval_max_episode_steps,
                seed=FLAGS.eval_seed,
            )
            if FLAGS.wandb:
                wandb.log(metrics, step=step)
            else:
                print(
                    f"[step {step}] return={metrics['eval/return_mean']:.2f} "
                    f"cost={metrics['eval/cost_mean']:.2f} "
                    f"len={metrics['eval/ep_len_mean']:.1f}"
                )

            append_history(
                step,
                FLAGS.env_name,
                experiment_name,
                FLAGS.seed,
                metrics,
                run_dir=run_dir,
            )

        if step % FLAGS.save_interval == 0:
            _save_checkpoint(agent, ckpt_dir, step)

    if FLAGS.max_steps % FLAGS.save_interval != 0:
        _save_checkpoint(agent, ckpt_dir, FLAGS.max_steps)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    app.run(main)
