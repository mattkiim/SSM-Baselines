#! /usr/bin/env python
from __future__ import annotations

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents import SACLearner, TD3Learner
from jaxrl5.data import ReplayBuffer
from jaxrl5.envs import make_safety_env
from jaxrl5.evaluation import evaluate
from jaxrl5.wrappers import SafetyRecordEpisodeStatistics, WANDBVideo


FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "jaxrl5_safety_online", "wandb project name.")
flags.DEFINE_string("run_name", "", "wandb run name.")
flags.DEFINE_string(
    "env_name",
    "SafetyPointGoal1-v0",
    "Safety-Gymnasium environment name.",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Evaluation episodes.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Evaluation interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of steps before learning starts."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", False, "Enable wandb logging.")
flags.DEFINE_boolean("save_video", True, "Upload videos during evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
config_flags.DEFINE_config_file(
    "config",
    "examples/states/configs/sac_config.py",
    "Path to training hyperparameter configuration.",
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


def _make_env(env_name: str, seed: int, render_mode: str | None = None):
    env = make_safety_env(env_name, seed=seed, render_mode=render_mode)
    env = SafetyRecordEpisodeStatistics(env, deque_size=1)
    if FLAGS.wandb and FLAGS.save_video and render_mode is None:
        env = WANDBVideo(env)
    return env


def _make_eval_policy(agent):
    eval_agent = agent

    def policy(obs):
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(obs)
        return np.asarray(action)

    return policy


def main(_):
    _maybe_init_wandb()

    train_env = _make_env(FLAGS.env_name, seed=FLAGS.seed)
    eval_env = _make_env(FLAGS.env_name, seed=FLAGS.seed + 42)

    obs_shape = train_env.observation_space.shape
    act_shape = train_env.action_space.shape

    replay_buffer = ReplayBuffer(obs_shape, act_shape, capacity=FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed,
        train_env.observation_space,
        train_env.action_space,
        **kwargs,
    )

    observation, _ = train_env.reset(seed=FLAGS.seed)
    episode_return, episode_cost, episode_length = 0.0, 0.0, 0

    print("Training started")
    for step in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if step < FLAGS.start_training:
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
        )

        episode_return += float(reward)
        episode_cost += float(cost)
        episode_length += 1
        observation = next_obs

        if step >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

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

        if step % FLAGS.eval_interval == 0:
            metrics = evaluate(
                eval_env,
                _make_eval_policy(agent),
                episodes=FLAGS.eval_episodes,
            )
            if FLAGS.wandb:
                wandb.log(metrics, step=step)
            else:
                print(
                    f"[step {step}] return={metrics['eval/return_mean']:.2f} "
                    f"cost={metrics['eval/cost_mean']:.2f} "
                    f"len={metrics['eval/ep_len_mean']:.1f}"
                )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    app.run(main)

    print("Done")
