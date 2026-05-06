#!/usr/bin/env python
"""RAC training for F16 stabilize-avoid v6.

Uses region_profile='v6', reset_box_mode='ours', init_curriculum_mode='box'.
Mirrors the structure of train_rac_quad3d.py.
"""
from __future__ import annotations

import datetime
import glob
import inspect
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import gymnasium as gym

from jaxrl5.agents.rac.rac_learner import RACLearner
from jaxrl5.data import ReplayBuffer
from jaxrl5.envs.registration import ensure_custom_envs_registered
from jaxrl5.utils import append_history
from jaxrl5.wrappers import AddCostFromInfo, WANDBVideo
from jaxrl5.wrappers.termination_penalty import TerminationPenaltyWrapper

from f16_gym_adapter import F16StabilizeGymWrapper


# ---------------------------------------------------------------------------
# Eval starts.
#
# Built around nominal_state_v5() with perturbations in (theta, h_alt) — the
# axes that the v6 diagnostic grid sweeps. Each tuple is (delta_theta, h_alt).
# All other state dims come from nominal_state_v5().
# ---------------------------------------------------------------------------
EVAL_PERTURBATIONS: List[Tuple[float, float]] = [
    (0.0,   500.0),   # nominal
    (0.4,   500.0),   # nose-up
    (-0.4,  500.0),   # nose-down
    (0.0,   200.0),   # low altitude (near safe_h_min=50)
    (0.0,   900.0),   # high altitude (near safe_h_max=1000)
    (0.6,   400.0),   # nose-up + low
    (-0.6,  600.0),   # nose-down + high
    (0.0,   100.0),   # very low
]


FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "jaxrl5_f16_stabilize_v6_rac", "wandb project name.")
flags.DEFINE_string("run_name", "", "wandb run name.")
flags.DEFINE_string("env_name", "F16StabilizeV6-v0", "Environment name (registered id).")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("eval_episodes", 8, "Evaluation episodes (one per EVAL_PERTURBATIONS entry).")
flags.DEFINE_integer("log_interval", 400, "Logging interval (steps).")
flags.DEFINE_integer("eval_interval", 10000, "Evaluation interval (steps).")
flags.DEFINE_integer("batch_size", 512, "Mini batch size.")
flags.DEFINE_integer("max_steps", 2_000_000, "Number of training steps.")
flags.DEFINE_integer("start_training", 25_000, "Number of steps before learning starts.")
flags.DEFINE_boolean("wandb", False, "Enable wandb logging.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_integer("eval_seed_offset", 12345, "Offset for evaluation environment seed.")
flags.DEFINE_boolean("save_video", False, "Upload videos during evaluation (wandb only).")
flags.DEFINE_integer("save_interval", 50_000, "Checkpoint save interval.")
flags.DEFINE_float(
    "termination_penalty",
    100.0,
    "Positive magnitude of the reward penalty applied on terminated transitions.",
)
flags.DEFINE_enum("mode", "training", ["training", "testing"], "Run mode.")
flags.DEFINE_string("load_dir", "", "Directory containing checkpoints for testing.")
flags.DEFINE_integer("load_step", None, "Checkpoint step to load for testing.")
flags.DEFINE_string(
    "results_root",
    "results/F16StabilizeV6-v0/jaxrl5_f16_stabilize_v6_rac",
    "Root directory for experiment outputs.",
)

# Env-specific flags exposed to the command line. Anything else falls back to
# the F16StabilizeEnvV6 defaults.
flags.DEFINE_string("region_profile", "v6", "F16 region profile.")
flags.DEFINE_string("reset_box_mode", "ours", "Reset proposal box.")
flags.DEFINE_boolean("init_curriculum", True, "Enable init-state curriculum.")
flags.DEFINE_string("init_curriculum_mode", "box", "Curriculum mode.")
flags.DEFINE_float("init_curriculum_frac", 0.5, "Curriculum start fraction.")
flags.DEFINE_float("init_curriculum_frac_end", 0.5, "Curriculum end fraction.")
flags.DEFINE_integer("init_curriculum_anneal_start", 200_000, "Anneal start step.")
flags.DEFINE_integer("init_curriculum_anneal_end", 800_000, "Anneal end step.")
flags.DEFINE_integer("max_episode_steps", 640, "Max steps per episode.")
flags.DEFINE_integer("goal_dwell_steps", 50, "Steps in goal needed for success.")

config_flags.DEFINE_config_file(
    "config",
    "examples/f16/configs/rac_f16_config.py",
    "Path to RAC hyperparameter configuration.",
    lock_config=False,
)


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------
def _env_kwargs() -> Dict:
    """Collect env kwargs from FLAGS."""
    return dict(
        max_episode_steps=FLAGS.max_episode_steps,
        goal_dwell_steps=FLAGS.goal_dwell_steps,
        region_profile=FLAGS.region_profile,
        reset_box_mode=FLAGS.reset_box_mode,
        init_curriculum=FLAGS.init_curriculum,
        init_curriculum_mode=FLAGS.init_curriculum_mode,
        init_curriculum_frac=FLAGS.init_curriculum_frac,
        init_curriculum_frac_end=FLAGS.init_curriculum_frac_end,
        init_curriculum_anneal_start=FLAGS.init_curriculum_anneal_start,
        init_curriculum_anneal_end=FLAGS.init_curriculum_anneal_end,
    )


def _is_cost_wrapped(env) -> bool:
    e = env
    while hasattr(e, "env"):
        if isinstance(e, AddCostFromInfo):
            return True
        e = e.env
    return isinstance(e, AddCostFromInfo)


def _make_env(seed: int, allow_video: bool = True) -> gym.Env:
    """Build a gymnasium-style F16 env wrapped for jaxrl5 RAC."""
    ensure_custom_envs_registered()
    env = F16StabilizeGymWrapper(seed=seed, **_env_kwargs())

    # AddCostFromInfo reads info["cost"] (we already set this in the wrapper)
    # and turns env.step into a 6-tuple (obs, r, c, term, trunc, info) which
    # is what jaxrl5 RAC training loop expects.
    if not _is_cost_wrapped(env):
        env = AddCostFromInfo(env)

    if allow_video and FLAGS.wandb and FLAGS.save_video:
        env = WANDBVideo(env)
    return env


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def _make_eval_policy(agent: RACLearner) -> Callable[[np.ndarray], np.ndarray]:
    eval_agent = agent

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal eval_agent
        action, eval_agent = eval_agent.eval_actions(np.asarray(obs, dtype=np.float32))
        return np.asarray(action, dtype=np.float32)

    return policy


def _build_eval_init(env: gym.Env, delta_theta: float, h_alt: float) -> Dict[str, float]:
    """Construct an eval init dict from nominal_state_v5() + perturbations."""
    base = env.unwrapped.nominal_state_v5()
    init = {
        "init_vt":   float(base[0]),
        "init_alpha": float(base[1]),
        "init_beta":  float(base[2]),
        "init_phi":   float(base[3]),
        "init_theta": float(base[4] + delta_theta),
        "init_psi":   float(base[5]),
        "init_p":     float(base[6]),
        "init_q":     float(base[7]),
        "init_r":     float(base[8]),
        "init_pn":    float(base[9]),
        "init_pe":    float(base[10]),
        "init_h":     float(h_alt),
        "init_pow":   float(base[12]),
    }
    return init


def evaluate_perturbation_grid(env, policy_fn, seed: int = 0) -> Dict[str, float]:
    """Evaluate on the EVAL_PERTURBATIONS grid."""
    returns, costs, ep_lens, viol_rates = [], [], [], []
    successes, crashes = [], []

    for i, (dtheta, h_alt) in enumerate(EVAL_PERTURBATIONS):
        init = _build_eval_init(env, dtheta, h_alt)
        obs, _ = env.reset(seed=seed + i, options=init)

        ep_ret, ep_cost, viol_sum, steps = 0.0, 0.0, 0.0, 0
        terminated = False
        truncated = False
        crashed = False
        success = False

        while not (terminated or truncated):
            act = policy_fn(obs)
            obs, r, c, terminated, truncated, info = env.step(act)
            ep_ret += float(r)
            ep_cost += float(c)
            viol_sum += float(c)
            steps += 1
            crashed = crashed or bool(info.get("crashed", False))
            success = success or bool(info.get("success", False))

        returns.append(ep_ret)
        costs.append(ep_cost)
        ep_lens.append(steps)
        viol_rates.append(viol_sum / max(1, steps))
        successes.append(float(success))
        crashes.append(float(crashed))

    returns = np.asarray(returns, np.float32)
    costs = np.asarray(costs, np.float32)
    ep_lens = np.asarray(ep_lens, np.float32)
    viol_rates = np.asarray(viol_rates, np.float32)
    successes = np.asarray(successes, np.float32)
    crashes = np.asarray(crashes, np.float32)

    return {
        "eval/return_mean": float(returns.mean()),
        "eval/return_std": float(returns.std()),
        "eval/cost_mean": float(costs.mean()),
        "eval/cost_std": float(costs.std()),
        "eval/ep_len_mean": float(ep_lens.mean()),
        "eval/ep_len_std": float(ep_lens.std()),
        "eval/violation_rate_mean": float(viol_rates.mean()),
        "eval/violation_rate_std": float(viol_rates.std()),
        "eval/success_rate": float(successes.mean()),
        "eval/crash_rate": float(crashes.mean()),
    }


# ---------------------------------------------------------------------------
# Boilerplate (mirrors train_rac_quad3d.py)
# ---------------------------------------------------------------------------
def _maybe_init_wandb():
    if not FLAGS.wandb:
        return
    try:
        import wandb
    except ImportError:
        print("wandb not installed; disabling wandb logging.")
        FLAGS.wandb = False
        return
    run_name = FLAGS.run_name or None
    try:
        wandb.init(project=FLAGS.project_name, name=run_name,
                   tags=[FLAGS.run_name] if FLAGS.run_name else None)
    except Exception as exc:
        print(f"wandb init failed ({exc}); disabling.")
        FLAGS.wandb = False
        return
    wandb.config.update(FLAGS)


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


def _save_checkpoint(agent: RACLearner, ckpt_dir: str, step: int) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{step}.msgpack")
    agent.save(path)
    return path


def _find_checkpoint(load_dir: str, load_step: Optional[int]) -> str:
    ckpt_dir = os.path.join(load_dir, "checkpoints")
    if load_step is not None:
        path = os.path.join(ckpt_dir, f"ckpt_{load_step}.msgpack")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = glob.glob(os.path.join(ckpt_dir, "ckpt_*.msgpack"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints under {ckpt_dir}")
    return sorted(candidates)[-1]


def _filter_create_kwargs(cfg: Dict) -> Dict:
    sig = inspect.signature(RACLearner.create)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def _run_evaluation(agent: RACLearner, eval_env, step: int, experiment_name: str):
    eval_policy = _make_eval_policy(agent)

    # Set eval mode on the underlying env (disables curriculum, freezes obs stats).
    eval_env.unwrapped.set_eval_mode()
    metrics = evaluate_perturbation_grid(eval_env, eval_policy,
                                         seed=FLAGS.seed + FLAGS.eval_seed_offset)
    eval_env.unwrapped.set_train_mode()

    if FLAGS.wandb:
        import wandb
        wandb.log(metrics, step=step)
    else:
        print(
            f"[step {step}] return={metrics['eval/return_mean']:.2f} "
            f"cost={metrics['eval/cost_mean']:.2f} "
            f"viol={metrics['eval/violation_rate_mean']:.3f} "
            f"len={metrics['eval/ep_len_mean']:.1f} "
            f"crash={metrics['eval/crash_rate']:.2f} "
            f"success={metrics['eval/success_rate']:.2f}"
        )

    append_history(
        step,
        FLAGS.env_name,
        experiment_name,
        FLAGS.seed,
        {
            "eval/return_mean":         metrics["eval/return_mean"],
            "eval/return_std":          metrics["eval/return_std"],
            "eval/cost_mean":           metrics["eval/cost_mean"],
            "eval/cost_std":            metrics.get("eval/cost_std", float("nan")),
            "eval/violation_rate_mean": metrics["eval/violation_rate_mean"],
            "eval/violation_rate_std":  metrics.get("eval/violation_rate_std", float("nan")),
            "eval/ep_len_mean":         metrics["eval/ep_len_mean"],
            "eval/success_rate":        metrics["eval/success_rate"],
            "eval/crash_rate":          metrics["eval/crash_rate"],
        },
    )
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _training_loop(run_dir: str) -> None:
    _maybe_init_wandb()

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    _save_config(run_dir)

    train_env = _make_env(seed=FLAGS.seed, allow_video=True)
    eval_env = _make_env(seed=FLAGS.seed + FLAGS.eval_seed_offset, allow_video=True)

    obs_shape = train_env.observation_space.shape
    act_shape = train_env.action_space.shape

    train_env = TerminationPenaltyWrapper(
        train_env,
        penalty=-abs(float(FLAGS.termination_penalty)),
        apply_on_truncated=False,
    )

    replay_buffer = ReplayBuffer(obs_shape, act_shape, capacity=FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    kwargs = _filter_create_kwargs(dict(FLAGS.config))
    model_cls = kwargs.pop("model_cls", "RACLearner")
    agent: RACLearner = globals()[model_cls].create(
        FLAGS.seed,
        train_env.observation_space,
        train_env.action_space,
        **kwargs,
    ) if model_cls in globals() else RACLearner.create(
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
        # Pass training step into the env so curriculum annealing works.
        train_env.unwrapped.set_train_step(step)

        if step < FLAGS.start_training:
            action = np.asarray(train_env.action_space.sample(), dtype=np.float32)
        else:
            action, agent = agent.sample_actions(np.asarray(observation, dtype=np.float32))
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, train_env.action_space.low, train_env.action_space.high)

        next_obs, reward, cost, terminated, truncated, info = train_env.step(action)
        done = bool(terminated or truncated)

        replay_buffer.insert(
            observation, action, float(reward), float(cost),
            next_obs, terminated, truncated,
        )

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

        if done:
            if FLAGS.wandb:
                import wandb
                wandb.log({
                    "training/return": episode_return,
                    "training/cost": episode_cost,
                    "training/length": episode_length,
                }, step=step)
            observation, _ = train_env.reset()
            episode_return, episode_cost, episode_length = 0.0, 0.0, 0

        if step % FLAGS.eval_interval == 0:
            _run_evaluation(agent, eval_env, step=step, experiment_name=experiment_name)

        if step % FLAGS.save_interval == 0:
            _save_checkpoint(agent, ckpt_dir, step)

    train_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# Testing loop
# ---------------------------------------------------------------------------
def _testing_loop() -> None:
    if not FLAGS.load_dir:
        raise ValueError("--load_dir is required in testing mode.")
    ckpt_path = _find_checkpoint(FLAGS.load_dir, FLAGS.load_step)
    agent = RACLearner.load(ckpt_path)

    eval_env = _make_env(seed=FLAGS.seed + FLAGS.eval_seed_offset, allow_video=False)
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
