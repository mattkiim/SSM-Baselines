from __future__ import annotations

import tempfile

import gymnasium as gym
import numpy as np

from jaxrl5.agents.sac.sac_cbf_learner import SACCbfLearner
from jaxrl5.tools.load_sac_cbf import load_sac_cbf


def _make_batch(batch_size: int, obs_dim: int, act_dim: int) -> dict:
    observations = np.random.normal(size=(batch_size, obs_dim)).astype(np.float32)
    observations[:, 2] = 2.0
    actions = np.random.uniform(-1.0, 1.0, size=(batch_size, act_dim)).astype(np.float32)
    rewards = np.random.normal(size=(batch_size,)).astype(np.float32)
    costs = np.random.randint(0, 2, size=(batch_size,)).astype(np.float32)
    next_observations = np.random.normal(size=(batch_size, obs_dim)).astype(np.float32)
    not_terminated = np.ones((batch_size,), dtype=np.float32)
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "costs": costs,
        "next_observations": next_observations,
        "not_terminated": not_terminated,
    }


def _assert_finite(metrics: dict) -> None:
    for key, value in metrics.items():
        arr = np.asarray(value)
        assert np.all(np.isfinite(arr)), f"Metric {key} has non-finite values: {arr}"


def main() -> None:
    obs_dim = 12
    act_dim = 2
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
    )

    agent = SACCbfLearner.create(
        seed=0,
        observation_space=observation_space,
        action_space=action_space,
        hidden_dims=(256, 256),
    )

    batch = _make_batch(batch_size=8, obs_dim=obs_dim, act_dim=act_dim)
    new_agent, metrics = agent.update(batch)

    required_keys = {
        "critic_loss",
        "q1_mean",
        "q2_mean",
        "target_q_mean",
        "actor_loss",
        "alpha",
        "entropy",
        "logp_mean",
        "cbf/residual_nom_mean",
        "cbf/residual_safe_mean",
        "cbf/success_rate",
        "cbf/delta_action_mean",
    }
    missing = required_keys.difference(metrics.keys())
    assert not missing, f"Missing metrics keys: {sorted(missing)}"
    _assert_finite(metrics)

    single_obs = batch["observations"][0]
    action_sample, _ = new_agent.sample_actions(single_obs)
    action_eval, _ = new_agent.eval_actions(single_obs)

    assert action_sample.shape == (act_dim,)
    assert action_eval.shape == (act_dim,)
    assert np.all(action_sample >= action_space.low - 1e-5)
    assert np.all(action_sample <= action_space.high + 1e-5)
    assert np.all(action_eval >= action_space.low - 1e-5)
    assert np.all(action_eval <= action_space.high + 1e-5)

    nominal = np.asarray(new_agent._eval_nominal_actions(single_obs[None]), dtype=np.float32)[0]
    residual_nom = new_agent._cbf_residual(single_obs, nominal)
    safe_action, info = new_agent._cbf_correct_action(single_obs, nominal)
    residual_safe = info["residual_safe"]
    assert residual_safe <= residual_nom + 1e-6

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = f"{tmpdir}/ckpt.msgpack"
        new_agent.save(ckpt_path)
        _, policy_fn, meta = load_sac_cbf(
            ckpt_path,
            observation_space=observation_space,
            action_space=action_space,
            deterministic=True,
        )
        loaded_action = policy_fn(single_obs)

    assert loaded_action.shape == (act_dim,)
    assert np.all(np.isfinite(loaded_action)), f"Non-finite action: {loaded_action}"
    assert np.all(loaded_action >= action_space.low - 1e-5)
    assert np.all(loaded_action <= action_space.high + 1e-5)
    assert meta.get("algo") == "sac_cbf"

    print("SAC-CBF smoke test passed.")


if __name__ == "__main__":
    main()
