from __future__ import annotations

import tempfile

import gymnasium as gym
import numpy as np

from jaxrl5.agents.rac.rac_learner import RACLearner


def _make_batch(batch_size: int, obs_dim: int, act_dim: int) -> dict:
    observations = np.random.normal(size=(batch_size, obs_dim)).astype(np.float32)
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

    agent = RACLearner.create(
        seed=0,
        observation_space=observation_space,
        action_space=action_space,
        hidden_dims=(32, 32),
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
        "safety_critic_loss",
        "qh_mean",
        "target_qh_mean",
        "h_mean",
        "lambda_mean",
        "lambda_max",
        "violation_mean",
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

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = f"{tmpdir}/ckpt.msgpack"
        new_agent.save(ckpt_path)

        loaded_obj = RACLearner.load(ckpt_path)

        # ✅ 兼容两种实现：
        # 1) load() 直接返回 RACLearner
        # 2) load() 返回 dict（raw state / payload）
        if isinstance(loaded_obj, RACLearner):
            loaded = loaded_obj
        else:
            from flax import serialization as flax_serialization

            # 有些实现会存 {"state": state_dict, "meta": ...}，有些直接就是 state_dict
            state = loaded_obj["state"] if isinstance(loaded_obj, dict) and "state" in loaded_obj else loaded_obj

            template = RACLearner.create(
                seed=0,
                observation_space=observation_space,
                action_space=action_space,
                hidden_dims=(32, 32),
            )
            loaded = flax_serialization.from_state_dict(template, state)

        loaded_action, _ = loaded.eval_actions(single_obs)

    np.testing.assert_allclose(
        np.asarray(action_eval), np.asarray(loaded_action), rtol=1e-6, atol=1e-6
    )

    print("RAC smoke test passed.")



if __name__ == "__main__":
    main()
