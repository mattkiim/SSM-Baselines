from __future__ import annotations

import tempfile

import gymnasium as gym
import numpy as np

from jaxrl5.agents.cal.cal_learner import CALAgent
from jaxrl5.tools.load_agent import load_agent


def main() -> None:
    obs_dim = 12
    act_dim = 2
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
    )

    agent = CALAgent.create(
        seed=0,
        observation_space=observation_space,
        action_space=action_space,
        hidden_dims=(256, 256), # 需要和训练时设定的结构一致
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = f"{tmpdir}/ckpt.msgpack"
        if hasattr(agent, "save"):
            agent.save(ckpt_path)
        else:
            from flax import serialization

            with open(ckpt_path, "wb") as f:
                f.write(serialization.to_bytes(agent))

        _, policy_fn, meta = load_agent(
            "cal",
            ckpt_path,
            observation_space=observation_space,
            action_space=action_space,
            deterministic=True,
        )

        obs = np.random.normal(size=(obs_dim,)).astype(np.float32)
        action = policy_fn(obs)

    assert action.shape == (act_dim,)
    assert np.all(np.isfinite(action)), f"Non-finite action: {action}"
    assert np.all(action >= action_space.low - 1e-5)
    assert np.all(action <= action_space.high + 1e-5)
    assert meta.get("algo") == "cal"

    print("CAL load smoke test passed.")


if __name__ == "__main__":
    main()
