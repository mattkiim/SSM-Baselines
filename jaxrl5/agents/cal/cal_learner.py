from __future__ import annotations

from functools import partial
from typing import Dict, Sequence

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.algorithms.cal_update import (
    CALTrainState,
    CALUpdateConfig,
    cal_update,
)
from jaxrl5.distributions import TanhNormal
from jaxrl5.networks import Ensemble, MLP, StateActionValue


class CALAgent(Agent):
    qr1: TrainState
    qr2: TrainState
    qc: TrainState
    qr1_target: TrainState
    qr2_target: TrainState
    qc_target: TrainState
    alpha: TrainState
    log_lambda: jnp.ndarray
    update_config: CALUpdateConfig = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        cost_lr: float = 5e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        gamma_c: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
        k_ucb: float = 0.5,
        alm_c: float = 10.0,
        lambda_lr: float = 1e-3,
        cost_limit: float = 25.0,
        init_temperature: float = 1.0,
        init_lambda: float = 1.0,
        qc_ens_size: int = 6,
    ):

        obs_sample = jnp.asarray(observation_space.sample())
        act_sample = jnp.asarray(action_space.sample())

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key1, critic_key2, qc_key, temp_key = jax.random.split(
            rng, 6
        )

        action_dim = action_space.shape[-1]
        if target_entropy is None:
            target_entropy = -float(action_dim)

        # Actor.
        actor_base = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_def = TanhNormal(actor_base, action_dim)
        actor_params = actor_def.init(actor_key, obs_sample)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(actor_lr),
        )

        # Reward critics.
        critic_base = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        critic_def = StateActionValue(critic_base)
        critic_params1 = critic_def.init(critic_key1, obs_sample, act_sample)["params"]
        critic_params2 = critic_def.init(critic_key2, obs_sample, act_sample)["params"]
        qr1 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params1,
            tx=optax.adam(critic_lr),
        )
        qr2 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params2,
            tx=optax.adam(critic_lr),
        )
        zero_tx = optax.GradientTransformation(lambda _: None, lambda *_: (None, None))
        qr1_target = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params1,
            tx=zero_tx,
        )
        qr2_target = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params2,
            tx=zero_tx,
        )

        # Cost critic ensemble.
        cost_base = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        cost_cls = partial(StateActionValue, base_cls=cost_base)
        qc_def = Ensemble(cost_cls, num=qc_ens_size)
        qc_params = qc_def.init(qc_key, obs_sample, act_sample)["params"]
        qc = TrainState.create(
            apply_fn=qc_def.apply,
            params=qc_params,
            tx=optax.adam(cost_lr),
        )
        qc_target = TrainState.create(
            apply_fn=qc_def.apply,
            params=qc_params,
            tx=zero_tx,
        )

        # Temperature / alpha.
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        alpha = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(alpha_lr),
        )

        update_config = CALUpdateConfig(
            gamma=gamma,
            gamma_c=gamma_c,
            tau=tau,
            k_ucb=k_ucb,
            alm_c=alm_c,
            lambda_lr=lambda_lr,
            cost_limit=cost_limit,
            target_entropy=target_entropy,
        )

        return cls(
            rng=rng,
            actor=actor,
            qr1=qr1,
            qr2=qr2,
            qc=qc,
            qr1_target=qr1_target,
            qr2_target=qr2_target,
            qc_target=qc_target,
            alpha=alpha,
            log_lambda=jnp.log(init_lambda),
            update_config=update_config,
        )

    def _build_state(self) -> CALTrainState:
        return CALTrainState(
            actor=self.actor,
            qr1=self.qr1,
            qr2=self.qr2,
            qc=self.qc,
            qr1_target=self.qr1_target,
            qr2_target=self.qr2_target,
            qc_target=self.qc_target,
            alpha=self.alpha,
            log_lambda=self.log_lambda,
        )

    def update(self, batch, utd_ratio: int = 1):
        agent = self
        metrics: Dict[str, jnp.ndarray] = {}

        for i in range(utd_ratio):

            def _slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(_slice, batch)

            cal_state = agent._build_state()
            rng, new_state, metrics = cal_update(
                agent.rng, cal_state, mini_batch, agent.update_config
            )

            agent = agent.replace(
                rng=rng,
                actor=new_state.actor,
                qr1=new_state.qr1,
                qr2=new_state.qr2,
                qc=new_state.qc,
                qr1_target=new_state.qr1_target,
                qr2_target=new_state.qr2_target,
                qc_target=new_state.qc_target,
                alpha=new_state.alpha,
                log_lambda=new_state.log_lambda,
            )

        return agent, metrics

