"""Soft Actor-Critic with a Lagrangian cost constraint (SAC-Lag)."""

from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax.serialization as serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal
from jaxrl5.networks import MLP, Ensemble, StateActionValue, subsample_ensemble


class SACLagLearner(Agent):
    """SAC learner with a scalar Lagrangian multiplier for cost constraints.

    The cost_limit parameter constrains the expected discounted cost-to-go
    (via the cost critic estimate). The lambda update performs projected
    gradient ascent on the constraint violation.
    """

    critic: TrainState
    target_critic: TrainState
    cost_critic: TrainState
    target_cost_critic: TrainState
    temp: TrainState

    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)

    lagrangian_lambda: jnp.ndarray
    lambda_lr: float
    lambda_max: float
    cost_limit: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        cost_critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        lambda_init: float = 0.0,
        lambda_lr: float = 1e-3,
        lambda_max: float = 1000.0,
        cost_limit: float = 0.0,
    ) -> "SACLagLearner":
        """Construct a SAC-Lag learner with reward and cost critics."""

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, cost_critic_key, temp_key = jax.random.split(rng, 5)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)

        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        cost_critic_def = Ensemble(critic_cls, num=num_qs)
        cost_critic_params = cost_critic_def.init(cost_critic_key, observations, actions)[
            "params"
        ]
        cost_critic = TrainState.create(
            apply_fn=cost_critic_def.apply,
            params=cost_critic_params,
            tx=optax.adam(learning_rate=cost_critic_lr),
        )
        target_cost_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_cost_critic = TrainState.create(
            apply_fn=target_cost_critic_def.apply,
            params=cost_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            cost_critic=cost_critic,
            target_cost_critic=target_cost_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            lagrangian_lambda=jnp.asarray(lambda_init, dtype=jnp.float32),
            lambda_lr=float(lambda_lr),
            lambda_max=float(lambda_max),
            cost_limit=float(cost_limit),
        )

    def _sample_policy(self, observations: jnp.ndarray, rng: jax.random.PRNGKey):
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def _eval_policy(self, observations: jnp.ndarray):
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        return dist.mode()

    def _format_obs(self, observations: jnp.ndarray):
        if observations.ndim == 1:
            return observations[None], True
        return observations, False

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray):
        obs, single = self._format_obs(observations)
        key, rng = jax.random.split(self.rng)
        actions, _ = self._sample_policy(obs, key)
        if single:
            actions = jnp.squeeze(actions, axis=0)
        return actions, self.replace(rng=rng)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        obs, single = self._format_obs(observations)
        actions = self._eval_policy(obs)
        if single:
            actions = jnp.squeeze(actions, axis=0)
        return actions, self

    def update_critic(self, batch: DatasetDict) -> Tuple["SACLagLearner", Dict[str, float]]:
        rng = self.rng

        dist = self.actor.apply_fn({"params": self.actor.params}, batch["next_observations"])
        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)
        next_log_probs = dist.log_prob(next_actions)

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        next_q = next_qs.min(axis=0)
        target_q = batch["rewards"] + self.discount * batch["not_terminated"] * next_q

        if self.backup_entropy:
            alpha = self.temp.apply_fn({"params": self.temp.params})
            target_q -= (
                self.discount
                * batch["not_terminated"]
                * alpha
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )
            loss = ((qs - target_q) ** 2).mean()
            return loss, {
                "critic_loss": loss,
                "q1_mean": qs[0].mean(),
                "q2_mean": qs[1].mean() if qs.shape[0] > 1 else qs[0].mean(),
                "target_q_mean": target_q.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update_cost_critic(
        self, batch: DatasetDict
    ) -> Tuple["SACLagLearner", Dict[str, float]]:
        rng = self.rng

        dist = self.actor.apply_fn({"params": self.actor.params}, batch["next_observations"])
        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_cost_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_cost_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        next_qc = next_qs.min(axis=0)
        target_qc = (
            batch["costs"] + self.discount * batch["not_terminated"] * next_qc
        )

        key, rng = jax.random.split(rng)

        def cost_critic_loss_fn(cost_params):
            qs = self.cost_critic.apply_fn(
                {"params": cost_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )
            loss = ((qs - target_qc) ** 2).mean()
            return loss, {
                "cost_critic_loss": loss,
                "qc1_mean": qs[0].mean(),
                "qc2_mean": qs[1].mean() if qs.shape[0] > 1 else qs[0].mean(),
                "target_qc_mean": target_qc.mean(),
            }

        grads, info = jax.grad(cost_critic_loss_fn, has_aux=True)(self.cost_critic.params)
        cost_critic = self.cost_critic.apply_gradients(grads=grads)

        target_cost_params = optax.incremental_update(
            cost_critic.params, self.target_cost_critic.params, self.tau
        )
        target_cost_critic = self.target_cost_critic.replace(params=target_cost_params)

        return (
            self.replace(cost_critic=cost_critic, target_cost_critic=target_cost_critic, rng=rng),
            info,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple["SACLagLearner", Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        key3, rng = jax.random.split(rng)

        lambda_value = self.lagrangian_lambda

        def actor_loss_fn(actor_params):
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)

            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )
            q = qs.min(axis=0)

            qcs = self.cost_critic.apply_fn(
                {"params": self.cost_critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key3},
            )
            qc = qcs.min(axis=0)

            alpha = self.temp.apply_fn({"params": self.temp.params})
            actor_loss = (alpha * log_probs - q + lambda_value * qc).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": -log_probs.mean(),
                "logp_mean": log_probs.mean(),
                "alpha": alpha,
            }

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: jnp.ndarray) -> Tuple["SACLagLearner", Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "alpha": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_lambda(self, batch: DatasetDict) -> Tuple["SACLagLearner", Dict[str, float]]:
        rng = self.rng
        key, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)

        dist = self.actor.apply_fn({"params": self.actor.params}, batch["observations"])
        actions = dist.sample(seed=key)
        qcs = self.cost_critic.apply_fn(
            {"params": self.cost_critic.params},
            batch["observations"],
            actions,
            True,
            rngs={"dropout": key2},
        )
        qc = qcs.min(axis=0)
        jc = qc.mean()

        new_lambda = jnp.clip(
            self.lagrangian_lambda + self.lambda_lr * (jc - self.cost_limit),
            0.0,
            self.lambda_max,
        )

        metrics = {
            "lambda": new_lambda,
            "Jc_mean": jc,
            "cost_limit": jnp.asarray(self.cost_limit),
            "violation_mean": batch["costs"].mean(),
        }
        return self.replace(lagrangian_lambda=new_lambda, rng=rng), metrics

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent, critic_info = self.update_critic(batch)
        new_agent, cost_critic_info = new_agent.update_cost_critic(batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        new_agent, lambda_info = new_agent.update_lambda(batch)

        metrics = {**critic_info, **cost_critic_info, **actor_info, **temp_info, **lambda_info}
        return new_agent, metrics

    def save(self, path: str) -> None:
        """Serialize the learner to a file using Flax msgpack."""
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(self))

    @classmethod
    def load(cls, path: str) -> "SACLagLearner":
        """Load a learner checkpoint from disk."""
        with open(path, "rb") as f:
            data = f.read()
        restored = None
        if hasattr(serialization, "msgpack_restore"):
            restored = serialization.msgpack_restore(data)
            if isinstance(restored, cls):
                return restored
            if isinstance(restored, dict):
                for value in restored.values():
                    if isinstance(value, cls):
                        return value

        try:
            return serialization.from_bytes(cls, data)
        except Exception as exc:
            restored_type = type(restored) if restored is not None else None
            raise TypeError(
                f"Loaded checkpoint type {restored_type} is not {cls.__name__} or a container of it"
            ) from exc
