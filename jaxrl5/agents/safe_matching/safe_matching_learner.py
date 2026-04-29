"""Safe Score Matching learner built on the QSM implementation."""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import flax.serialization as serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import (
    DDPM,
    FourierFeatures,
    MLP,
    StateActionValue,
    cosine_beta_schedule,
    ddpm_sampler,
    vp_beta_schedule,
)


tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def tensorstats(tensor, prefix=None):
    assert tensor.size > 0, tensor.shape
    metrics = {
        "mean": tensor.mean(),
        "std": tensor.std(),
        "mag": jnp.abs(tensor).max(),
        "min": tensor.min(),
        "max": tensor.max(),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


class SafeScoreMatchingLearner(Agent):
    score_model: TrainState
    critic_1: TrainState
    critic_2: TrainState
    target_critic_1: TrainState
    target_critic_2: TrainState
    safety_critic: TrainState
    target_safety_critic: TrainState
    lambda_net: TrainState

    discount: float
    tau: float
    safety_tau: float
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    M_q: float

    cost_limit: float
    safety_discount: float
    safety_lambda: float
    alpha_coef: float
    safety_threshold: float
    safety_grad_scale: float
    safe_lagrange_coef: float
    lambda_max: float
    lambda_update_coef: float
    actor_grad_coef: float
    actor_safety_grad_coef: float
    actor_grad_loss_coef: float
    actor_aux_loss_coef: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = "mlp",
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        safety_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        safety_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        lambda_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        safety_tau: Optional[float] = None,
        ddpm_temperature: float = 1.0,
        actor_layer_norm: bool = False,
        T: int = 5,
        time_dim: int = 64,
        clip_sampler: bool = True,
        beta_schedule: str = "vp",
        decay_steps: Optional[int] = int(2e6),
        M_q: float = 1.0,
        cost_limit: float = 25.0,
        safety_discount: float = 0.99,
        safety_lambda: float = 1.0,
        alpha_coef: float = 0.1,
        safety_threshold: float = 0.0,
        safety_grad_scale: float = 1.0,
        safe_lagrange_coef: float = 0.5,
        lambda_lr: Union[float, optax.Schedule] = 3e-4,
        lambda_max: float = 100.0,
        lambda_update_coef: float = 1.0,
        actor_grad_coef: float = 1.0,
        actor_safety_grad_coef: float = 1.0,
        actor_grad_loss_coef: float = 1.0,
        actor_aux_loss_coef: float = 0.1,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, safety_key, lambda_key = jax.random.split(rng, 5)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[-1]

        preprocess_time_cls = partial(
            FourierFeatures, output_size=time_dim, learnable=True
        )

        cond_model_cls = partial(
            MLP,
            hidden_dims=(128, 128),
            activations=mish,
            activate_final=False,
        )

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if actor_architecture == "mlp":
            base_model_cls = partial(
                MLP,
                hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                activations=mish,
                use_layer_norm=actor_layer_norm,
                activate_final=False,
            )

            actor_def = DDPM(
                time_preprocess_cls=preprocess_time_cls,
                cond_encoder_cls=cond_model_cls,
                reverse_encoder_cls=base_model_cls,
            )
        else:
            raise ValueError(f"Invalid actor architecture: {actor_architecture}")

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis=0)
        actor_params = actor_def.init(actor_key, observations, actions, time)["params"]

        score_model = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_base_cls = partial(
            MLP, hidden_dims=critic_hidden_dims, activate_final=True
        )
        critic_def = StateActionValue(critic_base_cls)
        critic_key_1, critic_key_2 = jax.random.split(critic_key, 2)
        critic_params_1 = critic_def.init(critic_key_1, observations, actions)["params"]
        critic_params_2 = critic_def.init(critic_key_2, observations, actions)["params"]
        critic_1 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params_1,
            tx=optax.adam(learning_rate=critic_lr),
        )
        critic_2 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params_2,
            tx=optax.adam(learning_rate=critic_lr),
        )

        target_critic_def = StateActionValue(critic_base_cls)
        target_critic_1 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params_1,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        target_critic_2 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params_2,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        safety_base_cls = partial(
            MLP, hidden_dims=safety_hidden_dims, activate_final=True
        )
        safety_def = StateActionValue(safety_base_cls)
        safety_params = safety_def.init(safety_key, observations, actions)["params"]
        safety_critic = TrainState.create(
            apply_fn=safety_def.apply,
            params=safety_params,
            tx=optax.adam(learning_rate=safety_lr),
        )
        target_safety_critic = TrainState.create(
            apply_fn=safety_def.apply,
            params=safety_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        lambda_def = MLP(
            hidden_dims=tuple(lambda_hidden_dims) + (1,),
            activations=mish,
            activate_final=False,
        )
        lambda_params = lambda_def.init(lambda_key, observations, training=True)["params"]
        lambda_net = TrainState.create(
            apply_fn=lambda_def.apply,
            params=lambda_params,
            tx=optax.adam(learning_rate=lambda_lr),
        )

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[: i + 1]) for i in range(T)])

        if safety_tau is None:
            safety_tau = tau

        return cls(
            actor=None,
            score_model=score_model,
            critic_1=critic_1,
            critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            safety_critic=safety_critic,
            target_safety_critic=target_safety_critic,
            lambda_net=lambda_net,
            tau=tau,
            safety_tau=safety_tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            clip_sampler=clip_sampler,
            M_q=M_q,
            cost_limit=cost_limit,
            safety_discount=safety_discount,
            safety_lambda=safety_lambda,
            alpha_coef=alpha_coef,
            safety_threshold=safety_threshold,
            safety_grad_scale=safety_grad_scale,
            safe_lagrange_coef=safe_lagrange_coef,
            lambda_max=lambda_max,
            lambda_update_coef=lambda_update_coef,
            actor_grad_coef=actor_grad_coef,
            actor_safety_grad_coef=actor_safety_grad_coef,
            actor_grad_loss_coef=actor_grad_loss_coef,
            actor_aux_loss_coef=actor_aux_loss_coef,
        )

    def _lambda_values(self, observations: jnp.ndarray, params=None) -> jnp.ndarray:
        """Compute non-negative, clipped lambda values for each state."""
        raw = self.lambda_net.apply_fn(
            {"params": params if params is not None else self.lambda_net.params},
            observations,
            training=True,
        )
        lam = jnp.squeeze(nn.softplus(raw), axis=-1)
        lam = jnp.clip(lam, 0.0, self.lambda_max)
        return lam

    def update_q(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        agent = self
        (B, _) = batch["observations"].shape
        (_, A) = batch["actions"].shape

        key, rng = jax.random.split(agent.rng)
        next_actions, rng = ddpm_sampler(
            agent.score_model.apply_fn,
            agent.score_model.params,
            agent.T,
            rng,
            agent.act_dim,
            batch["next_observations"],
            agent.alphas,
            agent.alpha_hats,
            agent.betas,
            agent.ddpm_temperature,
            agent.clip_sampler,
        )
        key, rng = jax.random.split(rng, 2)
        noise = jax.random.normal(key, shape=next_actions.shape) * 0.1
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
        key, rng = jax.random.split(rng, 2)
        assert next_actions.shape == (B, A)

        key, rng = jax.random.split(rng)
        next_q_1 = agent.target_critic_1.apply_fn(
            {"params": agent.target_critic_1.params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        key, rng = jax.random.split(rng)
        next_q_2 = agent.target_critic_2.apply_fn(
            {"params": agent.target_critic_2.params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        next_v = jnp.stack([next_q_1, next_q_2], 0).min(0)
        target_q = batch["rewards"] + agent.discount * batch["not_terminated"] * next_v
        metrics = tensorstats(target_q, "target_q")
        assert target_q.shape == (B,)

        def critic_loss_fn(critic_params):
            q = agent.critic_1.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                training=True,
            )
            loss = (q - sg(target_q)) ** 2
            assert loss.shape == (B,)
            loss_mean = loss.mean()
            met = {**tensorstats(loss, "c_loss"), **tensorstats(q, "q")}
            return loss_mean, met

        grads_c_1, metrics_c_1 = jax.grad(critic_loss_fn, has_aux=True)(
            agent.critic_1.params
        )
        metrics.update({f"{k}_1": v for k, v in metrics_c_1.items()})
        critic_1 = agent.critic_1.apply_gradients(grads=grads_c_1)

        grads_c_2, metrics_c_2 = jax.grad(critic_loss_fn, has_aux=True)(
            agent.critic_2.params
        )
        metrics.update({f"{k}_2": v for k, v in metrics_c_2.items()})
        critic_2 = agent.critic_2.apply_gradients(grads=grads_c_2)

        target_critic_1_params = optax.incremental_update(
            critic_1.params, agent.target_critic_1.params, agent.tau
        )
        target_critic_2_params = optax.incremental_update(
            critic_2.params, agent.target_critic_2.params, agent.tau
        )
        target_critic_1 = agent.target_critic_1.replace(params=target_critic_1_params)
        target_critic_2 = agent.target_critic_2.replace(params=target_critic_2_params)

        new_agent = agent.replace(
            critic_1=critic_1,
            critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            rng=rng,
        )
        return new_agent, metrics

    def update_lambda(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Update the state-dependent multiplier lambda via gradient ascent."""
        agent = self
        (B, _) = batch["observations"].shape

        key, rng = jax.random.split(agent.rng)
        pi_actions, rng = ddpm_sampler(
            agent.score_model.apply_fn,
            agent.score_model.params,
            agent.T,
            rng,
            agent.act_dim,
            batch["observations"],
            agent.alphas,
            agent.alpha_hats,
            agent.betas,
            agent.ddpm_temperature,
            agent.clip_sampler,
        )
        key, rng = jax.random.split(rng, 2)
        noise = jax.random.normal(key, shape=pi_actions.shape) * 0.1
        pi_actions = jnp.clip(pi_actions + noise, -1.0, 1.0)

        qh = agent.target_safety_critic.apply_fn(
            {"params": agent.target_safety_critic.params},
            batch["observations"],
            pi_actions,
            training=True,
        )
        vh = jnp.maximum(0.0, qh)
        violation = jnp.maximum(0.0, vh - agent.safety_threshold)

        lam = self._lambda_values(batch["observations"])
        lam_sg = sg(lam)

        def lambda_loss_fn(lambda_params):
            lam_values = self._lambda_values(batch["observations"], params=lambda_params)
            j_lambda = (lam_values * sg(violation)).mean()
            loss = -agent.lambda_update_coef * j_lambda
            metrics = {
                "lambda_j": j_lambda,
                "lambda_loss": loss,
                **tensorstats(lam_values, "lambda"),
                "violation_mean": violation.mean(),
            }
            return loss, metrics

        grads, metrics = jax.grad(lambda_loss_fn, has_aux=True)(agent.lambda_net.params)
        lambda_net = agent.lambda_net.apply_gradients(grads=grads)

        new_agent = agent.replace(lambda_net=lambda_net, rng=rng)
        metrics.update(tensorstats(lam_sg, "lambda_eval"))
        metrics["vh_mean"] = vh.mean()
        return new_agent, metrics

    def _safety_targets(
        self,
        agent,
        batch: DatasetDict,
        rng: jax.random.PRNGKey,
    ):
        next_actions, rng = ddpm_sampler(
            agent.score_model.apply_fn,
            agent.score_model.params,
            agent.T,
            rng,
            agent.act_dim,
            batch["next_observations"],
            agent.alphas,
            agent.alpha_hats,
            agent.betas,
            agent.ddpm_temperature,
            agent.clip_sampler,
        )
        key, rng = jax.random.split(rng, 2)
        noise = jax.random.normal(key, shape=next_actions.shape) * 0.1
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
        next_qh = agent.target_safety_critic.apply_fn(
            {"params": agent.target_safety_critic.params},
            batch["next_observations"],
            next_actions,
            training=True,
        )
        next_vh = jnp.maximum(0.0, next_qh)

        current_qh = agent.safety_critic.apply_fn(
            {"params": agent.safety_critic.params},
            batch["observations"],
            batch["actions"],
            training=True,
        )
        current_vh = jnp.maximum(0.0, current_qh)

        alpha_term = agent.alpha_coef * current_vh
        candidate = agent.safety_discount * batch["not_terminated"] * next_vh - current_vh + alpha_term
        positive_candidate = jnp.maximum(0.0, candidate)

        stage_violation = jnp.maximum(0.0, batch["costs"] - agent.cost_limit)
        target = jnp.maximum(stage_violation, positive_candidate)
        return target, current_qh, rng

    def update_safety(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        agent = self
        rng = agent.rng
        safety_target, current_qh, rng = self._safety_targets(agent, batch, rng)

        def safety_loss_fn(params):
            qh_pred = agent.safety_critic.apply_fn(
                {"params": params},
                batch["observations"],
                batch["actions"],
                training=True,
            )
            relu_pred = jnp.maximum(0.0, qh_pred)
            diff = relu_pred - sg(safety_target)
            hinge = jnp.maximum(0.0, qh_pred - sg(safety_target))
            loss = diff ** 2 + agent.safety_lambda * hinge ** 2
            loss = loss.mean()
            metrics = {
                **tensorstats(relu_pred, "qh_relu"),
                **tensorstats(diff, "qh_diff"),
                "qh_target_mean": safety_target.mean(),
                "qh_loss": loss,
            }
            return loss, metrics

        grads, metrics = jax.grad(safety_loss_fn, has_aux=True)(
            agent.safety_critic.params
        )
        safety_critic = agent.safety_critic.apply_gradients(grads=grads)
        target_params = optax.incremental_update(
            safety_critic.params, agent.target_safety_critic.params, agent.safety_tau
        )
        target_safety_critic = agent.target_safety_critic.replace(params=target_params)

        metrics.update(tensorstats(current_qh, "qh_current"))
        metrics["qh_target_stage_mean"] = jnp.maximum(
            0.0, batch["costs"] - agent.cost_limit
        ).mean()

        new_agent = agent.replace(
            safety_critic=safety_critic,
            target_safety_critic=target_safety_critic,
            rng=rng,
        )
        return new_agent, metrics

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        agent = self
        B, A = batch["actions"].shape

        key, rng = jax.random.split(agent.rng, 2)
        time = jax.random.randint(key, (B,), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(key, (B, agent.act_dim))
        key, rng = jax.random.split(rng, 2)
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

        dropout_key, rng = jax.random.split(rng)
        critic_1_jacobian = jax.grad(
            lambda actions: agent.critic_1.apply_fn(
                {"params": agent.critic_1.params},
                batch["observations"],
                actions,
            ).sum()
        )(noisy_actions)
        assert critic_1_jacobian.shape == (B, A)
        critic_2_jacobian = jax.grad(
            lambda actions: agent.critic_2.apply_fn(
                {"params": agent.critic_2.params},
                batch["observations"],
                actions,
            ).sum()
        )(noisy_actions)
        assert critic_2_jacobian.shape == (B, A)
        critic_jacobian = jnp.stack([critic_1_jacobian, critic_2_jacobian], 0).mean(0)

        lam = agent._lambda_values(batch["observations"])
        lam_sg = sg(lam)

        safety_q = agent.safety_critic.apply_fn(
            {"params": agent.safety_critic.params},
            batch["observations"],
            noisy_actions,
            training=True,
        )
        safety_value = jnp.maximum(0.0, safety_q)

        safety_jacobian = jax.grad(
            lambda actions: agent.safety_critic.apply_fn(
                {"params": agent.safety_critic.params},
                batch["observations"],
                actions,
            ).sum()
        )(noisy_actions)
        assert safety_jacobian.shape == (B, A)

        # Soft Lagrangian blend between reward gradient and safety gradient using statewise lambda.
        phi = (
            agent.actor_grad_coef * agent.M_q * critic_jacobian
            - agent.actor_safety_grad_coef
            * (lam_sg[:, None] * agent.safety_grad_scale * safety_jacobian)
        )

        # phi = agent.M_q * critic_jacobian - (agent.safe_lagrange_coef * agent.safety_grad_scale) * safety_jacobian

        def actor_loss_fn(score_model_params):
            eps_pred = agent.score_model.apply_fn(
                {"params": score_model_params},
                batch["observations"],
                noisy_actions,
                time,
                rngs={"dropout": dropout_key},
                training=True,
            )
            assert eps_pred.shape == (B, A)
            target = - sg(phi)
            loss_gradmatch = jnp.square(target - eps_pred).mean(-1)

            # Reconstruct the denoised action x0_hat to evaluate safety value under the learned policy.
            x0_hat = (noisy_actions - alpha_2 * eps_pred) / alpha_1
            x0_hat = jnp.clip(x0_hat, -1.0, 1.0)
            qh_hat = agent.safety_critic.apply_fn(
                {"params": agent.safety_critic.params},
                batch["observations"],
                x0_hat,
                training=False,
            )
            vh_hat = jnp.maximum(0.0, qh_hat)
            violation_hat = jnp.maximum(0.0, vh_hat - agent.safety_threshold)
            loss_aux = (lam_sg * violation_hat).mean()

            actor_loss = (
                agent.actor_grad_loss_coef * loss_gradmatch.mean(0)
                + agent.actor_aux_loss_coef * loss_aux
            )
            metrics = tensorstats(loss_gradmatch, "actor_loss")
            metrics.update(tensorstats(eps_pred, "eps_pred"))
            metrics.update(tensorstats(phi, "phi"))
            metrics.update(tensorstats(critic_jacobian, "critic_jacobian"))
            metrics.update(tensorstats(safety_jacobian, "safety_jacobian"))
            metrics.update(tensorstats(lam_sg, "lambda"))
            metrics["safety_value_mean"] = safety_value.mean()
            metrics["vh_hat_mean"] = vh_hat.mean()
            metrics["violation_hat_mean"] = violation_hat.mean()
            metrics["loss_aux"] = loss_aux
            metrics["loss_gradmatch"] = loss_gradmatch.mean()
            return actor_loss, metrics

        key, rng = jax.random.split(rng, 2)
        grads, metrics = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        new_agent = agent.replace(
            score_model=score_model,
            rng=rng,
        )
        return new_agent, metrics

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray):
        actions, new_agent = self.eval_actions(observations)
        key, rng = jax.random.split(new_agent.rng, 2)
        noise = jax.random.normal(key, shape=actions.shape) * 0.1
        actions = jnp.clip(actions + noise, -1.0, 1.0)
        key, rng = jax.random.split(rng, 2)
        return actions, new_agent.replace(rng=rng)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng
        assert len(observations.shape) == 1
        observations = observations[None]

        actions, rng = ddpm_sampler(
            self.score_model.apply_fn,
            self.score_model.params,
            self.T,
            rng,
            self.act_dim,
            observations,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.clip_sampler,
        )
        assert actions.shape == (1, self.act_dim)
        _, rng = jax.random.split(rng, 2)
        return jnp.squeeze(actions), self.replace(rng=rng)

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        new_agent, critic_info = new_agent.update_q(batch)
        new_agent, safety_info = new_agent.update_safety(batch)
        new_agent, lambda_info = new_agent.update_lambda(batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, {**actor_info, **critic_info, **safety_info, **lambda_info}

    def save(self, path: str) -> None:
        """Serialize the learner to a file using Flax msgpack."""
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(self))

    @classmethod
    def load(cls, path: str):
        """Load a learner checkpoint from disk."""
        with open(path, "rb") as f:
            data = f.read()
        # Prefer msgpack_restore (available in newer Flax); fall back to from_bytes.
        if hasattr(serialization, "msgpack_restore"):
            restored = serialization.msgpack_restore(data)
        else:
            restored = serialization.from_bytes(cls, data)

        if isinstance(restored, cls):
            return restored
        if isinstance(restored, dict):
            for value in restored.values():
                if isinstance(value, cls):
                    return value
        raise TypeError(
            f"Loaded checkpoint type {type(restored)} is not {cls.__name__} or a container of it"
        )
