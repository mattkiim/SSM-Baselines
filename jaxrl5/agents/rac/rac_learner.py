"""Reachable Actor-Critic (RAC) learner built on SAC updates."""
from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax.serialization as serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.algorithms.reachability import reachability_target
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal
from jaxrl5.networks import Ensemble, MLP, StateActionValue, subsample_ensemble


# def compute_h_from_obs(obs: jnp.ndarray) -> jnp.ndarray:
#     """Compute violation measure h(s) from observation.

#     h(s) = max(0.5 - z, z - 1.5) with optional out-of-bounds penalty.
#     Supports obs shape [obs_dim] or [B, obs_dim].
#     """
#     obs = jnp.asarray(obs)
#     single = obs.ndim == 1
#     if single:
#         obs = obs[None]

#     x = obs[:, 0]
#     z = obs[:, 2]
#     h = jnp.maximum(0.5 - z, z - 1.5)
#     out_of_bounds = (jnp.abs(x) > 2.0) | (jnp.abs(z) > 3.0)
#     h = jnp.where(out_of_bounds, jnp.maximum(h, 1.0), h)

#     if single:
#         return jnp.squeeze(h, axis=0)
#     return h

# def compute_h_from_obs(obs: jnp.ndarray) -> jnp.ndarray:
#     obs = jnp.asarray(obs)
#     single = obs.ndim == 1
#     if single:
#         obs = obs[None]

#     x = obs[:, 0]
#     z = obs[:, 2]
#     violated = (z < 0.5) | (z > 1.5) | (jnp.abs(x) > 2.0) | (jnp.abs(z) > 3.0)
#     h = jnp.where(violated, 1.0, -1.0).astype(jnp.float32)

#     if single:
#         return jnp.squeeze(h, axis=0)
#     return h

# def compute_h_from_obs(obs: jnp.ndarray) -> jnp.ndarray:
#     """Continuous Quad3D safety signal from observation.

#     Assumes obs = [state(9), ref(9)] and state =
#     [px, py, pz, vx, vy, vz, phi, theta, psi].

#     Safe set inspired by Dawson's Quad3D:
#       - floor constraint: pz <= 0.0
#       - radius constraint: ||position|| <= 3.0

#     Returns h(s) where h <= 0 is safe and h > 0 is unsafe.
#     """
#     obs = jnp.asarray(obs)
#     single = obs.ndim == 1
#     if single:
#         obs = obs[None]

#     px = obs[:, 0]
#     py = obs[:, 1]
#     pz = obs[:, 2]

#     radius = jnp.sqrt(px**2 + py**2 + pz**2)

#     h_floor = pz - 0.0
#     h_radius = radius - 3.0
#     h = jnp.maximum(h_floor, h_radius)

#     if single:
#         return jnp.squeeze(h, axis=0)
#     return h

SAFETY_H_MODE_QUAD2D = "quad2d"
SAFETY_H_MODE_QUAD2D_STAB_V2 = "quad2d_stab_v2"
SAFETY_H_MODE_QUAD2D_STAB_V3 = "quad2d_stab_v3"
SAFETY_H_MODE_QUAD3D = "quad3d"
SAFETY_H_MODE_F16_TASK_FEATURE = "f16_task_feature"
# Distance-to-nearest-hazard margin for SafetyCarGoal-v0 (Car robot), reconstructed
# from the hazards_lidar sensor. Only valid for this exact robot+task observation
# layout -- see compute_h_from_obs for the flattened index offsets it assumes.
SAFETY_H_MODE_CARGOAL_LIDAR = "cargoal_lidar"
# Use the environment's own per-transition `cost` as the safety signal instead
# of a hand-crafted geometric h(s). For envs (e.g. Safety-Gymnasium tasks) whose
# safety constraint is not recoverable from a single observation, this mode
# learns cumulative costs. reachability_transition instead accepts an exact
# signed transition margin from the environment through replay.
SAFETY_H_MODE_ENV_COST = "env_cost"
SAFETY_H_MODE_TRANSITION = "reachability_transition"
SAFETY_H_MODES = (
    SAFETY_H_MODE_QUAD2D,
    SAFETY_H_MODE_QUAD2D_STAB_V2,
    SAFETY_H_MODE_QUAD2D_STAB_V3,
    SAFETY_H_MODE_QUAD3D,
    SAFETY_H_MODE_F16_TASK_FEATURE,
    SAFETY_H_MODE_CARGOAL_LIDAR,
    SAFETY_H_MODE_ENV_COST,
    SAFETY_H_MODE_TRANSITION,
)


def _quad2d_stab_rect_h(
    x: jnp.ndarray,
    z: jnp.ndarray,
    cx: float,
    cz: float,
    sx: float,
    sz: float,
    margin: float,
) -> jnp.ndarray:
    hx = 0.5 * sx + margin
    hz = 0.5 * sz + margin
    qx = jnp.abs(x - cx) - hx
    qz = jnp.abs(z - cz) - hz
    outside = jnp.sqrt(jnp.square(jnp.maximum(qx, 0.0)) + jnp.square(jnp.maximum(qz, 0.0)))
    inside = jnp.minimum(jnp.maximum(qx, qz), 0.0)
    return -(outside + inside)


def _quad2d_stab_h_from_xz(x: jnp.ndarray, z: jnp.ndarray, layout: str) -> jnp.ndarray:
    if layout == "corridor_v3":
        rects = (
            (0.0, 0.30, 1.30, 0.30),
            (-1.72, 0.30, 0.32, 0.32),
            (1.72, 0.30, 0.32, 0.32),
            (0.0, 1.55, 0.34, 0.74),
        )
    else:
        rects = (
            (0.0, 0.30, 2.00, 0.30),
            (-1.84, 0.30, 0.44, 0.44),
            (1.84, 0.30, 0.44, 0.44),
            (0.0, 1.55, 0.38, 0.86),
        )
    h = _quad2d_stab_rect_h(x, z, *rects[0], margin=0.10)
    for rect in rects[1:]:
        h = jnp.maximum(h, _quad2d_stab_rect_h(x, z, *rect, margin=0.10))
    h_boundary = jnp.maximum(
        jnp.maximum(-2.78 + 0.10 - x, x - (2.78 - 0.10)),
        jnp.maximum(-1.55 + 0.10 - z, z - (3.10 - 0.10)),
    )
    return jnp.maximum(h, h_boundary)


def compute_h_from_obs(obs: jnp.ndarray, safety_h_mode: str) -> jnp.ndarray:
    obs = jnp.asarray(obs)
    single = obs.ndim == 1
    if single:
        obs = obs[None]

    if safety_h_mode == SAFETY_H_MODE_F16_TASK_FEATURE:
        # F16 observations are encoded as [state_enc(24), task_feats], where
        # the first task feature is tanh(h_margin).
        h_tanh = jnp.clip(obs[:, 24], -0.999, 0.999)
        h = jnp.arctanh(h_tanh)
    elif safety_h_mode == SAFETY_H_MODE_QUAD2D:
        z = obs[:, 2]
        h = jnp.maximum(0.5 - z, z - 1.5)
    elif safety_h_mode == SAFETY_H_MODE_QUAD2D_STAB_V2:
        h = _quad2d_stab_h_from_xz(obs[:, 0], obs[:, 2], "corridor_v2")
    elif safety_h_mode == SAFETY_H_MODE_QUAD2D_STAB_V3:
        h = _quad2d_stab_h_from_xz(obs[:, 0], obs[:, 2], "corridor_v3")
    elif safety_h_mode == SAFETY_H_MODE_QUAD3D:
        pz = obs[:, 2]
        state_norm = jnp.linalg.norm(obs[:, :9], axis=-1)

        h_floor = pz - 0.0
        h_radius = state_norm - 3.0
        h = jnp.maximum(h_floor, h_radius)
    elif safety_h_mode == SAFETY_H_MODE_CARGOAL_LIDAR:
        # SafetyCarGoal-v0 (Car robot) flattened obs layout (72D):
        #   accelerometer(3) velocimeter(3) gyro(3) magnetometer(3)
        #   ballangvel_rear(3) ballquat_rear(9) goal_lidar(16)
        #   hazards_lidar(16) vases_lidar(16)
        # Safety-Gymnasium's pseudo-lidar encodes each bin as
        # (max_dist - dist) / max_dist, so the closest hazard in any
        # direction is recovered from the single largest bin reading.
        # These constants match SafetyCarGoal1-v0's registered defaults
        # (LidarConf.max_dist=3, Hazards.size=0.2); verified against the
        # env's ground-truth hazard distance to be sign-exact.
        lidar_max_dist = 3.0
        hazard_size = 0.2
        hazards_lidar = obs[:, 40:56]
        closest_sensor = jnp.max(hazards_lidar, axis=-1)
        dist_to_nearest_hazard = lidar_max_dist * (1.0 - closest_sensor)
        h = hazard_size - dist_to_nearest_hazard
    else:
        raise ValueError(
            f"Unsupported safety_h_mode={safety_h_mode!r}. "
            f"Expected one of {SAFETY_H_MODES}."
        )

    if single:
        return jnp.squeeze(h, axis=0)
    return h


class RACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    safety_critic: TrainState
    target_safety_critic: TrainState
    lambda_net: TrainState
    temp: TrainState

    tau: float
    discount: float
    safety_discount: float
    safety_tau: float
    target_entropy: float
    lambda_max: float
    safety_threshold: float
    update_step: jnp.ndarray
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    policy_update_period: int = struct.field(pytree_node=False)
    multiplier_update_period: int = struct.field(pytree_node=False)
    safety_h_mode: str = struct.field(pytree_node=False, default=SAFETY_H_MODE_QUAD3D)

    reference_protocol: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_dims: Sequence[int] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        safety_lr: float = 3e-4,
        lambda_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        safety_discount: float = 0.99,
        safety_tau: Optional[float] = None,
        target_entropy: Optional[float] = None,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        lambda_max: float = 100.0,
        safety_threshold: float = 0.0,
        safety_h_mode: str = SAFETY_H_MODE_QUAD3D,
        policy_update_period: int = 1,
        multiplier_update_period: int = 1,
        init_temperature: float = 1.0,
        reference_protocol: bool = False,
        lr_decay_updates: int = 2000000,
    ) -> "RACLearner":
        if safety_h_mode not in SAFETY_H_MODES:
            raise ValueError(
                f"Unsupported safety_h_mode={safety_h_mode!r}. "
                f"Expected one of {SAFETY_H_MODES}."
            )

        if reference_protocol and safety_h_mode != SAFETY_H_MODE_TRANSITION:
            raise ValueError("Reference protocol requires transition reachability")

        def optimizer(lr, end_lr, period=1, clip_norm=10., ensemble=False):
            if not reference_protocol:
                return optax.adam(lr)
            schedule = optax.linear_schedule(lr, end_lr, max(1, lr_decay_updates // period))
            clip = optax.clip_by_global_norm(clip_norm)
            if ensemble:
                # Reference clips the two reward critics independently.
                clip = optax.GradientTransformation(
                    lambda _: optax.EmptyState(),
                    lambda updates, state, params=None: (
                        jax.vmap(lambda g: optax.clip_by_global_norm(clip_norm).update(
                            g, optax.EmptyState())[0])(updates), state))
            return optax.chain(clip, optax.adam(schedule, eps=1e-7))

        network_kwargs = dict(activations=nn.elu, kernel_init=nn.initializers.he_normal()) if reference_protocol else {}
        output_kwargs = dict(kernel_init=nn.initializers.he_normal()) if reference_protocol else {}
        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -float(action_dim)
        if safety_tau is None:
            safety_tau = tau

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, safety_key, lambda_key, temp_key = jax.random.split(
            rng, 6
        )

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True, **network_kwargs)
        actor_def = TanhNormal(actor_base_cls, action_dim, **output_kwargs,
                               **(dict(log_std_min=-5., log_std_max=1.) if reference_protocol else {}))
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optimizer(actor_lr, 1e-6, policy_update_period),
        )

        critic_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True, **network_kwargs)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls, **output_kwargs)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optimizer(critic_lr, 1e-6, ensemble=True),
        )
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        safety_def = StateActionValue(critic_base_cls, **output_kwargs)
        safety_params = safety_def.init(safety_key, observations, actions)["params"]
        safety_critic = TrainState.create(
            apply_fn=safety_def.apply,
            params=safety_params,
            tx=optimizer(safety_lr, 1e-6),
        )
        target_safety_critic = TrainState.create(
            apply_fn=safety_def.apply,
            params=safety_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        lambda_def = MLP(hidden_dims=tuple(hidden_dims) + (1,), activate_final=False, **network_kwargs)
        lambda_params = lambda_def.init(lambda_key, observations)["params"]
        lambda_net = TrainState.create(
            apply_fn=lambda_def.apply,
            params=lambda_params,
            tx=optimizer(lambda_lr, 1e-7, multiplier_update_period, clip_norm=3.),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optimizer(alpha_lr, 3e-6, policy_update_period),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            safety_critic=safety_critic,
            target_safety_critic=target_safety_critic,
            lambda_net=lambda_net,
            temp=temp,
            tau=tau,
            discount=discount,
            safety_discount=safety_discount,
            safety_tau=safety_tau,
            target_entropy=target_entropy,
            lambda_max=lambda_max,
            safety_threshold=safety_threshold,
            safety_h_mode=safety_h_mode,
            reference_protocol=reference_protocol,
            update_step=jnp.array(0, dtype=jnp.int32),
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            policy_update_period=policy_update_period,
            multiplier_update_period=multiplier_update_period,
        )

    def _lambda_values(self, observations: jnp.ndarray, params=None) -> jnp.ndarray:
        raw = self.lambda_net.apply_fn(
            {"params": params if params is not None else self.lambda_net.params},
            observations,
        )
        lam = jnp.squeeze(jax.nn.softplus(raw), axis=-1)
        return lam if self.reference_protocol else jnp.clip(lam, 0.0, self.lambda_max)

    def _format_obs(self, observations: jnp.ndarray):
        if observations.ndim == 1:
            return observations[None], True
        return observations, False

    # def eval_actions(self, observations: jnp.ndarray):
    #     obs, single = self._format_obs(observations)
    #     dist = self.actor.apply_fn({"params": self.actor.params}, obs)
    #     actions = dist.mode()
    #     if single:
    #         actions = jnp.squeeze(actions, axis=0)
    #     return actions, self

    # def sample_actions(self, observations: jnp.ndarray):
    #     obs, single = self._format_obs(observations)
    #     key, rng = jax.random.split(self.rng)
    #     dist = self.actor.apply_fn({"params": self.actor.params}, obs)
    #     actions = dist.sample(seed=key)
    #     if single:
    #         actions = jnp.squeeze(actions, axis=0)
    #     return actions, self.replace(rng=rng)
    
    @staticmethod
    @jax.jit
    def _eval_actions_jit(agent, observations):
        obs = jnp.atleast_2d(observations)
        dist = agent.actor.apply_fn({"params": agent.actor.params}, obs)
        actions = dist.mode()
        return jnp.squeeze(actions, axis=0), agent

    @staticmethod
    @jax.jit
    def _sample_actions_jit(agent, observations):
        obs = jnp.atleast_2d(observations)
        key, rng = jax.random.split(agent.rng)
        dist = agent.actor.apply_fn({"params": agent.actor.params}, obs)
        actions = dist.sample(seed=key)
        return jnp.squeeze(actions, axis=0), agent.replace(rng=rng)

    def eval_actions(self, observations: jnp.ndarray):
        return self._eval_actions_jit(self, jnp.asarray(observations, dtype=jnp.float32))

    def sample_actions(self, observations: jnp.ndarray):
        return self._sample_actions_jit(self, jnp.asarray(observations, dtype=jnp.float32))

    def update_critic(self, batch: DatasetDict) -> Tuple["RACLearner", Dict[str, float]]:
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

        target_q = batch["rewards"] + self.discount * batch["not_terminated"] * (
            next_q - self.temp.apply_fn({"params": self.temp.params}) * next_log_probs
        )
        
        # jax.debug.print("reward mean: {}", batch["rewards"].mean())
        # jax.debug.print("target_q mean: {}", target_q.mean())
        # jax.debug.print("not_terminated mean: {}", batch["not_terminated"].mean())
        # quit()
        
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
            # Mean over two critics equals the sum of their half-MSE losses.
            return loss, {
                "critic_loss": loss,
                "q1_mean": qs[0].mean(),
                "q2_mean": qs[1].mean() if qs.shape[0] > 1 else qs[0].mean(),
                "target_q_mean": target_q.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params,
            jnp.where(self.update_step % self.policy_update_period == 0, self.tau, 0.)
            if self.reference_protocol else self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update_safety_critic(
        self, batch: DatasetDict
    ) -> Tuple["RACLearner", Dict[str, float]]:
        rng = self.rng

        dist = self.actor.apply_fn({"params": self.actor.params}, batch["next_observations"])
        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        key, rng = jax.random.split(rng)
        qh_next = self.target_safety_critic.apply_fn(
            {"params": self.target_safety_critic.params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        not_terminated = jnp.asarray(batch["not_terminated"], dtype=jnp.float32)

        if self.safety_h_mode == SAFETY_H_MODE_TRANSITION:
            h = jnp.asarray(batch['safety_h'], dtype=jnp.float32)
            if self.reference_protocol:
                h = 20. * jnp.sign(h)
            target_qh = reachability_target(h, qh_next, not_terminated, self.safety_discount)
            h_mean = h.mean()
        elif self.safety_h_mode == SAFETY_H_MODE_ENV_COST:
            # `batch["costs"]` is the environment's own per-transition cost,
            # already aligned with (observations, actions) -> next_observations
            # exactly like batch["rewards"]. Bellman target is a plain
            # discounted expected-future-cost sum (no reachability max, no
            # geometric h(s) needed/available).
            costs = jnp.asarray(batch["costs"], dtype=jnp.float32)
            target_qh = costs + self.safety_discount * not_terminated * qh_next
            h_mean = costs.mean()
        else:
            h = compute_h_from_obs(batch["observations"], self.safety_h_mode)
            h_next = compute_h_from_obs(batch["next_observations"], self.safety_h_mode)
            h_step = jnp.maximum(h, h_next)
            qh_bootstrap = not_terminated * qh_next + (1.0 - not_terminated) * h_step
            target_qh = (1.0 - self.safety_discount) * h_step + self.safety_discount * jnp.maximum(
                h_step, qh_bootstrap
            )
            h_mean = h.mean()

        key, rng = jax.random.split(rng)

        def safety_loss_fn(params):
            qh = self.safety_critic.apply_fn(
                {"params": params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )
            loss = 0.5 * jnp.square(qh - jax.lax.stop_gradient(target_qh)).mean()
            return loss, {
                "safety_critic_loss": loss,
                "qh_mean": qh.mean(),
                "qh_min": qh.min(),
                "qh_max": qh.max(),
                "target_qh_mean": target_qh.mean(),
                "h_mean": h_mean,
            }

        grads, info = jax.grad(safety_loss_fn, has_aux=True)(self.safety_critic.params)
        info["safety_gradient_norm"] = optax.global_norm(grads)
        safety_critic = self.safety_critic.apply_gradients(grads=grads)
        target_params = optax.incremental_update(
            safety_critic.params, self.target_safety_critic.params,
            jnp.where(self.update_step % self.policy_update_period == 0, self.safety_tau, 0.)
            if self.reference_protocol else self.safety_tau
        )
        target_safety_critic = self.target_safety_critic.replace(params=target_params)

        return (
            self.replace(
                safety_critic=safety_critic,
                target_safety_critic=target_safety_critic,
                rng=rng,
            ),
            info,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple["RACLearner", Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        key3, rng = jax.random.split(rng)

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
            qh = self.safety_critic.apply_fn(
                {"params": self.safety_critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key3},
            )
            lam = jax.lax.stop_gradient(self._lambda_values(batch["observations"]))
            alpha = self.temp.apply_fn({"params": self.temp.params})
            actor_loss = (alpha * log_probs - q + lam * qh).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": -log_probs.mean(),
                "logp_mean": log_probs.mean(),
                "action_saturation_fraction": jnp.mean(jnp.abs(actions) > .99),
                "alpha": alpha,
            }

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), info

    def update_temperature(self, entropy: jnp.ndarray) -> Tuple["RACLearner", Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            coefficient = temp_params["log_temp"] if self.reference_protocol else temperature
            temp_loss = coefficient * jax.lax.stop_gradient(entropy - self.target_entropy).mean()
            return temp_loss, {
                "alpha": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_multiplier(self, batch: DatasetDict) -> Tuple["RACLearner", Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        dist = self.actor.apply_fn({"params": self.actor.params}, batch["observations"])
        actions = dist.sample(seed=key)
        qh = self.safety_critic.apply_fn(
            {"params": self.safety_critic.params},
            batch["observations"],
            actions,
            True,
            rngs={"dropout": key2},
        )
        qh_term = qh - self.safety_threshold
        if self.reference_protocol:
            qh_term = jnp.clip(qh_term, -10., 100.)

        def lambda_loss_fn(params):
            lam = self._lambda_values(batch["observations"], params=params)
            loss = -(lam * jax.lax.stop_gradient(qh_term)).mean()
            return loss, {
                "lambda_mean": lam.mean(),
                "lambda_max": jnp.asarray(self.lambda_max),
            }

        grads, info = jax.grad(lambda_loss_fn, has_aux=True)(self.lambda_net.params)
        lambda_net = self.lambda_net.apply_gradients(grads=grads)

        return self.replace(lambda_net=lambda_net, rng=rng), info

    def _zeros_like_metrics(self) -> Dict[str, jnp.ndarray]:
        # 固定结构的“空指标”，用于 lax.cond 的 skip 分支
        z = lambda: jnp.array(0.0, dtype=jnp.float32)
        alpha = self.temp.apply_fn({"params": self.temp.params}).astype(jnp.float32)
        return {
            # critic
            "critic_loss": z(),
            "q1_mean": z(),
            "q2_mean": z(),
            "target_q_mean": z(),
            # safety critic
            "safety_critic_loss": z(),
            "qh_mean": z(),
            "target_qh_mean": z(),
            "h_mean": z(),
            # actor
            "actor_loss": z(),
            "entropy": z(),
            "logp_mean": z(),
            "alpha": alpha,  # 这个保留真实 alpha，避免全是 0
            # temperature
            "temperature_loss": z(),
            # lambda
            "lambda_mean": z(),
            "lambda_max": jnp.asarray(self.lambda_max, dtype=jnp.float32),
        }

    @staticmethod
    @jax.jit
    def _update_jit(agent: "RACLearner", batch: DatasetDict) -> Tuple["RACLearner", Dict[str, jnp.ndarray]]:
        # 1) step++
        step = agent.update_step + jnp.array(1, dtype=jnp.int32)
        agent = agent.replace(update_step=step)

        snapshot = agent

        # 2) always update critics
        agent, critic_info = agent.update_critic(batch)
        agent, safety_info = agent.update_safety_critic(batch)

        # 3) policy (actor + temp) periodic update via lax.cond
        zeros = agent._zeros_like_metrics()

        def do_policy(a: "RACLearner"):
            current = a
            if a.reference_protocol:
                a = snapshot.replace(rng=a.rng)
            a, actor_info = a.update_actor(batch)
            a, temp_info = a.update_temperature(actor_info["entropy"])
            if a.reference_protocol:
                a = current.replace(actor=a.actor, temp=a.temp, rng=a.rng)
            return a, {
                "actor_loss": actor_info["actor_loss"],
                "entropy": actor_info["entropy"],
                "logp_mean": actor_info["logp_mean"],
                "action_saturation_fraction": actor_info["action_saturation_fraction"],
                "alpha": temp_info["alpha"],
                "temperature_loss": temp_info["temperature_loss"],
            }

        def skip_policy(a: "RACLearner"):
            return a, {
                "actor_loss": jnp.array(0.0, dtype=jnp.float32),
                "entropy": jnp.array(0.0, dtype=jnp.float32),
                "logp_mean": jnp.array(0.0, dtype=jnp.float32),
                "action_saturation_fraction": jnp.array(0.0, dtype=jnp.float32),
                "alpha": a.temp.apply_fn({"params": a.temp.params}).astype(jnp.float32),
                "temperature_loss": jnp.array(0.0, dtype=jnp.float32),
            }

        do_pol = (step % agent.policy_update_period) == 0
        agent, pol_info = jax.lax.cond(do_pol, do_policy, skip_policy, agent)

        # 4) multiplier periodic update via lax.cond
        def do_lambda(a: "RACLearner"):
            current = a
            if a.reference_protocol:
                a = snapshot.replace(rng=a.rng)
            a, mul_info = a.update_multiplier(batch)
            if a.reference_protocol:
                a = current.replace(lambda_net=a.lambda_net, rng=a.rng)
            return a, {
                "lambda_mean": mul_info["lambda_mean"],
                "lambda_max": mul_info["lambda_max"],
            }

        def skip_lambda(a: "RACLearner"):
            return a, {
                "lambda_mean": jnp.array(0.0, dtype=jnp.float32),
                "lambda_max": jnp.asarray(a.lambda_max, dtype=jnp.float32),
            }

        do_lam = (step % agent.multiplier_update_period) == 0
        agent, lam_info = jax.lax.cond(do_lam, do_lambda, skip_lambda, agent)

        # 5) merge metrics (固定 key，不会因为分支变化导致 jit 不稳定)
        metrics = dict(zeros)
        metrics.update(critic_info)
        metrics.update(safety_info)

        # policy info / lambda info 只覆盖它们负责的字段（其余保持 zeros）
        metrics.update(pol_info)
        metrics.update(lam_info)

        metrics["actor_updated"] = do_pol.astype(jnp.float32)
        metrics["multiplier_updated"] = do_lam.astype(jnp.float32)
        metrics["learner_update"] = step.astype(jnp.float32)
        metrics["violation_mean"] = jnp.asarray(batch["costs"]).mean().astype(jnp.float32)
        return agent, metrics

    def update(self, batch: DatasetDict) -> Tuple["RACLearner", Dict[str, jnp.ndarray]]:
        return self._update_jit(self, batch)

    def save(self, path: str) -> None:
        """Serialize the learner to a file using Flax msgpack."""
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(self))

    @classmethod
    def load(cls, path: str) -> "RACLearner":
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
