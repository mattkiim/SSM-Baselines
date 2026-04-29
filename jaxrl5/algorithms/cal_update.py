from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.utils.soft_update import soft_update
from jaxrl5.utils.ucb import ucb_from_ensemble


@struct.dataclass
class CALTrainState:
    actor: TrainState
    qr1: TrainState
    qr2: TrainState
    qc: TrainState
    qr1_target: TrainState
    qr2_target: TrainState
    qc_target: TrainState
    alpha: TrainState
    log_lambda: jnp.ndarray


@dataclass(frozen=True)
class CALUpdateConfig:
    gamma: float
    gamma_c: float
    tau: float
    k_ucb: float
    alm_c: float
    lambda_lr: float
    cost_limit: float
    target_entropy: float


def _get_not_terminated(batch):
    if "not_terminated" in batch:
        return batch["not_terminated"]
    if "not_dones" in batch:
        return batch["not_dones"]
    if "dones" in batch:
        return 1.0 - batch["dones"]
    raise KeyError("Batch must contain 'not_terminated', 'not_dones', or 'dones'.")


@partial(jax.jit, static_argnums=3)
def cal_update(
    rng: jax.Array,
    state: CALTrainState,
    batch,
    config: CALUpdateConfig,
) -> Tuple[jax.Array, CALTrainState, Dict[str, jnp.ndarray]]:
    """Performs a single Conservative Augmented Lagrangian update."""

    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    not_terminated = _get_not_terminated(batch)

    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    costs = batch["costs"]
    next_observations = batch["next_observations"]

    rng, key_next, key_actor = jax.random.split(rng, 3)

    # Sample next actions.
    next_dist = state.actor.apply_fn({"params": state.actor.params}, next_observations)
    next_actions = next_dist.sample(seed=key_next)
    next_logp = next_dist.log_prob(next_actions)

    alpha_val = state.alpha.apply_fn({"params": state.alpha.params})

    # Reward backups.
    q1_next = state.qr1_target.apply_fn(
        {"params": state.qr1_target.params}, next_observations, next_actions
    )
    q2_next = state.qr2_target.apply_fn(
        {"params": state.qr2_target.params}, next_observations, next_actions
    )
    q_next = jnp.minimum(q1_next, q2_next)
    target_q = rewards + config.gamma * not_terminated * (q_next - alpha_val * next_logp)
    target_q = jax.lax.stop_gradient(target_q)

    # Cost backups.
    qc_next = state.qc_target.apply_fn(
        {"params": state.qc_target.params}, next_observations, next_actions
    )
    y_c = costs[None, :] + config.gamma_c * not_terminated[None, :] * qc_next
    target_c = jax.lax.stop_gradient(y_c)

    # Online critics.
    q1 = state.qr1.apply_fn({"params": state.qr1.params}, observations, actions)
    q2 = state.qr2.apply_fn({"params": state.qr2.params}, observations, actions)
    qc_stack = state.qc.apply_fn(
        {"params": state.qc.params}, observations, actions
    )

    qc_ucb = ucb_from_ensemble(qc_stack, config.k_ucb)
    qc_ucb_flat = jnp.squeeze(qc_ucb, axis=-1)
    qc_ucb_mean = jnp.mean(qc_ucb_flat)
    qc_ucb_std = jnp.std(qc_ucb_flat)

    # lambda_val = jnp.exp(state.log_lambda)
    # tilde_lambda = jnp.maximum(
    #     0.0, lambda_val - config.alm_c * (config.cost_limit - qc_ucb_mean)
    # )

    lambda_val = jnp.exp(state.log_lambda)

    # rect = clamp(c * max(0, cost_limit - current_QC_mean), max=lambda)
    gap = jnp.maximum(0.0, config.cost_limit - qc_ucb_mean)
    rect = jnp.minimum(lambda_val, config.alm_c * gap)

    tilde_lambda = lambda_val - rect

    # Actor update.
    def actor_loss_fn(actor_params):
        dist = state.actor.apply_fn({"params": actor_params}, observations)
        actions_pi = dist.sample(seed=key_actor)
        logp = dist.log_prob(actions_pi)
        q1_pi = state.qr1.apply_fn(
            {"params": state.qr1.params}, observations, actions_pi
        )
        q2_pi = state.qr2.apply_fn(
            {"params": state.qr2.params}, observations, actions_pi
        )
        qc_pi_stack = state.qc.apply_fn(
            {"params": state.qc.params}, observations, actions_pi
        )
        qc_pi_ucb = jnp.squeeze(
            ucb_from_ensemble(qc_pi_stack, config.k_ucb), axis=-1
        )
        actor_loss = jnp.mean(
            alpha_val * logp - jnp.minimum(q1_pi, q2_pi) + tilde_lambda * qc_pi_ucb
        )
        entropy = -jnp.mean(logp)
        metrics = {
            "actor_loss": actor_loss,
            "entropy": entropy,
            "qc_pi_ucb": jnp.mean(qc_pi_ucb),
        }
        return actor_loss, (metrics, jnp.mean(logp))

    (actor_loss_val, (actor_metrics, logp_mean)), actor_grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(state.actor.params)
    actor = state.actor.apply_gradients(grads=actor_grads)

    # Alpha / temperature update.
    def alpha_loss_fn(alpha_params):
        alpha = state.alpha.apply_fn({"params": alpha_params})
        loss = alpha * (-logp_mean - config.target_entropy)
        return loss, {"alpha": alpha, "alpha_loss": loss}

    (alpha_loss_val, alpha_metrics), alpha_grads = jax.value_and_grad(
        alpha_loss_fn, has_aux=True
    )(state.alpha.params)
    alpha = state.alpha.apply_gradients(grads=alpha_grads)

    # Reward critic updates.
    def qr1_loss_fn(params):
        q = state.qr1.apply_fn({"params": params}, observations, actions)
        loss = jnp.mean((q - target_q) ** 2)
        return loss, loss

    (qr1_loss_val, _), qr1_grads = jax.value_and_grad(
        qr1_loss_fn, has_aux=True
    )(state.qr1.params)
    qr1 = state.qr1.apply_gradients(grads=qr1_grads)

    def qr2_loss_fn(params):
        q = state.qr2.apply_fn({"params": params}, observations, actions)
        loss = jnp.mean((q - target_q) ** 2)
        return loss, loss

    (qr2_loss_val, _), qr2_grads = jax.value_and_grad(
        qr2_loss_fn, has_aux=True
    )(state.qr2.params)
    qr2 = state.qr2.apply_gradients(grads=qr2_grads)

    # Cost critic update.
    def qc_loss_fn(params):
        qc = state.qc.apply_fn({"params": params}, observations, actions)
        loss = jnp.mean((qc - target_c) ** 2)
        return loss, loss

    (qc_loss_val, _), qc_grads = jax.value_and_grad(
        qc_loss_fn, has_aux=True
    )(state.qc.params)
    qc = state.qc.apply_gradients(grads=qc_grads)

    # Target updates.
    qr1_target = state.qr1_target.replace(
        params=soft_update(state.qr1_target.params, qr1.params, config.tau)
    )
    qr2_target = state.qr2_target.replace(
        params=soft_update(state.qr2_target.params, qr2.params, config.tau)
    )
    qc_target = state.qc_target.replace(
        params=soft_update(state.qc_target.params, qc.params, config.tau)
    )

    # Lambda update (projected gradient).
    # lambda_target = jnp.maximum(
    #     0.0, lambda_val - config.lambda_lr * (config.cost_limit - qc_ucb_mean)
    # )
    # lambda_target = jnp.maximum(lambda_target, 1e-6)
    # log_lambda = jnp.log(lambda_target)
    lambda_new = lambda_val + config.lambda_lr * (qc_ucb_mean - config.cost_limit)
    lambda_new = jnp.maximum(lambda_new, 1e-6)
    log_lambda = jnp.log(lambda_new)
    lambda_new_val = jnp.exp(log_lambda)

    total_critic_loss = 0.5 * (qr1_loss_val + qr2_loss_val)

    metrics = {
        "train/actor_loss": actor_loss_val,
        "train/q_reward_loss": total_critic_loss,
        "train/q_cost_loss": qc_loss_val,
        "train/alpha_loss": alpha_loss_val,
        "train/alpha": alpha_metrics["alpha"],
        "train/lambda": lambda_new_val,
        "train/tilde_lambda": tilde_lambda,
        "train/qc_ucb_mean": qc_ucb_mean,
        "train/qc_ucb_std": qc_ucb_std,
        "train/entropy": actor_metrics["entropy"],
    }

    new_state = state.replace(
        actor=actor,
        qr1=qr1,
        qr2=qr2,
        qc=qc,
        qr1_target=qr1_target,
        qr2_target=qr2_target,
        qc_target=qc_target,
        alpha=alpha,
        log_lambda=log_lambda,
    )

    return rng, new_state, metrics
