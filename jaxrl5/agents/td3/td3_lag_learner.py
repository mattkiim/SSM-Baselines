"""TD3-Lag learner with reward and cost critics."""

import glob
import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhDeterministic
from jaxrl5.networks import MLP, Ensemble, StateActionValue, subsample_ensemble


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(
    rng, apply_fn, params, observations: np.ndarray, action_noise: float
) -> np.ndarray:
    key, rng = jax.random.split(rng)
    actions = apply_fn({"params": params}, observations)
    noise = jax.random.normal(key, shape=actions.shape) * action_noise
    actions = actions + noise
    return jnp.clip(actions, -1.0, 1.0), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    return apply_fn({"params": params}, observations)


def _iter_array_leaves_with_paths(tree) -> list[Tuple[str, np.ndarray]]:
    """Traverse a pytree and collect array leaves with their string paths."""

    leaves = []

    def _walk(node, path):
        if isinstance(node, (dict, FrozenDict)):
            for key, value in node.items():
                key_str = str(key)
                new_path = key_str if path == "" else f"{path}/{key_str}"
                _walk(value, new_path)
            return
        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                key_str = str(idx)
                new_path = key_str if path == "" else f"{path}/{key_str}"
                _walk(value, new_path)
            return

        arr = None
        if isinstance(node, (np.ndarray, jnp.ndarray)) or isinstance(node, jax.Array):
            arr = np.asarray(node)
        if arr is not None and hasattr(arr, "ndim"):
            leaves.append((path, arr))

    _walk(tree, "")
    return leaves


class TD3LagLearner(Agent):
    critic: TrainState
    cost_critic: TrainState
    target_critic: TrainState
    target_cost_critic: TrainState
    target_actor: TrainState
    tau: float
    discount: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    exploration_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_policy_noise_clip: float = 0.5
    actor_delay: int = 2
    lagrangian_lambda: jnp.ndarray = jnp.array(0.0)
    lambda_lr: float = struct.field(pytree_node=False, default=1e-3)
    cost_limit: float = struct.field(pytree_node=False, default=0.0)
    lambda_max: Optional[float] = struct.field(pytree_node=False, default=1000.0)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_policy_noise_clip: float = 0.5,
        actor_delay: int = 2,
        cost_limit: float = 0.0,
        lambda_lr: float = 1e-3,
        lambda_init: float = 0.0,
        lambda_max: Optional[float] = 1000.0,
    ):
        """Create a TD3-Lag learner with reward and cost critics."""

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, cost_critic_key = jax.random.split(rng, 4)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_def = TanhDeterministic(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )
        target_actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
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

        cost_critic_params = critic_def.init(cost_critic_key, observations, actions)[
            "params"
        ]
        cost_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=cost_critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_cost_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=cost_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            cost_critic=cost_critic,
            target_critic=target_critic,
            target_cost_critic=target_cost_critic,
            target_actor=target_actor,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            exploration_noise=exploration_noise,
            target_policy_noise=target_policy_noise,
            target_policy_noise_clip=target_policy_noise_clip,
            actor_delay=actor_delay,
            lagrangian_lambda=jnp.array(lambda_init, dtype=jnp.float32),
            lambda_lr=lambda_lr,
            cost_limit=cost_limit,
            lambda_max=lambda_max,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng, num=2)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            actions = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key},
            )
            q = qs.mean(axis=0)
            cqs = self.cost_critic.apply_fn(
                {"params": self.cost_critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key},
            )
            cq = cqs.mean(axis=0)
            actor_loss = -(q - self.lagrangian_lambda * cq).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "q_pi": q.mean(),
                "cq_pi": cq.mean(),
                "lambda_value": self.lagrangian_lambda,
            }

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        target_actor_params = optax.incremental_update(
            actor.params, self.target_actor.params, self.tau
        )
        target_actor = self.target_actor.replace(params=target_actor_params)

        return self.replace(actor=actor, target_actor=target_actor, rng=rng), actor_info

    def update_lambda(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng, num=2)
        actions = self.actor.apply_fn({"params": self.actor.params}, batch["observations"])
        cqs = self.cost_critic.apply_fn(
            {"params": self.cost_critic.params},
            batch["observations"],
            actions,
            True,
            rngs={"dropout": key},
        )
        cq = cqs.mean()
        lambda_update = self.lambda_lr * (cq - self.cost_limit)
        if self.lambda_max is None:
            new_lambda = jnp.maximum(0.0, self.lagrangian_lambda + lambda_update)
        else:
            new_lambda = jnp.clip(
                self.lagrangian_lambda + lambda_update, 0.0, self.lambda_max
            )
        info = {
            "lambda_value": new_lambda,
            "lambda_update": lambda_update,
            "c_est": cq,
            "cost_limit": self.cost_limit,
        }
        return self.replace(lagrangian_lambda=new_lambda, rng=rng), info

    def update_critic(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_actions = self.target_actor.apply_fn(
            {"params": self.target_actor.params}, batch["next_observations"]
        )

        rng = self.rng
        key, rng = jax.random.split(rng)
        target_noise = jax.random.normal(key, next_actions.shape) * self.target_policy_noise
        target_noise = target_noise.clip(
            -self.target_policy_noise_clip, self.target_policy_noise_clip
        )
        next_actions += target_noise

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

        key, rng = jax.random.split(rng)
        target_cost_params = subsample_ensemble(
            key, self.target_cost_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_cqs = self.target_cost_critic.apply_fn(
            {"params": target_cost_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        next_cq = next_cqs.min(axis=0)

        target_cq = batch["costs"] + self.discount * batch["not_terminated"] * next_cq

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        cost_key, rng = jax.random.split(rng)

        def cost_critic_loss_fn(cost_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            cqs = self.cost_critic.apply_fn(
                {"params": cost_critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": cost_key},
            )
            cost_critic_loss = ((cqs - target_cq) ** 2).mean()
            return cost_critic_loss, {
                "cost_critic_loss": cost_critic_loss,
                "cq": cqs.mean(),
            }

        cost_grads, cost_info = jax.grad(cost_critic_loss_fn, has_aux=True)(
            self.cost_critic.params
        )
        cost_critic = self.cost_critic.apply_gradients(grads=cost_grads)

        target_cost_critic_params = optax.incremental_update(
            cost_critic.params, self.target_cost_critic.params, self.tau
        )
        target_cost_critic = self.target_cost_critic.replace(
            params=target_cost_critic_params
        )

        new_agent = self.replace(
            critic=critic,
            target_critic=target_critic,
            cost_critic=cost_critic,
            target_cost_critic=target_cost_critic,
            rng=rng,
        )

        return new_agent, {**info, **cost_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        new_agent = self
        actor_info: Dict[str, float] = {}
        lambda_info: Dict[str, float] = {}
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        true_steps = new_agent.critic.step / utd_ratio

        def _actor_and_lambda(agent):
            updated_agent, a_info = agent.update_actor(mini_batch)
            updated_agent, l_info = updated_agent.update_lambda(mini_batch)
            return updated_agent, {**a_info, **l_info}

        def _skip_actor(agent):
            return agent, {
                "actor_loss": 0.0,
                "q_pi": 0.0,
                "cq_pi": 0.0,
                "lambda_value": agent.lagrangian_lambda,
                "lambda_update": 0.0,
                "c_est": 0.0,
                "cost_limit": agent.cost_limit,
            }

        new_agent, actor_lambda_info = jax.lax.cond(
            true_steps % new_agent.actor_delay == 0,
            _actor_and_lambda,
            _skip_actor,
            new_agent,
        )

        combined_info = {**critic_info, **actor_lambda_info}
        return new_agent, combined_info

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions), self

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            self.exploration_noise,
        )
        return np.asarray(actions), self.replace(rng=new_rng)

    def save(self, ckpt_dir: str, step: int) -> str:
        """Serialize the learner to a msgpack checkpoint."""

        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"ckpt_{step}.msgpack")
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(self))
        return path

    @classmethod
    def load(
        cls,
        ckpt_path: str,
        step: Optional[int] = None,
        *,
        lambda_lr: float = 1e-3,
        cost_limit: float = 0.0,
        lambda_max: Optional[float] = 1000.0,
    ) -> "TD3LagLearner":
        """Load learner state from a checkpoint file or directory."""

        def _pick_file(base: str, requested_step: Optional[int]) -> str:
            if os.path.isfile(base):
                return base
            if not os.path.isdir(base):
                raise FileNotFoundError(f"Checkpoint path {base} does not exist")

            if requested_step is not None:
                for cand in (
                    os.path.join(base, f"ckpt_{requested_step}.msgpack"),
                    os.path.join(base, f"ckpt_{requested_step}"),
                ):
                    if os.path.isfile(cand):
                        return cand

            candidates = sorted(
                glob.glob(os.path.join(base, "ckpt_*.msgpack"))
                + glob.glob(os.path.join(base, "ckpt_*"))
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No checkpoint found under {base} (step={requested_step})"
                )
            return candidates[-1]

        resolved = _pick_file(ckpt_path, step)
        with open(resolved, "rb") as f:
            data = f.read()

        def _restore_from_state_dict_checkpoint(state_dict: dict) -> "TD3LagLearner":
            required_keys = {
                "actor",
                "critic",
                "target_actor",
                "target_critic",
                "cost_critic",
                "target_cost_critic",
                "rng",
                "tau",
                "discount",
                "exploration_noise",
                "target_policy_noise",
                "target_policy_noise_clip",
                "actor_delay",
                "lagrangian_lambda",
            }
            missing = required_keys.difference(state_dict.keys())
            if missing:
                raise TypeError(
                    "Checkpoint is missing required keys for TD3LagLearner: "
                    + ", ".join(sorted(missing))
                )

            def _unwrap_params(tree_like):
                current = tree_like
                while isinstance(current, (dict, FrozenDict)) and "params" in current:
                    maybe = current.get("params")
                    if isinstance(maybe, (dict, FrozenDict)):
                        current = maybe
                        continue
                    if len(current.keys()) == 1:
                        current = maybe
                        break
                    break
                return current

            def _infer_actor_dims(actor_params_tree):
                params_tree = _unwrap_params(actor_params_tree)
                leaves = _iter_array_leaves_with_paths(params_tree)

                kernel_candidates = [
                    (path, arr)
                    for path, arr in leaves
                    if arr.ndim == 2 and "kernel" in path.lower()
                ]
                if not kernel_candidates:
                    kernel_candidates = [(path, arr) for path, arr in leaves if arr.ndim == 2]

                if len(kernel_candidates) < 2:
                    preview = ", ".join(
                        [f"{p}:{np.asarray(a).shape}" for p, a in leaves[:30]]
                    )
                    raise TypeError(
                        "Unable to infer actor architecture from checkpoint params; "
                        f"found {len(kernel_candidates)} 2D kernels. Array leaves: {preview}"
                    )

                shapes = [(path, np.asarray(arr).shape) for path, arr in kernel_candidates]
                action_dim = min(shape[1] for _, shape in shapes)
                last_candidates = [
                    (path, shape) for path, shape in shapes if shape[1] == action_dim
                ]
                last_path, last_shape = max(last_candidates, key=lambda kv: kv[1][0])
                chain = [(last_path, last_shape)]
                remaining = [item for item in shapes if item != (last_path, last_shape)]
                current_in = last_shape[0]

                while True:
                    predecessors = [
                        (path, shape)
                        for path, shape in remaining
                        if shape[1] == current_in
                    ]
                    if not predecessors:
                        break
                    best = max(predecessors, key=lambda kv: kv[1][0])
                    chain.insert(0, best)
                    remaining.remove(best)
                    current_in = best[1][0]

                obs_dim = int(chain[0][1][0])
                hidden_dims = tuple(int(shape[1]) for _, shape in chain[:-1])
                if not hidden_dims:
                    hidden_dims = (int(chain[-1][1][0]),)
                return obs_dim, int(action_dim), hidden_dims

            def _infer_num_qs(params):
                for leaf in jax.tree_util.tree_leaves(params):
                    if hasattr(leaf, "shape") and leaf.shape:
                        return int(leaf.shape[0])
                raise TypeError("Unable to infer ensemble size from critic params")

            actor_params = _unwrap_params(state_dict["actor"].get("params", {}))
            obs_dim, action_dim, hidden_dims = _infer_actor_dims(actor_params)

            critic_params = state_dict["critic"]["params"]
            target_critic_params = state_dict["target_critic"]["params"]
            cost_critic_params = state_dict["cost_critic"]["params"]
            target_cost_critic_params = state_dict["target_cost_critic"]["params"]

            num_qs = _infer_num_qs(critic_params)
            num_target_qs = _infer_num_qs(target_critic_params)
            num_cost_qs = _infer_num_qs(cost_critic_params)
            num_target_cost_qs = _infer_num_qs(target_cost_critic_params)

            rng = jnp.asarray(state_dict["rng"])

            actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
            actor_def = TanhDeterministic(actor_base_cls, action_dim)
            dummy_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
            actor_vars = actor_def.init(jax.random.PRNGKey(0), dummy_obs)
            actor_template = TrainState.create(
                apply_fn=actor_def.apply,
                params=actor_vars["params"],
                tx=optax.adam(learning_rate=3e-4),
            )
            target_actor_template = TrainState.create(
                apply_fn=actor_def.apply,
                params=actor_vars["params"],
                tx=optax.GradientTransformation(lambda _: None, lambda _: None),
            )

            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=None,
                use_layer_norm=False,
            )
            critic_cls = partial(StateActionValue, base_cls=critic_base_cls)

            dummy_action = jnp.zeros((action_dim,), dtype=jnp.float32)

            critic_def = Ensemble(critic_cls, num=num_qs)
            critic_template = TrainState.create(
                apply_fn=critic_def.apply,
                params=critic_def.init(jax.random.PRNGKey(1), dummy_obs, dummy_action)[
                    "params"
                ],
                tx=optax.adam(learning_rate=3e-4),
            )

            target_critic_def = Ensemble(critic_cls, num=num_target_qs)
            target_critic_template = TrainState.create(
                apply_fn=target_critic_def.apply,
                params=target_critic_def.init(
                    jax.random.PRNGKey(2), dummy_obs, dummy_action
                )["params"],
                tx=optax.GradientTransformation(lambda _: None, lambda _: None),
            )

            cost_critic_def = Ensemble(critic_cls, num=num_cost_qs)
            cost_critic_template = TrainState.create(
                apply_fn=cost_critic_def.apply,
                params=cost_critic_def.init(
                    jax.random.PRNGKey(3), dummy_obs, dummy_action
                )["params"],
                tx=optax.adam(learning_rate=3e-4),
            )

            target_cost_critic_def = Ensemble(critic_cls, num=num_target_cost_qs)
            target_cost_critic_template = TrainState.create(
                apply_fn=target_cost_critic_def.apply,
                params=target_cost_critic_def.init(
                    jax.random.PRNGKey(4), dummy_obs, dummy_action
                )["params"],
                tx=optax.GradientTransformation(lambda _: None, lambda _: None),
            )

            actor = serialization.from_state_dict(actor_template, state_dict["actor"])
            target_actor = serialization.from_state_dict(
                target_actor_template, state_dict["target_actor"]
            )
            critic = serialization.from_state_dict(critic_template, state_dict["critic"])
            target_critic = serialization.from_state_dict(
                target_critic_template, state_dict["target_critic"]
            )
            cost_critic = serialization.from_state_dict(
                cost_critic_template, state_dict["cost_critic"]
            )
            target_cost_critic = serialization.from_state_dict(
                target_cost_critic_template, state_dict["target_cost_critic"]
            )

            return cls(
                rng=rng,
                actor=actor,
                critic=critic,
                cost_critic=cost_critic,
                target_critic=target_critic,
                target_cost_critic=target_cost_critic,
                target_actor=target_actor,
                tau=float(state_dict.get("tau", 0.005)),
                discount=float(state_dict.get("discount", 0.99)),
                num_qs=num_qs,
                num_min_qs=None,
                exploration_noise=float(state_dict.get("exploration_noise", 0.1)),
                target_policy_noise=float(state_dict.get("target_policy_noise", 0.2)),
                target_policy_noise_clip=float(
                    state_dict.get("target_policy_noise_clip", 0.5)
                ),
                actor_delay=int(state_dict.get("actor_delay", 2)),
                lagrangian_lambda=jnp.asarray(
                    state_dict.get("lagrangian_lambda", 0.0), dtype=jnp.float32
                ),
                lambda_lr=lambda_lr,
                cost_limit=cost_limit,
                lambda_max=lambda_max,
            )

        try:
            loaded = serialization.from_bytes(cls, data)
        except (TypeError, ValueError):
            loaded = serialization.msgpack_restore(data)

        if isinstance(loaded, dict):
            for v in loaded.values():
                if isinstance(v, cls):
                    return v
            return _restore_from_state_dict_checkpoint(loaded)
        if not isinstance(loaded, cls):
            return _restore_from_state_dict_checkpoint(
                serialization.msgpack_restore(data)
            )
        return loaded

    # Manual sanity checklist (informal):
    # 1) Run TD3LagLearner.create(...) with a toy Gym space to ensure construction works.
    # 2) Call agent.save(...); TD3LagLearner.load(...) should restore the learner type.
    # 3) Invoke agent.update(batch, utd_ratio=1) on a synthetic batch and confirm
    #    returned info contains lambda-related entries.
