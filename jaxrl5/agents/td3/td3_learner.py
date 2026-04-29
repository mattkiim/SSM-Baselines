"""Implementations of algorithms for continuous control."""

import glob
import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax import serialization
from flax.training.train_state import TrainState
from flax.core import FrozenDict

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhDeterministic
from jaxrl5.networks import MLP, Ensemble, StateActionValue, subsample_ensemble


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


def _unwrap_params(tree):
    unwrapped = tree
    while isinstance(unwrapped, (dict, FrozenDict)) and "params" in unwrapped:
        candidate = unwrapped["params"]
        if isinstance(candidate, (dict, FrozenDict)):
            unwrapped = candidate
            continue
        break
    return unwrapped


def _infer_actor_dims(actor_params_tree) -> Tuple[int, int, Tuple[int, ...]]:
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
        preview = ", ".join([f"{p}: {a.shape}" for p, a in leaves[:30]])
        raise TypeError(
            "Unable to infer actor architecture from checkpoint params; "
            f"found {len(kernel_candidates)} kernel candidates. Leaves preview: {preview}"
        )

    kernels = [(path, tuple(arr.shape)) for path, arr in kernel_candidates]
    action_dim = min(shape[1] for _, shape in kernels)
    last_candidates = [k for k in kernels if k[1][1] == action_dim]
    last = max(last_candidates, key=lambda item: item[1][0])
    chain = [last]
    remaining = [k for k in kernels if k is not last]
    current_in = last[1][0]

    while True:
        predecessors = [k for k in remaining if k[1][1] == current_in]
        if not predecessors:
            break
        prev = max(predecessors, key=lambda item: item[1][0])
        chain.append(prev)
        remaining.remove(prev)
        current_in = prev[1][0]

    chain = chain[::-1]
    if len(chain) < 1:
        preview = ", ".join([f"{p}: {s}" for p, s in kernels[:30]])
        raise TypeError(
            "Failed to build actor layer chain from kernels. Preview: " + preview
        )

    obs_dim = chain[0][1][0]
    hidden_dims = tuple(layer_shape[1] for _, layer_shape in chain[:-1])
    action_dim = chain[-1][1][1]

    if len(hidden_dims) < 1:
        preview = ", ".join([f"{p}: {s}" for p, s in kernels[:30]])
        raise TypeError(
            "Inferred actor network has no hidden layers; kernel preview: " + preview
        )

    return obs_dim, action_dim, hidden_dims


def _infer_num_qs_from_critic_params(critic_params_tree) -> int:
    params_tree = _unwrap_params(critic_params_tree)
    leaves = _iter_array_leaves_with_paths(params_tree)
    for _, arr in leaves:
        if hasattr(arr, "shape") and arr.ndim > 0:
            return int(arr.shape[0])
    raise TypeError("Unable to infer critic ensemble size from checkpoint params")


def _restore_from_state_dict_checkpoint(
    state: dict, *, actor_lr: float = 3e-4, critic_lr: float = 3e-4
) -> "TD3Learner":
    required = [
        "actor",
        "critic",
        "target_actor",
        "target_critic",
        "rng",
        "tau",
        "discount",
        "exploration_noise",
        "target_policy_noise",
        "target_policy_noise_clip",
        "actor_delay",
    ]
    missing = [k for k in required if k not in state]
    if missing:
        raise TypeError(f"Checkpoint missing required keys: {missing}")

    actor_state = state["actor"]
    critic_state = state["critic"]
    target_critic_state = state["target_critic"]

    obs_dim, action_dim, hidden_dims = _infer_actor_dims(actor_state.get("params", {}))
    num_qs = _infer_num_qs_from_critic_params(critic_state.get("params", {}))
    target_num_qs = _infer_num_qs_from_critic_params(
        target_critic_state.get("params", {})
    )

    actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
    actor_def = TanhDeterministic(actor_base_cls, action_dim)
    critic_base_cls = partial(
        MLP,
        hidden_dims=hidden_dims,
        activate_final=True,
        dropout_rate=None,
        use_layer_norm=False,
    )
    critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
    critic_def = Ensemble(critic_cls, num=num_qs)
    target_critic_def = Ensemble(critic_cls, num=target_num_qs)

    dummy_obs = np.zeros((obs_dim,), dtype=np.float32)
    dummy_act = np.zeros((action_dim,), dtype=np.float32)

    actor_params = actor_def.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    critic_params = critic_def.init(jax.random.PRNGKey(0), dummy_obs, dummy_act)["params"]
    target_actor_params = actor_params
    target_critic_params = target_critic_def.init(
        jax.random.PRNGKey(0), dummy_obs, dummy_act
    )["params"]

    noop_tx = optax.GradientTransformation(lambda _: None, lambda _: None)

    actor_template = TrainState.create(
        apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(actor_lr)
    )
    critic_template = TrainState.create(
        apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(critic_lr)
    )
    target_actor_template = TrainState.create(
        apply_fn=actor_def.apply, params=target_actor_params, tx=noop_tx
    )
    target_critic_template = TrainState.create(
        apply_fn=target_critic_def.apply, params=target_critic_params, tx=noop_tx
    )

    actor = serialization.from_state_dict(actor_template, actor_state)
    critic = serialization.from_state_dict(critic_template, critic_state)
    target_actor = serialization.from_state_dict(target_actor_template, state["target_actor"])
    target_critic = serialization.from_state_dict(
        target_critic_template, target_critic_state
    )

    num_min_qs = target_num_qs if target_num_qs < num_qs else None

    return TD3Learner(
        rng=jnp.asarray(state["rng"]),
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        target_actor=target_actor,
        tau=float(state["tau"]),
        discount=float(state["discount"]),
        num_qs=num_qs,
        num_min_qs=num_min_qs,
        exploration_noise=float(state["exploration_noise"]),
        target_policy_noise=float(state["target_policy_noise"]),
        target_policy_noise_clip=float(state["target_policy_noise_clip"]),
        actor_delay=int(state["actor_delay"]),
    )


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


class TD3Learner(Agent):
    critic: TrainState
    target_critic: TrainState
    target_actor: TrainState
    tau: float
    discount: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    exploration_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_policy_noise_clip: float = 0.5
    actor_delay: int = 2

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
    ):
        """
        An implementation of TD3: https://arxiv.org/abs/1802.09477
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

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

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            target_actor=target_actor,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            exploration_noise=exploration_noise,
            target_policy_noise=target_policy_noise,
            target_policy_noise_clip=target_policy_noise_clip,
            actor_delay=actor_delay,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng, num=2)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            actions = self.actor.apply_fn(
                {"params": actor_params}, batch["observations"]
            )
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = -q.mean()
            return actor_loss, {
                "actor_loss": actor_loss,
            }

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        target_actor_params = optax.incremental_update(
            actor.params, self.target_actor.params, self.tau
        )

        target_actor = self.target_actor.replace(params=target_actor_params)

        return self.replace(actor=actor, target_actor=target_actor, rng=rng), actor_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        next_actions = self.target_actor.apply_fn(
            {"params": self.target_actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        target_noise = (
            jax.random.normal(key, next_actions.shape) * self.target_policy_noise
        )
        target_noise = target_noise.clip(
            -self.target_policy_noise_clip, self.target_policy_noise_clip
        )
        next_actions += target_noise

        # Used only for REDQ.
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
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["not_terminated"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        true_steps = new_agent.critic.step / utd_ratio

        # Actor delay
        new_agent, actor_info = jax.lax.cond(
            true_steps % new_agent.actor_delay == 0,
            new_agent.update_actor,
            lambda _: (new_agent, {"actor_loss": 0.0}),
            mini_batch,
        )

        return new_agent, {**actor_info, **critic_info}

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
        """Serialize the learner to a msgpack checkpoint.

        The checkpoint is written to ``{ckpt_dir}/ckpt_{step}.msgpack``. This
        mirrors the SafeScoreMatchingLearner format and avoids Orbax path
        restrictions by using plain Flax serialization.
        """

        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"ckpt_{step}.msgpack")
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(self))
        return path

    @classmethod
    def load(cls, ckpt_path: str, step: Optional[int] = None) -> "TD3Learner":
        """Load learner state from a checkpoint file or directory.

        If ``ckpt_path`` is a directory, the loader searches for
        ``ckpt_{step}.msgpack`` (or the latest when ``step`` is ``None``).
        """

        def _pick_file(base: str, requested_step: Optional[int]) -> str:
            if os.path.isfile(base):
                return base
            if not os.path.isdir(base):
                raise FileNotFoundError(f"Checkpoint path {base} does not exist")

            candidates = []
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
        # The checkpoints are written via ``serialization.to_bytes(self)``; the
        # matching deserialization path is ``serialization.from_bytes``. Using
        # ``msgpack_restore`` on these bytes may yield a plain ``dict`` rather
        # than the structured learner, which triggered downstream attribute
        # errors. Prefer ``from_bytes`` and only fall back to any embedded
        # TD3Learner in a restored mapping for compatibility with older saves.
        loaded = None
        try:
            loaded = serialization.from_bytes(cls, data)
            if isinstance(loaded, cls):
                return loaded
        except TypeError:
            loaded = None

        state = serialization.msgpack_restore(data)
        if isinstance(state, dict):
            return _restore_from_state_dict_checkpoint(state)

        raise TypeError(
            f"Loaded checkpoint type {type(loaded) if loaded is not None else type(state)} is not TD3Learner"
        )
