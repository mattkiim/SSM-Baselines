import inspect
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from flax import serialization
from flax.core import frozen_dict

from jaxrl5.agents.safe_matching.safe_matching_learner import SafeScoreMatchingLearner
from jaxrl5.tools.checkpoints import resolve_checkpoint


PolicyFn = Callable[[np.ndarray], np.ndarray]


def _extract_step(path: str) -> Optional[int]:
    m = re.search(r"ckpt_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _iter_array_leaves_with_paths(tree) -> Iterable[Tuple[str, np.ndarray]]:
    stack = [("", tree)]
    while stack:
        path, node = stack.pop()
        if isinstance(node, (dict, frozen_dict.FrozenDict)):
            for k, v in node.items():
                new_path = f"{path}/{k}" if path else str(k)
                stack.append((new_path, v))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                new_path = f"{path}/{i}" if path else str(i)
                stack.append((new_path, v))
        else:
            try:
                arr = np.asarray(node)
            except Exception:
                continue
            if hasattr(arr, "ndim"):
                yield path, arr


def _unwrap_params(tree):
    current = tree
    while isinstance(current, (dict, frozen_dict.FrozenDict)) and "params" in current:
        next_tree = current.get("params")
        if isinstance(next_tree, (dict, frozen_dict.FrozenDict)):
            current = next_tree
        else:
            break
    return current


def _infer_mlp_hidden_dims_from_params(params_subtree) -> Tuple[int, ...]:
    params_subtree = _unwrap_params(params_subtree)
    leaves = list(_iter_array_leaves_with_paths(params_subtree))
    kernel_candidates = [
        (p, a)
        for p, a in leaves
        if a.ndim == 2 and (p.lower().endswith("kernel") or "kernel" in p.lower())
    ]
    if not kernel_candidates:
        kernel_candidates = [(p, a) for p, a in leaves if a.ndim == 2]
    if len(kernel_candidates) < 1:
        preview = ", ".join([f"{p}:{a.shape}" for p, a in leaves[:30]])
        raise TypeError(
            "Unable to infer MLP hidden dims: no 2D kernels found; leaves=" + preview
        )

    shapes = [(path, arr.shape) for path, arr in kernel_candidates]

    # Build longest forward chain where out_dim of one kernel matches in_dim of the next.
    chains = []
    for path, (in_dim, out_dim) in shapes:
        chain = [(path, (in_dim, out_dim))]
        used = {path}
        current_out = out_dim
        improved = True
        while improved:
            improved = False
            candidates = [
                (p, s)
                for p, s in shapes
                if p not in used and s[0] == current_out
            ]
            if candidates:
                # pick one with largest out_dim to favor continued expansion
                p_sel, s_sel = max(candidates, key=lambda x: x[1][1])
                chain.append((p_sel, s_sel))
                used.add(p_sel)
                current_out = s_sel[1]
                improved = True
        chains.append(chain)

    best_chain = max(chains, key=len)
    chain_shapes = [shape for _, shape in best_chain]
    if len(chain_shapes) == 1:
        hidden_dims: Tuple[int, ...] = ()
    else:
        hidden_dims = tuple(s[1] for s in chain_shapes[:-1])
    return hidden_dims


def _find_config_path(run_dir: Path) -> Path:
    candidates = [
        "config.json",
        "variant.json",
        "flags.json",
        "args.json",
        "params.json",
    ]
    for name in candidates:
        cand = run_dir / name
        if cand.exists():
            return cand
    listing = sorted(os.listdir(run_dir))[:50]
    raise FileNotFoundError(
        f"No config file found in {run_dir}. Tried {candidates}. Contents (first 50): {listing}"
    )


def _load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_create_kwargs(cfg: Dict) -> Dict:
    sig = inspect.signature(SafeScoreMatchingLearner.create)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def _score_hidden_dims_param_name() -> str:
    sig = inspect.signature(SafeScoreMatchingLearner.create)
    if "actor_hidden_dims" in sig.parameters:
        return "actor_hidden_dims"
    for name in sig.parameters:
        if name.endswith("hidden_dims"):
            return name
    return "actor_hidden_dims"


def load_ssm(
    ckpt_path: str,
    step: Optional[int] = None,
    *,
    observation_space=None,
    action_space=None,
    seed: int = 0,
    config_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    deterministic: bool = True,
    ddpm_temperature: Optional[float] = None,
    **kwargs,
) -> Tuple[SafeScoreMatchingLearner, PolicyFn, Dict]:
    if observation_space is None or action_space is None:
        raise ValueError("load_ssm requires observation_space and action_space to build a template agent.")

    resolved = resolve_checkpoint(ckpt_path, step)
    try:
        direct_agent = SafeScoreMatchingLearner.load(resolved)
    except Exception:
        direct_agent = None

    config_path_used: Optional[Path] = None
    if isinstance(direct_agent, SafeScoreMatchingLearner):
        agent = direct_agent
    else:
        data = Path(resolved).read_bytes()
        state = serialization.msgpack_restore(data)
        if not isinstance(state, dict):
            raise TypeError(
                f"Restored checkpoint is not a dict or SafeScoreMatchingLearner (got {type(state)})."
            )

        score_state = state.get("score_model")
        if not isinstance(score_state, dict) or "params" not in score_state:
            raise TypeError(
                f"Checkpoint at {resolved} missing score_model params; available top-level keys: {list(state.keys())}"
            )
        score_params = _unwrap_params(score_state.get("params"))
        inferred_score_hidden_dims = _infer_mlp_hidden_dims_from_params(score_params)

        resolved_run_dir = Path(run_dir) if run_dir is not None else Path(resolved).parent.parent
        config_path_used = Path(config_path) if config_path is not None else _find_config_path(resolved_run_dir)
        cfg_dict = _load_config(config_path_used)
        filtered_cfg = _filter_create_kwargs(cfg_dict)
        hidden_param_name = _score_hidden_dims_param_name()
        filtered_cfg[hidden_param_name] = inferred_score_hidden_dims
        try:
            template = SafeScoreMatchingLearner.create(
                seed=seed,
                observation_space=observation_space,
                action_space=action_space,
                **filtered_cfg,
            )
        except TypeError as exc:
            raise TypeError(
                f"Failed to create template SafeScoreMatchingLearner with config at {config_path_used}: {exc}"
            ) from exc
        try:
            agent = serialization.from_state_dict(template, state)
        except ValueError as exc:
            template_keys = list(template.score_model.params.keys()) if hasattr(template.score_model, "params") else []
            state_keys = list(score_params.keys()) if isinstance(score_params, (dict, frozen_dict.FrozenDict)) else []
            raise ValueError(
                f"Failed to restore SSM checkpoint at {resolved} with inferred hidden dims {inferred_score_hidden_dims}."
                f" Template score_model params keys: {template_keys}; state score_model params keys: {state_keys}; original error: {exc}"
            ) from exc

    if ddpm_temperature is not None and hasattr(agent, "ddpm_temperature"):
        agent = agent.replace(ddpm_temperature=ddpm_temperature)

    state_holder = {"agent": agent}

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)
        if deterministic:
            action, new_agent = state_holder["agent"].eval_actions(obs_np)
        else:
            action, new_agent = state_holder["agent"].sample_actions(obs_np)
        state_holder["agent"] = new_agent
        return np.asarray(action, dtype=np.float32)

    meta = {
        "algo": "ssm",
        "ckpt_resolved_path": str(resolved),
        "step": _extract_step(resolved),
        "deterministic": deterministic,
    }
    if ddpm_temperature is not None:
        meta["ddpm_temperature"] = ddpm_temperature
    if config_path_used is not None:
        meta["config_path"] = str(config_path_used)
    return state_holder["agent"], policy_fn, meta
