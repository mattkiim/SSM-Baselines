"""Loader for SAC-Lag (SACLagLearner) checkpoints."""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from flax import serialization
from flax.core import frozen_dict

from jaxrl5.agents.sac.sac_lag_learner import SACLagLearner

PolicyFn = Callable[[np.ndarray], np.ndarray]


def _extract_step(path: str) -> Optional[int]:
    match = re.search(r"ckpt_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def _list_checkpoint_candidates(search_dir: Path) -> list[str]:
    patterns = ("ckpt_*.msgpack", "ckpt_*")
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(str(p) for p in sorted(search_dir.glob(pattern)))
    return candidates


def _resolve_checkpoint(ckpt_path: str, step: Optional[int]) -> Path:
    path = Path(ckpt_path)
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path '{ckpt_path}' does not exist")

    search_dir = path
    if (path / "checkpoints").is_dir():
        search_dir = path / "checkpoints"

    if step is not None:
        explicit = [search_dir / f"ckpt_{step}.msgpack", search_dir / f"ckpt_{step}"]
        for cand in explicit:
            if cand.is_file():
                return cand
        candidates = _list_checkpoint_candidates(search_dir)
        preview = ", ".join(candidates[:10])
        raise FileNotFoundError(
            f"Checkpoint for step {step} not found under '{search_dir}'. "
            f"Candidates (first 10): {preview or '[]'}"
        )

    candidates = _list_checkpoint_candidates(search_dir)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under '{search_dir}'. Candidates:[]")
    return Path(candidates[-1])


def _find_config_path(ckpt_file: Path) -> Path:
    if ckpt_file.parent.name == "checkpoints":
        run_dir = ckpt_file.parent.parent
    else:
        run_dir = ckpt_file.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found at '{config_path}'. "
            "Provide a run_dir with config.json to rebuild SACLagLearner."
        )
    return config_path


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_create_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(SACLagLearner.create)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def _build_policy_fn(state_holder: Dict[str, SACLagLearner], *, deterministic: bool) -> PolicyFn:
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)
        single = obs_np.ndim == 1
        if single:
            obs_np = obs_np[None]

        agent = state_holder["agent"]
        if deterministic and hasattr(agent, "eval_actions"):
            out = agent.eval_actions(obs_np)
        elif (not deterministic) and hasattr(agent, "sample_actions"):
            out = agent.sample_actions(obs_np)
        elif hasattr(agent, "eval_actions"):
            out = agent.eval_actions(obs_np)
        else:
            raise AttributeError("Agent has neither eval_actions nor sample_actions.")

        if isinstance(out, (tuple, list)) and len(out) == 2:
            actions, new_agent = out
            state_holder["agent"] = new_agent
        else:
            actions = out

        if single:
            actions = np.asarray(actions)[0]
        return np.asarray(actions, dtype=np.float32)

    return policy_fn


def load_sac_lag(
    ckpt_path: str,
    step: Optional[int] = None,
    *,
    observation_space=None,
    action_space=None,
    seed: int = 0,
    deterministic: bool = True,
) -> Tuple[SACLagLearner, PolicyFn, Dict[str, Any]]:
    """Load a SACLagLearner checkpoint and return (agent, policy_fn, meta)."""
    if observation_space is None or action_space is None:
        raise ValueError(
            "load_sac_lag requires observation_space and action_space to build a template agent."
        )

    resolved = _resolve_checkpoint(ckpt_path, step)
    config_path = _find_config_path(resolved)
    raw_cfg = _load_config(config_path)
    cfg = raw_cfg.get("config", raw_cfg)

    create_kwargs = _filter_create_kwargs(cfg)
    template = SACLagLearner.create(
        seed=seed,
        observation_space=observation_space,
        action_space=action_space,
        **create_kwargs,
    )

    data = resolved.read_bytes()
    restored = None
    if hasattr(serialization, "msgpack_restore"):
        restored = serialization.msgpack_restore(data)
        if isinstance(restored, SACLagLearner):
            agent = restored
        elif isinstance(restored, (dict, frozen_dict.FrozenDict)):
            agent = serialization.from_state_dict(template, restored)
        else:
            agent = serialization.from_bytes(template, data)
    else:
        agent = serialization.from_bytes(template, data)

    if not isinstance(agent, SACLagLearner):
        raise TypeError(f"load_sac_lag produced {type(agent)} instead of SACLagLearner")

    state_holder = {"agent": agent}
    policy_fn = _build_policy_fn(state_holder, deterministic=deterministic)

    meta: Dict[str, Any] = {
        "algo": "sac_lag",
        "ckpt_path": str(resolved),
        "step": _extract_step(str(resolved)),
        "config_path": str(config_path),
        "deterministic": deterministic,
    }
    return state_holder["agent"], policy_fn, meta
