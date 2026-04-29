"""Loader for SAC-CBF checkpoints (not yet registered in load_agent)."""

import inspect
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from flax import serialization
from flax.core import frozen_dict

from jaxrl5.agents.sac.sac_cbf_learner import SACCbfLearner
from jaxrl5.tools.checkpoints import resolve_checkpoint


PolicyFn = Callable[[np.ndarray], np.ndarray]


def _extract_step(path: str) -> Optional[int]:
    m = re.search(r"ckpt_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


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
    sig = inspect.signature(SACCbfLearner.create)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def _extract_config(cfg: Dict) -> Dict:
    if "config" in cfg and isinstance(cfg["config"], dict):
        return cfg["config"]
    return cfg


def _build_policy_fn(
    state_holder: Dict[str, SACCbfLearner],
    *,
    deterministic: bool,
) -> PolicyFn:
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)
        single = obs_np.ndim == 1
        if single:
            obs_np = obs_np[None]

        if deterministic:
            actions, new_agent = state_holder["agent"].eval_actions(obs_np)
        else:
            actions, new_agent = state_holder["agent"].sample_actions(obs_np)

        state_holder["agent"] = new_agent
        if single:
            actions = actions[0]
        return np.asarray(actions, dtype=np.float32)

    return policy_fn


def load_sac_cbf(
    ckpt_path: str,
    step: Optional[int] = None,
    *,
    observation_space=None,
    action_space=None,
    seed: int = 0,
    config_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    deterministic: bool = True,
    **kwargs,
) -> Tuple[SACCbfLearner, PolicyFn, Dict]:
    """
    Load a SACCbfLearner checkpoint and return (agent, policy_fn, meta).

    This loader is robust to:
      - checkpoints saved via `agent.save(path)` (msgpack bytes)
      - checkpoints that restore to a state_dict (dict/FrozenDict)

    It avoids `serialization.from_bytes(SACCbfLearner, data)` because passing a class
    (instead of a template instance) can yield a raw dict and break `.eval_actions()`.
    """
    if observation_space is None or action_space is None:
        raise ValueError(
            "load_sac_cbf requires observation_space and action_space to build a template agent."
        )

    resolved = resolve_checkpoint(ckpt_path, step)

    # 0) Prefer direct class loader if available (fast path)
    direct_agent = None
    if hasattr(SACCbfLearner, "load"):
        try:
            direct_agent = SACCbfLearner.load(resolved)
        except Exception:
            direct_agent = None

    config_path_used: Optional[Path] = None

    if isinstance(direct_agent, SACCbfLearner):
        agent: SACCbfLearner = direct_agent
    else:
        # 1) Read raw bytes, restore to python object/state_dict
        data = Path(resolved).read_bytes()

        if not hasattr(serialization, "msgpack_restore"):
            raise RuntimeError(
                "flax.serialization.msgpack_restore not found; cannot restore checkpoint bytes."
            )

        state = serialization.msgpack_restore(data)

        # 2) If the checkpoint restores directly to a Learner (rare), use it
        if isinstance(state, SACCbfLearner):
            agent = state

        # 3) Typical: restores to a state dict -> build template -> from_state_dict
        elif isinstance(state, (dict, frozen_dict.FrozenDict)):
            resolved_path = Path(resolved)

            # Candidate directories where config might live
            # - explicit run_dir
            # - same directory as ckpt (tmp smoke test)
            # - parent.parent (common "run_dir/checkpoints/ckpt_xxx")
            candidate_dirs = []
            if run_dir is not None:
                candidate_dirs.append(Path(run_dir))
            candidate_dirs.append(resolved_path.parent)
            candidate_dirs.append(resolved_path.parent.parent)

            cfg_dict: Dict = {}
            if config_path is not None:
                config_path_used = Path(config_path)
                cfg_dict = _extract_config(_load_config(config_path_used))
            else:
                for d in candidate_dirs:
                    try:
                        config_path_used = _find_config_path(d)
                        cfg_dict = _extract_config(_load_config(config_path_used))
                        break
                    except FileNotFoundError:
                        continue

            # Filter config/kwargs to only those accepted by SACCbfLearner.create
            filtered_cfg = _filter_create_kwargs(cfg_dict)
            filtered_cfg.update(_filter_create_kwargs(kwargs))  # kwargs override config

            template = SACCbfLearner.create(
                seed=seed,
                observation_space=observation_space,
                action_space=action_space,
                **filtered_cfg,
            )
            agent = serialization.from_state_dict(template, state)

        else:
            raise TypeError(
                f"Loaded checkpoint type {type(state)} is not SACCbfLearner or state_dict"
            )

    # 4) Hard guard to avoid silent dict leakage
    if not isinstance(agent, SACCbfLearner):
        raise TypeError(f"load_sac_cbf produced {type(agent)} instead of SACCbfLearner")

    state_holder = {"agent": agent}
    policy_fn = _build_policy_fn(state_holder, deterministic=deterministic)

    meta: Dict = {
        "algo": "sac_cbf",
        "ckpt_resolved_path": str(resolved),
        "step": _extract_step(resolved),
        "deterministic": deterministic,
    }
    if config_path_used is not None:
        meta["config_path"] = str(config_path_used)

    return state_holder["agent"], policy_fn, meta

