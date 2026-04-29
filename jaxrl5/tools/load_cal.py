import inspect
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from flax import serialization
from flax.core import frozen_dict

from jaxrl5.agents.cal.cal_learner import CALAgent
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
    sig = inspect.signature(CALAgent.create)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def _extract_config(cfg: Dict) -> Dict:
    if "config" in cfg and isinstance(cfg["config"], dict):
        return cfg["config"]
    return cfg


def _build_policy_fn(
    state_holder: Dict[str, CALAgent],
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


def load_cal(
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
) -> Tuple[CALAgent, PolicyFn, Dict]:
    if observation_space is None or action_space is None:
        raise ValueError("load_cal requires observation_space and action_space to build a template agent.")

    resolved = resolve_checkpoint(ckpt_path, step)

    # Prefer direct class loader if available.
    direct_agent = None
    if hasattr(CALAgent, "load"):
        try:
            direct_agent = CALAgent.load(resolved)
        except Exception:
            direct_agent = None

    config_path_used: Optional[Path] = None
    if isinstance(direct_agent, CALAgent):
        agent = direct_agent
    else:
        data = Path(resolved).read_bytes()

        # 1) restore 原始内容（通常是 state_dict / FrozenDict）
        state = None
        if hasattr(serialization, "msgpack_restore"):
            state = serialization.msgpack_restore(data)
        else:
            # 极少数版本没有 msgpack_restore 时可以直接 raise 或自己处理
            raise RuntimeError("flax.serialization.msgpack_restore not found")

        # 2) 如果直接就是 CALAgent（很少见），直接用
        if isinstance(state, CALAgent):
            agent = state

        # 3) 如果是 state_dict，就必须先造 template 再 from_state_dict
        elif isinstance(state, (dict, frozen_dict.FrozenDict)):
            # ------- 更鲁棒的 run_dir 推断：先看同级，再看 parent.parent -------
            resolved_path = Path(resolved)
            candidate_dirs = []
            if run_dir is not None:
                candidate_dirs.append(Path(run_dir))
            candidate_dirs.append(resolved_path.parent)         # 适配你的 smoke test（ckpt 直接放 tmpdir）
            candidate_dirs.append(resolved_path.parent.parent)  # 适配常见结构 run_dir/checkpoints/ckpt_xxx

            cfg_dict = {}
            config_path_used = None
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

            filtered_cfg = _filter_create_kwargs(cfg_dict)
            # 允许调用者 kwargs 覆盖（比如 hidden_dims）
            filtered_cfg.update(_filter_create_kwargs(kwargs))

            template = CALAgent.create(
                seed=seed,
                observation_space=observation_space,
                action_space=action_space,
                **filtered_cfg,
            )
            agent = serialization.from_state_dict(template, state)

        else:
            raise TypeError(f"Loaded checkpoint type {type(state)} is not CALAgent or state_dict")

        # 4) 最后加一道硬检查，避免 silent fail
        if not isinstance(agent, CALAgent):
            raise TypeError(f"load_cal produced {type(agent)} instead of CALAgent")


    state_holder = {"agent": agent}
    policy_fn = _build_policy_fn(state_holder, deterministic=deterministic)

    meta = {
        "algo": "cal",
        "ckpt_resolved_path": str(resolved),
        "step": _extract_step(resolved),
        "deterministic": deterministic,
    }
    if config_path_used is not None:
        meta["config_path"] = str(config_path_used)
    return state_holder["agent"], policy_fn, meta
