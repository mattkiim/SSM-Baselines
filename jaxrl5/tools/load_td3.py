from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from jaxrl5.agents.td3.td3_learner import TD3Learner
from jaxrl5.tools.checkpoints import resolve_checkpoint


PolicyFn = Callable[[np.ndarray], np.ndarray]


def _extract_step(path: str) -> Optional[int]:
    import os
    import re

    m = re.search(r"ckpt_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_td3(
    ckpt_path: str, step: Optional[int] = None, **_: Any
) -> Tuple[TD3Learner, PolicyFn, Dict]:
    resolved = resolve_checkpoint(ckpt_path, step)
    agent = TD3Learner.load(resolved)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_b = np.asarray(obs, dtype=np.float32)[None]
        actions, _ = agent.eval_actions(obs_b)
        return np.asarray(actions[0], dtype=np.float32)

    meta = {"algo": "td3", "ckpt_resolved_path": resolved, "step": _extract_step(resolved)}
    return agent, policy_fn, meta
