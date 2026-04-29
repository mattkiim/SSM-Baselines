from __future__ import annotations

from typing import Any

import jax


def soft_update(target_params: Any, source_params: Any, tau: float):
    """Applies an exponential moving average update to target parameters."""

    def _update(tgt, src):
        return (1.0 - tau) * tgt + tau * src

    return jax.tree_util.tree_map(_update, target_params, source_params)

