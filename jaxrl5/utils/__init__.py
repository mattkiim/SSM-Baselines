"""Utility helpers for JAXRL5 agents."""

from jaxrl5.utils.history import append_history
from jaxrl5.utils.soft_update import soft_update

__all__ = ["soft_update", "append_history"]

