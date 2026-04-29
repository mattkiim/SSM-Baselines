import glob
import os
from typing import Optional


def resolve_checkpoint(ckpt_path: str, step: Optional[int] = None) -> str:
    """Resolve a checkpoint path to a concrete file.

    Args:
        ckpt_path: Path to a checkpoint file or directory.
        step: Optional training step to select when ``ckpt_path`` is a directory.

    Returns:
        Path to a checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint can be resolved.
    """

    if os.path.isfile(ckpt_path):
        return ckpt_path

    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path '{ckpt_path}' does not exist")

    def _list_candidates(directory: str):
        return sorted(
            glob.glob(os.path.join(directory, "ckpt_*.msgpack"))
            + glob.glob(os.path.join(directory, "ckpt_*"))
        )

    search_dir = ckpt_path
    if os.path.isdir(os.path.join(ckpt_path, "checkpoints")):
        search_dir = os.path.join(ckpt_path, "checkpoints")

    if step is not None:
        explicit = [
            os.path.join(search_dir, f"ckpt_{step}.msgpack"),
            os.path.join(search_dir, f"ckpt_{step}"),
        ]
        for cand in explicit:
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(
            f"Checkpoint for step {step} not found under '{search_dir}'. Candidates:"
            f" {', '.join(_list_candidates(search_dir)) or '[]'}"
        )

    candidates = _list_candidates(search_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under '{search_dir}'. Candidates:[]"
        )
    return candidates[-1]
