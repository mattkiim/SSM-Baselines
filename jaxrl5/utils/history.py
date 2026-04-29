"""Helpers for writing experiment metrics to CSV history files."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict


def _make_history_path(env_name: str, experiment_name: str, seed: int) -> Path:
    """Return the history.csv path for a given experiment setup.

    The path format matches `<experiment_env>/<experiment_name>/<YYYY-MM-DD>_seedXXXX/history.csv`.
    """

    date_str = datetime.now().strftime("%Y-%m-%d")
    seed_str = f"seed{seed:04d}"
    experiment = experiment_name or "default_experiment"
    base_dir = "results" / Path(env_name) / experiment / f"{date_str}_{seed_str}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "history.csv"


def append_history(
    step: int, env_name: str, experiment_name: str, seed: int, metrics: Dict[str, float]
) -> None:
    """Append a metrics row to the experiment history file.

    Args:
        step: Global training step for the recorded metrics.
        env_name: Environment name used for the run.
        experiment_name: User-provided experiment or run identifier.
        seed: Random seed for the run.
        metrics: Mapping of metric name → value to store.
    """

    history_path = _make_history_path(env_name, experiment_name, seed)
    row = {"step": step, **metrics}

    # Maintain a deterministic column order so repeated appends are consistent.
    fieldnames = ["step"] + sorted(metrics.keys())
    write_header = not history_path.exists()

    with history_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

