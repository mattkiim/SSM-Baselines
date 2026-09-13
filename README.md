# Safe Score Matching

Safe reinforcement learning implementations in JAX, adapted from QSM. The
`jaxrl5` library includes score matching agents, reachability constrained RL
(RAC), primal-dual safe RL (CAL), and SAC/TD3 variants.

## Repository layout

| Path | Contents |
| --- | --- |
| `jaxrl5/agents/` | Learners, grouped by algorithm |
| `jaxrl5/envs/` | Safety environments, quadrotor environments, and dynamics |
| `jaxrl5/networks/` and `jaxrl5/distributions/` | Networks and policy distributions |
| `jaxrl5/data/` | Datasets and replay buffers |
| `jaxrl5/tools/` | Checkpoint saving and policy loading |
| `jaxrl5/utils/`, `jaxrl5/wrappers/`, `jaxrl5/algorithms/` | Shared utilities, wrappers, and update helpers |
| `examples/states/` | State-based training, including Safety-Gymnasium |
| `examples/quadrotor/` | Quadrotor training, evaluation, and visualization |
| `examples/f16/` | F-16 adapters and training |
| `examples/pixels/` | Pixel-based training and data collection |
| `launcher/` | Experiment launchers and hyperparameter sweeps |
| `tests/smoke/` | Standalone learner and environment checks |
| `docs/` | Training examples and original infrastructure notes |

Each example family keeps hyperparameters in its own `configs/` directory.
Initial-state `.npz` assets remain beside those configurations.
Generated evaluation results are collected in `results/evaluations/`; see
[the output directory map](docs/evaluation.md) for their locations.

## Setup and training

For seed-0 Ant/Humanoid RCRL experiments on a new machine, follow
[the portable setup guide](docs/portable-setup.md) and
[the reference-style protocol](docs/rcrl-reference-velocity.md). These include
pinned core dependencies, smoke checks, and periodic evaluation commands.


Install the local package into your experiment environment:

```bash
python -m pip install -e .
```

The existing dependency list is in [requirements.txt](requirements.txt), moved
from `temp.txt`. It contains historical minimum versions, not a tested lockfile;
package installation does not automatically install these dependencies.
Environment-specific examples may need additional simulator or dataset packages.

Run this Safety-Gymnasium example from the repository root:

```bash
python examples/states/train_safe_matching_online.py \
  --config=examples/states/configs/safe_matching_config.py \
  --env_name=SafetyCarButton1-v0
```

See [training commands](docs/training.md), [state examples](examples/states/README.md),
and [pixel examples](examples/pixels/README.md). Commands in the state and pixel
READMEs use paths relative to their respective directories. The
[original infrastructure notes](docs/infrastructure.md) are retained for context;
some referenced external shell scripts are not included in this checkout.

## Checks

See [tests/README.md](tests/README.md) for standalone smoke-test commands. Depending
on the learner, these exercise updates, action sampling, and checkpoint loading.
They require the corresponding runtime dependencies.

## Working conventions

Keep reusable code in `jaxrl5/`, runnable experiments in `examples/`, and checks
in `tests/`. Use an editable installation to pick up library changes without
reinstalling. Store generated runs in `results/`, `logs/`, `wandb/`, or
`results/evaluations/`, which Git ignores. Local experiment outputs are retained.
