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

For RCRL velocity experiments on a new machine, follow
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

## RCRL velocity training commands

After following [the portable setup guide](docs/portable-setup.md), run these
commands from the repository root with your experiment environment activated.
The launcher uses `python` from that environment and defaults to CUDA; set
`RCRL_PYTHON=/path/to/python` to select another interpreter or
`JAX_PLATFORMS=cpu` for CPU execution.

These commands use the original reference-style protocol and the settings in
the saved experiment manifests. They show training seed 0; use `--seed=1`
through `--seed=4` for the other training seeds. Run names below omit the
historical batch timestamps.

```bash
# SafetySwimmerVelocity-v1
bash scripts/train_rcrl_reference.sh swimmer --seed=0 --max_steps=3000000 --save_interval=25000 --notqdm

# SafetyHopperVelocity-v1
bash scripts/train_rcrl_reference.sh hopper --seed=0 --max_steps=3000000 --save_interval=25000 --notqdm

# SafetyAntVelocity-v1
bash scripts/train_rcrl_reference.sh ant --seed=0 --max_steps=3000000 --save_interval=25000 --notqdm

# SafetyHumanoidVelocity-v1
bash scripts/train_rcrl_reference.sh humanoid --seed=0 --max_steps=3000000 --save_interval=25000 --notqdm

# SafetyHalfCheetahVelocity-v1
bash scripts/train_rcrl_reference.sh cheetah --seed=0 --max_steps=3000000 --save_interval=25000 --notqdm

# SafetyWalker2dVelocity-v1
bash scripts/train_rcrl_reference.sh walker --seed=0 --max_steps=3000000 --save_interval=25000 --notqdm
```

All six use [the reference configuration](examples/states/configs/rac_velocity_reference_config.py),
which inherits [the reachability configuration](examples/states/configs/rac_velocity_reachability_config.py).
The launchers set 10,000 random warmup steps, a 1,000,000-transition replay
buffer, batch size 256, and one learner update per environment step after
warmup. Evaluation runs every 30,000 steps over 50 deterministic episodes
with seeds 42–91, capped at 1,000 steps each. The explicit save interval above
matches the experiments: one checkpoint every 25,000 steps.

### Batch commands used for the sweeps

The batches were launched in these groups. Each command queues two concurrent
GPU jobs by default, assigns timestamped run names, and records its commands
and status under `logs/rcrl_batch_TIMESTAMP/`.

```bash
# Humanoid and Ant seeds 0–4, plus Swimmer and Hopper seed 0.
python scripts/train_rcrl_batch.py

# Remaining Swimmer and Hopper seeds.
python scripts/train_rcrl_batch.py --robots swimmer hopper --seeds 1 2 3 4

# HalfCheetah and Walker2d seeds 0–4.
python scripts/train_rcrl_batch.py --protocol reference --robots cheetah walker --seeds 0 1 2 3 4
```

Add `--dry-run` to a batch command to print its per-run commands without
starting training. Training outputs are saved under
`results/Safety<Robot>Velocity-v1/<run_name>/<date>_seedNNNN/`, including
`config.json`, evaluation history, training metrics, and checkpoints.
These outputs and batch logs are Git-ignored. See
[the reference protocol](docs/rcrl-reference-velocity.md) for algorithm details
and [the completed four-task results](docs/rcrl-results-3m-seeds0-4.md).

## Checks

See [tests/README.md](tests/README.md) for standalone smoke-test commands. Depending
on the learner, these exercise updates, action sampling, and checkpoint loading.
They require the corresponding runtime dependencies.

## Working conventions

Keep reusable code in `jaxrl5/`, runnable experiments in `examples/`, and checks
in `tests/`. Use an editable installation to pick up library changes without
reinstalling. Store generated runs in `results/`, `logs/`, `wandb/`, or
`results/evaluations/`, which Git ignores. Local experiment outputs are retained.
