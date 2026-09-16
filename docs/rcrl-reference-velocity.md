# Reference-style RCRL on velocity tasks

This experiment ports the cloned `Reachability_Constrained_RL` quadrotor
optimization protocol into our JAX learner. It is a Safety-Gymnasium adaptation,
not an exact reproduction of the original asynchronous TensorFlow experiment.

## Protocol

Run `bash scripts/train_rcrl_reference.sh ant` or replace `ant` with `humanoid`.
Fresh seed 0 runs train for 3,000,000 environment steps, with 10,000 random
warmup steps, a 1,000,000-transition replay buffer, batch size 256, and one
learner update per subsequent environment step. Each run evaluates and saves a
checkpoint every 30,000 steps. Evaluation uses 50 deterministic episodes,
seeds 42–91, capped at 1,000 steps each. Costs are the environment's original
undiscounted episode costs. Cost 10 is an evaluation criterion, not a learner
budget. Repeated evaluation seeds allow comparable curves but are not held-out
validation for choosing the best checkpoint.

## Reference choices

The optional `reference_protocol` learner setting enables:

- Safety target `max(20 sign(h), Qh_target)` with discount 1, where
  `h = transition_speed - environment_velocity_limit`. A true terminal
  transition targets just `20 sign(h)`; exact boundary h=0 maps to 0.
- Threshold 0 and uncapped `softplus(MLP(state))` multiplier. In saved configs
  and diagnostics `lambda_max=-1` denotes no cap; cap fraction is zero.
- Multiplier residual clipped to [-10, 100]. Global gradient norm clipping
  at 10 for each reward critic independently, safety critic, actor and
  temperature, and 3 for the multiplier.
- Actor and temperature updated every 4 learner updates, multiplier every 12.
  Target critics update with tau .005 only on actor-update steps.
  Actor and multiplier gradients use the pre-update critics and policy.
- Linear learning-rate decay: critics 1e-4 to 1e-6; actor 2e-5 to 1e-6;
  multiplier 6e-7 to 1e-7; temperature 8e-5 to 3e-6. Decay spans this run's
  2,990,001 learner updates, adjusted for each optimizer's update frequency.
- Two ELU hidden layers of 256 units, He-normal weight initialization,
  zero biases, Gaussian log-standard-deviation bounds [-5, 1].
- Log-temperature loss, as in the original, and Adam epsilon 1e-7.

Source locations: `worker.py` signed constraint encoding;
`learners/sac.py` feasibility target, policy, multiplier and temperature losses;
`policy.py` optimizer and target update schedules; `train_scripts/train_script.py`
default parameters, all within the cloned reference repository.

## Deliberate adaptations and remaining differences

Actions map from [-1,1] to each robot's native bounds (particularly Humanoid's
[-.4,.4]). Environment rewards and observations are retained without quadrotor
scaling. Both value targets bootstrap at time limits, and stop at true
terminations; the original safety target uses legacy `done`, while its reward
target does not mask terminal transitions. This is intentional for the velocity
tasks' continuing time limits and true falling terminations.

Execution is synchronous JAX on one GPU rather than asynchronous TensorFlow
workers/learners. We retain the existing replay size and warmup protocol.
Learning-rate decay is mapped to environment-driven learner updates instead of
the original 2M asynchronous iterations. Random sampling, separate actor output
heads, JAX initialization and optimizer numerics differ; target actions are sampled separately for reward and safety losses, while
the temperature update reuses the actor-loss action sample. No bitwise parity is claimed.

Existing configurations retain their prior behavior. Output directories are
`results/Safety{Ant,Humanoid}Velocity-v1/{ant,humanoid}_rcrl_reference_3m/DATE_seed0000/`.
Each contains config.json, history.csv, training_metrics.csv and checkpoints.
The launch log directory contains commands, source hashes and exit-status files.


## Twelve-run sweep

`python scripts/train_rcrl_batch.py` queues Humanoid and Ant seeds 0–4 plus
Swimmer and Hopper seed 0. Every run uses 3,000,000 steps and saves every
25,000 steps (120 checkpoints per completed run). Other reference settings,
including evaluation every 30,000 steps, are retained. Swimmer and Hopper
use signed forward velocity minus their environment threshold as the
constraint margin; Ant and Humanoid use planar speed.

The batch defaults to two concurrent GPU jobs (`--parallel` overrides this).
Use `--dry-run` to print commands. Run names include a UTC batch timestamp.
Logs, the initial queue manifest, per-job status JSON files, package versions,
and source hashes are saved under `logs/rcrl_batch_TIMESTAMP/`.
Per-job JSON files track live status; the manifest is finalized when all jobs
finish. Failed jobs are recorded and the remaining queue continues.

To persist across terminal/SSH disconnects, launch from the repository root:

```bash
mkdir -p logs
tmux new-session -d -s rcrl-batch \
  'exec .venv-rcrl/bin/python -u scripts/train_rcrl_batch.py > logs/rcrl-batch-console.log 2>&1'
tail -f logs/rcrl-batch-console.log
```

This survives disconnects while the host remains running; it does not restart
after reboot. The local `.venv-rcrl` environment on this host inherits simulator
packages from the `ssm_jax` conda environment and pins JAX/Flax/TFP to the
repository's recorded versions. Its CUDA plugin package is isolated from the
inherited newer plugin, and its `nvidia` symlink exposes the inherited CUDA
wheel libraries at the plugin's expected relative path.

To launch only Swimmer and Hopper seeds 1–4 with the same settings:

```bash
.venv-rcrl/bin/python scripts/train_rcrl_batch.py --robots swimmer hopper --seeds 1 2 3 4
```

Use a separate detached tmux session and console log when launching another batch.

See the [implementation fidelity audit](rcrl-fidelity-audit.md) for verified
optimizer, update-phase, sampling, and diagnostic differences from the clone.

## HalfCheetah and Walker2d

The `cheetah` and `walker` launcher names select `SafetyHalfCheetahVelocity-v1`
and `SafetyWalker2dVelocity-v1`. Both use signed forward velocity minus the
environment's threshold as the transition constraint, matching their costs.

```bash
.venv-rcrl/bin/python scripts/train_rcrl_batch.py \
  --protocol reference --robots cheetah walker --seeds 0 1 2 3 4
```

This uses the original reference-style configuration, 3M environment steps,
checkpoints every 25K, and evaluation every 30K. Batch launches freeze source for queued jobs. Completed Swimmer and Hopper runs are retained.
