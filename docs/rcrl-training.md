# RCRL velocity training

Run from the repository root. The launcher uses the `ssm` Python environment,
CUDA, seed 0, and the existing RAC learner settings:

```bash
bash scripts/train_rcrl_velocity.sh ant
bash scripts/train_rcrl_velocity.sh humanoid
```

To extend a completed run to 3 million total environment steps:

```bash
bash scripts/train_rcrl_velocity.sh ant \
  --resume_checkpoint=results/SafetyAntVelocity-v1/ant_rcrl_velocity/2026-09-08_seed0000/checkpoints/ckpt_1000000.msgpack \
  --resume_step=1000000 \
  --max_steps=3000000 \
  --resume_warmup_steps=10000 \
  --replay_capacity=1000000 \
  --run_name=ant_rcrl_velocity_3m
```

Use the corresponding Humanoid checkpoint and a distinct run name for Humanoid.
The trainer refuses to overwrite an existing run's configuration. GPU device
access is required; this workspace's sandbox hides the GPU. Set
`JAX_PLATFORMS=cpu` explicitly only when CPU training is intended.

Checkpoint continuation restores the learner, optimizer states, and learner RNG.
The replay buffer and simulator state are not saved, so this is not an exact
continuation of an uninterrupted run. A fresh replay buffer is filled using the
restored policy before updates resume. Warmup steps count toward `max_steps`.
The new run saves its source checkpoint in `config.json` and uses global step
numbers for history and checkpoints. Original run artifacts remain intact.

## Fresh threshold-10 experiments

```bash
bash scripts/train_rcrl_threshold10.sh ant
bash scripts/train_rcrl_threshold10.sh humanoid
```

These start fresh seed-0 learners with `safety_threshold=10.0`, train for 3 million
steps using a 1-million-transition replay buffer, and evaluate every 10,000
training steps over 50 episodes capped at 1,000 steps each. The threshold applies
to the discounted safety critic value, not the undiscounted episode cost.
Checkpoint saving remains every 50,000 steps. Use a new `--run_name` when
repeating an experiment to preserve existing results.
