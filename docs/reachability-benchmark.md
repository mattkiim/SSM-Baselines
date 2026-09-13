# RCRL versus RESPO: velocity benchmark

The benchmark uses fresh training runs on `SafetyAntVelocity-v1` and
`SafetyHumanoidVelocity-v1`, seed 0, with 3,000,000 training transitions each.
Evaluation transitions do not count toward this total.

## Shared environment and evaluation protocol

- Both use the same locally installed Safety-Gymnasium simulator, rewards,
  observations, binary costs, and native time limits.
- Both expose normalized actions in [-1, 1], clip there, and map to the native
  bounds using `SymmetricActionWrapper`. Humanoid's native bounds are [-0.4, 0.4].
- Evaluate and save post-update checkpoints every 30,000 training steps.
- Use 50 deterministic evaluation episodes, each capped at 1,000 steps, with
  reset seeds 42 through 91 repeated at each evaluation.
- Report undiscounted episode reward, cost, length, and violation rate. Budget 10
  is the common velocity evaluation criterion, not an RCRL multiplier cap.
- These are single-seed pilot experiments, not a multi-seed statistical comparison.

The previous runs used a different safety objective and lacked normalized
Humanoid actions, so they are not interchangeable with these runs.

## RCRL adaptation

```bash
bash scripts/train_rcrl_reachability.sh ant
bash scripts/train_rcrl_reachability.sh humanoid
```

The config selects `reachability_transition`, `safety_threshold=0`, and
`lambda_max=100`. Other learner learning rates and update periods remain those
of `rac_config.py`. Replay capacity is one million transitions; warmup is 10,000
steps. The existing SAC-style stochastic actor is retained.

The simulator penalizes **transition-average planar speed**, measured by torso
displacement for Ant and mass-center displacement for Humanoid. We take
`h = hypot(info['x_velocity'], info['y_velocity']) - env._velocity_threshold`.
The velocity limits are 2.6222 for Ant-v1 and 1.4149 for Humanoid-v1. The wrapper
checks every transition that `(h > 0)` agrees with the simulator's cost.

The safety backup is `(1-gamma)*h + gamma*max(h, next_Qh)`, with gamma 0.99,
and is `h` on true terminations. Time-limit truncations bootstrap, treating the
task as continuing for critic training. This adapts equation (9) of the uploaded
RCRL paper from h(s) to a transition constraint h(s,a). It does not reconstruct
instantaneous speed from observations, sum binary costs, or clip safety values
to be nonnegative. Negative reachability values represent safety margins.

This is an off-policy stochastic actor-critic adaptation, not a reproduction of
the paper's on-policy Safety-Gym RCO experiment. RCRL targets no speed-limit
violations; its training constraint differs from RESPO's mean-cost budget.

`training_metrics.csv` saves critic diagnostics and measured multiplier min,
mean, maximum, and cap-saturation fraction. This avoids the zero placeholder
metrics on steps without a multiplier optimizer update. History is anchored to
the run directory even across midnight. Standalone RAC evaluation restores the
saved action-mapping setting.

## RESPO adaptation

```bash
bash scripts/train_respo_velocity.sh ant
bash scripts/train_respo_velocity.sh humanoid
```

`respo-clone/train_velocity.py` bridges the same environments to the clone's old
Gym API, with 30,000 steps per epoch, 100 epochs, and `cost_limit=10`. It uses
one MPI rank and two Torch threads. Architecture is two 256-unit hidden layers.
The clone's PPO losses, state-dependent reachability weighting, scalar penalty,
learning-rate schedules, and optimizer update counts are retained.

Evaluation uses the clipped Gaussian mean; RCRL uses its squashed deterministic
action. The callback writes post-update `history.csv` and numbered checkpoints
in the root `results/` tree. The clone's original unnumbered saver still runs
before an epoch update; use the new numbered checkpoints for this benchmark.
RESPO checkpoints save model weights and penalty for evaluation; they do not
provide exact training resume with optimizer or buffer state.

The isolated `.venv-respo` inherits simulator dependencies from `ssm`. Added
packages: CPU Torch 2.14.0+cpu, mpi4py-mpich 3.1.5, and joblib 1.6.0. Recreate with:

```bash
python -m venv --system-site-packages .venv-respo
.venv-respo/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
.venv-respo/bin/python -m pip install mpi4py-mpich==3.1.5 joblib==1.6.0
```

GPU access for RCRL and MPI local sockets for RESPO require execution outside
this workspace's sandbox. Detached tmux sessions keep runs independent of the
interactive connection; the host must remain running.
