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
heads, JAX initialization and optimizer numerics differ; stochastic actions in
individual loss computations are sampled separately. No bitwise parity is claimed.

Existing configurations retain their prior behavior. Output directories are
`results/Safety{Ant,Humanoid}Velocity-v1/{ant,humanoid}_rcrl_reference_3m/DATE_seed0000/`.
Each contains config.json, history.csv, training_metrics.csv and checkpoints.
The launch log directory contains commands, source hashes and exit-status files.
