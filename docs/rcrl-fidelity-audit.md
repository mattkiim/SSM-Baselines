# RCRL implementation fidelity audit

Audited 2026-09-14 against `Reachability_Constrained_RL` commit
`83993a775081359275bae6dc77b28b0deb5bb499`, the official quadrotor implementation.
The reviewed RAC learner, policy distribution, reference configuration, and trainer
match the recorded source hashes of both completed training batches.

## Conclusion

The core reachability, actor, multiplier, and temperature objectives match the
reference formulas. There is no evidence of a missing reference value clamp,
multiplier cap, temperature cap, or default observation normalization. Several
implementation and execution differences remain; none has yet been established
as the cause of Swimmer's divergence. No training behavior or existing results
were changed during this audit.

## Findings

| Component | Reference | Current port | Assessment |
|---|---|---|---|
| Observation preprocessing | Quadrotor scale vector is all ones | Native observations | Identity preprocessing matches; task observation distributions differ |
| Reward preprocessing | `(reward + 0) * 1` | Native reward | Identity preprocessing matches; task reward scales differ |
| Safety encoding | Sign of maximum constraint, scaled by 20; zero stays zero | `20 * sign(transition velocity margin)` | Encoding matches; transition constraint is the velocity-task adaptation |
| Safety target | Terminal: h; otherwise `(1-gamma_h)h + gamma_h max(h,Qh_next)`, gamma_h=1 | Same with true-termination mask | Formula matches |
| Actor objective | Mean of `alpha*log_pi - min(Q1,Q2) + stop_gradient(lambda)*Qh` | Same | Matches; reference's clipped actor diagnostic does not clip the actual Qh penalty |
| Multiplier | State-dependent softplus MLP, uncapped; residual clipped to [-10,100] | Same | Matches |
| Temperature | Optimize log-alpha using `-log_alpha * stop_gradient(log_pi + target_entropy)` | `log_alpha * stop_gradient(entropy-target_entropy)` | Algebraically the same gradient; no sign error |
| Temperature sampling | Separate sample from the pre-update policy | Reuses actor-loss sample from pre-update policy | Same expected objective, different gradient correlation |
| Target-action sampling | One sample shared by reward and safety targets | Separate samples for the two targets | Same marginal target objectives, different correlation |
| Target entropy | Fixed -2 for 2-action quadrotor | Minus action dimension | Matches Swimmer (-2); task adaptation for other robots |
| Networks | Two 256-unit ELU layers, He-normal weights, zero biases | Same | Initialization family matches; framework RNG draws differ |
| Actor heads | One output layer split into mean/log-std | Separate mean/log-std output layers | Equivalent parameterization, different RNG initialization layout |
| Distribution | Tanh Gaussian, log-std clipped to [-5,1], action range 1 | Same normalized distribution | Matches; environment wrapper rescales to native action bounds |
| Gradient clipping | Global norm 10 per reward critic, safety critic, actor, alpha; 3 for multiplier | Same | Matches for the two-critic configuration |
| Adam | TensorFlow 2.5 Adam, epsilon-hat 1e-7 | Optax Adam, epsilon 1e-7 after second-moment bias correction | Numerical mismatch; equal epsilon literals do not give equal updates |
| Delayed update phase | Global iteration starts at 0; actor/targets/alpha on applications 1,5,9,...; multiplier 13,25,... | Learner counter starts at 1; actor/targets/alpha on 4,8,12,...; multiplier 12,24,... | Periods match, phase differs |
| Target averaging | Tau .005 on actor update iterations, using updated critics | Same cadence relative to actor updates | Matches apart from phase |
| Learning rates | Linear decay over 2M optimizer applications, adjusted for each optimizer's period | Same endpoints over 2,990,001 learner updates | Documented duration adaptation, not exact reference schedule |
| Reward target termination | Always bootstraps, including done transitions | Stops at true terminations | Documented semantic difference; does not directly distinguish Swimmer because it never truly terminates |
| Execution/replay | 8 workers, 12 learners, 8 buffers; asynchronous gradients; 500K capacity per buffer; 3K replay-start threshold per buffer; uniform sampling | One synchronous learner per run, 1M buffer, 10K random warmup, one update per environment step; uniform sampling | Material protocol difference; reference has no fixed environment-step-to-update ratio |
| Initial penalty use | Asynchronous learners initially compute unconstrained actor gradients until ascent is enabled after penalty_start=0 | Safety penalty active from first scheduled actor update | Early transient difference; queued gradients make exact duration timing-dependent |

### Adam epsilon mismatch

Using moments m and v, TensorFlow 2.5 computes the equivalent update

`-lr * m_hat / (sqrt(v_hat) + epsilon_hat / sqrt(1-beta2^t))`.

Optax computes

`-lr * m_hat / (sqrt(v_hat) + epsilon)`.

At the first update, with beta2=.999, epsilon=1e-7, unit learning rate, and a
scalar gradient of 1e-7, the TensorFlow formula gives about -0.03065 while the
installed Optax gives about -0.50000. This is a deliberately small-gradient
example, not a measurement of typical policy gradients. The discrepancy shrinks
as the second-moment bias correction approaches one. Its training impact is
unmeasured. A TensorFlow-compatible Adam update would improve numerical fidelity
without introducing a new RCRL objective.

Source: [TensorFlow 2.5 Adam implementation](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/optimizer_v2/adam.py).
Installed Optax source: `optax/_src/transform.py`, `scale_by_adam`.

### Episode ending: correction to the earlier hypothesis

The reference worker forwards environment done directly into replay; the learner
uses it for the safety target. In the simulator fork linked by the reference
README, the time-limit termination blocks in `benchmark_env.py` and
`gym_pybullet_drones/quadrotor.py` are commented out. Out-of-bounds termination
remains enabled by the reference training configuration.

Therefore, terminating Swimmer's safety bootstrap at 1,000 steps is not justified
as a reproduction of this quadrotor implementation. The port does reset the
simulator at the velocity environment's time limit while retaining the actual
pre-reset successor for bootstrapping. Swimmer has no true terminations, unlike
the reference quadrotor's out-of-bounds endings. That is a task difference, not
proof of a mask bug. The external fork was inspected on its current
`dev_ydj_drone` branch; the RCRL README does not pin its historical revision.

Sources: [reference worker](../Reachability_Constrained_RL/worker.py),
[reference safety target](../Reachability_Constrained_RL/learners/sac.py),
[simulator time-limit code](https://github.com/ManUtdMoon/safe-control-gym/blob/dev_ydj_drone/safe_control_gym/envs/benchmark_env.py#L410).

### Logging defect

With the default warmup, learner update number is `environment_step - 9999`.
All logged environment steps (multiples of 400) therefore have learner update
number 1 modulo 4. The actor updates only at 0 modulo 4. Consequently the CSV's
`actor_loss`, `entropy`, `logp_mean`, and `temperature_loss` fields contain skipped
update placeholders at every default logging step. They cannot be used to infer
zero entropy or zero actor gradients. Some multiplier diagnostics also contain
placeholders on skipped multiplier updates; lambda mean/min/max are explicitly
recomputed by the trainer.

The logged alpha on skipped actor updates is the actual temperature. Safety
critics update on every learner step, so their large logged values and losses
are not placeholders. Evaluation returns and costs are also unaffected.

## Implications for instability

Three Swimmer seeds have peak batch-mean safety values above one million even
though the constraint encoding has magnitude 20. The unbounded linear safety
critic, gamma_h=1 backup, uncapped multiplier and uncapped temperature are all
present in the reference. The reference does not supply an omitted value clamp
that can simply be restored. Finite gradient clipping does not bound predicted
values. The backup can propagate overestimates; this is a plausible mechanism,
not a demonstrated causal explanation of these runs.

There is no missing default normalization switch: the reference's observation
and reward scaling is identity. Choosing new task-specific normalization or
reward scaling would be an additional experiment, not a reference bug fix.

## Faithful next steps

1. Correct diagnostic sampling so actual actor-update entropy, losses, action
   saturation and update counters can be observed. This changes logging only.
2. In a separately identified compatibility configuration, match TensorFlow
   Adam epsilon semantics, update phase and independent temperature sampling.
   Preserve existing configurations/checkpoint loading and numerical results.
3. Add formula and optimizer parity tests, then compare across seeds before
   claiming any stability improvement. A full TensorFlow-vs-JAX single-update
   comparison with shared weights, batches and random samples is still needed.
4. Treat schedule duration, replay/execution model, and reward termination masks
   as explicit experiment choices. Reproducing them requires more than changing
   a few constants. Retain final-step reporting and separate checkpoint selection.

Discount changes, safety-value bounds, multiplier/temperature caps and PID
updates were not introduced or recommended as fidelity fixes.

## Validation and scope

- 13 existing reachability/reference/evaluation tests passed on CPU.
- Direct checks matched safety-target and temperature-gradient formulas;
  quantified the Adam first-step discrepancy; and verified update/logging phases.
- [Numerical checks](../logs/rcrl_fidelity_audit/numerical_checks.txt).
- [Test output](../logs/rcrl_fidelity_audit/tests.log).
- Source inspection covers reference `worker.py`, `preprocessor.py`, `model.py`,
  `policy.py`, `learners/sac.py`, `optimizer.py`, `buffer.py`, and training defaults.
- This is source and formula validation, not an executed TensorFlow reproduction
  or a causal ablation. No new training jobs were launched.
