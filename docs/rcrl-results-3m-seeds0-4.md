# RCRL velocity results: 3M steps, seeds 0–4

All 20 requested runs completed successfully. Each run has 120 checkpoints at 25,000-step intervals, including step 3,000,000 (2,400 checkpoints total).

## Evaluation and aggregation

These results use the final 3,000,000-step evaluation for every run, without selecting the best checkpoint. Each seed result is the mean of 50 deterministic evaluation episodes using environment seeds 42–91, capped at 1,000 steps. All runs use the reference-style RCRL protocol.

Aggregates give equal weight to training seeds 0–4. The ± values are sample standard deviations across the five seed means (ddof=1), not standard errors or episode-level standard deviations. Return is undiscounted episode reward; cost is undiscounted episode safety cost, with lower cost preferred. Violation rate is the mean per-episode fraction of steps with a safety violation. Short episodes can reduce total cost, so episode length and violation rate are also reported. Evaluation seeds were reused during training; these are not held-out test results.

## Aggregate final results

| Task | Seeds | Return, mean ± SD | Cost, mean ± SD | Mean episode length | Mean violation rate |
|---|---:|---:|---:|---:|---:|
| Humanoid | 5 | 5,766.32 ± 647.12 | 15.94 ± 35.51 | 983.52 | 1.605% |
| Ant | 5 | 2,418.28 ± 1,285.60 | 82.23 ± 183.79 | 1,000.00 | 8.223% |
| Swimmer | 5 | -5.40 ± 14.53 | 104.69 ± 209.80 | 1,000.00 | 10.469% |
| Hopper | 5 | 558.58 ± 448.79 | 16.61 ± 23.39 | 504.81 | 15.241% |

## Per-seed final results

| Task | Seed | Return | Cost | Episode length | Violation rate | Source |
|---|---:|---:|---:|---:|---:|---|
| Humanoid | 0 | 5,354.56 | 0.00 | 1,000.00 | 0.000% | [history](../results/SafetyHumanoidVelocity-v1/humanoid_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0000/history.csv) |
| Humanoid | 1 | 4,862.70 | 0.14 | 926.54 | 0.062% | [history](../results/SafetyHumanoidVelocity-v1/humanoid_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0001/history.csv) |
| Humanoid | 2 | 5,947.87 | 0.10 | 991.04 | 0.018% | [history](../results/SafetyHumanoidVelocity-v1/humanoid_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0002/history.csv) |
| Humanoid | 3 | 6,414.64 | 79.46 | 1,000.00 | 7.946% | [history](../results/SafetyHumanoidVelocity-v1/humanoid_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0003/history.csv) |
| Humanoid | 4 | 6,251.83 | 0.00 | 1,000.00 | 0.000% | [history](../results/SafetyHumanoidVelocity-v1/humanoid_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0004/history.csv) |
| Ant | 0 | 3,082.44 | 0.04 | 1,000.00 | 0.004% | [history](../results/SafetyAntVelocity-v1/ant_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0000/history.csv) |
| Ant | 1 | 3,152.36 | 411.00 | 1,000.00 | 41.100% | [history](../results/SafetyAntVelocity-v1/ant_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0001/history.csv) |
| Ant | 2 | 2,421.32 | 0.08 | 1,000.00 | 0.008% | [history](../results/SafetyAntVelocity-v1/ant_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0002/history.csv) |
| Ant | 3 | 192.96 | 0.00 | 1,000.00 | 0.000% | [history](../results/SafetyAntVelocity-v1/ant_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0003/history.csv) |
| Ant | 4 | 3,242.34 | 0.04 | 1,000.00 | 0.004% | [history](../results/SafetyAntVelocity-v1/ant_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0004/history.csv) |
| Swimmer | 0 | -15.62 | 3.88 | 1,000.00 | 0.388% | [history](../results/SafetySwimmerVelocity-v1/swimmer_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0000/history.csv) |
| Swimmer | 1 | -3.44 | 0.00 | 1,000.00 | 0.000% | [history](../results/SafetySwimmerVelocity-v1/swimmer_rcrl_reference_3m_20260913T234937Z/2026-09-13_seed0001/history.csv) |
| Swimmer | 2 | 17.12 | 478.74 | 1,000.00 | 47.874% | [history](../results/SafetySwimmerVelocity-v1/swimmer_rcrl_reference_3m_20260913T234937Z/2026-09-13_seed0002/history.csv) |
| Swimmer | 3 | -4.57 | 40.70 | 1,000.00 | 4.070% | [history](../results/SafetySwimmerVelocity-v1/swimmer_rcrl_reference_3m_20260913T234937Z/2026-09-13_seed0003/history.csv) |
| Swimmer | 4 | -20.50 | 0.14 | 1,000.00 | 0.014% | [history](../results/SafetySwimmerVelocity-v1/swimmer_rcrl_reference_3m_20260913T234937Z/2026-09-13_seed0004/history.csv) |
| Hopper | 0 | 1,039.44 | 0.00 | 1,000.00 | 0.000% | [history](../results/SafetyHopperVelocity-v1/hopper_rcrl_reference_3m_20260913T100023Z/2026-09-13_seed0000/history.csv) |
| Hopper | 1 | 1,049.06 | 0.00 | 1,000.00 | 0.000% | [history](../results/SafetyHopperVelocity-v1/hopper_rcrl_reference_3m_20260913T234937Z/2026-09-13_seed0001/history.csv) |
| Hopper | 2 | 153.26 | 49.64 | 80.16 | 61.921% | [history](../results/SafetyHopperVelocity-v1/hopper_rcrl_reference_3m_20260913T234937Z/2026-09-13_seed0002/history.csv) |
| Hopper | 3 | 344.00 | 33.22 | 233.22 | 14.244% | [history](../results/SafetyHopperVelocity-v1/hopper_rcrl_reference_3m_20260913T234937Z/2026-09-13_seed0003/history.csv) |
| Hopper | 4 | 207.12 | 0.20 | 210.66 | 0.038% | [history](../results/SafetyHopperVelocity-v1/hopper_rcrl_reference_3m_20260913T234937Z/2026-09-13_seed0004/history.csv) |

## Observations

- Humanoid seeds 0, 1, 2, and 4 have final mean costs at or below 0.14; seed 3 has higher return but cost 79.46.
- Ant varies substantially: seed 1 has cost 411.00, while seed 3 has return 192.96 despite zero cost.
- Swimmer has negative final mean returns for four seeds. Seed 2 has positive return 17.12 but cost 478.74 and a 47.874% violation rate.
- Hopper seeds 0 and 1 reach the 1,000-step evaluation cap with zero cost. Seeds 2–4 have shorter episodes and lower returns; seed 2 violates the constraint on about 61.9% of episode steps.

## Artifacts

- [Original batch manifest](../logs/rcrl_batch_20260913T100023Z/manifest.json): Ant/Humanoid seeds 0–4 and Swimmer/Hopper seed 0.
- [Additional batch manifest](../logs/rcrl_batch_20260913T234937Z/manifest.json): Swimmer/Hopper seeds 1–4.
- Each linked history file shares its directory with `config.json` and `checkpoints/ckpt_3000000.msgpack`. Retain `config.json` when copying checkpoints.
- [Reference protocol](rcrl-reference-velocity.md).
