# Evaluation outputs

Generated evaluation results are collected in `results/evaluations/`, which is
ignored by Git. Existing folder names and all files have been preserved.

| Previous location | Current location |
| --- | --- |
| `eval_f16/` | `results/evaluations/eval_f16/` |
| `eval_quad2d/` | `results/evaluations/eval_quad2d/` |
| `eval_quad2d_shared_inits/` | `results/evaluations/eval_quad2d_shared_inits/` |
| `eval_quad2d_shared_inits_cont/` | `results/evaluations/eval_quad2d_shared_inits_cont/` |
| `eval_quad2d_shared_inits_cont_test/` | `results/evaluations/eval_quad2d_shared_inits_cont_test/` |
| `eval_quad3d/` | `results/evaluations/eval_quad3d/` |
| `eval_quad3d_stab/` | `results/evaluations/eval_quad3d_stab/` |
| `eval_quad3d_stab_inits/` | `results/evaluations/eval_quad3d_stab_inits/` |
| `eval_quad3d_stab_inits2/` | `results/evaluations/eval_quad3d_stab_inits2/` |
| `eval_rac_quad2d_stab/` | `results/evaluations/eval_rac_quad2d_stab/` |
| `eval_states/` | `results/evaluations/eval_states/` |
| `Vh_figures/` | `results/evaluations/Vh_figures/` |
| `results/quad3d_results/` | `results/evaluations/quad3d_results/` |

Evaluation scripts retain their original locations. Their default output paths
and the local progression scripts now point into `results/evaluations/`. Run
commands from the repository root. For custom destinations, use the script's
`--out_dir` or `--out` option.

Saved JSON/NPZ contents are unchanged, so historical paths recorded inside them
may still refer to the previous locations. Use the mapping above when reopening
those artifacts. Training checkpoints remain in their existing run directories.
