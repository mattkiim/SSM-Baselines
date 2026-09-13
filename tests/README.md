# Smoke checks

Run these standalone checks from the repository root in an environment with the
project's runtime dependencies installed:

```bash
python -m tests.smoke.smoke_test_rac_learner
python -m tests.smoke.smoke_test_cal_learner
python -m tests.smoke.smoke_test_sac_cbf_learner
python -m tests.smoke.smoke_test_sac_lag_learner
python -m tests.smoke.smoke_test_safety_env
```

The learner scripts were moved unchanged from `jaxrl5/agents/*/smoke_test_*.py`.
The environment check was moved unchanged from `test_env.py`; it requires
Safety-Gymnasium and MuJoCo and inspects simulator internals.

These scripts execute their checks through `main()` and are not pytest test
functions. A successful import or pytest collection does not run the checks.

## RCRL regression checks

```bash
JAX_PLATFORMS=cpu python -m unittest tests.test_reachability tests.test_rcrl_reference tests.test_evaluation
```

The optional `tests.test_respo_velocity` check needs the reference checkout and
patch described in [portable setup](../docs/portable-setup.md); it skips if absent.
