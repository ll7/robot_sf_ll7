# MPC Clearance Sampler Smoke Retune (2026-04-29)

## Scope

Retune `mpc_clearance_sampler_v1` only enough to determine whether the smoke-stage
low-progress failure was caused by an overly conservative open-space speed cap.

## Change

`configs/policy_search/candidates/mpc_clearance_sampler_v1.yaml` now sets
`max_linear_speed: 1.8` for the candidate override. The clearance, obstacle, and
pedestrian-risk weights are unchanged.

## Commands

```bash
uv run pytest tests/validation/test_policy_search_common.py tests/validation/test_run_policy_search_candidate.py tests/planner/test_nmpc_social.py tests/planner/test_predictive_mppi_planner.py
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate mpc_clearance_sampler_v1 --stage smoke --workers 1
```

## Result

- Unit and runner helper tests: `28 passed`.
- Smoke summary: `output/policy_search/mpc_clearance_sampler_v1/smoke/latest/summary.json`.
- Runner report: `docs/context/policy_search/reports/2026-04-29_mpc_clearance_sampler_v1_smoke.md`.
- Smoke episodes: `3`.
- Success rate: `1.0000`.
- Collision rate: `0.0000`.
- Near-miss rate: `0.0000`.
- Mean average speed: `1.7896`.
- Failure taxonomy: no failures recorded.

## Interpretation

The prior smoke failure was consistent with insufficient route progress on the
80-step `planner_sanity_simple` slice, not an integration or solver failure. The
speed-cap retune is enough to clear smoke, but it is not nominal-sanity evidence.
The next local step is `nominal_sanity` for `mpc_clearance_sampler_v1`.
