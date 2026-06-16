# Issue #2944: Native CV-Only Closed-Loop Replay Smoke

## Summary

Issue #2944 narrows the issue #2941 native forecast-variant replay goal to the
smallest useful smoke: compare `none` and `cv`, compute the required
closed-loop metric surface, and decide whether the full forecast variant matrix
should expand.

The current result is **blocked**, not native evidence. The gate now defaults to
the `none` + `cv` smoke variants and reports one of `native`, `blocked`,
`degraded`, or `diagnostic_only`. On the current repository state it fails
closed as `blocked` because the live planner path still lacks the components
needed for a selectable forecast variant to influence closed-loop behavior.

## Evidence

- Smoke report JSON:
  [`evidence/issue_2944_native_cv_smoke_2026-06-16/live_forecast_replay_gate_report.json`](evidence/issue_2944_native_cv_smoke_2026-06-16/live_forecast_replay_gate_report.json)
- Smoke report Markdown:
  [`evidence/issue_2944_native_cv_smoke_2026-06-16/live_forecast_replay_gate_report.md`](evidence/issue_2944_native_cv_smoke_2026-06-16/live_forecast_replay_gate_report.md)

The report was generated from:

```bash
python scripts/validation/validate_live_forecast_replay_gate.py \
  --trace tests/fixtures/analysis_workbench/simulation_trace_export_v1/dense_pedestrian_stress_episode_0000.json \
  --generated-at-utc 2026-06-16T11:05:00+00:00 \
  --output-dir output/live_forecast_replay_gate_smoke
```

## Result

- Classification: `blocked`
- Full-matrix expansion recommended: `false`
- Required issue #2944 metric surface present for both smoke variants:
  `collision`, `near_miss`, `min_distance`, `stop_yield_timing`, `progress`,
  and `runtime`.
- Closed-loop metric source: `baseline_recorded_trace` for both `none` and
  `cv`, because `cv` is not yet consumed by a native planner.

The blocked components are:

- no `ProbabilisticPredictor` implementation registered for baseline forecast
  variants;
- no environment or planner config key for selecting `forecast_variant`.

## Claim Boundary

This is a reproducible smoke gate and negative/blocked result, not benchmark
evidence that CV changes behavior. It should gate the full forecast-variant
matrix by keeping expansion parked until the native planner/environment path can
consume selectable forecasts and produce per-variant closed-loop metrics without
fallback.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- python -m ruff check \
  robot_sf/benchmark/live_forecast_replay_gate.py \
  scripts/validation/validate_live_forecast_replay_gate.py \
  tests/benchmark/test_live_forecast_replay_gate.py \
  tests/validation/test_validate_live_forecast_replay_gate.py

scripts/dev/run_worktree_shared_venv.sh -- python -m pytest \
  tests/benchmark/test_live_forecast_replay_gate.py \
  tests/validation/test_validate_live_forecast_replay_gate.py \
  -n auto -x
```

Observed focused result: `41 passed`.

## Next Step

Proceed to issue #2941 only after a native planner/environment path exposes a
`forecast_variant` selector and a baseline `ProbabilisticPredictor` implementation
that can influence closed-loop actions. Until then, full-matrix expansion should
remain blocked.
