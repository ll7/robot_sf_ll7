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

## Result (Updated by #2941)

The original #2944 result was `blocked`.  Issue #2941 subsequently added the
missing components and the gate now reports `native` on the same fixture:

- Classification: `native`
- Full-matrix expansion recommended: `true`
- Native path eligibility: `live_path_available=true`, no missing components,
  `forecast_variant` config key present,
  `BaselineProbabilisticPredictor` registered.
- Closed-loop metrics now differ between `none` and `cv`:
  - `none`: collision `true`, progress `1.9 m`, stop/yield `0 s`
  - `cv`: collision `false`, progress `0.545 m`, stop/yield `0.4 s`

See [`issue_2941_native_forecast_replay.md`](issue_2941_native_forecast_replay.md)
for the implementation details and current evidence.

## Claim Boundary

The original #2944 result was a reproducible smoke gate and negative/blocked
result, not benchmark evidence that CV changes behavior.  After #2941, the gate
now demonstrates that CV **can influence** closed-loop metrics via a minimal
forecast-brake replay policy, but it still does not claim that CV improves
safety, success, or runtime in a production planner stack.

## Validation

```bash
uv run ruff check \
  robot_sf/gym_env/unified_config.py \
  robot_sf/nav/baseline_probabilistic_predictor.py \
  robot_sf/nav/__init__.py \
  robot_sf/benchmark/live_forecast_replay_gate.py \
  scripts/validation/validate_live_forecast_replay_gate.py \
  tests/benchmark/test_live_forecast_replay_gate.py \
  tests/validation/test_validate_live_forecast_replay_gate.py \
  tests/nav/test_baseline_probabilistic_predictor.py

uv run pytest \
  tests/benchmark/test_live_forecast_replay_gate.py \
  tests/validation/test_validate_live_forecast_replay_gate.py \
  tests/nav/test_baseline_probabilistic_predictor.py \
  -n auto -x
```

Observed focused result: `56 passed`.

## Next Step

Issue #2941 has implemented the unblocking layer.  The next step is to replace
the minimal forecast-brake replay with a real planner integration and run a
proper benchmark campaign to measure whether forecast variants improve
safety/success/runtime in the full stack.
