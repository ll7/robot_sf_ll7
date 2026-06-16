# Issue #2941: Native Forecast-Variant Replay

## Summary

Issue #2941 tracks making the live forecast replay gate native enough to stop
reporting purely blocked/diagnostic-only for the forecast-variant path.  The
work adds a config-first `forecast_variant` selector and a baseline
`ProbabilisticPredictor` implementation, then wires the gate to run a minimal
forecast-aware brake replay so each variant produces its own closed-loop
metrics instead of copying the recorded baseline.  Classification compares
`cv` against the same integrated replay surface with forecast braking disabled,
while the recorded closed-loop metrics remain in the report for provenance.

## Changes

- `robot_sf/gym_env/unified_config.py`: added `forecast_variant` field to
  `RobotSimulationConfig` (inherited by the deprecated `EnvSettings`) with
  validation for `none`/`cv`/`semantic`/`interaction_aware`/`risk_filtered`.
- `robot_sf/nav/baseline_probabilistic_predictor.py`: new
  `BaselineProbabilisticPredictor` class implementing the
  `ProbabilisticPredictor` protocol for all baseline variants.
- `robot_sf/benchmark/live_forecast_replay_gate.py`: added
  `run_variant_closed_loop_replay` using a forecast-brake policy, and updated
  the gate to compute per-variant closed-loop metrics.  The claim boundary and
  limitations now describe the minimal native replay policy honestly.
- Tests updated/added:
  - `tests/benchmark/test_live_forecast_replay_gate.py`
  - `tests/validation/test_validate_live_forecast_replay_gate.py`
  - `tests/nav/test_baseline_probabilistic_predictor.py`

## Evidence

- Smoke report (none + cv):
  [`evidence/issue_2941_native_forecast_replay_2026-06-16/live_forecast_replay_gate_report.json`](evidence/issue_2941_native_forecast_replay_2026-06-16/live_forecast_replay_gate_report.json)
- Full-matrix report (all five variants):
  [`evidence/issue_2941_native_forecast_replay_full_matrix_2026-06-16/live_forecast_replay_gate_report.json`](evidence/issue_2941_native_forecast_replay_full_matrix_2026-06-16/live_forecast_replay_gate_report.json)

Generated from the motion-rich fixture
`tests/fixtures/analysis_workbench/simulation_trace_export_v1/dense_pedestrian_stress_episode_0000.json`
(scenario `dense_pedestrian_stress`, seed 2765, planner `hybrid_rule_v0_minimal`).

## Result

- Classification: `native`
- Full-matrix expansion recommended: `true`
- Native path eligibility: `live_path_available=true`, no missing components,
  `forecast_variant` config key present,
  `BaselineProbabilisticPredictor` registered.
- Closed-loop metrics differ between `none` and `cv`:
  - `none` integrated no-forecast replay: collision `false`, progress `0.545 m`,
    stop/yield `0.4 s`
  - `cv` forecast-brake replay: collision `false`, progress `0.0 m`,
    stop/yield `2.0 s`

## Claim Boundary

This is reproducible evidence that the cv forecast **can influence** closed-loop
metrics through a minimal replay policy.  It does **not** claim that cv improves
safety, success, or runtime in a production planner stack.  The replay policy is
a deliberately simple forecast-brake heuristic, not a benchmark-ready planner.

## Validation

```bash
uv run ruff check \
  robot_sf/gym_env/unified_config.py \
  robot_sf/nav/baseline_probabilistic_predictor.py \
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

## Limitations

- The closed-loop replay uses a minimal forecast-brake policy, not a production
  planner.
- Open-loop forecast metrics are still computed from a single frame per trace.
- The `risk_filtered` variant uses a deterministic distance-based relevance
  filter; it is not a learned or calibrated risk model.

## Follow-Up

- Replace the minimal brake replay with integration into a real planner that
  consumes `ProbabilisticPredictor` baselines.
- Run a proper benchmark campaign to measure whether forecast variants improve
  safety/success/runtime in the full stack.
- Consider adding `forecast_variant` to training/evaluation YAML templates once
  a real planner integration exists.
