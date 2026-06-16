# Issue #2902: Prediction - Live Same-Seed Forecast Replay Gate v1

## Summary

Implemented the smallest valuable executable same-seed live forecast replay gate
for comparing forecast variants (`none`, CV, semantic, interaction-aware,
risk-filtered) on a motion-rich pedestrian fixture. The gate is diagnostic-only
because the repository does not yet expose a planner that consumes selectable
baseline forecast variants via the `ProbabilisticPredictor` protocol.

## Evidence

- Gate report (JSON):
  [`evidence/issue_2902_live_forecast_replay_gate_2026-06-16/live_forecast_replay_gate_report.json`](evidence/issue_2902_live_forecast_replay_gate_2026-06-16/live_forecast_replay_gate_report.json)
- Gate report (Markdown):
  [`evidence/issue_2902_live_forecast_replay_gate_2026-06-16/live_forecast_replay_gate_report.md`](evidence/issue_2902_live_forecast_replay_gate_2026-06-16/live_forecast_replay_gate_report.md)

The evidence was generated on the motion-rich
`tests/fixtures/analysis_workbench/simulation_trace_export_v1/dense_pedestrian_stress_episode_0000.json`
fixture (scenario `dense_pedestrian_stress`, seed 2765, planner
`hybrid_rule_v0_minimal`).

## Key Findings

- **Native live path: blocked.** The gate detects two missing components:
  - No `ProbabilisticPredictor` implementation is registered for the baseline
    forecast variants.
  - No environment or planner config key (`forecast_variant`) exists to select
    the forecast variant at runtime.
- **Baseline closed-loop metrics from the recorded trace:**
  - Collision: `true`
  - Near-miss timesteps: `9`
  - Min robot-pedestrian distance: `0.1562 m`
  - Stop/yield steps: `0`
  - Progress: `1.9 m`
  - False-positive stops: `0`
  - Runtime: `1.9 s`
- **Forecast variant comparison:** The `none` variant is represented by the
  recorded baseline trace metrics; CV, semantic, interaction-aware, and
  risk-filtered variants produce valid `ForecastBatch.v1` artifacts. Open-loop
  forecast metrics are computed from the first trace frame; horizons were
  automatically clamped from `[0.5, 1.0, 2.0]` to `[0.5, 1.0]` because the
  1.9-second fixture does not contain a 2.0-second future frame.

## Claim Boundary

This is a **diagnostic-only** gate. It does not prove that any forecast variant
improves navigation safety, success, or runtime. Closed-loop metrics are
properties of the recorded trace (the `none` variant) and do not vary by
forecast variant because no native planner consumes the non-`none` variants.

## Reproducible Command

```bash
uv run python scripts/validation/validate_live_forecast_replay_gate.py \
  --trace tests/fixtures/analysis_workbench/simulation_trace_export_v1/dense_pedestrian_stress_episode_0000.json \
  --full-matrix \
  --output-dir output/live_forecast_replay_gate
```

The validation CLI now defaults to the issue #2944 `none` + `cv` smoke. Use
`--full-matrix` when reproducing the original issue #2902 five-variant evidence.

## Validation

```bash
uv run pytest tests/benchmark/test_live_forecast_replay_gate.py
uv run pytest tests/benchmark/test_pedestrian_forecast.py tests/benchmark/test_pedestrian_forecast_cv_eval.py
uv run ruff check robot_sf/benchmark/live_forecast_replay_gate.py \
  robot_sf/benchmark/pedestrian_forecast.py \
  scripts/validation/validate_live_forecast_replay_gate.py \
  tests/benchmark/test_live_forecast_replay_gate.py \
  tests/benchmark/test_pedestrian_forecast.py \
  tests/benchmark/test_pedestrian_forecast_cv_eval.py
uv run ruff format robot_sf/benchmark/live_forecast_replay_gate.py \
  robot_sf/benchmark/pedestrian_forecast.py \
  scripts/validation/validate_live_forecast_replay_gate.py \
  tests/benchmark/test_live_forecast_replay_gate.py \
  tests/benchmark/test_pedestrian_forecast.py \
  tests/benchmark/test_pedestrian_forecast_cv_eval.py
```

## Limitations

- The gate cannot compare variant effects on closed-loop metrics until a
  planner/environment exposes a selectable `forecast_variant` config and a
  `ProbabilisticPredictor` implementation for the baseline families.
- Open-loop forecast metrics are computed from a single frame per trace by
  default.
- The `risk_filtered` variant uses a deterministic distance-based relevance
  filter; it is not a learned or calibrated risk model.
- The `risk_filtered` function is intentionally not registered in the generic
  `BASELINE_FUNCTIONS` map because generic forecast evaluation does not provide
  robot position and would otherwise degrade it to plain CV while reporting an
  evaluated row.

## Follow-Up Target

The next proof step is issue-class "planner integration": implement or wire an
existing planner to consume `ProbabilisticPredictor` baseline variants so the
same-seed replay can actually vary closed-loop behavior and produce the required
metrics (collision, near miss, min distance, stop/yield timing, progress,
false-positive stops, runtime) per variant. Until that path exists, the gate
must remain diagnostic-only and fail closed.
