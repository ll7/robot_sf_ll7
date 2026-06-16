# Issue #2960 Forecast Planner Consumer Smoke (2026-06-16)

Issue #2960 moved `forecast_variant` beyond replay-only wiring by making
`PredictionPlannerAdapter` consume `BaselineProbabilisticPredictor` outputs in
its normal sampled-rollout pedestrian-future path when `forecast_variant` is
non-`none`.

## Evidence Boundary

This is **smoke evidence**, not nominal benchmark evidence. It proves that a real
planner consumer can read forecast variants through the config-first
`SocNavPlannerConfig.forecast_variant` path and score the resulting pedestrian
futures. It does not prove that any forecast variant improves safety, success,
runtime, progress, or paper-facing benchmark performance.

## Implementation

- `robot_sf/planner/socnav.py`: `SocNavPlannerConfig` now carries
  `forecast_variant`, forecast horizons, forecast timestep, and risk-distance
  settings. `PredictionPlannerAdapter` builds `BaselineProbabilisticPredictor`
  for non-`none` variants and feeds those predicted trajectories into
  `_predict_trajectories`, the same future-pedestrian path used by the
  sampled-rollout scorer.
- `scripts/validation/validate_forecast_planner_consumer.py`: deterministic
  same-seed smoke over `none`, `cv`, `semantic`, `interaction_aware`, and
  `risk_filtered`, emitting JSON and Markdown.
- `tests/test_socnav_planner_adapter.py`: focused regression test showing
  `interaction_aware` forecasts change the planner-consumed pedestrian futures
  versus the no-forecast constant-velocity path.

## Smoke Result

Durable report:

- [`evidence/issue_2960_forecast_planner_consumer_2026-06-16/forecast_planner_consumer_smoke.json`](evidence/issue_2960_forecast_planner_consumer_2026-06-16/forecast_planner_consumer_smoke.json)
- [`evidence/issue_2960_forecast_planner_consumer_2026-06-16/forecast_planner_consumer_smoke.md`](evidence/issue_2960_forecast_planner_consumer_2026-06-16/forecast_planner_consumer_smoke.md)

Classification by variant on the deterministic motion-rich fixture:

| Variant | Classification | Reason |
|---|---|---|
| `none` | `native` | Native no-forecast planner baseline. |
| `cv` | `degraded` | Predictor built and consumed, but consumed futures match the no-forecast baseline in this fixture. |
| `semantic` | `degraded` | Predictor built and consumed, but no semantic context is available in the fixture, so futures match baseline. |
| `interaction_aware` | `native` | Predictor built and consumed; nearby pedestrians change the future trajectories scored by the planner. |
| `risk_filtered` | `degraded` | Predictor built and consumed, but risk filtering widens uncertainty only; planner mean futures match baseline in this fixture. |

The smoke reports the required metric fields: collision, near miss, minimum
distance, stop/yield timing, progress, false-positive stops, and runtime.

## Claim Map Update

`cm-v0.prediction.full_planner_integration` moves from `blocked` to `smoke`.
`cm-v0.prediction.native_replay` remains diagnostic because the earlier replay
gate still uses a minimal brake heuristic. The new planner-consumer smoke does
not promote the forecast lane to nominal or benchmark evidence; that requires a
full episode benchmark or scenario-matrix run with durable per-episode outcomes.

## Validation

```bash
uv run pytest tests/test_socnav_planner_adapter.py::test_prediction_adapter_consumes_configured_forecast_variant -q
uv run python scripts/validation/validate_forecast_planner_consumer.py --output-dir output/issue_2960_forecast_planner_consumer_smoke
uv run ruff check scripts/validation/validate_forecast_planner_consumer.py robot_sf/planner/socnav.py tests/test_socnav_planner_adapter.py
```
