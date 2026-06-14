# Semantic Forecast Baseline Comparison

## Claim Boundary

**Diagnostic-only, not paper-facing evidence.** Compares constant-velocity and semantic forecast baselines on bounded repository trace fixtures.

## Reproducibility

- **Issue:** #2758
- **Generated at (UTC):** 2026-06-14T23:18:12.251491+00:00
- **Command:** `uv run python scripts/benchmark/run_cv_forecast_eval.py --output-dir docs/context/evidence/issue_2758_semantic_forecast_baselines_2026-06-14 --compare-all`
- **Repo HEAD:** `9ae4a6eb`
- **Forecast horizons:** [0.5, 1.0, 2.0]

## Fixture Metadata Assessment

- **Fixtures have signal metadata:** False
- **Fixtures have intent metadata:** False

Existing durable fixtures lack signal and intent metadata. Semantic conditioning baselines run correctly but their calibration/collision-relevance improvement cannot be measured until fixtures include this metadata.

## Baseline Summary

| Baseline | Evaluated Traces | Total Samples |
|----------|------------------|---------------|
| cv | 3 | 71 |
| signal_aware | 3 | 71 |
| goal_aware | 3 | 71 |
| semantic | 3 | 71 |

## Comparison Table

| Baseline | Family | Label | Status | Samples | ADE 1s | NLL 1s | Miss Rate 1s |
|----------|--------|-------|--------|---------|--------|--------|-------------|
| cv | corridor_interaction | default_social_force | evaluated | 30 | 0.0769 | 1.9522 | 0.00% |
| cv | corridor_interaction | ammv_social_force | evaluated | 30 | 0.0769 | 1.9522 | 0.00% |
| cv | crossing_proxy | synthetic_crossing_proxy_orca | limited_no_pedestrian_motion | 0 | - | - | - |
| cv | bottleneck | minimal_fixture | limited_no_pedestrian_motion | 0 | - | - | - |
| cv | occluded_emergence | deterministic_occluded_emergence | evaluated | 11 | 0.0000 | 1.9355 | 0.00% |
| signal_aware | corridor_interaction | default_social_force | evaluated | 30 | 0.0769 | 1.9522 | 0.00% |
| signal_aware | corridor_interaction | ammv_social_force | evaluated | 30 | 0.0769 | 1.9522 | 0.00% |
| signal_aware | crossing_proxy | synthetic_crossing_proxy_orca | limited_no_pedestrian_motion | 0 | - | - | - |
| signal_aware | bottleneck | minimal_fixture | limited_no_pedestrian_motion | 0 | - | - | - |
| signal_aware | occluded_emergence | deterministic_occluded_emergence | evaluated | 11 | 0.0000 | 1.9355 | 0.00% |
| goal_aware | corridor_interaction | default_social_force | evaluated | 30 | 0.0769 | 1.6715 | 0.00% |
| goal_aware | corridor_interaction | ammv_social_force | evaluated | 30 | 0.0769 | 1.6715 | 0.00% |
| goal_aware | crossing_proxy | synthetic_crossing_proxy_orca | limited_no_pedestrian_motion | 0 | - | - | - |
| goal_aware | bottleneck | minimal_fixture | limited_no_pedestrian_motion | 0 | - | - | - |
| goal_aware | occluded_emergence | deterministic_occluded_emergence | evaluated | 11 | 0.0000 | 1.6493 | 0.00% |
| semantic | corridor_interaction | default_social_force | evaluated | 30 | 0.0769 | 2.4701 | 0.00% |
| semantic | corridor_interaction | ammv_social_force | evaluated | 30 | 0.0769 | 2.4701 | 0.00% |
| semantic | crossing_proxy | synthetic_crossing_proxy_orca | limited_no_pedestrian_motion | 0 | - | - | - |
| semantic | bottleneck | minimal_fixture | limited_no_pedestrian_motion | 0 | - | - | - |
| semantic | occluded_emergence | deterministic_occluded_emergence | evaluated | 11 | 0.0000 | 2.4602 | 0.00% |