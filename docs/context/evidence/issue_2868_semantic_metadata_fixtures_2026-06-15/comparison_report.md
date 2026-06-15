# Forecast Baseline Comparison

## Claim Boundary

**Diagnostic-only, not paper-facing evidence.** Compares constant-velocity, semantic, and interaction-aware forecast baselines on bounded repository trace fixtures when those baselines are requested.

## Reproducibility

- **Issue:** #2868
- **Generated at (UTC):** 2026-06-15T00:00:00+00:00
- **Command:** `uv run python scripts/benchmark/run_cv_forecast_eval.py --output-dir docs/context/evidence/issue_2868_semantic_metadata_fixtures_2026-06-15 --compare-all --issue 2868 --generated-at-utc 2026-06-15T00:00:00+00:00`
- **Repo HEAD:** `f983398d`
- **Forecast horizons:** [0.5, 1.0, 2.0]

## Fixture Metadata Assessment

- **Fixtures have signal metadata:** True
- **Fixtures have intent metadata:** True

## Metadata Coverage

- **Rows with metadata:** 20
- **Rows without metadata:** 25
- **Rows with signal metadata:** 20
- **Rows with intent metadata:** 15

## Interaction-Aware Diagnostic Effect

- **Matched evaluated rows:** 7
- **Mean ADE 1s delta vs CV:** 0.0106
- **Mean NLL 1s delta vs CV:** -0.5196
- **Conclusion:** Interaction-aware heuristic improved Gaussian likelihood/calibration proxy but worsened point accuracy on matched diagnostic rows; revise before closed-loop coupling claims.

## Baseline Summary

| Baseline | Evaluated Traces | Total Samples |
|----------|------------------|---------------|
| cv | 7 | 99 |
| goal_aware | 7 | 99 |
| interaction_aware | 7 | 99 |
| semantic | 7 | 99 |
| signal_aware | 7 | 99 |

## Comparison Table

| Baseline | Family | Label | Metadata | Status | Samples | ADE 1s | NLL 1s | Miss Rate 1s |
|----------|--------|-------|----------|--------|---------|--------|--------|-------------|
| cv | corridor_interaction | default_social_force | absent | evaluated | 30 | 0.0769 | 1.9522 | 0.00% |
| cv | corridor_interaction | ammv_social_force | absent | evaluated | 30 | 0.0769 | 1.9522 | 0.00% |
| cv | crossing_proxy | synthetic_crossing_proxy_orca | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| cv | bottleneck | minimal_fixture | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| cv | occluded_emergence | deterministic_occluded_emergence | absent | evaluated | 11 | 0.0000 | 1.9355 | 0.00% |
| cv | signalized_crossing | signalized_crossing_semantic_metadata | present | evaluated | 7 | 0.0000 | 1.9355 | 0.00% |
| cv | goal_directed_crossing | goal_directed_crossing_fixture | present | evaluated | 7 | 0.0000 | 1.1245 | 0.00% |
| cv | waiting_with_intent_change | waiting_intent_change_fixture | present | evaluated | 7 | 0.4500 | 1.3321 | 0.00% |
| cv | route_conflict_goal | route_conflict_goal_fixture | present | evaluated | 7 | 0.0000 | 1.9355 | 0.00% |
| goal_aware | corridor_interaction | default_social_force | absent | evaluated | 30 | 0.0769 | 1.6715 | 0.00% |
| goal_aware | corridor_interaction | ammv_social_force | absent | evaluated | 30 | 0.0769 | 1.6715 | 0.00% |
| goal_aware | crossing_proxy | synthetic_crossing_proxy_orca | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| goal_aware | bottleneck | minimal_fixture | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| goal_aware | occluded_emergence | deterministic_occluded_emergence | absent | evaluated | 11 | 0.0000 | 1.6493 | 0.00% |
| goal_aware | signalized_crossing | signalized_crossing_semantic_metadata | present | evaluated | 7 | 0.0000 | 1.6493 | 0.00% |
| goal_aware | goal_directed_crossing | goal_directed_crossing_fixture | present | evaluated | 7 | 0.1200 | 1.1392 | 0.00% |
| goal_aware | waiting_with_intent_change | waiting_intent_change_fixture | present | evaluated | 7 | 0.4500 | 1.3321 | 0.00% |
| goal_aware | route_conflict_goal | route_conflict_goal_fixture | present | evaluated | 7 | 0.1077 | 1.1364 | 0.00% |
| interaction_aware | corridor_interaction | default_social_force | absent | evaluated | 30 | 0.1139 | 1.3501 | 0.00% |
| interaction_aware | corridor_interaction | ammv_social_force | absent | evaluated | 30 | 0.1139 | 1.3501 | 0.00% |
| interaction_aware | crossing_proxy | synthetic_crossing_proxy_orca | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| interaction_aware | bottleneck | minimal_fixture | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| interaction_aware | occluded_emergence | deterministic_occluded_emergence | absent | evaluated | 11 | 0.0000 | 1.1245 | 0.00% |
| interaction_aware | signalized_crossing | signalized_crossing_semantic_metadata | present | evaluated | 7 | 0.0000 | 1.1245 | 0.00% |
| interaction_aware | goal_directed_crossing | goal_directed_crossing_fixture | present | evaluated | 7 | 0.0000 | 1.1245 | 0.00% |
| interaction_aware | waiting_with_intent_change | waiting_intent_change_fixture | present | evaluated | 7 | 0.4500 | 1.3321 | 0.00% |
| interaction_aware | route_conflict_goal | route_conflict_goal_fixture | present | evaluated | 7 | 0.0000 | 1.1245 | 0.00% |
| semantic | corridor_interaction | default_social_force | absent | evaluated | 30 | 0.0769 | 2.4701 | 0.00% |
| semantic | corridor_interaction | ammv_social_force | absent | evaluated | 30 | 0.0769 | 2.4701 | 0.00% |
| semantic | crossing_proxy | synthetic_crossing_proxy_orca | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| semantic | bottleneck | minimal_fixture | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| semantic | occluded_emergence | deterministic_occluded_emergence | absent | evaluated | 11 | 0.0000 | 2.4602 | 0.00% |
| semantic | signalized_crossing | signalized_crossing_semantic_metadata | present | evaluated | 7 | 0.4200 | 1.7558 | 0.00% |
| semantic | goal_directed_crossing | goal_directed_crossing_fixture | present | evaluated | 7 | 0.1200 | 1.1392 | 0.00% |
| semantic | waiting_with_intent_change | waiting_intent_change_fixture | present | evaluated | 7 | 0.4500 | 1.3321 | 0.00% |
| semantic | route_conflict_goal | route_conflict_goal_fixture | present | evaluated | 7 | 0.1077 | 1.9407 | 0.00% |
| signal_aware | corridor_interaction | default_social_force | absent | evaluated | 30 | 0.0769 | 1.9522 | 0.00% |
| signal_aware | corridor_interaction | ammv_social_force | absent | evaluated | 30 | 0.0769 | 1.9522 | 0.00% |
| signal_aware | crossing_proxy | synthetic_crossing_proxy_orca | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| signal_aware | bottleneck | minimal_fixture | absent | limited_no_pedestrian_motion | 0 | - | - | - |
| signal_aware | occluded_emergence | deterministic_occluded_emergence | absent | evaluated | 11 | 0.0000 | 1.9355 | 0.00% |
| signal_aware | signalized_crossing | signalized_crossing_semantic_metadata | present | evaluated | 7 | 0.4200 | 1.3045 | 0.00% |
| signal_aware | goal_directed_crossing | goal_directed_crossing_fixture | present | evaluated | 7 | 0.0000 | 1.1245 | 0.00% |
| signal_aware | waiting_with_intent_change | waiting_intent_change_fixture | present | evaluated | 7 | 0.4500 | 1.3321 | 0.00% |
| signal_aware | route_conflict_goal | route_conflict_goal_fixture | present | evaluated | 7 | 0.0000 | 1.9355 | 0.00% |