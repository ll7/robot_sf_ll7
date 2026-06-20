# Forecast Baseline Comparison (#2915)

## Claim Boundary

Analysis-only diagnostic evidence. This compares deterministic forecast baselines on a bounded set of durable Robot SF trace fixtures. It is not planner-promotion or paper-facing benchmark evidence.

## Reproducibility

- Issue: #2915
- Generated at UTC: `2026-06-20T21:50:00Z`
- Git HEAD: `bf81a40e62be7975afe1a6dff432cfeb6c887c51`
- Config: `configs/research/forecast_baseline_comparison_issue_2915.yaml`
- Command: `scripts/benchmark/run_forecast_baseline_comparison.py --config configs/research/forecast_baseline_comparison_issue_2915.yaml --date 2026-06-20 --generated-at-utc 2026-06-20T21:50:00Z`
- Evidence tier: `analysis_only`

## Strongest Baseline By Family

| Scenario family | Strongest baseline | Note |
| --- | --- | --- |
| bottleneck | not available | no evaluated baseline rows |
| corridor_interaction | cv | ranked on primary metrics, then ADE/FDE |
| crossing_proxy | not available | no evaluated baseline rows |
| dense_pedestrian_interaction | not available | no evaluated baseline rows |
| goal_directed_crossing | cv | ranked on primary metrics, then ADE/FDE |
| occluded_emergence | cv | ranked on primary metrics, then ADE/FDE |
| route_conflict_goal | cv | ranked on primary metrics, then ADE/FDE |
| signalized_crossing | cv | ranked on primary metrics, then ADE/FDE |
| waiting_with_intent_change | cv | ranked on primary metrics, then ADE/FDE |

## Aggregate Rows

| Scenario family | Baseline | Status | Evaluated rows | Excluded rows | Miss rate | Calibration error | Collision relevance error | Planner risk error | ADE | FDE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bottleneck | cv | not_available | 0 | 1 |  |  |  |  |  |  |
| bottleneck | interaction_aware | not_available | 0 | 1 |  |  |  |  |  |  |
| bottleneck | semantic | not_available | 0 | 1 |  |  |  |  |  |  |
| corridor_interaction | cv | evaluated | 2 | 0 | 0.0 | 0.050000000000000044 | 0.0 | 0.0 | 0.05060846568931593 | 0.07693663848497238 |
| corridor_interaction | interaction_aware | evaluated | 2 | 0 | 0.0 | 0.050000000000000044 | 0.0 | 0.0 | 0.07949928954877997 | 0.11391100018251157 |
| corridor_interaction | semantic | evaluated | 2 | 0 | 0.0 | 0.050000000000000044 | 0.0 | 0.0 | 0.05060846568931593 | 0.07693663848497238 |
| crossing_proxy | cv | not_available | 0 | 1 |  |  |  |  |  |  |
| crossing_proxy | interaction_aware | not_available | 0 | 1 |  |  |  |  |  |  |
| crossing_proxy | semantic | not_available | 0 | 1 |  |  |  |  |  |  |
| dense_pedestrian_interaction | cv | not_available | 0 | 1 |  |  |  |  |  |  |
| dense_pedestrian_interaction | interaction_aware | not_available | 0 | 1 |  |  |  |  |  |  |
| dense_pedestrian_interaction | semantic | not_available | 0 | 1 |  |  |  |  |  |  |
| goal_directed_crossing | cv | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 1.0 | 1.0 | 8.723180907769087e-17 | 1.1102230246251565e-16 |
| goal_directed_crossing | interaction_aware | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 1.0 | 1.0 | 8.723180907769087e-17 | 1.1102230246251565e-16 |
| goal_directed_crossing | semantic | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 1.0 | 1.0 | 0.09000000000000001 | 0.12 |
| occluded_emergence | cv | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 0.045454545454545456 | 0.045454545454545456 | 1.9134525613804783e-17 | 2.312964634635743e-17 |
| occluded_emergence | interaction_aware | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 0.045454545454545456 | 0.045454545454545456 | 1.9134525613804783e-17 | 2.312964634635743e-17 |
| occluded_emergence | semantic | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 0.045454545454545456 | 0.045454545454545456 | 1.9134525613804783e-17 | 2.312964634635743e-17 |
| route_conflict_goal | cv | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 1.0 | 1.0 | 9.912705577010326e-18 | 0.0 |
| route_conflict_goal | interaction_aware | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 1.0 | 1.0 | 9.912705577010326e-18 | 0.0 |
| route_conflict_goal | semantic | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 1.0 | 1.0 | 0.08077747210701756 | 0.10770329614269006 |
| signalized_crossing | cv | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 1.0 | 1.0 | 3.172065784643304e-17 | 0.0 |
| signalized_crossing | interaction_aware | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 1.0 | 1.0 | 3.172065784643304e-17 | 0.0 |
| signalized_crossing | semantic | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 1.0 | 1.0 | 0.27 | 0.41999999999999993 |
| waiting_with_intent_change | cv | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 0.7857142857142857 | 0.7857142857142857 | 0.285 | 0.44999999999999996 |
| waiting_with_intent_change | interaction_aware | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 0.7857142857142857 | 0.7857142857142857 | 0.285 | 0.44999999999999996 |
| waiting_with_intent_change | semantic | evaluated | 1 | 0 | 0.0 | 0.050000000000000044 | 0.7857142857142857 | 0.7857142857142857 | 0.2978571428571428 | 0.44999999999999996 |

## Learned-Predictor Gap

Residual learned-predictor gap remains diagnostic-only: learned predictor expansion should stay gated until this comparison is repeated on broader fixed-scope trace coverage with non-degenerate crossing, bottleneck, and dense-interaction families.

## Limitations

- Rows with missing files, no pedestrian motion, insufficient frames, or execution errors are excluded fail-closed and not ranked as successful evidence.
- `planner_relevant_risk_error` is the same collision-relevance forecast error surface used by the existing pedestrian forecast metric path; it is a proxy for planner-risk relevance, not closed-loop planner evidence.
