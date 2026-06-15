# Forecast Calibration Report

- Report id: issue_2865_forecast_calibration_report_2026-06-15
- Decision: wait
- Claim status: diagnostic-only
- Reliability rows: 40
- Limitation rows: 10

| scenario_family | horizon_s | observation_tier | predictor_family | actor_class | semantic_metadata | coverage | miss_rate | status | sharpness | failure_taxonomy | eligibility |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |
| bottleneck | 1 | deployable_tracked | cv | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| bottleneck | 1 | deployable_tracked | goal_aware | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| bottleneck | 1 | deployable_tracked | interaction_aware | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| bottleneck | 1 | deployable_tracked | semantic | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| bottleneck | 1 | deployable_tracked | signal_aware | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| corridor_interaction | 1 | deployable_tracked | cv | unavailable | absent | 1 | 0 | under_confident_over_coverage | 0.0769366 | over_coverage | diagnostic_only_actor_class_unavailable |
| corridor_interaction | 1 | deployable_tracked | goal_aware | unavailable | absent | 1 | 0 | under_confident_over_coverage | 0.0769366 | over_coverage | diagnostic_only_actor_class_unavailable |
| corridor_interaction | 1 | deployable_tracked | interaction_aware | unavailable | absent | 1 | 0 | under_confident_over_coverage | 0.113911 | over_coverage | diagnostic_only_actor_class_unavailable |
| corridor_interaction | 1 | deployable_tracked | semantic | unavailable | absent | 1 | 0 | under_confident_over_coverage | 0.0769366 | over_coverage | diagnostic_only_actor_class_unavailable |
| corridor_interaction | 1 | deployable_tracked | signal_aware | unavailable | absent | 1 | 0 | under_confident_over_coverage | 0.0769366 | over_coverage | diagnostic_only_actor_class_unavailable |
| crossing_proxy | 1 | deployable_tracked | cv | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| crossing_proxy | 1 | deployable_tracked | goal_aware | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| crossing_proxy | 1 | deployable_tracked | interaction_aware | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| crossing_proxy | 1 | deployable_tracked | semantic | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| crossing_proxy | 1 | deployable_tracked | signal_aware | unavailable | absent | NA | NA | unavailable | NA | unavailable_uncertainty_denominator | blocked_no_denominator |
| goal_directed_crossing | 1 | deployable_tracked | cv | unavailable | present | 1 | 0 | under_confident_over_coverage | 1.11022e-16 | over_coverage | diagnostic_only_actor_class_unavailable |
| goal_directed_crossing | 1 | deployable_tracked | goal_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.12 | over_coverage | diagnostic_only_actor_class_unavailable |
| goal_directed_crossing | 1 | deployable_tracked | interaction_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 1.11022e-16 | over_coverage | diagnostic_only_actor_class_unavailable |
| goal_directed_crossing | 1 | deployable_tracked | semantic | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.12 | over_coverage | diagnostic_only_actor_class_unavailable |
| goal_directed_crossing | 1 | deployable_tracked | signal_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 1.11022e-16 | over_coverage | diagnostic_only_actor_class_unavailable |
| occluded_emergence | 1 | deployable_tracked | cv | unavailable | absent | 1 | 0 | under_confident_over_coverage | 2.31296e-17 | over_coverage | diagnostic_only_actor_class_unavailable |
| occluded_emergence | 1 | deployable_tracked | goal_aware | unavailable | absent | 1 | 0 | under_confident_over_coverage | 2.31296e-17 | over_coverage | diagnostic_only_actor_class_unavailable |
| occluded_emergence | 1 | deployable_tracked | interaction_aware | unavailable | absent | 1 | 0 | under_confident_over_coverage | 2.31296e-17 | over_coverage | diagnostic_only_actor_class_unavailable |
| occluded_emergence | 1 | deployable_tracked | semantic | unavailable | absent | 1 | 0 | under_confident_over_coverage | 2.31296e-17 | over_coverage | diagnostic_only_actor_class_unavailable |
| occluded_emergence | 1 | deployable_tracked | signal_aware | unavailable | absent | 1 | 0 | under_confident_over_coverage | 2.31296e-17 | over_coverage | diagnostic_only_actor_class_unavailable |
| route_conflict_goal | 1 | deployable_tracked | cv | unavailable | present | 1 | 0 | under_confident_over_coverage | 0 | over_coverage | diagnostic_only_actor_class_unavailable |
| route_conflict_goal | 1 | deployable_tracked | goal_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.107703 | over_coverage | diagnostic_only_actor_class_unavailable |
| route_conflict_goal | 1 | deployable_tracked | interaction_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0 | over_coverage | diagnostic_only_actor_class_unavailable |
| route_conflict_goal | 1 | deployable_tracked | semantic | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.107703 | over_coverage | diagnostic_only_actor_class_unavailable |
| route_conflict_goal | 1 | deployable_tracked | signal_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0 | over_coverage | diagnostic_only_actor_class_unavailable |
| signalized_crossing | 1 | deployable_tracked | cv | unavailable | present | 1 | 0 | under_confident_over_coverage | 0 | over_coverage | diagnostic_only_actor_class_unavailable |
| signalized_crossing | 1 | deployable_tracked | goal_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0 | over_coverage | diagnostic_only_actor_class_unavailable |
| signalized_crossing | 1 | deployable_tracked | interaction_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0 | over_coverage | diagnostic_only_actor_class_unavailable |
| signalized_crossing | 1 | deployable_tracked | semantic | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.42 | over_coverage | diagnostic_only_actor_class_unavailable |
| signalized_crossing | 1 | deployable_tracked | signal_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.42 | over_coverage | diagnostic_only_actor_class_unavailable |
| waiting_with_intent_change | 1 | deployable_tracked | cv | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.45 | over_coverage | diagnostic_only_actor_class_unavailable |
| waiting_with_intent_change | 1 | deployable_tracked | goal_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.45 | over_coverage | diagnostic_only_actor_class_unavailable |
| waiting_with_intent_change | 1 | deployable_tracked | interaction_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.45 | over_coverage | diagnostic_only_actor_class_unavailable |
| waiting_with_intent_change | 1 | deployable_tracked | semantic | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.45 | over_coverage | diagnostic_only_actor_class_unavailable |
| waiting_with_intent_change | 1 | deployable_tracked | signal_aware | unavailable | present | 1 | 0 | under_confident_over_coverage | 0.45 | over_coverage | diagnostic_only_actor_class_unavailable |

## Limitations

- bottleneck / 1s / deployable_tracked / cv: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy
- bottleneck / 1s / deployable_tracked / goal_aware: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy
- bottleneck / 1s / deployable_tracked / interaction_aware: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy
- bottleneck / 1s / deployable_tracked / semantic: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy
- bottleneck / 1s / deployable_tracked / signal_aware: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy
- crossing_proxy / 1s / deployable_tracked / cv: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy
- crossing_proxy / 1s / deployable_tracked / goal_aware: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy
- crossing_proxy / 1s / deployable_tracked / interaction_aware: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy
- crossing_proxy / 1s / deployable_tracked / semantic: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy
- crossing_proxy / 1s / deployable_tracked / signal_aware: unavailable uncertainty metrics: coverage, likelihood, sharpness_proxy

Calibration summaries are analysis-only evidence. They do not prove planner safety or navigation benefit, and unavailable uncertainty denominators remain limitations.
