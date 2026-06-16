# Horizon x Timestep Denominator Health Report

## Claim Boundary

**analysis_only_not_navigation_evidence: this report measures denominator coverage of the horizon x timestep ablation matrix. It does not change forecast defaults, prove navigation value, closed-loop benefit, safety improvement, or benchmark-strength predictor quality.**

## Reproducibility

- **Issue:** #2937
- **Parent ablation issue:** #2837
- **Generated at (UTC):** 2026-06-16T00:00:00+00:00
- **Command:** `uv run python scripts/benchmark/build_horizon_timestep_denominator_report.py --issue 2937 --parent-issue 2837 --generated-at-utc 2026-06-16T00:00:00+00:00`
- **Repo HEAD:** `ae463792`
- **Horizon ladder (s):** [0.5, 1.0, 1.6, 2.0, 3.0]
- **dt ladder (s):** [0.1, 0.2, 0.4, 0.5]
- **Trace families:** 9
- **Expected total cells:** 180

## Category Totals

| category | count | fraction |
| --- | ---: | ---: |
| trace_too_short | 16 | 8.9% |
| no_pedestrian_motion | 0 | 0.0% |
| metadata_missing | 0 | 0.0% |
| actor_class_missing | 0 | 0.0% |
| observation_tier_missing | 0 | 0.0% |
| other_explicit_reason | 0 | 0.0% |
| evaluated | 164 | 91.1% |

## Matrix Coverage (horizon_s x dt_s)

| horizon_s | dt_s | evaluated/total | fraction | trace_too_short | no_pedestrian_motion | metadata_missing | actor_class_missing | observation_tier_missing | other_explicit_reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.1 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.5 | 0.2 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.5 | 0.4 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.5 | 0.5 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0.1 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0.2 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0.4 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0.5 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.6 | 0.1 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.6 | 0.2 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.6 | 0.4 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.6 | 0.5 | 9/9 | 100.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0.1 | 7/9 | 77.8% | 2 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0.2 | 7/9 | 77.8% | 2 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0.4 | 7/9 | 77.8% | 2 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0.5 | 7/9 | 77.8% | 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0.1 | 7/9 | 77.8% | 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0.2 | 7/9 | 77.8% | 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0.4 | 7/9 | 77.8% | 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0.5 | 7/9 | 77.8% | 2 | 0 | 0 | 0 | 0 | 0 |

## Per-Family Missingness

| family | label | total | evaluated | trace_too_short | no_pedestrian_motion | metadata_missing | actor_class_missing | observation_tier_missing | other_explicit_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bottleneck | minimal_fixture | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| corridor_interaction | ammv_social_force | 20 | 12 | 8 | 0 | 0 | 0 | 0 | 0 |
| corridor_interaction | default_social_force | 20 | 12 | 8 | 0 | 0 | 0 | 0 | 0 |
| crossing_proxy | synthetic_crossing_proxy_orca | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| goal_directed_crossing | goal_directed_crossing_fixture | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| occluded_emergence | deterministic_occluded_emergence | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| route_conflict_goal | route_conflict_goal_fixture | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| signalized_crossing | signalized_crossing_semantic_metadata | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| waiting_with_intent_change | waiting_intent_change_fixture | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |

## Spot Checks (one per missingness category)

| reason | family | label | horizon_s | dt_s | status | detail |
| --- | --- | --- | ---: | ---: | --- | --- |
| trace_too_short | corridor_interaction | default_social_force | 2 | 0.1 | horizon_longer_than_trace | Horizon 2s at dt 0.1s requires 20 steps, but resampled trace has only 20 frames. |

## Minimum Fixture Additions for 90% Coverage

- **Current coverage:** 164/180 (91.1%)
- **Target coverage:** 162/180 (90%)
- **Additional cells needed:** 0
- **No-motion families to replace:** [] (0 blocked cells)
- **Short families to extend:** ['corridor_interaction'] (16 blocked cells)
- **No-motion fixtures to replace:** []
- **Short fixtures observed:** ['corridor_interaction/ammv_social_force', 'corridor_interaction/default_social_force']

### Minimum Set Estimate

- **Fixture changes required:** 0
- **Replace these no-motion fixtures:** []
- **Extend these short fixtures:** []
- **Estimated additional cells:** 0
- **Estimated coverage:** 91.1%

### Full-Extension Estimate

- Replacing all no-motion families and extending all short families to cover the full horizon ladder would yield approximately 180 evaluated cells (100.0%).

> Estimates assume replacements are motion-rich and extensions cover the missing horizons. Actual coverage depends on the generated fixtures.

## Interpretation

This report is a denominator-health audit for the horizon x timestep ablation.  Forecast defaults are explicitly not changed by this report alone.  After the issue #2937 fixture repairs, remaining missing cells are concentrated in short corridor_interaction traces; no no-motion, metadata, actor-class, or observation-tier gaps were observed in the current durable fixture set.  The proposed fixture additions are planning estimates and must be validated by actually generating or extending the relevant traces.