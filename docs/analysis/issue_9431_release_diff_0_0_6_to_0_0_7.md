# Issue #9431 release diff: 0.0.6 → corrected 0.0.7

This report pairs rows by planner, scenario, and seed. Successor execution is admitted only when every row is native, adapter, or mixed mode with no fallback/degraded row. Trace-dependent goal-adjacent timeout labels remain `unavailable` when step traces were not recorded.

- predecessor archive SHA-256: `61b865fdde65455a39a68221d7c65b0eff315bfa51b4c0bfe34aed3c5d4f3e8e`
- exact source range: `31cdfe0361abe2c520117a17f99c1b7a0aba4359..07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- successor source commit: `07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- successor publication bundle SHA-256: `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`
- paired rows: **20160**; changed outcome rows: **622**
- successor execution audit: **pass**

## Per-arm outcome counts

| arm | paired | 0.0.6 success | 0.0.7 success | 0.0.6 collisions | 0.0.7 collisions | 0.0.6 timeouts | 0.0.7 timeouts | goal-adjacent labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `goal` | 1440 | 561 | 561 | 843 | 843 | 36 | 36 | old `unavailable=1440`; new `unavailable=1440` |
| `guarded_ppo` | 1440 | 329 | 332 | 506 | 513 | 605 | 595 | old `unavailable=1440`; new `unavailable=1440` |
| `hybrid_rule_v3_fast_progress_static_escape` | 1440 | 1281 | 1277 | 73 | 74 | 86 | 89 | old `unavailable=1440`; new `unavailable=1440` |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | 1440 | 1279 | 1275 | 50 | 50 | 111 | 115 | old `unavailable=1440`; new `unavailable=1440` |
| `orca` | 1440 | 1209 | 1206 | 194 | 197 | 37 | 37 | old `unavailable=1440`; new `unavailable=1440` |
| `ppo` | 1440 | 796 | 714 | 609 | 685 | 35 | 41 | old `unavailable=1440`; new `unavailable=1440` |
| `prediction_planner` | 1440 | 799 | 790 | 617 | 622 | 24 | 28 | old `unavailable=1440`; new `unavailable=1440` |
| `predictive_mppi` | 1440 | 157 | 96 | 977 | 1001 | 306 | 343 | old `unavailable=1440`; new `unavailable=1440` |
| `risk_dwa` | 1440 | 101 | 57 | 1253 | 1294 | 86 | 89 | old `unavailable=1440`; new `unavailable=1440` |
| `sacadrl` | 1440 | 448 | 445 | 990 | 993 | 2 | 2 | old `unavailable=1440`; new `unavailable=1440` |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | 1440 | 1279 | 1277 | 72 | 73 | 89 | 90 | old `unavailable=1440`; new `unavailable=1440` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 1440 | 1281 | 1277 | 72 | 73 | 87 | 90 | old `unavailable=1440`; new `unavailable=1440` |
| `social_force` | 1440 | 51 | 30 | 785 | 782 | 604 | 628 | old `unavailable=1440`; new `unavailable=1440` |
| `socnav_sampling` | 1440 | 500 | 497 | 940 | 940 | 0 | 3 | old `unavailable=1440`; new `unavailable=1440` |

## Seed-block bootstrap intervals

The interval is a deterministic 95% percentile bootstrap over whole seed blocks (3000 replicates, seed 123). The estimand is the risk difference 0.0.7 minus 0.0.6.

| arm | metric | observed delta | 95% interval | seed blocks |
|---|---|---:|---|---:|
| `goal` | collision_event | 0.000000 | [0.000000, 0.000000] | 30 |
| `goal` | route_complete | 0.000000 | [0.000000, 0.000000] | 30 |
| `goal` | timeout_event | 0.000000 | [0.000000, 0.000000] | 30 |
| `guarded_ppo` | collision_event | 0.004861 | [-0.008333, 0.018056] | 30 |
| `guarded_ppo` | route_complete | 0.002083 | [-0.006944, 0.011111] | 30 |
| `guarded_ppo` | timeout_event | -0.006944 | [-0.018750, 0.004167] | 30 |
| `hybrid_rule_v3_fast_progress_static_escape` | collision_event | 0.000694 | [0.000000, 0.002083] | 30 |
| `hybrid_rule_v3_fast_progress_static_escape` | route_complete | -0.002778 | [-0.006250, 0.000000] | 30 |
| `hybrid_rule_v3_fast_progress_static_escape` | timeout_event | 0.002083 | [0.000000, 0.005556] | 30 |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | collision_event | 0.000000 | [0.000000, 0.000000] | 30 |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | route_complete | -0.002778 | [-0.007639, 0.001389] | 30 |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | timeout_event | 0.002778 | [-0.001389, 0.007639] | 30 |
| `orca` | collision_event | 0.002083 | [0.000000, 0.005556] | 30 |
| `orca` | route_complete | -0.002083 | [-0.005556, 0.000000] | 30 |
| `orca` | timeout_event | 0.000000 | [0.000000, 0.000000] | 30 |
| `ppo` | collision_event | 0.052778 | [0.034722, 0.069444] | 30 |
| `ppo` | route_complete | -0.056944 | [-0.074306, -0.038194] | 30 |
| `ppo` | timeout_event | 0.004167 | [-0.004861, 0.013889] | 30 |
| `prediction_planner` | collision_event | 0.003472 | [0.000694, 0.006250] | 30 |
| `prediction_planner` | route_complete | -0.006250 | [-0.010417, -0.002778] | 30 |
| `prediction_planner` | timeout_event | 0.002778 | [0.000000, 0.006250] | 30 |
| `predictive_mppi` | collision_event | 0.016667 | [0.005538, 0.029167] | 30 |
| `predictive_mppi` | route_complete | -0.042361 | [-0.062500, -0.025000] | 30 |
| `predictive_mppi` | timeout_event | 0.025694 | [0.017361, 0.035417] | 30 |
| `risk_dwa` | collision_event | 0.028472 | [0.016649, 0.040972] | 30 |
| `risk_dwa` | route_complete | -0.030556 | [-0.044444, -0.017361] | 30 |
| `risk_dwa` | timeout_event | 0.002083 | [0.000000, 0.004861] | 30 |
| `sacadrl` | collision_event | 0.002083 | [0.000000, 0.004167] | 30 |
| `sacadrl` | route_complete | -0.002083 | [-0.004167, 0.000000] | 30 |
| `sacadrl` | timeout_event | 0.000000 | [0.000000, 0.000000] | 30 |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | collision_event | 0.000694 | [0.000000, 0.002083] | 30 |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | route_complete | -0.001389 | [-0.004167, 0.001389] | 30 |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | timeout_event | 0.000694 | [-0.001389, 0.003472] | 30 |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | collision_event | 0.000694 | [0.000000, 0.002083] | 30 |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | route_complete | -0.002778 | [-0.006250, 0.000000] | 30 |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | timeout_event | 0.002083 | [0.000000, 0.005556] | 30 |
| `social_force` | collision_event | -0.002083 | [-0.004861, 0.000000] | 30 |
| `social_force` | route_complete | -0.014583 | [-0.024306, -0.004861] | 30 |
| `social_force` | timeout_event | 0.016667 | [0.007639, 0.025694] | 30 |
| `socnav_sampling` | collision_event | 0.000000 | [0.000000, 0.000000] | 30 |
| `socnav_sampling` | route_complete | -0.002083 | [-0.004167, 0.000000] | 30 |
| `socnav_sampling` | timeout_event | 0.002083 | [0.000000, 0.004167] | 30 |

## Interpretation boundary

The full release rows are outcome evidence only. Because they do not contain simulation-step traces, the goal-adjacent timeout predicate is not recomputed here and `unavailable` is not counted as false. The separately pinned 0.0.7 worked-example trace bundle supplies the bounded head-on/group-crossing diagnostic traces.
