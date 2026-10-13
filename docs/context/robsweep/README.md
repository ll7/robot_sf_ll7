# Examiner robustness study — 2026-10-12

**Diagnostic-only development evidence; complete.** Requested 21,900; recorded 21,900; eligible 21,900; missing 0; excluded 0.

Seeds 1061–1090 only; host CPUs; differential drive; frozen model weights. One medium cell for each multi-cell family and five singleton Francis cells from Figure 7.3. These ten cells are a bounded sample, not the full release matrix or all parameterizations of those families. Narrow doorway is outside that figure. No manuscript claim is admitted by this report.

The comparator is a new matched control using 0.0.8 configurations on unchanged runtime source main `33a4735900153e742828227cc2171436ea09cd90`. It is not a comparison against historical release episode scores. Radius changes retain planner thresholds and PPO weights; they may also affect radius-aware spawn/goal sampling. Speed treatment is deterministic preferred speed, not a guarantee of actual walking speed during holds or interactions.

The [measured verdict](VERDICT.md) distinguishes robust leading-set and family conclusions from design-dependent strict rankings. [Reproduction](REPRODUCE.md), [study contract](PLAN.md), and [analysis test-value proof](TESTS.md) define the execution and interpretation boundaries.

Verified release facts:

| Quantity | Verified value / correction |
|---|---|
| Robot envelope / drive | 1.0 m radius disc; differential drive |
| Physical speed | default 2.0 m/s; pedestrian-overtaking scenario explicitly uses 0.7 m/s |
| Pedestrian speed | default preferred cap 1.3 × 0.5 = 0.65 m/s; actual speed can be lower. Authored actors also use 0.3, 0.7 and 0.8 m/s initial/role speeds. Therefore 0.65 is not universal across the release. |
| Narrow doorway | measured SVG gap 2.0 m; radius 1.0 gives zero diameter margin and no strictly positive-clearance crossing. The release requires 0.1 m safety margin per side. An ideal exactly tangent centerline is a mathematical limiting case, not a valid traversal under that contract. |
| Radius 0.5 / 0.3 doorway geometry | diameter margins 1.0 / 1.4 m; geometrically feasible centerline, without implying planner/crowd success |

| Planner | Release global command cap (m/s) | Requested swap (m/s) |
|---|---:|---:|
| prediction_planner | 1.6 | 2 |
| goal | 1 | 2 |
| social_force | 2 | 1.2 |
| orca | 2 | 1.2 |
| ppo | 2 | 1.2 |
| socnav_sampling | 2 | 1.2 |
| sacadrl | 2 | 1.2 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4 | 2 | 1.2 |
| scenario_adaptive_hybrid_orca_v2_collision_guard_v4 | 2 | 1.2 |
| hybrid_rule_v4_fast_progress_static_escape | 2 | 1.2 |
| hybrid_rule_v4_fast_progress_static_escape_continuous | 2 | 1.2 |
| guarded_ppo | 2 | 1.2 |
| predictive_mppi | 1.5 | 2 |
| risk_dwa | 1.2 | 2 |

The listed caps are confirmed, with two further qualifications: SA-CADRL's preferred command speed is 1.0 m/s despite its 2.0 ceiling; hybrid/guarded/predictive controllers retain conditional lower caps. The requested swap still has two cap groups. It is a cap-sensitivity check, not a fully equal-cap comparison. Swap configs change both the plant ceiling and planner ceiling; DWA's candidate count and relative spacing are retained while its velocity grid expands to the new ceiling.

Planner-rank correlations versus the matched legacy control. Radius/speed correlations use five planners; the cap swap uses all fourteen. Success and collision incidence are ranked separately; no SNQI robustness claim is made. Kendall tau-b retains ties. Brackets are paired seed-bootstrap 95% percentile intervals (1,000 resamples), conditional on defined tau draws (counts in summary.json); these quantify seed variation in these fixed cells, not scenario-selection uncertainty. Undefined means a constant/tied ordering has no tau, or the complete paired roster is unavailable.

| Radius / preferred speed or cap arm | Success tau-b | 95% CI | Collision tau-b | 95% CI | Family success tau-b |
|---|---:|---|---:|---|---:|
| swap | 0.834 | [0.622, 0.899] | 0.739 | [0.548, 0.921] | 0.956 |
| r0.3_v0.65 | 0.800 | [0.400, 0.800] | 0.949 | [0.738, 1.000] | 0.782 |
| r0.3_v1 | 0.105 | [-0.120, 0.317] | 0.359 | [0.333, 0.378] | 0.483 |
| r0.3_v1.3 | 0.316 | [0.105, 0.359] | 0.316 | [0.316, 0.333] | 0.659 |
| r0.5_v0.65 | 1.000 | [0.734, 1.000] | 0.949 | [0.892, 1.000] | 0.809 |
| r0.5_v1 | 0.400 | [0.200, 0.527] | 0.527 | [0.378, 0.707] | 0.518 |
| r0.5_v1.3 | 0.200 | [0.000, 0.316] | 0.359 | [0.359, 0.378] | 0.644 |
| r1_v0.65 | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] | 1.000 |
| r1_v1 | 1.000 | [0.600, 1.000] | 1.000 | [0.800, 1.000] | 0.809 |
| r1_v1.3 | 0.800 | [0.600, 0.950] | 0.600 | [0.600, 0.738] | 0.719 |

Five-planner aggregate success rates (300 episodes each). Bold values mark the success-leading set, retaining ties.

| Radius / preferred speed | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| r0.3_v0.65 | 93.7% | 99.3% | **100.0%** | 81.3% | 82.0% |
| r0.3_v1 | 98.7% | 99.3% | **100.0%** | **100.0%** | 81.3% |
| r0.3_v1.3 | 99.7% | 97.7% | **100.0%** | **100.0%** | 79.0% |
| r0.5_v0.65 | 94.7% | 91.7% | **100.0%** | 74.3% | 77.3% |
| r0.5_v1 | 97.3% | 90.3% | **100.0%** | 99.0% | 78.0% |
| r0.5_v1.3 | 97.7% | 83.0% | 99.7% | **100.0%** | 71.7% |
| r1_v0.65 | 92.7% | 62.7% | **94.3%** | 46.7% | 58.3% |
| r1_v1 | 80.3% | 58.7% | **85.7%** | 42.3% | 58.3% |
| r1_v1.3 | 70.7% | 47.0% | **74.0%** | 36.7% | 54.3% |

Family difficulty across the five sweep planners (150 episodes per family/treatment). Hardness rank 1 is lowest success; ties retain the minimum rank. Ranges cover the nine design settings, not confidence intervals.

| Family | Release-setting success | Success range | Release-setting collision | Collision range | Hardness rank baseline | Hardness rank range |
|---|---:|---|---:|---|---:|---|
| bottleneck | 39.3% | 34.7%–100.0% | 40.7% | 0.0%–65.3% | 2 | 2–8 |
| cross_trap | 52.7% | 52.7%–90.7% | 31.3% | 9.3%–40.7% | 3 | 1–5 |
| doorway | 70.7% | 10.0%–98.0% | 27.3% | 2.0%–78.0% | 5 | 2–7 |
| group_crossing | 92.7% | 77.3%–99.3% | 7.3% | 0.7%–14.0% | 8 | 6–8 |
| head_on_corridor | 82.7% | 62.7%–100.0% | 14.7% | 0.0%–35.3% | 6 | 6–8 |
| entering_room | 92.0% | 92.0%–98.0% | 6.0% | 0.7%–6.0% | 7 | 4–9 |
| exiting_room | 100.0% | 99.3%–100.0% | 0.0% | 0.0%–0.7% | 10 | 7–10 |
| circular_crossing | 98.0% | 96.0%–100.0% | 0.7% | 0.0%–2.7% | 9 | 7–9 |
| robot_crowding | 28.7% | 6.0%–88.7% | 46.7% | 8.0%–90.7% | 1 | 1–1 |
| crowd_navigation | 52.7% | 24.0%–95.3% | 30.7% | 4.7%–64.0% | 3 | 2–5 |

Complete-roster cap swap (each planner: 300 matched episodes; success ranks retain ties).

| Planner | Success baseline → swap | Collision baseline → swap | Success rank baseline → swap |
|---|---|---|---|
| prediction_planner | 49.7% → 53.0% | 0.7% → 1.3% | 12 → 12 |
| goal | 46.7% → 41.0% | 53.3% → 59.0% | 13 → 13 |
| social_force | 62.7% → 62.7% | 0.3% → 0.3% | 10 → 9 |
| orca | 92.7% → 90.7% | 6.7% → 8.3% | 4 → 5 |
| ppo | 58.3% → 62.7% | 41.7% → 35.3% | 11 → 9 |
| socnav_sampling | 91.3% → 93.0% | 2.3% → 2.3% | 5 → 1 |
| sacadrl | 37.3% → 37.3% | 62.7% → 62.7% | 14 → 14 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4 | 94.3% → 91.0% | 0.7% → 1.7% | 1 → 2 |
| scenario_adaptive_hybrid_orca_v2_collision_guard_v4 | 94.3% → 91.0% | 0.7% → 1.7% | 1 → 2 |
| hybrid_rule_v4_fast_progress_static_escape | 94.3% → 91.0% | 0.7% → 1.0% | 1 → 2 |
| hybrid_rule_v4_fast_progress_static_escape_continuous | 91.0% → 90.3% | 1.0% → 0.7% | 6 → 6 |
| guarded_ppo | 65.0% → 56.7% | 0.3% → 0.0% | 8 → 11 |
| predictive_mppi | 63.0% → 71.0% | 0.7% → 0.0% | 9 → 8 |
| risk_dwa | 81.0% → 88.3% | 0.0% → 0.7% | 7 → 7 |

**legacy: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 29/1 | 0/0 | 30/0 | 0/30 | 0/30 |
| cross_trap: `classic_cross_trap_medium` | 27/2 | 7/0 | 30/0 | 0/30 | 15/15 |
| doorway: `classic_doorway_medium` | 30/0 | 27/0 | 30/0 | 12/18 | 7/23 |
| group_crossing: `classic_group_crossing_medium` | 29/1 | 30/0 | 30/0 | 25/5 | 25/5 |
| head_on_corridor: `classic_head_on_corridor_medium` | 27/3 | 26/0 | 30/0 | 12/18 | 29/1 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 27/0 | 30/0 | 21/9 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 30/0 | 28/0 | 30/0 | 29/1 |
| robot_crowding: `francis2023_robot_crowding` | 20/9 | 0/1 | 21/2 | 0/30 | 2/28 |
| crowd_navigation: `francis2023_crowd_navigation` | 26/4 | 8/0 | 27/0 | 1/29 | 17/13 |

**r0.3_v0.65: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 30/0 | 30/0 | 30/0 | 0/30 | 30/0 |
| cross_trap: `classic_cross_trap_medium` | 30/0 | 30/0 | 30/0 | 29/1 | 17/13 |
| doorway: `classic_doorway_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 27/3 |
| group_crossing: `classic_group_crossing_medium` | 23/6 | 30/0 | 30/0 | 30/0 | 30/0 |
| head_on_corridor: `classic_head_on_corridor_medium` | 29/1 | 30/0 | 30/0 | 28/2 | 30/0 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 30/0 | 30/0 | 21/9 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| robot_crowding: `francis2023_robot_crowding` | 23/7 | 28/0 | 30/0 | 18/12 | 9/19 |
| crowd_navigation: `francis2023_crowd_navigation` | 26/4 | 30/0 | 30/0 | 19/11 | 22/8 |

**r0.3_v1: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| cross_trap: `classic_cross_trap_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 16/14 |
| doorway: `classic_doorway_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 21/9 |
| group_crossing: `classic_group_crossing_medium` | 29/1 | 30/0 | 30/0 | 30/0 | 30/0 |
| head_on_corridor: `classic_head_on_corridor_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 30/0 | 30/0 | 22/8 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| robot_crowding: `francis2023_robot_crowding` | 28/2 | 28/0 | 30/0 | 30/0 | 11/18 |
| crowd_navigation: `francis2023_crowd_navigation` | 29/1 | 30/0 | 30/0 | 30/0 | 24/6 |

**r0.3_v1.3: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 26/4 |
| cross_trap: `classic_cross_trap_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 13/17 |
| doorway: `classic_doorway_medium` | 29/0 | 30/0 | 30/0 | 30/0 | 19/11 |
| group_crossing: `classic_group_crossing_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 27/3 |
| head_on_corridor: `classic_head_on_corridor_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 30/0 | 30/0 | 25/5 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| robot_crowding: `francis2023_robot_crowding` | 30/0 | 26/0 | 30/0 | 30/0 | 17/12 |
| crowd_navigation: `francis2023_crowd_navigation` | 30/0 | 27/0 | 30/0 | 30/0 | 20/10 |

**r0.5_v0.65: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 30/0 | 30/0 | 30/0 | 0/30 | 30/0 |
| cross_trap: `classic_cross_trap_medium` | 30/0 | 29/0 | 30/0 | 29/1 | 16/14 |
| doorway: `classic_doorway_medium` | 30/0 | 30/0 | 30/0 | 28/2 | 22/8 |
| group_crossing: `classic_group_crossing_medium` | 24/6 | 30/0 | 30/0 | 29/1 | 28/2 |
| head_on_corridor: `classic_head_on_corridor_medium` | 29/1 | 30/0 | 30/0 | 26/4 | 30/0 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 30/0 | 30/0 | 23/7 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 30/0 | 30/0 | 30/0 | 28/2 |
| robot_crowding: `francis2023_robot_crowding` | 25/5 | 13/0 | 30/0 | 8/22 | 3/24 |
| crowd_navigation: `francis2023_crowd_navigation` | 26/4 | 23/0 | 30/0 | 13/17 | 22/8 |

**r0.5_v1: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| cross_trap: `classic_cross_trap_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 14/16 |
| doorway: `classic_doorway_medium` | 30/0 | 26/0 | 30/0 | 30/0 | 16/14 |
| group_crossing: `classic_group_crossing_medium` | 29/1 | 30/0 | 30/0 | 30/0 | 30/0 |
| head_on_corridor: `classic_head_on_corridor_medium` | 29/1 | 30/0 | 30/0 | 30/0 | 30/0 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 30/0 | 30/0 | 23/7 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| robot_crowding: `francis2023_robot_crowding` | 26/4 | 10/0 | 30/0 | 27/3 | 8/21 |
| crowd_navigation: `francis2023_crowd_navigation` | 28/2 | 25/0 | 30/0 | 30/0 | 23/7 |

**r0.5_v1.3: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 19/11 |
| cross_trap: `classic_cross_trap_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 13/17 |
| doorway: `classic_doorway_medium` | 29/0 | 9/0 | 30/0 | 30/0 | 15/15 |
| group_crossing: `classic_group_crossing_medium` | 30/0 | 30/0 | 30/0 | 30/0 | 27/3 |
| head_on_corridor: `classic_head_on_corridor_medium` | 30/0 | 29/0 | 30/0 | 30/0 | 29/1 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 30/0 | 30/0 | 27/3 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| robot_crowding: `francis2023_robot_crowding` | 27/3 | 12/0 | 30/0 | 30/0 | 7/22 |
| crowd_navigation: `francis2023_crowd_navigation` | 27/3 | 19/0 | 29/0 | 30/0 | 18/12 |

**r1_v0.65: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 29/1 | 0/0 | 30/0 | 0/30 | 0/30 |
| cross_trap: `classic_cross_trap_medium` | 27/2 | 7/0 | 30/0 | 0/30 | 15/15 |
| doorway: `classic_doorway_medium` | 30/0 | 27/0 | 30/0 | 12/18 | 7/23 |
| group_crossing: `classic_group_crossing_medium` | 29/1 | 30/0 | 30/0 | 25/5 | 25/5 |
| head_on_corridor: `classic_head_on_corridor_medium` | 27/3 | 26/0 | 30/0 | 12/18 | 29/1 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 27/0 | 30/0 | 21/9 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 30/0 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 30/0 | 28/0 | 30/0 | 29/1 |
| robot_crowding: `francis2023_robot_crowding` | 20/9 | 0/1 | 21/2 | 0/30 | 2/28 |
| crowd_navigation: `francis2023_crowd_navigation` | 26/4 | 8/0 | 27/0 | 1/29 | 17/13 |

**r1_v1: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 30/0 | 1/0 | 30/0 | 0/30 | 0/30 |
| cross_trap: `classic_cross_trap_medium` | 28/2 | 29/1 | 30/0 | 0/30 | 13/17 |
| doorway: `classic_doorway_medium` | 0/8 | 0/1 | 29/1 | 3/27 | 1/29 |
| group_crossing: `classic_group_crossing_medium` | 27/3 | 22/0 | 30/0 | 24/6 | 26/4 |
| head_on_corridor: `classic_head_on_corridor_medium` | 29/1 | 25/0 | 30/0 | 9/21 | 27/3 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 27/0 | 30/0 | 24/6 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 29/1 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 30/0 | 28/0 | 30/0 | 30/0 |
| robot_crowding: `francis2023_robot_crowding` | 14/16 | 1/8 | 1/26 | 0/30 | 6/24 |
| crowd_navigation: `francis2023_crowd_navigation` | 23/7 | 8/2 | 22/2 | 1/29 | 19/11 |

**r1_v1.3: per-cell success/collision episode counts (requested 30 each).**

| Family / selected cell | ORCA | Social force | Static hybrid v4 | Goal | PPO |
|---|---:|---:|---:|---:|---:|
| bottleneck: `classic_bottleneck_medium` | 22/8 | 0/30 | 30/0 | 0/30 | 0/30 |
| cross_trap: `classic_cross_trap_medium` | 22/8 | 25/3 | 30/0 | 0/30 | 10/20 |
| doorway: `classic_doorway_medium` | 1/24 | 0/18 | 13/16 | 0/30 | 1/29 |
| group_crossing: `classic_group_crossing_medium` | 28/2 | 15/2 | 28/2 | 19/11 | 26/4 |
| head_on_corridor: `classic_head_on_corridor_medium` | 30/0 | 8/19 | 30/0 | 2/28 | 24/6 |
| entering_room: `francis2023_entering_room` | 30/0 | 30/0 | 27/0 | 30/0 | 29/1 |
| exiting_room: `francis2023_exiting_room` | 30/0 | 30/0 | 30/0 | 30/0 | 29/1 |
| circular_crossing: `francis2023_circular_crossing` | 30/0 | 29/0 | 27/2 | 29/1 | 29/1 |
| robot_crowding: `francis2023_robot_crowding` | 7/23 | 0/25 | 0/30 | 0/30 | 2/28 |
| crowd_navigation: `francis2023_crowd_navigation` | 12/18 | 4/16 | 7/15 | 0/30 | 13/17 |

Crowd sanity (30 dev seeds; 40 pedestrians; preferred speed 0.65 m/s).

| Probe | Result | Interpretation |
|---|---|---|
| Bidirectional corridor, 7.5–15 s | segregation 0.170; purity 0.395; weak 16/30; clear 1/30 | mainly weak transient separation; sustained lanes not established |
| Bidirectional corridor, 15–30 s | segregation 0.266; purity 0.433; weak 21/30; clear 0/30 | mainly weak transient separation; sustained lanes not established |
| Bidirectional corridor, 30–60 s | segregation 0.051; purity 0.220; weak 1/30; clear 0/30 | mainly weak transient separation; sustained lanes not established |
| Bidirectional 1.2 m door | 0.0000 people/s; 0–0/40 crossing in 60 s | opposing-flow bottleneck largely blocks |
| Unidirectional, goal-aligned 1.2 m door | 0.268 people/s; 9–21/40 crossings in 60 s | passage emerges |
| Unidirectional actors with no wall overlap at any recorded step | 0.208 people/s | 6,144 wall-overlap pedestrian-steps remain in the complete population |

Final probes explicitly use the release's wrapped_v2 pair kernel and historical legacy_shifted_gradient_v1 wall law. All corridor/door resets are independently clear for 0.4 m pedestrian bodies. Force radius remains 0.35 m, while the physical check uses 0.4 m. Earlier legacy-kernel, invalid-reset, failed rejection and off-axis doorway probes are retained as excluded diagnostics. Throughput counts each actor once at the door plane, interpolates lateral position, and requires its disc to fit inside the aperture. The wall-clear subset excludes any actor with a sampled wall overlap during the run; it does not prove swept or pedestrian-pair collision freedom. These finite cohorts do not test continuous-flow capacity or prove that sustained lanes cannot emerge under other densities or boundaries.

The model produces mainly weak transient lane sorting and some one-way door flow in these finite-population fixtures. It does not establish stable lanes, realistic counterflow, or empirically calibrated throughput. Known social-force phenomena are discussed in [Helbing and Molnár (1995)](https://doi.org/10.1103/PhysRevE.51.4282). These model-specific diagnostics are not empirical crowd validation.

Detailed outcomes for **every** requested planner/treatment/family cell, including the full cap roster, typed collision incidence and missingness: [cells.csv](evidence/cells.csv). Machine-readable paired correlations and planner/family aggregates: [summary.json](evidence/summary.json). Geometry proof: [door_geometry.json](evidence/door_geometry.json). All-release fact audit: [release_fact_audit.json](evidence/release_fact_audit.json).

Custody: compressed canonical episode records (checkout-prefix normalization only), complete crowd trajectories, canary traces, exact acquisition driver and input bytes are retained. Producer sidecar references and file checksums are rebound to portable compressed artifacts in the [custody ledger](evidence/portable_custody.json); scientific values remain unchanged. [Source identity](evidence/acquisition_source.json) records an administrative Git-epoch change during acquisition. The recovered episode commit has [archived Git commit/tree identity and an independently verifiable public-base diff](evidence/runtime_custody/README.md): every tracked base path and all 42 archived acquisition-source files match. Later tooling cleanup is separate from the archived acquisition driver. This is diagnostic provenance, not a frozen-release replay certificate.

Archived YAML files under `inputs/` preserve exact bytes at their original relative-path layout; regenerate execution configs under `configs/benchmarks/robsweep/` before running them. [Input locations](evidence/input_locations.json) map each archived artifact to that original config path. [SHA256SUMS](evidence/SHA256SUMS) is checked from the repository root. No frozen config, release artifact, benchmark default, model weight, or locked thesis text is changed.
