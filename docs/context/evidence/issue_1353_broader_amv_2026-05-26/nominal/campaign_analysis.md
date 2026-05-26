# Camera-Ready Campaign Analysis

- Campaign ID: `issue_1353_paired_nominal_v1_broader_baselines_issue1353-nominal-main-rvo2-20260526_20260526_062603`
- Campaign root: `output/benchmarks/issue_1353/issue_1353_paired_nominal_v1_broader_baselines_issue1353-nominal-main-rvo2-20260526_20260526_062603`
- Runtime sec: `165.5303267678246`
- Episodes/sec: `0.507459881462203`

## Planner Diagnostics

| planner | algo | kinematics | preflight | episodes | success(ep) | collision(ep) | snqi(ep) | abs map paths | runtime(s) | eps/s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | differential_drive | ok | 12 | 0.2500 | 0.3333 | -0.09674198266936922 | 0 | 22.1500 | 0.5418 |
| orca | orca | differential_drive | ok | 12 | 0.2500 | 0.0833 | -0.2999373776212828 | 0 | 12.6250 | 0.9505 |
| ppo | ppo | differential_drive | ok | 12 | 0.3333 | 0.0833 | -0.4518696193725835 | 0 | 27.0776 | 0.4432 |
| prediction_planner | prediction_planner | differential_drive | ok | 12 | 0.3333 | 0.2500 | -0.12461610754903583 | 0 | 59.1144 | 0.2030 |
| sacadrl | sacadrl | differential_drive | ok | 12 | 0.2500 | 0.2500 | -0.0993124229230289 | 0 | 16.2300 | 0.7394 |
| social_force | social_force | differential_drive | ok | 12 | 0.0000 | 0.0000 | -1.0432326550481905 | 0 | 11.4492 | 1.0481 |
| socnav_bench | socnav_bench | differential_drive | skipped | 0 | 0.0000 | 0.0000 | nan | 0 | 0.1871 | 0.0000 |
| socnav_sampling | socnav_sampling | differential_drive | ok | 12 | 0.2500 | 0.3333 | -0.08140486488876582 | 0 | 7.9529 | 1.5089 |

## Runtime Hotspots

| planner | runtime(s) | wall_time_mean(s) | wall_time_p95(s) |
|---|---:|---:|---:|
| prediction_planner (differential_drive) | 59.1144 | 4.8505 | 7.5726 |
| ppo (differential_drive) | 27.0776 | 1.6383 | 1.8348 |
| goal (differential_drive) | 22.1500 | 1.7720 | 2.8938 |

- `prediction_planner` (differential_drive) top slow scenarios:
  - `single_ped_crossing_orthogonal` mean=7.6296s p95=7.7575s (episodes=3)
  - `classic_doorway_low` mean=4.8398s p95=6.0388s (episodes=3)
  - `classic_bottleneck_low` mean=4.3854s p95=6.3529s (episodes=3)
- `ppo` (differential_drive) top slow scenarios:
  - `single_ped_crossing_orthogonal` mean=1.7403s p95=1.7446s (episodes=3)
  - `classic_bottleneck_low` mean=1.7026s p95=1.8348s (episodes=3)
  - `empty_map_8_directions_east` mean=1.5922s p95=2.7033s (episodes=3)
- `goal` (differential_drive) top slow scenarios:
  - `empty_map_8_directions_east` mean=4.3985s p95=12.5504s (episodes=3)
  - `classic_doorway_low` mean=1.1321s p95=2.8938s (episodes=3)
  - `single_ped_crossing_orthogonal` mean=0.9355s p95=1.0768s (episodes=3)

## Scenario Difficulty

- Primary proxy: `consensus_outcome_rank_v1`
- Description: Weighted consensus score across core benchmark-success planners using success, collisions, near-misses, and normalized time-to-goal.
- Consensus planners: `3` (core benchmark-success planners)
- Supporting metric: `snqi_mean`

### Hardest Scenarios

| rank | scenario | family | difficulty | success | collisions | near misses | time_to_goal_norm | seed success CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | classic_doorway_low | doorway | 0.8000 | 0.0000 | 0.4444 | 9.3333 | 1.0000 | 0.0000 |
| 2 | classic_bottleneck_low | bottleneck | 0.6167 | 0.0000 | 0.1111 | 0.0000 | 1.0000 | 0.0000 |
| 3 | single_ped_crossing_orthogonal | single_ped_crossing_orthogonal | 0.4917 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 4 | empty_map_8_directions_east | empty_map_8_directions | 0.0917 | 0.6667 | 0.0000 | 0.0000 | 0.7089 | 0.0000 |

### Family Rollup

| family | scenarios | difficulty | success | collisions | near misses | seed success CI | hardest scenarios |
|---|---:|---:|---:|---:|---:|---:|---|
| bottleneck | 1 | 0.6167 | 0.0000 | 0.1111 | 0.0000 | 0.0000 | classic_bottleneck_low |
| doorway | 1 | 0.8000 | 0.0000 | 0.4444 | 9.3333 | 0.0000 | classic_doorway_low |
| empty_map_8_directions | 1 | 0.0917 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | empty_map_8_directions_east |
| single_ped_crossing_orthogonal | 1 | 0.4917 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | single_ped_crossing_orthogonal |

### Planner Residual Summary

| planner | group | mean residual | max residual | easy-scenario mismatches | worst scenarios |
|---|---|---:|---:|---:|---|
| goal | core | 0.0338 | 0.1667 | 1 | classic_bottleneck_low, classic_doorway_low, single_ped_crossing_orthogonal |
| orca | core | -0.0377 | 0.1285 | 0 | classic_doorway_low, single_ped_crossing_orthogonal, classic_bottleneck_low |
| ppo | experimental | -30000.0870 | 0.0000 | 0 | single_ped_crossing_orthogonal, classic_bottleneck_low, empty_map_8_directions_east |
| prediction_planner | experimental | -26037.4440 | 0.1667 | 1 | classic_bottleneck_low, single_ped_crossing_orthogonal, empty_map_8_directions_east |
| sacadrl | experimental | -0.0269 | 0.0853 | 0 | classic_doorway_low, single_ped_crossing_orthogonal, classic_bottleneck_low |
| social_force | core | 0.0039 | 0.3040 | 1 | empty_map_8_directions_east, single_ped_crossing_orthogonal, classic_bottleneck_low |
| socnav_sampling | experimental | -0.0141 | 0.1667 | 1 | classic_bottleneck_low, classic_doorway_low, single_ped_crossing_orthogonal |

### Verified-Simple Assessment

- Status: `candidate_supported`
- Matched scenarios: `2`
- Rank correlation: `1.0000`
- Recommendation: The verified-simple subset broadly preserves planner ordering while keeping seed noise comparable to the full campaign. Use it as a calibration aid, not as a replacement benchmark.

### Difficulty Findings

- No additional scenario-difficulty warnings.

## Findings

- No inconsistencies detected by automated checks.
