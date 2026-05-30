# Camera-Ready Campaign Analysis

- Campaign ID: `issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441`
- Campaign root: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441`
- Runtime sec: `86.0142236140091`
- Episodes/sec: `0.2441456670496499`

## Planner Diagnostics

| planner | algo | kinematics | preflight | episodes | success(ep) | collision(ep) | snqi(ep) | abs map paths | runtime(s) | eps/s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.429660865607279 | 0 | 0.6015 | 1.6625 |
| goal | goal | differential_drive | ok | 1 | 0.0000 | 0.0000 | -0.4126535486592599 | 0 | 14.9820 | 0.0667 |
| goal | goal | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.10834588948566656 | 0 | 0.6001 | 1.6664 |
| orca | orca | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.11486736792187859 | 0 | 0.9204 | 1.0865 |
| orca | orca | differential_drive | ok | 1 | 0.0000 | 0.0000 | -0.17537249117335946 | 0 | 0.9102 | 1.0987 |
| orca | orca | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.5054352687555362 | 0 | 0.7314 | 1.3671 |
| ppo | ppo | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.40168950119545055 | 0 | 4.1474 | 0.2411 |
| ppo | ppo | differential_drive | ok | 1 | 0.0000 | 0.0000 | -0.36489649272443586 | 0 | 22.1824 | 0.0451 |
| ppo | ppo | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.29142105055583606 | 0 | 2.6857 | 0.3723 |
| prediction_planner | prediction_planner | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.10232213653698202 | 0 | 6.2081 | 0.1611 |
| prediction_planner | prediction_planner | differential_drive | ok | 1 | 0.0000 | 0.0000 | -0.09998383103973885 | 0 | 6.3611 | 0.1572 |
| prediction_planner | prediction_planner | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.10039033325596686 | 0 | 6.2260 | 0.1606 |
| sacadrl | sacadrl | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.4925880178596274 | 0 | 1.6761 | 0.5966 |
| sacadrl | sacadrl | differential_drive | ok | 1 | 0.0000 | 0.0000 | -0.46226609684403575 | 0 | 1.5163 | 0.6595 |
| sacadrl | sacadrl | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.40917822487994254 | 0 | 1.5438 | 0.6478 |
| social_force | social_force | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.12896830062809556 | 0 | 0.6353 | 1.5740 |
| social_force | social_force | differential_drive | ok | 1 | 0.0000 | 0.0000 | -0.2663483342611581 | 0 | 0.9116 | 1.0970 |
| social_force | social_force | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.41206322959262204 | 0 | 0.6314 | 1.5837 |
| socnav_sampling | socnav_sampling | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.10853146985401904 | 0 | 0.7977 | 1.2535 |
| socnav_sampling | socnav_sampling | differential_drive | ok | 1 | 0.0000 | 1.0000 | -0.28103524279580777 | 0 | 0.9128 | 1.0955 |
| socnav_sampling | socnav_sampling | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.09998383103973885 | 0 | 0.8194 | 1.2204 |

## Runtime Hotspots

| planner | runtime(s) | wall_time_mean(s) | wall_time_p95(s) |
|---|---:|---:|---:|
| ppo (differential_drive) | 22.1824 | 4.1722 | 4.1722 |
| goal (differential_drive) | 14.9820 | 14.8905 | 14.8905 |
| prediction_planner (differential_drive) | 6.3611 | 6.2608 | 6.2608 |

- `ppo` (differential_drive) top slow scenarios:
  - `classic_cross_trap_low` mean=4.1722s p95=4.1722s (episodes=1)
- `goal` (differential_drive) top slow scenarios:
  - `classic_cross_trap_low` mean=14.8905s p95=14.8905s (episodes=1)
- `prediction_planner` (differential_drive) top slow scenarios:
  - `classic_cross_trap_low` mean=6.2608s p95=6.2608s (episodes=1)

## Scenario Difficulty

- Primary proxy: `consensus_outcome_rank_v1`
- Description: Weighted consensus score across core benchmark-success planners using success, collisions, near-misses, and normalized time-to-goal.
- Consensus planners: `3` (core benchmark-success planners)
- Supporting metric: `snqi_mean`

### Hardest Scenarios

| rank | scenario | family | difficulty | success | collisions | near misses | time_to_goal_norm | seed success CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | classic_cross_trap_low | cross_trap | 0.0000 | 0.0000 | 0.0000 | 10.2222 | 1.0000 | 0.0000 |

### Family Rollup

| family | scenarios | difficulty | success | collisions | near misses | seed success CI | hardest scenarios |
|---|---:|---:|---:|---:|---:|---:|---|
| cross_trap | 1 | 0.0000 | 0.0000 | 0.0000 | 10.2222 | 0.0000 | classic_cross_trap_low |

### Planner Residual Summary

| planner | group | mean residual | max residual | easy-scenario mismatches | worst scenarios |
|---|---|---:|---:|---:|---|
| goal | core | 0.1495 | 0.1495 | 1 | classic_cross_trap_low |
| orca | core | -0.1005 | -0.1005 | 0 | classic_cross_trap_low |
| ppo | experimental | -0.1225 | -0.1225 | 0 | classic_cross_trap_low |
| prediction_planner | experimental | -0.2255 | -0.2255 | 0 | classic_cross_trap_low |
| sacadrl | experimental | 0.5980 | 0.5980 | 1 | classic_cross_trap_low |
| social_force | core | -0.0490 | -0.0490 | 0 | classic_cross_trap_low |
| socnav_sampling | experimental | 83324.8113 | 83324.8113 | 1 | classic_cross_trap_low |

### Verified-Simple Assessment

- Status: `rerun_required`
- Matched scenarios: `0`
- Rank correlation: `nan`
- Recommendation: The current camera-ready campaign does not include the verified-simple candidate scenarios. Keep the subset as a debugging or promotion gate for now and run one bounded pilot before treating it as a calibration set.

### Difficulty Findings

- No additional scenario-difficulty warnings.

## Findings

- No inconsistencies detected by automated checks.
