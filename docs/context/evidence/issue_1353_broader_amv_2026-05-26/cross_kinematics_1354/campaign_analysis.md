# Camera-Ready Campaign Analysis

- Campaign ID: `paper_cross_kinematics_v1_issue1354-paper-compact-main-rvo2-20260526_20260526_062604`
- Campaign root: `/home/luttkule/git/robot_sf_ll7/output/benchmarks/issue_1354_20260526/paper_cross_kinematics_v1_issue1354-paper-compact-main-rvo2-20260526_20260526_062604`
- Runtime sec: `22.543566548265517`
- Episodes/sec: `0.39922698037735527`

## Planner Diagnostics

| planner | algo | kinematics | preflight | episodes | success(ep) | collision(ep) | snqi(ep) | abs map paths | runtime(s) | eps/s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.429660865607279 | 0 | 0.5959 | 1.6780 |
| goal | goal | differential_drive | ok | 1 | 0.0000 | 0.0000 | -0.4126535486592599 | 0 | 12.5537 | 0.0797 |
| goal | goal | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.10834588948566656 | 0 | 0.6157 | 1.6241 |
| orca | orca | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.11486736792187859 | 0 | 0.9227 | 1.0838 |
| orca | orca | differential_drive | ok | 1 | 0.0000 | 0.0000 | -0.17537249117335946 | 0 | 0.9227 | 1.0838 |
| orca | orca | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.5054352687555362 | 0 | 0.7427 | 1.3464 |
| social_force | social_force | bicycle_drive | ok | 1 | 0.0000 | 0.0000 | -0.12896830062809556 | 0 | 0.6431 | 1.5549 |
| social_force | social_force | differential_drive | ok | 1 | 0.0000 | 0.0000 | -0.2663483342611581 | 0 | 0.9396 | 1.0643 |
| social_force | social_force | holonomic | ok | 1 | 0.0000 | 0.0000 | -0.41206322959262204 | 0 | 0.6225 | 1.6064 |

## Runtime Hotspots

| planner | runtime(s) | wall_time_mean(s) | wall_time_p95(s) |
|---|---:|---:|---:|
| goal (differential_drive) | 12.5537 | 12.4700 | 12.4700 |
| social_force (differential_drive) | 0.9396 | 0.8651 | 0.8651 |
| orca (differential_drive) | 0.9227 | 0.8481 | 0.8481 |

- `goal` (differential_drive) top slow scenarios:
  - `classic_cross_trap_low` mean=12.4700s p95=12.4700s (episodes=1)
- `social_force` (differential_drive) top slow scenarios:
  - `classic_cross_trap_low` mean=0.8651s p95=0.8651s (episodes=1)
- `orca` (differential_drive) top slow scenarios:
  - `classic_cross_trap_low` mean=0.8481s p95=0.8481s (episodes=1)

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
| social_force | core | -0.0490 | -0.0490 | 0 | classic_cross_trap_low |

### Verified-Simple Assessment

- Status: `rerun_required`
- Matched scenarios: `0`
- Rank correlation: `nan`
- Recommendation: The current camera-ready campaign does not include the verified-simple candidate scenarios. Keep the subset as a debugging or promotion gate for now and run one bounded pilot before treating it as a calibration set.

### Difficulty Findings

- No additional scenario-difficulty warnings.

## Findings

- No inconsistencies detected by automated checks.
