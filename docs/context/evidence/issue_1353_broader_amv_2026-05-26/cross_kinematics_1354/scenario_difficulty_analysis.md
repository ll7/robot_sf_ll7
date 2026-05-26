# Scenario Difficulty

- Primary proxy: `consensus_outcome_rank_v1`
- Description: Weighted consensus score across core benchmark-success planners using success, collisions, near-misses, and normalized time-to-goal.
- Consensus planners: `3` (core benchmark-success planners)
- Supporting metric: `snqi_mean`

## Hardest Scenarios

| rank | scenario | family | difficulty | success | collisions | near misses | time_to_goal_norm | seed success CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | classic_cross_trap_low | cross_trap | 0.0000 | 0.0000 | 0.0000 | 10.2222 | 1.0000 | 0.0000 |

## Family Rollup

| family | scenarios | difficulty | success | collisions | near misses | seed success CI | hardest scenarios |
|---|---:|---:|---:|---:|---:|---:|---|
| cross_trap | 1 | 0.0000 | 0.0000 | 0.0000 | 10.2222 | 0.0000 | classic_cross_trap_low |

## Planner Residual Summary

| planner | group | mean residual | max residual | easy-scenario mismatches | worst scenarios |
|---|---|---:|---:|---:|---|
| goal | core | 0.1495 | 0.1495 | 1 | classic_cross_trap_low |
| orca | core | -0.1005 | -0.1005 | 0 | classic_cross_trap_low |
| social_force | core | -0.0490 | -0.0490 | 0 | classic_cross_trap_low |

## Verified-Simple Assessment

- Status: `rerun_required`
- Matched scenarios: `0`
- Rank correlation: `nan`
- Recommendation: The current camera-ready campaign does not include the verified-simple candidate scenarios. Keep the subset as a debugging or promotion gate for now and run one bounded pilot before treating it as a calibration set.

## Difficulty Findings

- No additional scenario-difficulty warnings.
