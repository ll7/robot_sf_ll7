# Scenario Difficulty

- Primary proxy: `consensus_outcome_rank_v1`
- Description: Weighted consensus score across core benchmark-success planners using success, collisions, near-misses, and normalized time-to-goal.
- Consensus planners: `3` (core benchmark-success planners)
- Supporting metric: `snqi_mean`

## Hardest Scenarios

| rank | scenario | family | difficulty | success | collisions | near misses | time_to_goal_norm | seed success CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | classic_doorway_low | doorway | 0.8000 | 0.0000 | 0.4444 | 9.3333 | 1.0000 | 0.0000 |
| 2 | classic_bottleneck_low | bottleneck | 0.6167 | 0.0000 | 0.1111 | 0.0000 | 1.0000 | 0.0000 |
| 3 | single_ped_crossing_orthogonal | single_ped_crossing_orthogonal | 0.4917 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 4 | empty_map_8_directions_east | empty_map_8_directions | 0.0917 | 0.6667 | 0.0000 | 0.0000 | 0.7089 | 0.0000 |

## Family Rollup

| family | scenarios | difficulty | success | collisions | near misses | seed success CI | hardest scenarios |
|---|---:|---:|---:|---:|---:|---:|---|
| bottleneck | 1 | 0.6167 | 0.0000 | 0.1111 | 0.0000 | 0.0000 | classic_bottleneck_low |
| doorway | 1 | 0.8000 | 0.0000 | 0.4444 | 9.3333 | 0.0000 | classic_doorway_low |
| empty_map_8_directions | 1 | 0.0917 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | empty_map_8_directions_east |
| single_ped_crossing_orthogonal | 1 | 0.4917 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | single_ped_crossing_orthogonal |

## Planner Residual Summary

| planner | group | mean residual | max residual | easy-scenario mismatches | worst scenarios |
|---|---|---:|---:|---:|---|
| goal | core | 0.0338 | 0.1667 | 1 | classic_bottleneck_low, classic_doorway_low, single_ped_crossing_orthogonal |
| orca | core | -0.0377 | 0.1285 | 0 | classic_doorway_low, single_ped_crossing_orthogonal, classic_bottleneck_low |
| ppo | experimental | -30000.0870 | 0.0000 | 0 | single_ped_crossing_orthogonal, classic_bottleneck_low, empty_map_8_directions_east |
| prediction_planner | experimental | -26037.4440 | 0.1667 | 1 | classic_bottleneck_low, single_ped_crossing_orthogonal, empty_map_8_directions_east |
| sacadrl | experimental | -0.0269 | 0.0853 | 0 | classic_doorway_low, single_ped_crossing_orthogonal, classic_bottleneck_low |
| social_force | core | 0.0039 | 0.3040 | 1 | empty_map_8_directions_east, single_ped_crossing_orthogonal, classic_bottleneck_low |
| socnav_sampling | experimental | -0.0141 | 0.1667 | 1 | classic_bottleneck_low, classic_doorway_low, single_ped_crossing_orthogonal |

## Verified-Simple Assessment

- Status: `candidate_supported`
- Matched scenarios: `2`
- Rank correlation: `1.0000`
- Recommendation: The verified-simple subset broadly preserves planner ordering while keeping seed noise comparable to the full campaign. Use it as a calibration aid, not as a replacement benchmark.

## Difficulty Findings

- No additional scenario-difficulty warnings.
