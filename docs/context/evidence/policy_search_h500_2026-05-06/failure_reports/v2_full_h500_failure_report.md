# Policy Search Failure Report

- Source JSONL: `output/policy_search/scenario_adaptive_hybrid_orca_v2_collision_guard/full_matrix_h500/policy_search_full_matrix_h500_collision_guard_20260506_0800/full_matrix_h500__scenario_adaptive_hybrid_orca_v2_collision_guard__combined.jsonl`

## Failure Taxonomy

| Failure Mode | Count |
|---|---:|
| raw_success_rate | 0.9028 |
| raw_collision_rate | 0.0139 |
| evidence_adjusted_success_rate | 0.9028 |
| evidence_adjusted_collision_rate | 0.0139 |
| excluded_episodes | 0 |
| near_miss_intrusive | 4 |
| static_collision | 2 |
| timeout_low_progress | 8 |

## Scenario Exclusions

These rows are not classified as policy failures because they carry explicit exclusion metadata with evidence.

| Scenario | Seed | Status | Reason | Evidence |
|---|---:|---|---|---|

## Top Failure Scenarios

| Failure Mode | Scenario | Count |
|---|---|---:|
| near_miss_intrusive | classic_station_platform_medium | 3 |
| timeout_low_progress | classic_merging_low | 3 |
| timeout_low_progress | francis2023_narrow_doorway | 3 |
| timeout_low_progress | classic_merging_medium | 2 |
| static_collision | classic_cross_trap_high | 1 |
| near_miss_intrusive | classic_merging_medium | 1 |
| static_collision | francis2023_circular_crossing | 1 |
