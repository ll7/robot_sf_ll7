# Policy Search Failure Report

- Source JSONL: `output/policy_search/scenario_adaptive_hybrid_orca_v1/full_matrix_h500/policy_search_full_matrix_h500_leaders_clean_20260505_204501/full_matrix_h500__scenario_adaptive_hybrid_orca_v1__combined.jsonl`

## Failure Taxonomy

| Failure Mode | Count |
|---|---:|
| raw_success_rate | 0.9097 |
| raw_collision_rate | 0.0208 |
| evidence_adjusted_success_rate | 0.9097 |
| evidence_adjusted_collision_rate | 0.0208 |
| excluded_episodes | 0 |
| near_miss_intrusive | 4 |
| static_collision | 3 |
| timeout_low_progress | 6 |

## Scenario Exclusions

These rows are not classified as policy failures because they carry explicit exclusion metadata with evidence.

| Scenario | Seed | Status | Reason | Evidence |
|---|---:|---|---|---|

## Top Failure Scenarios

| Failure Mode | Scenario | Count |
|---|---|---:|
| near_miss_intrusive | classic_station_platform_medium | 3 |
| timeout_low_progress | francis2023_narrow_doorway | 3 |
| timeout_low_progress | classic_merging_medium | 2 |
| static_collision | classic_cross_trap_high | 1 |
| static_collision | classic_merging_low | 1 |
| timeout_low_progress | classic_merging_low | 1 |
| near_miss_intrusive | classic_merging_medium | 1 |
| static_collision | francis2023_circular_crossing | 1 |
