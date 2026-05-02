# Policy Search Failure Report

- Source JSONL: `output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_full_h500_w2/full_matrix__hybrid_rule_v3_fast_progress_static_escape.jsonl`

## Failure Taxonomy

| Failure Mode | Count |
|---|---:|
| near_miss_intrusive | 5 |
| static_collision | 3 |
| timeout_low_progress | 7 |

## Top Failure Scenarios

| Failure Mode | Scenario | Count |
|---|---|---:|
| near_miss_intrusive | classic_realworld_double_bottleneck_high | 3 |
| timeout_low_progress | francis2023_narrow_doorway | 3 |
| timeout_low_progress | classic_merging_medium | 2 |
| static_collision | classic_cross_trap_high | 1 |
| static_collision | classic_merging_low | 1 |
| timeout_low_progress | classic_merging_low | 1 |
| near_miss_intrusive | classic_merging_medium | 1 |
| near_miss_intrusive | francis2023_leave_group | 1 |
| timeout_low_progress | francis2023_perpendicular_traffic | 1 |
| static_collision | francis2023_circular_crossing | 1 |
