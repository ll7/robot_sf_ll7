# Trace Failure Predicate Table

Aggregate predicate rows are diagnostic-only unless tied to a predeclared benchmark matrix.

- input traces: 3
- table kind: aggregate_by_predicate_group
- schema version: trace_failure_predicates.v1

## Aggregate Rows

| scenario_family | planner_id | seed | predicate_id | validity_status | severity | predicate_count | trace_denominator | predicate_rate_per_trace |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crossing_proxy | orca | 111 | bottleneck_deadlock | valid | high | 1 | 1 | 1.0000 |
| crossing_proxy | orca | 111 | clearance_critical_interaction | valid | high | 2 | 1 | 2.0000 |
| crossing_proxy | orca | 111 | clearance_critical_interaction | valid | medium | 3 | 1 | 3.0000 |
| crossing_proxy | orca | 111 | late_evasive_reaction | valid | high | 1 | 1 | 1.0000 |
| crossing_proxy | orca | 111 | occlusion_triggered_near_miss | not_available | not_available | 1 | 1 | 1.0000 |
| crossing_proxy | orca | 111 | oscillatory_local_control | valid | medium | 1 | 1 | 1.0000 |
| crossing_proxy | orca | 111 | zero_motion_timeout_behavior | valid | high | 1 | 1 | 1.0000 |
