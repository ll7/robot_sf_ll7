# Issue #2544 Static-Deadlock Suite Smoke

## Outcome Summary

- Planner/scenario: `social_force + orca` on `static_deadlock_recovery: classic_bottleneck_low, classic_head_on_corridor_low, narrow_passage`.
- Outcome: The suite executes end-to-end for two native/core planners, but required mechanism trace fields are missing.
- Execution status: `missing_required_trace_fields`.
- Metrics: `orca_success_count`=2, `rows`=6, `social_force_success_count`=0.

## Mechanism Activation

- Mechanism: `static_deadlock_or_local_minimum`.
- Activation status: `missing_required_trace_fields`.
- Evidence: Episode rows do not include low_progress_window, recenter_activation_count, distance_to_goal_delta, local_minimum_indicator, or row_status.

## Failure Mechanism Classification

- Classification: `static_deadlock_or_local_minimum`.
- Rationale: The selected suite targets static-deadlock recovery, but the current runner output cannot confirm the mechanism fields required by #2452.

## Paired Comparator

- Comparator: `social_force versus orca smoke rows`.
- Comparator outcome: ORCA succeeds on two of three rows; social_force times out on all three, but this is not planner-ranking evidence because mechanism trace fields are absent.
- Delta: `classification`=diagnostic_revise, `rows`=6.

## Trace Evidence

- raw_output_policy: raw JSONL was worktree-local and summarized into tracked compact evidence.
- summary: docs/context/evidence/issue_2544_mechanism_aware_suite_smoke/summary.json.
- table: docs/context/evidence/issue_2544_mechanism_aware_suite_smoke/suite_smoke_table.csv.

## Alternative Explanations

- The selected horizon is a bounded smoke horizon, not the full suite horizon.
- Two selected static maps intentionally have zero pedestrian density, limiting dynamic social-force interpretation.
- The runner may need explicit row-status and static-deadlock trace instrumentation before this suite can become benchmark-candidate evidence.

## Continue / Revise / Stop Decision

- Decision: `revise`.
- Rationale: Execution is available, but the required mechanism trace contract is incomplete.
- Next step: Add or bind runner instrumentation for static-deadlock trace fields before interpreting metric deltas as mechanism evidence.

## Claim Boundary

Report strength is limited to the compact input evidence.
