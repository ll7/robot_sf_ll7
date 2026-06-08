# Issue #2586 Static-Deadlock Trace Field Smoke

## Outcome Summary

- Planner/scenario: `social_force + orca` on `static_deadlock_recovery: classic_bottleneck_low, classic_head_on_corridor_low, narrow_passage`.
- Outcome: The static-deadlock smoke rows now emit all required trace/reportability fields for two native/core planners..
- Execution status: `diagnostic_revise`.
- Metrics: `rows`=6, `social_force_local_minimum_rows`=1, `orca_local_minimum_rows`=0, `required_fields_present`=true.

## Mechanism Activation

- Mechanism: `static_deadlock_or_local_minimum`.
- Activation status: `trace_fields_present_baseline_only`.
- Evidence: All six smoke rows emit low_progress_window, recenter_activation_count, distance_to_goal_delta, local_minimum_indicator, execution_mode, and row_status..

## Failure Mechanism Classification

- Classification: `static_deadlock_or_local_minimum`.
- Rationale: The trace fields make static-deadlock reportability possible, but the smoke does not include a matched static-recenter intervention comparison..

## Paired Comparator

- Comparator: `social_force versus orca smoke rows`.
- Comparator outcome: ORCA succeeds on two of three rows; social_force times out on all three. This remains diagnostic because it is baseline-only and one seed..
- Delta: `classification`=diagnostic_revise, `rows`=6.

## Trace Evidence

- summary: docs/context/evidence/issue_2586_static_deadlock_trace_fields/summary.json.
- table: docs/context/evidence/issue_2586_static_deadlock_trace_fields/suite_smoke_table.csv.
- field_presence: docs/context/evidence/issue_2586_static_deadlock_trace_fields/field_presence.json.
- raw_output_policy: raw JSONL was worktree-local and summarized into tracked compact evidence.

## Alternative Explanations

- The selected horizon is still a bounded smoke horizon, not the full suite horizon.
- The run uses one seed and baseline planners only, so it cannot establish controlled intervention evidence.
- The recenter activation count remains zero because no static-recenter intervention planner was run.

## Continue / Revise / Stop Decision

- Decision: `revise`.
- Rationale: Trace-field reportability is fixed, but benchmark-candidate evidence requires matched intervention rows and the declared seed set.
- Next step: Run a controlled static-deadlock baseline/intervention slice using the emitted trace fields; follow-up issue #2588 tracks that proof..

## Claim Boundary

Report strength is limited to the compact input evidence.
