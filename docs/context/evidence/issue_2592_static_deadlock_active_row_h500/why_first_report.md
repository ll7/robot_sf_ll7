# Issue #2592 Static-Deadlock Active-Row Horizon Sensitivity

## Outcome Summary

- Planner/scenario: `static_recenter_only and escape_recenter_pair h500 active-row sensitivity` on `static_deadlock_recovery: classic_bottleneck_low seed 113, h500`.
- Outcome: Both h500 active-row pairings completed with all required static-deadlock trace fields; both recenter interventions converted the previously 120-step trace-only change into a terminal success at 122 steps while the matched baselines remained 500-step failures.
- Execution status: `controlled_trace_horizon_sensitivity`.
- Metrics: `rows`=2, `completed_pairs`=2, `terminal_rescue_pairs`=2, `activation_count_total`=8, `intervention_success_steps`=[122], `baseline_failure_steps`=[500], `required_fields_present`=true.

## Mechanism Activation

- Mechanism: `static_recenter`.
- Activation status: `delayed_rescue_candidate`.
- Evidence: Each intervention row had four positive static_recenter decision terms beginning at step 7, changed command/trajectory/local-minimum fields, and reached success at step 122.

## Failure Mechanism Classification

- Classification: `horizon_limited_static_deadlock_or_local_minimum`.
- Rationale: The baselines stayed in a low-progress local minimum through h500, while recentering removed the local-minimum indicator and reached success after the longer horizon.

## Paired Comparator

- Comparator: `two active-row h500 recenter pairings`.
- Comparator outcome: `hybrid_rule_v3_fast_progress` versus `issue_2170_static_recenter_only` and `issue_2170_static_escape_only` versus `issue_2170_static_escape_recenter_no_transit` both classified as `mechanism_active_terminal_changed`.
- Delta: `rows`=2, `terminal_rescue_pairs`=2, `classification`=delayed_rescue_candidate.

## Trace Evidence

- summary: docs/context/evidence/issue_2592_static_deadlock_active_row_h500/summary.json.
- field_presence: docs/context/evidence/issue_2592_static_deadlock_active_row_h500/field_presence.json.
- table: docs/context/evidence/issue_2592_static_deadlock_active_row_h500/horizon_sensitivity_table.csv.
- raw_output_policy: raw activation_trace JSON files were worktree-local and summarized into tracked compact evidence.

## Alternative Explanations

- This is a one-row probe selected after observing the active row in #2588/#2590, so it is not an unbiased benchmark slice.
- The longer horizon changes the stop rule and cannot be compared directly to the 120-step controlled traces as a planner-ranking result.
- Both intervention candidates share the same observed recenter behavior on this row, so the result supports a recentering horizon-sensitivity hypothesis rather than a full planner-family claim.

## Continue / Revise / Stop Decision

- Decision: `continue`.
- Rationale: The active row now has delayed-rescue evidence under h500, but broader predeclared rows are needed before any benchmark-candidate or transfer claim.
- Next step: Run a separately predeclared h500 static-deadlock controlled slice or broadened scenario/seed probe before considering promotion.

## Claim Boundary

Report strength is limited to the compact input evidence.
