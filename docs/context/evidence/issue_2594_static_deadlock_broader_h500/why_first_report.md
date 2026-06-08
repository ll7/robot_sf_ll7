# Issue #2594 Broader H500 Static-Deadlock Recenter Slice

## Outcome Summary

- Planner/scenario: `static_recenter_only and escape_recenter_pair broader h500 static-deadlock slice` on `static_deadlock_recovery: 3 scenarios x 3 seeds, h500`.
- Outcome: Both h500 recenter pairings completed the predeclared 3-scenario x 3-seed static-deadlock slice with all required trace fields. Each pairing reproduced terminal rescue on classic_bottleneck_low seed 113, while the remaining 16 pair-rows were already solved by both comparator and intervention..
- Execution status: `controlled_trace_horizon_sensitivity`.
- Metrics: `pair_rows`=18, `completed_pair_rows`=18, `terminal_rescue_pair_rows`=2, `already_solved_pair_rows`=16, `activation_count_total`=8, `intervention_success_steps`=[122], `baseline_failure_steps`=[500], `required_fields_present`=true.

## Mechanism Activation

- Mechanism: `static_recenter`.
- Activation status: `broader_delayed_rescue_supported`.
- Evidence: Both intervention candidates had four positive static_recenter decision terms beginning at step 7 on classic_bottleneck_low seed 113 and converted the matched baseline failure into terminal success..

## Failure Mechanism Classification

- Classification: `horizon_limited_static_deadlock_or_local_minimum`.
- Rationale: The only unsolved baseline row in the broader slice remained a 500-step low-progress local minimum, while recentering removed the local-minimum indicator and reached success at step 122..

## Paired Comparator

- Comparator: `two broader h500 recenter pairings`.
- Comparator outcome: `hybrid_rule_v3_fast_progress` versus `issue_2170_static_recenter_only` and `issue_2170_static_escape_only` versus `issue_2170_static_escape_recenter_no_transit` each produced 8 already-solved rows and 1 `mechanism_active_terminal_changed` row..
- Delta: `pair_rows`=18, `terminal_rescue_pair_rows`=2, `already_solved_pair_rows`=16, `classification`=broader_delayed_rescue_supported.

## Trace Evidence

- summary: docs/context/evidence/issue_2594_static_deadlock_broader_h500/summary.json.
- field_presence: docs/context/evidence/issue_2594_static_deadlock_broader_h500/field_presence.json.
- table: docs/context/evidence/issue_2594_static_deadlock_broader_h500/broader_h500_table.csv.
- raw_output_policy: raw activation_trace JSON files were worktree-local and summarized into tracked compact evidence.

## Alternative Explanations

- The broader 3x3 slice contained only one unsolved active row; the other rows cannot show a rescue because both comparator and intervention already succeeded..
- The longer h500 stop rule differs from the h120 controlled traces and should not be presented as direct planner-ranking improvement..
- Both intervention candidates share the same observed recenter behavior on the active row, so this supports a static-recenter horizon-sensitivity claim rather than a broader planner-family claim..

## Continue / Revise / Stop Decision

- Decision: `continue`.
- Rationale: The predeclared broader h500 slice supports the delayed-rescue mechanism signal, but the evidence remains concentrated in one unsolved active row and is not benchmark-candidate proof..
- Next step: Synthesize the static-deadlock recenter claim boundary or predeclare a harder unsolved-row expansion before any promotion beyond controlled trace evidence..

## Claim Boundary

Report strength is limited to the compact input evidence.
