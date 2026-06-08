# Issue #2588 Static-Deadlock Controlled Trace

## Outcome Summary

- Planner/scenario: `hybrid_rule_v3_fast_progress vs issue_2170_static_recenter_only` on `static_deadlock_recovery: classic_bottleneck_low, classic_head_on_corridor_low, narrow_passage; seeds 111, 112, 113`.
- Outcome: The matched baseline/intervention slice completed with all required static-deadlock trace fields; static recenter activated in one of nine pairs and changed trace/local-minimum state, but did not create a terminal success rescue..
- Execution status: `controlled_trace`.
- Metrics: `activation_count_total`=4, `active_pairs`=1, `completed_pairs`=9, `required_fields_present`=true, `rows`=9, `terminal_rescue_pairs`=0.

## Mechanism Activation

- Mechanism: `static_deadlock_or_local_minimum`.
- Activation status: `activated_trace_changed_no_terminal_rescue`.
- Evidence: One classic_bottleneck_low seed-113 intervention row had four positive static_recenter decision terms and changed command/trajectory/local-minimum fields; terminal outcome remained max_steps failure..

## Failure Mechanism Classification

- Classification: `static_deadlock_or_local_minimum`.
- Rationale: The discriminating active row removes the local-minimum indicator under the intervention but still times out, so the mechanism signal is present without benchmark-strength success evidence..

## Paired Comparator

- Comparator: `hybrid_rule_v3_fast_progress baseline versus issue_2170_static_recenter_only intervention`.
- Comparator outcome: All nine pairs completed. Four pairs were inactive, four were already solved by both candidates, and one active pair changed trace state without terminal success..
- Delta: `active_trace_changed_pairs`=1, `classification`=controlled_trace_negative_mixed, `rows`=9.

## Trace Evidence

- field_presence: docs/context/evidence/issue_2588_static_deadlock_controlled_trace/field_presence.json.
- raw_output_policy: raw activation_trace.json was worktree-local and summarized into tracked compact evidence.
- summary: docs/context/evidence/issue_2588_static_deadlock_controlled_trace/summary.json.
- table: docs/context/evidence/issue_2588_static_deadlock_controlled_trace/controlled_trace_table.csv.

## Alternative Explanations

- The 120-step stop rule is intentionally bounded to the #2544 static-deadlock smoke horizon, so this is not benchmark-candidate evidence..
- Only one intervention candidate was run; the semantic escape_recenter_pair intervention remains a possible follow-up..
- The active row improved the local-minimum trace classification but did not reach the goal, so the intervention signal is not a planner-improvement claim..

## Continue / Revise / Stop Decision

- Decision: `continue`.
- Rationale: Controlled trace evidence now exists, but the result is mixed/negative and not strong enough for benchmark-candidate promotion..
- Next step: If more evidence is needed, run the escape_recenter_pair candidate or a longer predeclared horizon slice before considering benchmark-candidate status..

## Claim Boundary

Report strength is limited to the compact input evidence.
