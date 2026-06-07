# Issue #2590 Escape-Recenter Static-Deadlock Controlled Trace

## Outcome Summary

- Planner/scenario: `issue_2170_static_escape_only vs issue_2170_static_escape_recenter_no_transit` on `static_deadlock_recovery: classic_bottleneck_low, classic_head_on_corridor_low, narrow_passage; seeds 111, 112, 113`.
- Outcome: The matched escape-only versus escape+recenter slice completed with all required static-deadlock trace fields; static recenter activated in one of nine pairs and changed trace/local-minimum state, but did not create a terminal success rescue.
- Execution status: `controlled_trace`.
- Metrics: `rows`=9, `completed_pairs`=9, `active_pairs`=1, `activation_count_total`=4, `terminal_rescue_pairs`=0, `required_fields_present`=true.

## Mechanism Activation

- Mechanism: `escape_recenter_pair`.
- Activation status: `activated_trace_changed_no_terminal_rescue`.
- Evidence: One classic_bottleneck_low seed-113 intervention row had four positive static_recenter decision terms and changed command/trajectory/local-minimum fields; terminal outcome remained max_steps failure.

## Failure Mechanism Classification

- Classification: `static_deadlock_or_local_minimum`.
- Rationale: The discriminating active row removes the local-minimum indicator under the escape+recenter intervention but still times out, so the recenter signal is present without benchmark-strength success evidence.

## Paired Comparator

- Comparator: `issue_2170_static_escape_only baseline versus issue_2170_static_escape_recenter_no_transit intervention`.
- Comparator outcome: All nine pairs completed. Four pairs were inactive, four were already solved by both candidates, and one active pair changed trace state without terminal success.
- Delta: `rows`=9, `active_trace_changed_pairs`=1, `classification`=controlled_trace_negative_mixed.

## Trace Evidence

- summary: docs/context/evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/summary.json.
- field_presence: docs/context/evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/field_presence.json.
- table: docs/context/evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/controlled_trace_table.csv.
- raw_output_policy: raw activation_trace.json was worktree-local and summarized into tracked compact evidence.

## Alternative Explanations

- The 120-step stop rule is intentionally comparable to #2588, so this is not benchmark-candidate evidence.
- The baseline already has static escape enabled; rows solved by both candidates cannot demonstrate added recenter value.
- The active row improved the local-minimum trace classification but did not reach the goal, so the intervention signal is not a planner-improvement claim.

## Continue / Revise / Stop Decision

- Decision: `continue`.
- Rationale: Controlled trace evidence exists for the semantic pair, but the result is mixed/negative and not strong enough for benchmark-candidate promotion.
- Next step: If more evidence is needed, run a separately predeclared horizon-sensitivity slice or broadened scenario slice before considering benchmark-candidate status.

## Claim Boundary

Report strength is limited to the compact input evidence.
