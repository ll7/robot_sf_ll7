# Issue #2596 Static-Deadlock Recenter Claim Boundary

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2596>
Date: 2026-06-11
Status: current, diagnostic-only synthesis.

## Goal

Synthesize the static-deadlock recenter evidence from Issues #2588, #2590, #2592, and #2594 before
adding more diagnostic families or promoting the mechanism beyond controlled-trace status.

This note decides what the current evidence supports, what remains diagnostic-only, and what the
next smallest empirical step should be if the lane continues.

## Evidence Chain

| Issue | Scope | Result | Claim boundary |
| --- | --- | --- | --- |
| [#2588](issue_2588_static_deadlock_controlled_trace.md) | `static_recenter_only` controlled trace, 3 scenarios x 3 seeds, h120 | One active row changed trace/local-minimum state, but zero terminal rescues. | `controlled_trace_negative_mixed`; no planner-improvement claim. |
| [#2590](issue_2590_escape_recenter_static_deadlock_controlled_trace.md) | `escape_recenter_pair` controlled trace, same h120 slice | Same active row changed trace/local-minimum state, but zero terminal rescues. | `controlled_trace_negative_mixed`; no planner-improvement claim. |
| [#2592](issue_2592_static_deadlock_active_row_h500.md) | Selected active row rerun at h500 | Both recenter pairings converted the baseline 500-step local-minimum failure into success at step `122`. | `delayed_rescue_candidate` for one selected active row only. |
| [#2594](issue_2594_static_deadlock_broader_h500.md) | Predeclared 3-scenario x 3-seed h500 slice for both pairings | All 18 pair-rows completed with required fields; the same `classic_bottleneck_low`, seed `113` row was rescued in both pairings; the other 16 pair-rows were already solved by comparator and intervention. | `broader_delayed_rescue_supported`, still controlled-trace evidence. |

## Decision

Decision: `promotion_blocked_by_scope`.

Observed evidence supports a narrow delayed-rescue mechanism signal for static recentering on the
single unsolved active row found so far. It does not support generalized planner improvement,
benchmark-candidate status, transfer, or paper-facing promotion because the broader h500 slice added
completion and field-presence confidence but did not add additional unsolved active rescue rows.

The current evidence is still useful: it shows that the h120 negative/mixed trace was horizon-limited
on `classic_bottleneck_low`, seed `113`, and that both semantic recenter pairings can rescue that row
when given h500. The limitation is equally clear: most rows in the broader slice were already solved,
so they cannot demonstrate mechanism value.

## Next Smallest Empirical Step

If the lane continues, the next empirical issue should be a harder unsolved-row expansion, not another
repeat of the already-solved 3x3 slice. The child should predeclare:

- a source for additional rows expected to be unsolved by the comparator at the chosen horizon;
- the same static-deadlock trace-field contract used by #2588/#2590/#2592/#2594;
- a stop rule that treats no new unsolved active rescue rows as `synthesize_stop`;
- an explicit claim boundary of controlled-trace diagnostic evidence only.

If no credible harder unsolved-row source is available, the lane should stop at controlled-trace
evidence and route research effort to another mechanism blocker.

## Mechanism Closure Update

The mechanism closure state for static recentering is:

- held-out transfer: `stop`, because the unsolved row stayed inactive;
- h120 static-deadlock controlled traces: `controlled_trace_negative_mixed`;
- selected h500 active row: `delayed_rescue_candidate`;
- broader h500 slice: `broader_delayed_rescue_supported`;
- promotion decision: `promotion_blocked_by_scope`.

## Validation

Validated this docs-only synthesis with:

```bash
rtk uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2596_static_deadlock_recenter_claim_boundary.md --path docs/context/mechanism_closure_status.md --path docs/context/README.md --path docs/context/catalog.yaml
rtk git diff --check
```

No new benchmark, planner, metric, or schema evidence was generated for this issue.
