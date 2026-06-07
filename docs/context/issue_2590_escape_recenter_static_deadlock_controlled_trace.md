# Issue #2590 Escape-Recenter Static-Deadlock Controlled Trace

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2590>

Related:
- Issue #2452
- Issue #2544
- Issue #2586
- Issue #2588

## Scope

Issue #2590 ran the semantic `escape_recenter_pair` follow-up named by the #2452
`static_deadlock_recovery` suite and the #2588 successor boundary.

The predeclared controlled slice used:

- scenario matrix: `configs/scenarios/sets/issue_2590_escape_recenter_static_deadlock_controlled_trace.yaml`;
- funnel config: `configs/policy_search/transfer/issue_2590_escape_recenter_static_deadlock_controlled_trace.yaml`;
- scenarios: `classic_bottleneck_low`, `classic_head_on_corridor_low`, `narrow_passage`;
- seeds: `111`, `112`, `113`;
- horizon: `120`;
- baseline candidate: `issue_2170_static_escape_only`;
- intervention candidate: `issue_2170_static_escape_recenter_no_transit`.

The 120-step stop rule intentionally matches Issue #2588 for comparability. This is controlled
trace evidence for one semantic baseline/intervention pairing. It is not benchmark-candidate
evidence, planner ranking, mechanism transfer, or paper-facing proof.

## Result

The controlled trace wrote nine matched baseline/intervention pairs:

- all nine pairs completed with `paired_row_status: completed`;
- all required static-deadlock trace fields were present on both baseline and intervention rows;
- four pairs were `mechanism_inactive`;
- four pairs were `comparator_already_solved_case`;
- one pair, `classic_bottleneck_low` seed `113`, was `mechanism_active_trace_changed`;
- zero pairs had a terminal success rescue.

The active row recorded four positive `static_recenter` decision terms beginning at step `7`. The
escape+recenter intervention changed command/trajectory/local-minimum trace state, but both
baseline and intervention still ended in `max_steps` failure. The local-minimum indicator changed
from detected on the escape-only baseline row to not detected on the escape+recenter intervention
row, while the terminal outcome remained unchanged.

Overall classification: `controlled_trace_negative_mixed`.

## Evidence

Tracked compact evidence:

- [evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/summary.json](evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/summary.json)
- [evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/field_presence.json](evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/field_presence.json)
- [evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/controlled_trace_table.csv](evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/controlled_trace_table.csv)
- [evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/why_first_input.json](evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/why_first_input.json)
- [evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/why_first_report.md](evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/why_first_report.md)

The raw `activation_trace.json` used to build this compact evidence stayed in worktree-local ignored
`output/` and is disposable. The tracked files above are the durable review surface.

## Validation

```bash
rtk uv run robot_sf_bench --quiet validate-config --matrix configs/scenarios/sets/issue_2590_escape_recenter_static_deadlock_controlled_trace.yaml
rtk env LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/validation/run_static_recenter_activation_trace.py --funnel-config configs/policy_search/transfer/issue_2590_escape_recenter_static_deadlock_controlled_trace.yaml --baseline-candidate issue_2170_static_escape_only --mechanism-candidate issue_2170_static_escape_recenter_no_transit --stage full_matrix --output-json output/benchmarks/issue_2590_escape_recenter_static_deadlock_controlled_trace/activation_trace.json
rtk uv run python scripts/tools/generate_why_first_report.py --input docs/context/evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/why_first_input.json --output docs/context/evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/why_first_report.md
rtk python -m json.tool docs/context/evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/summary.json
rtk python -m json.tool docs/context/evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/field_presence.json
rtk python -m json.tool docs/context/evidence/issue_2590_escape_recenter_static_deadlock_controlled_trace/why_first_input.json
```

## Claim Boundary

This result extends the #2588 static-deadlock controlled trace to the semantic escape-recenter pair.
It does not establish planner improvement because the only active intervention row changed trace
state without terminal success. Benchmark-candidate promotion would require a separately
predeclared broader contract, such as horizon sensitivity or a broader scenario slice, with no
fallback/degraded/failed rows counted as success.
