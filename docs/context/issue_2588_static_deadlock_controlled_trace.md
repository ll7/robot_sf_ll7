# Issue #2588 Static-Deadlock Controlled Trace

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2588>

Related:
- Issue #2452
- Issue #2544
- Issue #2586
- Issue #2590

## Scope

Issue #2588 ran the first matched baseline/intervention controlled trace for the #2452
`static_deadlock_recovery` suite after Issue #2586 made the required row fields reportable.

The predeclared controlled slice used:

- scenario matrix: `configs/scenarios/sets/issue_2588_static_deadlock_controlled_trace.yaml`;
- funnel config: `configs/policy_search/transfer/issue_2588_static_deadlock_controlled_trace.yaml`;
- scenarios: `classic_bottleneck_low`, `classic_head_on_corridor_low`, `narrow_passage`;
- seeds: `111`, `112`, `113`;
- horizon: `120`;
- baseline candidate: `hybrid_rule_v3_fast_progress`;
- intervention candidate: `issue_2170_static_recenter_only`.

This is controlled-trace evidence for one predeclared baseline/intervention pairing. It is not
benchmark-candidate evidence, planner ranking, mechanism transfer, or paper-facing proof.

## Result

The controlled trace wrote nine matched baseline/intervention pairs:

- all nine pairs completed with `paired_row_status: completed`;
- all required static-deadlock trace fields were present on both baseline and intervention rows;
- four pairs were `mechanism_inactive`;
- four pairs were `comparator_already_solved_case`;
- one pair, `classic_bottleneck_low` seed `113`, was `mechanism_active_trace_changed`;
- zero pairs had a terminal success rescue.

The active row recorded four positive `static_recenter` decision terms beginning at step `7`. The
intervention changed command/trajectory/local-minimum trace state, but both baseline and
intervention still ended in `max_steps` failure. The local-minimum indicator changed from detected
on the baseline row to not detected on the intervention row, while the terminal outcome remained
unchanged.

Overall classification: `controlled_trace_negative_mixed`.

## Evidence

Tracked compact evidence:

- [evidence/issue_2588_static_deadlock_controlled_trace/summary.json](evidence/issue_2588_static_deadlock_controlled_trace/summary.json)
- [evidence/issue_2588_static_deadlock_controlled_trace/field_presence.json](evidence/issue_2588_static_deadlock_controlled_trace/field_presence.json)
- [evidence/issue_2588_static_deadlock_controlled_trace/controlled_trace_table.csv](evidence/issue_2588_static_deadlock_controlled_trace/controlled_trace_table.csv)
- [evidence/issue_2588_static_deadlock_controlled_trace/why_first_input.json](evidence/issue_2588_static_deadlock_controlled_trace/why_first_input.json)
- [evidence/issue_2588_static_deadlock_controlled_trace/why_first_report.md](evidence/issue_2588_static_deadlock_controlled_trace/why_first_report.md)

The raw `activation_trace.json` used to build this compact evidence stayed in worktree-local ignored
`output/` and is disposable. The tracked files above are the durable review surface.

## Validation

```bash
rtk uv run pytest tests/validation/test_static_recenter_activation_trace.py -q
rtk uv run robot_sf_bench --quiet validate-config --matrix configs/scenarios/sets/issue_2588_static_deadlock_controlled_trace.yaml
rtk env LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/validation/run_static_recenter_activation_trace.py --funnel-config configs/policy_search/transfer/issue_2588_static_deadlock_controlled_trace.yaml --baseline-candidate hybrid_rule_v3_fast_progress --mechanism-candidate issue_2170_static_recenter_only --stage full_matrix --output-json output/benchmarks/issue_2588_static_deadlock_controlled_trace/activation_trace.json
rtk uv run python scripts/tools/generate_why_first_report.py --input docs/context/evidence/issue_2588_static_deadlock_controlled_trace/why_first_input.json --output docs/context/evidence/issue_2588_static_deadlock_controlled_trace/why_first_report.md
rtk python -m json.tool docs/context/evidence/issue_2588_static_deadlock_controlled_trace/summary.json
rtk python -m json.tool docs/context/evidence/issue_2588_static_deadlock_controlled_trace/field_presence.json
rtk python -m json.tool docs/context/evidence/issue_2588_static_deadlock_controlled_trace/why_first_input.json
```

## Claim Boundary

This result moves the #2452 static-deadlock suite from reportability-only smoke to a bounded
controlled-trace result for one intervention. It does not establish planner improvement because the
only active intervention row changed trace state without terminal success. Benchmark-candidate
promotion would require a separately predeclared broader contract. Follow-up issue #2590 tracks the
next `escape_recenter_pair` controlled trace, with an explicit choice of same horizon, longer
horizon, or another justified stop rule before any benchmark-candidate discussion.
