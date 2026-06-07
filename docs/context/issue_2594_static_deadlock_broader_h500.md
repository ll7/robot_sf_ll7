# Issue #2594 Broader H500 Static-Deadlock Recenter Slice

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2594>

Related:
- Issue #2452
- Issue #2588
- Issue #2590
- Issue #2592
- Issue #2596

## Scope

Issue #2594 extended the Issue #2592 active-row h500 rescue signal to the predeclared
Issue #2588/Issue #2590 static-deadlock controlled slice:

- scenario matrix: `configs/scenarios/sets/issue_2594_static_deadlock_broader_h500.yaml`;
- funnel config: `configs/policy_search/transfer/issue_2594_static_deadlock_broader_h500.yaml`;
- scenarios: `classic_bottleneck_low`, `classic_head_on_corridor_low`, `narrow_passage`;
- seeds: `111`, `112`, `113`;
- horizon: `500`;
- pair 1: `hybrid_rule_v3_fast_progress` versus `issue_2170_static_recenter_only`;
- pair 2: `issue_2170_static_escape_only` versus
  `issue_2170_static_escape_recenter_no_transit`.

This is broader h500 controlled-trace evidence. It is not planner-ranking, benchmark-candidate,
mechanism-transfer, or paper-facing evidence.

## Result

Both h500 pairings completed the 3-scenario x 3-seed slice with all required static-deadlock trace
fields:

- pair-rows: `18`;
- completed pair-rows: `18`;
- terminal rescue pair-rows: `2`;
- already-solved pair-rows: `16`;
- missing required fields: none;
- active terminal-change row: `classic_bottleneck_low`, seed `113`, in both pairings;
- first recenter activation step: `7`;
- baseline outcome on the active row: failed low-progress local-minimum termination at `500` steps;
- intervention outcome on the active row: success at `122` steps.

Overall classification: `broader_delayed_rescue_supported`.

The result supports the delayed-rescue mechanism signal from Issue #2592 under a predeclared 3x3
h500 slice. The claim remains narrow because the broader slice contains only one unsolved active
row; the other sixteen pair-rows were already solved by both comparator and intervention.

## Evidence

Tracked compact evidence:

- [evidence/issue_2594_static_deadlock_broader_h500/summary.json](evidence/issue_2594_static_deadlock_broader_h500/summary.json)
- [evidence/issue_2594_static_deadlock_broader_h500/field_presence.json](evidence/issue_2594_static_deadlock_broader_h500/field_presence.json)
- [evidence/issue_2594_static_deadlock_broader_h500/broader_h500_table.csv](evidence/issue_2594_static_deadlock_broader_h500/broader_h500_table.csv)
- [evidence/issue_2594_static_deadlock_broader_h500/why_first_input.json](evidence/issue_2594_static_deadlock_broader_h500/why_first_input.json)
- [evidence/issue_2594_static_deadlock_broader_h500/why_first_report.md](evidence/issue_2594_static_deadlock_broader_h500/why_first_report.md)

The raw activation trace JSON files stayed in worktree-local ignored `output/` and are disposable.
The tracked files above are the durable review surface.

## Validation

```bash
rtk uv run robot_sf_bench --quiet validate-config --matrix configs/scenarios/sets/issue_2594_static_deadlock_broader_h500.yaml
rtk env LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/validation/run_static_recenter_activation_trace.py --funnel-config configs/policy_search/transfer/issue_2594_static_deadlock_broader_h500.yaml --baseline-candidate hybrid_rule_v3_fast_progress --mechanism-candidate issue_2170_static_recenter_only --stage full_matrix --output-json output/benchmarks/issue_2594_static_deadlock_broader_h500/static_recenter_only_activation_trace.json
rtk env LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/validation/run_static_recenter_activation_trace.py --funnel-config configs/policy_search/transfer/issue_2594_static_deadlock_broader_h500.yaml --baseline-candidate issue_2170_static_escape_only --mechanism-candidate issue_2170_static_escape_recenter_no_transit --stage full_matrix --output-json output/benchmarks/issue_2594_static_deadlock_broader_h500/escape_recenter_pair_activation_trace.json
rtk uv run python scripts/tools/generate_why_first_report.py --input docs/context/evidence/issue_2594_static_deadlock_broader_h500/why_first_input.json --output docs/context/evidence/issue_2594_static_deadlock_broader_h500/why_first_report.md
rtk python -m json.tool docs/context/evidence/issue_2594_static_deadlock_broader_h500/summary.json
rtk python -m json.tool docs/context/evidence/issue_2594_static_deadlock_broader_h500/field_presence.json
rtk python -m json.tool docs/context/evidence/issue_2594_static_deadlock_broader_h500/why_first_input.json
```

Config validation warned that two static-deadlock scenarios have `ped_density=0.0`, and that
`narrow_passage` lacks `metadata.density`. These warnings are inherited from the selected controlled
slice and do not indicate missing execution rows.

## Claim Boundary

The broader h500 slice supports a controlled-trace delayed-rescue interpretation for static
recentering on the observed unsolved active row. It does not establish generalized planner
improvement across static-deadlock scenarios, benchmark-candidate status, transfer, or manuscript
claims. Follow-up Issue #2596 tracks the synthesis decision before any harder unsolved-row expansion
or promotion attempt.
