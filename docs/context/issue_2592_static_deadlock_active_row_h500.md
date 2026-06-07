# Issue #2592 Static-Deadlock Active-Row H500 Horizon Sensitivity

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2592>

Related:
- Issue #2452
- Issue #2588
- Issue #2590
- Issue #2594

## Scope

Issue #2592 tested whether the only active static-recenter row from Issues #2588 and #2590 was
horizon-limited under the 120-step controlled trace stop rule.

The predeclared one-row sensitivity slice used:

- scenario matrix: `configs/scenarios/sets/issue_2592_static_deadlock_active_row_h500.yaml`;
- funnel config: `configs/policy_search/transfer/issue_2592_static_deadlock_active_row_h500.yaml`;
- scenario: `classic_bottleneck_low`;
- seed: `113`;
- horizon: `500`;
- pair 1: `hybrid_rule_v3_fast_progress` versus `issue_2170_static_recenter_only`;
- pair 2: `issue_2170_static_escape_only` versus
  `issue_2170_static_escape_recenter_no_transit`.

This is a selected active-row horizon-sensitivity probe. It is not planner-ranking,
benchmark-candidate, mechanism-transfer, or paper-facing evidence.

## Result

Both h500 pairings completed with all required static-deadlock trace fields:

- pair count: `2`;
- completed pairs: `2`;
- terminal rescue pairs: `2`;
- activation count total: `8`;
- first activation step: `7` in both intervention rows;
- baseline outcome: failed local-minimum timeout at `500` steps in both pairings;
- intervention outcome: success at `122` steps in both pairings.

Both rows classified as `mechanism_active_terminal_changed`. The 120-step traces from Issue #2588
and Issue #2590 were therefore horizon-limited for this selected active row: the same recenter trace
signal that changed local-minimum state at h120 becomes a terminal rescue under h500.

Overall classification: `delayed_rescue_candidate`.

## Evidence

Tracked compact evidence:

- [evidence/issue_2592_static_deadlock_active_row_h500/summary.json](evidence/issue_2592_static_deadlock_active_row_h500/summary.json)
- [evidence/issue_2592_static_deadlock_active_row_h500/field_presence.json](evidence/issue_2592_static_deadlock_active_row_h500/field_presence.json)
- [evidence/issue_2592_static_deadlock_active_row_h500/horizon_sensitivity_table.csv](evidence/issue_2592_static_deadlock_active_row_h500/horizon_sensitivity_table.csv)
- [evidence/issue_2592_static_deadlock_active_row_h500/why_first_input.json](evidence/issue_2592_static_deadlock_active_row_h500/why_first_input.json)
- [evidence/issue_2592_static_deadlock_active_row_h500/why_first_report.md](evidence/issue_2592_static_deadlock_active_row_h500/why_first_report.md)

The raw activation trace JSON files stayed in worktree-local ignored `output/` and are disposable.
The tracked files above are the durable review surface.

## Validation

```bash
rtk uv run robot_sf_bench --quiet validate-config --matrix configs/scenarios/sets/issue_2592_static_deadlock_active_row_h500.yaml
rtk env LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/validation/run_static_recenter_activation_trace.py --funnel-config configs/policy_search/transfer/issue_2592_static_deadlock_active_row_h500.yaml --baseline-candidate hybrid_rule_v3_fast_progress --mechanism-candidate issue_2170_static_recenter_only --stage full_matrix --output-json output/benchmarks/issue_2592_static_deadlock_active_row_h500/static_recenter_only_activation_trace.json
rtk env LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/validation/run_static_recenter_activation_trace.py --funnel-config configs/policy_search/transfer/issue_2592_static_deadlock_active_row_h500.yaml --baseline-candidate issue_2170_static_escape_only --mechanism-candidate issue_2170_static_escape_recenter_no_transit --stage full_matrix --output-json output/benchmarks/issue_2592_static_deadlock_active_row_h500/escape_recenter_pair_activation_trace.json
rtk uv run python scripts/tools/generate_why_first_report.py --input docs/context/evidence/issue_2592_static_deadlock_active_row_h500/why_first_input.json --output docs/context/evidence/issue_2592_static_deadlock_active_row_h500/why_first_report.md
rtk python -m json.tool docs/context/evidence/issue_2592_static_deadlock_active_row_h500/summary.json
rtk python -m json.tool docs/context/evidence/issue_2592_static_deadlock_active_row_h500/field_presence.json
rtk python -m json.tool docs/context/evidence/issue_2592_static_deadlock_active_row_h500/why_first_input.json
```

## Claim Boundary

This result supports a narrow `delayed_rescue_candidate` hypothesis for a selected static-deadlock
active row. It does not establish planner improvement across the suite. A broader predeclared h500
slice or scenario/seed probe is required before any benchmark-candidate, transfer, or paper-facing
claim. Follow-up Issue #2594 tracks that broader h500 static-deadlock slice.
