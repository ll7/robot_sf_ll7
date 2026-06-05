# Issue #2306 Static-Recenter Activation Trace Evidence

This directory contains compact diagnostic evidence for Issue #2306. The raw instrumented rerun
records were generated in a worktree-local diagnostics directory and were not promoted; the tracked
`summary.json` preserves the activation fields needed for review.

Scope:

- Baseline: `hybrid_rule_v3_fast_progress`
- Mechanism candidate: `issue_2170_static_recenter_only`
- Scenario matrix: `configs/scenarios/sets/issue_2128_heldout_family_transfer_pilot_eval.yaml`
- Seed: `111`
- Horizon: `500`
- Evidence tier: diagnostic-only trace rerun

Command:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_static_recenter_activation_trace.py --output-json <worktree-local diagnostics summary>
```

Result:

- `classic_station_platform_medium`: `mechanism_inactive`
- `francis2023_intersection_wait`: `comparator_already_solved_case`

Static recentering did not activate on either held-out row. This is not benchmark-strength,
transfer, or planner-improvement evidence.
