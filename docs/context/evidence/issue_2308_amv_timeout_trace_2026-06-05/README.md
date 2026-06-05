# Issue #2308 AMV Timeout Trace Evidence

This directory contains the compact, reviewable evidence for Issue #2308. The raw rerun JSONL
traces were generated in a worktree-local diagnostics directory and were not promoted; `summary.json`
preserves the small fields needed for review.

Scope:

- Stage: `amv_actuation_smoke`
- Scenario: `classic_cross_trap_high`
- Seed: `101`
- Horizon: `80`
- Baseline: `hybrid_rule_v3_fast_progress`
- Intervention: `actuation_aware_hybrid_rule_v0`
- Evidence tier: diagnostic-only synthetic AMV actuation evidence

Commands:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_fast_progress --stage amv_actuation_smoke --output-dir <worktree-local diagnostics dir>/hybrid_rule_v3_fast_progress
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_policy_search_candidate.py --candidate actuation_aware_hybrid_rule_v0 --stage amv_actuation_smoke --output-dir <worktree-local diagnostics dir>/actuation_aware_hybrid_rule_v0
```

Result classification: `feasibility_improved_but_route_blocked`.

The actuation-aware candidate reduced total command clip steps from `22` to `15`, while yaw-rate
saturation stayed zero and both candidates timed out with about `3.6 m` remaining. This is not
benchmark-strength or calibrated AMV evidence.
