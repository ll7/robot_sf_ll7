# Issue #2221 Static Recenter Transfer Evidence Manifest

Date: 2026-06-04

This directory contains compact durable evidence for the static-recentering held-out transfer smoke.
Raw JSONL outputs were used to derive the tracked summaries and are intentionally not tracked.

## Tracked Files

- `comparison_summary.json`: paired baseline/mechanism outcome summary and deltas.

## Tracked Reports

- `docs/context/policy_search/reports/2026-06-04_hybrid_rule_v3_fast_progress_full_matrix.md`
- `docs/context/policy_search/reports/2026-06-04_issue_2170_static_recenter_only_full_matrix.md`

## Canonical Command Shape

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress \
  --stage full_matrix \
  --funnel-config configs/policy_search/transfer/issue_2221_static_recenter_heldout_smoke.yaml \
  --output-dir output/policy_search/issue2221/hybrid_rule_v3_fast_progress/heldout_smoke \
  --workers 1 \
  --allow-expensive-stage
```

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate issue_2170_static_recenter_only \
  --stage full_matrix \
  --funnel-config configs/policy_search/transfer/issue_2221_static_recenter_heldout_smoke.yaml \
  --output-dir output/policy_search/issue2221/issue_2170_static_recenter_only/heldout_smoke \
  --workers 1 \
  --allow-expensive-stage
```
