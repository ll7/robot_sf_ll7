# Issue #2266 Static-Recenter Activation Manifest 2026-06-05

This directory preserves compact durable evidence for the #2266 static-recenter activation
diagnostic.

## Tracked Files

- `summary.json`: machine-readable activation diagnostic classification, per-scenario rows,
  available evidence, missing evidence, and recommendation.
- `activation_table.csv`: two-row held-out smoke activation table with terminal outcomes and
  explicit missing activation/command-source fields.

## Source Evidence

- `docs/context/issue_2221_static_recenter_transfer.md`
- `docs/context/evidence/issue_2221_static_recenter_transfer_2026-06-04/comparison_summary.json`
- `docs/context/evidence/issue_2221_static_recenter_transfer_2026-06-04/manifest.md`
- `docs/context/policy_search/reports/2026-06-04_hybrid_rule_v3_fast_progress_full_matrix.md`
- `docs/context/policy_search/reports/2026-06-04_issue_2170_static_recenter_only_full_matrix.md`

## Claim Boundary

This is activation-diagnostic evidence only. It records that terminal outcomes were durable and
identical, but activation and command-source evidence was not preserved. It must not be cited as
proof that static recentering did or did not activate, and it is not transfer, planner-improvement,
benchmark-strength, or paper-facing evidence.

The source #2221 manifest says raw JSONL outputs were used to derive the tracked summaries and were
intentionally not tracked. A future activation table needs either those untracked local outputs or a
targeted rerun/extractor that preserves activation and selected-command-source fields durably.
