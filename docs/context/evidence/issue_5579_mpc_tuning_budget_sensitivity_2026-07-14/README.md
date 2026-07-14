# Issue #5579 tuning-budget evidence surface

Status: preregistration and configuration-history evidence only. This directory does not contain
episode results or a benchmark claim.

* [`tuning_effort_table.csv`](tuning_effort_table.csv) records retained config-history commits
  for the two prediction-MPC target arms and four hybrid incumbents. `unknown` runs/hours mean no
  trial ledger was found; commit counts are not treated as tuning trials.
* The executable packet is
  [`configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml`](../../../../configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml).
* The runner's compact derived report names are `sensitivity_report.json`,
  `sensitivity_report.md`, and `sensitivity_candidate_rows.csv`. Raw episode JSONL stays in
  worktree-local `output/` and is not durable evidence.

The report is diagnostic-only. Fallback, degraded, failed, and unavailable rows must remain
excluded, and an incomplete denominator blocks the preregistered read. No campaign was run when
this evidence surface was created.
