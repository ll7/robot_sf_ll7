<!-- AI-GENERATED (robot_sf#5579, 2026-07-15) - NEEDS-REVIEW -->

# Issue #5579 tuning-budget evidence surface

Status: completed bounded diagnostic with a blocked preregistered read. This directory contains
compact derived results, not raw episode artifacts or a benchmark claim.

* [`tuning_effort_table.csv`](tuning_effort_table.csv) records retained config-history commits
  for the two prediction-MPC target arms and four hybrid incumbents. `unknown` runs/hours mean no
  trial ledger was found; commit counts are not treated as tuning trials.
* The executable packet is
  [`configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml`](../../../../configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml).
* [`sensitivity_report.json`](sensitivity_report.json) is the machine-readable result and records
  its execution-start commit, configuration hash, claim boundary, and source-artifact hashes.
* [`sensitivity_report.md`](sensitivity_report.md) presents the blocked read before the compact
  arm summaries.
* [`sensitivity_candidate_rows.csv`](sensitivity_candidate_rows.csv) contains the 40 target
  candidate summaries and four unchanged incumbent summaries.
* [`checksums.sha256`](checksums.sha256) covers the promoted compact result files.
* [`agent_run_manifest.yaml`](agent_run_manifest.yaml) records the agent-assisted execution and
  validation boundary.

The local CPU campaign produced all 396 preregistered rows: 295 were eligible and 101 were excluded
(92 solver-failure rows and 9 fallback rows). Every eligible candidate and incumbent summary had
zero observed successes. The excluded denominator therefore blocks the preregistered read; these
results do not distinguish a tuning-budget limitation from a structural limitation.

The report remains diagnostic-only. Raw JSONL, generated configs, and execution logs are retained
outside the repository for seven days and were not promoted. No Slurm or GPU submission occurred.
