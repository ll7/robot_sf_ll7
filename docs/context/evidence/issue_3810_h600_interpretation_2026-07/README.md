# Issue 4195 h600 Aggregation Artifact

This directory contains a diagnostic-only aggregation artifact for h600 jobs 13268, 13273.

## Claim Boundary

- Evidence status: `diagnostic-only`.
- Scope: mechanical per-planner aggregation from retrieved local artifacts for issue #4195 checklist item 1.
- This artifact does not assert benchmark success, dissertation-ready evidence, paper-grade evidence, or a planner ranking claim.
- No full benchmark campaign, Slurm submission, graphics processing unit job, retention decision, SNQI recalibration, horizon-sensitivity synthesis, or dissertation claim edit was run for this slice.
- Comparability is limited to shared planner arms whose `scenario_matrix_hash` and `comparability_mapping_hash` match across the two campaign summaries.

## Contents

- `planner_metric_summary.csv`: one row per job, planner, and metric with per-seed values where available plus bootstrap confidence intervals.
- `planner_metric_summary.md`: Markdown rendering of the same rows.
- `comparability_check.json` and `comparability_check.md`: shared-arm scenario matrix comparability check.
- `source_manifest.json`: input paths, campaign metadata, and source file SHA-256 digests.
- `SHA256SUMS`: checksums for generated files in this directory.

## Notes

- Metric rows: 80.
- Shared-arm comparability status: `pass`.
- Comfort rows: 16.
- Comfort per-seed values are not present in `seed_episode_rows.csv`; those rows preserve the campaign-summary aggregate mean confidence interval and are marked `no_seed_episode_column`.
