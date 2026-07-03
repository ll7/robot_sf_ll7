# Issue 4195 h600 Aggregation Artifact

This directory contains diagnostic-only h600 interpretation artifacts for jobs 13268, 13273.

## Claim Boundary

- Evidence status: `diagnostic-only`.
- Scope: per-planner aggregation plus issue #4195 checklist items 3-5.
- This artifact does not assert benchmark success, dissertation-ready evidence, paper-grade evidence, or a planner ranking claim.
- No full benchmark campaign, Slurm submission, graphics processing unit job, retention decision, or dissertation claim edit was run for this slice.
- SNQI recalibration, horizon-sensitivity, and exposure diagnostics are diagnostic-only interpretation artifacts.
- Comparability is limited to shared planner arms whose `scenario_matrix_hash` and `comparability_mapping_hash` match across the two campaign summaries.

## Contents

- `planner_metric_summary.csv`: one row per job, planner, and metric with per-seed values where available plus bootstrap confidence intervals.
- `planner_metric_summary.md`: Markdown rendering of the same rows.
- `comparability_check.json` and `comparability_check.md`: shared-arm scenario matrix comparability check.
- `snqi_recalibration_bundle.json` and `snqi_recalibration_report.md`: analysis-only h600 recalibration and h500 reversal checks.
- `horizon_sensitivity_report.json` and `horizon_sensitivity_report.md`: h600-vs-h500 rank-stability and rank-flip diagnostic.
- `interaction_exposure_diagnostics.json` and `interaction_exposure_diagnostics.md`: episode-level exposure coverage readiness; fail-closed when required fields are absent.
- `h600_mechanism_labels_sidecar.csv` `h600_interaction_exposure_sidecar.csv`: retained h600 episode-row sidecars from issue #4242. All retained rows are explicit `not_derivable_missing_trace`; no geometry-only mechanism labels or exposure zero-imputation are used.
- `h600_mechanism_exposure_backfill_manifest.json` `h600_mechanism_exposure_backfill_report.md`: compact sidecar provenance, status counts, and diagnostic-only claim boundary.
- `source_manifest.json`: input paths, campaign metadata, and source file SHA-256 digests.
- `SHA256SUMS`: checksums for generated files in this directory.

## Notes

- Metric rows: 80.
- Shared-arm comparability status: `pass`.
- SNQI recalibration status: `ok`.
- Horizon-sensitivity status: `ok`.
- Interaction-exposure status: `blocked_missing_required_fields`.
- Comfort rows: 16.
- Comfort per-seed values are not present in `seed_episode_rows.csv`; those rows preserve the campaign-summary aggregate mean confidence interval and are marked `no_seed_episode_column`.
