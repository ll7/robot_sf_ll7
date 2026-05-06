# Camera-Ready All-Planners Evidence (2026-05-04)

Compact, tracked evidence copied from the May 4 camera-ready all-planners campaign output.

## Source

`output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/`

## Contents

- `campaign/`: campaign manifest and run metadata.
- `reports/`: compact campaign, comparability, matrix, scenario, and statistical reports.
- `runs/*/summary.json`: per-planner run summaries.
- `manifest.sha256`: checksums for the copied files.

## Storage Decision

This bundle keeps the small, reviewable campaign evidence in git. Raw per-planner episode JSONL,
large seed-variability tables, Slurm logs, videos, and coverage artifacts are intentionally not
tracked. They are reproducible or should be archived outside git only when needed for a release
package or paper artifact bundle.
