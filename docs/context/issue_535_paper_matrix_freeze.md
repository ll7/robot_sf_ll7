# Issue #535 Execution Note: Paper Matrix Freeze

This note records the first frozen paper-facing benchmark execution contract.

## Frozen Profile

- config: `configs/benchmarks/paper_experiment_matrix_v1.yaml`
- `paper_facing: true`
- `paper_profile_version: paper-matrix-v1`
- matrix scope: mixed planners with explicit `planner_group` tags (`core|experimental`)
- kinematics set: `["differential_drive"]` (v1 freeze)
- seed policy: pinned in config (`seed_policy` block; resolved seeds emitted in artifacts)

## Canonical Commands

1. Preflight only (no episodes):

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1.yaml \
  --mode preflight \
  --label preflight
```

2. Full execution:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1.yaml \
  --label run
```

3. Inspect matrix summary:

```bash
jq '.rows[0] | {planner_key, planner_group, kinematics, repeats, paper_profile_version}' \
  output/benchmarks/camera_ready/<campaign_id>/reports/matrix_summary.json
```

## Required Artifacts

- `preflight/validate_config.json`
- `preflight/preview_scenarios.json`
- `reports/matrix_summary.json`
- `reports/matrix_summary.csv`
- `reports/campaign_summary.json`
- `campaign_manifest.json`

## Cross-Repo Link

Paper-side decision source: https://github.com/ll7/amv_benchmark_paper/issues/11
