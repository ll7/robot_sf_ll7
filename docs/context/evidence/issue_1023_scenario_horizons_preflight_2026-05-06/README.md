# Issue 1023 Scenario-Horizon Preflight Evidence

Compact, tracked preflight evidence for the issue-1023 scenario-specific horizon benchmark
surface.

## Source

Generated from:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --campaign-id issue1023_scenario_horizons_preflight_2026-05-06 \
  --log-level INFO
```

Source output root:
`output/benchmarks/camera_ready/issue1023_scenario_horizons_preflight_2026-05-06/`

## Contents

- `campaign_manifest.json`: generated campaign manifest.
- `preflight/validate_config.json`: config validation payload.
- `preflight/preview_scenarios.json`: scenario preview with patched horizon metadata.
- `reports/matrix_summary.{json,csv}`: planner matrix definition.
- `reports/amv_coverage_summary.{json,md}`: AMV coverage preflight.
- `reports/comparability_matrix.{json,md}`: Alyassi comparability preflight.
- `manifest.sha256`: checksums for the copied evidence files.

## Key Preflight Facts

- Scenario count: 48.
- Planner count: 7.
- Seed schedule: `eval` = `[111, 112, 113]`.
- Horizon mode: `scenario_horizons`.
- Horizon source: `configs/policy_search/scenario_horizons_h500.yaml`.
- Horizon range: 102 to 600 steps.
- Horizon status counts: 45 `recommended`, 3 `planner_blocked`.

## Storage Decision

This bundle intentionally preserves only compact preflight evidence. Full benchmark run outputs,
episode JSONL files, videos, Slurm logs, and publication bundles should stay out of git unless a
future paper-release handoff promotes small reviewable summaries or external artifact pointers.
