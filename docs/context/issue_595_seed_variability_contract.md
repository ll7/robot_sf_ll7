# Issue 595 Seed-Variability Contract

Date: 2026-03-27
Related issues:
- `robot_sf_ll7#595`
- `amv_benchmark_paper#74`

## Decision

Treat the camera-ready benchmark campaign as the canonical upstream contract for fixed-scenario
multi-seed variability analysis.

This supersedes the older idea of extending only the full-classic benchmark path for this issue.
The camera-ready stack already carries the needed benchmark provenance, seed-policy handling, and
publication-facing artifact structure.

## Frozen Contract

Canonical entrypoint:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_seed_variability_pilot_v1.yaml \
  --label pilot
```

Required config properties:

- explicit scenario manifest
- explicit planner set
- explicit ordered seed list
- explicit bootstrap settings
- paper-facing differential-drive-only profile

Canonical artifact bundle for paper consumption:

- `campaign_manifest.json`
- `run_meta.json`
- `reports/seed_variability_by_scenario.json`
- `reports/seed_variability_by_scenario.csv`
- `reports/seed_episode_rows.csv`
- `reports/statistical_sufficiency.json`

## Artifact Semantics

### `reports/seed_variability_by_scenario.json`

Primary aggregate artifact grouped by `(scenario_id, planner_key)`.

Each row records:

- scenario id
- planner key / algo
- seed count and executed seed list
- per-seed metric means
- across-seed mean/std/cv
- bootstrap confidence interval bounds and half-widths
- campaign/config/git provenance

Confidence method:

- `bootstrap_mean_over_seed_means`

### `reports/seed_episode_rows.csv`

Flat per-episode traceability export for paper-side grouping without path parsing.

Required columns:

- `episode_id`
- `scenario_id`
- `planner_key`
- `algo`
- `seed`
- `repeat_index`
- `success`
- `collision`
- `near_miss`
- `time_to_goal`
- `snqi`

`repeat_index` is a deterministic zero-based index within each
`(scenario_id, planner_key, seed)` group.

### `reports/statistical_sufficiency.json`

Thin side artifact over the seed-variability rows.

Purpose:

- expose seed counts
- expose metric-level CI half-widths
- preserve the bootstrap settings used for those half-widths

This artifact is intentionally narrow. It is not a new benchmark precision framework.

## Recommended Pilot Slice For `amv_benchmark_paper#74`

Config:

- `configs/benchmarks/paper_seed_variability_pilot_v1.yaml`

Scenario subset:

- `classic_crossing_low`
- `classic_head_on_corridor_low`
- `classic_overtaking_low`
- `classic_t_intersection_low`

Planner subset:

- `orca`
- `ppo`

Seed list:

- `111, 112, 113, 114, 115, 116, 117, 118`

Why this slice:

- small enough to run repeatedly
- maintained classic scenarios already used elsewhere in the benchmark stack
- one strong classical planner plus one learned planner gives interpretable contrast without
  overexpanding scope
