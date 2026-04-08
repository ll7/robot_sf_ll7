# Issue 751 Seed-Variability Pilot Execution

Date: 2026-04-08

Related issues:
- Upstream: `ll7/robot_sf_ll7#751`
- Export contract: `ll7/robot_sf_ll7#595`
- Downstream manuscript consumer: `ll7/amv_benchmark_paper#74`

## Scope

This note records the fixed-scenario multi-seed pilot execution used to provide
paper-ready seed-variability evidence for downstream manuscript issue 74. It is a
small pilot slice, not a full benchmark matrix rerun and not evidence for
field-wide generalization.

Pilot slice:
- Scenarios: `classic_crossing_low`, `classic_head_on_corridor_low`,
  `classic_overtaking_low`, `classic_t_intersection_low`
- Planners: `orca`, `ppo`
- Kinematics: `differential_drive`
- Seeds: `111,112,113,114,115,116,117,118`
- Repeats: one episode per scenario/planner/seed

## Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_seed_variability_pilot_v1.yaml \
  --label pilot
```

Analyzer command:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/paper_seed_variability_pilot_v1_pilot_20260408_082538
```

## Run Provenance

- Campaign id: `paper_seed_variability_pilot_v1_pilot_20260408_082538`
- Campaign root: `output/benchmarks/camera_ready/paper_seed_variability_pilot_v1_pilot_20260408_082538`
- Config: `configs/benchmarks/paper_seed_variability_pilot_v1.yaml`
- Scenario matrix: `configs/scenarios/sets/paper_seed_variability_pilot_v1.yaml`
- Git commit recorded by the run: `237ff48825104d70bab562f439fd896fb8832e61`
- Git branch recorded by the run:
  `751-run-fixed-scenario-multi-seed-pilot-and-export-paper-ready-seed-variability-evidence-74-downstream`
- Benchmark success: `true`
- Total runs: `2`
- Successful runs: `2`
- Total episodes: `64`
- Runtime: `98.37006720800127` seconds

Confidence metadata:
- Method: `bootstrap_mean_over_seed_means`
- Confidence: `0.95`
- Bootstrap samples: `1000`
- Bootstrap seed: `123`

## Paper-Ready Artifacts

Use these generated artifacts for downstream issue 74:

- Aggregate seed-variability JSON:
  `output/benchmarks/camera_ready/paper_seed_variability_pilot_v1_pilot_20260408_082538/reports/seed_variability_by_scenario.json`
- Aggregate seed-variability CSV:
  `output/benchmarks/camera_ready/paper_seed_variability_pilot_v1_pilot_20260408_082538/reports/seed_variability_by_scenario.csv`
- Per-episode seed traceability CSV:
  `output/benchmarks/camera_ready/paper_seed_variability_pilot_v1_pilot_20260408_082538/reports/seed_episode_rows.csv`
- Statistical sufficiency JSON:
  `output/benchmarks/camera_ready/paper_seed_variability_pilot_v1_pilot_20260408_082538/reports/statistical_sufficiency.json`
- Campaign analysis report:
  `output/benchmarks/camera_ready/paper_seed_variability_pilot_v1_pilot_20260408_082538/reports/campaign_analysis.md`
- Scenario difficulty report:
  `output/benchmarks/camera_ready/paper_seed_variability_pilot_v1_pilot_20260408_082538/reports/scenario_difficulty_analysis.md`

The visualization-ready artifact for downstream plots is
`reports/seed_variability_by_scenario.csv` for aggregate rows and
`reports/seed_episode_rows.csv` for per-seed traces. The per-episode CSV was
verified to contain populated `planner_key` and `algo` values for both `orca`
and `ppo`.

## Observed Pilot Results

| Scenario | Planner | Seeds | Episodes | Success mean | Collision mean | Near-miss mean | Near-miss 95% CI | TTG norm mean | SNQI mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `classic_crossing_low` | `orca` | 8 | 8 | 0 | 0 | 2.5 | [0.375, 5] | 1 | n/a |
| `classic_crossing_low` | `ppo` | 8 | 8 | 0 | 0 | 2.12 | [0.375, 4.38] | 1 | n/a |
| `classic_head_on_corridor_low` | `orca` | 8 | 8 | 0 | 0 | 3.25 | [1, 6] | 1 | n/a |
| `classic_head_on_corridor_low` | `ppo` | 8 | 8 | 0 | 0 | 1.25 | [0, 3] | 1 | n/a |
| `classic_overtaking_low` | `orca` | 8 | 8 | 0 | 0 | 0 | [0, 0] | 1 | n/a |
| `classic_overtaking_low` | `ppo` | 8 | 8 | 0 | 0 | 0 | [0, 0] | 1 | n/a |
| `classic_t_intersection_low` | `orca` | 8 | 8 | 0 | 0 | 0 | [0, 0] | 1 | n/a |
| `classic_t_intersection_low` | `ppo` | 8 | 8 | 0 | 0 | 1.5 | [0, 3.75] | 1 | n/a |

Interpretation:
- The pilot demonstrates seed-dependent variation in near-miss counts for
  `classic_crossing_low`, `classic_head_on_corridor_low`, and the `ppo`
  `classic_t_intersection_low` row.
- Success, collision, and normalized time-to-goal were flat in this low-density
  pilot; do not use this run to claim broad success or safety variation.
- SNQI values are not used here because the run reported the SNQI contract as a
  soft warning/fail state.

## Warnings

The run completed successfully but reported these caveats:

- `Adjusted degenerate baseline for 'time_to_goal_norm' (p95 <= med) using fallback width 0.05`
- `Adjusted degenerate baseline for 'collisions' (p95 <= med) using fallback width 1`
- `SNQI contract status=fail with snqi_contract.enforcement=warn; campaign marked with soft contract warning.`

These warnings do not invalidate the seed-traceability export, but they limit
how the pilot should be described. Use it as seed-variability evidence, not as a
headline SNQI or overall benchmark-performance claim.

## Validation

Commands run for this issue:

```bash
uv run ruff check robot_sf/benchmark/seed_variance.py \
  tests/benchmark/test_seed_variance.py \
  tests/benchmark/test_camera_ready_campaign.py
```

```bash
uv run pytest -o addopts='' \
  tests/benchmark/test_seed_variance.py \
  tests/benchmark/test_camera_ready_campaign.py \
  -k 'seed_episode_rows or seed_variability_pilot or paper_seed_variability or seed_variability'
```

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_seed_variability_pilot_v1.yaml \
  --label pilot
```

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/paper_seed_variability_pilot_v1_pilot_20260408_082538
```

Validation checks added:
- `build_seed_episode_rows` now preserves planner metadata after metric
  flattening.
- Camera-ready campaign tests now assert the generated `seed_episode_rows.csv`
  contains populated planner and algorithm values, not just the column headers.
