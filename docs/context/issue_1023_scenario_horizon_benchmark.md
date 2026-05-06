# Issue 1023 Scenario-Horizon Benchmark Surface

Date: 2026-05-06

Related:

- Upstream issue: `ll7/robot_sf_ll7#1023`
- Scenario horizon schedule: `configs/policy_search/scenario_horizons_h500.yaml`
- New benchmark config: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`
- Preflight evidence:
  `docs/context/evidence/issue_1023_scenario_horizons_preflight_2026-05-06/`
- Local full-campaign evidence:
  `docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06/`
- Fixed-horizon reference evidence:
  `docs/context/evidence/camera_ready_all_planners_2026-05-04/`
- H500 policy-search source evidence:
  `docs/context/evidence/policy_search_h500_2026-05-06/`
- Fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

## Goal

Issue 1023 asks whether the h500-derived scenario-specific horizon schedule should become a
paper-facing benchmark surface. This branch makes that schedule runnable through the camera-ready
benchmark path, preserves preflight proof, records a full local non-Slurm campaign requested by the
maintainer, and compares it against the May 4 fixed-horizon reference.

## What Changed

- `robot_sf.benchmark.camera_ready_campaign` now accepts an optional top-level
  `scenario_horizons` YAML path.
- The schedule is joined by scenario name/id and patches each loaded scenario's
  `simulation_config.max_episode_steps` before the map runner sees the matrix.
- Each patched scenario receives `metadata.scenario_horizon` with source, horizon, status, and
  bucket. This keeps the three `planner_blocked` scenarios visible instead of excluding them.
- Preflight and matrix artifacts now report `horizon_mode=scenario_horizons`, the schedule path,
  horizon range, and status counts.
- The versioned issue-1023 benchmark config is paper-facing for validation/comparability/SNQI
  contracts but disables publication bundle export until maintainers decide whether this
  sensitivity surface belongs in a release package.

## Canonical Commands

Local preflight, already run:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --campaign-id issue1023_scenario_horizons_preflight_2026-05-06 \
  --log-level INFO
```

Local full campaign, run from this non-Slurm worktree per maintainer direction:

```bash
.venv/bin/python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --output-root output/benchmarks/issue_1023 \
  --campaign-id issue1023_scenario_horizons_h500_local_2026-05-06 \
  --mode run \
  --log-level INFO
```

Post-run diagnostics:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06
```

Fixed-vs-scenario comparison:

```bash
uv run python scripts/tools/compare_camera_ready_campaigns.py \
  --base-campaign-root docs/context/evidence/camera_ready_all_planners_2026-05-04 \
  --candidate-campaign-root output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06 \
  --output-json output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/fixed_vs_scenario_horizon_comparison.json \
  --output-md output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/fixed_vs_scenario_horizon_comparison.md
```

Canonical Auxme Slurm command retained for reproducibility if maintainers later require a
cluster-side rerun:

```bash
ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
ISSUE791_BENCHMARK_LABEL=issue1023-scenario-horizons-h500 \
ISSUE791_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_1023 \
sbatch SLURM/Auxme/issue_791_benchmark.sl
```

## Preflight Evidence

The tracked preflight bundle records:

- 48 scenarios.
- 7 planners.
- Seed set `eval` = `[111, 112, 113]`.
- Horizon mode `scenario_horizons`.
- Horizon range 102-600 steps.
- Status counts: 45 `recommended`, 3 `planner_blocked`.

The three `planner_blocked` scenarios remain in the matrix with the h600 cap and must be reported
separately in the final comparison. They should not be excluded from aggregate accounting without a
separate paper-table decision.

## Comparison Boundary

The current fixed-horizon reference remains the May 4 camera-ready all-planners evidence. The local
scenario-horizon campaign is valid issue-1023 evidence for this branch, but it is not a Slurm
reproduction and should be labeled as local non-Slurm evidence in PR and paper-facing handoff text.

The tracked fixed-vs-scenario comparison reports:

- planner,
- scenario,
- scenario family,
- success,
- collision,
- near-miss,
- `unfinished_mean = 1 - success_mean` as the route-incomplete comparison metric,
- fallback/degraded/not-available status.

Raw per-episode timeout attribution is not preserved in the compact May 4 fixed-horizon evidence,
so this note does not claim a raw timeout-vs-collision decomposition for that reference. Use
`unfinished_mean` for the comparable route-incomplete signal and the raw local JSONL only for
local-only debugging.

Fallback/degraded statuses must be interpreted through the fail-closed benchmark fallback policy.
`orca` and `sacadrl` keep their existing fallback prereq policies in the config, while
`socnav_sampling` remains fail-fast.

## Local Full Campaign Evidence

The tracked local full-campaign bundle records:

- 7 successful planner runs.
- 1008 episodes.
- Runtime: 819.7857 seconds.
- Campaign warning: SNQI contract status `warn`.
- Analyzer finding: no automated campaign inconsistencies.
- Runtime hotspots: `prediction_planner`, `ppo`, and `social_force`.
- Hardest scenarios by analyzer: `classic_cross_trap_high`,
  `francis2023_narrow_hallway`, `francis2023_robot_crowding`, and
  `classic_station_platform_medium`.
- Verified-simple assessment: `rerun_required` because this campaign does not include the
  verified-simple candidate scenarios.

Planner-level local scenario-horizon outcomes:

| planner | success | collisions | SNQI |
|---|---:|---:|---:|
| goal | 0.0556 | 0.6181 | -0.1904 |
| orca | 0.7569 | 0.1667 | -0.2513 |
| ppo | 0.8056 | 0.1667 | -0.2074 |
| prediction_planner | 0.4931 | 0.4514 | -0.1408 |
| sacadrl | 0.0833 | 0.6667 | -0.2726 |
| social_force | 0.0139 | 0.3819 | -0.9537 |
| socnav_sampling | 0.4028 | 0.5972 | -0.0848 |

Fixed-vs-scenario planner-level deltas for the seven matched planners show success increased under
scenario-specific horizons, but every matched planner also saw higher collision and near-miss means.
The comparison is therefore a horizon-confounding analysis, not a clean benchmark improvement.
`socnav_bench` is a coverage gap because it is present in the May 4 fixed-horizon evidence but not
in the issue-1023 scenario-horizon config.

## Recommendation

Do not promote the scenario-horizon surface as the paper headline benchmark. The local full run
shows the longer scenario-specific budgets reduce route-incompletion and raise success, but the same
surface also increases collisions and near-misses.

Use this evidence as a sensitivity/confounding analysis for the paper-facing benchmark. If the paper
mentions scenario-specific horizons, present them beside the fixed-horizon surface with explicit
safety caveats, the SNQI `warn` status, the local non-Slurm provenance, and the `socnav_bench`
coverage gap.

## Validation Run

Commands run in this branch:

```bash
uv run pytest tests/benchmark/test_camera_ready_campaign.py::test_scenario_horizon_schedule_applies_to_loaded_campaign_scenarios \
  tests/benchmark/test_camera_ready_campaign.py::test_preflight_reports_scenario_horizon_schedule_summary \
  tests/benchmark/test_camera_ready_campaign.py::test_issue_1023_scenario_horizon_config_uses_h500_schedule -q
```

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --campaign-id issue1023_scenario_horizons_preflight_2026-05-06 \
  --log-level INFO
```

```bash
.venv/bin/python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --output-root output/benchmarks/issue_1023 \
  --campaign-id issue1023_scenario_horizons_h500_local_2026-05-06 \
  --mode run \
  --log-level INFO
```

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06
```

```bash
uv run python scripts/tools/compare_camera_ready_campaigns.py \
  --base-campaign-root docs/context/evidence/camera_ready_all_planners_2026-05-04 \
  --candidate-campaign-root output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06 \
  --output-json output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/fixed_vs_scenario_horizon_comparison.json \
  --output-md output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/fixed_vs_scenario_horizon_comparison.md
```

```bash
uv run pytest tests/tools/test_compare_camera_ready_campaigns.py -q
```

The Slurm command remains documented above for optional cluster reproduction, but the issue-1023
full campaign evidence preserved here is the local non-Slurm run requested for this worktree.
