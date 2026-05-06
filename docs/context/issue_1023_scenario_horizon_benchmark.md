# Issue 1023 Scenario-Horizon Benchmark Surface

Date: 2026-05-06

Related:

- Upstream issue: `ll7/robot_sf_ll7#1023`
- Scenario horizon schedule: `configs/policy_search/scenario_horizons_h500.yaml`
- New benchmark config: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`
- Preflight evidence:
  `docs/context/evidence/issue_1023_scenario_horizons_preflight_2026-05-06/`
- Fixed-horizon reference evidence:
  `docs/context/evidence/camera_ready_all_planners_2026-05-04/`
- H500 policy-search source evidence:
  `docs/context/evidence/policy_search_h500_2026-05-06/`
- Fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

## Goal

Issue 1023 asks whether the h500-derived scenario-specific horizon schedule should become a
paper-facing benchmark surface. The immediate implementation goal is to make that schedule runnable
through the camera-ready benchmark path, preserve preflight proof, and define the remaining full-run
comparison gate.

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
  contracts but disables publication bundle export until the full rerun is reviewed.

## Canonical Commands

Local preflight, already run:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --campaign-id issue1023_scenario_horizons_preflight_2026-05-06 \
  --log-level INFO
```

Canonical Auxme Slurm run, not submitted from this worktree because `local.machine.md` has
`allow_slurm_submission: false`:

```bash
ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
ISSUE791_BENCHMARK_LABEL=issue1023-scenario-horizons-h500 \
ISSUE791_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_1023 \
sbatch SLURM/Auxme/issue_791_benchmark.sl
```

Post-run diagnostics once the Slurm campaign exists:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1023/<campaign_id>
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

The current fixed-horizon reference remains the May 4 camera-ready all-planners evidence. This
branch does not replace that evidence because the scenario-specific benchmark run has not yet been
executed on Slurm.

The full issue-1023 comparison should report fixed-horizon vs scenario-horizon deltas by:

- planner,
- scenario,
- scenario family,
- success,
- collision,
- near-miss,
- timeout/unfinished outcome,
- fallback/degraded/not-available status.

Fallback/degraded statuses must be interpreted through the fail-closed benchmark fallback policy.
`orca` and `sacadrl` keep their existing fallback prereq policies in the config, while
`socnav_sampling` remains fail-fast.

## Recommendation

Do not promote the scenario-horizon surface into the paper yet. Treat this branch as the runnable
paper-facing setup and preflight proof.

After the Slurm rerun, present both fixed-horizon and scenario-specific results if longer budgets
materially change success without hiding safety regressions. If success improves while collisions,
near-misses, fallback/degraded execution, or `planner_blocked` concentration worsen, keep the fixed
horizon as the headline surface and use the scenario-horizon results only as horizon-confounding
analysis.

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

Full Slurm execution and post-run diagnostics remain open because this local machine is marked
`allow_slurm_submission: false`.
