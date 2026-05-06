# Camera-Ready All-Planners SLURM Check 2026-05-04

Date checked: 2026-05-05

Related context:

- [Issue #562 SocNavBench Re-Entry Gate](issue_562_socnav_bench_reentry.md)
- [Issue #691 Benchmark Fallback Policy](issue_691_benchmark_fallback_policy.md)
- [Benchmark: Camera-ready / Scenario Reports](../benchmark_camera_ready.md)
- [SocNav Asset Setup](../socnav_assets_setup.md)

## Goal

Check the new camera-ready benchmark SLURM jobs that should have finished by 2026-05-05, identify
whether any outputs are publication-ready, and record the next decision boundary.

## Scheduler Evidence

`squeue --me` had no active jobs at check time.

Recent benchmark-related `sacct` rows since 2026-05-01:

| Job | Name | State | Exit | Partition | Elapsed | Start | End |
|---|---|---|---|---|---:|---|---|
| 12273 | `rsf-allbench` | `FAILED` | `2:0` | `a30` | 00:32:41 | 2026-05-04 17:11:49 | 2026-05-04 17:44:30 |
| 12274 | `rsf-analyze` | `COMPLETED` | `0:0` | `a30` | 00:00:17 | 2026-05-04 17:44:30 | 2026-05-04 17:44:47 |

SLURM log:

- `output/slurm/12273-issue791-benchmark.out`
- `output/slurm/12274-rsf-analyze.out`

## Campaign

Campaign root:

```text
output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217
```

Config and provenance:

- Config: `configs/benchmarks/paper_experiment_matrix_all_planners_v1.yaml`
- Command: `scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_all_planners_v1.yaml --output-root output/benchmarks/camera_ready --mode run --log-level INFO --label main_latest_all`
- Branch: `benchmark_2026-05-04`
- Commit: `1d7acbaac53b32dd4d656c5a31466b018dd131f6`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Scenarios: 48
- Seeds: 111, 112, 113
- Kinematics: `differential_drive`
- Workers: 1

Analyzer command rerun on 2026-05-05:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217
```

Analyzer artifacts:

- `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/campaign_analysis.json`
- `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/campaign_analysis.md`

## Outcome

The campaign produced partial benchmark evidence but is not publication-ready.

- `benchmark_success=false`
- Runs: 8 total, 7 successful
- Episodes: 1008
- Runtime: 1894.28 s
- Publication bundle: skipped
- Automated campaign-analysis consistency findings: none

Completed planner rows:

| Planner | Group | Mode | Status | Episodes | Success | Collision | Near misses | TTG norm | SNQI |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `ppo` | experimental | native | ok | 144 | 0.2500 | 0.0903 | 3.3542 | 0.9306 | -0.3060 |
| `orca` | core | adapter | ok | 144 | 0.1806 | 0.0347 | 4.9097 | 0.9650 | -0.2589 |
| `socnav_sampling` | experimental | adapter | ok | 144 | 0.1736 | 0.5278 | 1.4097 | 0.9528 | -0.1390 |
| `prediction_planner` | experimental | adapter | ok | 144 | 0.0694 | 0.2083 | 8.3681 | 0.9867 | -0.1945 |
| `goal` | core | native | ok | 144 | 0.0139 | 0.2361 | 2.9097 | 0.9981 | -0.1656 |
| `social_force` | core | adapter | ok | 144 | 0.0000 | 0.2083 | 2.3264 | 1.0000 | -0.8535 |
| `sacadrl` | experimental | adapter | ok | 144 | 0.0000 | 0.3889 | 3.5694 | 1.0000 | -0.2834 |
| `socnav_bench` | experimental | unknown/degraded | failed | 0 | n/a | n/a | n/a | n/a | n/a |

## Failure Cause

`socnav_bench` failed its fail-fast preflight because required SocNavBench assets were absent:

```text
Failed to find the wayptnav_data dir at
third_party/socnavbench/wayptnav_data
```

The run summary reports:

```text
SocNav preflight failed for algorithm 'socnav_bench': SocNavBench control pipeline parameters
failed to load. Ensure the SocNavBench data directories exist. See docs/socnav_assets_setup.md
and run uv run python scripts/tools/prepare_socnav_assets.py.
```

This matches the prior issue #562 re-entry gate: `socnav_bench` remains blocked on local
license-sensitive assets and should not be treated as benchmark evidence until a focused fail-fast
probe completes without fallback.

## Additional Diagnostics

Runtime:

- `prediction_planner` dominated runtime at 1006.07 s, about 53% of campaign wall time.
- Slowest prediction-planner slices were `classic_bottleneck_low`,
  `classic_realworld_double_bottleneck_high`, and `classic_bottleneck_high`.

SNQI:

- SNQI contract status: `pass`
- Spearman rank alignment: 0.75
- Outcome separation: 0.2029
- Caveats: collisions were degenerate on this slice, and one-at-a-time weight ablations changed
  planner ordering.

AMV/comparability:

- AMV coverage status: `warn`
- Required AMV dimensions were not observed in the coverage summary.
- Alyassi comparability mapping lists all current scenario families as `amv_extension`, with only
  metric-level comparability/proxy mappings available.

Scenario and route caveats:

- Route-clearance warnings: 18 scenarios.
- Negative clearance margins were reported for `classic_merging_low`,
  `classic_merging_medium`, and `classic_station_platform_medium`.
- Verified-simple assessment: `rerun_required`; this campaign did not include the verified-simple
  candidate scenarios.

Statistical sufficiency:

- 336 scenario/planner rows were reported, corresponding to 48 scenarios times 7 completed
  planners.
- The failed `socnav_bench` row is absent from sufficiency evidence.
- Some three-seed confidence intervals remain wide; for example, maximum success half-width was
  0.6667 on `prediction_planner` / `francis2023_exiting_elevator`.

## Recommendation

Do not publish this campaign as the final all-planners paper bundle. Use it only as partial
internal evidence for the seven planners that completed.

Preferred next path:

1. Hydrate SocNavBench assets locally without committing them:

   ```bash
   uv run python scripts/tools/prepare_socnav_assets.py \
     --report-json output/tmp/socnav_asset_report.json
   ```

   If assets already exist in `output/SocNavBench`, stage them with:

   ```bash
   uv run python scripts/tools/prepare_socnav_assets.py --copy-from-source
   ```

2. Run the focused re-entry probe on a compute node before spending another full all-planners
   allocation. On the auxme login node, treat this as the SLURM payload command, not a local
   benchmark command:

   ```bash
   uv run python scripts/tools/run_camera_ready_benchmark.py \
     --config configs/benchmarks/socnav_bench_reentry_probe.yaml \
     --label socnav_asset_sanity \
     --log-level INFO
   ```

3. Only after that probe completes with `socnav_bench` episodes, resubmit the all-planners
   camera-ready campaign through SLURM and rerun the analyzer.

Alternative if SocNavBench assets cannot be made available:

- Keep `socnav_bench` out of the publication matrix, run an explicit config that excludes it, and
  document the exclusion as `not available` / dependency-blocked rather than degraded success.
- The completed core rows (`goal`, `orca`, `social_force`) are useful sanity evidence, but the
  all-planners campaign itself remains fail-closed because `benchmark_success=false`.

Before public paper claims:

- Resolve or explicitly scope the AMV coverage warning.
- Inspect the 18 route-clearance warnings, especially the negative-margin classic scenarios.
- Treat adapter-heavy planners and high projection rates as caveats in any comparison table.
