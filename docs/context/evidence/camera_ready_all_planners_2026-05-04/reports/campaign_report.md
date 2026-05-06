# Camera-Ready Benchmark Campaign Report

- Campaign ID: `paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217`
- Name: `paper_experiment_matrix_all_planners_v1`
- Created (UTC): `2026-05-04T15:12:17.036639Z`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Scenario matrix hash: `8ac8ab9387f4`
- Git commit: `1d7acbaac53b32dd4d656c5a31466b018dd131f6`
- Runtime sec: `1894.2798869700637`
- Episodes/sec: `0.5321283337977657`
- Interpretation profile: `baseline-ready-core`
- Command: `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_all_planners_v1.yaml --output-root output/benchmarks/camera_ready --mode run --log-level INFO --label main_latest_all`

## Planner Summary

| planner | algo | planner group | kinematics | status | started (UTC) | runtime (s) | episodes | eps/s | success | collisions | snqi | proj_rate | infeasible_rate |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | core | differential_drive | ok | 2026-05-04T15:29:04.807179Z | 92.5234 | 144 | 1.5564 | 0.0139 | 0.2361 | -0.1656 | 0.0000 | 0.0000 |
| orca | orca | core | differential_drive | ok | 2026-05-04T15:32:28.153313Z | 153.8507 | 144 | 0.9360 | 0.1806 | 0.0347 | -0.2589 | 0.7723 | 0.7723 |
| ppo | ppo | experimental | differential_drive | ok | 2026-05-04T15:35:03.301061Z | 233.9481 | 144 | 0.6155 | 0.2500 | 0.0903 | -0.3060 | 0.0000 | 0.0000 |
| prediction_planner | prediction_planner | experimental | differential_drive | ok | 2026-05-04T15:12:17.429530Z | 1006.0669 | 144 | 0.1431 | 0.0694 | 0.2083 | -0.1945 | 0.0000 | 0.0000 |
| sacadrl | sacadrl | experimental | differential_drive | ok | 2026-05-04T15:40:43.276787Z | 186.4920 | 144 | 0.7722 | 0.0000 | 0.3889 | -0.2834 | 0.0000 | 0.0000 |
| social_force | social_force | core | differential_drive | ok | 2026-05-04T15:30:38.622622Z | 108.2956 | 144 | 1.3297 | 0.0000 | 0.2083 | -0.8535 | 0.1359 | 0.1359 |
| socnav_sampling | socnav_sampling | experimental | differential_drive | ok | 2026-05-04T15:38:58.548405Z | 103.4304 | 144 | 1.3922 | 0.1736 | 0.5278 | -0.1390 | 0.9722 | 0.9722 |
| socnav_bench | socnav_bench | experimental | differential_drive | failed | 2026-05-04T15:43:51.009713Z | 0.1830 | 0 | 0.0000 | nan | nan | nan | 0.0000 | 0.0000 |

## Readiness & Degraded/Fallback Status

| planner | planner group | execution mode | execution detail | planner cmd | benchmark cmd | projection policy | readiness status | tier | preflight | learned contract | run status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goal | core | native | unspecified | unicycle_vw | unknown | unknown | native | baseline-ready | ok | not_applicable | ok |
| orca | core | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | baseline-ready | ok | not_applicable | ok |
| ppo | experimental | native | unspecified | unicycle_vw | unknown | unknown | native | experimental | ok | pass | ok |
| prediction_planner | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| sacadrl | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| social_force | core | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | baseline-ready | ok | not_applicable | ok |
| socnav_sampling | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| socnav_bench | experimental | unknown | unspecified | unknown | unknown | unknown | degraded | unknown | unknown | not_applicable | failed |

Planners in fallback/degraded mode:
- `socnav_bench`: readiness=degraded, preflight=unknown, tier=unknown

## SocNav Strict-vs-Fallback Disclosure

| planner | algo | planner group | prereq policy | preflight status | readiness status |
|---|---|---|---|---|---|
| goal | goal | core | fail-fast | ok | native |
| orca | orca | core | fallback | ok | adapter |
| ppo | ppo | experimental | fail-fast | ok | native |
| prediction_planner | prediction_planner | experimental | fail-fast | ok | adapter |
| sacadrl | sacadrl | experimental | fallback | ok | adapter |
| social_force | social_force | core | fail-fast | ok | adapter |
| socnav_sampling | socnav_sampling | experimental | fail-fast | ok | adapter |
| socnav_bench | socnav_bench | experimental | fail-fast | unknown | degraded |

- No within-campaign strict-vs-fallback pair available for direct comparison.

## Scenario Diagnostics

- Per-scenario breakdown: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/scenario_breakdown.csv`
- Per-family breakdown: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/scenario_family_breakdown.csv`

## Kinematics Parity

- Planner x kinematics parity table: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/kinematics_parity_table.csv`
- Skipped planner/kinematics combinations: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/kinematics_skipped_combinations.csv`

## AMV Coverage Contract

- Coverage JSON: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/amv_coverage_summary.json`
- Coverage Markdown: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/amv_coverage_summary.md`
- Coverage status: `warn` (enforcement: `warn`)

## Alyassi Comparability

- Comparability JSON: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/comparability_matrix.json`
- Comparability Markdown: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/comparability_matrix.md`
- Mapping version: `alyassi-comparability-v1`

## SNQI Contract

- Contract status: `pass`
- Rank alignment (Spearman): `0.75`
- Outcome separation: `0.20291047090364647`
- Positioning recommendation: `strengthen_as_operational_multi_objective_aggregation`
- Weights version: `snqi_weights_camera_ready_v3`
- Baseline version: `snqi_baseline_camera_ready_v3`
- Diagnostics JSON: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/snqi_diagnostics.json`
- Diagnostics Markdown: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/snqi_diagnostics.md`
- Sensitivity CSV: `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217/reports/snqi_sensitivity.csv`

## Campaign Warnings

- Planner 'socnav_bench' failed for kinematics 'differential_drive': SocNav preflight failed for algorithm 'socnav_bench': SocNavBench control pipeline parameters failed to load. Ensure the SocNavBench data directories exist. See `docs/socnav_assets_setup.md` and run `uv run python scripts/tools/prepare_socnav_assets.py`.. Check missing dependencies/models or choose a different prereq policy.
- Planner failure recorded: planner='socnav_bench' kinematics='differential_drive' status='failed' most_likely_reason='RuntimeError("SocNav preflight failed for algorithm 'socnav_bench': SocNavBench control pipeline parameters failed to load. Ensure the SocNavBench data directories exist. See `docs/socnav_assets_setup.md` and run `uv run python scripts/tools/prepare_socnav_assets.py`.. Check missing dependencies/models or choose a different prereq policy.")'
- Publication bundle export skipped because benchmark_success=false.

## Failed Planners And Likely Reasons

| planner | status | most likely reason |
|---|---|---|
| socnav_bench | failed | RuntimeError("SocNav preflight failed for algorithm 'socnav_bench': SocNavBench control pipeline parameters failed to load. Ensure the SocNavBench data directories exist. See `docs/socnav_assets_setup.md` and run `uv run python scripts/tools/prepare_socnav_assets.py`.. Check missing dependencies/models or choose a different prereq policy.") |
