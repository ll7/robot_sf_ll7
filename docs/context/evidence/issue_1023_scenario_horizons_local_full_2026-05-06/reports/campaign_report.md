# Camera-Ready Benchmark Campaign Report

- Campaign ID: `issue1023_scenario_horizons_h500_local_2026-05-06`
- Name: `paper_experiment_matrix_v1_scenario_horizons_h500`
- Created (UTC): `2026-05-06T12:01:35.670439Z`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Scenario matrix hash: `fed70fa089d8`
- Git commit: `521a5d798cb1aca9dd62c20eb76790f4cf5ee79d`
- Runtime sec: `819.7856649390014`
- Episodes/sec: `1.229589688025349`
- Interpretation profile: `baseline-ready-core`
- Command: `/home/luttkule/git/robot_sf_ll7.worktrees/origin-1023-validate-scenario-h500-bench/.venv/bin/python /home/luttkule/git/robot_sf_ll7.worktrees/origin-1023-validate-scenario-h500-bench/scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml --output-root output/benchmarks/issue_1023 --campaign-id issue1023_scenario_horizons_h500_local_2026-05-06 --mode run --log-level INFO`

## Planner Summary

| planner | algo | planner group | kinematics | status | started (UTC) | runtime (s) | episodes | eps/s | success | collisions | snqi | proj_rate | infeasible_rate |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | core | differential_drive | ok | 2026-05-06T12:08:13.057999Z | 45.0499 | 144 | 3.1965 | 0.0556 | 0.6181 | -0.1904 | 0.0000 | 0.0000 |
| orca | orca | core | differential_drive | ok | 2026-05-06T12:10:14.588783Z | 71.7924 | 144 | 2.0058 | 0.7569 | 0.1667 | -0.2513 | 0.6469 | 0.6469 |
| ppo | ppo | experimental | differential_drive | ok | 2026-05-06T12:11:26.865266Z | 140.0520 | 144 | 1.0282 | 0.8056 | 0.1667 | -0.2074 | 0.0000 | 0.0000 |
| prediction_planner | prediction_planner | experimental | differential_drive | ok | 2026-05-06T12:01:35.798848Z | 396.7669 | 144 | 0.3629 | 0.4931 | 0.4514 | -0.1408 | 0.0000 | 0.0000 |
| sacadrl | sacadrl | experimental | differential_drive | ok | 2026-05-06T12:14:17.129889Z | 57.7939 | 144 | 2.4916 | 0.0833 | 0.6667 | -0.2726 | 0.0000 | 0.0000 |
| social_force | social_force | core | differential_drive | ok | 2026-05-06T12:08:58.591008Z | 75.5148 | 144 | 1.9069 | 0.0139 | 0.3819 | -0.9537 | 0.0726 | 0.0726 |
| socnav_sampling | socnav_sampling | experimental | differential_drive | ok | 2026-05-06T12:13:47.409492Z | 29.2304 | 144 | 4.9264 | 0.4028 | 0.5972 | -0.0848 | 0.9730 | 0.9730 |

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

- No fallback/degraded planners detected.

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

- No within-campaign strict-vs-fallback pair available for direct comparison.

## Scenario Diagnostics

- Per-scenario breakdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/scenario_breakdown.csv`
- Per-family breakdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/scenario_family_breakdown.csv`

## Kinematics Parity

- Planner x kinematics parity table: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/kinematics_parity_table.csv`
- Skipped planner/kinematics combinations: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/kinematics_skipped_combinations.csv`

## AMV Coverage Contract

- Coverage JSON: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/amv_coverage_summary.json`
- Coverage Markdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/amv_coverage_summary.md`
- Coverage status: `warn` (enforcement: `warn`)

## Alyassi Comparability

- Comparability JSON: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/comparability_matrix.json`
- Comparability Markdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/comparability_matrix.md`
- Mapping version: `alyassi-comparability-v1`

## SNQI Contract

- Contract status: `warn`
- Rank alignment (Spearman): `0.32142857142857145`
- Outcome separation: `0.21996748943799715`
- Positioning recommendation: `strengthen_as_operational_multi_objective_aggregation`
- Weights version: `snqi_weights_camera_ready_v3`
- Baseline version: `snqi_baseline_camera_ready_v3`
- Diagnostics JSON: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/snqi_diagnostics.json`
- Diagnostics Markdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/snqi_diagnostics.md`
- Sensitivity CSV: `output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/reports/snqi_sensitivity.csv`

## Campaign Warnings

- SNQI contract status=warn with snqi_contract.enforcement=warn; campaign marked with soft contract warning.

## Failed Planners And Likely Reasons

- No failed/partial/not-available planners.
