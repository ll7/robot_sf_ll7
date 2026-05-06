# Camera-Ready Benchmark Campaign Report

- Campaign ID: `issue1023_scenario_horizons_candidates_local_2026-05-06`
- Name: `paper_experiment_matrix_v1_scenario_horizons_h500`
- Created (UTC): `2026-05-06T17:52:18.544160Z`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Scenario matrix hash: `fed70fa089d8`
- Git commit: `424e8deea5c62c5aeafd13c8c4a01181e76c38aa`
- Runtime sec: `1966.5879543470073`
- Episodes/sec: `0.6590094265223588`
- Interpretation profile: `baseline-ready-core`
- Command: `/home/luttkule/git/robot_sf_ll7.worktrees/origin-1023-validate-scenario-h500-bench/.venv/bin/python /home/luttkule/git/robot_sf_ll7.worktrees/origin-1023-validate-scenario-h500-bench/scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml --output-root output/benchmarks/issue_1023 --campaign-id issue1023_scenario_horizons_candidates_local_2026-05-06 --mode run --log-level INFO`

## Planner Summary

| planner | algo | planner group | kinematics | status | started (UTC) | runtime (s) | episodes | eps/s | success | collisions | snqi | proj_rate | infeasible_rate |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | core | differential_drive | ok | 2026-05-06T17:58:55.769510Z | 44.7352 | 144 | 3.2189 | 0.0556 | 0.6181 | -0.1904 | 0.0000 | 0.0000 |
| hybrid_rule_v3_fast_progress_static_escape | hybrid_rule_local_planner | experimental | differential_drive | ok | 2026-05-06T18:15:27.965848Z | 576.6053 | 144 | 0.2497 | 0.9028 | 0.0278 | -0.0874 | 0.0000 | 0.0000 |
| orca | orca | core | differential_drive | ok | 2026-05-06T18:00:56.619994Z | 71.2254 | 144 | 2.0217 | 0.7569 | 0.1667 | -0.2513 | 0.6469 | 0.6469 |
| ppo | ppo | experimental | differential_drive | ok | 2026-05-06T18:02:08.334194Z | 140.1452 | 144 | 1.0275 | 0.8056 | 0.1667 | -0.2074 | 0.0000 | 0.0000 |
| prediction_planner | prediction_planner | experimental | differential_drive | ok | 2026-05-06T17:52:18.668771Z | 396.6133 | 144 | 0.3631 | 0.4931 | 0.4514 | -0.1408 | 0.0000 | 0.0000 |
| sacadrl | sacadrl | experimental | differential_drive | ok | 2026-05-06T18:04:58.486725Z | 57.2194 | 144 | 2.5166 | 0.0833 | 0.6667 | -0.2726 | 0.0000 | 0.0000 |
| scenario_adaptive_hybrid_orca_v1 | hybrid_rule_local_planner | experimental | differential_drive | ok | 2026-05-06T18:05:56.195892Z | 570.8285 | 144 | 0.2523 | 0.9097 | 0.0278 | -0.0835 | 0.0000 | 0.0000 |
| social_force | social_force | core | differential_drive | ok | 2026-05-06T17:59:40.994347Z | 75.1397 | 144 | 1.9164 | 0.0139 | 0.3819 | -0.9537 | 0.0726 | 0.0726 |
| socnav_sampling | socnav_sampling | experimental | differential_drive | ok | 2026-05-06T18:04:28.978949Z | 29.0170 | 144 | 4.9626 | 0.4028 | 0.5972 | -0.0848 | 0.9730 | 0.9730 |

## Readiness & Degraded/Fallback Status

| planner | planner group | execution mode | execution detail | planner cmd | benchmark cmd | projection policy | readiness status | tier | preflight | learned contract | run status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goal | core | native | unspecified | unicycle_vw | unknown | unknown | native | baseline-ready | ok | not_applicable | ok |
| hybrid_rule_v3_fast_progress_static_escape | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| orca | core | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | baseline-ready | ok | not_applicable | ok |
| ppo | experimental | native | unspecified | unicycle_vw | unknown | unknown | native | experimental | ok | pass | ok |
| prediction_planner | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| sacadrl | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| scenario_adaptive_hybrid_orca_v1 | experimental | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | experimental | ok | not_applicable | ok |
| social_force | core | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | baseline-ready | ok | not_applicable | ok |
| socnav_sampling | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |

- No fallback/degraded planners detected.

## SocNav Strict-vs-Fallback Disclosure

| planner | algo | planner group | prereq policy | preflight status | readiness status |
|---|---|---|---|---|---|
| goal | goal | core | fail-fast | ok | native |
| hybrid_rule_v3_fast_progress_static_escape | hybrid_rule_local_planner | experimental | fail-fast | ok | adapter |
| orca | orca | core | fallback | ok | adapter |
| ppo | ppo | experimental | fail-fast | ok | native |
| prediction_planner | prediction_planner | experimental | fail-fast | ok | adapter |
| sacadrl | sacadrl | experimental | fallback | ok | adapter |
| scenario_adaptive_hybrid_orca_v1 | hybrid_rule_local_planner | experimental | fail-fast | ok | adapter |
| social_force | social_force | core | fail-fast | ok | adapter |
| socnav_sampling | socnav_sampling | experimental | fail-fast | ok | adapter |

- No within-campaign strict-vs-fallback pair available for direct comparison.

## Scenario Diagnostics

- Per-scenario breakdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/scenario_breakdown.csv`
- Per-family breakdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/scenario_family_breakdown.csv`

## Kinematics Parity

- Planner x kinematics parity table: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/kinematics_parity_table.csv`
- Skipped planner/kinematics combinations: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/kinematics_skipped_combinations.csv`

## AMV Coverage Contract

- Coverage JSON: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/amv_coverage_summary.json`
- Coverage Markdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/amv_coverage_summary.md`
- Coverage status: `warn` (enforcement: `warn`)

## Alyassi Comparability

- Comparability JSON: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/comparability_matrix.json`
- Comparability Markdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/comparability_matrix.md`
- Mapping version: `alyassi-comparability-v1`

## SNQI Contract

- Contract status: `fail`
- Rank alignment (Spearman): `0.18333333333333332`
- Outcome separation: `0.2034624335903155`
- Positioning recommendation: `strengthen_as_operational_multi_objective_aggregation`
- Weights version: `snqi_weights_camera_ready_v3`
- Baseline version: `snqi_baseline_camera_ready_v3`
- Diagnostics JSON: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/snqi_diagnostics.json`
- Diagnostics Markdown: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/snqi_diagnostics.md`
- Sensitivity CSV: `output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06/reports/snqi_sensitivity.csv`

## Campaign Warnings

- SNQI contract status=fail with snqi_contract.enforcement=warn; campaign marked with soft contract warning.

## Failed Planners And Likely Reasons

- No failed/partial/not-available planners.
