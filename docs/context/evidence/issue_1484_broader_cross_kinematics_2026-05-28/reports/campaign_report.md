# Camera-Ready Benchmark Campaign Report

- Campaign ID: `issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441`
- Name: `issue_1484_broader_cross_kinematics`
- Created (UTC): `2026-05-28T15:24:41.783271Z`
- Scenario matrix: `configs/scenarios/sets/cross_kinematics_v1.yaml`
- Scenario matrix hash: `ab8352ea8083`
- Git commit: `f35fb3b5d47c217594a7d6e923b14bc22772ab5f`
- Runtime sec: `86.0142236140091`
- Episodes/sec: `0.2441456670496499`
- Campaign status: `benchmark_success`
- Campaign execution status: `completed`
- Evidence status: `valid`
- Status reason: `all planner rows were benchmark-success`
- Benchmark success: `True`
- Successful rows: `21` / `21`
- Accepted unavailable/excluded rows: `0`
- Unexpected failed rows: `0`
- Row status summary: `{'successful_evidence_rows': 21, 'accepted_unavailable_rows': 0, 'unexpected_failed_rows': 0, 'fallback_or_degraded_rows': 0}`
- Interpretation profile: `issue-1484-broader-cross-kinematics`
- Command: `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/issue_1484_broader_cross_kinematics.yaml --output-root <ignored-output-root>/benchmarks/issue_1484 --mode run --log-level INFO --label issue1484-broader-cross-kinematics-20260528-gf35fb3b5 --skip-publication-bundle`

## Planner Summary

| planner | algo | planner group | kinematics | status | started (UTC) | runtime (s) | episodes | eps/s | success | collisions | snqi | proj_rate | infeasible_rate |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | core | differential_drive | ok | 2026-05-28T15:24:42.213841Z | 14.9820 | 1 | 0.0667 | 0.0000 | 0.0000 | -0.4127 | 0.0000 | 0.0000 |
| goal | goal | core | bicycle_drive | ok | 2026-05-28T15:24:57.598097Z | 0.6015 | 1 | 1.6625 | 0.0000 | 0.0000 | -0.4297 | 0.0000 | 0.0000 |
| goal | goal | core | holonomic | ok | 2026-05-28T15:24:58.602638Z | 0.6001 | 1 | 1.6664 | 0.0000 | 0.0000 | -0.1083 | 0.0000 | 0.0000 |
| orca | orca | core | differential_drive | ok | 2026-05-28T15:25:02.992621Z | 0.9102 | 1 | 1.0987 | 0.0000 | 0.0000 | -0.1754 | 0.9750 | 0.9750 |
| orca | orca | core | bicycle_drive | ok | 2026-05-28T15:25:04.303096Z | 0.9204 | 1 | 1.0865 | 0.0000 | 0.0000 | -0.1149 | 0.9125 | 0.9125 |
| orca | orca | core | holonomic | ok | 2026-05-28T15:25:05.625072Z | 0.7314 | 1 | 1.3671 | 0.0000 | 0.0000 | -0.5054 | 0.0000 | 0.0000 |
| ppo | ppo | experimental | differential_drive | ok | 2026-05-28T15:25:06.760976Z | 22.1824 | 1 | 0.0451 | 0.0000 | 0.0000 | -0.3649 | 0.0000 | 0.0000 |
| ppo | ppo | experimental | bicycle_drive | ok | 2026-05-28T15:25:29.546611Z | 4.1474 | 1 | 0.2411 | 0.0000 | 0.0000 | -0.4017 | 0.0000 | 0.0000 |
| ppo | ppo | experimental | holonomic | ok | 2026-05-28T15:25:34.209916Z | 2.6857 | 1 | 0.3723 | 0.0000 | 0.0000 | -0.2914 | 0.0000 | 0.0000 |
| prediction_planner | prediction_planner | experimental | differential_drive | ok | 2026-05-28T15:25:37.494711Z | 6.3611 | 1 | 0.1572 | 0.0000 | 0.0000 | -0.1000 | 0.0000 | 0.0000 |
| prediction_planner | prediction_planner | experimental | bicycle_drive | ok | 2026-05-28T15:25:44.258493Z | 6.2081 | 1 | 0.1611 | 0.0000 | 0.0000 | -0.1023 | 0.0000 | 0.0000 |
| prediction_planner | prediction_planner | experimental | holonomic | ok | 2026-05-28T15:25:50.870075Z | 6.2260 | 1 | 0.1606 | 0.0000 | 0.0000 | -0.1004 | 0.0000 | 0.0000 |
| sacadrl | sacadrl | experimental | differential_drive | ok | 2026-05-28T15:25:57.500406Z | 1.5163 | 1 | 0.6595 | 0.0000 | 0.0000 | -0.4623 | 0.0000 | 0.0000 |
| sacadrl | sacadrl | experimental | bicycle_drive | ok | 2026-05-28T15:25:59.607900Z | 1.6761 | 1 | 0.5966 | 0.0000 | 0.0000 | -0.4926 | 0.0000 | 0.0000 |
| sacadrl | sacadrl | experimental | holonomic | ok | 2026-05-28T15:26:01.879681Z | 1.5438 | 1 | 0.6478 | 0.0000 | 0.0000 | -0.4092 | 0.0000 | 0.0000 |
| social_force | social_force | core | differential_drive | ok | 2026-05-28T15:24:59.607667Z | 0.9116 | 1 | 1.0970 | 0.0000 | 0.0000 | -0.2663 | 0.1500 | 0.1500 |
| social_force | social_force | core | bicycle_drive | ok | 2026-05-28T15:25:00.919668Z | 0.6353 | 1 | 1.5740 | 0.0000 | 0.0000 | -0.1290 | 0.2750 | 0.2750 |
| social_force | social_force | core | holonomic | ok | 2026-05-28T15:25:01.957336Z | 0.6314 | 1 | 1.5837 | 0.0000 | 0.0000 | -0.4121 | 0.0000 | 0.0000 |
| socnav_sampling | socnav_sampling | experimental | differential_drive | ok | 2026-05-28T15:26:04.019698Z | 0.9128 | 1 | 1.0955 | 0.0000 | 1.0000 | -0.2810 | 1.0000 | 1.0000 |
| socnav_sampling | socnav_sampling | experimental | bicycle_drive | ok | 2026-05-28T15:26:05.333386Z | 0.7977 | 1 | 1.2535 | 0.0000 | 0.0000 | -0.1085 | 1.0000 | 1.0000 |
| socnav_sampling | socnav_sampling | experimental | holonomic | ok | 2026-05-28T15:26:06.533478Z | 0.8194 | 1 | 1.2204 | 0.0000 | 0.0000 | -0.1000 | 0.0000 | 0.0000 |

## Readiness & Degraded/Fallback Status

| planner | planner group | execution mode | execution detail | planner cmd | benchmark cmd | projection policy | readiness status | tier | preflight | learned contract | run status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goal | core | native | unspecified | unicycle_vw | unknown | unknown | native | baseline-ready | ok | not_applicable | ok |
| goal | core | native | unspecified | unicycle_vw | unknown | unknown | native | baseline-ready | ok | not_applicable | ok |
| goal | core | native | unspecified | holonomic_vxy_world | unknown | unknown | native | baseline-ready | ok | not_applicable | ok |
| orca | core | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | baseline-ready | ok | not_applicable | ok |
| orca | core | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | baseline-ready | ok | not_applicable | ok |
| orca | core | adapter | direct_holonomic_world_velocity | holonomic_vxy_world | holonomic_vxy_world | world_velocity_passthrough | adapter | baseline-ready | ok | not_applicable | ok |
| ppo | experimental | native | unspecified | unicycle_vw | unknown | unknown | native | experimental | ok | pass | ok |
| ppo | experimental | native | unspecified | unicycle_vw | unknown | unknown | native | experimental | ok | pass | ok |
| ppo | experimental | native | unspecified | holonomic_vxy_world | unknown | unknown | native | experimental | ok | warn | ok |
| prediction_planner | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| prediction_planner | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| prediction_planner | experimental | adapter | unspecified | holonomic_vxy_world | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| sacadrl | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| sacadrl | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| sacadrl | experimental | adapter | unspecified | holonomic_vxy_world | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| social_force | core | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | baseline-ready | ok | not_applicable | ok |
| social_force | core | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | baseline-ready | ok | not_applicable | ok |
| social_force | core | adapter | direct_holonomic_world_velocity | holonomic_vxy_world | holonomic_vxy_world | world_velocity_passthrough | adapter | baseline-ready | ok | not_applicable | ok |
| socnav_sampling | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| socnav_sampling | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| socnav_sampling | experimental | adapter | unspecified | holonomic_vxy_world | unknown | unknown | adapter | experimental | ok | not_applicable | ok |

- No fallback/degraded planners detected.

## SocNav Strict-vs-Fallback Disclosure

| planner | algo | planner group | prereq policy | preflight status | readiness status |
|---|---|---|---|---|---|
| goal | goal | core | fail-fast | ok | native |
| goal | goal | core | fail-fast | ok | native |
| goal | goal | core | fail-fast | ok | native |
| orca | orca | core | fallback | ok | adapter |
| orca | orca | core | fallback | ok | adapter |
| orca | orca | core | fallback | ok | adapter |
| ppo | ppo | experimental | fail-fast | ok | native |
| ppo | ppo | experimental | fail-fast | ok | native |
| ppo | ppo | experimental | fail-fast | ok | native |
| prediction_planner | prediction_planner | experimental | fail-fast | ok | adapter |
| prediction_planner | prediction_planner | experimental | fail-fast | ok | adapter |
| prediction_planner | prediction_planner | experimental | fail-fast | ok | adapter |
| sacadrl | sacadrl | experimental | fallback | ok | adapter |
| sacadrl | sacadrl | experimental | fallback | ok | adapter |
| sacadrl | sacadrl | experimental | fallback | ok | adapter |
| social_force | social_force | core | fail-fast | ok | adapter |
| social_force | social_force | core | fail-fast | ok | adapter |
| social_force | social_force | core | fail-fast | ok | adapter |
| socnav_sampling | socnav_sampling | experimental | skip-with-warning | ok | adapter |
| socnav_sampling | socnav_sampling | experimental | skip-with-warning | ok | adapter |
| socnav_sampling | socnav_sampling | experimental | skip-with-warning | ok | adapter |

- No within-campaign strict-vs-fallback pair available for direct comparison.

## Scenario Diagnostics

- Per-scenario breakdown: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/scenario_breakdown.csv`
- Per-family breakdown: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/scenario_family_breakdown.csv`

## Kinematics Parity

- Planner x kinematics parity table: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/kinematics_parity_table.csv`
- Skipped planner/kinematics combinations: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/kinematics_skipped_combinations.csv`

## AMV Coverage Contract

- Coverage JSON: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/amv_coverage_summary.json`
- Coverage Markdown: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/amv_coverage_summary.md`
- Coverage status: `pass` (enforcement: `warn`)

## Alyassi Comparability

- Comparability JSON: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/comparability_matrix.json`
- Comparability Markdown: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/comparability_matrix.md`
- Mapping version: `alyassi-comparability-v1`

## SNQI Contract

- Contract status: `warn`
- Rank alignment (Spearman): `0.9596411929959341`
- Outcome separation: `0.0`
- Positioning recommendation: `strengthen_as_operational_multi_objective_aggregation`
- Weights version: `snqi_weights_camera_ready_v3`
- Baseline version: `snqi_baseline_camera_ready_v3`
- Diagnostics JSON: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/snqi_diagnostics.json`
- Diagnostics Markdown: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/snqi_diagnostics.md`
- Sensitivity CSV: `<ignored-output-root>/benchmarks/issue_1484/issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441/reports/snqi_sensitivity.csv`

## Accepted Unavailable/Excluded Planners

- No accepted unavailable/excluded planners.

## Unexpected Failed/Partial Planners

- No unexpected failed/partial planners.

## Campaign Warnings

- No campaign-level warnings.
