# Camera-Ready Benchmark Campaign Report

- Campaign ID: `issue1454-s10-h500-candidates`
- Name: `issue_1454_s10_scenario_horizons_h500_candidates`
- Created (UTC): `2026-05-22T14:48:37.106517Z`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Scenario matrix hash: `8973cf0d26dd`
- Git commit: `4941ac48f1f4e65053bbfcbbc94a55a336fad9ea`
- Runtime sec: `34507.69299255498`
- Episodes/sec: `0.16691930119010615`
- Interpretation profile: `issue-1454-s10-scenario-horizons-h500-candidates`
- Command: `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/issue_1454_s10_scenario_horizons_h500_candidates.yaml --campaign-id issue1454-s10-h500-candidates`

## Planner Summary

| planner | algo | planner group | kinematics | status | started (UTC) | runtime (s) | episodes | eps/s | success | collisions | snqi | proj_rate | infeasible_rate |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | core | differential_drive | ok | 2026-05-22T14:48:37.317496Z | 258.4807 | 480 | 1.8570 | 0.0479 | 0.6125 | -0.2594 | 0.0000 | 0.0000 |
| hybrid_rule_v3_fast_progress | hybrid_rule_local_planner | experimental | differential_drive | ok | 2026-05-22T16:02:17.366573Z | 6291.2782 | 480 | 0.0763 | 0.7875 | 0.0292 | -0.1160 | 0.0000 | 0.0000 |
| hybrid_rule_v3_fast_progress_static_escape | hybrid_rule_local_planner | experimental | differential_drive | ok | 2026-05-22T17:47:09.277901Z | 5885.3257 | 480 | 0.0816 | 0.8646 | 0.0354 | -0.1069 | 0.0000 | 0.0000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous | hybrid_rule_local_planner | experimental | differential_drive | ok | 2026-05-22T19:25:15.229581Z | 6223.2562 | 480 | 0.0771 | 0.8771 | 0.0250 | -0.0972 | 0.0000 | 0.0000 |
| orca | orca | core | differential_drive | ok | 2026-05-22T15:00:05.504852Z | 416.3719 | 480 | 1.1528 | 0.7750 | 0.1604 | -0.2476 | 0.6507 | 0.6507 |
| ppo | ppo | experimental | differential_drive | ok | 2026-05-22T15:07:02.472562Z | 433.6603 | 480 | 1.1069 | 0.7854 | 0.1938 | -0.2415 | 0.0000 | 0.0000 |
| prediction_planner | prediction_planner | experimental | differential_drive | ok | 2026-05-22T15:14:16.730871Z | 2283.0118 | 480 | 0.2102 | 0.5312 | 0.4000 | -0.1815 | 0.0000 | 0.0000 |
| sacadrl | sacadrl | experimental | differential_drive | ok | 2026-05-22T15:55:39.186933Z | 397.5824 | 480 | 1.2073 | 0.0688 | 0.6604 | -0.3514 | 0.0000 | 0.0000 |
| scenario_adaptive_hybrid_orca_v1 | hybrid_rule_local_planner | experimental | differential_drive | ok | 2026-05-22T21:08:59.112314Z | 5851.2596 | 480 | 0.0820 | 0.8729 | 0.0333 | -0.1037 | 0.0000 | 0.0000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard | hybrid_rule_local_planner | experimental | differential_drive | ok | 2026-05-22T22:46:31.712397Z | 5831.3324 | 480 | 0.0823 | 0.8729 | 0.0333 | -0.1037 | 0.0000 | 0.0000 |
| social_force | social_force | core | differential_drive | ok | 2026-05-22T14:52:56.383575Z | 428.5337 | 480 | 1.1201 | 0.0167 | 0.3854 | -0.9916 | 0.0765 | 0.0765 |
| socnav_sampling | socnav_sampling | experimental | differential_drive | ok | 2026-05-22T15:52:20.565432Z | 198.0263 | 480 | 2.4239 | 0.4000 | 0.6000 | -0.1565 | 0.9715 | 0.9715 |

## Readiness & Degraded/Fallback Status

| planner | planner group | execution mode | execution detail | planner cmd | benchmark cmd | projection policy | readiness status | tier | preflight | learned contract | run status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goal | core | native | unspecified | unicycle_vw | unknown | unknown | native | baseline-ready | ok | not_applicable | ok |
| hybrid_rule_v3_fast_progress | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| hybrid_rule_v3_fast_progress_static_escape | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| hybrid_rule_v3_fast_progress_static_escape_continuous | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| orca | core | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | baseline-ready | ok | not_applicable | ok |
| ppo | experimental | native | unspecified | unicycle_vw | unknown | unknown | native | experimental | ok | pass | ok |
| prediction_planner | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| sacadrl | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| scenario_adaptive_hybrid_orca_v1 | experimental | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | experimental | ok | not_applicable | ok |
| scenario_adaptive_hybrid_orca_v2_collision_guard | experimental | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | experimental | ok | not_applicable | ok |
| social_force | core | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | baseline-ready | ok | not_applicable | ok |
| socnav_sampling | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |

- No fallback/degraded planners detected.

## SocNav Strict-vs-Fallback Disclosure

| planner | algo | planner group | prereq policy | preflight status | readiness status |
|---|---|---|---|---|---|
| goal | goal | core | fail-fast | ok | native |
| hybrid_rule_v3_fast_progress | hybrid_rule_local_planner | experimental | fail-fast | ok | adapter |
| hybrid_rule_v3_fast_progress_static_escape | hybrid_rule_local_planner | experimental | fail-fast | ok | adapter |
| hybrid_rule_v3_fast_progress_static_escape_continuous | hybrid_rule_local_planner | experimental | fail-fast | ok | adapter |
| orca | orca | core | fallback | ok | adapter |
| ppo | ppo | experimental | fail-fast | ok | native |
| prediction_planner | prediction_planner | experimental | fail-fast | ok | adapter |
| sacadrl | sacadrl | experimental | fallback | ok | adapter |
| scenario_adaptive_hybrid_orca_v1 | hybrid_rule_local_planner | experimental | fail-fast | ok | adapter |
| scenario_adaptive_hybrid_orca_v2_collision_guard | hybrid_rule_local_planner | experimental | fail-fast | ok | adapter |
| social_force | social_force | core | fail-fast | ok | adapter |
| socnav_sampling | socnav_sampling | experimental | fail-fast | ok | adapter |

- No within-campaign strict-vs-fallback pair available for direct comparison.

## Scenario Diagnostics

- Per-scenario breakdown: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/scenario_breakdown.csv`
- Per-family breakdown: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/scenario_family_breakdown.csv`

## Kinematics Parity

- Planner x kinematics parity table: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/kinematics_parity_table.csv`
- Skipped planner/kinematics combinations: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/kinematics_skipped_combinations.csv`

## AMV Coverage Contract

- Coverage JSON: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/amv_coverage_summary.json`
- Coverage Markdown: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/amv_coverage_summary.md`
- Coverage status: `warn` (enforcement: `warn`)

## Alyassi Comparability

- Comparability JSON: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/comparability_matrix.json`
- Comparability Markdown: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/comparability_matrix.md`
- Mapping version: `alyassi-comparability-v1`

## SNQI Contract

- Contract status: `fail`
- Rank alignment (Spearman): `-0.20665530815981215`
- Outcome separation: `0.2662956780720449`
- Positioning recommendation: `strengthen_as_operational_multi_objective_aggregation`
- Weights version: `snqi_weights_camera_ready_v3`
- Baseline version: `snqi_baseline_camera_ready_v3`
- Diagnostics JSON: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/snqi_diagnostics.json`
- Diagnostics Markdown: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/snqi_diagnostics.md`
- Sensitivity CSV: `output/benchmarks/camera_ready/issue1454-s10-h500-candidates/reports/snqi_sensitivity.csv`

## Campaign Warnings

- No campaign-level warnings.

## Failed Planners And Likely Reasons

- No failed/partial/not-available planners.
