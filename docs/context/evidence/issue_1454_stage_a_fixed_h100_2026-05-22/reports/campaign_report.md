# Camera-Ready Benchmark Campaign Report

- Campaign ID: `issue1454-s10-fixed-h100`
- Name: `issue_1454_s10_fixed_h100_broader_baselines`
- Created (UTC): `2026-05-22T11:23:25.274801Z`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Scenario matrix hash: `d2e3bdbeafba`
- Git commit: `17b179007e6292dd0365c53cff719cccea1276a9`
- Runtime sec: `2722.0126118178014`
- Episodes/sec: `1.2343807612838873`
- Interpretation profile: `issue-1454-s10-fixed-h100-broader-robustness`
- Command: `/home/luttkule/git/robot_sf_ll7/.venv/bin/python3 /home/luttkule/git/robot_sf_ll7/scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/issue_1454_s10_fixed_h100_broader_baselines.yaml --campaign-id issue1454-s10-fixed-h100`

## Planner Summary

| planner | algo | planner group | kinematics | status | started (UTC) | runtime (s) | episodes | eps/s | success | collisions | snqi | proj_rate | infeasible_rate |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | core | differential_drive | ok | 2026-05-22T11:23:25.453999Z | 161.0327 | 480 | 2.9808 | 0.0063 | 0.2396 | -0.1838 | 0.0000 | 0.0000 |
| orca | orca | core | differential_drive | ok | 2026-05-22T11:29:13.300737Z | 247.8762 | 480 | 1.9365 | 0.1812 | 0.0604 | -0.2497 | 0.7673 | 0.7673 |
| ppo | ppo | experimental | differential_drive | ok | 2026-05-22T11:33:21.762749Z | 342.7629 | 480 | 1.4004 | 0.2250 | 0.1229 | -0.3280 | 0.0000 | 0.0000 |
| prediction_planner | prediction_planner | experimental | differential_drive | ok | 2026-05-22T11:39:05.130583Z | 1341.0938 | 480 | 0.3579 | 0.0625 | 0.2229 | -0.2076 | 0.0000 | 0.0000 |
| sacadrl | sacadrl | experimental | differential_drive | ok | 2026-05-22T12:04:15.916537Z | 270.6134 | 480 | 1.7737 | 0.0104 | 0.3792 | -0.3111 | 0.0000 | 0.0000 |
| social_force | social_force | core | differential_drive | ok | 2026-05-22T11:26:07.068575Z | 185.4653 | 480 | 2.5881 | 0.0000 | 0.2250 | -0.8704 | 0.1438 | 0.1438 |
| socnav_sampling | socnav_sampling | experimental | differential_drive | ok | 2026-05-22T12:01:26.820302Z | 168.5053 | 480 | 2.8486 | 0.1604 | 0.4792 | -0.1923 | 0.9701 | 0.9701 |
| socnav_bench | socnav_bench | experimental | differential_drive | failed | 2026-05-22T12:08:47.127427Z | 0.0108 | 0 | 0.0000 | nan | nan | nan | 0.0000 | 0.0000 |

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

- Per-scenario breakdown: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/scenario_breakdown.csv`
- Per-family breakdown: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/scenario_family_breakdown.csv`

## Kinematics Parity

- Planner x kinematics parity table: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/kinematics_parity_table.csv`
- Skipped planner/kinematics combinations: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/kinematics_skipped_combinations.csv`

## AMV Coverage Contract

- Coverage JSON: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/amv_coverage_summary.json`
- Coverage Markdown: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/amv_coverage_summary.md`
- Coverage status: `warn` (enforcement: `warn`)

## Alyassi Comparability

- Comparability JSON: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/comparability_matrix.json`
- Comparability Markdown: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/comparability_matrix.md`
- Mapping version: `alyassi-comparability-v1`

## SNQI Contract

- Contract status: `warn`
- Rank alignment (Spearman): `0.32142857142857145`
- Outcome separation: `0.20901141789641814`
- Positioning recommendation: `strengthen_as_operational_multi_objective_aggregation`
- Weights version: `snqi_weights_camera_ready_v3`
- Baseline version: `snqi_baseline_camera_ready_v3`
- Diagnostics JSON: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/snqi_diagnostics.json`
- Diagnostics Markdown: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/snqi_diagnostics.md`
- Sensitivity CSV: `output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/snqi_sensitivity.csv`

## Campaign Warnings

- Planner 'socnav_bench' failed for kinematics 'differential_drive': SocNav preflight failed for algorithm 'socnav_bench': SocNavBench control pipeline parameters failed to load. Ensure the SocNavBench data directories exist. See `docs/socnav_assets_setup.md` and run `uv run python scripts/tools/prepare_socnav_assets.py`.. Check missing dependencies/models or choose a different prereq policy.
- Planner failure recorded: planner='socnav_bench' kinematics='differential_drive' status='failed' most_likely_reason='RuntimeError("SocNav preflight failed for algorithm 'socnav_bench': SocNavBench control pipeline parameters failed to load. Ensure the SocNavBench data directories exist. See `docs/socnav_assets_setup.md` and run `uv run python scripts/tools/prepare_socnav_assets.py`.. Check missing dependencies/models or choose a different prereq policy.")'

## Failed Planners And Likely Reasons

| planner | status | most likely reason |
|---|---|---|
| socnav_bench | failed | RuntimeError("SocNav preflight failed for algorithm 'socnav_bench': SocNavBench control pipeline parameters failed to load. Ensure the SocNavBench data directories exist. See `docs/socnav_assets_setup.md` and run `uv run python scripts/tools/prepare_socnav_assets.py`.. Check missing dependencies/models or choose a different prereq policy.") |
