# Camera-Ready Benchmark Campaign Report

- Campaign ID: `issue4830_safety_wrapper_factorial_v1`
- Name: `issue_4830_safety_wrapper_factorial_v1`
- Created (UTC): `2026-07-29T10:27:02.624054Z`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Scenario matrix hash: `c10df617a87c`
- Git commit: `8b5ce0b1ca5b7845ae05b0fdc07761079c75d380`
- Runtime sec: `800.6803789430123`
- Episodes/sec: `1.07908226893305`
- Campaign status: `benchmark_success`
- Campaign execution status: `completed`
- Evidence status: `valid`
- Aggregate integrity: `valid`
- Status reason: `all planner rows were benchmark-success`
- Benchmark success: `True`
- Successful rows: `6` / `6`
- Accepted unavailable/excluded rows: `0`
- Unexpected failed rows: `0`
- Row status summary: `{'successful_evidence_rows': 6, 'accepted_unavailable_rows': 0, 'unexpected_failed_rows': 0, 'fallback_or_degraded_rows': 0}`
- Interpretation profile: `diagnostic-safety-wrapper-factorial-v1-preregistered`
- Command: `python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/issue_4830_safety_wrapper_factorial_v1.yaml --output-root output/benchmarks/camera_ready --mode run --log-level INFO --campaign-id issue4830_safety_wrapper_factorial_v1 --skip-publication-bundle --arm-isolation subprocess`

## Planner Summary

| planner | algo | planner group | kinematics | status | started (UTC) | runtime (s) | episodes | eps/s | success | collisions | snqi | proj_rate | infeasible_rate |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| orca__wrapper_off | orca | core | differential_drive | ok | 2026-07-29T10:28:26.700879Z | 0.0000 | 144 | 18554305.1045 | 0.0556 | 0.1042 | -0.2465 | 0.7382 | 0.7382 |
| orca__wrapper_on | orca | core | differential_drive | ok | 2026-07-29T10:29:57.607648Z | 0.0000 | 144 | 18250973.1720 | 0.0417 | 0.0417 | -0.3949 | 0.6314 | 0.6314 |
| prediction_planner__wrapper_off | prediction_planner | experimental | differential_drive | ok | 2026-07-29T10:36:10.500540Z | 0.0000 | 144 | 19672153.4274 | 0.0625 | 0.2708 | -0.2110 | 0.0000 | 0.0000 |
| prediction_planner__wrapper_on | prediction_planner | experimental | differential_drive | ok | 2026-07-29T10:40:20.174397Z | 0.0000 | 144 | 18508981.1236 | 0.0208 | 0.1458 | -0.1981 | 0.0000 | 0.0000 |
| social_force__wrapper_off | social_force | core | differential_drive | ok | 2026-07-29T10:31:03.353714Z | 0.0000 | 144 | 17955053.6895 | 0.0000 | 0.3750 | -0.7361 | 0.2272 | 0.2272 |
| social_force__wrapper_on | social_force | core | differential_drive | ok | 2026-07-29T10:32:10.759097Z | 0.0000 | 144 | 18947369.2625 | 0.0000 | 0.2222 | -0.7858 | 0.3366 | 0.3366 |

## Aggregate Integrity

Final arm aggregates have exact logical coverage and compatible provenance.
Claim boundary: A derived clean slice is diagnostic-only unless it was predeclared and provenance-complete.

## Arm Rollup

| planner | kinematics | status | written | failed |
|---|---|---|---:|---:|
| orca__wrapper_off | differential_drive | ok | 144 | 0 |
| orca__wrapper_on | differential_drive | ok | 144 | 0 |
| social_force__wrapper_off | differential_drive | ok | 144 | 0 |
| social_force__wrapper_on | differential_drive | ok | 144 | 0 |
| prediction_planner__wrapper_off | differential_drive | ok | 144 | 0 |
| prediction_planner__wrapper_on | differential_drive | ok | 144 | 0 |

## Credibility Scorecard

NASA-STD-7009B-inspired campaign credibility metadata. Unscored factors are shown as `not_assessed`.

- Schema: `campaign_credibility_scorecard.v1`
- Overall status: `partial`
- Overall score: `1.5`
- Claim boundary: `Scorecard is reporting metadata for campaign credibility dimensions; it is not benchmark proof, paper evidence, or real-world validation.`

| factor | status | score | justification |
|---|---|---:|---|
| Verification | weak | 1 | Report was generated from structured campaign summary/table artifacts; test-suite evidence is not embedded in the campaign artifact. |
| Validation | not_assessed |  | No campaign artifact supplied enough evidence for this factor. |
| Input pedigree | partial | 2 | Campaign records git commit and scenario matrix hash; external input lineage remains limited to recorded artifacts. |
| Uncertainty characterization | partial | 2 | Campaign records 3 seed(s) and seed-variability artifacts when available; no claim beyond campaign-level uncertainty. |
| Results robustness | weak | 1 | Campaign reports 6/6 successful planner row(s); fallback/degraded rows remain caveats, not success evidence. |
| Use history | not_assessed |  | No campaign artifact supplied enough evidence for this factor. |

## Readiness & Degraded/Fallback Status

| planner | planner group | execution mode | execution detail | planner cmd | benchmark cmd | projection policy | readiness status | tier | preflight | learned contract | run status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| orca__wrapper_off | core | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | baseline-ready | ok | not_applicable | ok |
| orca__wrapper_on | core | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | baseline-ready | ok | not_applicable | ok |
| prediction_planner__wrapper_off | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| prediction_planner__wrapper_on | experimental | adapter | unspecified | unicycle_vw | unknown | unknown | adapter | experimental | ok | not_applicable | ok |
| social_force__wrapper_off | core | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | baseline-ready | ok | not_applicable | ok |
| social_force__wrapper_on | core | adapter | unspecified | unicycle_vw | unicycle_vw | heading_safe_velocity_to_unicycle_vw | adapter | baseline-ready | ok | not_applicable | ok |

- No fallback/degraded planners detected.

## SocNav Strict-vs-Fallback Disclosure

| planner | algo | planner group | prereq policy | preflight status | readiness status |
|---|---|---|---|---|---|
| orca__wrapper_off | orca | core | fail-fast | ok | adapter |
| orca__wrapper_on | orca | core | fail-fast | ok | adapter |
| prediction_planner__wrapper_off | prediction_planner | experimental | fail-fast | ok | adapter |
| prediction_planner__wrapper_on | prediction_planner | experimental | fail-fast | ok | adapter |
| social_force__wrapper_off | social_force | core | fail-fast | ok | adapter |
| social_force__wrapper_on | social_force | core | fail-fast | ok | adapter |

- No within-campaign strict-vs-fallback pair available for direct comparison.

## Scenario Diagnostics

- Per-scenario breakdown: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/scenario_breakdown.csv`
- Per-family breakdown: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/scenario_family_breakdown.csv`

## Kinematics Parity

- Planner x kinematics parity table: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/kinematics_parity_table.csv`
- Skipped planner/kinematics combinations: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/kinematics_skipped_combinations.csv`

## AMV Coverage Contract

- Coverage JSON: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/amv_coverage_summary.json`
- Coverage Markdown: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/amv_coverage_summary.md`
- Coverage status: `warn` (enforcement: `warn`)

## Alyassi Comparability

- Comparability JSON: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/comparability_matrix.json`
- Comparability Markdown: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/comparability_matrix.md`
- Mapping version: `alyassi-comparability-v1`

## SNQI Contract

- Contract status: `pass`
- Rank alignment (Spearman): `0.7714285714285715`
- Outcome separation: `0.23162092145025104`
- Positioning recommendation: `strengthen_as_operational_multi_objective_aggregation`
- Weights version: `snqi_weights_camera_ready_v3`
- Baseline version: `snqi_baseline_camera_ready_v3`
- Diagnostics JSON: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/snqi_diagnostics.json`
- Diagnostics Markdown: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/snqi_diagnostics.md`
- Sensitivity CSV: `output/benchmarks/camera_ready/issue4830_safety_wrapper_factorial_v1/reports/snqi_sensitivity.csv`

## Fairness Contract

- Ranking claim allowed: `True`
- Fair subset size: `3`
- Excluded planners: `0`
- Hard mismatches: `0`
- Soft mismatches (caveats): `3`

Fair comparison subset:
- `orca`
- `prediction_planner`
- `social_force`

Soft mismatches (caveats):
- **adapter**: orca vs prediction_planner — Adapter name differs: orca=ORCAPlannerAdapter vs prediction_planner=PredictionPlannerAdapter.
- **adapter**: orca vs social_force — Adapter name differs: orca=ORCAPlannerAdapter vs social_force=SocialForcePlannerAdapter.
- **adapter**: prediction_planner vs social_force — Adapter name differs: prediction_planner=PredictionPlannerAdapter vs social_force=SocialForcePlannerAdapter.

## Accepted Unavailable/Excluded Planners

- No accepted unavailable/excluded planners.

## Unexpected Failed/Partial Planners

- No unexpected failed/partial planners.

## Campaign Warnings

- No campaign-level warnings.
