# Issue #1462 S10 H500 Failure-Mode Evidence

Date: 2026-05-24

This bundle derives compact scenario, candidate-vs-core, and seed tables from the issue #1454
S10/h500 candidate evidence. It does not rerun the benchmark.

## Source Inputs

- Compact source: `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/`
- Raw archive: `artifact/issue1454-s10-h500-candidates-2026-05-23`
- Source campaign: `issue1454-s10-h500-candidates`
- Episode count: 5,760

## Headline Tables

- `scenario_difficulty_table.csv`: all S10/h500 scenarios ranked by aggregate success, collision,
  timeout/unfinished rate, candidate-vs-core delta, and taxonomy.
- `candidate_vs_core_matrix.csv`: per-scenario candidate group versus core runnable planner group.
- `seed_difficulty_table.csv`: seed-level aggregate success/collision/near-miss difficulty.
- `planner_scenario_seed_variability.csv`: planner-scenario cells sorted by seed success range.
- `summary.json`: compact machine-readable manifest and highlights.

## Highlights

- Hardest aggregate scenarios: `francis2023_narrow_doorway` (0.000), `classic_station_platform_medium` (0.042), `classic_cross_trap_high` (0.317), `classic_doorway_medium` (0.350), `classic_doorway_high` (0.383).
- Largest candidate-vs-core success gains: `classic_bottleneck_high` (0.866), `francis2023_robot_crowding` (0.800), `francis2023_narrow_hallway` (0.786), `classic_realworld_double_bottleneck_high` (0.700), `classic_bottleneck_medium` (0.694).
- Candidate weak spots: `francis2023_narrow_doorway` (0.000), `classic_station_platform_medium` (0.040), `classic_cross_trap_high` (0.560), `classic_doorway_high` (0.600), `classic_doorway_medium` (0.620).
- Highest seed-sensitive planner/scenario cells: `orca` on `classic_bottleneck_medium` (1.000), `prediction_planner` on `classic_doorway_high` (1.000), `prediction_planner` on `classic_doorway_low` (1.000), `prediction_planner` on `classic_doorway_medium` (1.000), `socnav_sampling` on `classic_group_crossing_medium` (1.000).
- Hardest seeds by aggregate success: `116` (success 0.530, collision 0.257), `111` (success 0.535, collision 0.312), `117` (success 0.545, collision 0.271).

## Interpretation Boundary

The taxonomy is aggregate evidence. It can distinguish broad difficulty, candidate-specific gains,
collision-heavy cells, and timeout/unfinished-heavy cells. It does not prove behavioral mechanisms
such as waiting, yielding, or hesitation without trace/video review.
