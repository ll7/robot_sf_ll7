# Issue #1462 S10 H500 Failure-Mode Evidence

Date: 2026-05-24

This bundle derives compact scenario, candidate-vs-core, and seed tables from the issue #1454
S10/h500 candidate evidence. It does not rerun the benchmark.

## Source Inputs

- Compact source: `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/`
- Raw archive: https://github.com/ll7/robot_sf_ll7/releases/tag/artifact/issue1454-s10-h500-candidates-2026-05-23
- Source campaign: `issue1454-s10-h500-candidates`
- Episode count: 5,760

## Headline Tables

- `scenario_difficulty_table.csv`: all S10/h500 scenarios ranked by aggregate success, collision,
  timeout/unfinished rate, time-to-goal, comfort-exposure, diagnostic SNQI, candidate-vs-core delta,
  and taxonomy.
- `candidate_vs_core_matrix.csv`: per-scenario candidate group versus core runnable planner group,
  with time-to-goal and comfort-exposure deltas.
- `seed_difficulty_table.csv`: seed-level aggregate success/collision/near-miss difficulty with
  time-to-goal and diagnostic SNQI.
- `planner_scenario_seed_variability.csv`: planner-scenario cells sorted by seed success range, with
  time-to-goal and diagnostic SNQI.
- `summary.json`: compact machine-readable manifest and highlights.

## Highlights

- Hardest aggregate scenarios: `francis2023_narrow_doorway` (0.000), `classic_station_platform_medium` (0.042), `classic_cross_trap_high` (0.317), `classic_doorway_medium` (0.350), `classic_doorway_high` (0.383).
- Easiest aggregate scenarios: `francis2023_entering_elevator` (0.758), `francis2023_pedestrian_overtaking` (0.758), `francis2023_entering_room` (0.725), `francis2023_exiting_room` (0.725), `classic_bottleneck_low` (0.708).
- Largest candidate-vs-core success gains: `classic_bottleneck_high` (0.866), `francis2023_robot_crowding` (0.800), `francis2023_narrow_hallway` (0.786), `classic_realworld_double_bottleneck_high` (0.700), `classic_bottleneck_medium` (0.694).
- Candidate weak spots: `francis2023_narrow_doorway` (0.000), `classic_station_platform_medium` (0.040), `classic_cross_trap_high` (0.560), `classic_doorway_high` (0.600), `classic_doorway_medium` (0.620).
- Highest seed-sensitive planner/scenario cells: `orca` on `classic_bottleneck_medium` (1.000), `prediction_planner` on `classic_doorway_high` (1.000), `prediction_planner` on `classic_doorway_low` (1.000), `prediction_planner` on `classic_doorway_medium` (1.000), `socnav_sampling` on `classic_group_crossing_medium` (1.000).
- Hardest seeds by aggregate success: `116` (success 0.530, collision 0.257), `111` (success 0.535, collision 0.312), `117` (success 0.545, collision 0.271).

## Easiest Scenarios

The most broadly solvable S10/h500 scenarios (highest aggregate success mean across all 12
planners) are: `francis2023_entering_elevator` (0.758), `francis2023_pedestrian_overtaking` (0.758), `francis2023_entering_room` (0.725), `francis2023_exiting_room` (0.725), `classic_bottleneck_low` (0.708), `francis2023_intersection_no_gesture` (0.708), `francis2023_intersection_proceed` (0.708), `francis2023_intersection_wait` (0.708), `classic_group_crossing_medium` (0.692), `francis2023_crowd_navigation` (0.683).

These scenarios are useful as high-probability anchor points and for isolating the hardest
seeds/planners in otherwise solvable geometry.

## Relation to H500 Solvability Mechanisms

This taxonomy extends the issue #1045 solvability-mechanism classification
(`docs/context/issue_1045_h500_solvability_mechanisms.md`) to the full S10 candidate surface.
The #1045 categories (clean budget relief, exposure-enabled completion, partial timeout relief,
and safety-regressed completion) were built on ORCA-only trace evidence on the reduced
continuous-h500 matrix. The scenario-level failure-mode taxonomy here is coarser but covers
all 12 planners and 48 scenarios. It cannot confirm or reject individual #1045 mechanism labels
for specific planner-scenario cells without per-step trace or video evidence.

## SNQI Field

SNQI (`snqi_mean`) fields are included in the CSV tables and summary for diagnostic inspection
purposes only. The source campaign `snqi_contract_status` is `fail` (rank alignment:
-0.207, outcome separation: 0.266, dominant component: success_reward), so SNQI is **not**
treated as a decisive quality signal.

## Interpretation Boundary

The taxonomy is aggregate evidence. It can distinguish broad difficulty, candidate-specific gains,
collision-heavy cells, and timeout/unfinished-heavy cells. It does not prove behavioral mechanisms
such as waiting, yielding, or hesitation without trace/video review.
