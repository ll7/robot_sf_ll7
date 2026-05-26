# Issue #1462 S10 H500 Failure Modes

Date: 2026-05-24

## Goal

Analyze the issue #1454 S10 scenario-horizon h500 candidate campaign at scenario and seed level
without rerunning the 9.6 hour benchmark. The source campaign is
`issue1454-s10-h500-candidates`, recorded in
`docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/` with the raw archive preserved
at the GitHub artifact release
https://github.com/ll7/robot_sf_ll7/releases/tag/artifact/issue1454-s10-h500-candidates-2026-05-23.

This note is a follow-up to
[`issue_1454_s10_h500_candidate_comparison.md`](issue_1454_s10_h500_candidate_comparison.md) and
keeps SNQI diagnostic-only because the campaign SNQI contract status is `fail` (rank alignment:
-0.207, outcome separation: 0.266, dominant component: success_reward).

## Method

The analysis is reproduced by:

```bash
python3 scripts/tools/analyze_issue_1462_h500_failure_modes.py \
  --raw-campaign-dir output/issue_1462_raw/extracted/issue1454-s10-h500-candidates
```

Before running that command locally, the raw archive was downloaded from the release and verified
against SHA256 `44ec1d4eb89d450eb204398a3807185ce9bdd4aae0eeb5e55af0704fd4a8b0fc`. The raw archive
was used only to count termination/status outcomes; committed evidence remains compact.

Derived evidence is in
[`evidence/issue_1462_s10_h500_failure_modes_2026-05-24/README.md`](evidence/issue_1462_s10_h500_failure_modes_2026-05-24/README.md):

- `scenario_difficulty_table.csv`
- `candidate_vs_core_matrix.csv`
- `seed_difficulty_table.csv`
- `planner_scenario_seed_variability.csv`
- `summary.json`

The comparison group boundary is important: `core` means the seven functioning issue #1454 Stage A
rows (`goal`, `social_force`, `orca`, `ppo`, `prediction_planner`, `socnav_sampling`, `sacadrl`),
while `candidate` means the five local h500 policy-search candidate rows.

## Findings

The hardest aggregate scenarios are stable and match the preliminary issue framing:
`francis2023_narrow_doorway` has `0.000` mean success, `classic_station_platform_medium` has
`0.042`, `classic_cross_trap_high` has `0.317`, `classic_doorway_medium` has `0.350`, and
`classic_doorway_high` has `0.383`.

The largest candidate-vs-core success gains are concentrated in bottleneck/crowding/hallway cases:
`classic_bottleneck_high` gains `+0.866`, `francis2023_robot_crowding` gains `+0.800`,
`francis2023_narrow_hallway` gains `+0.786`, `classic_realworld_double_bottleneck_high` gains
`+0.700`, and `classic_bottleneck_medium` gains `+0.694`.

Candidate weak spots remain narrow-doorway and station-platform geometry. The candidate mean is
`0.000` on `francis2023_narrow_doorway`, `0.040` on `classic_station_platform_medium`, `0.560` on
`classic_cross_trap_high`, and `0.600` on `classic_doorway_high`. These should be treated as
scenario-specific failure surfaces rather than a global candidate-family collapse.

The seed-level table shows modest global seed ordering but strong planner-scenario cell effects.
The hardest aggregate seeds are `116`, `111`, and `117`; however, the highest variability rows are
specific cells such as ORCA on `classic_bottleneck_medium`, prediction planner on the doorway
variants, `socnav_sampling` on `classic_group_crossing_medium`, and PPO/prediction planner on
`classic_realworld_double_bottleneck_high`.

## Time-to-Goal and Comfort Exposure

The scenario-level tables include `time_to_goal_norm_mean_all` and `comfort_exposure_mean_all`
because these metrics are available from the source `scenario_breakdown.csv`. The candidate-vs-core
matrix adds core/candidate deltas for both metrics, so reviewers can inspect comfort-exposure
tradeoffs alongside success deltas.

Quoting a few representative cells: on `classic_bottleneck_high` (candidate success gain +0.866),
the candidate TTG norm is 0.738 versus core 0.968 (delta -0.230); comfort exposure is negligible
for both groups (cand 0.000, core 0.000). The large success gain comes with a meaningful TTG
improvement.

On `francis2023_robot_crowding` (candidate success gain +0.800), candidate TTG norm is 0.629
versus core 0.858 (delta -0.229) and comfort exposure is 0.082 versus 0.228 (delta -0.146).
Both TTG and comfort-exposure improve with the candidate, likely because fewer collisions
translate into fewer force-exceed events.

On `classic_station_platform_medium` (consistently unsolved, candidate mean 0.040), candidate TTG
norm is 0.999 versus core 0.991 (delta +0.009). Both groups stay near the TTG ceiling
despite different collision rates, consistent with early termination dominating any speed
difference.

The seed-level tables add `time_to_goal_norm_mean` at the seed and planner-scenario cell level
from the `time_to_goal_norm_per_seed_mean` column, so reviewers can inspect whether difficult
seeds also show systematic TTG shifts.

No explicit comfort-exposure columns are available at the seed level because
`seed_variability_by_scenario.csv` does not include comfort-exposure per-seed breakdowns.

## Easiest Scenarios

The most broadly solvable S10/h500 scenarios (highest aggregate success mean across all 12
planners) are: `francis2023_pedestrian_overtaking` (0.758), `francis2023_entering_elevator`
(0.758), `francis2023_entering_room` (0.725), `francis2023_exiting_room` (0.725), and
`francis2023_intersection_proceed` (0.708). These are useful as high-probability anchor points
and for isolating the hardest seeds/planners in otherwise solvable geometry.

## Relation to Issue #1045 Solvability Mechanisms

The scenario-level failure-mode taxonomy here extends the issue #1045 solvability-mechanism
classification ([`issue_1045_h500_solvability_mechanisms.md`](issue_1045_h500_solvability_mechanisms.md))
to the full S10 candidate surface. The #1045 categories (clean budget relief, exposure-enabled
completion, partial timeout relief, safety-regressed completion, and late clean completion) were
built on ORCA-only aggregate trace evidence on the reduced continuous-h500 matrix. The
scenario-level taxonomy here is coarser (scenario family and aggregate metrics only) but covers
all 12 planners and 48 scenarios.

The hardest aggregate scenarios (`francis2023_narrow_doorway`, `classic_station_platform_medium`)
remain consistently unsolved for both core and candidate groups, which is consistent with the
Issue #1045 observation that narrow-doorway and station-platform geometry is difficult independent of
horizon length. The candidate-specific improvement cases (e.g., `classic_bottleneck_high`,
`francis2023_narrow_hallway`) are consistent with the Issue #1045 clean budget relief and
exposure-enabled completion categories for those scenario families, but the aggregate
scenario-level data cannot confirm which individual Issue #1045 mechanism labels apply to which
planner-scenario cells without per-step trace or video evidence.

Causal mechanism claims (waiting, yielding, delayed continuous progress, recovery after blockage,
or risk-taking through denser interaction) still require the trace/video review workflow described
in #1045 and #1049.

## SNQI Diagnostic Fields

SNQI (`snqi_mean` in scenario tables, `snqi_mean` and `snqi_std` in seed tables) is included for
diagnostic inspection only. The source campaign SNQI contract status is `fail` (rank alignment:
-0.207, outcome separation: 0.266, dominant component: success_reward), so SNQI values should not
be treated as decisive quality signals. The prior SNQI contract decision for h500 is documented in
[`issue_1038_h500_snqi_contract.md`](issue_1038_h500_snqi_contract.md).

## Interpretation Boundary

The taxonomy can separate consistently unsolved scenarios, candidate-specific improvements,
collision-heavy rows, timeout/unfinished-heavy rows, and seed-sensitive cells. It cannot prove
behavioral causes such as waiting, yielding, hesitation, or intentional risk-taking without trace or
video review. Near-miss-heavy candidate wins such as `francis2023_robot_crowding` should therefore
be described as aggregate safety/exposure tradeoffs, not as a known behavioral mechanism.

## Follow-Up

Useful next work would be a small trace/video review for the hardest candidate weak spots and the
largest near-miss-heavy wins. Prioritize `francis2023_narrow_doorway`,
`classic_station_platform_medium`, `francis2023_robot_crowding`, and
`francis2023_narrow_hallway` if causal mechanism claims are needed.
