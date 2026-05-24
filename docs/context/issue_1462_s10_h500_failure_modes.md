# Issue #1462 S10 H500 Failure Modes

Date: 2026-05-24

## Goal

Analyze the issue #1454 S10 scenario-horizon h500 candidate campaign at scenario and seed level
without rerunning the 9.6 hour benchmark. The source campaign is
`issue1454-s10-h500-candidates`, recorded in
`docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/` with the raw archive preserved
at the GitHub artifact release
`artifact/issue1454-s10-h500-candidates-2026-05-23`.

This note is a follow-up to
[`issue_1454_s10_h500_candidate_comparison.md`](issue_1454_s10_h500_candidate_comparison.md) and
keeps SNQI diagnostic-only because the campaign SNQI contract status is `fail`.

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

