# Issue #1674 Topology-Hypothesis Diagnostics

Date: 2026-05-30

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1674>

Evidence summary:
[`evidence/issue_1674_topology_hypothesis_diagnostics_2026-05-30/summary.json`](evidence/issue_1674_topology_hypothesis_diagnostics_2026-05-30/summary.json)

Follow-up issue: <https://github.com/ll7/robot_sf_ll7/issues/1692>

## Scope

This note evaluates whether a small existing Robot SF local-planner slice points to missing
alternate topology/homotopy hypotheses, using only existing scenario definitions, route/clearance
signals, and planner step diagnostics. It treats `tud-amr/mpc_planner` / T-MPC++ as design
inspiration only; no external code, assets, or dependencies were imported.

## Scenario Slice

The slice deliberately includes one easy control, one known hard bottleneck, and one crossing/trap
case that had appeared in prior h500 failure notes:

* `classic_bottleneck_medium`, seed `111`: control row for a bottleneck that current local
  candidates can solve.
* `classic_realworld_double_bottleneck_high`, seed `111`: hard double-bottleneck row from the
  h500 failure notes.
* `classic_cross_trap_high`, seed `112`: crossing/trap row that previously failed early.

The candidates were:

* `hybrid_rule_v0_minimal`: DWA-style local reactive baseline.
* `hybrid_rule_v3_waypoint2_route_lookahead8`: route-aware local candidate, used only on the two
  bottleneck rows to test whether route guidance already acts as a topology hypothesis.

## Commands

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v0_minimal \
  --stage full_matrix \
  --scenario-name classic_bottleneck_medium \
  --seed 111 \
  --horizon 240 \
  --output-dir output/issue_1674_topology_diagnostics/hybrid_v0_bottleneck_medium_111_h240

LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_waypoint2_route_lookahead8 \
  --stage full_matrix \
  --scenario-name classic_bottleneck_medium \
  --seed 111 \
  --horizon 240 \
  --output-dir output/issue_1674_topology_diagnostics/route_lookahead8_bottleneck_medium_111_h240

LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v0_minimal \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 240 \
  --output-dir output/issue_1674_topology_diagnostics/hybrid_v0_double_bottleneck_high_111_h240

LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_waypoint2_route_lookahead8 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 240 \
  --output-dir output/issue_1674_topology_diagnostics/route_lookahead8_double_bottleneck_high_111_h240

LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v0_minimal \
  --stage full_matrix \
  --scenario-name classic_cross_trap_high \
  --seed 112 \
  --horizon 120 \
  --output-dir output/issue_1674_topology_diagnostics/hybrid_v0_cross_trap_high_112_h120
```

## Metrics

| Scenario | Candidate | Outcome | Selected source counts | Rejections | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `classic_bottleneck_medium` seed `111` | `hybrid_rule_v0_minimal` | success at step `144`; no collision, near miss, force exceed, or comfort exposure | `dynamic_window=138`, `path_follow_0.5m=7` | `dynamic_collision=279`, `static_clearance=2010` | Control row; local command lattice is sufficient. |
| `classic_bottleneck_medium` seed `111` | `hybrid_rule_v3_waypoint2_route_lookahead8` | success at step `152`; no collision, near miss, force exceed, or comfort exposure | `dynamic_window=139`, `path_follow_0.5m=9`, `route_guide=5` | `dynamic_collision=345`, `static_clearance=1468` | Route guidance appears occasionally but is not needed for this easy row. |
| `classic_realworld_double_bottleneck_high` seed `111` | `hybrid_rule_v0_minimal` | no route completion by horizon `240`; last goal distance `4.209 m`; min observed goal distance `2.049 m` | `dynamic_window=237`, `path_follow_0.5m=3` | `dynamic_collision=881`, `static_clearance=525` | Hard row dominated by one local trajectory family; no explicit alternate route hypothesis is visible. |
| `classic_realworld_double_bottleneck_high` seed `111` | `hybrid_rule_v3_waypoint2_route_lookahead8` | no route completion by horizon `240`; last goal distance `4.563 m`; min observed goal distance `2.019 m` | `dynamic_window=236`, `path_follow_0.5m=2`, `route_guide=2` | `dynamic_collision=973`, `static_clearance=293` | Existing route guidance does not create a robust alternate-hypothesis diagnostic; it is selected only `2/240` steps. |
| `classic_cross_trap_high` seed `112` | `hybrid_rule_v0_minimal` | pedestrian collision at step `0`; comfort exposure `0.5833`; fallback emergency stop | `all_candidates_rejected=1` | `dynamic_collision=67` | Initial dynamic infeasibility/collision, not strong evidence for topology ambiguity. |

## Interpretation

The useful topology signal is the hard double-bottleneck row. Both tested candidates remained
within local command families for nearly all steps. The route-aware variant did expose a
`route_guide` source, but only on `2` of `240` decisions and without route completion by the
bounded horizon. That means the existing traces can show local-source dominance and clearance
rejections, but they do not yet expose multiple named route/corridor hypotheses that a reviewer can
compare.

The crossing/trap row should not drive topology work. It fails at step `0` because every candidate
is rejected for dynamic collision and the emergency stop still terminates with a pedestrian
collision. That is a scenario initial-condition or dynamic-feasibility issue, not a homotopy-choice
diagnostic.

## Follow-Up Decision

A follow-up implementation issue is justified, but it should be a diagnostic issue rather than a
planner issue. Issue #1692 tracks the next step: add an explicit topology-hypothesis diagnostic
that:

* extracts at least two route or corridor alternatives around static bottleneck obstacles,
* reports per-hypothesis static clearance, dynamic clearance, route progress, and selected local
  command source,
* runs first on `classic_realworld_double_bottleneck_high` seed `111`,
* and fails closed as diagnostic-only if fewer than two distinct hypotheses are available.

Do not add a topology-aware planner until this diagnostic proves that alternate hypotheses are
available and that the current local planner is choosing among them incorrectly.

## Validation

This note is evidence-only. Validation for the branch should include:

```bash
git diff --check origin/main...HEAD
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```
