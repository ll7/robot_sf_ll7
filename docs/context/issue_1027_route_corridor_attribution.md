# Issue #1027 Route-Corridor Attribution Diagnostics

Date: 2026-05-06

Related issues:

- #1027: <https://github.com/ll7/robot_sf_ll7/issues/1027>
- #1022: <https://github.com/ll7/robot_sf_ll7/issues/1022>
- #884: <https://github.com/ll7/robot_sf_ll7/issues/884>

## Goal

Add diagnostic-only route-corridor attribution for the #884 classic-merging failures. This note
records the proof that the new fields appear in step traces and summarizes what they show. It does
not claim a benchmark improvement and does not change planner behavior.

## Diagnostic Surface

`GridRoutePlannerAdapter.route_geometry(observation)` now exposes a JSON-ready route snapshot when
structured occupancy-grid routing is available:

- `route_start_world`,
- `route_goal_world`,
- `route_waypoint_world`,
- `route_waypoint_index`,
- `route_path_cell_count`,
- `route_remaining_distance`,
- `route_distance_to_waypoint`,
- `route_corner_distance`,
- `route_tangent_heading`,
- `route_heading_error`,
- `corridor_center_clearance`,
- `corridor_width_estimate`,
- `robot_lateral_offset_to_corridor`.

`HybridRuleLocalPlannerAdapter.last_decision()` now includes a nested `route_corridor` payload when
`route_guide_enabled` is active and route geometry is available. The payload also contains
`route_arc_progress_windows` computed from route remaining distance over 1 s, 3 s, and 5 s windows.

Missing route geometry is represented as `route_corridor: null`; it does not change selected
commands or candidate scoring.

## Validation Commands

Focused code checks:

```bash
rtk uv run ruff check robot_sf/planner/grid_route.py \
  robot_sf/planner/hybrid_rule_local_planner.py \
  tests/planner/test_grid_route.py \
  tests/planner/test_hybrid_rule_local_planner.py

rtk uv run pytest tests/planner/test_grid_route.py \
  tests/planner/test_hybrid_rule_local_planner.py -q
```

Result:

- Ruff passed.
- Targeted tests passed: `37 passed`.

Five regenerated #884 traces used this command shape:

```bash
rtk env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name <scenario> \
  --seed <seed> \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_1027_route_corridor_diagnostics/<scenario>_<seed>_h500
```

The command was run for:

- `classic_merging_low` seeds `111`, `113`,
- `classic_merging_medium` seeds `111`, `112`, `113`.

## Five-Seed Outcome

| Scenario | Seed | Outcome | Route diagnostic steps | Last route diagnostic step | Final route geometry |
|---|---:|---|---:|---:|---|
| `classic_merging_low` | 111 | timeout, no obstacle collision | `200` | `202` | unavailable at terminal stall |
| `classic_merging_low` | 113 | route-complete success at step `330` | `280` | `330` | available |
| `classic_merging_medium` | 111 | timeout, no obstacle collision | `196` | `200` | unavailable at terminal stall |
| `classic_merging_medium` | 112 | timeout, no obstacle collision | `252` | `253` | unavailable at terminal stall |
| `classic_merging_medium` | 113 | timeout, no obstacle collision | `479` | `499` | available |

Behavior matched the #1022 current-main reproduction: low seed `113` still succeeds, and the other
four named seeds still time out without obstacle collisions. The diagnostic change is additive.

## Route-Corridor Findings

Last available route geometry for the timeout seeds:

| Scenario | Seed | Remaining route | Corner distance | Heading error | Width estimate | Lateral offset | Route arc progress |
|---|---:|---:|---:|---:|---:|---:|---|
| `classic_merging_low` | 111 | `0.8828` | `0.6000` | `0.7854` | `0.8000` | `0.0728` | `1s=2.3657`, `3s=5.9314`, `5s=7.6142` |
| `classic_merging_medium` | 111 | `1.0000` | n/a | `0.0000` | `0.8000` | `0.1000` | `1s=0.1657`, `3s=5.2142`, `5s=6.3314` |
| `classic_merging_medium` | 112 | `0.8828` | `0.6000` | `0.7854` | `0.8000` | `0.0728` | `1s=6.8142`, `3s=3.5314`, `5s=4.6485` |
| `classic_merging_medium` | 113 | `11.0083` | `1.9799` | `-0.7854` | `2.0000` | `0.1414` | `1s=-0.1172`, `3s=-0.3515`, `5s=-0.5858` |

Implications:

- For low `111` and medium `112`, the last available route geometry shows a near route corner
  roughly `0.6 m` ahead, a `45 deg` heading error, and a narrow estimated corridor width of
  about `0.8 m`. The terminal stall later lacks route geometry, which means a behavior PR needs to
  handle route-geometry dropout explicitly.
- Medium `111` also loses geometry before the terminal stall, but its last route geometry shows
  a narrow corridor estimate and little 1 s route progress.
- Medium `113` keeps route geometry through the terminal stall and shows negative route-arc
  progress windows, so a future trigger can distinguish route regression from generic goal-distance
  stagnation.
- The successful low `113` trace keeps route geometry through route completion and has broad
  final clearance. Future changes must preserve this success.

## Follow-Up Boundary

Issue `#1027` satisfies the diagnostic prerequisite for `#1028`. A future `corridor_subgoal`
behavior PR should use these fields to fail closed when route geometry disappears, and should treat
negative route-arc progress as a stronger recovery trigger than ordinary goal-distance stagnation.

The next behavior PR must still prove:

- no obstacle-collision regressions on the five #884 target seeds,
- preservation of `classic_merging_low` seed `113` success,
- no h500 `nominal_sanity` or `stress_slice` obstacle-collision regression,
- no fallback/degraded success classification.
