# Issue 7883 — biased route-condition generator

Status: implemented (diagnostic-only). PR: the deterministic
neutral/left/right route-condition generator for parent issue
[#7883](https://github.com/ll7/robot_sf_ll7/issues/7883) (passing-side bias
and route predictability research), built on the merged
`route_choice_observability.v1` contract from #7890.

## Method card

- Owner module: `robot_sf/nav/biased_route_generator.py` (pure, typed,
  deterministic; no I/O, no simulator, no planner-default changes).
- Conditions: `neutral`, `left`, `right` (`ROUTE_CONDITIONS`).
- Generation: deterministic 8-connected A* over a boolean ``(row, col)``
  occupancy grid. The `neutral` condition plans with both strict axis sides
  masked; `left`/`right` mask everything except the named side (plus the
  shared endpoints). Ties break by
  `(f, heuristic, row, col, axis-distance)` so identical inputs always
  produce identical paths and equal-cost candidates hug the axis.
- Verification: every variant is checked with the #7890 contract —
  `classify_route_side` must return the requested condition's side, and
  `homotopy_identity` must be stable under a deterministic cell-repetition
  replan. Output: `route_condition_report.v1`.
- Canonical fixtures: `corridor_map()` (three-corridor barrier, cols 6-8) and
  `doorway_map()` (vertical wall at col 7 with openings at rows 1, 4, 7).

## Fixture receipts (synthetic, deterministic)

- Corridor map identities at start `(4, 1)` / goal `(4, 13)`:
  `neutral=4,6;4,7;4,8`, `left=1,6;1,7;1,8;2,5;2,9;3,4;3,10`,
  `right=5,4;5,10;6,5;6,9;7,6;7,7;7,8`.
- Doorway map identities: `neutral=4,7`, `left=1,7`, `right=7,7`.
- Failure fixtures: blocked endpoints, fully walled doorway, and non-boolean
  grids fail closed (`no_feasible_route` / `ValueError`).

## Claim boundary

Planner-route observability and deterministic synthetic route generation
only. No pedestrian-intent inference, no passing-side or predictability
claims about humans, no benchmark or campaign evidence, and no changes to
default navigation route selection. Campaign use of these conditions is
parent-issue work and is separately gated.
