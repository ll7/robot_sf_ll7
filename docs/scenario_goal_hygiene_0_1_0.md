# Goal geometry for 0.1.0 authoring

Diagnostic successor matrix:
`configs/scenarios/classic_interactions_francis2023_goal_hygiene_0_1_0_v1.yaml`.
It inherits all 48 scenarios from the unchanged 0.0.8 matrix, including their
explicit `footprint_clearance_v1` robot-goal sampling contract. Fresh main
already selects that policy for every release row; changing the global legacy
default is unnecessary and would change historical configuration semantics.
The sampler rejects footprint/wall or footprint/bounds contact and explicitly
fails after 400 candidates when no valid target is found.

The three `classic_bottleneck` density variants use a new map copied from
`issue_9762_classic_bottleneck_goal_zone_entry_v2.svg`. Only pedestrian goal
zone 0 changes: bounds become (18, 27)–(22, 31), including the existing route
endpoint (20, 28). The old parsed rectangle (17.801584, 38.135078)–
(21.801584, 42.135078) extends 8.540312 m² beyond the 40 m map and intersects
the bottom wall. The successor rectangle is wholly in bounds and clears every
obstacle. Robot geometry and pedestrian route waypoints are identical.

This matrix is an independent geometry overlay, not a benchmark promotion.
Future authoring must compose it with the speed/path successor from #10175;
that PR's files are deliberately untouched. The density/horizon/seed contract
is inherited; development diagnostics use seeds 1001–1030 only. Changed maps
must not be mixed with historical release rows in planner comparisons.
Released scenario bytes and frozen artifacts remain historical inputs.

The regression uses the real scenario loader and SVG parser. To reproduce the
pre-fix failure, set `SCENARIO_MATRIX` to the released matrix and run
`uv run pytest tests/validation/test_bottleneck_goal_hygiene.py -q`.
The successor passes the same containment, obstacle-overlap and route-endpoint
assertions. Existing sampler tests exercise explicit failure for infeasible
goals; the new test needs no simulator or production test seam.
