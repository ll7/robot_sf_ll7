# Issue #9762: versioned goal-zone route margins

The 0.0.7 scenario matrix, registry, waiver file, and all historical `maps/svg_maps/`
files remain byte-identical. The #9348 doorway preregistration therefore keeps using its
2.0 m reference map with the pinned SHA-256
`7538ed173d462a5107afc1a1e43b5b2e6d2bc5c9604035cdec9a551e20a8b15e`.

For a future matrix, `classic_interactions_francis2023_goal_zone_entry_v3.yaml` extends
the current goal-zone matrix and routes 46 scenario rows through versioned map overrides.
It uses 31 #9762 successor SVGs. The station-platform successor starts from the already
merged #9725 map (`classic_station_platform_v2.svg`, SHA-256
`6a106e4e1a44ea927ffe1fa0089d865991d97c9ae85fe4a1608d5c8fd78f136c`) and changes only
the robot route endpoint from `(77, 22.5)` to `(78, 22.5)`; its path segment is checked
against obstacles and its endpoint has a 1.0 m zone-boundary margin. The override clears
`map_id` so the scenario loader resolves `map_file` rather than silently selecting the
historical registry map.

Review of every release-matrix route also found zero-margin endpoints in the real-world
double-bottleneck and urban-crossing scenarios. Their versioned successors extend only the
final waypoint from `(53, 15)` to `(54, 15)` and `(39, 25)` to `(40, 25)`, respectively.
The added segments are obstacle-clear and each endpoint has a 1.0 m goal-zone margin.

The route-consistency test loads both matrices through the scenario loader and checks
that every final route waypoint enters its declared goal zone with at least the configured
robot-radius margin. For the crossing family, it preserves the certified interaction
route as an unchanged prefix and adds a final segment to the goal; for station-platform,
it preserves the #9725 spawn-corrected map and extends only the robot route endpoint.
Both added segments are checked against obstacles and the configured radius. Other
successor files change only robot route data. These are input-consistency checks, not
campaign results.

The v3 matrix is a candidate input for 0.0.8 integration after review. It does not change
the 0.0.7 release config or establish any outcome, planner, or release claim.
