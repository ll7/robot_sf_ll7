# Issue #9762: versioned goal-zone route margins

The 0.0.7 scenario matrix, registry, waiver file, and all historical `maps/svg_maps/`
files remain byte-identical. The #9348 doorway preregistration therefore keeps using its
2.0 m reference map with the pinned SHA-256
`7538ed173d462a5107afc1a1e43b5b2e6d2bc5c9604035cdec9a551e20a8b15e`.

For a future matrix, `classic_interactions_francis2023_goal_zone_entry_v3.yaml` extends
the current goal-zone matrix and routes 44 scenario rows through versioned map overrides.
It uses 28 #9762 successor SVGs and the already-versioned #9725 station-platform map.
The override clears `map_id` so the scenario loader resolves `map_file` rather than
silently selecting the historical registry map.

The route-consistency test loads both matrices through the scenario loader and checks
that each final route waypoint enters its declared goal zone with at least the configured
robot-radius margin. For the crossing family, it preserves the certified interaction
route as an unchanged prefix and adds a final segment to the goal; the test checks that
segment against obstacles and the configured radius. Other successor files change only
the robot route data. These are input-consistency checks, not campaign results.

The v3 matrix is a candidate input for 0.0.8 integration after review. It does not change
the 0.0.7 release config or establish any outcome, planner, or release claim.
