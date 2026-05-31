# Issue #1939 Corridor Trace-Response Evidence

This bundle contains compact diagnostic evidence for issue #1939, the trace-level follow-up to the
#1937 pedestrian-route-offset pilot under parent issue #1610.

- `closest_approach_trace_slices.json`: closest-approach slices for
  `classic_head_on_corridor_low` no-op versus `pedestrian_route_offset` pairs across `goal`,
  `orca`, and `scenario_adaptive_hybrid_orca_v2_collision_guard`.
- `report.md`: compact generated Markdown summary with aggregate and per-planner deltas.
- `SHA256SUMS`: checksums for the tracked evidence files.

Raw rerun outputs, generated scenario matrices, and route override files remain under ignored
`output/` paths and are not mirrored here.

Claim boundary: diagnostic local trace inspection only; not benchmark-strength or paper-facing
evidence.
