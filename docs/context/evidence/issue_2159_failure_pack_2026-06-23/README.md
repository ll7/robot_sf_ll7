# Research v1 Failure-Case Trace Review Pack

Generated: 2026-06-23T00:00:00+00:00
Source issues: #2159, #2269
Claim boundary: diagnostic trace-review evidence only; not benchmark or paper evidence

## Selected Cases

- **Head-on Corridor Route Offset Response** (`head_on_corridor_route_offset_response`)
  - Scenario: `classic_head_on_corridor_low`
  - Planners: goal, orca, scenario_adaptive_hybrid_orca_v2_collision_guard
  - Priority: 3
  - Claim boundary: diagnostic local trace inspection only

- **Leave-Group Speed Outcome Flip** (`leave_group_speed_outcome_flip`)
  - Scenario: `francis2023_leave_group`
  - Planners: orca
  - Priority: 4
  - Claim boundary: diagnostic local trace inspection only

- **Intersection-Wait Speed p050 Phase Response** (`intersection_wait_speed_p050_phase_response`)
  - Scenario: `francis2023_intersection_wait`
  - Planners: goal, orca, scenario_adaptive_hybrid_orca_v2_collision_guard
  - Priority: 5
  - Claim boundary: diagnostic local trace inspection only

## Non-Goals

- No broad browser/UI viewer work.
- No benchmark-strength evidence claims.
- No paper-facing claims from qualitative trace inspection alone.
- AMV-specific cases are blocked pending renderable trace export (see #2269).