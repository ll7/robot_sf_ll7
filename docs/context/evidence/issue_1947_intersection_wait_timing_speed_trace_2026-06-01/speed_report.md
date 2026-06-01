# Scenario Perturbation Trace Response

## Boundary

diagnostic local trace inspection only; not benchmark-strength or paper-facing evidence

## Scope

- Source scenario: `francis2023_intersection_wait`
- Perturbed family: `single_pedestrian_speed_offset`
- Planners: `goal`, `orca`, `scenario_adaptive_hybrid_orca_v2_collision_guard`
- Pair rows: `9`
- Pair statuses: `{"completed": 9}`

## Mean Closest-Approach Deltas

- `center_distance_m`: `-2.002917`
- `clearance_m`: `-2.002917`
- `goal_distance_m`: `0.022215`
- `progress_m`: `-0.022215`
- `time_s`: `0.011111`

Positive clearance/center-distance deltas mean the perturbed run was farther from the closest pedestrian at its closest approach.

## By Planner

| Planner | Pairs | Center Distance Delta | Progress Delta | Time Delta |
|---|---:|---:|---:|---:|
| `goal` | 3 | `-2.298689` | `0.0` | `0.0` |
| `orca` | 3 | `-1.820168` | `0.0` | `0.0` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 3 | `-1.889893` | `-0.066644` | `0.033333` |
