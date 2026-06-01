# Scenario Perturbation Trace Response

## Boundary

diagnostic local trace inspection only; not benchmark-strength or paper-facing evidence

## Scope

- Source scenario: `francis2023_intersection_wait`
- Perturbed family: `single_pedestrian_start_delay_offset`
- Planners: `goal`, `orca`, `scenario_adaptive_hybrid_orca_v2_collision_guard`
- Pair rows: `9`
- Pair statuses: `{"completed": 9}`

## Mean Closest-Approach Deltas

- `center_distance_m`: `4.159358`
- `clearance_m`: `4.159358`
- `goal_distance_m`: `-1.95458`
- `progress_m`: `1.95458`
- `time_s`: `-0.977778`

Positive clearance/center-distance deltas mean the perturbed run was farther from the closest pedestrian at its closest approach.

## By Planner

| Planner | Pairs | Center Distance Delta | Progress Delta | Time Delta |
|---|---:|---:|---:|---:|
| `goal` | 3 | `4.861856` | `0.0` | `0.0` |
| `orca` | 3 | `3.714579` | `2.998506` | `-1.5` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 3 | `3.901639` | `2.865234` | `-1.433333` |
