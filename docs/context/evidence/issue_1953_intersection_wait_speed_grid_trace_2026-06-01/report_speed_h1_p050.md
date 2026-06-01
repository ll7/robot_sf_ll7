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

- `center_distance_m`: `-3.862581`
- `clearance_m`: `-3.862581`
- `goal_distance_m`: `-0.288782`
- `progress_m`: `0.288782`
- `time_s`: `-0.144444`

Positive clearance/center-distance deltas mean the perturbed run was farther from the closest pedestrian at its closest approach.

## By Planner

| Planner | Pairs | Center Distance Delta | Progress Delta | Time Delta |
|---|---:|---:|---:|---:|
| `goal` | 3 | `-4.530414` | `0.0` | `0.0` |
| `orca` | 3 | `-3.464482` | `0.399855` | `-0.2` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 3 | `-3.592846` | `0.46649` | `-0.233333` |
