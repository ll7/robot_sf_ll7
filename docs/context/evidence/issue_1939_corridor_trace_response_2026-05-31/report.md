# Scenario Perturbation Trace Response

## Boundary

diagnostic local trace inspection only; not benchmark-strength or paper-facing evidence

## Scope

- Source scenario: `classic_head_on_corridor_low`
- Perturbed family: `pedestrian_route_offset`
- Planners: `goal`, `orca`, `scenario_adaptive_hybrid_orca_v2_collision_guard`
- Pair rows: `12`
- Pair statuses: `{"completed": 12}`

## Mean Closest-Approach Deltas

- `center_distance_m`: `0.153489`
- `clearance_m`: `0.153489`
- `goal_distance_m`: `-0.506475`
- `progress_m`: `0.506475`
- `time_s`: `-0.25`

Positive clearance/center-distance deltas mean the perturbed run was farther from the closest pedestrian at its closest approach.

## By Planner

| Planner | Pairs | Center Distance Delta | Progress Delta | Time Delta |
|---|---:|---:|---:|---:|
| `goal` | 4 | `0.159909` | `-0.024578` | `0.025` |
| `orca` | 4 | `0.157236` | `-0.049732` | `0.025` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 4 | `0.143321` | `1.593735` | `-0.8` |
