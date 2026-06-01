# Scenario Perturbation Trace Response

## Boundary

diagnostic local trace inspection only; not benchmark-strength or paper-facing evidence

## Scope

- Source scenario: `francis2023_leave_group`
- Perturbed family: `single_pedestrian_speed_offset`
- Planners: `orca`
- Pair rows: `3`
- Pair statuses: `{"completed": 3}`

## Mean Closest-Approach Deltas

- `center_distance_m`: `0.034504`
- `clearance_m`: `0.034504`
- `goal_distance_m`: `-0.930346`
- `progress_m`: `0.930346`
- `time_s`: `-0.433333`

Positive clearance/center-distance deltas mean the perturbed run was farther from the closest pedestrian at its closest approach.

## By Planner

| Planner | Pairs | Center Distance Delta | Progress Delta | Time Delta |
|---|---:|---:|---:|---:|
| `orca` | 3 | `0.034504` | `0.930346` | `-0.433333` |
