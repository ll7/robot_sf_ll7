# CARLA Replay Diagnostics

- Schema: `carla-replay-diagnostics.v1`
- Status: `available`
- Boundary: Diagnostics classify comparability surfaces only; they are not simulator-equivalence or benchmark-transfer evidence by themselves.

## Capability Matrix

| Axis | Status | Reason |
| --- | --- | --- |
| `summary_schema_version` | `available` |  |
| `replay_status` | `available` |  |
| `carla_map` | `not_available` | required CARLA summary field missing |
| `actor_summary` | `not_available` | required CARLA summary field missing |
| `replay_status` | `available` |  |
| `static_geometry_support` | `not_available` | static-geometry replay support metadata is absent |
| `map_coordinate_frame` | `not_available` | map or coordinate-frame metadata is missing |
| `timing_step_sync` | `not_available` | trajectory step synchronization metadata is missing |
| `robot_pose_terminal_event` | `not_available` | success/collision terminal fields are not present in both inputs |
| `pedestrian_replay` | `not_available` | actor summary does not expose pedestrian replay count |

## Metric Fields

| Metric | Status | Reason |
| --- | --- | --- |
| `success` | `not_available` | missing CARLA metric |
| `collision` | `not_available` | missing Robot-SF metric |
| `ttc_min_s` | `not_available` | missing Robot-SF metric |
| `min_distance_m` | `not_available` | missing Robot-SF metric |
| `comfort` | `not_available` | missing Robot-SF metric |
| `jerk` | `not_available` | missing Robot-SF metric |
| `curvature` | `not_available` | missing Robot-SF metric |
| `intervention_rate` | `not_available` | missing Robot-SF metric |
| `snqi` | `not_available` | missing Robot-SF metric |

## Unsupported Semantics

| Semantic | Status | Reason |
| --- | --- | --- |
| `sensor_perception_replay` | `unsupported` | CARLA replay diagnostics do not compare sensor or perception pipelines |
| `broad_simulator_equivalence` | `unsupported` | live replay diagnostics are not simulator-equivalence evidence |
