# CARLA Replay Diagnostics

- Schema: `carla-replay-diagnostics.v1`
- Status: `degraded`
- Boundary: Diagnostics classify comparability surfaces only; they are not simulator-equivalence or benchmark-transfer evidence by themselves.

## Capability Matrix

| Axis | Status | Reason |
| --- | --- | --- |
| `summary_schema_version` | `available` |  |
| `replay_status` | `available` |  |
| `carla_map` | `not_available` | required CARLA summary field missing |
| `actor_summary` | `not_available` | required CARLA summary field missing |
| `replay_status` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |
| `static_geometry_support` | `not_available` | static-geometry replay support metadata is absent |
| `map_coordinate_frame` | `not_available` | map or coordinate-frame metadata is missing |
| `timing_step_sync` | `not_available` | trajectory step synchronization metadata is missing |
| `robot_pose_terminal_event` | `not_available` | success/collision terminal fields are not present in both inputs |
| `pedestrian_replay` | `not_available` | actor summary does not expose pedestrian replay count |

## Metric Fields

| Metric | Status | Reason |
| --- | --- | --- |
| `success` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |
| `collision` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |
| `ttc_min_s` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |
| `min_distance_m` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |
| `comfort` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |
| `jerk` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |
| `curvature` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |
| `intervention_rate` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |
| `snqi` | `degraded` | CARLA replay mode/status is not native/comparable: oracle-replay-adapted |

## Unsupported Semantics

| Semantic | Status | Reason |
| --- | --- | --- |
| `sensor_perception_replay` | `unsupported` | CARLA replay diagnostics do not compare sensor or perception pipelines |
| `broad_simulator_equivalence` | `unsupported` | live replay diagnostics are not simulator-equivalence evidence |
