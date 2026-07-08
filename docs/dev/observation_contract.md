# Observation Contract

This document defines the observation schemas used by Robot SF environments. It is the
source of truth for shapes, keys, and normalization conventions expected by policies
and downstream tooling.

## Default Observation Mode (`ObservationMode.DEFAULT_GYM`)

**Producer:** `robot_sf/sensor/sensor_fusion.py`  
**Consumer:** `robot_sf/gym_env/robot_env.py`

### Keys and Shapes

| Key | Shape | Description |
| --- | --- | --- |
| `drive_state` | `(timesteps, 5)` | `[speed_x, speed_rot, target_distance, target_angle, next_target_angle]` |
| `rays` | `(timesteps, num_rays)` | LiDAR ranges stacked over time |

### Target Semantics

- Robot environments set `target_distance` and `target_angle` from the robot pose to the current
  route waypoint, and `next_target_angle` from that waypoint toward the following route waypoint
  when `SimulationSettings.use_next_goal` is enabled.
- Pedestrian environments set the ego-pedestrian target to the robot position. When
  `use_next_goal` is enabled, `next_target_angle` points from that robot target toward the robot's
  current route waypoint. This preserves the ego-pedestrian "track the robot" goal contract while
  exposing the route direction of the moving target.

### Stacking Rules

- `timesteps` equals `observation_stack.stack_steps` on the environment config.
- Stacking is chronological with newest samples appended and older samples shifted.

### Normalization

- `drive_state` and `rays` are normalized by dividing by the corresponding
  `orig_obs_space.high` values (per-dimension).
- Normalized values are expected to fall within `[-1, 1]` or `[0, 1]` depending
  on the underlying raw bounds.

## SocNav Structured Observation Mode (`ObservationMode.SOCNAV_STRUCT`)

**Producer:** `robot_sf/sensor/socnav_observation.py`  
**Consumer:** `robot_sf/gym_env/robot_env.py`

### Keys and Shapes (Structured)

| Key | Shape | Description |
| --- | --- | --- |
| `robot.position` | `(2,)` | Robot position `(x, y)` clipped to 50m |
| `robot.heading` | `(1,)` | Heading in radians, wrapped to `[-pi, pi]` |
| `robot.speed` | `(2,)` | Robot speed contract `(linear_speed, angular_speed)` |
| `robot.velocity_xy` | `(2,)` | Robot translational velocity `(vx, vy)` in world coordinates |
| `robot.angular_velocity` | `(1,)` | Robot yaw rate in radians per second |
| `robot.radius` | `(1,)` | Robot radius |
| `goal.current` | `(2,)` | Current goal position |
| `goal.next` | `(2,)` | Next goal position (or zeros if none) |
| `pedestrians.positions` | `(max_pedestrians, 2)` | Ped positions, sorted by distance |
| `pedestrians.velocities` | `(max_pedestrians, 2)` | Ped velocities in robot frame |
| `pedestrians.radius` | `(1,)` | Ped radius |
| `pedestrians.count` | `(1,)` | Count of visible pedestrians |
| `map.size` | `(2,)` | Map width/height capped to 50m |
| `sim.timestep` | `(1,)` | Simulation step duration in seconds |

`max_pedestrians` is derived from `SimulationSettings.max_total_pedestrians` or defaults to 64.

### Frame Semantics

- `robot.position`, `goal.current`, `goal.next`, and `robot.velocity_xy` are expressed in the
  global/world frame.
- `pedestrians.velocities` are rotated into the robot ego frame.
- `robot.heading` is always present, including for holonomic robots.
- In holonomic `vx_vy` mode, heading is *not* independently actuated. When the robot is moving,
  heading aligns with the current world-frame velocity direction. With zero translational velocity,
  the previous heading is retained.
- `robot.speed` is not a duplicate of `robot.velocity_xy`. It preserves the benchmark-wide
  `(linear_speed, angular_speed)` contract even when the robot is holonomic.

### Flattened Keys (SB3 Compatibility)

When using StableBaselines3, nested dicts are flattened with underscore separators
by `robot_sf/gym_env/robot_env.py`. Example: `robot.position` becomes `robot_position`.

## Occupancy Grid Augmentation

When `use_occupancy_grid=True` and `include_grid_in_observation=True`, the observation
dict includes the grid itself plus flattened metadata fields.

### Grid Keys

| Key | Shape | Description |
| --- | --- | --- |
| `occupancy_grid` | `(channels, height, width)` | Occupancy grid tensor |

### Metadata Keys

Each metadata key is prefixed with `occupancy_grid_meta_`:

- `occupancy_grid_meta_origin` - `(2,)`
- `occupancy_grid_meta_resolution` - `(1,)`
- `occupancy_grid_meta_size` - `(2,)`
- `occupancy_grid_meta_use_ego_frame` - `(1,)`
- `occupancy_grid_meta_center_on_robot` - `(1,)`
- `occupancy_grid_meta_channel_indices` - `(4,)`
- `occupancy_grid_meta_robot_pose` - `(3,)`

Sources: `robot_sf/nav/occupancy_grid.py`, `robot_sf/gym_env/robot_env.py`.

## Configuration Hooks

- Observation mode: `RobotSimulationConfig.observation_mode`
- Stacking: `RobotSimulationConfig.observation_stack.stack_steps`
- Grid: `RobotSimulationConfig.grid_config`, `use_occupancy_grid`, `include_grid_in_observation`

## Benchmark Observation Levels

Benchmark runs may also declare an observation level. The level records the
perception assumption separately from the raw environment observation mode:

| Level | Meaning |
| --- | --- |
| `oracle_full_state` | Privileged simulator state. |
| `tracked_agents_no_noise` | Perfect tracked-agent state without synthetic noise. |
| `tracked_agents_with_noise` | Tracked-agent state with benchmark observation-noise metadata. |
| `lidar_2d` | Range-sensor or lidar-style projection metadata. |
| `occluded_partial_state` | Partial-state contract with visibility/occlusion assumptions. |

The vocabulary lives in `robot_sf/benchmark/observation_levels.py` and is wired
through planner compatibility metadata. Unsupported planner/level combinations
fail before episodes are written. These levels are benchmark provenance labels;
they are not detector, camera, lidar, or sim-to-real certification claims.

## Checkpoint Observation Contract (Model Registry)

Learned-policy checkpoints that use dict-family observation modes (`dict`, `native_dict`,
`multi_input`) must declare their observation contract in the model registry. This prevents
running checkpoints with the wrong observation producer and ensures reproducibility.

### When Observation Contract is Required

A checkpoint requires `benchmark_promotion` metadata with observation contract fields when:
- The checkpoint uses `obs_mode: dict`, `native_dict`, or `multi_input`
- The algorithm's default observation mode is not `socnav_state`

For example, `guarded_ppo` uses `obs_mode: dict` with default `sensor_fusion_state`, so its
checkpoint must declare the observation level and mode used during training.

### Model Registry Metadata

Add a `benchmark_promotion` block to the model registry entry (`model/registry.yaml`):

```yaml
- model_id: ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200
  # ... other fields ...
  benchmark_promotion:
    claim_boundary: benchmark_candidate  # or benchmark_promoted, research_only
    benchmark_track: grid_socnav_v1
    track_schema_version: observation-track.v1
    observation_level: tracked_agents_no_noise
    observation_mode: dict
    allowed_observation_keys:
    - occupancy_grid
    - occupancy_grid_meta_origin
    - occupancy_grid_meta_resolution
    - occupancy_grid_meta_width
    - occupancy_grid_meta_height
    - robot_position
    - robot_heading
    - robot_velocity_xy
    - robot_speed
    - robot_angular_velocity
    - robot_radius
    - goal_current
    - goal_next
    - pedestrians_positions
    goal_encoding: polar
    sensor_geometry: differential_drive
    privileged_input_status: none
```

### Required Fields

- `observation_level`: Must be a valid level from `robot_sf/benchmark/observation_levels.py`
- `observation_mode`: The planner-side observation mode (`dict`, `socnav_state`, etc.)
- `allowed_observation_keys`: List of observation keys the checkpoint expects
- `goal_encoding`, `sensor_geometry`, `privileged_input_status`: Additional metadata

### Resolution Behavior

The observation contract is resolved by `resolve_learned_checkpoint_observation_contract`
in `robot_sf/benchmark/algorithm_metadata.py`. Resolution follows this priority:
1. Explicit `observation_mode` or `observation_level` override (command-line or config)
2. `algo_config.observation_contract` dict
3. `algo_config.benchmark_promotion` dict
4. Model registry `benchmark_promotion` metadata (lookup by `model_id`)
5. Algorithm default observation mode (fallback when metadata not required)

If none of the above sources provide metadata and the checkpoint requires it,
a ValueError is raised with guidance on where to add the metadata.

### Regression Testing

The test `test_guarded_ppo_checkpoint_observation_contract_resolves_from_registry`
in `tests/benchmark/test_algorithm_metadata_contract.py` validates this contract.
See issue #4837 for context on why this metadata is required.

## Reference Files

- `robot_sf/sensor/sensor_fusion.py`
- `robot_sf/sensor/socnav_observation.py`
- `robot_sf/gym_env/robot_env.py`
- `robot_sf/nav/occupancy_grid.py`
- `robot_sf/benchmark/observation_levels.py`
- `robot_sf/benchmark/algorithm_metadata.py`
- `robot_sf/models/registry.py`
- `model/registry.yaml`
