# Defining Single Pedestrians in Maps

This guide explains how to define individually controlled pedestrians with explicit start positions, goals, or trajectories in robot-sf simulation maps.

## Table of Contents
- [Overview](#overview)
- [Use Cases](#use-cases)
- [Pedestrian Definition Format](#pedestrian-definition-format)
- [SVG Map Definition](#svg-map-definition)
- [JSON Map Definition](#json-map-definition)
- [Scenario YAML Overrides](#scenario-yaml-overrides)
- [Preview Helper](#preview-helper)
- [Programmatic Definition](#programmatic-definition)
- [Validation Rules](#validation-rules)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

Single pedestrians are individually controlled pedestrians that spawn at specific positions with either:
- **Fixed goals**: Navigate using Social Force Model toward a goal position
- **Predefined trajectories**: Follow a sequence of waypoints
- **Static positioning**: Remain at the start position (neither goal nor trajectory)

Unlike zone-spawned crowd pedestrians, single pedestrians have deterministic start positions and can follow precise navigation paths.

## Use Cases

- **Testing scenarios**: Create reproducible test cases with specific pedestrian configurations
- **Crossing scenarios**: Simulate pedestrians crossing the robot's path at predictable locations
- **Static obstacles**: Model stationary people (e.g., vendors, information booths)
- **Guided tours**: Simulate pedestrians following specific routes through the environment
- **Interaction studies**: Study robot behavior with controlled pedestrian movements

## Pedestrian Definition Format

Each single pedestrian is defined by:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier for the pedestrian |
| `start` | `Vec2D` | Yes | Starting position as `(x, y)` tuple |
| `goal` | `Vec2D \| None` | No | Goal position for navigation (mutually exclusive with trajectory) |
| `trajectory` | `list[Vec2D] \| None` | No | List of waypoints to follow (mutually exclusive with goal) |
| `speed_m_s` | `float \| None` | No | Optional speed override for this pedestrian (m/s) |
| `wait_at` | `list[PedestrianWaitRule] \| None` | No | Optional waits at trajectory waypoints (trajectory required) |
| `note` | `str \| None` | No | Optional note for scenario documentation |

### Constraints

- **Mutually Exclusive**: A pedestrian can have either a `goal` OR a `trajectory`, but not both
- **Static Pedestrians**: If neither goal nor trajectory is provided, the pedestrian remains static
- **Unique IDs**: All pedestrian IDs within a map must be unique
- **Wait Rules**: `wait_at` requires a trajectory; wait indices refer to trajectory waypoints

### Wait Rule Format

Each wait rule includes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `waypoint_index` | `int` | Yes | Index into the trajectory list where the pedestrian waits |
| `wait_s` | `float` | Yes | Wait duration in seconds |
| `note` | `str \| None` | No | Optional note for documentation |

## SVG Map Definition

### Circle-Based Markers

Single pedestrians can be defined in SVG maps using circle elements with specific labels.

**Label Convention:**
- Start position: `single_ped_<id>_start`
- Goal position: `single_ped_<id>_goal`

**Example SVG snippet:**
```xml
<!-- Pedestrian with goal -->
<circle cx="30" cy="100" r="2" 
        inkscape:label="single_ped_ped1_start" 
        id="ped1_start" />
<circle cx="170" cy="100" r="2" 
        inkscape:label="single_ped_ped1_goal" 
        id="ped1_goal" />

<!-- Static pedestrian (no goal marker) -->
<circle cx="100" cy="50" r="2" 
        inkscape:label="single_ped_vendor_start" 
        id="vendor_start" />
```

**Notes:**
- Trajectories cannot be defined in SVG (use JSON for trajectory support)
- If only a start marker is present, the pedestrian will be static
- Circle radius (`r`) is not used for simulation, only for visualization

## JSON Map Definition

### Structure

Add a `single_pedestrians` array to your map JSON:

```json
{
  "width": 200,
  "height": 200,
  "obstacles": [...],
  "robot_routes": [...],
  "ped_routes": [...],
  "single_pedestrians": [
    {
      "id": "ped1",
      "start": [30, 100],
      "goal": [170, 100]
    },
    {
      "id": "ped2",
      "start": [50, 50],
      "trajectory": [
        [80, 80],
        [120, 120],
        [160, 80]
      ],
      "speed_m_s": 1.2,
      "wait_at": [
        {
          "waypoint_index": 1,
          "wait_s": 2.0,
          "note": "pause at midpoint"
        }
      ],
      "note": "tourist loop"
    },
    {
      "id": "vendor",
      "start": [100, 50]
    }
  ]
}
```

### Coordinate System

- Coordinates are absolute within the map bounds
- The map loader will automatically normalize coordinates based on `x_margin` and `y_margin`
- Ensure all coordinates are within map bounds for proper validation

## Scenario YAML Overrides

Scenario configuration files can override map-defined single pedestrians without editing the SVG/JSON.
Use the `single_pedestrians` list under a scenario entry to override by `id`.
POI labels come from SVG circles tagged as POIs (see `docs/SVG_MAP_EDITOR.md`).

Supported override keys (mutually exclusive pairs):
- `goal` or `goal_poi`
- `trajectory` or `trajectory_poi`

Optional override fields:
- `speed_m_s`
- `wait_at`
- `note`

If the map defines a `goal` but you want a `trajectory`, set `goal: null` in the override.
When using `trajectory_poi`, each `wait_at` entry can specify `poi` instead of `waypoint_index`.

### Example (POI-based)

```yaml
scenarios:
  - name: frontal_approach_slow
    map_file: "../../maps/svg_maps/francis2023/francis2023_frontal_approach.svg"
    single_pedestrians:
      - id: ped1
        goal_poi: "hallway_end"
        speed_m_s: 0.6
        note: "slow walker"
      - id: ped2
        goal: null
        trajectory_poi: ["cross_start", "cross_mid", "cross_end"]
        wait_at:
          - poi: "cross_mid"
            wait_s: 2.0
            note: "yield gesture"
```

### Example (Coordinate-based)

```yaml
scenarios:
  - name: overtaking_demo
    map_file: "../../maps/svg_maps/francis2023/francis2023_ped_overtaking.svg"
    single_pedestrians:
      - id: ped1
        goal: null
        trajectory:
          - [5.0, 8.0]
          - [15.0, 8.0]
          - [25.0, 8.0]
        speed_m_s: 1.0
```

## Preview Helper

Use the preview helper to visualize trajectories and wait points on top of the map geometry:

```bash
uv run python scripts/tools/preview_scenario_trajectories.py \
  --scenario configs/scenarios/classic_interactions.yaml \
  --scenario-id classic_head_on_corridor
```

For headless usage, set `MPLBACKEND=Agg` before running.

## Programmatic Definition

### Using `SinglePedestrianDefinition`

```python
from robot_sf.nav.map_config import SinglePedestrianDefinition

# Goal-based pedestrian
ped_goal = SinglePedestrianDefinition(
    id="commuter_1",
    start=(10.0, 10.0),
    goal=(90.0, 90.0)
)

# Trajectory-based pedestrian
ped_traj = SinglePedestrianDefinition(
    id="tourist_1",
    start=(20.0, 20.0),
    trajectory=[(40.0, 40.0), (60.0, 60.0), (80.0, 80.0)]
)

# Static pedestrian
ped_static = SinglePedestrianDefinition(
    id="vendor_1",
    start=(50.0, 50.0)
)
```

### Adding to MapDefinition

```python
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.gym_env.environment_factory import make_robot_env

# Create map with single pedestrians
map_def = MapDefinition(
    width=100.0,
    height=100.0,
    obstacles=[...],
    robot_spawn_zones=[...],
    ped_spawn_zones=[...],
    robot_goal_zones=[...],
    bounds=[...],
    robot_routes=[...],
    ped_goal_zones=[...],
    ped_crowded_zones=[...],
    ped_routes=[...],
    single_pedestrians=[ped_goal, ped_traj, ped_static]
)

# Create environment with custom map
map_pool = MapDefinitionPool(map_defs={"custom_map": map_def})
config = RobotSimulationConfig(map_pool=map_pool)
env = make_robot_env(config=config)
```

## Validation Rules

The system automatically validates single pedestrian definitions:

### Critical Errors (Raise `ValueError`)

1. **Invalid ID**: Empty string or non-string type
2. **Invalid coordinates**: Non-tuple or wrong length (must be 2-tuples)
3. **Mutually exclusive violation**: Both goal and trajectory provided
4. **Duplicate IDs**: Multiple pedestrians with the same ID
5. **Invalid trajectory**: Non-list type or invalid waypoint format

### Warnings (Log but don't fail)

1. **Static pedestrian**: Neither goal nor trajectory specified
2. **Goal equals start**: Pedestrian will remain at same location
3. **Out of bounds**: Coordinates outside map bounds
4. **Overlapping starts**: Multiple pedestrians start within 0.5m of each other
5. **Duplicate waypoints**: Consecutive identical waypoints in trajectory
6. **Empty trajectory**: Trajectory list is empty

## Examples

### Example 1: Crossing Scenario

```python
# Pedestrian crossing robot's path
crossing_ped = SinglePedestrianDefinition(
    id="crossing_1",
    start=(10.0, 50.0),
    goal=(90.0, 50.0)  # Crosses horizontally
)
```

### Example 2: Complex Trajectory

```python
# Tourist following a specific route
tourist = SinglePedestrianDefinition(
    id="tourist",
    start=(5.0, 5.0),
    trajectory=[
        (20.0, 20.0),   # First landmark
        (50.0, 30.0),   # Second landmark
        (70.0, 60.0),   # Third landmark
        (90.0, 90.0)    # Final destination
    ]
)
```

### Example 3: Multiple Pedestrians

```python
pedestrians = [
    SinglePedestrianDefinition(
        id="ped_north",
        start=(50.0, 10.0),
        goal=(50.0, 90.0)  # Moving north
    ),
    SinglePedestrianDefinition(
        id="ped_south",
        start=(50.0, 90.0),
        goal=(50.0, 10.0)  # Moving south (opposite direction)
    ),
    SinglePedestrianDefinition(
        id="vendor",
        start=(30.0, 50.0)  # Static vendor
    )
]
```

## Troubleshooting

### Error: "Pedestrian ID must be a non-empty string"

**Cause**: Invalid or missing pedestrian ID

**Solution**: Ensure each pedestrian has a unique, non-empty string ID
```python
# Wrong
ped = SinglePedestrianDefinition(id="", start=(10, 10))
ped = SinglePedestrianDefinition(id=123, start=(10, 10))

# Correct
ped = SinglePedestrianDefinition(id="ped1", start=(10, 10))
```

### Error: "goal and trajectory are mutually exclusive"

**Cause**: Both goal and trajectory were provided

**Solution**: Choose either goal-based OR trajectory-based navigation
```python
# Wrong
ped = SinglePedestrianDefinition(
    id="ped1",
    start=(10, 10),
    goal=(50, 50),
    trajectory=[(30, 30)]
)

# Correct (goal-based)
ped = SinglePedestrianDefinition(
    id="ped1",
    start=(10, 10),
    goal=(50, 50)
)

# Correct (trajectory-based)
ped = SinglePedestrianDefinition(
    id="ped1",
    start=(10, 10),
    trajectory=[(30, 30), (50, 50)]
)
```

### Error: "Duplicate single pedestrian IDs found"

**Cause**: Multiple pedestrians have the same ID

**Solution**: Ensure all IDs are unique within the map
```python
# Wrong
peds = [
    SinglePedestrianDefinition(id="ped1", start=(10, 10)),
    SinglePedestrianDefinition(id="ped1", start=(20, 20))  # Duplicate!
]

# Correct
peds = [
    SinglePedestrianDefinition(id="ped1", start=(10, 10)),
    SinglePedestrianDefinition(id="ped2", start=(20, 20))
]
```

### Warning: "Neither goal nor trajectory specified"

**Cause**: Static pedestrian (intentional or accidental)

**Solution**: If intentional, this is expected behavior. Otherwise, add a goal or trajectory:
```python
# Static pedestrian (intentional)
vendor = SinglePedestrianDefinition(
    id="vendor",
    start=(50, 50)  # Will remain here
)

# Add goal if movement is desired
moving_ped = SinglePedestrianDefinition(
    id="ped1",
    start=(50, 50),
    goal=(80, 80)
)
```

### Warning: "Pedestrian start position outside map bounds"

**Cause**: Coordinates exceed map dimensions

**Solution**: Adjust coordinates to be within `[0, width] × [0, height]`
```python
# Map is 100x100
# Wrong
ped = SinglePedestrianDefinition(id="ped1", start=(150, 150))

# Correct
ped = SinglePedestrianDefinition(id="ped1", start=(50, 50))
```

## See Also

- [Example: Single Pedestrian Spawning](../examples/example_single_pedestrian.py)
- [Test Suite: Single Pedestrian Tests](../tests/test_single_pedestrian.py)
- [Environment Documentation](ENVIRONMENT.md)
- [Simulation View Documentation](SIM_VIEW.md)
