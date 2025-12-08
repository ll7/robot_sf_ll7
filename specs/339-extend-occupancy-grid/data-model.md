# Data Model: Extended Occupancy Grid

**Date**: 2025-12-04  
**Phase**: Phase 1 (Design & Contracts)  
**Status**: Complete

This document defines the core data entities, their fields, relationships, and invariants for the occupancy grid feature.

---

## Core Entities

### 1. OccupancyGrid

**Purpose**: Main container for occupancy information at a point in time. Represents the navigable/obstacle space around the robot.

**Fields**:
- `size_m: Tuple[float, float]` — Grid dimensions in meters (width, height)
- `resolution_m: float` — Meters per grid cell (e.g., 0.1m = 10cm per cell)
- `frame: Literal["ego", "world"]` — Coordinate frame (ego = robot-relative; world = global)
- `timestamp: float` — Seconds since epoch or environment step counter
- `robot_pose: RobotPose` — Robot position/heading (origin for ego-frame; reference for world-frame)
- `channels: Dict[str, GridChannel]` — Named occupancy layers (e.g., "obstacles", "pedestrians")

**Invariants**:
- `size_m[0] > 0 and size_m[1] > 0` — Positive dimensions
- `resolution_m > 0` — Positive resolution
- `frame in ["ego", "world"]` — Valid frame mode
- All channels in dict have same grid shape: `(height, width)` where `height = size_m[1] / resolution_m`, `width = size_m[0] / resolution_m`
- `robot_pose` is not None when frame == "ego"

**State Transitions**:
- Created at environment reset with config-specified size/resolution
- Updated each environment step via `update(robot_pose, obstacles, pedestrians)`
- Channels are independent; pedestrian channel updates don't affect obstacle channel

**Relationships**:
- Contains: Multiple `GridChannel` objects
- Created by: `create_occupancy_grid(config, obstacles, pedestrians, robot_pose) -> OccupancyGrid`
- Queried by: `query_poi(grid, query) -> POIResult`
- Observed by: Gymnasium env via `get_observation(grid) -> np.ndarray`
- Rendered by: `render_grid_pygame(grid, surface, visible_channels)`

---

### 2. GridChannel

**Purpose**: Individual occupancy layer representing one semantic type of space occupancy (e.g., static obstacles or pedestrians).

**Fields**:
- `name: str` — Channel identifier (e.g., "static_obstacles", "pedestrians", "dynamic_agents")
- `data: np.ndarray[float32]` — 2D grid of occupancy values (shape: `(height, width)`)
- `occupancy_type: Literal["binary", "continuous"]` — Semantics of occupancy values
- `valid: bool` — Whether channel is enabled/valid (soft delete without removing data)

**Invariants**:
- `data.dtype == np.float32`
- `0.0 ≤ data ≤ 1.0` — All cells in [0, 1] range
- `occupancy_type in ["binary", "continuous"]`
- If `occupancy_type == "binary"`: values only 0.0 or 1.0 (no intermediate values)
- If `occupancy_type == "continuous"`: any value in [0, 1] allowed
- `data.shape == (height, width)` matches parent grid

**Operations**:
- `get_cell(row, col) -> float` — O(1) lookup
- `set_cell(row, col, value) -> None` — O(1) write
- `to_observation() -> np.ndarray` — Return flattened or reshaped for gymnasium

**Relationships**:
- Owned by: `OccupancyGrid.channels[name]`
- Created by: Grid initialization or update
- Rendered: Independently togglable in visualization

---

### 3. GridConfig

**Purpose**: Configuration specification for grid creation and behavior. Separates tuning parameters from runtime state.

**Fields**:
- `size_m: Tuple[float, float]` — Grid dimensions in meters
- `resolution_m: float` — Meters per cell
- `frame: Literal["ego", "world"]` — Coordinate frame mode
- `occupancy_type: Literal["binary", "continuous"]` — Cell value semantics
- `enabled_channels: List[str]` — Which layers to include (e.g., ["static_obstacles", "pedestrians"])
- `include_static_obstacles: bool` — Whether to add obstacle channel
- `include_pedestrians: bool` — Whether to add pedestrian channel

**Defaults**:
```python
GridConfig(
    size_m=(10.0, 10.0),
    resolution_m=0.1,
    frame="ego",
    occupancy_type="binary",
    enabled_channels=["static_obstacles", "pedestrians"],
    include_static_obstacles=True,
    include_pedestrians=True,
)
```

**Invariants**:
- All fields are immutable or validated on set (e.g., no negative resolution)
- `enabled_channels` subset of available channels
- Used at grid creation; immutable during grid lifetime (new grid requires new config)

**Relationships**:
- Input to: `create_occupancy_grid(config, ...)`
- Stored in: `RobotSimulationConfig` or standalone
- Example usage: `RobotSimulationConfig(grid_config=GridConfig(...))`

---

### 4. POIQuery

**Purpose**: Request object for point-of-interest or area-of-interest occupancy checks (e.g., "is this spawn point safe?").

**Fields**:
- `query_type: Literal["point", "circle", "rectangle"]` — Spatial region type
- `world_x: float` — X coordinate in world frame
- `world_y: float` — Y coordinate in world frame
- `radius_m: Optional[float]` — For circle queries (must be >0 if type=="circle")
- `width_m: Optional[float]` — For rectangle queries
- `height_m: Optional[float]` — For rectangle queries
- `grid_frame: Literal["ego", "world"]` — Which frame to interpret coordinates in
- `channel_names: Optional[List[str]]` — Specific channels to query (None = all enabled)

**Invariants**:
- `query_type in ["point", "circle", "rectangle"]`
- If `query_type == "circle"`: `radius_m > 0`
- If `query_type == "rectangle"`: `width_m > 0` and `height_m > 0`
- `grid_frame` matches grid's frame mode

**Relationships**:
- Input to: `query_occupancy(grid, query) -> POIResult`
- Created by: User code or scenario validation routines

---

### 5. POIResult

**Purpose**: Result of occupancy query. Answers "is this location free?"

**Fields**:
- `query_type: str` — Echo of query type for context
- `is_occupied: bool` — Conservative answer: True if ANY cell in query region is occupied
- `occupancy_fraction: float` — [0, 1] = fraction of cells in region that are occupied
- `is_within_bounds: bool` — Whether query region intersects grid bounds
- `max_occupancy_value: float` — Highest occupancy value in queried region (if continuous mode)
- `channel_results: Dict[str, float]` — Per-channel occupancy fractions (e.g., {"obstacles": 0.0, "pedestrians": 0.1})
- `safe_to_spawn: bool` — True if all cells free (occupancy_fraction == 0.0); convenience flag for spawn validation

**Invariants**:
- `0.0 ≤ occupancy_fraction ≤ 1.0`
- `0.0 ≤ max_occupancy_value ≤ 1.0`
- `is_occupied == (occupancy_fraction > 0.0)`
- `safe_to_spawn == (occupancy_fraction == 0.0)`
- `is_within_bounds` True if any cell of query intersects grid

**Relationships**:
- Output of: `query_occupancy(grid, query)`
- Consumed by: Scenario validation, spawn placement logic, navigation planners

---

## Relationships & Data Flow

### Grid Lifecycle

```
1. Config Created
   ↓
2. Environment Reset
   ├─ Load map static obstacles
   ├─ Create GridConfig
   ├─ Call create_occupancy_grid(config, obstacles, pedestrians, robot_pose)
   │  ├─ Allocate OccupancyGrid
   │  ├─ Create GridChannels (obstacles, pedestrians)
   │  ├─ Rasterize obstacles onto obstacle channel
   │  ├─ Rasterize pedestrians onto pedestrian channel
   │  └─ Return OccupancyGrid
   ├─ Store grid in environment state
   └─ Include grid in observation
   
3. Environment Step
   ├─ Move robot, update pedestrians (physics sim)
   ├─ Call grid.update(robot_pose, obstacles, pedestrians)
   │  ├─ Update pedestrian channel (obstacles unchanged)
   │  └─ Update grid timestamp
   ├─ Include grid in observation
   └─ Optionally render grid visualization
   
4. Optional: POI Query
   ├─ Create POIQuery (world_x, world_y, radius_m, ...)
   ├─ Call query_occupancy(grid, query)
   └─ Use result for spawn validation / planning
   
5. Optionally: Visualization Interaction
   ├─ Toggle channel visibility in pygame
   ├─ Call render_grid_pygame(..., visible_channels)
   └─ Render to screen
```

### Relationship Diagram

```
RobotSimulationConfig
├─ GridConfig
│  ├─ size_m, resolution_m, frame, occupancy_type
│  └─ enabled_channels: List[str]
│
Environment (Gymnasium)
├─ Reset:
│  ├─ Load obstacles from map
│  ├─ Create OccupancyGrid
│  │  ├─ GridChannel("static_obstacles")
│  │  └─ GridChannel("pedestrians")
│  └─ Store in observation
│
├─ Step:
│  ├─ Update OccupancyGrid.update()
│  └─ Observation.occupancy_grid ← GridChannel.data
│
└─ Query (optional):
   ├─ POIQuery → query_occupancy(grid, POIQuery)
   └─ POIResult ← (is_occupied, safe_to_spawn, ...)

Visualization (pygame, optional)
├─ render_grid_pygame(grid, surface, visible_channels)
├─ Interactive toggle: visible_channels ← user input
└─ Each frame: update rendering
```

---

## Validation Rules

### GridConfig Validation

```python
def validate_grid_config(config: GridConfig) -> None:
    assert config.size_m[0] > 0 and config.size_m[1] > 0, "Grid size must be positive"
    assert config.resolution_m > 0, "Resolution must be positive"
    assert config.frame in ["ego", "world"], "Frame must be 'ego' or 'world'"
    assert config.occupancy_type in ["binary", "continuous"], "Occupancy type invalid"
    assert len(config.enabled_channels) > 0, "At least one channel must be enabled"
    for ch in config.enabled_channels:
        assert ch in ["static_obstacles", "pedestrians", ...], f"Unknown channel: {ch}"
```

### GridChannel Validation

```python
def validate_grid_channel(ch: GridChannel) -> None:
    assert 0 <= ch.data.min() and ch.data.max() <= 1.0, "Occupancy out of [0, 1]"
    if ch.occupancy_type == "binary":
        assert set(ch.data.unique()) <= {0.0, 1.0}, "Binary channel must have only 0 or 1"
```

### POIQuery Validation

```python
def validate_poi_query(q: POIQuery, grid: OccupancyGrid) -> None:
    assert q.query_type in ["point", "circle", "rectangle"], "Invalid query type"
    if q.query_type == "circle":
        assert q.radius_m > 0, "Circle radius must be positive"
    if q.query_type == "rectangle":
        assert q.width_m > 0 and q.height_m > 0, "Rectangle dimensions must be positive"
    assert q.grid_frame == grid.frame, "Query frame must match grid frame"
```

---

## Python Type Definitions (Pseudocode)

```python
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np

@dataclass
class GridConfig:
    size_m: Tuple[float, float] = (10.0, 10.0)
    resolution_m: float = 0.1
    frame: Literal["ego", "world"] = "ego"
    occupancy_type: Literal["binary", "continuous"] = "binary"
    enabled_channels: List[str] = ("static_obstacles", "pedestrians")
    include_static_obstacles: bool = True
    include_pedestrians: bool = True

@dataclass
class GridChannel:
    name: str
    data: np.ndarray  # shape (height, width), dtype float32
    occupancy_type: Literal["binary", "continuous"] = "binary"
    valid: bool = True

@dataclass
class OccupancyGrid:
    size_m: Tuple[float, float]
    resolution_m: float
    frame: Literal["ego", "world"]
    timestamp: float
    robot_pose: "RobotPose"  # (x, y, theta)
    channels: Dict[str, GridChannel]

@dataclass
class POIQuery:
    query_type: Literal["point", "circle", "rectangle"]
    world_x: float
    world_y: float
    radius_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    grid_frame: Literal["ego", "world"] = "world"
    channel_names: Optional[List[str]] = None

@dataclass
class POIResult:
    query_type: str
    is_occupied: bool
    occupancy_fraction: float
    is_within_bounds: bool
    max_occupancy_value: float
    channel_results: Dict[str, float]
    safe_to_spawn: bool
```

---

## Summary

- **OccupancyGrid**: Main container; holds multiple channels; updated per step
- **GridChannel**: Individual layer (obstacles, pedestrians); independent occupancy values
- **GridConfig**: Specification for grid creation; immutable during grid lifetime
- **POIQuery**: Request for occupancy info at a location (point, circle, or rectangle)
- **POIResult**: Response with occupancy status and per-channel breakdown

All entities validate inputs; maintain invariants (positive dimensions, values in [0,1], consistent shapes). Data flow is deterministic; grid updates are reproducible given seed and config.

---

## Next Steps

1. Implement entities in `robot_sf/nav/occupancy.py`
2. Create API contracts in `contracts/occupancy_api.md`
3. Generate `quickstart.md` with usage examples
4. Proceed to Phase 2: Task breakdown
