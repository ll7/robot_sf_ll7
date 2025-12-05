# API Contracts: Occupancy Grid Module

**Date**: 2025-12-04  
**Phase**: Phase 1 (Design & Contracts)  
**Format**: Python function signatures and Gymnasium integration

---

## Core API: Grid Generation & Management

### Function: `create_occupancy_grid`

**Signature**:
```python
def create_occupancy_grid(
    robot_pose: RobotPose,
    config: GridConfig,
    obstacles: List[Obstacle],
    pedestrians: List[Pedestrian],
) -> OccupancyGrid
```

**Purpose**: Create a new occupancy grid from configuration and current world state.

**Parameters**:
- `robot_pose: RobotPose` — Robot position (x, y) and heading θ in world frame
- `config: GridConfig` — Grid configuration (size, resolution, frame mode, channels)
- `obstacles: List[Obstacle]` — Static obstacles from map (line segments or polygons)
- `pedestrians: List[Pedestrian]` — Pedestrian states (position, radius) from simulator

**Returns**:
- `OccupancyGrid` — Fully populated grid with channels

**Throws**:
- `ValueError` — If config invalid or dimensions mismatched
- `RuntimeError` — If rasterization fails

**Guarantees**:
- Deterministic: Same input → same output (no randomness)
- Reproducible: Seeded environment produces identical grids
- Performance: <5ms for 10m×10m at 0.1m resolution with ≤100 pedestrians
- All grid cells in [0, 1] range

**Example**:
```python
from robot_sf.nav.occupancy import create_occupancy_grid, GridConfig

grid = create_occupancy_grid(
    robot_pose=(5.0, 5.0, 0.5),  # x, y, theta
    config=GridConfig(width=10.0, height=10.0, resolution=0.1, use_ego_frame=True), 
    obstacles=map_obstacles,
    pedestrians=sim_pedestrians,
)
```

---

### Method: `OccupancyGrid.update`

**Signature**:
```python
def update(
    self,
    robot_pose: RobotPose,
    obstacles: List[Obstacle],
    pedestrians: List[Pedestrian],
) -> None
```

**Purpose**: Update an existing grid's pedestrian channel and timestamp (obstacles static, frame fixed at creation).

**Parameters**:
- `robot_pose: RobotPose` — Current robot pose (for ego-frame recomputation)
- `obstacles: List[Obstacle]` — Static obstacles (required; pass the current list)
- `pedestrians: List[Pedestrian]` — Updated pedestrian positions

**Side Effects**:
- Modifies `self.pedestrian_channel.data` in-place
- Updates `self.timestamp`
- If frame="ego", rotates grid based on new robot heading

**Performance**:
- <2ms typical for pedestrian channel update

**Example**:
```python
for step in range(100):
    # Step simulation
    new_robot_pose = robot.get_pose()
    new_pedestrians = sim.get_pedestrians()
    
    # Update grid
    grid.update(
        robot_pose=new_robot_pose,
        obstacles=map_obstacles,
        pedestrians=new_pedestrians,
    )
```

---

## Query API: Point-of-Interest Checks

### Function: `query_occupancy`

**Signature**:
```python
def query_occupancy(
    grid: OccupancyGrid,
    query: POIQuery,
) -> POIResult
```

**Purpose**: Check occupancy status at a specific location or region.

**Parameters**:
- `grid: OccupancyGrid` — Grid to query
- `query: POIQuery` — Query specification (point, circle, or rectangle)

**Returns**:
- `POIResult` — Occupancy status with per-channel breakdown

**Throws**:
- `ValueError` — If query frame doesn't match grid frame
- `TypeError` — If query type invalid

**Guarantees**:
- <1ms execution time for any grid size up to 20m×20m at 0.1m resolution
- Deterministic: Same query on same grid → same result
- Conservative: `is_occupied=True` if ANY cell in region is occupied

**Example**:
```python
from robot_sf.nav.occupancy import query_occupancy, POIQuery

# Check single point
point_query = POIQuery(query_type="point", world_x=3.0, world_y=3.0)
result = query_occupancy(grid, point_query)
print(f"Safe to spawn: {result.safe_to_spawn}")

# Check circular region
circle_query = POIQuery(
    query_type="circle", 
    world_x=3.0, 
    world_y=3.0, 
    radius_m=1.0
)
result = query_occupancy(grid, circle_query)
if result.occupancy_fraction < 0.1:  # <10% blocked
    print("Region mostly free")
```

---

## Observation API: Gymnasium Integration

### Function: `grid_to_observation`

**Signature**:
```python
def grid_to_observation(grid: OccupancyGrid) -> np.ndarray
```

**Purpose**: Convert OccupancyGrid to numpy array suitable for gymnasium Box observation space.

**Parameters**:
- `grid: OccupancyGrid` — Grid to convert

**Returns**:
- `np.ndarray` with shape `(num_channels, height, width)` and dtype `float32`

**Guarantees**:
- Shape: `(len(grid.channels), height, width)` where dimensions computed from grid size/resolution
- Dtype: Always `float32`
- Values: All in [0, 1] range
- Channel order: Matches `grid.channels.keys()` iteration order (use OrderedDict or list if order critical)

**Example**:
```python
from robot_sf.nav.occupancy import grid_to_observation

obs_array = grid_to_observation(grid)
# obs_array.shape = (2, 100, 100) for 2-channel 10m×10m grid at 0.1m resolution
```

---

### Gymnasium Environment Integration

**Observation Space**:
```python
from gymnasium import spaces
import numpy as np

# When use_occupancy_grid=True in RobotSimulationConfig:
obs_space = spaces.Box(
    low=0.0,
    high=1.0,
    shape=(num_channels, height, width),
    dtype=np.float32,
)

# Included in environment observation under key "occupancy_grid":
obs = {
    "occupancy_grid": np.ndarray(...),  # from grid_to_observation()
    # ... other observation keys (if any)
}
```

**Reset Behavior**:
```python
obs, info = env.reset(seed=42)
grid_obs = obs["occupancy_grid"]  # Updated
```

**Step Behavior**:
```python
obs, reward, terminated, truncated, info = env.step(action)
grid_obs = obs["occupancy_grid"]  # Updated with latest pedestrian positions
```

---

## Visualization API: Pygame Integration

### Function: `render_grid_pygame`

**Signature**:
```python
def render_grid_pygame(
    grid: OccupancyGrid,
    surface: pygame.Surface,
    robot_pose: RobotPose,
    visible_channels: List[str],
    alpha: float = 0.3,
) -> None
```

**Purpose**: Render occupancy grid overlay onto pygame surface.

**Parameters**:
- `grid: OccupancyGrid` — Grid to render
- `surface: pygame.Surface` — Pygame surface to draw on
- `robot_pose: RobotPose` — Robot position (for coordinate transforms)
- `visible_channels: List[str]` — Which channels to render (e.g., ["static_obstacles"])
- `alpha: float` — Transparency for overlay (0.0 = transparent, 1.0 = opaque); default 0.3

**Side Effects**:
- Draws directly onto `surface` (modifies in-place)
- No return value

**Performance**:
- <2ms per frame at 10m×10m grid, 100×100 cells

**Color Scheme**:
- static_obstacles: Yellow (255, 255, 0) with alpha
- pedestrians: Red (255, 0, 0) with alpha
- Other channels: Cyan (0, 255, 255) with alpha
- Free cells: Transparent (no drawing)

**Example**:
```python
from robot_sf.render.sim_view import render_grid_pygame
import pygame

surface = pygame.Surface((800, 800))
render_grid_pygame(
    grid=grid,
    surface=surface,
    robot_pose=robot.get_pose(),
    visible_channels=["static_obstacles", "pedestrians"],
    alpha=0.4,
)
```

---

### Function: `toggle_grid_channel_visibility`

**Signature**:
```python
def toggle_grid_channel_visibility(channel_name: str) -> None
```

**Purpose**: Toggle visualization of a specific grid channel (interactive control).

**Parameters**:
- `channel_name: str` — Name of channel (e.g., "pedestrians")

**Side Effects**:
- Updates global or instance visualization state
- Next `render_grid_pygame()` call respects new visibility state

**Example** (in pygame event loop):
```python
for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_p:  # 'P' key
            toggle_grid_channel_visibility("pedestrians")
        elif event.key == pygame.K_o:  # 'O' key
            toggle_grid_channel_visibility("static_obstacles")
```

---

## Configuration API

### Class: `GridConfig`

**Constructor**:
```python
GridConfig(
    resolution: float = 0.1,
    width: float = 20.0,
    height: float = 20.0,
    use_ego_frame: bool = False,
    channels: List[GridChannel] = [GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
)
```

**Validation**:
- `resolution > 0` — Positive resolution
- `width > 0` and `height > 0` — Positive dimensions
- `isinstance(use_ego_frame, bool)` — Frame selection is boolean
- `len(channels) > 0` — At least one channel

**Integration with RobotSimulationConfig**:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig

config = RobotSimulationConfig(
    use_occupancy_grid=True,
    grid_config=GridConfig(
        width=10.0,
        height=10.0,
        resolution=0.1,
        use_ego_frame=True,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    ),
    # ... other config fields
)
```

---

## Error Handling

### Standard Exceptions

**`ValueError`**:
- Invalid grid config (negative size, zero resolution, unknown frame)
- Invalid query (out-of-range radius, unknown query type)
- Mismatched grid frame and query frame

**`RuntimeError`**:
- Rasterization failure (numerical issues, memory)
- Missing pedestrian state in simulator

**`TypeError`**:
- Wrong argument type to any function
- Gymnasium Box/Space mismatch

**Handling Pattern**:
```python
from robot_sf.nav.occupancy import POIQuery, query_occupancy

try:
    result = query_occupancy(grid, POIQuery(...))
except ValueError as e:
    print(f"Invalid query: {e}")
    # Handle gracefully
except RuntimeError as e:
    print(f"Grid error: {e}")
    # Fallback behavior
```

---

## Version & Compatibility

**API Version**: 1.0  
**Introduced**: 2025-12 (feature 339)  
**Breaking Changes**: None (new API, extends legacy `occupancy.py`)

**Future Compatibility**:
- GridConfig fields may be extended (new optional fields only)
- POIResult may gain new fields (safe-to-add without breaking)
- Function signatures remain stable

---

## Summary Table

| Function/Method | Input | Output | Performance |
|-----------------|-------|--------|-------------|
| `create_occupancy_grid()` | Config, obstacles, pedestrians | OccupancyGrid | <5ms |
| `grid.update()` | Robot pose, pedestrians | None (in-place) | <2ms |
| `query_occupancy()` | Grid, POIQuery | POIResult | <1ms |
| `grid_to_observation()` | OccupancyGrid | np.ndarray | <1ms |
| `render_grid_pygame()` | Grid, surface, channels | None (draws on surface) | <2ms |
| `toggle_grid_channel_visibility()` | Channel name | None (state change) | Instant |

---

## References

- **Data Model**: `specs/339-extend-occupancy-grid/data-model.md`
- **Implementation**: Public grid API in `robot_sf/nav/occupancy_grid.py` (with helpers in
  `occupancy_grid_rasterization.py` and `occupancy_grid_utils.py`); continuous checks live in
  `robot_sf/nav/occupancy.py`
- **Tests**: `tests/test_occupancy_*.py` (pytest test suites)
- **Examples**: `examples/` (usage examples and demos)
