# Occupancy Grid Usage Guide

This guide explains how to enable the multi-channel occupancy grid, add it to Gymnasium
observations, query spawn safety, and visualize overlays. The grid is opt-in and configured
via `RobotSimulationConfig`/`GridConfig`; existing environments continue to work without it.

## Quickstart (environment)

```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import GridConfig, RobotSimulationConfig
from robot_sf.nav.occupancy_grid import GridChannel

grid_config = GridConfig(
    width=8.0,
    height=8.0,
    resolution=0.2,
    channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
    use_ego_frame=False,
)

config = RobotSimulationConfig(
    use_occupancy_grid=True,
    include_grid_in_observation=True,
    grid_config=grid_config,
)

env = make_robot_env(config=config, debug=False)
obs, _ = env.reset(seed=42)
grid_obs = obs["occupancy_grid"]  # shape: (channels, height, width), dtype float32
env.close()
```

## Core API reference

**GridConfig**

```python
GridConfig(
    width: float = 20.0,
    height: float = 20.0,
    resolution: float = 0.1,
    channels: list[GridChannel] = [GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    dtype: type = np.float32,
    max_distance: float = 0.5,
    use_ego_frame: bool = False,
)
```

- `width`/`height`: meters covered by the grid; `grid_width/height` derive from `ceil(size/resolution)`.
- `resolution`: meters per cell (higher → finer grid).
- `channels`: ordered list of layers; `COMBINED` is automatically max-pooled from the others.
- `dtype`: `np.float32` for continuous occupancy, `np.uint8` for binary.
- `use_ego_frame`: rotate grid with the robot; world frame stays fixed.

**OccupancyGrid methods**

- `generate(obstacles, pedestrians, robot_pose, ego_frame=False) -> np.ndarray`: rasterize obstacles
  (line segments) and pedestrians (circles) into `[C, H, W]`. Uses ego-frame if either `config.use_ego_frame`
  or `ego_frame=True`.
- `query(POIQuery) -> POIResult`: evaluate occupancy for POINT/CIRCLE/RECT/LINE. Returns per-channel
  means and helpers (`is_occupied`, `safe_to_spawn`, `occupancy_fraction`). Coordinates are clamped
  to the grid and rotated automatically for ego-frame grids.
- `to_observation() -> np.ndarray`: float32 array in `[0, 1]`, channel order matches `GridConfig.channels`.
- `reset()`: clear cached grid/origin/pose.

### Query types

- `POIQuery(query_type=POIQueryType.POINT|CIRCLE|RECT|LINE, x, y, radius, width, height, x2, y2,
  channels=None)`; coordinates are in world frame unless the grid is built in ego frame.
- `POIResult` fields: `occupancy` (mean of selected cells), `min_occupancy`, `max_occupancy`,
  `mean_occupancy`, `num_cells`, `channel_results` (per-channel means), plus convenience properties
  `is_occupied` (`>0.1`) and `safe_to_spawn` (`<0.05`).

## Configuration guide

- `RobotSimulationConfig.use_occupancy_grid`: opt-in switch to build grids.
- `RobotSimulationConfig.grid_config`: `GridConfig` instance (auto-created when `use_occupancy_grid`
  is `True` and none is supplied).
- `RobotSimulationConfig.include_grid_in_observation`: add `"occupancy_grid"` to Gymnasium
  observation dict (`spaces.Box(low=0, high=1, shape=[C,H,W], dtype=float32)`).
- `RobotSimulationConfig.show_occupancy_grid` and `grid_visualization_alpha`: enable pygame overlay
  with configurable transparency; requires `use_occupancy_grid=True`.
- `GridChannel.COMBINED` automatically max-pools all non-combined channels into a single layer.

Frame selection tips:
- **World frame** (default): stable axis alignment; good for map debugging and global planners.
- **Ego frame** (`use_ego_frame=True`): grid rotates with the robot heading; good for CNN policies that
  expect a robot-centric view.

Channel guidance:
- Start with `[OBSTACLES, PEDESTRIANS, COMBINED]` for RL.
- Add `ROBOT` if you need explicit robot footprint; disable `COMBINED` to keep channels independent.

## Queries & spawn validation

```python
from robot_sf.nav.occupancy_grid import POIQuery, POIQueryType

grid = env.unwrapped.occupancy_grid  # created when include_grid_in_observation=True
query = POIQuery(x=4.0, y=4.0, radius=0.5, query_type=POIQueryType.CIRCLE)
result = grid.query(query)
print(f"Safe to spawn: {result.safe_to_spawn}, occupancy={result.occupancy_fraction:.3f}")
print("Per-channel:", {ch.value: v for ch, v in result.per_channel_results.items()})
```

- Use circle/rect queries for spawn clearance; point queries are cheap (<1 ms).
- Ego-frame grids accept world-frame coordinates and internally rotate using the stored robot pose.
- Combine channels by setting `channels=[GridChannel.COMBINED]` on the `POIQuery` to avoid
  double-counting.

## Visualization

- Headless overlay rendering:

```python
import os, pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
scale = 4  # pixels per cell
surface = pygame.Surface((grid.shape[2] * scale, grid.shape[1] * scale), pygame.SRCALPHA)
grid.render_pygame(surface, robot_pose=robot_pose, scale=scale, alpha=160)
pygame.quit()
```

- In-sim overlay: set `show_occupancy_grid=True` and `grid_visualization_alpha` on `RobotSimulationConfig`.
  The overlay is drawn after entities inside `SimulationView`.

## Troubleshooting

- Grid not updating: ensure `include_grid_in_observation=True` so `RobotEnv` rebuilds per step; verify
  pedestrians/obstacles are passed to `generate`.
- Ego-frame misalignment: confirm `use_ego_frame=True` and that `robot_pose` is the latest pose (tests
  cover tuple/ndarray handling).
- Shape mismatch: check `width/height/resolution` math and channel count; observation space derives from
  `GridConfig`.
- Values outside [0, 1]: `to_observation()` clips; choose `dtype=np.uint8` for binary occupancy when
  you want hard bounds.
- Performance slow: reduce `grid_width/height` (coarser resolution), drop unused channels, or query with
  smaller AOIs (POINT/CIRCLE instead of large RECT).
- Out-of-bounds queries: coordinates clamp to the nearest cell; validate with
  `robot_sf.nav.occupancy_grid_utils.is_within_grid` if you prefer explicit guards.

## Worked examples

- Quickstart walkthrough: `examples/quickstart/04_occupancy_grid.py`
- Advanced spawn validation + reward shaping: `examples/advanced/20_occupancy_grid_workflow.py`
- RL reward shaping demo: `examples/occupancy_reward_shaping.py`
