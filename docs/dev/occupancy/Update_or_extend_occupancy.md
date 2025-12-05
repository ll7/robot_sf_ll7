# Occupancy Grid Usage Guide

This guide explains how to enable the multi-channel occupancy grid, add it to Gymnasium
observations, run spawn-safety queries, and debug common issues. The grid is opt-in and
configured via `RobotSimulationConfig`/`GridConfig`; existing environments continue to work
without it.

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

- `GridConfig`: `width`, `height` (meters), `resolution` (m/cell), `channels` (list of
  `GridChannel`, e.g., OBSTACLES/PEDESTRIANS/ROBOT/COMBINED), `dtype` (`np.float32` or `np.uint8`),
  `use_ego_frame` (rotate grid with the robot pose).
- `OccupancyGrid.generate(obstacles, pedestrians, robot_pose, ego_frame=False)`: rasterize line
  obstacles and circular pedestrians into `[C, H, W]` grid data. Uses `config.use_ego_frame` unless
  explicitly overridden by `ego_frame=True`.
- `OccupancyGrid.query(POIQuery) -> POIResult`: evaluate occupancy for a point/line/circle/rectangle,
  returning per-channel means and boolean helpers (`is_occupied`, `safe_to_spawn`,
  `occupancy_fraction`). Queries automatically clamp out-of-bounds coordinates and honour ego-frame
  transforms using the last robot pose.
- `OccupancyGrid.to_observation() -> np.ndarray`: float32 array in `[0, 1]`, channel order matches
  `GridConfig.channels`.
- `OccupancyGrid.reset()`: clear cached grid/origin/pose.

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

## Visualization & troubleshooting

- Rendering: `OccupancyGrid.render_pygame(surface, scale=1.0, alpha=128)` draws a translucent overlay
  per channel (obstacles=yellow, pedestrians=red, robot=blue, combined=orange). For headless runs,
  set `SDL_VIDEODRIVER=dummy`.
- Empty grids: confirm `use_occupancy_grid=True` and that obstacles/pedestrians are supplied; for
  ego-frame grids ensure the robot pose is set before queries.
- Unexpected shapes: verify `GridConfig.width/height/resolution` produce the intended `[H, W]`.
- Out-of-range values: `to_observation()` clips to `[0, 1]`; prefer `dtype=np.uint8` for binary
  occupancy if you need tighter bounds.
- Out-of-bounds queries: coordinates are clamped; if you want strict errors, validate with
  `grid_utils.is_within_grid(...)` before querying.

## Worked examples

- Quickstart walkthrough: `examples/quickstart/04_occupancy_grid.py`
- Advanced spawn validation + reward shaping: `examples/advanced/20_occupancy_grid_workflow.py`
