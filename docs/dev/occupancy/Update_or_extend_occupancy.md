# Update or extend the exisiting occupancy grid 2025-12-04

For our observation space, we want to be able to provide grid based occupancy grids. We would like to provide grids, because they work well with "image"-based observation spaces.
If we have to create occupancy grids dynamically in each frame, it could be worth to do this very efficiently. 
We should extend the grid based occupancy grid in our package.

## Feature Requirements

The solution should include all options.
We should create new layer for each new form of information. It is easy to merge this information when needed, but hard to seperate.

Local occupancy grid (recommended): configurable size (width/height in meters), resolution (meters per cell), and frame (ego-rotated vs. world-aligned). Include occupancy values for static obstacles; optionally add separate channels for dynamic agents (pedestrians, other robots). This keeps the interface compact and planner-friendly.

Configurable grid_size_m, resolution_m, frame (ego/world), and optional channels (static obstacles, pedestrians). Static + dynamic on separate channels helps learned/pluggable planners

## Avoid the following weaknesses

- No occupancy grid or costmap; only continuous geometric checks.
- O(N) over all obstacle segments/pedestrians each step; no spatial indexing for larger scenes.
- No caching/rasterization, so large maps or dense crowds will scale poorly.
- Only collision flags; no distance-to-obstacle/clearance fields to inform planners directly.

## Use cases that should be met

- Configurable gymnasium observation space to percieve the environment.
- Point of interest retrieval to check if a point or area is free. This can be helpful to empty determin spawn points.

## Implementation Requirements

- Increase the test coverage of `robot_sf/nav/occupancy.py` to 100% first.
- Extend or modify the previous occupancy grid specified in `robot_sf/nav/occupancy.py`.
- Write tests for the new implementation.
- Write visual tests where the new occupancy grid and its inforamtion can be visualized in the pygame visualization in `robot_sf/render/sim_view.py`. I imagine that we load a map, enable the grid visualization and we recieve a gird overlay where each cell of the grid can be visualized. cells with obstacles could have a light yellow tint and. Empty cells are transparent.
	- Visual tests go to test_pygame.
- Update the documentation in `docs/dev/occupancy/Update_or_extend_occupancy.md` to explain how to use the new occupancy grid.