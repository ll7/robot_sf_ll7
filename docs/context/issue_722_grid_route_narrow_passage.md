# Grid Route Narrow Passage Improvement (Issue 722)

## Goal

Document the fix for `grid_route` narrow-passage handling that was preventing the six-scenario static slice from reaching stable success.

## Summary

* Planner: `grid_route`
* Implementation: `robot_sf/planner/grid_route.py`
* Config: `configs/algos/grid_route_camera_ready.yaml`
* Proof surface: `configs/scenarios/sets/safety_barrier_static_slice_v1.yaml`

## Validation

### Commands

* `uv run python scripts/validation/run_safety_barrier_static_slice.py --algo grid_route --algo-config configs/algos/grid_route_camera_ready.yaml --output-dir output/tmp/issue722_full_static_slice --workers 1`

### Result

* Episodes: `18`
* Success rate: `1.0`
* Collision rate: `0.0`

Per-scenario success:
* `empty_map_8_directions_east`: `3/3`
* `goal_behind_robot`: `3/3`
* `line_wall_detour`: `3/3`
* `narrow_passage`: `3/3`
* `single_obstacle_circle`: `3/3`
* `single_obstacle_rectangle`: `3/3`

## Notes

* A narrow-passage regression test was added to ensure A* routing goes through the single available opening.
* The camera-ready grid-route config now explicitly includes `clearance_penalty_weight: 0.5`.
* This issue is now validated on the existing static-slice proof surface without regression.
