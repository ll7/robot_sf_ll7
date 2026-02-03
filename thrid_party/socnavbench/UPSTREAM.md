# SocNavBench (planner subset) upstream reference

Origin: https://github.com/CMU-TBD/SocNavBench
Commit: 2724ea85ff22ca61a88ee95285dfc3fa056656c6
License: MIT (see LICENSE)

## Notes
- This directory vendors only the planner-related subset of SocNavBench required for
  the SamplingPlanner integration in Robot SF.
- Imports are preserved in the SocNavBench style (e.g., `control_pipelines.*`) and
  resolved by inserting this directory on `sys.path`.
- A minimal `dotmap.py` shim is included to replace the external `dotmap` dependency.

## Files vendored
- agents/agent.py
- agents/agent_base.py
- agents/humans/human.py
- agents/robot_agent.py
- agents/robot_utils.py
- control_pipelines/base.py
- control_pipelines/control_pipeline_v0.py
- control_pipelines/control_pipeline_v0_helper.py
- costs/cost.py
- costs/quad_cost_with_wrapping.py
- metrics/cost_functions.py
- metrics/cost_utils.py
- objectives/angle_distance.py
- objectives/goal_distance.py
- objectives/objective_function.py
- objectives/obstacle_avoidance.py
- objectives/personal_space_cost.py
- obstacles/obstacle_map.py
- obstacles/sbpd_map.py
- optCtrl/lqr.py
- params/central_params.py
- planners/planner.py
- planners/sampling_planner.py
- sbpd/sbpd_renderer.py
- simulators/sim_state.py
- systems/dubins_3d.py
- systems/dubins_car.py
- systems/dubins_v2.py
- systems/dynamics.py
- trajectory/spline/spline.py
- trajectory/spline/spline_3rd_order.py
- trajectory/trajectory.py
- utils/angle_utils.py
- utils/fmm_map.py
- utils/utils.py
- utils/voxel_map_utils.py
- waypoint_grids/base.py
- waypoint_grids/projected_image_space_grid.py
- waypoint_grids/uniform_sampling_grid.py
