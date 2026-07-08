# ORCA Radius Change Comparison (1.0 -> 0.6)

## Why change the radius?
The default robot collision radius is 1.0 m (2.0 m diameter), which is larger than
most indoor robots. To test a more realistic footprint, we evaluated ORCA with
`robot_config.radius = 0.6` (1.2 m diameter) using the same scenario sweep.

## What changed
- New config: `configs/training/ppo_imitation/expert_ppo_issue_403_grid_radius_0_6.yaml`
  - `env_overrides.robot_config.radius: 0.6`
- ORCA sweep run:
  - `output/recordings/scenario_videos_classic_interactions_francis2023_socnav_orca_20260123_130937`
  - Failure frames extracted to `.../failure_frames`

Baseline run (radius = 1.0):
- `output/recordings/scenario_videos_classic_interactions_francis2023_socnav_orca_20260123_102719`

## Summary (overall)
- **Baseline (r=1.0):** 102/129 success, 27 collisions
- **Radius 0.6:** 86/129 success, 42 collisions, 1 terminated

Net effect: **success decreased by 16** and **collisions increased by 15**.

## Scenario-level changes
Improved (success rate increased):
- `classic_group_crossing_low`: 0.67 -> 1.0 (1 collision -> 0)
- `francis2023_parallel_traffic`: 0.33 -> 0.67 (2 collisions -> 1)

Regressed (success rate decreased):
- `classic_head_on_corridor_medium`: 1.0 -> 0.33 (0 -> 2 collisions)
- `francis2023_frontal_approach`: 1.0 -> 0.0 (0 -> 3 collisions)
- `francis2023_pedestrian_obstruction`: 1.0 -> 0.0 (0 -> 3 collisions)
- `francis2023_join_group`: 1.0 -> 0.0 (0 -> 3 collisions)
- `francis2023_leave_group`: 0.67 -> 0.0 (1 -> 3 collisions)
- `francis2023_crowd_navigation`: 1.0 -> 0.33 (0 -> 2 collisions)
- `francis2023_perpendicular_traffic`: 1.0 -> 0.33 (0 -> 2 collisions)
- `francis2023_exiting_elevator`: 0.67 -> 0.33 (1 -> 2 collisions)

Persistent failures (0 success in both runs):
- `classic_overtaking_low`
- `classic_overtaking_medium`
- `classic_merging_low`
- `classic_merging_medium`
- `francis2023_narrow_hallway`
- `francis2023_narrow_doorway`
- `francis2023_robot_crowding`

## Qualitative failure notes (radius 0.6)
Sample contact sheets inspected from the new failure frames:
- `classic_head_on_corridor_medium`: collisions appear in narrow corridor crossings (new failures vs baseline).
- `francis2023_frontal_approach` and `francis2023_pedestrian_obstruction`: repeated frontal collisions in straight corridors.
- `francis2023_crowd_navigation`: collisions when threading through dense groups.
- `classic_overtaking_low`: same persistent wall/obstacle interactions as baseline.

## Interpretation
Reducing the collision radius **did not improve ORCA performance** in these scenarios.
Instead, collision rates increased, especially in straight corridor interactions and
frontal approaches. This suggests the ORCA policy is **not limited by robot footprint
size**, but by its avoidance logic under dense or constrained geometry.

The few improved cases appear scenario-specific and do not offset the broader regression.

## Next steps
- Keep **r=1.0** as baseline for now (more stable results).
- If we want to revisit footprint tuning, first improve ORCA obstacle handling or
  consider integrating a stronger planner (e.g., fast-pysf local planner).

