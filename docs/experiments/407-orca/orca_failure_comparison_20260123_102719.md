# ORCA Failure Comparison (2026-01-23 10:27:19)

This file compares the new ORCA run:
* output/recordings/scenario_videos_classic_interactions_francis2023_socnav_orca_20260123_102719

against the prior best sweep:
* output/recordings/scenario_videos_classic_interactions_francis2023_socnav_orca_20260123_083815

## Config/context

* New run uses: configs/training/ppo_imitation/expert_ppo_issue_403_grid_diffdrive.yaml
* Previous run uses: configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml
* ORCA overrides (both): time_horizon=6.0, neighbor_dist=10.0
* socnav_use_grid: true
* robot_speed_m_s: 1.0 (from manifest)

## Aggregate outcome (summary.md)

* 083815: 87/129 success, 39 collisions, 3 terminated
* 102719: 102/129 success, 27 collisions, 0 terminated
* Delta: +15 success, -12 collisions, -3 terminated

## Scenario-level changes

Improved (fail -> success):
* classic_crossing_medium (all 3 seeds)
* classic_crossing_high (all 3 seeds)
* classic_t_intersection_low (all 3 seeds)
* classic_t_intersection_medium (all 3 seeds)
* francis2023_frontal_approach (all 3 seeds)
* classic_doorway_medium (seed 141)
* francis2023_pedestrian_obstruction (seed 206)
* francis2023_parallel_traffic (seed 265)

Regressed (success/terminate -> collision):
* classic_merging_medium (seed 153)
* classic_group_crossing_low (seed 172)
* francis2023_leave_group (seed 259)
* francis2023_robot_crowding (seeds 273, 275)

## Persistent failures (0/3 success)

* classic_overtaking_low
* classic_overtaking_medium
* classic_merging_low
* classic_merging_medium
* francis2023_narrow_hallway
* francis2023_narrow_doorway
* francis2023_robot_crowding

## Failure frame observations (102719)

Source frames: output/recordings/scenario_videos_classic_interactions_francis2023_socnav_orca_20260123_102719/failure_frames

* classic_overtaking / classic_merging: collisions happen early while turning into the narrow mid corridor or passing the central pillar. The trajectory points at the choke and clips the wall/obstacle, suggesting insufficient clearance for non-holonomic turning radius in these layouts.
* francis2023_narrow_hallway / narrow_doorway: contact sheets show the robot colliding at the doorway/hallway pinch point. This aligns with geometry width vs robot footprint/clearance rather than pedestrian dynamics.
* francis2023_robot_crowding: dense pedestrian cluster surrounds the robot; collision appears after the crowd compresses near the robot body. This looks like a multi-agent collision or squeeze failure rather than a single static obstacle.
* francis2023_parallel_traffic: collisions appear mid-corridor while passing by oncoming pedestrians; indicates ORCA still chooses a velocity that intersects pedestrians when corridor width is tight.
* francis2023_circular_crossing (seed 271): collision at step 1 suggests an initial overlap/spawn contact or immediate obstacle overlap.

## Net read

The new run is materially better overall (notably intersections and high-density crossings now succeed), while failures are concentrated in narrow-passage geometry (doorways, hallways, merging/overtaking) and high crowding density. That points to remaining sensitivity to corridor width + turning radius + crowd compression, not general navigation capability.
