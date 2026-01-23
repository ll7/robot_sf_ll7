# SocNav ORCA Failure Analysis (2026-01-23)

Run details
* policy: socnav_orca (static obstacle constraints enabled)
* scenario set: configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml
* output root: output/recordings/scenario_videos_classic_interactions_francis2023_socnav_orca_20260123_081218
* summary: summary.md
* failure frames: failure_frames (contact sheets per failed video)

## Reproduction settings (explicit)

Base config
* training config: configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml
* observation mode: socnav_struct
* occupancy grid: enabled
  + resolution: 0.2 m
  + width/height: 32.0 m
  + channels: [obstacles, pedestrians, combined]
  + use_ego_frame: true
  + center_on_robot: true
* robot: bicycle (allow_backwards=true)
* sim: peds_have_obstacle_forces=true; prf_config.is_active=true; max_peds_per_group=3

ORCA defaults (robot_sf/planner/socnav.py)
* max_linear_speed=3.0, max_angular_speed=1.0
* orca_neighbor_dist=5.0
* orca_time_horizon=2.0
* orca_time_horizon_obst=3.0
* orca_avoidance_weight=1.2
* orca_obstacle_threshold=0.5
* orca_obstacle_range=6.0
* orca_obstacle_max_points=80
* orca_obstacle_radius_scale=1.0
* orca_heading_slowdown=0.2

Sweep overrides
* Sweep A: --socnav-orca-time-horizon 4.0 --socnav-orca-neighbor-dist 8.0
* Sweep B: --socnav-orca-time-horizon 6.0 --socnav-orca-neighbor-dist 10.0

Command template (best sweep)

```
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/tools/render_scenario_videos.py \
  --training-config configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml \
  --policy socnav_orca --socnav-use-grid \
  --socnav-orca-time-horizon 6.0 --socnav-orca-neighbor-dist 10.0 --all
```

## Headline results

* total videos: 129
* success: 69 (53.5%)
* failures: 60 (46.5%)
* stop reasons: 59 collisions, 1 terminated, 0 timeouts

Change vs previous ORCA baseline (2026-01-22)
* success improved from 53/129 (41.1%) to 69/129 (53.5%).
* collision failures reduced from 75 to 59.

## Parameter sweep comparison (time horizon + neighbor distance)

Sweep A (time_horizon=4.0, neighbor_dist=8.0)
* output: output/recordings/scenario_videos_classic_interactions_francis2023_socnav_orca_20260123_082221
* summary: summary.md
* failure frames: failure_frames
* results: 80/129 success (62.0%), 46 collisions, 3 terminated

Sweep B (time_horizon=6.0, neighbor_dist=10.0)
* output: output/recordings/scenario_videos_classic_interactions_francis2023_socnav_orca_20260123_083815
* summary: summary.md
* failure frames: failure_frames
* results: 87/129 success (67.4%), 39 collisions, 3 terminated

Observations
* Success rate keeps improving as the horizon/neighbor radius increase.
* Head‑on corridor scenarios recover in sweep B (no longer zero‑success).
* Some regressions appear in sweep B (classic_crossing_high/medium go to 0/3).

## Zero‑success scenarios (0/3 seeds)

Classic scenarios
* classic_head_on_corridor_low / medium
* classic_merging_low / medium
* classic_overtaking_low / medium
* classic_t_intersection_medium

Francis 2023 scenarios
* francis2023_frontal_approach
* francis2023_parallel_traffic
* francis2023_pedestrian_obstruction
* francis2023_robot_crowding
* francis2023_robot_overtaking
* francis2023_crowd_navigation
* francis2023_narrow_hallway
* francis2023_narrow_doorway
* francis2023_join_group
* francis2023_leave_group
* francis2023_exiting_elevator

## Zero‑success scenarios (Sweep B: time_horizon=6.0, neighbor_dist=10.0)

Classic scenarios
* classic_crossing_high / medium
* classic_merging_low / medium
* classic_overtaking_low / medium
* classic_t_intersection_low / medium

Francis 2023 scenarios
* francis2023_frontal_approach
* francis2023_narrow_doorway
* francis2023_narrow_hallway
* francis2023_parallel_traffic

## Partial‑success scenarios

* classic_crossing_medium: 2/3 success
* classic_doorway_medium: 2/3 success
* classic_t_intersection_low: 2/3 success
* francis2023_circular_crossing: 2/3 success
* francis2023_perpendicular_traffic: 1/3 success

## Partial‑success scenarios (Sweep B)

* classic_doorway_medium: 2/3 success
* francis2023_circular_crossing: 2/3 success
* francis2023_exiting_elevator: 2/3 success
* francis2023_pedestrian_obstruction: 2/3 success
* francis2023_robot_crowding: 1/3 success

## Consistent successes (3/3 seeds)

Classic scenarios
* classic_crossing_low / high
* classic_bottleneck_low / medium / high
* classic_group_crossing_low / medium
* classic_doorway_low

Francis 2023 scenarios
* francis2023_down_path
* francis2023_blind_corner
* francis2023_entering_room
* francis2023_exiting_room
* francis2023_entering_elevator
* francis2023_following_human
* francis2023_leading_human
* francis2023_accompanying_peer
* francis2023_intersection_no_gesture
* francis2023_intersection_wait
* francis2023_intersection_proceed

## Failure patterns (from summary + contact sheets)

* Collisions still dominate failures. The improvement suggests static obstacle constraints help, but dense traffic and head‑on interactions remain problematic.
* Head‑on corridor, merging, and overtaking still fail consistently: these require negotiation and higher‑level yielding behaviors beyond velocity obstacles alone.
* Narrow hallway/doorway scenarios still fail in the Francis set, indicating obstacle constraints help but not enough to handle tight passages with pedestrians present.

## Suspected causes (algorithm + implementation)

* **Static obstacles now modeled, but pedestrian interactions remain brittle.** Dense flows (crowd/parallel/robot‑crowding) likely need longer horizons or more conservative neighbor selection.
* **Holonomic ORCA vs. non‑holonomic robot** still limits lateral avoidance in tight head‑on cases.
* **Time horizon sensitivity** is likely still limiting anticipation in crossings and merging.

## Suspicious signals (Option C checks recommended)

* Head‑on corridor remains 0% success despite static obstacles being considered. This could indicate:
  + Pedestrian velocity frame mismatch (world vs. ego) in ORCA constraints.
  + ORCA constraint solver degeneracy in symmetric head‑on setups.
  + Missing obstacle segments (grid‑based approximation might under‑represent walls).
* Recommended Option C follow‑ups:
  + Compare ORCA step outputs in a synthetic head‑on test against a reference implementation.
  + Verify pedestrian velocity frame in structured observations and convert to ego if needed.
  + Add a minimal static‑wall ORCA unit test with a corridor boundary to validate obstacle handling.

## Notes

* Frame contact sheets were generated for all failed runs under `failure_frames/`.
* This analysis is based on `summary.md` and the captured frames; no additional metrics were computed.

## Option D (execution environment / tmux)

* Rendering and planning logic are deterministic for a given seed; tmux should not change results unless it alters environment variables (e.g., display, SDL, or backend selection).
* If you suspect tmux‑specific behavior, confirm by running a single scenario outside tmux with identical env vars and comparing the summary entry.
