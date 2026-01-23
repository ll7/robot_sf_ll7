# SocNav Social Force Failure Analysis (2026-01-23)

Source run:
- output/recordings/scenario_videos_classic_interactions_francis2023_socnav_social_force_20260123_112305

## Aggregate results
- Total: 129
- Success: 61
- Collision: 69
- Terminated/timeout: 0

## Failure distribution (collisions)
Collisions are widespread across both classic and Francis scenarios. Key clusters:
- Classic: crossing (low/medium/high), overtaking (low/medium), doorway (low/medium), merging (low/medium)
- Francis: frontal_approach, pedestrian_obstruction, robot_overtaking, down_path, blind_corner,
  narrow_hallway, narrow_doorway, exiting_elevator, following_human, parallel_traffic,
  perpendicular_traffic, robot_crowding

## Frame-based observations (examples)
Reference frames in:
- output/recordings/scenario_videos_classic_interactions_francis2023_socnav_social_force_20260123_112305/failure_frames

Key patterns observed:
- **Crossing scenarios**: collisions occur near the central obstacle/pillar; the trajectory curves into the obstacle rather than around it. Suggests insufficient static obstacle clearance in the reactive force field.
- **Overtaking/merging**: repeated early collisions at the narrow mid-corridor choke, indicating that the social-force dynamics are not reliably finding collision-free corridors under non-holonomic constraints.
- **Narrow doorway/hallway**: robot collides at the pinch points; suggests geometry clearance is too tight for the chosen forces or robot footprint.
- **Following/parallel traffic**: collisions happen when staying near pedestrians in a narrow corridor, likely due to unstable local minima or insufficient lateral clearance.
- **Robot crowding**: dense pedestrian cluster compresses around the robot; collision occurs under crowd pressure.

## Strengths vs weaknesses
Strengths:
- Completes a subset of interaction scenarios without termination issues.
- Provides a classical baseline for crowd interactions.

Weaknesses:
- High collision rate overall (over half of episodes).
- Fails across both static obstacle-heavy layouts and dense pedestrian cases.
- Particularly weak in narrow passage geometry and overtaking/merging archetypes.

## Benchmark readiness
- **Not benchmark‑ready as a strong baseline** given current collision rate.
- It can still serve as a *weak classical reference* if the benchmark explicitly includes a low-performing social-force baseline.
