# Flat-observation frame contract and per-arm audit (issue #9752)

Flat SocNav fields and their frames (producer `SocNavObservationFusion.next_obs`;
mirrored unchanged by `normalize_map_observation`):

| Flat field | Frame |
|---|---|
| `robot_position`, `pedestrians_positions`, `goal_current`/`goal_next` | world, meters |
| `robot_heading` | world CCW yaw, radians |
| `pedestrians_velocities` | **robot ego**, m/s (world rotated by `-heading`) |
| radii, speeds, counts, timesteps | frame-free scalars |

## Audited arms (code evidence, 2026-09-27)

| Arm | Nested path | Flat path | Evidence |
|---|---|---|---|
| hybrid_rule v3 | OK (converts via `_ego_velocity_to_world`) | **affected, ruled unchanged** (pass-through; author keeps v3 for comparison) | `hybrid_rule_local_planner.py:651-658`, characterization test below |
| hybrid_rule v4 | OK | OK (rotates flat too) | same site, `_v4_clearance_braking` branch |
| socnav_social_force | OK (always rotates via `_rotate_velocities_to_world`) | OK | `socnav_social_force.py:469` |
| risk_dwa | **affected** (no conversion; `_ttc_proxy` mixes ego `ped_vel` with world `robot_vel`) | **affected** | `risk_dwa.py:195` |

## Unaudited (follow-up)

`prediction_planner`, `predictive_mppi`, `orca` pedestrian inputs, `sacadrl`,
`socnav_sampling`, `guarded_ppo`/`ppo` (training vs eval frame). Each needs the
same read: does the arm convert ego velocities before world-frame use?

## Characterization coverage

`test_hybrid_v3_flat_observation_keeps_historical_velocity_frame` pins the ruled
v3-flat pass-through (ego in, ego out) and the nested conversion, so any future
behavior change fails loudly and deliberately.
