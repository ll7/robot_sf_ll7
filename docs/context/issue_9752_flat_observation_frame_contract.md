# Flat-observation frame contract and per-arm audit (issue #9752)

This note records the contract against main commit
(`a979a38316d8df2a9a0950825bd35d3687865c00`). The 14-arm roster at that revision is the
planner list in `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml`.

## Producer contract

`SocNavObservationFusion.next_obs` emits the following frame semantics. The
map-runner bridge `normalize_map_observation` mirrors the flat leaves without
changing their values or frames.

| Flat field | Frame |
|---|---|
| `robot_position`, `pedestrians_positions`, `goal_current`/`goal_next` | world, meters |
| `robot_heading` | world CCW yaw, radians |
| `pedestrians_velocities` | **robot ego**, m/s (world velocity rotated by `-heading`) |
| radii, speeds, counts, timesteps | frame-free scalars |

For a world-frame consumer, an ego velocity `(vx, vy)` is converted with
`R(heading) @ [vx, vy]`. Robot translation is not subtracted from a pedestrian
velocity.

## Current release-arm audit

“N/A” means the arm does not consume pedestrian velocities in the applicable
path. “Ego-native” means the velocity is part of a learned or local model input
whose documented frame is ego; it is not silently treated as a world vector.
The evidence column names source symbols and existing or added tests. These are
source and focused-test anchors, not campaign acceptance evidence.

| Current release arm | Velocity use | Nested observation | Flat observation | Source and test evidence |
|---|---|---|---|---|
| `prediction_planner` | Model input is ego-native | Ego-native | Ego-native | `socnav_prediction.py:PredictionPlannerAdapter._build_model_input`; `tests/planner/test_socnav_prediction_module.py` |
| `goal` | **N/A** | N/A | N/A | `map_runner.py:_goal_policy`; `tests/benchmark/test_map_runner_view_integrity.py::test_goal_reference_declares_pedestrian_blind` |
| `social_force` | World-frame force interaction | Converts ego to world | Converts ego to world | `socnav_social_force.py:SocialForcePlannerAdapter._rotate_velocities_to_world`; `tests/metamorphic/test_mirror_symmetry.py::test_release_arm_trace_is_mirror_and_rotation_equivariant` |
| `orca` | World-frame ORCA/heuristic interaction | Converts ego to world at the world-velocity boundary | Converts ego to world at the world-velocity boundary | `socnav_orca.py:ORCAPlannerAdapter._rvo2_velocity_world` and `_heuristic_velocity_world`; same metamorphic test |
| `ppo` (training) | Policy features preserve the SOCNAV observation contract | Ego input where structured SOCNAV is used | Ego input where flattened SOCNAV is used | `docs/dev/observation_contract.md` (SOCNAV frame semantics); `tests/baselines/test_ppo_planner.py` |
| `ppo` (release evaluation) | Checkpoint receives the producer fields; PPO does no world-frame pedestrian rollout | Ego input | Ego input | `baselines/ppo.py:PPOPlanner._build_model_obs_dict`; `tests/baselines/test_ppo_planner.py` |
| `socnav_sampling` | Release config leaves pedestrian prediction off; opt-in prediction converts | N/A in release default; opt-in converts | N/A in release default; opt-in converts | `socnav_sampling_v2.py:_pedestrian_world_velocities`; `tests/benchmark/test_issue_9727_socnav_sampling.py::test_approaching_pedestrian_blocks_earlier_with_prediction` |
| `sacadrl` | Network agent states use global pedestrian velocities | Converts ego to global | Converts ego to global | `socnav_sacadrl.py:SACADRLPlannerAdapter._ego_to_global_velocities`; `tests/planner/test_socnav_sacadrl_module.py::test_adapter_builds_network_input_and_agent_states` |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | Hybrid v3 base; one scenario override selects ORCA | Hybrid v3 converts | Hybrid v3 pass-through; ORCA override converts | `hybrid_rule_local_planner.py:HybridRuleLocalPlannerAdapter._extract_state`; candidate config; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | Hybrid v3 base; one scenario override selects ORCA | Hybrid v3 converts | Hybrid v3 pass-through; ORCA override converts | Same hybrid and ORCA symbols; candidate config; characterization test above |
| `hybrid_rule_v3_fast_progress_static_escape` | Hybrid v3 local rollout | Converts | **Historical pass-through retained** | `hybrid_rule_local_planner.py:HybridRuleLocalPlannerAdapter._extract_state`; characterization test above |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | Hybrid v3 local rollout | Converts | **Historical pass-through retained** | Same source and characterization test |
| `guarded_ppo` | PPO primary input stays ego-native; safety and fallback rollouts use world vectors | Converts for guard rollout | Converts for guard rollout | `guarded_ppo.py:GuardedPPOAdapter._extract_state`; `tests/planner/test_guarded_ppo.py::test_guarded_ppo_observation_rotates_pedestrian_velocity_to_world` |
| `predictive_mppi` | Learned predictor consumes ego-native velocity features; MPPI scores predicted positions | Ego-native predictor input | Ego-native predictor input | `socnav_prediction.py:PredictionPlannerAdapter._build_model_input`, `predictive_mppi.py:PredictiveMPPIAdapter._predict_future`; `tests/planner/test_predictive_mppi_planner.py` |
| `risk_dwa` | World-frame TTC and constant-velocity rollout | Converts ego to world | Converts ego to world | `risk_dwa.py:RiskDWAPlannerAdapter._extract_robot_goal_ped`; `tests/planner/test_risk_dwa.py::test_risk_dwa_observation_rotates_pedestrian_velocity_to_world`; `tests/metamorphic/test_mirror_symmetry.py::test_risk_dwa_release_trace_is_rotation_equivariant` |

## Scoped outcome and limitations

Risk-DWA and the Guarded-PPO world-frame safety path now perform the missing
ego-to-world conversion for both nested and flat observations. The PPO policy
input remains in its training frame.

Hybrid v3 flat-observation pass-through is intentionally unchanged so the
historical 0.0.7 comparison remains reproducible. It is the known 0.0.7 frame
defect and is carried into the 0.0.8 draft note. The two hybrid v3 trace
relations remain characterized as existing behavior, including their known
rotation/selection limitation.

The focused tests provide adapter-level frame evidence and bounded deterministic
metamorphic coverage. Stochastic learned policies and sampling arms do not
receive exact scene-trace claims here. No campaign result or per-arm release
acceptance claim is made by this note.
