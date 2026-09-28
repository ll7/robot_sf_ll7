# Flat-observation frame contract and per-arm audit (issues #9752 and #9845)

This crosswalk was verified against the 14-arm roster reached through
`configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml::canonical_campaign_config`,
which points to
`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml`. The roster
and four candidate files below were checked against current main
`6fb2cfdd9fdad68858c9559de0170a0e7aca5e3f`; these files are unchanged since main
`5d7aba11f7ae34fc88cd99620ba179d0ebfec635`. Candidate arms point to their own files under
`configs/policy_search/candidates/`. Source symbols and focused test anchors below were checked
against the combined PR branch with current main merged; they provide implementation-integrity
evidence only.

## Producer contract

`SocNavObservationFusion.next_obs` emits the nested SOCNAV observation; its flat wrapper preserves
the same leaves. Pedestrian world velocities are rotated by negative robot heading only. The
map-runner bridge `robot_sf/benchmark/map_runner/map_runner_observations.py::normalize_map_observation`
maps flat leaves to nested keys without changing values or frames; this is checked by
`tests/benchmark/test_map_runner_observations.py::test_normalize_map_observation_preserves_velocity_values_unchanged`.

| Field | Frame |
|---|---|
| `robot_position`, `pedestrians_positions`, `goal_current` / `goal_next` | world, meters |
| `robot_heading` | world counter-clockwise yaw, radians |
| `pedestrians_velocities` | robot ego, m/s: world velocity rotated by `-heading` |
| radii, speeds, counts, timesteps | frame-free scalars |

A world-frame consumer converts ego velocity with `R(heading) @ [vx, vy]`. Robot translational
velocity is not subtracted. The producer and flattening evidence is
`tests/test_socnav_observation.py::test_socnav_observation_rotates_pedestrian_velocities_to_ego_frame`
and `tests/test_socnav_observation.py::test_flat_socnav_observation_preserves_declared_frames`.

## Current release-arm crosswalk

“Ego-native” means the planner's learned/model feature contract accepts the producer's ego velocity.
“World conversion” means the adapter rotates the pedestrian vector before a world-frame calculation.
Test anchors establish implementation-integrity behavior only; they are not campaign evidence.

| Release arm | Velocity use and nested / flat behavior | Source and exact test anchors |
|---|---|---|
| `prediction_planner` | Learned predictor feature; ego-native in both paths. | `robot_sf/planner/socnav_prediction.py::PredictionPlannerAdapter._build_model_input`; `tests/planner/test_socnav_prediction_module.py::test_predictive_model_input_preserves_ego_pedestrian_velocity`; flat feature test at `tests/planner/test_socnav_prediction_module.py::test_flat_observation_pedestrian_velocity_stays_ego_native_in_model_input` |
| `goal` | **N/A**: does not consume pedestrian velocity. | `robot_sf/benchmark/map_runner/map_runner.py::_goal_policy`; `tests/benchmark/test_map_runner_view_integrity.py::test_goal_reference_declares_itself_pedestrian_blind` |
| `social_force` | World-frame force interaction; converts ego to world. | `robot_sf/planner/socnav_social_force.py::SocialForcePlannerAdapter._rotate_velocities_to_world`; `tests/metamorphic/test_mirror_symmetry.py::test_release_arm_trace_is_mirror_and_rotation_equivariant[social_force]` is a strict expected failure for the separately recorded branch-cut defect; it is not green rotation evidence. |
| `orca` | World-frame RVO2 and heuristic interactions; converts ego at each world-velocity boundary. | `robot_sf/planner/socnav_orca.py::ORCAPlannerAdapter._rvo2_velocity_world` and `robot_sf/planner/socnav_orca.py::ORCAPlannerAdapter._heuristic_velocity_world`; `tests/metamorphic/test_mirror_symmetry.py::test_release_arm_trace_is_mirror_and_rotation_equivariant[orca]` asserts flat leaves for the base and transformed map-runner episodes |
| `ppo` (training) | Training observations already carry ego pedestrian velocity; PPO features retain that frame. | `robot_sf/sensor/socnav_observation.py::SocNavObservationFusion.next_obs`; `tests/test_socnav_observation.py::test_socnav_observation_rotates_pedestrian_velocities_to_ego_frame`; `tests/test_socnav_observation.py::test_flat_socnav_observation_preserves_declared_frames` |
| `ppo` (release evaluation) | Structured leaves flatten into checkpoint keys unchanged; flat leaves pass through. No world-frame pedestrian rollout occurs in this path. | `robot_sf/baselines/ppo.py::PPOPlanner._flatten_nested_observation` and `robot_sf/baselines/ppo.py::PPOPlanner._build_model_obs_dict`; `tests/baselines/test_ppo_planner.py::test_build_model_obs_dict_flattens_structured_socnav_observation` checks structured and flat feature preservation at `heading=pi/2` |
| `socnav_sampling` | The release roster omits an arm `algo_config`, so `None` resolves to legacy v1; the default does not consume pedestrian velocity. An explicit bounded-v2 prediction opt-in rotates ego velocity to world. | `robot_sf/planner/socnav_base.py::resolve_socnav_sampling_version` and `SocNavPlannerConfig.sampling_pedestrian_prediction`; `robot_sf/planner/socnav_sampling_v2.py::_pedestrian_world_velocities`; `tests/benchmark/test_issue_9727_socnav_sampling.py::test_default_stays_legacy_and_release_config_opts_in` checks defaults/config, and `tests/benchmark/test_issue_9727_socnav_sampling.py::test_pedestrian_world_velocities_converts_ego_at_nonzero_heading` checks flat-frame conversion at `heading=pi/2`. No release-default rotation claim applies. |
| `sacadrl` | Converts ego velocity to world, then projects it into goal-parallel / goal-lateral network features. | `robot_sf/planner/socnav_sacadrl.py::SACADRLPlannerAdapter._ego_to_global_velocities` and `robot_sf/planner/socnav_sacadrl.py::SACADRLPlannerAdapter._build_network_input`; `tests/planner/test_socnav_sacadrl_module.py::test_network_input_converts_ego_velocity_before_goal_frame_projection`; flat conversion at `tests/planner/test_socnav_sacadrl_module.py::test_flat_observation_pedestrian_velocity_converts_to_global_frame` |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | Hybrid-v3 base converts nested ego velocity; flat v3 retains historical pass-through. Its `francis2023_leave_group` algorithm override selects ORCA, which converts to world. | `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`; ORCA override test below |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | Same hybrid-v3 base and ORCA override frame behavior as the preceding arm. | Same hybrid and ORCA sources; the same v3 characterization and ORCA override test below |
| `hybrid_rule_v3_fast_progress_static_escape` | Hybrid-v3 rollout converts nested ego velocity; flat v3 retains historical pass-through. | `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame` and `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`; `tests/metamorphic/test_mirror_symmetry.py::test_release_hybrid_v3_outcome_is_mirror_and_rotation_invariant` |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | Same hybrid-v3 frame path and historical flat defect. | Same hybrid source and characterization tests as the preceding arm; the tested v3 candidate's rotation outcome anchor is `tests/metamorphic/test_mirror_symmetry.py::test_release_hybrid_v3_outcome_is_mirror_and_rotation_invariant`. |
| `guarded_ppo` | PPO primary features remain ego-native. Guard and fallback rollouts convert ego velocities to world for both nested and flat inputs. | `robot_sf/planner/guarded_ppo.py::GuardedPPOAdapter._extract_state`; `tests/planner/test_guarded_ppo.py::test_guarded_ppo_observation_rotates_pedestrian_velocity_to_world`; `tests/metamorphic/test_mirror_symmetry.py::test_flat_velocity_rotation_equivariance_for_world_frame_rollouts[guarded_ppo]` |
| `predictive_mppi` | Shared learned predictor consumes ego-native velocity features; MPPI scores predictor positions in its robot-relative rollout. Sampling is stochastic, so use seeded replay rather than scene-trace equivalence. | `robot_sf/planner/predictive_mppi.py::PredictiveMPPIAdapter._predict_future`; shared flat predictor boundary at `tests/planner/test_socnav_prediction_module.py::test_flat_observation_pedestrian_velocity_stays_ego_native_in_model_input`; seeded replay test `tests/planner/test_predictive_mppi_planner.py::test_predictive_mppi_is_deterministic_for_fixed_seed` |
| `risk_dwa` | TTC and constant-velocity rollout are world-frame; converts ego to world for nested and flat inputs. `pedestrians_count` limits both arrays before rollout, including zero visible rows. | `robot_sf/planner/risk_dwa.py::RiskDWAPlannerAdapter._extract_robot_goal_ped`; `tests/planner/test_risk_dwa.py::test_risk_dwa_observation_rotates_pedestrian_velocity_to_world`; `tests/planner/test_risk_dwa.py::test_risk_dwa_ignores_padded_flat_rows_when_visible_count_is_zero`; real flat map-runner scene rotation at `tests/metamorphic/test_mirror_symmetry.py::test_risk_dwa_flat_release_trace_is_rotation_equivariant` |

## Scenario-adaptive override crosswalk

The four candidate files referenced by the roster are:
`configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_bottleneck_yield_s30_h600_release.yaml`,
`configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard_s30_h600_release.yaml`,
`configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape_s30_h600_release.yaml`,
and
`configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape_continuous_s30_h600_release.yaml`.
Their `scenario_overrides` change planner parameters only; none changes velocity fields, the selected
planner, or frame handling. Each parameter row therefore inherits the hybrid-v3 consumer and frame
characterization. A separate frame test per parameter row is N/A because the common
`HybridRuleLocalPlannerAdapter._extract_state` boundary performs the conversion before those
parameter values affect planning.

| Candidate arm | `scenario_overrides` entry | Frame and exact evidence |
|---|---|---|
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | `classic_bottleneck_high` | Hybrid v3: nested ego-to-world; historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| same | `francis2023_blind_corner` | Parameter-only; hybrid-v3 nested ego-to-world and historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| same | `francis2023_perpendicular_traffic` | Parameter-only; hybrid-v3 nested ego-to-world and historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `classic_merging_low` | Hybrid v3: nested ego-to-world; historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| same | `francis2023_blind_corner` | Parameter-only; hybrid-v3 nested ego-to-world and historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| same | `francis2023_perpendicular_traffic` | Parameter-only; hybrid-v3 nested ego-to-world and historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| `hybrid_rule_v3_fast_progress_static_escape` | `francis2023_blind_corner` | Hybrid v3: nested ego-to-world; historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| same | `francis2023_perpendicular_traffic` | Parameter-only; hybrid-v3 nested ego-to-world and historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | `francis2023_blind_corner` | Hybrid v3: nested ego-to-world; historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| same | `francis2023_perpendicular_traffic` | Parameter-only; hybrid-v3 nested ego-to-world and historical flat pass-through. `robot_sf/planner/hybrid_rule_local_planner.py::HybridRuleLocalPlannerAdapter._extract_state`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame`; `tests/planner/test_hybrid_rule_local_planner.py::test_hybrid_v3_flat_observation_keeps_historical_velocity_frame`. |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | `francis2023_leave_group` in `scenario_algo_overrides` | Effective algorithm is ORCA; converts ego to world. `robot_sf/benchmark/policy_search_manifest.py::resolve_candidate_manifest_runtime`; `tests/metamorphic/test_mirror_symmetry.py::test_release_scenario_orca_override_trace_is_rotation_equivariant[scenario_adaptive_hybrid_orca_v2_bottleneck_yield]`. |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `francis2023_leave_group` in `scenario_algo_overrides` | Effective algorithm is ORCA; converts ego to world. `robot_sf/benchmark/policy_search_manifest.py::resolve_candidate_manifest_runtime`; `tests/metamorphic/test_mirror_symmetry.py::test_release_scenario_orca_override_trace_is_rotation_equivariant[scenario_adaptive_hybrid_orca_v2_collision_guard]`. |

The ORCA override episode test asserts both release candidate configs resolve to `orca` and retain
their configured symmetry/head-on bias values before comparing the transformed command and pose
traces. The test also asserts that reset, step, and rotated observations expose flat SOCNAV leaves.
It uses a synthetic in-memory scene and proves implementation behavior only.

## PPO training and evaluation comparison

Training's SOCNAV producer rotates pedestrian world velocity by `-heading` in
`SocNavObservationFusion.next_obs`; its structured and flattened observation paths preserve this
input contract. Release evaluation's `PPOPlanner._flatten_nested_observation` promotes nested
fields to model keys and `_build_model_obs_dict` aligns their shapes and dtypes without another
rotation. The producer tests above and
`tests/baselines/test_ppo_planner.py::test_build_model_obs_dict_flattens_structured_socnav_observation`
verify the two sides of that boundary. This establishes matching feature frames, not checkpoint
training provenance or policy action equivalence.

## Outcome and limitations

Risk-DWA previously retained padded position and velocity rows when the declared visible count was
zero. `_extract_robot_goal_ped` now slices both arrays to the count before the rollout score sees
them; the negative test supplies four nonzero padded rows and verifies empty arrays reach scoring.
This is a focused input-contract regression. No rollout impact has been measured.

The historical hybrid-v3 flat-observation pass-through remains unchanged and is the known 0.0.7
frame defect. The focused characterization test pins both the nested conversion and the flat
pass-through. Under the #9668 ruling, each affected 0.0.8 arm needs an explicitly versioned
corrected hybrid path selected and validated before release admission; the historical v3 path
remains available for the 0.0.7 comparison.

The Risk-DWA closed-loop episode uses map-runner's occupancy-grid-enabled flattened SOCNAV output;
the test harness asserts flat pedestrian keys are present at reset and after every step. Its
90-degree trace relation therefore exercises the real flat planner path, while the direct negative
case isolates count/padding normalization. `run_arm_episode` records this observation shape as
`ArmEpisode.flat_socnav_observation`; the ORCA release episode, both ORCA scenario overrides, and
Risk-DWA assert the flag for baseline and rotated episodes.

Social Force's current rotation case is a strict expected failure for its separately recorded
branch-cut defect; it is not counted as passing evidence. Learned and stochastic arms receive
feature-boundary or seeded-replay evidence only. The deterministic ORCA override test and Risk-DWA
flat trace test use real planner episodes over rotated synthetic scenes. These tests establish no
campaign, release-row, safety, performance, or paper-facing claim. No campaign was run and no
campaign artifact was produced.

## Crosswalk corrections and learned-feature boundary

The earlier Goal anchor `test_goal_reference_declares_pedestrian_blind` did not exist; the exact
test is `tests/benchmark/test_map_runner_view_integrity.py::test_goal_reference_declares_itself_pedestrian_blind`.
This arm is pedestrian-velocity-blind, so its velocity frame is `N/A`.

The `SOCNAV_STRUCT` selector in the metamorphic environment setup does not mean the map runner
hands the planner a nested observation: occupancy-grid mode emits flat SOCNAV keys. The episode
helper now records the actual keys it sees, and the applicable 90-degree episode tests assert that
shape. `robot_sf/benchmark/map_runner/map_runner.py::_build_common_adapter_policy` passes the same
observation dictionary directly to `adapter.plan`. The Social Force case remains a strict expected
failure for its known branch-cut defect (tracked separately as #9764); it is not counted as passing
rotation evidence.

The flat-path feature tests use nonzero heading where conversion is under test: the prediction
model retains ego-native features, SACADRL converts ego velocity before global/goal projection, and
the sampling opt-in converts ego velocity for world-frame prediction. The release-default sampling
configuration disables pedestrian prediction, so no velocity-rotation claim applies to that arm.
Predictive MPPI is covered by the shared flat predictor boundary plus seeded replay; no exact
scene-command trace claim is made for the stochastic path.

For PPO, training's observation producer rotates world pedestrian velocity to ego frame, while
release evaluation flattens structured leaves or passes flat leaves through without another
rotation. The producer-frame tests and PPO model-input test at nonzero heading cover both sides.
They establish a matching feature-frame contract, not checkpoint provenance or action equivalence.

## Validation record

The required focused command over `tests/metamorphic/test_mirror_symmetry.py`,
`tests/planner/test_risk_dwa.py`, and `tests/planner/test_guarded_ppo.py` reported 216 passed and
3 expected xfails on the merged head. The prediction, SACADRL, and sampling test modules reported
70 passed; eight additional producer, normalizer, Goal, hybrid-v3 characterization, PPO, and replay
anchors reported 8 passed. Ruff check and format passed for all changed Python files, and
`git diff --check` passed. The PR readiness command stopped during collection because the worktree
environment lacks `imageio_ffmpeg` for the unrelated
`tests/analysis_workbench/test_audit_materialize.py`; full readiness did not complete. These are
implementation-integrity/smoke results, not campaign evidence. All 34 referenced paths and 83 exact
source/config/test anchors resolved; the 14 roster arms and 12 candidate override keys also
resolved against the current checkout.
