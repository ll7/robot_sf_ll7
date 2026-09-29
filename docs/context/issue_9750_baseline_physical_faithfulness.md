# Issue #9750: 0.0.7 baseline physical-faithfulness audit

> Status (2026-09-29): source/config audit and deterministic unit-oracle inventory for the actual
> Release 0.0.7 roster. This is implementation-integrity evidence only. No planner campaign was
> run for this audit; it establishes no outcome, safety, ranking, or paper claim.

## Frozen evidence and shared physical contract

The audit uses the accepted Release 0.0.7 publication bundle, not its readiness matrix:

- Bundle: `issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922_publication_bundle.tar.gz`
- Bundle SHA-256: `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`
- Source commit: `07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- Resolved campaign config: `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml`
  (SHA-256 `095331329b06673dc165109c8523579549f769c98542b207a712f6e2bf9ed6ad`)
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml`
  (SHA-256 `03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c`)
- Campaign: 48 scenario identities, 30 seeds (`111`–`140`), horizon 600, environment `dt=0.1 s`,
  differential-drive kinematics, 14 roster slots, 20,160 planned episodes.

The source defaults resolved by that config are robot radius `1.0 m`, pedestrian radius `0.4 m`,
maximum linear/angular speed `2.0 m/s` / `1.0 rad/s`, linear/angular acceleration `1.0 m/s²` /
`1.0 rad/s²`, and braking deceleration `1.0 m/s²` (`max_linear_decel=None` resolves to the
acceleration value). These are the physical environment and actuator bounds. An internal planner
rollout step such as `0.2 s` is not the campaign control step `0.1 s`; release corrections below
bind physical-clearance and control-sensitive rollout settings explicitly.

The source defaults have known historical radius divergence: `GridConfig.robot_radius=0.3 m` and
the global-planner fallback radius `0.4 m` are not the physics/metrics robot radius. They are not
silently treated as the physical footprint in this audit. Observation fields `robot_radius` and
`pedestrians_radius` are defined in metres in [`observation_contract.md`](../dev/observation_contract.md).

## 0.0.8 release geometry binding

The actual 0.0.8 campaign template selects
`configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml`, as fixed by #9972.
The versioned planner configs below bind the physical robot and pedestrian radii and the drive
limits. A contract test resolves all 48 scenarios through the map-runner environment builder
without resetting or stepping them. The #9972 preflight, rather than this config test, checks
spawn and route feasibility. The frozen 0.0.7 config and scenario matrix remain untouched.

## Per-arm audit

“Correctness” denotes a mismatch that changes the claimed method or its physical collision
interpretation; it must be corrected in an explicitly versioned successor while the historical
default remains replayable. “Enhancement” denotes behavior absent from the reference method; it
stays off in the release baseline. A test anchor below is a hand-built, deterministic unit oracle,
not a release episode.

| Roster arm (0.0.7 or v4 replacement) | Actual config identity and resolved settings | Reference and observed deviation/classification | Unit oracle |
|---|---|---|---|
| `prediction_planner` | `configs/algos/prediction_planner_camera_ready.yaml`, SHA-256 `a5f3775110b7c1351e183016afe8a17348a66d35088eeb6ff55a182dc924a481`; speed `1.6`, angular `1.2`, rollout `0.2 s`, configured radii `0.25/0.25 m`. | Repository-defined predictor/scoring adapter; no external planner equivalence is asserted. Center-distance scoring plus `0.25/0.25 m` radii misstates the `1.0/0.4 m` bodies (**correctness**); angular cap exceeds the drive bound (**correctness**). Successor selects surface-distance clearance and the drive angular limit. | [`test_prediction_planner_surface_clearance_subtracts_both_body_radii_once`](../../tests/planner/test_predictive_mppi_planner.py) |
| `goal` | No `algo_config` in the resolved campaign; built-in `goal` alias to `simple_policy`. | Direct goal-seeking control is the declared naive comparator, not a collision-avoidance method. No footprint-aware avoidance claim is made; this is a comparator limitation, not a deviation from a collision-avoidance reference. Commands remain subject to environment speed/acceleration limits. | [`test_runner_goal_alias_executes_like_simple_policy`](../../tests/test_runner_smoke.py) |
| `social_force` | `configs/algos/social_force_terminal_goal_v1.yaml`, SHA-256 `bcd785fff7753dd31bcb899b1bab0cfec8d0c706b053ff28cd2988ca58761afb`; only `terminal_goal_v1` is explicit; wall, pedestrian, and command selectors otherwise resolve to legacy. Resolved planner speed cap is `3.0 m/s`. | Helbing–Molnár social force, pedestrian interaction Eqs. 8–12 ([paper](https://doi.org/10.1103/PhysRevE.51.4282)). Historical wall force sums raster cells and changes with grid resolution; historical pedestrian force evaluates center distance and ignores both body radii; legacy force-to-command mapping can exceed desired speed (**correctness**, #9724, #9758). Successor selects `resolution_independent_v2`, `surface_v3`, and the separately versioned wrapped angular kernel (#9764). | [`test_straight_wall_yields_one_exponential_term`](../../tests/test_issue_9724_social_force_resolution_independent.py), [`test_obstacle_force_is_invariant_when_grid_resolution_is_halved`](../../tests/test_issue_9724_social_force_resolution_independent.py), [`test_v3_brakes_for_close_head_on_pedestrian`](../../tests/planner/test_socnav_ped_surface_v3.py) |
| `orca` | No historical `algo_config`; defaults include speed `3.0 m/s`. The 0.0.8 template binds `configs/algos/socnav_release_v0_0_8.yaml` with `2.0/1.0` speed/turn caps. | ORCA reciprocal velocity-obstacle half-plane method ([paper, §4](https://doi.org/10.1007/978-3-642-19457-3_1)). The historical command cap exceeds the drive limit (**correctness**). Observation radii remain the geometry input; flat-observation velocity conversion is tracked by #9752. | [`test_orca_release_command_clips_three_metre_preference_to_drive_bound`](../../tests/planner/test_release_action_bounds.py) |
| `ppo` | `configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml`, SHA-256 `51ccfbf4400a306b355e2c3f0f46eda3489d5ce3bc85beaa023a6a1da9c9fb41`; action caps `2.0 m/s` / `1.0 rad/s`; predictive-feature cadence `0.2 s`. | PPO ([paper, §3](https://arxiv.org/abs/1707.06347)) with the registry-pinned trained checkpoint. Action caps match the drive. Keep the checkpoint's trained observation/feature schema and `0.2 s` predictive-feature cadence; changing them would change model inputs. The cadence is not the environment action `dt`. The policy is not claimed to provide an explicit clearance certificate. | [`test_vectorize_and_action_mapping_paths`](../../tests/baselines/test_ppo_planner.py) |
| `socnav_sampling` | No `algo_config`; legacy sampling selector and `SocNavPlannerConfig` speed default `3.0 m/s`. | SocNavBench path-distance sampling reference ([paper, §§4.1–4.2](https://arxiv.org/abs/2103.00047)). The historical additive pedestrian-repulsion vector has no counterpart in the reference sampler (**enhancement**; disable in release config). Default speed cap exceeds the drive bound (**correctness**). Versioned bounded sampling accounts for the robot footprint and drive/braking bounds. | [`test_goal_path_field_routes_through_the_gap`](../../tests/benchmark/test_issue_9727_socnav_sampling.py) |
| `sacadrl` | No historical `algo_config`; resolved speed default `3.0 m/s`. The 0.0.8 template binds `configs/algos/socnav_release_v0_0_8.yaml` with `2.0/1.0` speed/turn caps. | Chen et al., decentralized non-communicating collision avoidance with deep reinforcement learning ([paper](https://arxiv.org/abs/1609.07845)). Observation radii are passed to the model adapter; the historical command cap exceeds the drive bound (**correctness**). | [`test_sacadrl_release_command_clips_speed_and_turn_to_drive_bounds`](../../tests/planner/test_release_action_bounds.py) |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4` | [`scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4_s30_h600_release_0_0_8_frozen.yaml`](../../configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4_s30_h600_release_0_0_8_frozen.yaml), SHA-256 `28aafe6352543e4c4d24a146bd3ba7260105362052505af9308f6d45d9d94458`; v4 base `hybrid_rule_v4_clearance_braking.yaml`, SHA-256 `5dbc4f17a1ca8b44c2f78dc4e5c83e94a60615ebd7e97892109124323bc9c4ab`; speed cap `2.0`, drive-bound acceleration and angular cap, observation radii `1.0/0.4 m`. | Frozen v4 replacement uses surface clearance, drive-limited rollouts and braking checks; `francis2023_leave_group` resolves to the named ORCA override. Scenario yield thresholds are expressed in surface metres. | [`test_v4_rollout_follows_drive_braking_limit`](../../tests/planner/test_hybrid_rule_v4_clearance_braking.py) |
| `scenario_adaptive_hybrid_orca_v2_collision_guard_v4` | [`scenario_adaptive_hybrid_orca_v2_collision_guard_v4_s30_h600_release_0_0_8_frozen.yaml`](../../configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard_v4_s30_h600_release_0_0_8_frozen.yaml), SHA-256 `54435e7fd39e592d4684fe40f804ab1c080f39925dceb0781dd68c1ce172ef94`; same v4 base and drive/geometry settings as preceding row. | Frozen v4 collision-guard replacement retains its scenario override and ORCA hand-off. Surface clearance and braking are v4 method changes, with the historical v3 arm retained as comparison. | [`test_v4_braking_check_rejects_unstoppable_candidate`](../../tests/planner/test_hybrid_rule_v4_clearance_braking.py) |
| `hybrid_rule_v4_fast_progress_static_escape` | [`hybrid_rule_v4_fast_progress_static_escape_s30_h600_release_0_0_8_frozen.yaml`](../../configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_s30_h600_release_0_0_8_frozen.yaml), SHA-256 `1329d3934decb0bf50a803071af69b99e3f12e534083fc3d82e20d4706c76811`; same v4 base, `2.0 m/s` cap, observation body radii, drive-bound rollout. | Frozen v4 fast-progress replacement keeps its escape policy while evaluating candidate motion against the drive and physical clearance. | [`test_v4_rollout_follows_drive_braking_limit`](../../tests/planner/test_hybrid_rule_v4_clearance_braking.py) |
| `hybrid_rule_v4_fast_progress_static_escape_continuous` | [`hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release_0_0_8_frozen.yaml`](../../configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release_0_0_8_frozen.yaml), SHA-256 `044931a38f31a7b7c01ba8a750a1d34a79a58200309806e538761a94eff94888`; same v4 base plus hard-safety margin `0.2 m`, continuous static check and corridor subgoals. | Frozen v4 continuous replacement has explicit corridor behavior and exact obstacle geometry checks; these remain variant-specific behavior. | [`test_v4_braking_check_rejects_unstoppable_candidate`](../../tests/planner/test_hybrid_rule_v4_clearance_braking.py) |
| `guarded_ppo` | `configs/algos/guarded_ppo_camera_ready_cpu.yaml`, SHA-256 `69f273f311590009a344f3a88592cb19f54524d252f469ab5fc66f7cf2c9e772`; PPO caps `2.0/1.0`, guard and fallback rollout `0.2 s`; fallback DWA caps `0.9/1.2`. No explicit clearance model/radii. | Repository-defined safety wrapper around the trained PPO policy and DWA fallback. Fallback angular cap exceeds the drive bound, the guard uses center-distance clearance, and rollout cadence is `0.2 s` (**correctness**). Successor explicitly selects surface clearance with `1.0/0.4 m` radii and `0.1 s` rollouts. | [`test_guarded_ppo_uses_fallback_when_ppo_is_unsafe`](../../tests/planner/test_guarded_ppo.py), [`test_surface_v2_guard_uses_body_to_body_clearance`](../../tests/planner/test_guarded_ppo.py) |
| `predictive_mppi` | `configs/algos/predictive_mppi_camera_ready.yaml`, SHA-256 `5213a9e44c74cb79f78466d414645f6ca233d761e8fa5b77e5cfc3161ed94adb`; speed `1.5`, angular `1.3`, rollout `0.2 s`; nested predictor radii default to `0.3/0.3 m`, center-distance model. | Model predictive path integral control ([paper](https://doi.org/10.2514/1.G001921)). Body clearance and TTC are center-based or use undersized predictor radii, and angular limit exceeds the drive limit (**correctness**). Successor uses surface clearance and explicit `1.0/0.4 m` radii. | [`test_surface_v2_mppi_rejects_center_distance_that_overlaps_bodies`](../../tests/planner/test_predictive_mppi_planner.py) |
| `risk_dwa` | `configs/algos/risk_dwa_camera_ready.yaml`, SHA-256 `1351439539ed02334891f6a1ef6ac652745791b1b830a5f8616850fbd5ccb37c`; speed `1.2`, angular `1.2`, rollout `0.2 s`; center-distance model and default radii `0/0 m`. | Dynamic Window Approach ([paper](https://doi.org/10.1109/100.580977)). Angular limit exceeds the drive limit; center-distance pedestrian/occupied-cell margins and closest-approach TTC do not express physical contact (**correctness**, tracked by #9750). Successor uses circle-contact TTC, continuous-point occupied-square clearance, explicit `1.0/0.4 m` radii and a one-step drive-limited dynamic window. | [`test_release_dynamic_window_scores_only_next_step_reachable_commands`](../../tests/planner/test_risk_dwa.py), [`test_release_risk_dwa_sees_pedestrian_crossing_safety_boundary_at_one_second`](../../tests/planner/test_risk_dwa.py) |

## Versioned successor disposition

The implementation adds `center_v1` and `surface_v2` clearance modes. Omitting a selector retains
the historical center-distance path, including the serialized config shape; `surface_v2` fails
closed unless both body radii are finite and positive. Pedestrian contact uses the exact first
circle-intersection time. Occupancy clearance treats a marked cell as its square footprint and
subtracts the robot radius. This removes duplicate radius padding when the physical clearance is
already selected.

For `surface_v2`, the occupied-square distance now uses the continuous point
inside its grid cell and searches a metre-derived radius large enough to cover
the robot body plus the largest local safety margin. The 0.0.8 Risk-DWA profile
selects `drive_limited_v2`: sampled commands lie within one `0.1 s` drive step
at `1.0 m/s²` linear acceleration/braking and `1.0 rad/s²` angular acceleration.
The historical fixed lattice remains `fixed_v1` by default. The guarded PPO
fallback selects the same versioned window. The release rollouts keep their
historical durations in seconds by doubling step counts as cadence changes
from `0.2 s` to `0.1 s`, including nested and boosted predictor horizons.
The Risk-DWA crossing oracle enters the physical safety margin near `t=1.0 s`:
an `0.8 s` truncated rollout misses it; the restored `1.6 s` rollout rejects it.
These are deterministic contract checks, not episode or release evidence.

The 0.0.8 release template binds these corrections through new config files for the predictor, Guarded
PPO, MPPI, RiskDWA, SocialForce, SocNav/ORCA/SACADRL, and bounded SocNav sampling. The PPO learned
feature cadence remains `0.2 s` to preserve the registered checkpoint input contract while the
environment control step remains `0.1 s`. The four hybrid v4 slots remain bound to the frozen
#9748/#9751 configs from #9972. The obsolete v3 diagnostic twins are not used.

Relevant correctness findings are tracked in [#9724](https://github.com/ll7/robot_sf_ll7/issues/9724),
[#9742](https://github.com/ll7/robot_sf_ll7/issues/9742),
[#9727](https://github.com/ll7/robot_sf_ll7/issues/9727),
[#9746](https://github.com/ll7/robot_sf_ll7/issues/9746),
[#9758](https://github.com/ll7/robot_sf_ll7/issues/9758),
[#9764](https://github.com/ll7/robot_sf_ll7/issues/9764), and
[#9752](https://github.com/ll7/robot_sf_ll7/issues/9752). No 0.0.7 artifact, tag, or Zenodo record
was modified.

The separate standalone DWA grid-resolution defect is already tracked by [#9740](https://github.com/ll7/robot_sf_ll7/issues/9740),
with active [PR #9804](https://github.com/ll7/robot_sf_ll7/pull/9804); this audit does not duplicate
that implementation. [PR #9744](https://github.com/ll7/robot_sf_ll7/pull/9744) is the metamorphic
review that surfaced the DWA/body-radius follow-up now covered here by #9750.

## Validation and limits

Relevant deterministic unit oracles are the tests linked per arm above, plus the new physical
geometry tests in [`test_clearance_geometry.py`](../../tests/planner/test_clearance_geometry.py)
and the resolved 0.0.8 candidate contract in
[`test_planner_unit_consistency.py`](../../tests/metamorphic/test_planner_unit_consistency.py).
They check source-level behavior and config resolution only. They do not show that a release arm
avoids collision in campaign episodes. The actual campaign, new metrics/SNQI calibration, doorway
slice, and release candidate remain gated by #9668 and its dependencies.

**Quarantined validation incident:** an earlier focused pytest selection also executed the
`test_release_cell_replay_reaches_goal_without_wall_contact` parametrizations for both representative
scenarios on held-out seed 111 (bounded-v2, horizon 600). Both tests passed, but those two planner
episodes violate the current #9668 restriction on planner steps for seeds 111–140 before the SNQI
anchor gate. They are excluded from all release and dissertation claims and were disclosed in
[the #9668 custody comment](https://github.com/ll7/robot_sf_ll7/issues/9668#issuecomment-5881598950).
The test now uses diagnostic seed 103 and is explicitly named as a diagnostic replay; future
validation may run it under the allowed diagnostic-seed policy.
