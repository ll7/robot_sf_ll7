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
| `sacadrl` | No historical `algo_config`; resolved speed default `3.0 m/s`. The 0.0.8 template binds `configs/algos/socnav_release_v0_0_8.yaml` with `2.0/1.0` speed/turn caps, preferred speed `2.0 m/s` and 19 observed-agent slots (issue #10007). Heading choices saturate under the drive rate limit: [FXS adaptation note](issue_10007_fxs_sacadrl_sampling.md). | Chen et al., decentralized non-communicating collision avoidance with deep reinforcement learning ([paper](https://arxiv.org/abs/1609.07845)). Observation radii are passed to the model adapter; the historical command cap exceeds the drive bound (**correctness**). | [`test_sacadrl_release_command_clips_speed_and_turn_to_drive_bounds`](../../tests/planner/test_release_action_bounds.py) |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4` | [`scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4_s30_h600_release_0_0_8_frozen.yaml`](../../configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4_s30_h600_release_0_0_8_frozen.yaml), SHA-256 `28aafe6352543e4c4d24a146bd3ba7260105362052505af9308f6d45d9d94458`; v4 base `hybrid_rule_v4_clearance_braking.yaml`, SHA-256 `5dbc4f17a1ca8b44c2f78dc4e5c83e94a60615ebd7e97892109124323bc9c4ab`; speed cap `2.0`, drive-bound acceleration and angular cap, observation radii `1.0/0.4 m`. | Frozen v4 replacement uses surface clearance, drive-limited rollouts and braking checks; `francis2023_leave_group` resolves to the named ORCA override. Scenario yield thresholds are expressed in surface metres. | [`test_v4_rollout_follows_drive_braking_limit`](../../tests/planner/test_hybrid_rule_v4_clearance_braking.py) |
| `scenario_adaptive_hybrid_orca_v2_collision_guard_v4` | [`scenario_adaptive_hybrid_orca_v2_collision_guard_v4_s30_h600_release_0_0_8_frozen.yaml`](../../configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard_v4_s30_h600_release_0_0_8_frozen.yaml), SHA-256 `54435e7fd39e592d4684fe40f804ab1c080f39925dceb0781dd68c1ce172ef94`; same v4 base and drive/geometry settings as preceding row. | Frozen v4 collision-guard replacement retains its scenario override and ORCA hand-off. Surface clearance and braking are v4 method changes, with the historical v3 arm retained as comparison. | [`test_v4_braking_check_rejects_unstoppable_candidate`](../../tests/planner/test_hybrid_rule_v4_clearance_braking.py) |
| `hybrid_rule_v4_fast_progress_static_escape` | [`hybrid_rule_v4_fast_progress_static_escape_s30_h600_release_0_0_8_frozen.yaml`](../../configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_s30_h600_release_0_0_8_frozen.yaml), SHA-256 `1329d3934decb0bf50a803071af69b99e3f12e534083fc3d82e20d4706c76811`; same v4 base, `2.0 m/s` cap, observation body radii, drive-bound rollout. | Frozen v4 fast-progress replacement keeps its escape policy while evaluating candidate motion against the drive and physical clearance. | [`test_v4_rollout_follows_drive_braking_limit`](../../tests/planner/test_hybrid_rule_v4_clearance_braking.py) |
| `hybrid_rule_v4_fast_progress_static_escape_continuous` | [`hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release_0_0_8_frozen.yaml`](../../configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release_0_0_8_frozen.yaml), SHA-256 `044931a38f31a7b7c01ba8a750a1d34a79a58200309806e538761a94eff94888`; same v4 base plus hard-safety margin `0.2 m`, continuous static check and corridor subgoals. | Frozen v4 continuous replacement has explicit corridor behavior and exact obstacle geometry checks; these remain variant-specific behavior. | [`test_v4_braking_check_rejects_unstoppable_candidate`](../../tests/planner/test_hybrid_rule_v4_clearance_braking.py) |
| `guarded_ppo` | `configs/algos/guarded_ppo_camera_ready_cpu.yaml`, SHA-256 `69f273f311590009a344f3a88592cb19f54524d252f469ab5fc66f7cf2c9e772`; PPO caps `2.0/1.0`, guard and fallback rollout `0.2 s`; fallback DWA caps `0.9/1.2`. No explicit clearance model/radii. | Repository-defined safety wrapper around the trained PPO policy and DWA fallback. Fallback angular cap exceeds the drive bound, the guard uses center-distance clearance, and rollout cadence is `0.2 s` (**correctness**). Successor explicitly selects surface clearance with `1.0/0.4 m` radii and `0.1 s` rollouts. | [`test_guarded_ppo_uses_fallback_when_ppo_is_unsafe`](../../tests/planner/test_guarded_ppo.py), [`test_surface_v2_guard_uses_body_to_body_clearance`](../../tests/planner/test_guarded_ppo.py) |
| `predictive_mppi` | `configs/algos/predictive_mppi_camera_ready.yaml`, SHA-256 `5213a9e44c74cb79f78466d414645f6ca233d761e8fa5b77e5cfc3161ed94adb`; speed `1.5`, angular `1.3`, rollout `0.2 s`; nested predictor radii default to `0.3/0.3 m`, center-distance model. | Model predictive path integral control ([paper](https://doi.org/10.2514/1.G001921)). Body clearance and TTC are center-based or use undersized predictor radii, and angular limit exceeds the drive limit (**correctness**). Successor uses surface clearance and explicit `1.0/0.4 m` radii. | [`test_surface_v2_mppi_rejects_center_distance_that_overlaps_bodies`](../../tests/planner/test_predictive_mppi_planner.py) |
| `risk_dwa` | `configs/algos/risk_dwa_camera_ready.yaml`, SHA-256 `1351439539ed02334891f6a1ef6ac652745791b1b830a5f8616850fbd5ccb37c`; speed `1.2`, angular `1.2`, rollout `0.2 s`; center-distance model and default radii `0/0 m`. | Dynamic Window Approach ([paper](https://doi.org/10.1109/100.580977)). Angular limit exceeds the drive limit; center-distance pedestrian/occupied-cell margins and closest-approach TTC do not express physical contact (**correctness**, tracked by #9750). Successor uses circle-contact TTC, continuous-point occupied-square clearance, explicit `1.0/0.4 m` radii and a one-step drive-limited dynamic window. | [`test_release_dynamic_window_scores_only_next_step_reachable_commands`](../../tests/planner/test_risk_dwa.py), [`test_release_risk_dwa_horizon_scoring_sees_pedestrian_crossing_at_one_second`](../../tests/planner/test_risk_dwa.py) |

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
fallback selects the same versioned window. Risk-DWA and Guarded PPO keep their
historical durations by doubling step counts at `0.1 s`. Checkpoint-backed
prediction and MPPI now use the source-proven training cadence: eight forecast
steps at `0.1 s`, giving `0.8 s`. MPPI requests 24 optimizer steps at `0.1 s`,
retaining the pre-FX2 `604794af` optimizer contract (24 × 0.1 s requested),
but `_predict_future` caps execution to the eight available decoder outputs.
Extending or interpolating forecasts beyond the learned window would invent
prediction evidence. Thus the effective optimizer window is also `0.8 s`.
Adaptive horizon requests remain capped by those outputs. Risk-DWA and Guarded
PPO durations/cadence are unchanged by FX3. The environment executes at `0.1 s`.
[`test_release_checkpoint_effective_window_is_zero_point_eight_seconds`](../../tests/planner/test_release_horizons.py)
loads each actual registry-pinned checkpoint without fallback. The retained
crossing oracle now scores `t=0.5 s`, with a `0.3 s` truncation negative control.
It claims scoring cadence, not that the learned model predicts that trajectory.
The Risk-DWA crossing oracle enters the physical safety margin near `t=1.0 s`:
an `0.8 s` truncated rollout misses it; the restored `1.6 s` rollout rejects it.
The Risk-DWA test exercises horizon scoring for a prescribed `0.5 m/s` command,
not the from-rest dynamic-window selector. These are deterministic contract checks,
not episode or release evidence.

### Declared emergency fallback and grid-edge convention

When every Risk-DWA candidate scores `-inf`, no admissible DWA command exists.
The adapter retains the one-step reachable maximum brake, including residual
forward/turn velocity when the drive cannot stop immediately. This is a declared
emergency actuator fallback, a deviation from admissible DWA command selection;
it does not certify a collision-free trajectory. Diagnostics expose the per-plan
`no_admissible_command` flag and cumulative `no_admissible_command_count`, also
copied into the simulation step trace's `planner` block. Reaching the goal or a
later admissible plan clears the flag. The direct Risk-DWA adapter owns these
fields; Guarded PPO now forwards its fallback adapter diagnostics as well.
[`test_release_all_rejected_commands_report_emergency_brake_and_reset_flag`](../../tests/planner/test_risk_dwa.py)
pins a `0.7/0.4` moving state and overlapping occupied cell: the reachable brake
is `0.6/0.3`, with the fallback flag set.

Grid edges preserve the existing convention, which is **occupied outside**, not
free: `OccupancyAwarePlannerMixin._world_to_grid` returns no index strictly outside,
`_grid_value` returns `1.0`, and the three surface-clearance adapters return
`-robot_radius`. Exactly at `origin + size`, indexing clamps to the last cell.
An empty grid therefore returns infinite clearance at the exact edge, despite
body overlap beyond it; no virtual boundary obstacle is added. This discontinuity
is a declared representation limitation, unchanged by this repair.
[`test_grid_upper_edge_is_clamped_and_outside_is_occupied_characterization`](../../tests/planner/test_surface_geometry_adversarial.py)
pins the edge, an epsilon beyond it, and an occupied last cell for all three arms.
It passes on the pre-fix head and is characterization evidence.

### FX3 training cadence provenance and static geometry

At `cef93136b92ddca9b0c4436bc44049412461a2fd`,
`scripts/training/collect_predictive_planner_data.py:185-186` indexes consecutive
frames (`frames[t+k]`), `:281` constructs `RobotSimulationConfig()`, and `:309-311`
records a frame and steps the environment exactly once. At that same commit,
`robot_sf/sim/sim_config.py:18` sets `time_per_step_in_secs=0.1`.
The eight decoder targets therefore represent `0.8 s`. The checkpoint payloads
lack an embedded duration; the source collection contract, rather than the old
`0.2 s` inference default, supplies its provenance. No checkpoint bytes change.

Risk-DWA, Guarded PPO and predictive MPPI bind static polygons and wall/boundary
segments through the existing benchmark `bind_env` hook after each reset.
The source is `RobotEnv._get_static_grid_obstacles()`, exactly the geometry used
for occupancy-grid generation, including compound polygon holes. No pedestrian,
future, waypoint or policy information enters this evaluator, and PPO model
observations retain their trained schema. Mathematical distance to these
primitives has **zero raster discretization error**; GEOS uses double precision.
The executable independent face-distance checks allow **1e-7 m numerical error**
on the release maps (coordinates below 100 m). This is a tested floating-point
budget, not a global GEOS precision theorem. No margin or cell correction is
subtracted. Distances are signed inside solid polygons; motion between rollout
samples is checked as a swept line segment plus the circular body radius.
Unbound standalone/grid-only observations retain the conservative occupied-square
fallback. The grid-edge characterization above applies to that fallback, not to
an exact bound map: the bound map's physical wall segments define its boundary.

For a positive current obstacle clearance below `0.3 m`, Risk-DWA and MPPI admit
only rollouts with strictly positive clearance that never falls below the current
clearance. At-rest rotation qualifies for a circular body. Otherwise the full
hard margin applies. MPPI retains `0.35 m` first-step padding when the current state meets it.
With bound exact geometry and a supported native drive, a state already below
that padding instead requires strictly positive, nondecreasing first-step
clearance. The independent `0.30 m` whole-rollout hard gate remains intact;
below that hard margin, the existing whole-rollout monotone recovery rule applies.
This avoids the TDIAG2 `0.30–0.35 m` dead band without lowering either threshold: a
centered `2.6 m` corridor can retain its hard margin rather than demanding an
unreachable instantaneous clearance increase. At an exactly `2.7 m` corridor the first-step threshold is a floating
point boundary, conservatively rejected if rounded below `0.35 m`.
MPPI explicitly scores a zero-speed recovery turn because continuous CEM samples
almost never contain one, and chooses it over a stationary recovery winner.
The forward progress escape still goes through the same hard constraints.

Guarded PPO keeps veto authority. Best-effort ranking uses pedestrian clearance,
then static clearance, then progress; on a complete tie it prefers a feasible
nonzero recovery turn to stasis. A penetrating fallback cannot be selected in
surface mode. Per-plan no-admissible flags/counts and recovery counts are exposed
in runtime diagnostics and step traces, including Guarded PPO's fallback details.
[`test_fx3_static_recovery.py`](../../tests/planner/test_fx3_static_recovery.py)
uses real release configs, registered MPPI checkpoint bytes, production SVG
parsing/rasterization, TDIAG coordinates, three headings and three grid phases.
The width controls procedurally vary the named wall faces in the real corridor
map to 1.9/2.0/2.6/2.7 m and include penetrating poses; no tracked fixture is
rewritten; the release configs and checkpoint bytes remain the actual inputs.
No episode sweep runs on this workstation. The orchestrator owns the final-head
48 × 14 dev-seed Slurm sweep; its results will be diagnostic development
evidence, not release admission. The earlier class labels were superseded by TDIAG2; its class f now has the
native-drive shield correction described below.

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

## FX6 TDIAG2 physical prediction and arbitration repairs

The surface-clearance guard binds the actual `DifferentialDriveSettings` and
uses FX4's native velocity-to-acceleration conversion, wheel initialization,
trapezoidal odometry and midpoint heading for PPO, fallback and stop. Native
limits remain `2.0 m/s`, no reverse and the bound accel/decel/yaw authority.
The stop forecast includes residual translation and the full terminal braking
coast, including when stopping exceeds the configured rollout horizon. MPPI's
bound surface-clearance rollouts reuse the same native implementation in nominal
states as well as recovery. Legacy center-distance dynamics remain replayable.
The braking tail checks static geometry; the pedestrian prediction horizon and
all hard/first-step pedestrian and TTC thresholds remain unchanged. An emergency
brake from an already inevitable-contact state is reported unsafe, never certified.

In empty worlds with bound exact geometry, guard best-effort arbitration compares
fallback with stop, excluding vetoed PPO/prior proposals. A below-margin fallback
must satisfy the same strictly positive, nondecreasing swept-clearance recovery
contract, now evaluated using native motion and its braking tail. With pedestrians
or unbound geometry, FX4/FX5's strict pedestrian-clearance arbitration is retained.
Fallback scoring, checkpoint inputs and learned feature cadence are unchanged.

[`test_fx6_tdiag2_repairs.py`](../../tests/planner/test_fx6_tdiag2_repairs.py) uses
TDIAG2 dev-seed 1001/1002 poses, actual release maps and configs: ten contact
states, seven dead-band states and the three exact-target forward-recovery
arbitration errors. These are deterministic command/geometry regressions, not
new episode outcomes or a guarantee of global progress.

### Documented limitations

TDIAG2 classified 54 empty-world failures at sweep head `2e6f8171`, checked against
`cb3cf2ee`. Counts below describe those historical traces, not FX6 outcomes.

- **a (six):** the Francis narrow doorway is exactly `2.0 m`, equal to the body's
  diameter, and smaller than the `2.6 m` required for two-sided hard margins.
  Refusal is correct; widening the map or shrinking radius/margins changes the comparison.
- **c (six):** cross-trap/doorway MPPI stasis has admissible commands. Bounded
  local search and re-evaluation of a sampled winner's first action as a constant
  sequence can favor stop near corners. Candidate ledgers were not retained, so
  no unique scoring defect is proven. This arbitration is a repository method
  deviation if claiming textbook MPPI equivalence; no such equivalence is established.
- **d recovery (three):** cross-trap seed 1001 at all densities admits in-place
  recovery turns, but the direct target-bearing turn approaches zero yaw at
  `(12.788823,17.725279,2.125855)`. Positive forward motion initially reduces
  clearance. Monotone local recovery need not make progress; reverse, temporary
  clearance loss or global rerouting would be separately declared enhancements.
- **e remaining scoring limits:** fixing the three demonstrated forward-fallback
  rejections does not promise to resolve the other fifteen guard deadlocks.
  Fourteen exact-target fallback probes already selected stop; entering-elevator's
  sampled final target was unavailable. Arbitrary escape turns are outside this repair.
- **h (four):** Risk-DWA station-platform seeds 1001/1002 require at least
  `62.014497/61.345231 s` at its unchanged `1.2 m/s` cap, exceeding the `60 s`
  horizon even before acceleration/bends. Guarded crowd-navigation exhibits
  PPO/intervention sensitivity without a uniquely established cause; guarded
  station-platform overshoots a middle waypoint and tries to return. These are
  horizon, learned-trajectory and waypoint-capture limits; no failed trace entered
  its goal zone. Speed/horizon/waypoint tuning is not a physical-faithfulness repair.

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
