# Lane HZN: effective horizon audit

Inspected main `cf67fa7ef5467d964c90d77289679e1780e50c99` in `~/hzn`; release source `07f7e8d43084de748915e1b1eb8b2a1603357c6e`. Diagnostic analysis of recorded 0.0.7 outcomes and static 0.0.8 configuration; no held-out episodes or counterfactual outcomes. Source bundle SHA-256 `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f` verified locally. Auditor producer `6196fd1013d1afc81f5256db2ab6be0415ecef52`; all 48 after-scan shard digests are in `evidence/issue_9668_campaign_horizon/source_manifest.json`.

## 1. Precedence and timeout labels

On main, campaign `horizon: 600` controls the runner loop; the scenario limit independently controls simulator termination. The effective budget is the smaller limit (for this release dt=0.1, no timestep mismatch). A collision or completion may of course end an episode earlier.

File:line evidence on the inspected main SHA:

- `robot_sf/benchmark/camera_ready/campaign.py:643` selects planner horizon override or campaign horizon; `:561` passes it to `run_batch`. Subprocess execution forwards the same horizon at `robot_sf/benchmark/camera_ready/resource_lifecycle.py:249`.
- `robot_sf/benchmark/map_runner/map_runner_episode.py:1427` builds the environment from the scenario; `:1437`–`:1442` independently resolves the runner horizon, without applying that horizon to the simulator; `:3481` loops over `range(horizon_val)`.
- `robot_sf/benchmark/map_runner/map_runner_env.py:32` delegates environment construction to the scenario loader. `robot_sf/training/scenario_loader.py:2927`–`:2929` converts scenario `max_episode_steps` to simulator seconds using the simulator timestep.
- `robot_sf/gym_env/robot_env.py:638`–`:639` binds timestep and duration into `RobotState`. `robot_sf/robot/robot_state.py:61`–`:63` computes `ceil(sim_time_limit/d_t)`; `:125` marks timeout when timestep reaches that limit; `:73` makes timeout terminal; `:168` exposes `is_timesteps_exceeded`.
- `robot_sf/gym_env/robot_env.py:1180` takes `state.is_terminal` as termination and returns Gymnasium termination (`:1223`–`:1228`, `truncated=False`). `robot_sf/benchmark/map_runner/map_runner_episode.py:3282` records the timeout flag and `:3317` stops on termination. `robot_sf/benchmark/termination_reason.py:255`–`:261` labels an otherwise-unspecified terminal event `terminated`; `map_runner_episode.py:4979`–`:4988` independently sets `timeout_event` from the simulator flag.

**A 400-step simulator timeout is recorded as `timeout_event: true`, `termination_reason: terminated`, with `horizon: 600`. It is not labelled as the runner reaching its H600 horizon (`max_steps`).** Downstream generic timeout rates pool it with timeouts at other limits; the published horizon field alone cannot distinguish them.

## 2. Published 0.0.7 impact

**1,505/20,160 rows (7.465%) are flagged, all early timeouts: 1,057 at step 400 and 448 at step 500.** All-outcome exposure is **15,960/20,160 (79.167%)**: 25 scenarios × 420 = 10,500 rows at effective H400; 13 × 420 = 5,460 at H500. The other 10 scenarios have effective H600 (eight authored 600, one 650, one 700). Every scenario/arm cell has 30 rows; no duplicate or missing release row was dropped.

The 15,960 shorter-budget rows comprise **8,362 completions, 6,093 collisions, and 1,505 timeouts**. These are budgets, not observed episode lengths; a success in fewer than 400 steps still belongs to the H400 budget denominator.

Full scenario × arm counts, including all outcomes and zero-flag cells, are in `evidence/issue_9668_campaign_horizon/release_0_0_7_cells.csv`. Arm key for the table below:

- A1: `goal` — 0 flagged timeouts.
- A2: `guarded_ppo` — 503 flagged timeouts.
- A3: `hybrid_rule_v3_fast_progress_static_escape` — 46 flagged timeouts.
- A4: `hybrid_rule_v3_fast_progress_static_escape_continuous` — 68 flagged timeouts.
- A5: `orca` — 33 flagged timeouts.
- A6: `ppo` — 28 flagged timeouts.
- A7: `prediction_planner` — 18 flagged timeouts.
- A8: `predictive_mppi` — 278 flagged timeouts.
- A9: `risk_dwa` — 50 flagged timeouts.
- A10: `sacadrl` — 0 flagged timeouts.
- A11: `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` — 46 flagged timeouts.
- A12: `scenario_adaptive_hybrid_orca_v2_collision_guard` — 46 flagged timeouts.
- A13: `social_force` — 386 flagged timeouts.
- A14: `socnav_sampling` — 3 flagged timeouts.

Flagged timeout counts by scenario and arm. Every listed scenario has 420 all-outcome rows with effective horizon <600 (30 per arm).

| Scenario | Limit | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 | A12 | A13 | A14 | Flagged total | All outcomes <600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| classic_bottleneck_high | 500 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 420 |
| classic_bottleneck_low | 500 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 30 | 420 |
| classic_bottleneck_medium | 500 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 420 |
| classic_doorway_high | 500 | 0 | 6 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 38 | 420 |
| classic_doorway_low | 500 | 0 | 19 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 51 | 420 |
| classic_doorway_medium | 500 | 0 | 13 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 30 | 0 | 47 | 420 |
| classic_group_crossing_high | 500 | 0 | 27 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 58 | 420 |
| classic_group_crossing_low | 500 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 39 | 420 |
| classic_group_crossing_medium | 500 | 0 | 22 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 54 | 420 |
| classic_head_on_corridor_low | 500 | 0 | 6 | 0 | 0 | 0 | 2 | 0 | 25 | 9 | 0 | 0 | 0 | 12 | 0 | 54 | 420 |
| classic_head_on_corridor_medium | 500 | 0 | 18 | 0 | 0 | 0 | 1 | 0 | 25 | 4 | 0 | 0 | 0 | 5 | 0 | 53 | 420 |
| classic_t_intersection_low | 500 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 420 |
| classic_t_intersection_medium | 500 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 420 |
| francis2023_accompanying_peer | 400 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 41 | 420 |
| francis2023_blind_corner | 400 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 420 |
| francis2023_circular_crossing | 400 | 0 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 23 | 0 | 36 | 420 |
| francis2023_crowd_navigation | 400 | 0 | 29 | 0 | 0 | 0 | 3 | 0 | 21 | 2 | 0 | 0 | 0 | 2 | 0 | 57 | 420 |
| francis2023_down_path | 400 | 0 | 30 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 55 | 420 |
| francis2023_entering_elevator | 400 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 31 | 420 |
| francis2023_entering_room | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 30 | 420 |
| francis2023_exiting_elevator | 400 | 0 | 29 | 0 | 4 | 0 | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 39 | 420 |
| francis2023_exiting_room | 400 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 58 | 420 |
| francis2023_following_human | 400 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 420 |
| francis2023_frontal_approach | 400 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 29 | 22 | 0 | 0 | 0 | 1 | 0 | 53 | 420 |
| francis2023_intersection_no_gesture | 400 | 0 | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0 | 0 | 13 | 420 |
| francis2023_intersection_proceed | 400 | 0 | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0 | 0 | 13 | 420 |
| francis2023_intersection_wait | 400 | 0 | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0 | 0 | 13 | 420 |
| francis2023_join_group | 400 | 0 | 30 | 0 | 2 | 0 | 5 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 51 | 420 |
| francis2023_leading_human | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 420 |
| francis2023_leave_group | 400 | 0 | 30 | 0 | 1 | 0 | 2 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 38 | 420 |
| francis2023_narrow_doorway | 400 | 0 | 0 | 30 | 30 | 30 | 0 | 0 | 0 | 0 | 0 | 30 | 30 | 30 | 0 | 180 | 420 |
| francis2023_narrow_hallway | 400 | 0 | 5 | 3 | 6 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 3 | 0 | 0 | 21 | 420 |
| francis2023_parallel_traffic | 400 | 0 | 25 | 2 | 1 | 0 | 2 | 0 | 25 | 10 | 0 | 2 | 2 | 0 | 0 | 69 | 420 |
| francis2023_pedestrian_obstruction | 400 | 0 | 30 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 49 | 420 |
| francis2023_pedestrian_overtaking | 400 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 30 | 0 | 0 | 0 | 0 | 12 | 0 | 43 | 420 |
| francis2023_perpendicular_traffic | 400 | 0 | 30 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 36 | 420 |
| francis2023_robot_crowding | 400 | 0 | 28 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 35 | 420 |
| francis2023_robot_overtaking | 400 | 0 | 30 | 0 | 0 | 0 | 4 | 0 | 11 | 1 | 0 | 0 | 0 | 0 | 0 | 46 | 420 |

The remaining scenarios have no horizon-consistency flags and no below-600 exposure: `classic_realworld_double_bottleneck_high`, `classic_station_platform_medium`, `classic_cross_trap_low`, `classic_cross_trap_medium`, `classic_cross_trap_high`, `classic_merging_low`, `classic_merging_medium`, `classic_overtaking_low`, `classic_overtaking_medium`, `classic_urban_crossing_medium`.

**Room to succeed:** the 1,057 H400 timeouts lost 200 declared runner steps (20 s); the 448 H500 timeouts lost 100 (10 s). This leaves a possible completion opportunity, not proof that any particular row would succeed at H600. Continued progress supports investigating budget censoring; extrapolation cannot account for later contacts, delays, route changes, or goal semantics. The example hybrid parallel-traffic row has positive terminal planner `progress_windows[5s]=1.5583861586771341 m` and route remaining distance about 5.4 m, consistent with such an opportunity, but not a measured H600 success.

**Requested last-50-step fraction: unavailable.** All 1,505 flags have `record_simulation_step_trace=false` and `record_planner_decision_trace=false`; no bundle member is a trace/trajectory/step telemetry artifact, and no flagged row has a top-level distance trajectory. There are no distances at steps 350 and 400, so an exact fraction or zero fraction would be invented.

Available partial diagnostic: among the **1,057 step-400 flagged timeouts**, **197** have terminal hybrid-planner `progress_windows[5s]`; **50/197 = 25.381%** are positive, with **860/1,057 lacking that proxy**. Across all 1,505 flags the proxy is present in 206 and positive in 52 (25.243% of observed proxies). This is not the requested full-population last-50-step fraction: the value is computed at the last planner call before the final simulator step; its five-second reference depends on planner observation dt and is not a saved step-350/400 endpoint pair. It also demonstrates net change, not monotonic decrease at every step.

Proxy definition at release source: `robot_sf/planner/hybrid_rule_local_planner.py:1481`–`:1501` uses reference minus current distance over nominal 1/3/5-second windows; `:2974`–`:2975` timestamps planner history. Current corresponding definitions are `:1996`–`:2016` and `:3518`–`:3519`. No outcome after the published stop exists in this bundle.

## 3. Actual 0.0.8 template exposure on main

The requested template selects `configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml` at line 11 and H600 at line 45. The canonical campaign loader expands **48 scenarios, 38 below H600** before this correction. Limits and source scalar lines below were verified with the loader against current checkout bytes; inherited overrides do not replace these limits.

| Scenario | Effective horizon on main | Limit source file:line |
|---|---:|---|
| classic_bottleneck_high | 500 | `configs/scenarios/archetypes/classic_bottleneck.yaml:71` |
| classic_bottleneck_low | 500 | `configs/scenarios/archetypes/classic_bottleneck.yaml:11` |
| classic_bottleneck_medium | 500 | `configs/scenarios/archetypes/classic_bottleneck.yaml:41` |
| classic_doorway_high | 500 | `configs/scenarios/archetypes/classic_doorway.yaml:73` |
| classic_doorway_low | 500 | `configs/scenarios/archetypes/classic_doorway.yaml:11` |
| classic_doorway_medium | 500 | `configs/scenarios/archetypes/classic_doorway.yaml:42` |
| classic_group_crossing_high | 500 | `configs/scenarios/archetypes/classic_group_crossing.yaml:69` |
| classic_group_crossing_low | 500 | `configs/scenarios/archetypes/classic_group_crossing.yaml:11` |
| classic_group_crossing_medium | 500 | `configs/scenarios/archetypes/classic_group_crossing.yaml:40` |
| classic_head_on_corridor_low | 500 | `configs/scenarios/archetypes/classic_head_on_corridor.yaml:11` |
| classic_head_on_corridor_medium | 500 | `configs/scenarios/archetypes/classic_head_on_corridor.yaml:39` |
| classic_t_intersection_low | 500 | `configs/scenarios/archetypes/classic_t_intersection.yaml:11` |
| classic_t_intersection_medium | 500 | `configs/scenarios/archetypes/classic_t_intersection.yaml:39` |
| francis2023_accompanying_peer | 400 | `configs/scenarios/single/francis2023_accompanying_peer.yaml:5` |
| francis2023_blind_corner | 400 | `configs/scenarios/single/francis2023_blind_corner.yaml:5` |
| francis2023_circular_crossing | 400 | `configs/scenarios/single/francis2023_circular_crossing.yaml:5` |
| francis2023_crowd_navigation | 400 | `configs/scenarios/single/francis2023_crowd_navigation.yaml:5` |
| francis2023_down_path | 400 | `configs/scenarios/single/francis2023_down_path.yaml:5` |
| francis2023_entering_elevator | 400 | `configs/scenarios/single/francis2023_entering_elevator.yaml:5` |
| francis2023_entering_room | 400 | `configs/scenarios/single/francis2023_entering_room.yaml:5` |
| francis2023_exiting_elevator | 400 | `configs/scenarios/single/francis2023_exiting_elevator.yaml:5` |
| francis2023_exiting_room | 400 | `configs/scenarios/single/francis2023_exiting_room.yaml:5` |
| francis2023_following_human | 400 | `configs/scenarios/single/francis2023_following_human.yaml:5` |
| francis2023_frontal_approach | 400 | `configs/scenarios/single/francis2023_frontal_approach.yaml:5` |
| francis2023_intersection_no_gesture | 400 | `configs/scenarios/single/francis2023_intersection_no_gesture.yaml:5` |
| francis2023_intersection_proceed | 400 | `configs/scenarios/single/francis2023_intersection_proceed.yaml:5` |
| francis2023_intersection_wait | 400 | `configs/scenarios/single/francis2023_intersection_wait.yaml:5` |
| francis2023_join_group | 400 | `configs/scenarios/single/francis2023_join_group.yaml:5` |
| francis2023_leading_human | 400 | `configs/scenarios/single/francis2023_leading_human.yaml:5` |
| francis2023_leave_group | 400 | `configs/scenarios/single/francis2023_leave_group.yaml:5` |
| francis2023_narrow_doorway | 400 | `configs/scenarios/single/francis2023_narrow_doorway.yaml:5` |
| francis2023_narrow_hallway | 400 | `configs/scenarios/single/francis2023_narrow_hallway.yaml:5` |
| francis2023_parallel_traffic | 400 | `configs/scenarios/single/francis2023_parallel_traffic.yaml:5` |
| francis2023_pedestrian_obstruction | 400 | `configs/scenarios/single/francis2023_pedestrian_obstruction.yaml:5` |
| francis2023_pedestrian_overtaking | 400 | `configs/scenarios/single/francis2023_pedestrian_overtaking.yaml:5` |
| francis2023_perpendicular_traffic | 400 | `configs/scenarios/single/francis2023_perpendicular_traffic.yaml:5` |
| francis2023_robot_crowding | 400 | `configs/scenarios/single/francis2023_robot_crowding.yaml:5` |
| francis2023_robot_overtaking | 400 | `configs/scenarios/single/francis2023_robot_overtaking.yaml:5` |

The full include closure and all 48 resolved limits are in `evidence/issue_9668_campaign_horizon/template_0_0_8_limits_on_main.json`. The 400-step limits in other archetypes outside that closure are not an exposure of this template.

The separately authored three-width doorway file has H400 scalars at `configs/scenarios/francis2023_narrow_doorway_three_width_release_0_0_8_v1.yaml:6`, `:43`, `:80`. It is **not included in the 48-scenario main template**. Its preregistered separate H400 slice remains intentional and untouched.

## 4. Intent and design record

The documented design supports **fixed campaign horizon authority**, while per-scenario horizons are an explicit alternative mode. I did not find an explicit maintainer sentence saying “override the 400 defaults”; the conclusion below is an inference from the mode contract and ablation design, not a fabricated author ruling.

- `docs/context/issue_1023_scenario_horizon_benchmark.md:37`–`:47` records the explicit `scenario_horizons` sidecar, joining by scenario identity, patching simulator limits, and reporting the horizon mode. `docs/benchmark_camera_ready.md` (main lines 113–117) describes this as replacing the fixed campaign horizon. `robot_sf/benchmark/camera_ready/_config.py` (main lines 843–854) refuses mixing a schedule with fixed or planner horizons.
- `docs/context/issue_5409_paired_horizon_analysis.md:3`–`:5` says the treatment changes the **fixed episode horizon** from 500 to 600 while holding the scenario matrix and other factors constant. `configs/benchmarks/issue_5409_horizon_ablation_h500.yaml:4`, `:50` and its H600 counterpart declare that only the **campaign-level** horizon changes. Their inherited 400/500 scenario values would suppress the treatment for 38 scenarios under the old runner behavior.
- [Issue #5409](https://github.com/ll7/robot_sf_ll7/issues/5409) explicitly approves estimating the within-simulator effect of changing only episode horizon 500→600 on a matched frozen roster/scenario/seed suite. [Issue #8889](https://github.com/ll7/robot_sf_ll7/issues/8889) requires configs without changing source scenario definitions and horizon as the only treatment axis. Its recorded blocker concerns planner readiness, not an intent to preserve hidden H400 limits.
- `git log -S "max_episode_steps: 400" -- configs/scenarios/single/francis2023_parallel_traffic.yaml` traces the limit to `52b3fc58ddc1a43b47f503a2bb15f7a13b5ce2f6` (2026-01-29, scenario manifest/source split); the added file line 5 carries 400. The classic bottleneck 500 limits entered in the same commit. Later archetype edits adjust layout, spawn metadata, density units, and plausibility; the inspected history provides no H600-specific cutoff decision. Archetype files with 400 added in #596, #4154, #6125 are separate scenario suites, not this release matrix.

Thus the inherited values are valid defaults for scenario-controlled runs, but stale ceilings when a fixed H600 campaign is requested. Making the campaign authoritative matches the documented fixed/scheduled distinction; an admission-only refusal would leave an already declared fixed episode budget unresolved.

## Implementation and validation

Implemented on `fix/campaign-horizon-authority` in `~/hzn-work`, from the inspected latest main.

- Campaign loading binds positive fixed horizons to copied scenarios before preflight; planner scoping reapplies the effective arm override. Both execution paths share the prepared scenarios.
- `metadata.campaign_horizon` preserves the original authored limit and effective fixed budget. Repeated arm binding keeps the original provenance. No scenario YAML, frozen config, seed roster, planner, historical artifact or release identity changes.
- Episode context binds simulator seconds using the effective step budget and actual runner timestep. Scenario-controlled runs retain existing limits and explicit schedules; standalone map-runner callers without the campaign binding retain existing behavior.
- Real-template regression failed on main at `classic_bottleneck_low: assert 500 == 600`; three planner-override cases also failed for incorrect limits (four intended failures, one preservation pass). Added dt controls failed at 1200 and 300 simulator steps for a 600-step runner, then passed after duration binding.
- Final seven-node regression file against unchanged main: **6 failed, 1 passed**, with failures on the intended budget assertions; on fixed code: **7 passed**.
- Focused complete-environment suite: **327 passed, 1 skipped**. Final expanded seven regression nodes include all 48 real scenarios, H500/H700 arm overrides, authored provenance preservation, absent fixed budget, and dt 0.05/0.2. No environment step or planner invocation is required by these tests.

Test-value gate: (1) protects declared campaign budgets reaching the simulator, including arm and timestep overrides; (2) removing scenario binding or converting seconds before dt makes a credible regression; (3) nearest existing schedule test covers explicit sidecars, not the real H600 release matrix, and existing episode decomposition tests use already-matched toy limits; (4) existing loader/config/context boundaries suffice, no test-only production seam.

Evidence boundary: static/configuration and orchestrator implementation proof only. No held-out planner steps, campaign, Slurm, success uplift, corrected release, horizon-ablation finding, or release admission. Broad readiness and domain review are reported in the delivery section of `~/hzn_report.md`. Historical rows require their pinned old runtime to reproduce; any future correction needs a fresh source-bound candidate and review under #9668.
