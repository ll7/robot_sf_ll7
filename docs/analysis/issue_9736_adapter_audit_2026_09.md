# VV-6: 0.0.8 planner-adapter static audit (2026-09-28)

This is the independent source audit for the 14 **slots** in #9736, pinned to
`robot_sf_ll7` main `17bd03e09ad01529d974d457075339760b6f1e21`.
It is implementation-integrity evidence, **not** a preflight pass, oracle pass,
nominal campaign, baseline-faithfulness certification, or release admission.
The release contract is the corrected-results ruling in #9668. The four
historical hybrid slots must be replaced by v4-named keys under #9751; its
final keys, configs, and freeze are pending. Accordingly, the old keys in
`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml:67-155`
and the template are audit indices, not an approved 0.0.8 roster.

## Method and common boundary

I read each slot's campaign binding, map-runner builder, adapter target and
state extraction, action projection, and fallback path. The checklist was:
units/frames; robot and pedestrian radii; drive speed, acceleration, braking,
and turn limits; force magnitude where applicable; active goal/waypoint;
fallback semantics. The common flat observation has world XY positions and
goals, robot-ego pedestrian velocity in m/s, and scalar radii
(`robot_sf/benchmark/map_runner/map_runner_observations.py:22-30`;
`robot_sf/sensor/socnav_observation.py:919-936`). Runtime defaults are a
1.0 m robot radius, 0.4 m pedestrian radius, 2.0 m/s linear speed,
1.0 rad/s angular speed, and 1.0 m/s² linear acceleration
(`robot_sf/common/robot_defaults.py:34`;
`robot_sf/sim/sim_config.py:300`;
`robot_sf/robot/differential_drive.py:25-40`).

All generic SocNav and rule adapters pass their `(v, omega)` output through
the benchmark projection (`map_runner.py:2089-2092,2122-2133`;
`map_runner_policy_common.py:104-121`). The projection clips *speed and
angular speed*, but its stateless model has no acceleration or braking check
(`robot_sf/planner/kinematics_model.py:48-81,175-209`). This is a shared
adapter limit: a projected command is not proof that its trajectory respects
the drive's acceleration envelope. The simulator's drive remains a separate
execution layer; release interpretation must inspect command projection and
executed motion. The per-arm config and reference-method audit is still #9750.

## Per-slot checklist

Legend: **F** units/frames, **R** body radii, **D** drive limits,
**M** force or risk magnitude, **G** goal/waypoint, **B** fallback.
`N/A` means the mechanism is not part of that arm, not that all behavior is
validated. Source anchors refer to the pinned main commit.

| Slot | F; R; D | M; G; B | Disposition |
| --- | --- | --- | --- |
| `prediction_planner` | Predictor input keeps ego velocity beside ego relative position (`socnav_prediction.py:477-503`). Config fixes robot/ped radii at 0.25/0.25 m and speed at 1.6 m/s (`configs/algos/prediction_planner_camera_ready.yaml:20-22,42-55`); these differ from the physical body and need #9750 audit. Generic command projection applies. | Learned future risk/clearance uses configured radius sum (`socnav_prediction.py:1005-1011`); targets `goal.current` (`:1716-1748`). Registry defaults `allow_fallback=False` (`map_runner_policies/socnav_family.py:299-304,396-425`); runtime fallback metadata must still be checked. | **Open**: #9750 geometry and model-provenance/foresight checks; no static admission. |
| `goal` | World XY active goal; no pedestrian input by design. Builder projects `(v,omega)` (`map_runner_policies/goal.py:54-93`). | Force N/A; built-in goal policy only; no alternate planner fallback. | **Characterized control**, not a crowd-safe oracle. |
| `social_force` | Pedestrian ego velocity rotates to world before pair forces (`socnav_social_force.py:513-545`); obstacle v2 reads robot radius (`:644-657`). The corrected template selects `resolution_independent_v2` (`configs/algos/social_force_resolution_independent_v2.yaml:13`). | v2 replaces the per-cell wall sum (`socnav_social_force.py:644-647`) and total force can be clipped (`:1075-1086`). It targets `goal.current` (`:142`). Current config does **not** select the opt-in surface-distance pedestrian term (`socnav_base.py:420-426`). No external-model fallback. | **Open**: #9758 pedestrian term and #9764 angular branch cut; force ratios and outcomes still need corrected-run evidence. |
| `orca` | RVO2 path rotates ego pedestrian velocity to world (`socnav_orca.py:1272-1287`), takes observed robot/ped radius (`:1263,1272`), converts world velocity to unicycle (`:1153-1168`), then projects. | Force N/A; `goal.current` is used (`:1240-1252`). Missing RVO2 fails unless explicit fallback is permitted (`:157-176`); release stanza says fail-fast (`paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml:84-88`). | **Source path plausible**; prove RVO2/native runtime and oracle results before admission. |
| `ppo` | Release uses dict-observation, unicycle action and a CPU checkpoint (`configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml:18-34`), so it receives the producer's ego pedestrian features; action path projects (`map_runner.py:1338-1359`). Checkpoint feature interpretation requires #9845. | Force N/A; goal feature is the producer's active goal. `fallback_to_goal:false` is explicit; failure must remain non-success. | **Open**: #9845 training/evaluation frame crosswalk and checkpoint provenance. |
| `socnav_sampling` | `bounded_v2` is selected (`configs/algos/socnav_sampling_bounded_v2.yaml:15-17`); bound environment supplies speed, acceleration, deceleration, radius (`socnav_base.py:1135-1150`). Pedestrian velocity prediction is off, so frame use is N/A for this config. | Heuristic pedestrian repulsion and robot-footprint sweep are versioned; braking envelope is explicitly off (`socnav_sampling_bounded_v2.yaml:6-17`). It uses current goal and bound map path (`socnav_base.py:698-705`; `socnav_base.py:449-456`). This is the intended in-repo heuristic, not an unlabelled RVO2-style fallback. | **Pending** route/preflight and #9750 reference audit. |
| `sacadrl` | Ego pedestrian velocity rotates to global for the network (`socnav_sacadrl.py:475-483`); observed robot and shared pedestrian radii enter state features (`:381-383,468-483`). Network turn delta is divided by observed timestep then clipped (`:168-180`), followed by generic projection. | Force N/A; `goal.current` (`:287-305,470-472`). Missing checkpoint fails with release `allow_fallback=False` (`map_runner_policies/socnav_family.py:293-297,396-425`), but runtime metadata must confirm no heuristic use. | **Open**: model/reference and checkpoint verification under #9750. |
| Hybrid slot 1: bottleneck-yield | Current v2-named key binds a v3 hybrid config (`paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml:111-116`). Historical v3 flat velocity pass-through is known (#9752); v4 code rotates it (`hybrid_rule_local_planner.py:655-660`). v4 consumes observed body radii and timestep (`:607-675`) and adds braking checks (`:333-357`). | Hybrid has no social-force law; its rollout risk and selected source need final v4 config and scenario-override inspection. Current target switches only near active waypoint (`:619-626`). Child selector/fallback must be accounted by decision metadata, not silently called native. | **Hold** for #9751 final key/config/freeze and #9748 tuning. |
| Hybrid slot 2: collision-guard | Same shared v3/v4 adapter boundary as slot 1; current v2-named key binds `scenario_adaptive_hybrid_orca_v2_collision_guard_s30_h600_release.yaml` (`paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml:118-123`). | Scenario override can select ORCA; effective config and source-selection frequencies are unverified here. Goal switch and fallback accounting as above. | **Hold** for #9751/#9748. |
| Hybrid slot 3: fast-progress static-escape | Current v3 key uses a release twin setting 3.0 m/s and 3.0 m/s² (`configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape_s30_h600_release.yaml:3-10`), above the 2.0/1.0 drive; v3 also retains the known flat-frame defect. | No force law; goal switch is `hybrid_rule_local_planner.py:619-626`. V4 surface-clearance/braking path is not admitted until keyed and frozen. | **Hold** for #9751/#9748; old v3 remains only historical. |
| Hybrid slot 4: continuous static-escape | Same shared adapter, velocity-frame and body/drive concerns; current v3 config is bound at `paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml:132-137`. | Continuous static-clearance branch needs final v4 config and exact scenario overrides; current goal/fallback boundaries as above. | **Hold** for #9751/#9748. |
| `guarded_ppo` | Primary PPO input remains in its trained ego feature frame. Guard rollout converts pedestrian ego velocity to world (`guarded_ppo.py:316-363`). Builder chooses guard/fallback/prior, then projects chosen command (`map_runner.py:1575-1581,1640-1665`). | Force N/A; current goal is selected until reached (`guarded_ppo.py:319-328`). Exact `fallback_safe` is a declared guard action; best-effort/degraded markers are forbidden by the identity-bound policy (`docs/context/issue_691_benchmark_fallback_policy.md`). | **Open**: #9845 and #9750, plus guard decision counters at runtime. |
| `predictive_mppi` | Predictor keeps ego velocity; sampled rollout uses predicted positions in local coordinates (`predictive_mppi.py:83-125,334-369`). Its nested predictor has configured radii rather than proven physical radii; generic projection applies (`map_runner.py:1126-1169`). | Risk cost rather than force; **wrong goal selector** chooses `goal.next` or zero sentinel (`predictive_mppi.py:103-108,505-513`). Foresight fallback provenance is surfaced (`:664-674`) and release `allow_fallback` defaults false. | **Blocker #9883**; #9750 geometry also open. |
| `risk_dwa` | Pedestrian ego velocity rotates to world for TTC/rollout (`risk_dwa.py:99-134`). Candidate speeds and turn rates are capped by config and generic projection; planner does not model the drive acceleration envelope (`:48-79`). | Risk distance uses fixed `safe_distance:0.35` and `near_distance:0.7` defaults rather than observed body radii (`:68-70,198-225`), requiring #9750 classification. **Wrong goal selector** chooses `goal.next` or zero sentinel (`:104-106`). No alternate-model fallback. | **Blocker #9883**; #9750 geometry also open. |

All four historical hybrid release twins resolve from the v3 base and set
`max_linear_speed: 3.0` and `max_linear_accel: 3.0`, including their
scenario-adaptive variants. The values are visible in their
`configs/policy_search/candidates/*_s30_h600_release.yaml` files and exceed
the physical drive defaults; #9751 must resolve the effective v4 parameters
before this table can describe the corrected 0.0.8 implementations.

## Reproduced blocker and release consequence

At this head, I invoked both adapter state-extraction paths with robot
`(5,5)`, active goal `(8,5)`, next-goal sentinel `(0,0)`, no pedestrians:
both returned target `(0,0)`. With robot `(0,0)`, active `(1,0)`, next
`(0,1)`, both returned `(0,1)`. The producer's sentinel is explicit
(`socnav_observation.py:931-936`) and the observation contract distinguishes
current from next (`docs/dev/observation_contract.md:108-109`). This proves
the target-selection defect, **not** a measured episode failure or rate change.
The bounded correction and required route replay are tracked as [#9883](https://github.com/ll7/robot_sf_ll7/issues/9883),
a native child of #9736. Preserve the 0.0.7 implementation and attribute any
0.0.8 row change to a versioned source/config correction.

## Remaining VV-6 work

This report does not close #9736. The final 14-key audit must be reconciled
after #9751 freezes v4 names/configs and #9748 development tuning. #9845 owns
the per-arm frame test crosswalk; #9750 owns reference-method fidelity and
physical geometry; #9764 and #9758 own social-force defects. For **every**
release scenario × planner cell, VV-6 still needs detector/reviewer coverage
accounting and an explicit uncovered-cell ledger. The 0.0.2–0.0.7 retro-check
must run the corresponding oracle and anomaly checks against restored
historical releases and report defect carriage without rewriting them.
Current focused reproduction and source reading do not discharge either task.
