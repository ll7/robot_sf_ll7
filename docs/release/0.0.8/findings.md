# Release 0.0.8: findings register

This register lists every finding from the internal audits and the external
reviews that fed release 0.0.8, one row each, with what happened to it. It
supersedes the manual defect-by-experiment matrix of
[#10012](https://github.com/ll7/robot_sf_ll7/issues/10012) once merged.

Sources:

- **Internal audits:** the read-only planner hunt in four groups (A-D, umbrella
  [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007); hybrids in [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006)), the pedestrian-model audit (PED,
  [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017)), and one finding from the verification of the external
  pedestrian review (PED-V1).
- **External reviews (GPT-6 Pro, 2026-09-30):** pedestrian simulation (X-PED),
  scenarios and maps (X-SCN), campaign comparison (X-CMP), private campaign
  admission (X-ADM), reporting statistics (X-STAT) and planner adapters
  (X-ADP). External findings are leads until verified; the verified column
  says whether a lane reproduced them.

Severity follows the source: P1 changes results, P2 could change results, P3
hygiene. "(open)" means the fixing pull request is not merged yet. Decision
IDs (D-0NN) refer to [decisions.md](decisions.md). Full reports are in the
private review archive (`docs/reviews/0.0.8/` in the private operations
repository). State checked on 2026-10-01 against main and the current open PR heads. Measurements are attributed to the archived reviewer reports; this records lane did not rerun simulations. A disclosure disposition records the adopted disclosure policy, not confirmation that the release notes or thesis have already been edited.

## Planner integration

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| A1 | internal audit | prediction_planner model trained on pedestrian velocities rotated twice | P2 | confirmed | disclosed; collectors fixed in [#10011](https://github.com/ll7/robot_sf_ll7/pull/10011) (merged); retrain in 0.1.0 [#10018](https://github.com/ll7/robot_sf_ll7/issues/10018) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| A2 | internal audit | prediction_planner discards the wall part of its occupancy cost | P2 | confirmed | disclosed as a method limit (decision D-021, [#10011](https://github.com/ll7/robot_sf_ll7/pull/10011)); wall term in 0.1.0 [#10018](https://github.com/ll7/robot_sf_ll7/issues/10018) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| A3 | internal audit | predictive_mppi silently caps the horizon at the model's 8 steps | P2 | confirmed | fixed in [#10011](https://github.com/ll7/robot_sf_ll7/pull/10011) (merged): fails fast, configs state 8 x 0.1 s | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| A4 | internal audit | prediction heading lattice collapses to three turn rates | P3 | confirmed | fixed in [#10011](https://github.com/ll7/robot_sf_ll7/pull/10011) (merged) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| A5 | internal audit | predictive_mppi v1 model may have been trained on targets that switch pedestrians | P2 | plausible (code confirmed, dataset attribution conditional) | disclose as a model limitation; retrain in 0.1.0 [#10033](https://github.com/ll7/robot_sf_ll7/issues/10033) (confirmed by orchestrator 2026-09-30) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B1 | internal audit | SA-CADRL never drives faster than 1.0 m/s | P2 | confirmed | kept on purpose and disclosed (decision D-018, [#10008](https://github.com/ll7/robot_sf_ll7/pull/10008)) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B2 | internal audit | SA-CADRL observes only the 3 nearest pedestrians | P2 | confirmed | fixed in [#10008](https://github.com/ll7/robot_sf_ll7/pull/10008) (merged): 19 agents | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B3 | internal audit | SA-CADRL discrete turns all saturate at the turn-rate limit | P2 | confirmed (mechanism) | kept and documented as a method note ([#10008](https://github.com/ll7/robot_sf_ll7/pull/10008)) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B4 | internal audit | sampling planner rollout ignores angular acceleration and current turn rate | P2 | confirmed by the fix lane | fixed in [#10008](https://github.com/ll7/robot_sf_ll7/pull/10008) (merged): forecasts through the native drive | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B5 / D-F6 | internal audit | occupancy grid draws pedestrians with 0.35 m instead of 0.4 m radius | P3 | confirmed (still on main and open PRs) | 0.1.0 [#10034](https://github.com/ll7/robot_sf_ll7/issues/10034); fix changes PPO inputs (confirmed by orchestrator 2026-09-30) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B6 | internal audit | SA-CADRL and sampling silently fall back to dt 0.1 | P3 | confirmed | fixed in [#10009](https://github.com/ll7/robot_sf_ll7/pull/10009) (merged) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B7 | internal audit | guarded PPO goal selection can target the [0,0] placeholder on the final leg | P3 | confirmed (unreachable in practice; still on main) | not exposed; 0.1.0 [#10034](https://github.com/ll7/robot_sf_ll7/issues/10034) (confirmed by orchestrator 2026-09-30) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B8 | internal audit | PPO observation alignment fills missing keys with zeros without a record | P3 | confirmed (no key missing today) | not exposed; 0.1.0 [#10034](https://github.com/ll7/robot_sf_ll7/issues/10034) (confirmed by orchestrator 2026-09-30) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| C-F1 | internal audit | hybrid v4 static-escape and corridor-transit keys never act | P2 | confirmed | disclosed (decision D-010); fail-closed key check in 0.1.0 [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) | [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) |
| C-F2 | internal audit | all four hybrids use parameter overrides keyed on scenario names | P2 | confirmed | disclosed (decision D-010, [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006)) | [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) |
| C-F3 | internal audit | v4 tuning ran without the perpendicular_traffic override used at release | P3 | confirmed | disclosed ([#10006](https://github.com/ll7/robot_sf_ll7/issues/10006), [#10002](https://github.com/ll7/robot_sf_ll7/issues/10002)) | [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) |
| C-F4 | internal audit | v3 centre-distance gates read as surface clearance in v4 | P3 | plausible | 0.1.0 issue [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) (confirm and document) | [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) |
| D-F1 | internal audit | social_force planner integrates with dt 0.5 s instead of 0.1 s | P1 | confirmed | fixed in [#10009](https://github.com/ll7/robot_sf_ll7/pull/10009) (merged) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F2 | internal audit | ORCA adapter drives forward at full speed whatever the heading error | P1 | confirmed in code | fixed in [#10009](https://github.com/ll7/robot_sf_ll7/pull/10009) (merged), decision D-020 | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F3 | internal audit | ORCA solver runs with 3.0 m/s on a 2.0 m/s robot | P2 | confirmed (release template already 2.0 m/s) | fixed in [#10009](https://github.com/ll7/robot_sf_ll7/pull/10009) (merged) for the candidate manifest | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-P1 | internal audit | release social_force keeps the legacy pedestrian kernel | P2 | refuted for the release (template selects the corrected kernel) | refuted | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| X-ADP-1 | external review | adapter rollouts reach the target velocity instantly, including an instant stop | P1 | confirmed by ADVV execution; checked against corrected #9926 release bindings | not exposed in the corrected release config where #9926 supplies native drive rollouts; residual guard braking tail and prediction lattice: 0.1.0 #10051 | advv_report.md, private-ops review archive; [#10051](https://github.com/ll7/robot_sf_ll7/issues/10051) |
| X-ADP-2 | external review | bounded_v2 sampler turns instantly and integrates the pose differently | P1 | confirmed by ADVV execution; checked against corrected #9926 release bindings | fixed in #10008 (merged); residual non-release bounded sampler details: 0.1.0 #10051 | advv_report.md, private-ops review archive; [#10051](https://github.com/ll7/robot_sf_ll7/issues/10051) |
| X-ADP-3 | external review | guard and Risk-DWA 'TTC' is time to closest approach, not to collision | P1 | confirmed by ADVV execution; checked against corrected #9926 release bindings | fixed in #9926 (merged) for surface_v2 contact TTC; historical adapters retain the old meaning | advv_report.md, private-ops review archive; [#10051](https://github.com/ll7/robot_sf_ll7/issues/10051) |
| X-ADP-4 | external review | moving pedestrians also counted as static obstacles at old positions | P1 | confirmed by ADVV execution; checked against corrected #9926 release bindings | not exposed with the exact static geometry in #9926; fallback robustness: 0.1.0 #10051 | advv_report.md, private-ops review archive; [#10051](https://github.com/ll7/robot_sf_ll7/issues/10051) |
| X-ADP-5 | external review | non-finite SA-CADRL network output becomes a full-speed turn | P2 | confirmed by ADVV execution; checked against corrected #9926 release bindings | not exposed (fault injection only); 0.1.0 #10051 | advv_report.md, private-ops review archive; [#10051](https://github.com/ll7/robot_sf_ll7/issues/10051) |
| X-ADP-6 | external review | native-command parser accepts vx/vy as (v, w) | P2 | confirmed by ADVV execution; checked against corrected #9926 release bindings | not exposed (no release roster arm uses native-command parser); 0.1.0 #10051 | advv_report.md, private-ops review archive; [#10051](https://github.com/ll7/robot_sf_ll7/issues/10051) |

## Metrics and reporting

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| D-F4 | internal audit | metric goal is the waypoint active at episode end, not the route goal | P2 | confirmed | fixed in [#10014](https://github.com/ll7/robot_sf_ll7/pull/10014) (merged), decision D-029 | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F5 | internal audit | deadlock detector flags any slow episode and counts collisions | P2 | confirmed | fixed in [#10014](https://github.com/ll7/robot_sf_ll7/pull/10014) (merged), decision D-027 | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F7 | internal audit | jerk_mean is not divided by dt | P3 | confirmed | fixed in [#10014](https://github.com/ll7/robot_sf_ll7/pull/10014) (merged) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F8 | internal audit | step-index conventions one step off (time to goal, first path segment) | P3 | confirmed | fixed in [#10014](https://github.com/ll7/robot_sf_ll7/pull/10014) (merged) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-P2 | internal audit | recorded pedestrian forces lag positions by one step | P3 | plausible (code path still on main) | 0.1.0 [#10035](https://github.com/ll7/robot_sf_ll7/issues/10035) (confirmed by orchestrator 2026-09-30) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-N1 | internal audit | shortest-path reference uses no inflation, so it hugs walls | P3 | confirmed (kept in #10014) | disclose as a metric definition note (optimistic for every planner); 0.1.0 [#10035](https://github.com/ll7/robot_sf_ll7/issues/10035) (confirmed by orchestrator 2026-09-30) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| X-STAT-1 | external review | table means bypass the foresight eligibility filter | P1 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (merged) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-2 | external review | Spearman over a key intersection keeps absolute ranks | P1 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (merged) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-3 | external review | paired differences keep the last duplicate row | P2 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (merged) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-4 | external review | SNQI v2 mean and confidence interval use different weightings | P2 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (merged) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-5 | external review | bootstrap seed 0 silently becomes 123 | P2 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (merged) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-6 | external review | numeric group_by values fall back to the algorithm | P2 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (merged) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |

## Pedestrian simulation

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| PED-P1-1 | internal audit | wall force reaches far and adds up, so pedestrians stop before narrow openings | P1 | confirmed (qualified by the external review) | disclosed (decision D-038); fix in 0.1.0 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P1-2 | internal audit | goal-only single pedestrians walk into obstacles and stay stuck | P1 | confirmed | disclosed (decision D-038); fix in 0.1.0 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P1-3 | internal audit | declared pedestrian speed is multiplied by 1.3 for the cap | P1 | confirmed | kept as the standard convention and documented (decision D-039, [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017)) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-1 | internal audit | route respawns draw from the global random generator | P2 | confirmed (mechanism) | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (merged) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-2 | internal audit | simulation_config.groups is ignored | P2 | confirmed | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (merged) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-3 | internal audit | per-scenario robot_radius override of the reaction force is replaced | P2 | confirmed | not exposed (not in the 0.0.8 grid) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-4 | internal audit | pedestrian moved off the robot at reset lands 1.5 m away and keeps walking in | P2 | confirmed | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (merged), decision D-037 | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-5 | internal audit | resetting the same env twice leaves route navigators behind | P2 | confirmed | not exposed (benchmark builds one env per episode); 0.1.0 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-1 | internal audit | reaction-force range uses radius 0.35 m instead of 0.4 m | P3 | confirmed | 0.1.0 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-2 | internal audit | pedestrians overlap at spawn; group repulsion starts too late | P3 | confirmed | 0.1.0 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-3 | internal audit | pedestrians with speed 0 cannot step aside | P3 | confirmed | 0.1.0 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-4 | internal audit | relaxation time is effectively 0.448 s and configured twice | P3 | confirmed | 0.1.0 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-5 | internal audit | respawned groups keep their end-of-route velocity | P3 | confirmed (code) | 0.1.0 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-6 | internal audit | legacy wall force drops to zero at contact | P3 | plausible | covered by the wall-law fix, 0.1.0 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-7 | internal audit | route completion ignores the waypoint index | P3 | plausible; static scan found 0 of 29 release routes exposed | not exposed; 0.1.0 [#10036](https://github.com/ll7/robot_sf_ll7/issues/10036) (confirmed by orchestrator 2026-09-30) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-V1 | internal audit | join_group and leave_group never contain a group | P2 | confirmed | disclosed (decision D-041); fix in 0.1.0 [#10028](https://github.com/ll7/robot_sf_ll7/issues/10028) | [#10028](https://github.com/ll7/robot_sf_ll7/issues/10028) |
| X-PED-1 | external review | group gaze scales with 1/goal distance, can push forward, ignores field of view | P1 | confirmed by execution | disclosed (decision D-040); fix in 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-2 | external review | intra-group repulsion weakens as members get closer | P1 | confirmed by execution | disclosed (decision D-040); fix in 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-3 | external review | a waiting pedestrian can drift during an authored wait | P1 | confirmed by execution (drift at most 0.13 m in release) | 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-4 | external review | goal-less roles start with an eastward velocity | P1 | confirmed by execution (path deviation at most 0.2 m) | 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-5 | external review | coincident robot and pedestrian centres divide by zero | P2 | confirmed by execution | not exposed (closest approach 1.33 m); 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-6 | external review | join/leave reach the group forces one step late | P2 | confirmed by execution | not exposed (join never completes); 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-7 | external review | reset keeps changed group membership | P2 | confirmed by execution | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (merged); not exposed in the benchmark | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-8 | external review | timed holds behave wrongly at boundary values | P2 | confirmed by execution | not exposed; 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-9 | external review | joining group 0 does not latch an automatic target | P2 | confirmed by execution | not exposed; 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-10 | external review | FastPysfWrapper queries use the old social kernel | P2 | confirmed by execution | not exposed (no 0.0.8 arm uses the wrapper); 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-11 | external review | archetype speed factors ignored in crowded zones | P2 | confirmed by execution | not exposed (no release scenario uses archetypes); 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-12 | external review | archetype labels depend on dictionary key order | P2 | confirmed by execution | not exposed; 0.1.0 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |

## Scenarios and maps

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| X-SCN-1 | external review | planner random draws change pedestrian goals and respawns | P2 | confirmed | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (merged), decision D-034 | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-2 | external review | rectangular spawn and goal zones sampled as triangles | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (merged), decisions D-042, D-043 | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-3 | external review | crowded-zone behaviours lose obstacle constraints for later goals | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (merged) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-4 | external review | pedestrian spawns check centres, not footprints | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (merged) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-5 | external review | unknown simulation_config keys accepted and ignored | P2 | confirmed | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (merged), decision D-036 | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-6 | external review | episode seed does not seed optional archetype assignment | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (merged) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-7 | external review | included scenario's map shadowed by a same-named root file | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (merged) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-8 | external review | sparse SVG zone indices compacted | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (merged) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |

## Campaign integrity and 0.0.7-to-0.0.8 comparison

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| X-CMP-1 | external review | equal-sized sets of wrong episodes pass campaign integrity | P1 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (merged) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-2 | external review | correct slots with the wrong algorithm or config pass integrity | P2 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (merged) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-3 | external review | checksum check verifies a different file than the payload file | P2 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (merged) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-4 | external review | an ordinary slot can resolve to another algorithm by scenario override | P2 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (merged) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-5 | external review | comparison worker rejects the frozen v4 config (missing lineage) | P2 | confirmed | fixed in [#10001](https://github.com/ll7/robot_sf_ll7/pull/10001) (merged) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-6 | external review | comparison worker rejects the approved ORCA hand-off | P2 | confirmed | fixed in [#10001](https://github.com/ll7/robot_sf_ll7/pull/10001) (merged), decision D-024 | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-7 | external review | comparator checks a seeded scenario against an unseeded expectation | P2 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (merged) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |

## Campaign admission (private operations)

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| X-ADM-1 | external review | two flags select another canonical queue while production commands stay allowed | P1 | confirmed (fail-on-base tests) | fixed in [private-ops #413](https://github.com/ll7/robot_sf_ll7-private-ops/pull/413) (merged), decision D-047 | [private-ops #412](https://github.com/ll7/robot_sf_ll7-private-ops/issues/412) |
| X-ADM-2 | external review | reviewed public commit not part of the launch-packet comparison | P1 | confirmed (fail-on-base tests) | fixed in [private-ops #413](https://github.com/ll7/robot_sf_ll7-private-ops/pull/413) (merged), decision D-047 | [private-ops #412](https://github.com/ll7/robot_sf_ll7-private-ops/issues/412) |
| X-ADM-3 | external review | Git replacement refs can fake freeze blobs | - | confirmed | fixed in [private-ops #407](https://github.com/ll7/robot_sf_ll7-private-ops/pull/407) (merged) | [private-ops #407](https://github.com/ll7/robot_sf_ll7-private-ops/pull/407) |

## Thesis wording

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| D-F9 | internal audit | thesis attributes the 2.0 m/s clamp to code the release path never calls | P3 | confirmed | thesis intake [diss#3021](https://github.com/ll7/diss/issues/3021) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |


## Verified catch-up findings

### Pedestrian simulation

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| WALL-1 | wall-order replay | The emergent-phenomena builders passed walls as (x1, y1, x2, y2), but the simulator reads (start_x, end_x, start_y, end_y). So the face-validity (exit arching, doorway oscillation, lane formation) and lane-formation campaigns ran without their walls. | P1 for thesis evidence | confirmed; six geometry tests fail on base | **fixed** in #10057 (merged in #10080). The replay ran 428 corrected runs plus 428 same-runtime replays. The exit effect is weaker but still over its threshold. Doorway oscillation at released speed disappears. Lane control drops from 3/3 to 1/3 clear hits. Stage A has no eligible candidate. **thesis**: diss#3028 | #10056 |
| WALL-2 | wall-order replay | With correct walls, a lone pedestrian stops 1.51 m before the 1.2 m door: the obstacle force (−1.30) cancels the walking force (+1.30). In 20-agent runs pedestrians mill upstream and throughput is 0. | P2 | confirmed by execution (the force-off control crosses) | **disclosed** (D-055, verdict B: headline success moves at most 2.5 points; ranking unchanged); **0.1.0** | #10061 |
| WALL-3 | doorway check | In the benchmark environment a lone pedestrian stops about 2.25 m in front of any opening up to about 3.0 m wide and never passes. At 2.2 m the wall force is exactly 1.30 = v0/τ. Cause: when a wall segment's projection misses, it pushes from its nearest endpoint, and several corner terms 10/(d+0.57)³ add up. In the three-width slice pedestrians queue instead of flowing through. | P2 | confirmed by execution | **disclosed** (D-056); **0.1.0** acceptance on #10074; **thesis** (diss#3027) | #10074 comment 5932010889 |
| WALL-4 | wall calibration round | No wall-force setting passes (legacy refit, true gradient, body-edge exponential). Narrow-bottleneck flow misses Seyfried 2009 at 0.8/1.0/1.2 m, and wall penetration reaches 6–14 cm. Root cause: non-overlapping 0.40 m discs cannot exceed 1.80 /m², below the experiments' 3.3 /m². | P2 | confirmed (dev seeds 1001–1010) | **0.1.0** (joint calibration); legacy_v1 stays the default | #10061 comment 5927381513, #10073 |
| WALL-5 | validation baseline | Case V2 reported 0 of 210 lone aperture passages at every width, which seemed to contradict the benchmark doorway crossings. | – | explained by WALL-3 (model behaviour, not a test error) | folded into WALL-3 | #10075 |
| SPEED-1 | validation baseline and source check | Every crowd pedestrian walks at one shared 0.65 m/s (0.5 × 1.3). That value is both its desired speed and its hard cap, with no spread. Measured adult free walking speed is 1.29–1.34 m/s. | P2 (realism) | confirmed in code and by V1 | **disclosed** (D-057); **0.1.0** | #10074 comment 5932171567 |
| SPEED-2 | source check | The `typical` speed-tier docstring cites Moussaïd 2010; the measurement is Moussaïd 2009. | P3 | confirmed | **0.1.0** | #10074 |
| SPEED-3 | overlap measurement | The scenario loader silently drops `ped_speed_tier` and the desired-speed fields, so a requested tier has no effect. | P2 for 0.1.0 (no 0.0.8 scenario sets it) | confirmed by execution | **0.1.0** | #10083 |
| RADIUS-1 | validation work | Pedestrian radius is 0.40 m for collision, metrics and placement but 0.35 m in the force kernel. Real shoulder half-width is about 0.23–0.26 m. | P2 (realism) | confirmed in code | **disclosed** (discs 0.7–0.8 m wide); **0.1.0** (one 0.28 m parameter) | #10074 |
| OVL-1 | overlap measurement (1,530 episodes) | Pedestrian–pedestrian overlap is neither counted nor prevented. Group pairs are closer than 0.80 m in 55.8 % of pair-steps, closer than 0.45 m in 0.22 %, minimum 0.028 m. Non-group pairs: 0.34 % and 0.016 %. | P2 (realism) | confirmed | **disclosed** (thesis); **0.1.0** | #10027 comment 5933982284 |
| OVL-2 | overlap measurement | The group gaze force divides by the distance to the pedestrian's own waypoint (ε 1e-6), so it blows up near a waypoint. `fov_phi` is declared but unused. Cohesion still pulls below contact distance. | P2 | confirmed | **0.1.0** (#10027 items 1, 2, 5) | #10027 comment 5933982284 |
| CONTACT-1 | verification of the contact research | The capped-contact-force proposal was rejected after a reported head-on-pair verification (0.26 m overlap and elastic rebound). The separate five-person residual-overlap number is chat-only and is not admitted as independently verified evidence here. | design | author-approved switch; #10074 comment records the probe conclusion, not a committed reproducibility artifact | **0.1.0** (switched to velocity projection) | #10074 comment 5934117678 |
| PLAUS-3 | plausibility check | Pedestrian–robot repulsion is radial with no anticipatory side selection; collisions are not attributed. The reported 2.0 m centre cutoff is refuted: code adds both force radii, giving about 3.35 m with defaults; collision geometry separately uses 0.40 m pedestrian radius. | P2 | confirmed | **disclosed** (D-058); **0.1.0**; **thesis** | #10065 |

### Planners, learned policies, merge trains

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| PLAUS-1 | plausibility check | francis2023_pedestrian_overtaking: the robot spawns inside the pedestrian's lane on 12/30 seeds, and risk_dwa collides on 11/30. | P1 | confirmed | fix pending in #10067 (open, train 2 part 2) | #10063 |
| PLAUS-2 | plausibility check | When no command is admissible, risk_dwa, predictive_mppi and the guarded-PPO guard brake to zero and can never reach their escape. This caused most of the 353 rehearsal timeouts. | P1 | confirmed | **fixed** in #10066 (merged in #10080; predictive_mppi successes 105 → 149) | #10064 |
| PLAUS-4 | plausibility check | Published PPO success fell from about 50 % to 18 % under the correct action reading. That is an honest out-of-distribution result. | – | confirmed | **disclosed**; arm replacement pending in #10077 (D-053, D-054) | #9995 |
| PLAUS-5 | plausibility check | The 0.0.8 gains come mainly from configs that fix 0.0.7 plant mismatches. The 0.0.7 predictive_mppi baseline is invalid. | – | confirmed | **disclosed** (D-062); #10058 | plaus_report.md, private-ops review archive |
| PPO-1 | PPO evaluation | The retrained PPO collides in 89/240 episodes, 85 % of them with static geometry. Variant B drives at the 2.0 m/s cap on 97–100 % of steps, and the policy std grows during training. | P2 | confirmed | **disclosed**; **0.1.0** | #10071 |
| GRID-1 | thesis figure capture | The circle rasteriser places pedestrians about half a cell off (+0.08 to +0.14 m) in the grid that PPO and guarded PPO read. | P2 | confirmed | **disclosed** (D-060); **0.1.0** | #10082 |
| REPRO-1 | adapter verification | guarded_ppo episodes are not bit-reproducible across hosts (8 of 20 reruns differ). | P2 | confirmed | process: campaign on one node (D-072) | #10052 |
| TRAIN-1 | rr10080, finding F1 | The corrected crowd world lowers socnav_sampling: success −2.0 pp [−3.5, −0.8], pedestrian collisions +1.5 pp [+0.5, +2.8]. The train report had first blamed #10008; the review refuted that. | P1 by measurement | confirmed (17,280 episodes) | **disclosed**, accepted (D-050) | #10080 |
| TRAIN-2 | train 2 build | A non-release sampler test cell now times out safely at 400 steps instead of succeeding in 182. | P3 | confirmed by bisect | **fixed** (test re-scoped, D-065) | #10080 |
| TRAIN-3 | train 1 | PR CI skips slow tests, so train 1 turned main red after merging (20 failing tests). | P2 (process) | confirmed | **fixed** in #10069; rule D-071; **0.1.0** #10048, #10070 | #10046 |

### Metrics and reporting

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| METGEO-D1 | metric geometry | The wall/agent collision metric measures from the robot centre (0.25 m) and ignores its 1.0 m radius. | P2 (latent) | confirmed; release rows use the footprint flag instead, with 0/20 disagreements | not exposed; **0.1.0** | #10079 |
| METGEO-D2 | metric geometry | Time to collision ignores both radii and overstates the time by about 0.7 s at 2 m/s. | P2 | confirmed | **disclosed** (D-061); **0.1.0** | #10079 |
| METGEO-D4..D6 | metric geometry | Space compliance, the critical-interval tools and the runner's default radii are all centre-based. | P3 | confirmed | not exposed; **0.1.0** | #10079 |
| JERK-1 | rehearsal 1 | The published 0.0.7 breakdown tables have an empty `jerk_mean` column. | P3 | confirmed | **disclosed** (erratum in the 0.0.8 notes) | #10044 |
| CURV-1 | rehearsal 1 / auditor | v1 curvature explodes near standstill (p99 9.2e10). | P2 | confirmed | **fixed** in #10054 (merged), D-066 | #10065 item 3 |
| SNQIV-1 | external SNQI v2 review | SNQI v2 anchors are fragile (K drops 31.5 % without guarded_ppo; ranges 1.70–3.55 across seed pairs), but the ranking never changes (Spearman 1.0 in all 64 variants). | P2 | confirmed | **disclosed** (diss#3021); D-067; F2/F3 **0.1.0** | snqiv_report.md, private-ops review archive |

### Release chain and scenarios

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| CHAIN-1 | post-freeze chain | Release acceptance refuses the 1,260-cell doorway slice. | P1 (freeze blocker) | confirmed | fix pending in #10081 (open; its review returned FIX) | #10078 |
| CHAIN-2 | release-notes lane | The authored release template uses seeds 111–140 and is outside the sealed allowlist, so the real run would be refused. | P1 (release day) | confirmed | open; [#10085](https://github.com/ll7/robot_sf_ll7/issues/10085); D-074 | [#10085](https://github.com/ll7/robot_sf_ll7/issues/10085) |
| CHAIN-3 | intake prep | Rehearsal 1 never ran the release packaging tool. | P2 | confirmed | process (D-070) | diss#3026 |
| CHAIN-4 | runbook | The post-freeze chain has 12 gaps (G01–G12). | P1 | confirmed | G05, G07, G08-mint and G11 fixed in private-ops #421; G09 partly fixed; the rest open | runbook_report.md, private-ops review archive |
| CHAIN-5 | release-notes lane | The SNQI calibration file name says dev101_102 but holds 1001/1002 data. | P3 | confirmed | open (#10045 refresh) | #10045 |
| SCN-A1 | obstacle-force investigation | Scenario authoring defects: the double bottleneck, a station platform waypoint, a stale slice manifest. | P2 | confirmed | **0.1.0** | #10062 |
| HZN-1 | #9999 review | The first legacy-horizon fix extended historical runs to 600 steps under the same episode id (exposure: 1,505 published 0.0.7 rows). | P1 | confirmed | fix pending in #9999 (open), D-064 | #9999 |
| HZN-2 | #9999 review | The fixed-horizon binding broke 37 historical configs and re-hashed 88 more. | P1 | confirmed | fix pending in #9999 (open) | #9999 |
| REPRO-2 | seed review | `uv sync --frozen` does not refresh a stale installed pysocialforce copy. | P2 | confirmed | **fixed** for sealed runs (#10039); virtual environments rebuilt | #10039 |

### Seed hygiene (all on retired seeds 111–140; sealed seeds never stepped)

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| SEED-1 | seed-guard lane | A probe child process reset seed 111 without stepping it. | P3 | disclosed by the lane | **fixed** (#10053) | #10010 |
| SEED-2 | seed review | A reviewer's full-suite run stepped seed 115 for 12 steps, because `--ignore` was overridden. | P3 | confirmed | **fixed** (#10053) | #10010 |
| SEED-3 | scenario review | 27 existing test files step seed 111. | P3 | confirmed | rule changed (D-051); migration remains later work; current runtime guard still rejects retired seeds | #10010 |
| SEED-4 | train 2 | Guard tests use the real sealed seed 50036 as a refusal witness. | P3 | confirmed | permitted (D-052); #10076 item 5 | #10053 |

### Thesis wording

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| THESIS-1 | impact map | The Chapter 6 worked example depends on the wall stall, and on the 0.40 m radius for its 1.37 m collision. | P2 | confirmed | **thesis** | diss#3028 |
| THESIS-2 | impact map | The face-validity claims rest on the misplaced-wall campaign. | P1 | confirmed | **thesis** (diss#3025) | diss#3028 |
| THESIS-3 | intake prep | The manuscript implies that pedestrians avoid the robot. | P2 | confirmed | **thesis** | diss#3027 |
| INTAKE-1 | intake tooling | No release bundle records the 15 physics facts the thesis states. | P2 | confirmed | **thesis**: diss#3031 refuses intake; for 0.0.8, check each fact against the source at the release commit. **0.1.0**: #10084 | diss#3031 |
| THESIS-4 | doorway check | The doorway slice cannot support a door-width claim. | P2 | confirmed | **thesis** | #10061 |


The draft’s X-ADP summary updates the six existing X-ADP rows above; it is not a seventh adapter finding. Thus 48 draft rows yield 47 new finding IDs and six refreshed adapter dispositions.

## Counts per disposition

Each finding is counted once under its main disposition (the first one in its row). Pending PR fixes are counted separately from merged fixes.

| Disposition | Findings |
|---|---|
| 0.1.0 issue only | 17 |
| disclosure or retained method | 29 |
| explained by another finding | 1 |
| fix pending in an open pull request | 4 |
| fixed in a merged pull request | 50 |
| not exposed in the 0.0.8 release | 18 |
| open release-chain follow-up | 2 |
| partly fixed release-chain follow-up | 1 |
| process or seed policy | 4 |
| refuted | 1 |
| thesis correction / intake | 6 |
| **total** | **133** |

Source groups: 44 original internal-audit findings, 42 original external-review findings, and 47 catch-up findings. No adapter finding is counted twice.
