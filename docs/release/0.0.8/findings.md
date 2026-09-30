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
repository). State as of 2026-09-30, 12:30 UTC.

## Planner integration

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| A1 | internal audit | prediction_planner model trained on pedestrian velocities rotated twice | P2 | confirmed | disclosed; collectors fixed in [#10011](https://github.com/ll7/robot_sf_ll7/pull/10011) (open); retrain in 0.0.9 [#10018](https://github.com/ll7/robot_sf_ll7/issues/10018) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| A2 | internal audit | prediction_planner discards the wall part of its occupancy cost | P2 | confirmed | disclosed as a method limit (decision D-021, [#10011](https://github.com/ll7/robot_sf_ll7/pull/10011)); wall term in 0.0.9 [#10018](https://github.com/ll7/robot_sf_ll7/issues/10018) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| A3 | internal audit | predictive_mppi silently caps the horizon at the model's 8 steps | P2 | confirmed | fixed in [#10011](https://github.com/ll7/robot_sf_ll7/pull/10011) (open): fails fast, configs state 8 x 0.1 s | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| A4 | internal audit | prediction heading lattice collapses to three turn rates | P3 | confirmed | fixed in [#10011](https://github.com/ll7/robot_sf_ll7/pull/10011) (open) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| A5 | internal audit | predictive_mppi v1 model may have been trained on targets that switch pedestrians | P2 | plausible (code confirmed, dataset attribution conditional) | no disposition recorded | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B1 | internal audit | SA-CADRL never drives faster than 1.0 m/s | P2 | confirmed | kept on purpose and disclosed (decision D-018, [#10008](https://github.com/ll7/robot_sf_ll7/pull/10008)) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B2 | internal audit | SA-CADRL observes only the 3 nearest pedestrians | P2 | confirmed | fixed in [#10008](https://github.com/ll7/robot_sf_ll7/pull/10008) (open): 19 agents | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B3 | internal audit | SA-CADRL discrete turns all saturate at the turn-rate limit | P2 | confirmed (mechanism) | kept and documented as a method note ([#10008](https://github.com/ll7/robot_sf_ll7/pull/10008)) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B4 | internal audit | sampling planner rollout ignores angular acceleration and current turn rate | P2 | confirmed by the fix lane | fixed in [#10008](https://github.com/ll7/robot_sf_ll7/pull/10008) (open): forecasts through the native drive | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B5 / D-F6 | internal audit | occupancy grid draws pedestrians with 0.35 m instead of 0.4 m radius | P3 | confirmed | no disposition recorded | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B6 | internal audit | SA-CADRL and sampling silently fall back to dt 0.1 | P3 | confirmed | fixed in [#10009](https://github.com/ll7/robot_sf_ll7/pull/10009) (open) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B7 | internal audit | guarded PPO goal selection can target the [0,0] placeholder on the final leg | P3 | confirmed (unreachable in practice) | no disposition recorded | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| B8 | internal audit | PPO observation alignment fills missing keys with zeros without a record | P3 | confirmed (no key missing today) | no disposition recorded | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| C-F1 | internal audit | hybrid v4 static-escape and corridor-transit keys never act | P2 | confirmed | disclosed (decision D-010); fail-closed key check in 0.0.9 [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) | [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) |
| C-F2 | internal audit | all four hybrids use parameter overrides keyed on scenario names | P2 | confirmed | disclosed (decision D-010, [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006)) | [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) |
| C-F3 | internal audit | v4 tuning ran without the perpendicular_traffic override used at release | P3 | confirmed | disclosed ([#10006](https://github.com/ll7/robot_sf_ll7/issues/10006), [#10002](https://github.com/ll7/robot_sf_ll7/issues/10002)) | [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) |
| C-F4 | internal audit | v3 centre-distance gates read as surface clearance in v4 | P3 | plausible | 0.0.9 issue [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) (confirm and document) | [#10006](https://github.com/ll7/robot_sf_ll7/issues/10006) |
| D-F1 | internal audit | social_force planner integrates with dt 0.5 s instead of 0.1 s | P1 | confirmed | fixed in [#10009](https://github.com/ll7/robot_sf_ll7/pull/10009) (open) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F2 | internal audit | ORCA adapter drives forward at full speed whatever the heading error | P1 | confirmed in code | fixed in [#10009](https://github.com/ll7/robot_sf_ll7/pull/10009) (open), decision D-020 | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F3 | internal audit | ORCA solver runs with 3.0 m/s on a 2.0 m/s robot | P2 | confirmed (release template already 2.0 m/s) | fixed in [#10009](https://github.com/ll7/robot_sf_ll7/pull/10009) (open) for the candidate manifest | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-P1 | internal audit | release social_force keeps the legacy pedestrian kernel | P2 | refuted for the release (template selects the corrected kernel) | refuted | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| X-ADP-1 | external review | adapter rollouts reach the target velocity instantly, including an instant stop | P1 | not verified yet | triage pending; parts may be covered by [#9926](https://github.com/ll7/robot_sf_ll7/pull/9926) (merged) - unverified | adapter review, 2026-09-30 |
| X-ADP-2 | external review | bounded_v2 sampler turns instantly and integrates the pose differently | P1 | not verified yet | triage pending; same finding as B4, fixed in [#10008](https://github.com/ll7/robot_sf_ll7/pull/10008) (open) - unverified against this witness | adapter review, 2026-09-30 |
| X-ADP-3 | external review | guard and Risk-DWA 'TTC' is time to closest approach, not to collision | P1 | not verified yet | triage pending; [#9926](https://github.com/ll7/robot_sf_ll7/pull/9926) (merged) introduced contact-based TTC under surface_v2 - unverified | adapter review, 2026-09-30 |
| X-ADP-4 | external review | moving pedestrians also counted as static obstacles at old positions | P1 | not verified yet | triage pending | adapter review, 2026-09-30 |
| X-ADP-5 | external review | non-finite SA-CADRL network output becomes a full-speed turn | P2 | not verified yet | triage pending (fault injection only) | adapter review, 2026-09-30 |
| X-ADP-6 | external review | native-command parser accepts vx/vy as (v, w) | P2 | not verified yet | triage pending (reviewer: no roster arm uses it) | adapter review, 2026-09-30 |

## Metrics and reporting

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| D-F4 | internal audit | metric goal is the waypoint active at episode end, not the route goal | P2 | confirmed | fixed in [#10014](https://github.com/ll7/robot_sf_ll7/pull/10014) (open), decision D-029 | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F5 | internal audit | deadlock detector flags any slow episode and counts collisions | P2 | confirmed | fixed in [#10014](https://github.com/ll7/robot_sf_ll7/pull/10014) (open), decision D-027 | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F7 | internal audit | jerk_mean is not divided by dt | P3 | confirmed | fixed in [#10014](https://github.com/ll7/robot_sf_ll7/pull/10014) (open) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-F8 | internal audit | step-index conventions one step off (time to goal, first path segment) | P3 | confirmed | fixed in [#10014](https://github.com/ll7/robot_sf_ll7/pull/10014) (open) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-P2 | internal audit | recorded pedestrian forces lag positions by one step | P3 | plausible | no disposition recorded | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| D-N1 | internal audit | shortest-path reference uses no inflation, so it hugs walls | P3 | confirmed | no disposition recorded (related notes in [#10022](https://github.com/ll7/robot_sf_ll7/issues/10022)) | [#10007](https://github.com/ll7/robot_sf_ll7/issues/10007) |
| X-STAT-1 | external review | table means bypass the foresight eligibility filter | P1 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (open, review FIX) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-2 | external review | Spearman over a key intersection keeps absolute ranks | P1 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (open) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-3 | external review | paired differences keep the last duplicate row | P2 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (open) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-4 | external review | SNQI v2 mean and confidence interval use different weightings | P2 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (open) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-5 | external review | bootstrap seed 0 silently becomes 123 | P2 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (open) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |
| X-STAT-6 | external review | numeric group_by values fall back to the algorithm | P2 | confirmed | fixed in [#10019](https://github.com/ll7/robot_sf_ll7/pull/10019) (open) | [#10015](https://github.com/ll7/robot_sf_ll7/issues/10015) |

## Pedestrian simulation

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| PED-P1-1 | internal audit | wall force reaches far and adds up, so pedestrians stop before narrow openings | P1 | confirmed (qualified by the external review) | disclosed (decision D-038); fix in 0.0.9 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P1-2 | internal audit | goal-only single pedestrians walk into obstacles and stay stuck | P1 | confirmed | disclosed (decision D-038); fix in 0.0.9 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P1-3 | internal audit | declared pedestrian speed is multiplied by 1.3 for the cap | P1 | confirmed | kept as the standard convention and documented (decision D-039, [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017)) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-1 | internal audit | route respawns draw from the global random generator | P2 | confirmed (mechanism) | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (open, review FIX) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-2 | internal audit | simulation_config.groups is ignored | P2 | confirmed | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (open; meaning being corrected, decision D-035) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-3 | internal audit | per-scenario robot_radius override of the reaction force is replaced | P2 | confirmed | not exposed (not in the 0.0.8 grid) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-4 | internal audit | pedestrian moved off the robot at reset lands 1.5 m away and keeps walking in | P2 | confirmed | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (open), decision D-037 | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P2-5 | internal audit | resetting the same env twice leaves route navigators behind | P2 | confirmed | not exposed (benchmark builds one env per episode); 0.0.9 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-1 | internal audit | reaction-force range uses radius 0.35 m instead of 0.4 m | P3 | confirmed | 0.0.9 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-2 | internal audit | pedestrians overlap at spawn; group repulsion starts too late | P3 | confirmed | 0.0.9 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-3 | internal audit | pedestrians with speed 0 cannot step aside | P3 | confirmed | 0.0.9 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-4 | internal audit | relaxation time is effectively 0.448 s and configured twice | P3 | confirmed | 0.0.9 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-5 | internal audit | respawned groups keep their end-of-route velocity | P3 | confirmed (code) | 0.0.9 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-6 | internal audit | legacy wall force drops to zero at contact | P3 | plausible | covered by the wall-law fix, 0.0.9 [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-P3-7 | internal audit | route completion ignores the waypoint index | P3 | plausible | no disposition recorded | [#10017](https://github.com/ll7/robot_sf_ll7/issues/10017) |
| PED-V1 | internal audit | join_group and leave_group never contain a group | P2 | confirmed | disclosed (decision D-041); fix in 0.0.9 [#10028](https://github.com/ll7/robot_sf_ll7/issues/10028) | [#10028](https://github.com/ll7/robot_sf_ll7/issues/10028) |
| X-PED-1 | external review | group gaze scales with 1/goal distance, can push forward, ignores field of view | P1 | confirmed by execution | disclosed (decision D-040); fix in 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-2 | external review | intra-group repulsion weakens as members get closer | P1 | confirmed by execution | disclosed (decision D-040); fix in 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-3 | external review | a waiting pedestrian can drift during an authored wait | P1 | confirmed by execution (drift at most 0.13 m in release) | 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-4 | external review | goal-less roles start with an eastward velocity | P1 | confirmed by execution (path deviation at most 0.2 m) | 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-5 | external review | coincident robot and pedestrian centres divide by zero | P2 | confirmed by execution | not exposed (closest approach 1.33 m); 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-6 | external review | join/leave reach the group forces one step late | P2 | confirmed by execution | not exposed (join never completes); 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-7 | external review | reset keeps changed group membership | P2 | confirmed by execution | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (open); not exposed in the benchmark | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-8 | external review | timed holds behave wrongly at boundary values | P2 | confirmed by execution | not exposed; 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-9 | external review | joining group 0 does not latch an automatic target | P2 | confirmed by execution | not exposed; 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-10 | external review | FastPysfWrapper queries use the old social kernel | P2 | confirmed by execution | not exposed (no 0.0.8 arm uses the wrapper); 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-11 | external review | archetype speed factors ignored in crowded zones | P2 | confirmed by execution | not exposed (no release scenario uses archetypes); 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |
| X-PED-12 | external review | archetype labels depend on dictionary key order | P2 | confirmed by execution | not exposed; 0.0.9 [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) | [#10027](https://github.com/ll7/robot_sf_ll7/issues/10027) |

## Scenarios and maps

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| X-SCN-1 | external review | planner random draws change pedestrian goals and respawns | P2 | confirmed | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (open), decision D-034 | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-2 | external review | rectangular spawn and goal zones sampled as triangles | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (open), decisions D-042, D-043 | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-3 | external review | crowded-zone behaviours lose obstacle constraints for later goals | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (open) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-4 | external review | pedestrian spawns check centres, not footprints | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (open) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-5 | external review | unknown simulation_config keys accepted and ignored | P2 | confirmed | fixed in [#10024](https://github.com/ll7/robot_sf_ll7/pull/10024) (open), decision D-036 | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-6 | external review | episode seed does not seed optional archetype assignment | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (open) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-7 | external review | included scenario's map shadowed by a same-named root file | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (open) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |
| X-SCN-8 | external review | sparse SVG zone indices compacted | P2 | confirmed | fixed in [#10026](https://github.com/ll7/robot_sf_ll7/pull/10026) (open) | [#10020](https://github.com/ll7/robot_sf_ll7/issues/10020) |

## Campaign integrity and 0.0.7-to-0.0.8 comparison

| ID | Source | Finding | Severity | Verified | Disposition | Link |
|---|---|---|---|---|---|---|
| X-CMP-1 | external review | equal-sized sets of wrong episodes pass campaign integrity | P1 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (open, review FIX) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-2 | external review | correct slots with the wrong algorithm or config pass integrity | P2 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (open) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-3 | external review | checksum check verifies a different file than the payload file | P2 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (open) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-4 | external review | an ordinary slot can resolve to another algorithm by scenario override | P2 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (open) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-5 | external review | comparison worker rejects the frozen v4 config (missing lineage) | P2 | confirmed | fixed in [#10001](https://github.com/ll7/robot_sf_ll7/pull/10001) (merged) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-6 | external review | comparison worker rejects the approved ORCA hand-off | P2 | confirmed | fixed in [#10001](https://github.com/ll7/robot_sf_ll7/pull/10001) (merged), decision D-024 | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |
| X-CMP-7 | external review | comparator checks a seeded scenario against an unseeded expectation | P2 | confirmed | fixed in [#10023](https://github.com/ll7/robot_sf_ll7/pull/10023) (open) | [#10021](https://github.com/ll7/robot_sf_ll7/issues/10021) |

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

## Counts per disposition

Each finding is counted once, under its main disposition (the first one in
its row).

| Disposition | Findings |
|---|---|
| fixed in a pull request (merged or open) | 40 |
| disclosed in 0.0.8 (most also have a 0.0.9 issue) | 13 |
| 0.0.9 issue only | 9 |
| not exposed in the 0.0.8 release | 9 |
| refuted | 1 |
| to be corrected in the thesis text | 1 |
| triage pending (adapter review) | 6 |
| no disposition recorded | 7 |
| **total** | **86** |

Of the 40 fixed findings, 5 are fixed in merged pull requests and 35 in pull requests that are still open.

By source: 44 from internal audits, 42 from external reviews.
