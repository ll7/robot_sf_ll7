# Release 0.0.8: decision ledger

New decisions are added here when they are made, each with its enforcing test or "no test yet".

This file records the decisions taken while preparing benchmark release 0.0.8
(mainly 2026-09-29 and 2026-09-30). For each decision it states the question,
the choice, the reason, who decided, what was rejected, where it is
implemented, and which test enforces it. Reviews can then check the code
against the intention written here.

How to read the fields:

- **Decided by.** "author" means the repository author ruled. "delegated"
  means the orchestrating agent decided under the author's standing
  delegation; the author reviews these afterwards.
- **Implemented in.** Pull requests in this repository are written `#N`. Pull
  requests in the private operations repository are written `private-ops #N`;
  that repository is not public.
- **Enforcing test.** Written as `file::test`. "(open PR)" means the test
  exists on the pull request branch but is not yet on `main`. "no test yet"
  means nothing enforces the decision today; those decisions are listed again
  at the end.
- **unverified** marks a statement that could not be checked against GitHub or
  the current branch heads when this file was written (2026-09-30, 12:30 UTC).

The findings that led to these decisions are listed in
[findings.md](findings.md). The full lane and review reports are kept in the
private review archive (`docs/reviews/0.0.8/` in the private operations
repository).

## Release process

### D-001: Which new findings still block 0.0.8 (freeze rule)
- **Date:** 2026-09-30
- **Question:** Each review round was finding new defects and reopening the
  freeze. Which findings may still block the 0.0.8 release?
- **Choice:** After the fixes already in flight, a new finding blocks 0.0.8
  only if it is a P1 that measurably changes reported numbers. Every other
  finding gets an issue on milestone 0.0.9 and, where it affects how results
  should be read, a disclosure line in the release notes and the thesis.
- **Reason:** Without a rule, every review round reopened the freeze and 0.0.8
  never arrived. Tracking the rest on 0.0.9 keeps the defects visible.
- **Decided by:** author.
- **Rejected alternatives:** block on every P1 and P2 finding (the earlier
  rule, D-002); block on every finding.
- **Implemented in:** process rule; milestone 0.0.9.
- **Enforcing test:** no test yet.

### D-002: Earlier admission rule for side-quest findings (superseded by D-001)
- **Date:** 2026-09-30 (morning)
- **Question:** Do findings from side work (audits, reviews) block the freeze?
- **Choice:** A finding blocked the freeze only if it was P1 or P2 (changes
  results). P3 findings and review nits on critical pull requests became
  issues, not extra review rounds.
- **Reason:** Keep the critical path moving while not losing result-changing
  defects.
- **Decided by:** delegated.
- **Rejected alternatives:** treat every review finding as a merge blocker.
- **Implemented in:** process rule, stated on tracker #10013. Tightened by D-001.
- **Enforcing test:** no test yet.

### D-003: Fix result-changing planner defects before the freeze
- **Date:** 2026-09-30
- **Question:** What to do with the defects found by the read-only planner
  hunt (#10007)?
- **Choice:** Fix every defect that changes results before the 0.0.8 freeze,
  each through the behaviour-change gate; disclose method facts that are not
  defects. The freeze waits for these fixes.
- **Reason:** A planner that sees or does something different from what the
  benchmark claims silently corrupts the comparison.
- **Decided by:** delegated.
- **Rejected alternatives:** release 0.0.8 first and fix in 0.0.9.
- **Implemented in:** #9926, #9995, #10008, #10009, #10011, #10014 (see the
  decisions below).
- **Enforcing test:** no test yet for the rule itself; the individual fixes
  carry their own tests.

### D-004: 0.0.8 runs from a pinned freeze commit
- **Date:** 2026-09-29
- **Question:** Must `main` stand still while 0.0.8 is calibrated and run?
- **Choice:** Calibration, smoke run, campaign, comparison and tag all use one
  pinned commit. `main` may keep moving. The branch `release/0.0.8-freeze`
  points to the #9932 merge commit (`5c27a404`). A new freeze commit is chosen
  only when a blocking defect forces it, with the reason recorded in #9668.
  The pre-freeze fixes above mean a new freeze commit is still to be chosen
  (tracker #10013, "Freeze" step).
- **Reason:** Holding `main` would stall every other pull request; a pinned
  commit gives the same reproducibility.
- **Decided by:** author.
- **Rejected alternatives:** freeze `main` itself for the duration.
- **Implemented in:** #9932; the private mint binds a freeze commit
  (private-ops #407).
- **Enforcing test:** no test yet in this repository. The private mint refuses
  inputs that are not the pinned freeze blobs
  (`ops/jobs/scripts/test_full_campaign_authority.py::test_forged_scientific_authority_should_be_refused`,
  private).

### D-005: Rehearsal campaign before the real campaign
- **Date:** 2026-09-30
- **Question:** How to catch pipeline failures before the 20,160-episode
  campaign?
- **Choice:** Run the whole 0.0.8 pipeline at small scale on development seeds
  1001-1003, from an integration branch of `main` plus the open fix pull
  requests, in parallel with the merges.
- **Reason:** It exercises the full path without waiting for every merge and
  without touching the held-out evaluation seeds.
- **Decided by:** author.
- **Rejected alternatives:** go straight from merges to the full campaign.
- **Implemented in:** operational (no pull request).
- **Enforcing test:** no test yet.

### D-006: One last external review for 0.0.8
- **Date:** 2026-09-30
- **Question:** Should external model reviews continue during the freeze?
- **Choice:** One last external review for 0.0.8, on the planner adapters.
  Later external reviews go to 0.0.9 or to checks of thesis claims.
- **Reason:** Same as D-001: reviews kept reopening the freeze.
- **Decided by:** author.
- **Rejected alternatives:** keep reviewing every area before release.
- **Implemented in:** process rule. The adapter review arrived on 2026-09-30
  and is listed in findings.md, triage pending.
- **Enforcing test:** no test yet.

### D-007: Behaviour-change gate for planner and simulator changes
- **Date:** 2026-09-30
- **Question:** How to catch planner or simulator changes that aggregate
  metrics hide?
- **Choice:** Pull requests that change planner behaviour, adapters,
  simulator or drive dynamics, maps, scenario configs, the release campaign
  config, row writers or metrics run the empty-world sweep on development
  seeds, classify every new failure, and get a refute review. The gate does not
  require a success rate. Scenarios that are infeasible by design are listed in
  a checked exception list.
- **Reason:** The empty-world sweep found several defects that plausible
  aggregate numbers had hidden. A success-rate target would invite tuning
  towards the metric.
- **Decided by:** author (approved on #10000).
- **Rejected alternatives:** a success-rate threshold.
- **Implemented in:** applied by hand in 0.0.8; automation is #10000
  (milestone 0.0.9).
- **Enforcing test:** no test yet (no CI check requires a gate receipt).

### D-008: Held-out evaluation seeds 111-140 are never stepped outside the campaign
- **Date:** 2026-09-30
- **Question:** Which seeds may tests and development runs use?
- **Choice:** Every integer from 111 to 140 is reserved for the release
  evaluation. Tests and diagnostics use development seeds 1001-1030. An earlier
  lane rule that treated seed 123 as allowed was wrong and was corrected;
  existing tests that step held-out seeds are inventoried and moved (#10010).
- **Reason:** Tuning or bounding anything on held-out outcomes would leak
  evaluation information into development.
- **Decided by:** delegated.
- **Rejected alternatives:** allow held-out seeds in tests that only check
  pass/fail.
- **Implemented in:** the diff gate `scripts/validation/check_seed_holdout_diff.py`;
  test moves in #10009 and #10026; whole-tree inventory in #10010 (0.0.9).
- **Enforcing test:**
  `tests/validation/test_check_seed_holdout_diff.py::test_planner_seed_patterns_fail`
  (checks added lines only; a whole-tree and runtime check does not exist yet).

## Planner integration

### D-009: Hybrid v4 parameters frozen from the pre-registered tuning
- **Date:** 2026-09-29
- **Question:** Which parameters do the four hybrid v4 release slots run?
- **Choice:** Apply the pre-registered ranking rule (fewer collisions, then
  more completions, then lower success-only goal time) as written. The static
  fast-progress slot gets goal-progress weight 4.5; the continuous slot keeps
  the baseline v4 parameters; the two scenario-adaptive slots are untuned.
  The 0.0.7-to-0.0.8 comparison reports this as the rule's output and claims no
  significant improvement from tuning.
- **Reason:** Margins were small (0 against 1 collision in 120 episodes).
  Choosing differently after seeing results would be a post-hoc protocol change.
- **Decided by:** delegated (decision comment on #9748).
- **Rejected alternatives:** pick a trial by judgement after seeing the results.
- **Implemented in:** #9932 (merged).
- **Enforcing test:**
  `tests/benchmark/test_release_0_0_8_hybrid_v4_slots.py::test_v4_slots_bind_reviewed_frozen_configs_for_real_v4_twins`,
  `tests/benchmark/test_release_0_0_8_hybrid_v4_slots.py::test_frozen_v4_slot_requires_resolved_v4_variant`.

### D-010: Hybrid v4 inert keys and scenario-name overrides are disclosed, not changed
- **Date:** 2026-09-30
- **Question:** The v4 "static escape" and corridor-transit keys never act, and
  all four hybrids use parameter overrides keyed on scenario names. Change the
  frozen arms?
- **Choice:** No behaviour change for 0.0.8. Disclose that the static-escape
  and corridor-transit keys are inert, that all four hybrids carry
  scenario-name overrides (the adaptive arms in four named scenarios), and that
  the #9748 tuning ran without one override. For 0.0.9, add a check that fails
  on configuration keys that are present but never read.
- **Reason:** Changing the frozen arms would need a re-freeze and new tuning.
  These are method facts that change how the arms should be described, not
  wrong results.
- **Decided by:** delegated (#10006).
- **Rejected alternatives:** run the escape rules inside the continuous branch
  (a behaviour change); drop the scenario overrides.
- **Implemented in:** disclosure (#10006, #10002, thesis intake); fail-closed
  key check in 0.0.9.
- **Enforcing test:** no test yet.

### D-011: Planner horizons keep their length in seconds after the 0.1 s step fix
- **Date:** 2026-09-29
- **Question:** When planner configs move from a 0.2 s to the true 0.1 s
  control step, should the step counts stay (halving the horizon) or double?
- **Choice:** Keep the horizon in seconds and double the step counts.
- **Reason:** Matching the 0.1 s control step is the correction; the horizon
  length is the method's parameter and must not change as a side effect.
- **Decided by:** delegated (#9926 review).
- **Rejected alternatives:** keep step counts and silently halve the lookahead.
- **Implemented in:** #9926 (merged).
- **Enforcing test:**
  `tests/planner/test_release_horizons.py::test_release_step_count_retains_historical_horizon_seconds`.

### D-012: Prediction models run at their training cadence (8 x 0.1 s)
- **Date:** 2026-09-30
- **Question:** At what time step are the learned pedestrian forecasts used by
  prediction_planner and predictive_mppi?
- **Choice:** Use the training cadence: 8 outputs, 0.1 s apart, a 0.8 s
  window. A configured horizon longer than the model output fails fast instead
  of being capped silently. The release configs state 8 x 0.1 s.
- **Reason:** The training collector recorded one frame per 0.1 s step. The old
  0.2 s reading stretched forecasts in time, so pedestrians looked half as
  fast. An earlier review claim of a 0.2 s training window was withdrawn after
  the training source was read.
- **Decided by:** delegated (#9926 comment, #10011 verdict).
- **Rejected alternatives:** restore a 1.6 s window; silently hold the last
  forecast beyond 8 steps.
- **Implemented in:** #9926 (merged), #10011 (open).
- **Enforcing test:**
  `tests/planner/test_release_horizons.py::test_release_checkpoint_effective_window_is_zero_point_eight_seconds`;
  `tests/planner/test_prediction_fxp_regressions.py::test_mppi_rejects_24_steps_from_eight_step_forecast`
  (open PR).

### D-013: Risk-DWA gets a real dynamic window for 0.0.8
- **Date:** 2026-09-29
- **Question:** Risk-DWA scored commands the drive cannot reach in one step.
  Fix or keep?
- **Choice:** The 0.0.8 Risk-DWA binding gets a versioned dynamic window
  bounded by the drive's acceleration limits and control step. The old sampling
  stays reproducible for 0.0.7.
- **Reason:** A dynamic-window method must score only velocities reachable in
  the next interval; otherwise the scored rollout is not the executed one.
- **Decided by:** delegated (#9926 review).
- **Rejected alternatives:** keep unbounded sampling.
- **Implemented in:** #9926 (merged).
- **Enforcing test:**
  `tests/planner/test_risk_dwa.py::test_release_dynamic_window_scores_only_next_step_reachable_commands`.

### D-014: Residual empty-world failures are method limits, not tuned away
- **Date:** 2026-09-30
- **Question:** After the #9926 fixes, 28 empty-world non-successes remained
  for risk_dwa, predictive_mppi and guarded_ppo. Fix or record?
- **Choice:** Record them as method limits (narrow-doorway refusal, MPPI
  stasis classes, guarded_ppo finite-horizon limits) in the release docs. The
  one guarded_ppo contact is stated as evidence against any global
  collision-safety claim for that arm.
- **Reason:** Every case was classified and none is a correctness defect.
  Tuning for success would violate the faithfulness rule of #9750.
- **Decided by:** delegated (#9926 verdict).
- **Rejected alternatives:** tune the planners until the empty-world sweep
  passes.
- **Implemented in:** docs task #10002 (open).
- **Enforcing test:** no test yet.

### D-015: guarded_ppo keeps its one-step waypoint lookahead
- **Date:** 2026-09-29
- **Question:** At a waypoint boundary the guarded_ppo outer guard targets the
  next waypoint while its fallback targets the current one. Unify?
- **Choice:** Keep the outer guard's lookahead, document it and pin it with a
  test, so the "one active-waypoint target" claim is stated precisely.
- **Reason:** It is guarded_ppo's own method, it lasts one step, and the
  simulator advances the waypoint on the next step.
- **Decided by:** delegated (#9891 review).
- **Rejected alternatives:** force the outer guard onto the current waypoint.
- **Implemented in:** #9891 (merged).
- **Enforcing test:**
  `tests/planner/test_guarded_ppo.py::test_guarded_ppo_outer_guard_looks_ahead_for_one_waypoint_boundary_step`.

### D-016: PPO checkpoint outputs are velocity changes; keep the correct reading
- **Date:** 2026-09-30
- **Question:** The PPO checkpoints output velocity changes, but the benchmark
  read them as absolute targets. With the correct reading PPO scores worse
  (39 % against 83 % success on the training motion model, development seeds).
  Which reading does the release use?
- **Choice:** Use the correct reading (next target = current velocity plus
  the output, then the release limits). Report the drop; do not keep the wrong
  reading because it scored better.
- **Reason:** The old reading scored better by accident. The drop comes from
  the training-versus-benchmark motion mismatch, which D-017 addresses.
- **Decided by:** delegated (#9995 verdicts and diagnostic).
- **Rejected alternatives:** keep the absolute-target reading.
- **Implemented in:** #9995 (open, approved).
- **Enforcing test:**
  `tests/baselines/test_ppo_action_semantics.py::test_signed_delta_precedes_release_clipping`,
  `tests/baselines/test_ppo_action_semantics.py::test_release_checkpoints_fail_closed_without_delta_declaration`
  (open PR).

### D-017: Retrain PPO on the benchmark motion model
- **Date:** 2026-09-30
- **Question:** The PPO policies were trained with 3 m/s, instant velocity
  change, reverse driving and an instant turn rate; the benchmark robot has
  2 m/s, 1 m/s^2, no reverse and a turn-rate ramp. What to do?
- **Choice:** Retrain the same two recipes (plain PPO and the guarded PPO base
  policy), seeds 1001 and 1002, with training and evaluation using the same
  applied command and the benchmark drive, including the angular acceleration
  limit. The retrained policies form a supplementary arm on the same freeze.
  The one-factor ablation showed the turn-rate ramp alone explains nearly the
  whole drop.
- **Reason:** Model-based planners take the drive limits as parameters; a
  learned policy has to see them in training.
- **Decided by:** author. On 2026-09-30 the author approved retraining PPO on
  the benchmark motion model ("that is good"), and approved the three-part PPO
  sensitivity study (the one-factor ablation, the retraining, and a thesis
  section framed as a defect found by the checks followed by a designed
  study) with "yes, do all three". The thesis section is planned in
  ll7/diss#3023.
- **Rejected alternatives:** evaluate only the old checkpoints; replace the
  release PPO arm with the retrained one.
- **Implemented in:** #10003 (open, approved); private-ops #408, #409, #411,
  #414, #418 (merged).
- **Enforcing test:**
  `tests/training/test_ppo_release_contract.py::test_training_release_applied_command_parity`
  (open PR).

### D-018: SA-CADRL is evaluated as published: preferred speed 1.0 m/s, 19 observed agents
- **Date:** 2026-09-30
- **Question:** SA-CADRL never exceeds 1.0 m/s and observed only the 3 nearest
  pedestrians. Raise the speed to 2.0 m/s, and observe how many agents?
- **Choice:** Keep the preferred speed at 1.0 m/s and observe up to 19
  agents, the network's capacity. Keep the per-step heading mapping (turns at
  the rate limit). Refuse, before inference, a configured agent count above 19.
- **Reason:** Both speeds are inside the checkpoint's training range, but the
  benchmark robot brakes at 1 m/s^2 while the training dynamics stop at once.
  At 2.0 m/s the stop action does not stop in time (success 2/30 and 3/30
  against 11/30 and 14/30 at 1.0 m/s, development seeds). 19 agents raised
  success from 11 to 14 of 30.
- **Decided by:** delegated (#10008 review comment).
- **Rejected alternatives:** 2.0 m/s; 3 agents; a heading-hold adapter (would
  change the checkpoint's decision timing).
- **Implemented in:** #10008 (open, approved).
- **Enforcing test:**
  `tests/planner/test_fxs_release_contract.py::test_release_preferred_speed_reaches_checkpoint_host_input`,
  `tests/planner/test_fxs_release_contract.py::test_release_observes_nineteen_nearest_agents_without_padding_in_sequence`,
  `tests/planner/test_fxs_release_contract.py::test_checkpoint_rejects_configured_agent_count_above_nineteen`
  (open PR). The heading mapping has no test yet.

### D-019: Planners read the simulation clock and fail closed without it
- **Date:** 2026-09-30
- **Question:** social_force read a missing clock as 0.5 s; ORCA, SA-CADRL and
  sampling used a 0.1 s default by coincidence. How should planners get dt?
- **Choice:** One shared resolver reads the simulation clock from the
  observation (flat `sim_timestep` or a nested block), accepts a top-level `dt`
  only as a last, recorded source, and raises on a missing or invalid primary
  clock instead of falling through to a default.
- **Reason:** A silent default made the social-force arm integrate at five
  times the real step.
- **Decided by:** delegated.
- **Rejected alternatives:** keep per-planner defaults.
- **Implemented in:** #10009 (open, approved). Planners outside the release
  roster are listed on #10007 for 0.0.9.
- **Enforcing test:**
  `tests/benchmark/test_issue_10007_fxb.py::test_social_force_uses_real_flat_sim_dt`,
  `tests/benchmark/test_issue_10007_fxb.py::test_invalid_primary_clock_does_not_fall_through_to_dt`,
  `tests/benchmark/test_issue_10007_fxb.py::test_timestep_consumers_reject_missing_or_invalid_dt`
  (open PR).

### D-020: ORCA commands only the forward part of its velocity
- **Date:** 2026-09-30
- **Question:** The ORCA adapter drove forward at full ORCA speed even when
  ORCA's velocity pointed sideways or backwards. How to map ORCA's velocity
  onto a forward-only robot?
- **Choice:** Command the speed times the positive cosine of the heading error
  (with the configured slowdown and occupancy penalty applied to the requested
  speed); sideways or backward targets turn in place. The ORCA solver runs at
  the robot's 2.0 m/s.
- **Reason:** Projecting onto the forward axis is the usual heuristic for a
  forward-only robot. It is a faithful adaptation, not a guarantee-preserving
  one, and the docs say so.
- **Decided by:** delegated. Decision record: the #10009 verdict comment
  (https://github.com/ll7/robot_sf_ll7/pull/10009#issuecomment-5907593590),
  which accepted the rule implemented by the fix lane.
- **Rejected alternatives:** keep the old cap-only slowdown; a
  guarantee-preserving nonholonomic ORCA variant (a new method).
- **Implemented in:** #10009 (open, approved).
- **Enforcing test:**
  `tests/benchmark/test_issue_10007_fxb.py::test_orca_default_adapter_projects_forward_component`,
  `tests/benchmark/test_issue_10007_fxb.py::test_orca_release_binding_sets_native_solver_speed_cap`
  (open PR).

### D-021: prediction_planner keeps no wall term in 0.0.8
- **Date:** 2026-09-30
- **Question:** prediction_planner discards the static-obstacle part of its
  occupancy cost. The attempted fix wired the term in, but it never fired in
  real episodes. Build a working wall term now?
- **Choice:** Do not build a new wall-avoidance method during the freeze.
  Revert to the historical pedestrian-only cost, state "no effective
  static-obstacle term" as a method limitation, and keep a real-grid test that
  documents it. A footprint-aware wall term is planned for 0.0.9 (#10018).
- **Reason:** A new method during the freeze would need its own validation;
  the reverted code reproduces the reviewed episodes exactly.
- **Decided by:** delegated (#10011 review comment).
- **Rejected alternatives:** ship the non-firing term; design a new term now.
- **Implemented in:** #10011 (open, approved).
- **Enforcing test:**
  `tests/planner/test_prediction_fxp_regressions.py::test_prediction_real_doorway_grid_has_no_effective_static_obstacle_cost`
  (open PR).

### D-022: The prediction model is not retrained for 0.0.8
- **Date:** 2026-09-30
- **Question:** The prediction_planner model was trained on pedestrian
  velocities rotated twice. Retrain before 0.0.8?
- **Choice:** Fix the data collectors now; retrain the model with a new model
  ID in 0.0.9 (#10018). Disclose the training defect for 0.0.8.
- **Reason:** Retraining needs new data collection and validation that does
  not fit the freeze.
- **Decided by:** delegated.
- **Rejected alternatives:** retrain now; keep the collectors unchanged.
- **Implemented in:** collector fix in #10011 (open); retrain in #10018.
- **Enforcing test:**
  `tests/planner/test_prediction_fxp_regressions.py::test_collector_preserves_already_ego_velocity_at_north_heading`
  (open PR; enforces the collector fix, not the retrain).

### D-023: guarded_ppo rows keep "ppo" as the base-policy algorithm
- **Date:** 2026-09-30
- **Question:** guarded_ppo rows carry `algorithm_metadata.algorithm = "ppo"`.
  Relabel it?
- **Choice:** Keep it; it is the base-policy identity. Consumers that separate
  arms must use the top-level `algo`. Every contributing episode must carry
  the complete guarded identity.
- **Reason:** The acceptance contract of #9996 treats relabelling the base as
  guarded_ppo as a wrong base.
- **Decided by:** delegated (#10013 comments).
- **Rejected alternatives:** relabel the base algorithm.
- **Implemented in:** #9996 (merged).
- **Enforcing test:**
  `tests/benchmark/test_fallback_policy.py::test_shield_dictionary_requires_complete_identity`,
  `tests/benchmark/test_guarded_summary_identity.py::test_every_contributing_episode_must_bind_identity`.

### D-024: Approved ORCA hand-offs of the adaptive hybrids are the only allowed slot override
- **Date:** 2026-09-30
- **Question:** The comparison worker rejected the approved ORCA hand-off of
  the two scenario-adaptive hybrids. How wide should the exemption be?
- **Choice:** Exempt only the exact approved tuple (algorithm, base config
  file) from one source of truth; reject every other override on a v4 slot.
- **Reason:** A wide exemption would let a slot silently run another planner.
- **Decided by:** delegated (#10001 review).
- **Rejected alternatives:** exempt any ORCA override.
- **Implemented in:** #10001 (merged).
- **Enforcing test:**
  `tests/analysis/test_pinned_successor_lineage.py::test_real_orca_handoff_acceptance_keeps_substitutions_rejected`.

## Episode budgets

### D-025: Keep the authored per-scenario step budgets
- **Date:** 2026-09-30
- **Question:** 0.0.7 was described as "H600", but 38 of 48 scenarios had a
  shorter simulator limit (400 or 500 steps). Which budget is authoritative
  for 0.0.8?
- **Choice:** Keep the authored per-scenario budgets and declare them
  explicitly (a pinned schedule file). An undeclared mismatch is refused at
  admission, a budget timeout is labelled `max_steps`, and rows record the
  effective budget. SNQI v2 calibration uses the same schedule.
- **Reason:** The author confirmed the 400/500-step limits were deliberate:
  they keep each scenario inside its interactive part, so waiting until the
  pedestrians are gone is not a winning strategy.
- **Decided by:** author. This replaced an earlier delegated decision that the
  fixed 600-step horizon should win.
- **Rejected alternatives:** a fixed 600-step horizon for every scenario.
- **Implemented in:** #9999 (open, approved).
- **Enforcing test:**
  `tests/benchmark/test_campaign_horizon_authority.py::test_real_release_0_0_8_template_preserves_authored_budgets`,
  `tests/benchmark/test_campaign_horizon_authority.py::test_undeclared_shorter_scenario_limit_is_refused`,
  `tests/benchmark/test_campaign_horizon_authority.py::test_real_simulator_budget_timeout_and_terminal_controls`,
  `tests/unit/benchmark/test_snqi_v2.py::test_development_calibration_matches_candidate_and_preserves_frozen_007`
  (open PR).

### D-026: Budgets stay after the data check; four scenarios are disclosed
- **Date:** 2026-09-30
- **Question:** Does each authored budget really end inside the interactive
  part of its scenario?
- **Choice:** Keep all budgets. In 44 of 48 scenarios the data check
  confirmed it. The 4 flagged scenarios (classic_bottleneck_low,
  francis2023_blind_corner, francis2023_entering_room,
  francis2023_intersection_no_gesture) are disclosed as a limitation and get a
  wait-then-go probe; classic_bottleneck_low's design is checked separately.
- **Reason:** The flag came from an optimistic timing test that was not
  executed; changing budgets on that basis would be premature.
- **Decided by:** delegated (#9999 comment).
- **Rejected alternatives:** lengthen or shorten the four budgets now.
- **Implemented in:** disclosure; probe queued.
- **Enforcing test:** no test yet.

## Metrics and reporting

### D-027: Metric v2: a stall is measured as progress along the route
- **Date:** 2026-09-30
- **Question:** The deadlock detector flagged any episode slower than 0.5 m/s
  towards the goal, and counted collision episodes.
- **Choice:** Measure stalls as progress along the route frozen at reset, over
  the window the docstring describes, and exclude collision episodes.
- **Reason:** Straight-line distance flagged a goal planner moving at
  0.66 m/s on a curved route as deadlocked.
- **Decided by:** delegated (#10014 review comment).
- **Rejected alternatives:** keep Euclidean distance with a corrected window.
- **Implemented in:** #10014 (open, approved).
- **Enforcing test:**
  `tests/benchmark/test_fxm2_metrics.py::test_real_merging_timeout_keeps_route_progress`,
  `tests/benchmark/test_fxm_metric_definitions.py::test_slow_steady_approach_is_not_a_deadlock`,
  `tests/benchmark/test_fxm_metric_definitions.py::test_collision_episode_is_not_a_deadlock`
  (open PR).

### D-028: Metric v2: path efficiency is defined only for successful episodes
- **Date:** 2026-09-30
- **Question:** Path efficiency was clipped, so collisions could look perfectly
  efficient.
- **Choice:** Path efficiency exists only for successful episodes; otherwise
  it is NaN (null in JSON), and no clipping is applied.
- **Reason:** This matches the thesis definition; clipping hid failures.
- **Decided by:** delegated (#10014 review comment).
- **Rejected alternatives:** keep clipping; report zero for failures.
- **Implemented in:** #10014 (open, approved).
- **Enforcing test:**
  `tests/benchmark/test_fxm2_metrics.py::test_failure_efficiency_is_undefined`,
  `tests/benchmark/test_fxm2_metrics.py::test_completion_with_collision_has_no_path_efficiency`,
  `tests/benchmark/test_fxm2_metrics.py::test_success_efficiency_is_unclipped_and_reference_violation_flagged`
  (open PR).

### D-029: Metric v2: the goal reference is the goal-zone polygon
- **Date:** 2026-09-30
- **Question:** The metric goal was whichever waypoint was active at the end of
  the episode, so path metrics depended on where a planner entered the goal
  zone.
- **Choice:** Capture the route's final goal at reset. When completion means
  entering the goal zone, the reference path is the shortest path to the zone
  polygon, not to the sampled point. The same reference feeds efficiency and
  ideal time.
- **Reason:** A planner-dependent reference makes path efficiency and the
  SNQI v2 time term incomparable across planners.
- **Decided by:** delegated (#10014 review comment).
- **Rejected alternatives:** the sampled goal point; the last active waypoint.
- **Implemented in:** #10014 (open, approved).
- **Enforcing test:**
  `tests/benchmark/test_fxm2_metrics.py::test_goal_zone_shortest_path_is_to_continuous_polygon`,
  `tests/benchmark/test_fxm2_metrics.py::test_map_producer_uses_same_zone_reference_for_efficiency_and_ideal_time`,
  `tests/benchmark/test_fxm_metric_definitions.py::test_final_route_goal_is_captured_at_reset`
  (open PR).

### D-030: Metric v2: a schema version fence separates old and new meanings
- **Date:** 2026-09-30
- **Question:** How to stop rows with old and new metric meanings from being
  mixed?
- **Choice:** Rows carry `metric_schema_version`. Consumers refuse mixed v1/v2
  cohorts and unmarked rows that contain v2-only fields; old anchors cannot
  normalize new definitions; trace schema versions whose field meanings changed
  are bumped. Jerk now has physical time units and the first step counts in
  time and path metrics. The change is made now because the SNQI v2 anchors are
  not frozen yet.
- **Reason:** Silent mixing would average different quantities under one
  name. A later v3 would cost more.
- **Decided by:** delegated (#10014 review comment).
- **Rejected alternatives:** change the definitions in place without a version.
- **Implemented in:** #10014 (open, approved); the auditor accepts only v1 or
  v2 markers (#9997). Remaining gaps are #10022 (0.0.9).
- **Enforcing test:**
  `tests/benchmark/test_fxm2_metrics.py::test_unmarked_v2_only_diagnostics_are_refused`,
  `tests/benchmark/test_fxm_metric_definitions.py::test_aggregation_rejects_mixed_metric_meanings`,
  `tests/benchmark/test_fxm_metric_definitions.py::test_old_anchors_cannot_normalize_new_metric_definitions`,
  `tests/benchmark/test_fxm_metric_definitions.py::test_jerk_has_physical_time_units`
  (open PR).

### D-031: SNQI v2 uses the robot-attributable force
- **Date:** 2026-09-30 (thesis rule); the metric and index were decided
  earlier in #9666 and #9667.
- **Question:** The comfort metrics `force_exceed_events` and
  `comfort_exposure` count the total force on a pedestrian, including
  pedestrian, wall and group forces. What measures discomfort caused by the
  robot?
- **Choice:** The SNQI v2 force term uses the robot's own force component
  (`robot_force_impulse_total`, or its pedestrian-equivalent variant if the
  calibration's force decision selects it) and does not use the two
  total-force metrics.
  Where the thesis means "discomfort caused by the robot" it reports the
  robot-force metric; the old metrics appear only as 0.0.7 quantities,
  described as total pedestrian force.
- **Reason:** The group gaze term alone causes 8-11 % of total-force
  exceedances, so the old metrics are not attributable to the robot.
- **Decided by:** delegated (thesis rule); the SNQI v2 design is #9667.
- **Rejected alternatives:** keep the total-force metrics in the index.
- **Implemented in:** `robot_sf/benchmark/snqi/v2_spec.py` on `main`; thesis
  intake.
- **Enforcing test:**
  `tests/unit/benchmark/test_snqi_v2.py::test_snqi_v2_rejects_force_values_without_recorded_provenance`,
  `tests/unit/benchmark/test_snqi_v2.py::test_duplicate_and_derived_source_rejected`.

### D-032: Table column `episodes` is the eligible count, with total and excluded beside it
- **Date:** 2026-09-30
- **Question:** Table means already excluded ineligible episodes, but the count
  column did not say how many were excluded.
- **Choice:** `episodes` is the eligible N used by the means. New columns
  `episodes_total` and `episodes_excluded` (with reasons) sit beside it.
- **Reason:** A reader must be able to see how many episodes each mean rests
  on and how many were dropped.
- **Decided by:** delegated (#10019 review rounds).
- **Rejected alternatives:** keep `episodes` as the raw row count.
- **Implemented in:** #10019 (open; last review FIX).
- **Enforcing test:**
  `tests/benchmark/test_release_reporting_cohort.py::test_campaign_table_counts_match_metric_cohort_and_writers`,
  `tests/benchmark/test_release_reporting_cohort.py::test_all_excluded_arms_remain_in_breakdown_and_seed_outputs`
  (open PR).

### D-033: Release acceptance counts `episodes_total`; excluded rows are named
- **Date:** 2026-09-30
- **Question:** After D-032, the full-release acceptance check compared the
  eligible count with the planned N, so any excluded row blocked 0.0.8 with a
  misleading message.
- **Choice:** Acceptance checks `episodes_total` against the planned N.
  Invalid-spawn rows get their own named blocker. Rows excluded for missing
  foresight evidence are reported, not blocking. The publication check of SNQI
  ordering uses the same eligible cohort as the diagnostics.
- **Reason:** An invalid spawn is a campaign defect; a foresight exclusion is a
  reporting choice. The block reason must name the real cause.
- **Decided by:** delegated.
- **Rejected alternatives:** keep the eligible count as a silent gate (option
  (a) of the review without a named blocker).
- **Implemented in:** #10019, fix round in progress (not yet pushed when this
  file was written).
- **Enforcing test:** no test yet.

## Pedestrian simulation

### D-034: Pedestrians draw from private random streams seeded by the episode seed
- **Date:** 2026-09-30
- **Question:** Pedestrian respawns and goals drew from the global NumPy
  generator, so a planner that used random numbers changed the crowd.
- **Choice:** Pedestrians use private random streams derived from the episode
  seed, independent of the global generator. Every seeded entry point must
  pass the seed through.
- **Reason:** For the same seed every planner must face the same crowd.
- **Decided by:** delegated.
- **Rejected alternatives:** re-seed the global generator each step.
- **Implemented in:** #10024 (open; last review FIX: 13 existing tests break
  and some entry points seeded only through the global generator lose
  reproducibility).
- **Enforcing test:**
  `tests/ped_npc/test_pedfix_episode_contract.py::test_trajectories_ignore_global_numpy_after_respawn`,
  `tests/ped_npc/test_pedfix_episode_contract.py::test_vendored_population_leaves_global_numpy_untouched`
  (open PR).

### D-035: `simulation_config.groups` is the fraction of pedestrians in groups
- **Date:** 2026-09-30
- **Question:** The scenario key `groups` was ignored. When applied, what does
  0.5 mean?
- **Choice:** The fraction of pedestrians who walk in groups, as the scenario
  schema, the scenario generator and the scenario docs already define it.
- **Reason:** The first implementation read it as the probability that a
  sampled group has more than one member; for 0.5 that puts 69 % of
  pedestrians in groups and inflated the effect on the goal planner.
- **Decided by:** delegated.
- **Rejected alternatives:** probability of a multi-member group; rename the
  key.
- **Implemented in:** #10024, fix round in progress.
- **Enforcing test:** no test yet for the fraction meaning. The current test
  `tests/ped_npc/test_pedfix_episode_contract.py::test_groups_override_changes_real_population`
  (open PR) only checks that the key has an effect.

### D-036: Unknown `simulation_config` keys fail closed
- **Date:** 2026-09-30
- **Question:** Unknown keys in `simulation_config` were silently ignored.
- **Choice:** The scenario loader refuses unknown keys.
- **Reason:** A silently ignored key (such as `groups`) makes a scenario claim
  something it does not do.
- **Decided by:** delegated.
- **Rejected alternatives:** warn only.
- **Implemented in:** #10024 (open). The review found shipped non-release
  configs and runtime writers that now fail; they must be migrated.
- **Enforcing test:**
  `tests/ped_npc/test_pedfix_episode_contract.py::test_unknown_simulation_key_fails_at_real_loader`
  (open PR).

### D-037: A pedestrian moved off the robot at reset keeps a reaction buffer
- **Date:** 2026-09-30
- **Question:** A pedestrian relocated off the robot at reset landed 1.5 m from
  its centre and kept walking towards it, causing contacts no planner could
  avoid.
- **Choice:** Relocate with reaction clearance (contact distance plus 0.1 m
  plus the distance the pedestrian covers in one second) and point its
  velocity along its route.
- **Reason:** An unavoidable early collision was charged to the planner.
- **Decided by:** delegated.
- **Rejected alternatives:** keep the 0.1 m margin.
- **Implemented in:** #10024 (open).
- **Enforcing test:**
  `tests/ped_npc/test_pedfix_episode_contract.py::test_relocation_has_reaction_clearance_and_route_velocity`,
  `tests/ped_npc/test_pedfix_episode_contract.py::test_relocation_keeps_reaction_buffer_when_goal_is_inside_robot`
  (open PR).

### D-038: Wall law and stuck pedestrians: disclosed for 0.0.8, fixed in 0.0.9
- **Date:** 2026-09-30
- **Question:** The pedestrian wall force reaches far and adds up over wall
  segments, so pedestrians stop in front of narrow openings, and goal-only
  pedestrians walk into obstacles and stay stuck. Fix before 0.0.8?
- **Choice:** Disclose for 0.0.8, including which release scenarios have
  blocked pedestrians; fix in 0.0.9 as a declared behaviour change, validated
  against pedestrian-flow references. The disclosure uses the measured,
  shape-dependent wording (openings of 3 m or less stopped a pedestrian at
  0.65 m/s in every tested geometry; pedestrians already inside a straight
  corridor keep walking), not "cannot pass openings of 3 m or less".
- **Reason:** A new wall law changes every scenario and needs its own
  validation.
- **Decided by:** author (#10017).
- **Rejected alternatives:** change the wall law before 0.0.8.
- **Implemented in:** disclosure (thesis intake diss#3021); fix in #10017.
- **Enforcing test:** no test yet.

### D-039: Declared pedestrian speed is the desired speed; the 1.3 cap stays
- **Date:** 2026-09-30
- **Question:** `single_pedestrians[].speed_m_s` is multiplied by 1.3 to form
  the speed cap, so pedestrians walk up to 30 % faster than declared. Change?
- **Choice:** Keep the standard convention (maximum = 1.3 x desired speed) and
  document that the declared value is the desired speed.
- **Reason:** It is the standard social-force convention, not a defect.
- **Decided by:** author (#10017, item 3).
- **Rejected alternatives:** divide by 1.3 when seeding the speed.
- **Implemented in:** documentation (#10017).
- **Enforcing test:** no test yet.

### D-040: Group gaze and group repulsion: disclosed for 0.0.8, fixed in 0.0.9
- **Date:** 2026-09-30
- **Question:** The external pedestrian review found that the group gaze force
  scales with the inverse goal distance and ignores the field of view, and
  that group repulsion weakens as members get closer. All 12 of its findings
  were confirmed by execution. Which ones matter for 0.0.8?
- **Choice:** Disclose findings 1 and 2 (group forces differ from the
  published laws; groups form in 14-18 of 48 scenarios) for 0.0.8 and fix them
  in 0.0.9 with recalibration. Findings 3-12 go to 0.0.9 only: they are either
  not exposed in the release scenarios or change positions by at most 0.13 m.
- **Reason:** Under D-001 none of them is a P1 that measurably changes reported
  numbers; the group laws still change how results should be read.
- **Decided by:** delegated.
- **Rejected alternatives:** fix the group laws before 0.0.8.
- **Implemented in:** #10027 (0.0.9); disclosure in the thesis intake.
- **Enforcing test:** no test yet.

### D-041: join_group and leave_group are disclosed as containing no group
- **Date:** 2026-09-30
- **Question:** In `francis2023_join_group` the join never completes and every
  pedestrian starts alone; `francis2023_leave_group` has no group to leave.
- **Choice:** Disclose this in the scenario descriptions for 0.0.8; the
  scenarios still test the robot among those pedestrians. Fix the authored
  groups and the join threshold in 0.0.9.
- **Reason:** Changing scenarios after the freeze would change the release
  matrix.
- **Decided by:** delegated.
- **Rejected alternatives:** drop or rename the scenarios for 0.0.8.
- **Implemented in:** #10028 (0.0.9).
- **Enforcing test:** no test yet.

## Scenarios and maps

### D-042: Rectangular zones keep the 3-corner encoding but are sampled over the full rectangle
- **Date:** 2026-09-30
- **Question:** SVG rectangle zones are stored as 3 corners and were sampled as
  a triangle. Change the encoding to 4 corners?
- **Choice:** Keep the 3-corner encoding and sample the whole rectangle. A
  separate triangle marker keeps real triangles (synthetic crowd triangles,
  authored 3-vertex crowd paths) triangular.
- **Reason:** The type contract, map files, goal completion and the vendored
  sampler already read 3 corners as a full rectangle; 4-tuples would ripple
  through about ten consumers and the map format.
- **Decided by:** delegated (fix lane choice, recorded in #10026).
- **Rejected alternatives:** switch to a 4-corner encoding.
- **Implemented in:** #10026 (open, review pending).
- **Enforcing test:**
  `tests/test_scenario_map_review_fixes.py::test_svg_rectangle_zone_samples_the_full_rectangle`,
  `tests/test_scenario_map_review_fixes.py::test_parsed_svg_spawn_zone_covers_the_whole_rectangle`,
  `tests/test_scenario_map_review_fixes.py::test_true_triangle_zones_stay_triangular`
  (open PR).

### D-043: Triangle sampling in 0.0.7 is a coverage limitation, not a numbers error
- **Date:** 2026-09-30
- **Question:** How to describe the 0.0.7 triangle sampling?
- **Choice:** State it in the 0.0.7-to-0.0.8 verification as a limitation of
  start and goal coverage: every planner faced the same restricted starts, so
  the reported 0.0.7 numbers are not wrong for what they measured.
- **Reason:** It affected all 48 scenarios equally for all planners.
- **Decided by:** delegated (diss#3021 comment).
- **Rejected alternatives:** present it as an error in 0.0.7 results.
- **Implemented in:** thesis intake.
- **Enforcing test:** no test yet.

### D-044: The elevator wall geometry change is disclosed
- **Date:** 2026-09-29
- **Question:** #9972 changed the elevator scenario's interior walls.
- **Choice:** Disclose the change in the 0.0.8 release notes and the thesis
  intake.
- **Reason:** A reader comparing 0.0.7 and 0.0.8 must know the geometry
  differs.
- **Decided by:** delegated (#9972 review).
- **Rejected alternatives:** none recorded.
- **Implemented in:** #9972 (merged); disclosure pending.
- **Enforcing test:** no test yet.

### D-045: The narrow-doorway probe counts contact only, not force
- **Date:** 2026-09-29
- **Question:** Should the doorway probe's safe-failure metric count robot
  force on a pedestrian as contact?
- **Choice:** Contact means a collision event or a positive collision count
  (walls included). Force stays a descriptive column. A timeout without
  contact is a safe failure. The gate checks the exact (arm, seed) slot set.
- **Reason:** The robot force acts at up to about 3.4 m centre distance, so
  counting it would class safe waits as contacts.
- **Decided by:** delegated (#9976 review).
- **Rejected alternatives:** count force above zero as contact.
- **Implemented in:** #9976 (merged).
- **Enforcing test:**
  `tests/benchmark/test_infeasible_probe_safe_failure.py::test_force_above_zero_without_collision_is_safe_failure`,
  `tests/benchmark/test_infeasible_probe_safe_failure.py::test_gate_refuses_duplicate_that_offsets_a_missing_slot`.

## Campaign operations

### D-046: Private-ops pull requests that carry a runtime pin are merged with a merge commit
- **Date:** 2026-09-30
- **Question:** A squash merge had left a runtime pin that is not an ancestor
  of `main`, and the submit guard refused every real submission. How should
  such pull requests merge?
- **Choice:** A private-ops pull request whose runtime pin lives on its own
  branch is merged with a merge commit, so the pin stays an ancestor of
  `main`; no re-pin is needed.
- **Reason:** The submit guard correctly refuses a diverged pin; a merge commit
  keeps the reviewed bytes reachable.
- **Decided by:** delegated.
- **Rejected alternatives:** squash and re-pin afterwards (as in
  private-ops #411).
- **Implemented in:** private-ops #414 and #418 (merged with merge commits).
- **Enforcing test:** the guard is tested by
  `ops/jobs/scripts/test_submit_remote_sync.py::test_pinned_runtime_sync_accepts_reviewed_ancestor_and_refuses_dirty_or_diverged`
  (private). The merge method itself has no test yet.

### D-047: Admission uses one fixture rule and binds the reviewed public commit
- **Date:** 2026-09-30
- **Question:** The external admission review found that two flags could
  select another queue while production commands stayed allowed, and that the
  reviewed public commit was not part of the launch packet.
- **Choice:** One fixture-authorization predicate at every admission point,
  with partial fixture setups refused; the reviewed public commit is part of the
  packet and is compared with the observed remote head before submission.
- **Reason:** Both gaps let a reviewed row run something other than what was
  reviewed.
- **Decided by:** delegated.
- **Rejected alternatives:** none recorded.
- **Implemented in:** private-ops #413 (merged).
- **Enforcing test:**
  `ops/jobs/scripts/test_side_effect_guard.py::test_partial_fixture_configuration_is_refused_before_child_runs`,
  `ops/jobs/scripts/test_canonical_submit_guard.py::test_submitted_public_commit_pin_must_match_the_reviewed_row`
  (private).

### D-048: The SNQI v2 anchor and seed ruling is issued under delegation after calibration
- **Date:** 2026-09-30
- **Question:** The private mint of the 0.0.8 campaign rows needs a reviewed
  ruling that binds the freeze, manifest, config, SNQI v2 anchor file and the
  evaluation seeds 111-140. Who issues it, and when?
- **Choice:** The orchestrator issues it as a delegated decision once the
  development calibration (seeds 101-102) has produced frozen anchors and a
  refute review has checked them; the author discusses it afterwards. Until
  then the mint's trust list stays empty and its output stays "proposed". The
  ruling enters as exactly one reviewed entry in the mint's trusted sources,
  with a positive control on the real pinned bytes.
- **Reason:** Speed: the author trusts the gates and reviews afterwards. The
  empty trust list means no campaign row can be minted without that ruling.
- **Decided by:** author (delegation); the ruling itself will be delegated.
  Not issued yet.
- **Rejected alternatives:** wait for an author ruling before the campaign.
- **Implemented in:** private-ops #407 (merged); the follow-up pull request
  "ops: pin calibrated SNQI-v2 0.0.8 scientific sources" is not opened yet.
- **Enforcing test:**
  `ops/jobs/scripts/test_mint_snqi_v2_full_campaign.py::test_refuses_unfrozen_anchors_and_undecided_seeds`,
  `ops/jobs/scripts/test_full_campaign_authority.py::test_forged_scientific_authority_should_be_refused`
  (private).

## Decisions without an enforcing test

These decisions have no test that would fail if the code or process drifted
from them. Partial coverage is noted.

| ID | Decision | Note |
|---|---|---|
| D-001 | Freeze rule: only measurable P1s block 0.0.8 | process |
| D-002 | Earlier P1/P2 admission rule (superseded) | process |
| D-003 | Fix result-changing planner defects before freeze | process; fixes have own tests |
| D-004 | Pinned freeze commit | only the private mint checks freeze blobs |
| D-005 | Rehearsal campaign | process |
| D-006 | One last external review | process |
| D-007 | Behaviour-change gate | automation in #10000 |
| D-010 | Hybrid v4 inert keys disclosed | consumed-key check planned for 0.0.9 |
| D-014 | Empty-world residuals are method limits | docs in #10002 |
| D-018 | SA-CADRL per-step heading mapping (part of D-018) | speed and agent count are tested |
| D-026 | Budgets kept; four scenarios disclosed | probe queued |
| D-033 | Acceptance on `episodes_total` with named blocker | fix round not pushed |
| D-035 | `groups` = fraction of pedestrians in groups | fix round not pushed |
| D-038 | Wall law disclosed, fixed in 0.0.9 | #10017 |
| D-039 | Declared speed is desired speed | documentation only |
| D-040 | Group forces disclosed, fixed in 0.0.9 | #10027 |
| D-041 | join/leave scenarios disclosed | #10028 |
| D-043 | 0.0.7 triangle sampling described as coverage limit | thesis wording |
| D-044 | Elevator geometry change disclosed | disclosure pending |
| D-046 | Merge-commit rule for runtime-pinned private-ops PRs | the guard is tested, the merge method is not |

D-008 (held-out seeds) is only partly enforced: the existing test covers added
lines, not the whole tree or the seeds actually passed at runtime (#10010).
