# Release 0.0.8: decision ledger

New decisions are added here with evidence and an enforcing test or an intended check.

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
- **Enforced by.** Written as `file::test`. "(open PR)" means the test
  exists on the pull request branch but is not yet on `main`. "none yet:"
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
- **Alternatives:** block on every P1 and P2 finding (the earlier
  rule, D-002); block on every finding.
- **Implemented in:** process rule; milestone 0.0.9.
- **Evidence:** Original ledger in PR #10032 (author or delegated ruling as recorded above); original ledger provenance preserved.
- **Enforced by:** no test yet.

### D-002: Earlier admission rule for side-quest findings (superseded by D-001)
- **Date:** 2026-09-30 (morning)
- **Question:** Do findings from side work (audits, reviews) block the freeze?
- **Choice:** A finding blocked the freeze only if it was P1 or P2 (changes
  results). P3 findings and review nits on critical pull requests became
  issues, not extra review rounds.
- **Reason:** Keep the critical path moving while not losing result-changing
  defects.
- **Decided by:** delegated.
- **Alternatives:** treat every review finding as a merge blocker.
- **Implemented in:** process rule, stated on tracker #10013. Tightened by D-001.
- **Evidence:** #10013; original ledger provenance preserved.
- **Enforced by:** no test yet.

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
- **Alternatives:** release 0.0.8 first and fix in 0.0.9.
- **Implemented in:** #9926, #9995, #10008, #10009, #10011, #10014 (see the
  decisions below).
- **Evidence:** #9926, #9995, #10008, #10009, #10011, #10014; original ledger provenance preserved.
- **Enforced by:** no test yet for the rule itself; the individual fixes
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
- **Alternatives:** freeze `main` itself for the duration.
- **Implemented in:** #9932; the private mint binds a freeze commit
  (private-ops #407).
- **Evidence:** #9932, private-ops #407; original ledger provenance preserved.
- **Enforced by:** no test yet in this repository. The private mint refuses
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
- **Alternatives:** go straight from merges to the full campaign.
- **Implemented in:** operational (no pull request).
- **Evidence:** Original ledger in PR #10032 (author or delegated ruling as recorded above); original ledger provenance preserved.
- **Enforced by:** no test yet.

### D-006: One last external review for 0.0.8
- **Date:** 2026-09-30
- **Question:** Should external model reviews continue during the freeze?
- **Choice:** One last external review for 0.0.8, on the planner adapters.
  Later external reviews go to 0.0.9 or to checks of thesis claims.
- **Reason:** Same as D-001: reviews kept reopening the freeze.
- **Decided by:** author.
- **Alternatives:** keep reviewing every area before release.
- **Implemented in:** process rule. The adapter review arrived on 2026-09-30
  and is listed in findings.md, triage pending.
- **Evidence:** Original ledger in PR #10032 (author or delegated ruling as recorded above); original ledger provenance preserved.
- **Enforced by:** no test yet.

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
- **Alternatives:** a success-rate threshold.
- **Implemented in:** applied by hand in 0.0.8; automation is #10000
  (milestone 0.0.9).
- **Evidence:** #10000; original ledger provenance preserved.
- **Enforced by:** no test yet (no CI check requires a gate receipt).

### D-008: Held-out evaluation seeds 111-140 are never stepped outside the campaign

**Superseded for new 0.0.8 evaluation by D-049 below; historical archived-artifact records remain unchanged.**
- **Date:** 2026-09-30
- **Question:** Which seeds may tests and development runs use?
- **Choice:** Every integer from 111 to 140 is reserved for the release
  evaluation. Tests and diagnostics use development seeds 1001-1030. An earlier
  lane rule that treated seed 123 as allowed was wrong and was corrected;
  existing tests that step held-out seeds are inventoried and moved (#10010).
- **Reason:** Tuning or bounding anything on held-out outcomes would leak
  evaluation information into development.
- **Decided by:** delegated.
- **Alternatives:** allow held-out seeds in tests that only check
  pass/fail.
- **Implemented in:** the diff gate `scripts/validation/check_seed_holdout_diff.py`;
  test moves in #10009 and #10026; whole-tree inventory in #10010 (0.0.9).
- **Evidence:** #10009, #10026, #10010; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** pick a trial by judgement after seeing the results.
- **Implemented in:** #9932 (merged).
- **Evidence:** #9932; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** run the escape rules inside the continuous branch
  (a behaviour change); drop the scenario overrides.
- **Implemented in:** disclosure (#10006, #10002, thesis intake); fail-closed
  key check in 0.0.9.
- **Evidence:** #10006, #10002; original ledger provenance preserved.
- **Enforced by:** no test yet.

### D-011: Planner horizons keep their length in seconds after the 0.1 s step fix
- **Date:** 2026-09-29
- **Question:** When planner configs move from a 0.2 s to the true 0.1 s
  control step, should the step counts stay (halving the horizon) or double?
- **Choice:** Keep the horizon in seconds and double the step counts.
- **Reason:** Matching the 0.1 s control step is the correction; the horizon
  length is the method's parameter and must not change as a side effect.
- **Decided by:** delegated (#9926 review).
- **Alternatives:** keep step counts and silently halve the lookahead.
- **Implemented in:** #9926 (merged).
- **Evidence:** #9926; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** restore a 1.6 s window; silently hold the last
  forecast beyond 8 steps.
- **Implemented in:** #9926 (merged), #10011 (open).
- **Evidence:** #9926, #10011; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** keep unbounded sampling.
- **Implemented in:** #9926 (merged).
- **Evidence:** #9926; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** tune the planners until the empty-world sweep
  passes.
- **Implemented in:** docs task #10002 (open).
- **Evidence:** #10002; original ledger provenance preserved.
- **Enforced by:** no test yet.

### D-015: guarded_ppo keeps its one-step waypoint lookahead
- **Date:** 2026-09-29
- **Question:** At a waypoint boundary the guarded_ppo outer guard targets the
  next waypoint while its fallback targets the current one. Unify?
- **Choice:** Keep the outer guard's lookahead, document it and pin it with a
  test, so the "one active-waypoint target" claim is stated precisely.
- **Reason:** It is guarded_ppo's own method, it lasts one step, and the
  simulator advances the waypoint on the next step.
- **Decided by:** delegated (#9891 review).
- **Alternatives:** force the outer guard onto the current waypoint.
- **Implemented in:** #9891 (merged).
- **Evidence:** #9891; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** keep the absolute-target reading.
- **Implemented in:** #9995 (open, approved).
- **Evidence:** #9995; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** evaluate only the old checkpoints; replace the
  release PPO arm with the retrained one.
- **Implemented in:** #10003 (open, approved); private-ops #408, #409, #411,
  #414, #418 (merged).
- **Evidence:** #10003, private-ops #408, #409, #411, #414, #418; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** 2.0 m/s; 3 agents; a heading-hold adapter (would
  change the checkpoint's decision timing).
- **Implemented in:** #10008 (open, approved).
- **Evidence:** #10008; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** keep per-planner defaults.
- **Implemented in:** #10009 (open, approved). Planners outside the release
  roster are listed on #10007 for 0.0.9.
- **Evidence:** #10009, #10007; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** keep the old cap-only slowdown; a
  guarantee-preserving nonholonomic ORCA variant (a new method).
- **Implemented in:** #10009 (open, approved).
- **Evidence:** #10009; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** ship the non-firing term; design a new term now.
- **Implemented in:** #10011 (open, approved).
- **Evidence:** #10011; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** retrain now; keep the collectors unchanged.
- **Implemented in:** collector fix in #10011 (open); retrain in #10018.
- **Evidence:** #10011, #10018; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** relabel the base algorithm.
- **Implemented in:** #9996 (merged).
- **Evidence:** #9996; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** exempt any ORCA override.
- **Implemented in:** #10001 (merged).
- **Evidence:** #10001; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** a fixed 600-step horizon for every scenario.
- **Implemented in:** #9999 (open, approved).
- **Evidence:** #9999; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** lengthen or shorten the four budgets now.
- **Implemented in:** disclosure; probe queued.
- **Evidence:** Original ledger in PR #10032 (author or delegated ruling as recorded above); original ledger provenance preserved.
- **Enforced by:** no test yet.

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
- **Alternatives:** keep Euclidean distance with a corrected window.
- **Implemented in:** #10014 (open, approved).
- **Evidence:** #10014; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** keep clipping; report zero for failures.
- **Implemented in:** #10014 (open, approved).
- **Evidence:** #10014; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** the sampled goal point; the last active waypoint.
- **Implemented in:** #10014 (open, approved).
- **Evidence:** #10014; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** change the definitions in place without a version.
- **Implemented in:** #10014 (open, approved); the auditor accepts only v1 or
  v2 markers (#9997). Remaining gaps are #10022 (0.0.9).
- **Evidence:** #10014, #9997, #10022; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** keep the total-force metrics in the index.
- **Implemented in:** `robot_sf/benchmark/snqi/v2_spec.py` on `main`; thesis
  intake.
- **Evidence:** Original ledger in PR #10032 (author or delegated ruling as recorded above); original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** keep `episodes` as the raw row count.
- **Implemented in:** #10019 (open; last review FIX).
- **Evidence:** #10019; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** keep the eligible count as a silent gate (option
  (a) of the review without a named blocker).
- **Implemented in:** #10019, fix round in progress (not yet pushed when this
  file was written).
- **Evidence:** #10019; original ledger provenance preserved.
- **Enforced by:** no test yet.

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
- **Alternatives:** re-seed the global generator each step.
- **Implemented in:** #10024 (open; last review FIX: 13 existing tests break
  and some entry points seeded only through the global generator lose
  reproducibility).
- **Evidence:** #10024; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** probability of a multi-member group; rename the
  key.
- **Implemented in:** #10024, fix round in progress.
- **Evidence:** #10024; original ledger provenance preserved.
- **Enforced by:** no test yet for the fraction meaning. The current test
  `tests/ped_npc/test_pedfix_episode_contract.py::test_groups_override_changes_real_population`
  (open PR) only checks that the key has an effect.

### D-036: Unknown `simulation_config` keys fail closed
- **Date:** 2026-09-30
- **Question:** Unknown keys in `simulation_config` were silently ignored.
- **Choice:** The scenario loader refuses unknown keys.
- **Reason:** A silently ignored key (such as `groups`) makes a scenario claim
  something it does not do.
- **Decided by:** delegated.
- **Alternatives:** warn only.
- **Implemented in:** #10024 (open). The review found shipped non-release
  configs and runtime writers that now fail; they must be migrated.
- **Evidence:** #10024; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** keep the 0.1 m margin.
- **Implemented in:** #10024 (open).
- **Evidence:** #10024; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** change the wall law before 0.0.8.
- **Implemented in:** disclosure (thesis intake diss#3021); fix in #10017.
- **Evidence:** #3021, #10017; original ledger provenance preserved.
- **Enforced by:** no test yet.

### D-039: Declared pedestrian speed is the desired speed; the 1.3 cap stays
- **Date:** 2026-09-30
- **Question:** `single_pedestrians[].speed_m_s` is multiplied by 1.3 to form
  the speed cap, so pedestrians walk up to 30 % faster than declared. Change?
- **Choice:** Keep the standard convention (maximum = 1.3 x desired speed) and
  document that the declared value is the desired speed.
- **Reason:** It is the standard social-force convention, not a defect.
- **Decided by:** author (#10017, item 3).
- **Alternatives:** divide by 1.3 when seeding the speed.
- **Implemented in:** documentation (#10017).
- **Evidence:** #10017; original ledger provenance preserved.
- **Enforced by:** no test yet.

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
- **Alternatives:** fix the group laws before 0.0.8.
- **Implemented in:** #10027 (0.0.9); disclosure in the thesis intake.
- **Evidence:** #10027; original ledger provenance preserved.
- **Enforced by:** no test yet.

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
- **Alternatives:** drop or rename the scenarios for 0.0.8.
- **Implemented in:** #10028 (0.0.9).
- **Evidence:** #10028; original ledger provenance preserved.
- **Enforced by:** no test yet.

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
- **Alternatives:** switch to a 4-corner encoding.
- **Implemented in:** #10026 (open, review pending).
- **Evidence:** #10026; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** present it as an error in 0.0.7 results.
- **Implemented in:** thesis intake.
- **Evidence:** Original ledger in PR #10032 (author or delegated ruling as recorded above); original ledger provenance preserved.
- **Enforced by:** no test yet.

### D-044: The elevator wall geometry change is disclosed
- **Date:** 2026-09-29
- **Question:** #9972 changed the elevator scenario's interior walls.
- **Choice:** Disclose the change in the 0.0.8 release notes and the thesis
  intake.
- **Reason:** A reader comparing 0.0.7 and 0.0.8 must know the geometry
  differs.
- **Decided by:** delegated (#9972 review).
- **Alternatives:** none recorded.
- **Implemented in:** #9972 (merged); disclosure pending.
- **Evidence:** #9972; original ledger provenance preserved.
- **Enforced by:** no test yet.

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
- **Alternatives:** count force above zero as contact.
- **Implemented in:** #9976 (merged).
- **Evidence:** #9976; original ledger provenance preserved.
- **Enforced by:**
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
- **Alternatives:** squash and re-pin afterwards (as in
  private-ops #411).
- **Implemented in:** private-ops #414 and #418 (merged with merge commits).
- **Evidence:** private-ops #414, #418; original ledger provenance preserved.
- **Enforced by:** the guard is tested by
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
- **Alternatives:** none recorded.
- **Implemented in:** private-ops #413 (merged).
- **Evidence:** private-ops #413; original ledger provenance preserved.
- **Enforced by:**
  `ops/jobs/scripts/test_side_effect_guard.py::test_partial_fixture_configuration_is_refused_before_child_runs`,
  `ops/jobs/scripts/test_canonical_submit_guard.py::test_submitted_public_commit_pin_must_match_the_reviewed_row`
  (private).

### D-048: The SNQI v2 anchor and seed ruling is issued under delegation after calibration

**Superseded for new 0.0.8 evaluation by D-049 below; historical archived-artifact records remain unchanged.**
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
- **Alternatives:** wait for an author ruling before the campaign.
- **Implemented in:** private-ops #407 (merged); the follow-up pull request
  "ops: pin calibrated SNQI-v2 0.0.8 scientific sources" is not opened yet.
- **Evidence:** private-ops #407; original ledger provenance preserved.
- **Enforced by:**
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


### D-049: Fresh sealed evaluation seeds

- **Date:** 2026-09-30
- **Question:** Which seeds can support a fresh 0.0.8 evaluation after fixes were
  developed and checked on outcomes from the 0.0.7 evaluation band?
- **Choice:** Use the fresh sealed `EVAL_SEEDS_0_0_8` tuple in
  `robot_sf/benchmark/seed_bands.py`, transported by
  `configs/benchmarks/seed_sets_0_0_8.yaml`. Development seeds remain 1001..1030.
  No planner or environment step on either held-out band is permitted **anywhere,
  except the sealed campaign (including its own spawn preflight) at the freeze commit**.
  The sealed campaign includes the main 0.0.8 evaluation and the three-width
  doorway slice, minted together and bound to the same freeze commit. The slice
  runs from its materialized v0.2 identity, generated from
  `three_width_doorway_release_0_0_8_v1.template.yaml` by the same
  `resolve_benchmark_release_identity.py generate` command as the main campaign.
  Both require `source_sha` equal to HEAD at execution. Retired seeds 111..140 have no execution exception; historical
  pins permit STATIC validation of archived artifacts only. A failed sealed
  spawn preflight requires an author decision before fixing and rerunning it.
  The full-release stress pre-run gate requires dev seed 1001.
- **Reason:** Social force, socnav_sampling and hybrid v4 were corrected and
  checked using outcomes from the retired evaluation band. Reusing that band
  would not support a fresh release claim. Preserve 0.0.7 artifacts unchanged.
- **Decided by:** The author approved the fresh seed decision. The orchestrator
  issued the delegated scope, stress-seed, freeze-bound doorway slice, protocol
  and diff-checker rulings; those are separate from the author's seed decision.
- **Alternatives:** Reusing 111..140; historical execution exemptions; sealed seeds
  in unnamed releases, development, or unfrozen slices.
- **Implemented in:** `robot_sf/benchmark/release_protocol.py` checks resolved
  seeds in every policy mode, refuses retired seeds in non-historical releases,
  and applies the reverse sealed-seed identity/source check. Sealed campaign,
  scenario matrix, seed-set and referenced planner configs must use canonical
  repository paths, be tracked, and match their blobs at `source_sha` byte for
  byte; matching basenames or self-declared hashes confer no permission. The shared guard in
  `robot_sf/benchmark/spawn_preflight.py` refuses retired execution even with
  historical pins and protects the release runner and standalone preflight.
  `scripts/validation/check_seed_holdout_diff.py` treats every value in
  `seed_set*.yaml` and `seed_list*.yaml` as a seed and requires the same explicit
  allowlist for held-out names as for sealed literals. The #9748 validator and
  tuning runner protect both bands. SEEDGUARD (#10010) must import
  `HELD_OUT_SEEDS` from `robot_sf/benchmark/seed_bands.py`.
- **Evidence:** #9748, #10010; original ledger provenance preserved.
- **Enforced by:** `tests/benchmark/test_newseeds.py`,
  `tests/benchmark/test_sealed_execution_policy.py`, and
  `tests/validation/test_check_seed_holdout_diff.py`; all holdout witnesses are
  static or use recording stubs, with environment creation forbidden.
  `test_forged_sealed_inputs_refused_before_workers` and
  `test_canonical_inputs_must_equal_source_blobs` in
  `tests/benchmark/test_sealed_source_pins.py` enforce source pins.
  `test_real_materialized_identity_passes_frozen_guard`,
  `test_slice_requires_freeze_source` and
  `test_freeze_bound_slice_preflight_uses_runtime_source` enforce the reachable
  materialized slice and its freeze binding.

Derivation label: `robot_sf_ll7 release 0.0.8 evaluation seeds v1 (sealed 2026-09-30)`.
SHA-256: `166597da1e0e813d8a9cdc810f4b85db1407e9286c3113821423e17c50908dc0`.
Initialize `random.Random(int(hash, 16))`, sample 30 from `range(50000, 60000)`,
then sort. The private mint independently cross-checks the derivation and list.

**Supersession:** D-049 supersedes D-008's reservation of 111..140 for new
release evaluation and D-048's choice of those seeds for the 0.0.8 campaign.
Those entries originate in PR #10032, merge train 1. This supersession applies
when both changes land; retain their historical records and annotate them with
D-049. Their historical pins are for static archived-artifact validation only.

**Reopen:** Only new material evidence or an explicit author decision reopens
this ruling; never reuse an observed evaluation band for a fresh release claim.

### D-050: Accept the corrected combo crowd distribution as the 0.0.8 world

- **Date:** 2026-10-01
- **Question:** Accept the combo #10024/#10026 world-generation correction knowing
  that method-faithful release socnav_sampling loses success and gains pedestrian
  collisions on development seeds, or tune the planner to recover those outcomes?
- **Choice:** Accept the combo crowd distribution as the 0.0.8 world. Keep the
  planner and `configs/algos/socnav_sampling_release_v0_0_8.yaml` unchanged.
  The release configuration deliberately mirrors SocNavBench: no pedestrian
  prediction, no footprint margin, and no repulsion enhancement (#9926).
  The measured loss is a property of this method in the corrected world, not a
  defect to tune away.
- **Decided by:** The orchestrator issued this delegated ruling on 2026-10-01 in
  TRAIN2 fix round 1 for PR #10080, following refute review rr10080, finding F1.
- **Reason:** The reviewed combo corrects private per-episode pedestrian streams,
  group-member probabilities, the spawn reaction buffer, and velocity toward the
  current route goal when pedestrians are relocated. No sampler planner bytes
  changed. This world correction exposes the release method's omission of
  pedestrian motion prediction.
- **Paired evidence:** rr10080 compared 48 release scenarios from
  `configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml`
  on dev seeds 1001-1030, 1440 paired cells per arm, with authored per-scenario
  budgets. The review reports 17,280 episodes and zero errors. Against main
  `7ecb1ce5c5c96b607d3eb52b0c8200160e9ddb4a`, train head
  `26daa3947f030d5bcb4541540c0454ebfc3101e7` changes socnav_sampling success by
  **-2.0 percentage points [95% CI -3.5, -0.8]** and pedestrian collisions by
  **+1.5 percentage points [95% CI +0.5, +2.8]**. Success is 1321/1440 (91.7%)
  on main and 1292/1440 (89.7%) on the train; pedestrian collisions are 37/1440
  (2.6%) and 59/1440 (4.1%). These are paired deltas with scenario-cluster
  bootstrap intervals (B=4000), not sealed evaluation results. Loss concentrates
  in `doorway_high`, `station_platform_medium`, `robot_crowding`,
  `crowd_navigation`, `doorway_low`, and `merging_medium`. ORCA and goal show
  no statistically significant change; they do not establish an offsetting gain.
- **Distribution bisect:** The whole delta is introduced by combo merge
  `126e438dda94144c52ae0df89efd2cc8f9a0f63d`: main and pre-combo
  `fef9a94ebc23f1b23d5bd67096356f351f83171d` have zero differing outcomes across
  1440 cells, and post-combo and the reviewed train head have zero differences,
  for both socnav_sampling and ORCA. **#10008 is not the cause at distribution
  level**; its interaction with the combo is not statistically significant.
- **Mechanism probe:** All 40 new collision cells are pedestrian contacts at
  centre distances 1.31-1.40 m; 24 occur by step 66. Enabling pedestrian
  prediction on those selected cells rescues 35/40 (five still collide).
  This is a selected mechanism probe, not a fix or an unbiased performance
  estimate. It does not authorize enabling prediction in the release config.
- **Earlier sampler ruling re-scoped:** The earlier 2026-10-01 moving-timeout
  ruling, implemented in `ea0f729e265e04a5d56ab95b0586fc486125e832`, concerns
  only the non-release `classic_interactions_francis2023_goal_zone_entry_v1.yaml`
  / `socnav_sampling_bounded_v2.yaml` test cell. Its 182-to-400-step change is
  not release evidence. In the release configuration, `robot_crowding` / dev
  seed 1004 changes from main success at step 325 to a train **pedestrian
  collision at step 32**, minimum distance 1.39 m. The non-release safety,
  freeze, and progress checks remain; they make no release-success claim.
- **Evidence:** Review rr10080 (rr10080_report.md), archived in the private-ops review archive, finding F1 and
  claim 4; PR #10080. The review binds Slurm jobs 16110, 16112, and 16117 to
  development seeds only.
- **Alternatives:** Tune the release sampler to recover the old outcomes;
  retain the incorrect crowd world. Neither was adopted.
- **Enforced by:** none yet: distribution-acceptance receipt check. The existing
  `tests/benchmark/test_issue_9727_socnav_sampling.py::test_non_release_goal_zone_entry_v1_bounded_v2_replay_safety`
  enforces only the earlier non-release safety contract.
- **Reopen:** New material evidence about the corrected world or method
  fidelity, or an explicit author/orchestrator decision, may reopen this ruling.
  Keep the dev-seed disclosure separate from sealed release evaluation; other
  roster arms were not distribution-swept by this review.

## Verified catch-up decisions (2026-10-01)

Verification snapshot: main `2d145e3a80f12445877c3709b8c1aecb30072897`; GitHub issue/PR metadata and fetched open PR heads checked on 2026-10-01.
Historical measurements below are attributed to the named review report; they were not rerun for this records lane. Planned disclosures are decisions, not claims that the release notes or thesis already contain them.

### D-051: Existing tests may step the retired seeds 111-140; the sealed seeds stay forbidden
- **Date:** 2026-10-01
- **Question:** 27 existing test files step the retired 0.0.7 evaluation seeds
  111-140. Every full test run had to deselect them, so no full suite ever ran
  undeselected. Do they have to stay excluded?
- **Choice:** Existing tests may step 111-140. New code, probes, experiments and
  reviews may not. The 30 sealed 0.0.8 seeds stay strictly forbidden everywhere
  outside the sealed campaign. Moving these tests to development seeds is
  follow-up work for 0.1.0.
- **Reason:** 111-140 were the published 0.0.7 evaluation set. Running them
  cannot leak anything about the new sealed seeds, and full suites become
  complete again.
- **Decided by:** author; expressly reconfirmed on 2026-10-01 evening (records-lane instruction).
- **Alternatives:** keep deselecting the 27 files in every full run;
  migrate all of them to development seeds before the freeze.
- **Amends:** D-049, which says "Retired seeds 111..140 have no execution
  exception". The exception is now: existing tests only.
- **Evidence:** Author records-lane instruction, 2026-10-01 evening;
  author-rule summary, 2026-10-01 13:32 ("Seed rule (user)"); lane rules.
  These sources confirm the existing-test exception without an unsupported
  chat timestamp.
- **Implemented in:** process rule (lane rules); #10053 had already migrated
  the seeds it touched to development seeds.
- **Enforced by:** none yet: distinguish existing retired-seed tests from new episode work. `tests/validation/test_check_seed_holdout_diff.py::test_new_sealed_names_require_explicit_allowlist` covers added sealed names only; the current runtime test guard still rejects the retired band. This implementation gap does not rescind the author ruling.
- **Reopen:** New material evidence or an explicit author ruling.

### D-052: Held-out guard refusal witnesses may name a sealed seed
- **Date:** 2026-10-01
- **Question:** Merge train 2 stopped because three tests from #10053 pass a
  real sealed seed (50036) to the held-out guard. Is that a sealed-seed use?
- **Choice:** Permitted. A test may attempt a sealed seed if it expects the
  guard to refuse before any reset or step. If such a test ever reaches a
  reset or step, that is a STOP, reported with the node id.
- **Reason:** These tests prove the guard works at the simulation boundary.
  They never step the seed while the guard holds.
- **Decided by:** orchestrator (delegated), 2026-10-01 10:32. The private
  notes list it among the day's "author decisions" with the annotation
  "(my ruling)"; it is a delegated ruling, not an author decision.
- **Alternatives:** deselect the witnesses; treat them as violations.
- **Evidence:** train-2 lane stop and resume instruction; #10076 item 5.
- **Implemented in:** #10053 (merged in #10080).
- **Enforced by:** `tests/test_heldout_seed_guard.py::test_standalone_static_rng_only_rejects_at_simulation_boundary`, `tests/test_heldout_seed_guard.py::test_multiprocessing_child_protected_before_reset`, `tests/test_heldout_seed_guard.py::test_worker_guard_active`. These are refusal witnesses, never permission to reset or step sealed seeds. #10076 item 5 tracks replacing real-seed witnesses with injected sentinels.
- **Reopen:** New material evidence or an explicit author ruling.

### D-053: The plain `ppo` arm uses PPO retrained on the release robot
- **Date:** 2026-10-01
- **Question:** The retrained PPO (release robot: 2.0 m/s, no reverse,
  +-1 m/s^2, +-1 rad/s^2, velocity-delta actions) is about four times better
  than the published checkpoint but still collides often. What goes into the
  0.0.8 `ppo` row?
- **Choice:** Option 1. Replace the published checkpoint with the retrained
  policy `ppo_release_robot_b1002_last_20261001`, chosen by a pre-registered
  rule (most successes on 48 scenarios x dev seeds 1001-1005, then fewer
  collisions within five successes of the maximum). The release notes call it
  "a collision-prone learned reference", not a competitive baseline.
- **Reason:** The published checkpoint was trained on a different robot
  (3 m/s, instant velocity change, reverse, instant turn rate). Its 85 %
  collisions mainly measure that mismatch, which an examiner would question.
  Dev evidence (240 episodes per arm): retrained 151 successes / 89
  collisions; published 35 / 204; guarded_ppo 153 / 5; social_force 162 / 2;
  orca 213 / 22. About 85 % of the retrained policy's collisions are with
  static geometry.
- **Decided by:** author: "Option 1 is good. But yes, more training for 0.0.9
  could become interesting." (chat, 2026-10-01 07:28).
- **Alternatives:** option 2, keep the published checkpoint and
  explain the mismatch; option 3, drop the plain `ppo` row.
- **Supersedes:** the rejected alternative in D-017 ("replace the release PPO
  arm with the retrained one") and D-017's "supplementary arm" framing. The
  release tracker #10013 still lists the retrain as a supplementary campaign.
- **Evidence:** PPO evaluation lane report (dev seeds); #10071; #10077 body;
  model prerelease `models-ppo-release-robot-2026-10-01`, asset SHA-256
  `764a7d88f5b608237641d973634899e05a67b25e65f8b1607cfca025459824bc`;
  model card `model/cards/ppo_release_robot_0_0_8.md`; ppoeval_report.md, private-ops review archive.
- **Implemented in:** #10077 (open; supersedes #10072). Further training: #10071
  (milestone 0.1.0).
- **Enforced by:** (open PR #10077) `tests/benchmark/test_ppo_release_robot_binding.py::test_release_campaign_resolver_binds_plain_ppo_to_release_robot`, `tests/benchmark/test_ppo_release_robot_binding.py::test_release_ppo_registry_has_training_and_observation_contract`, `tests/baselines/test_ppo_action_semantics.py::test_new_registry_checkpoint_decodes_signed_velocity_delta`, `tests/integration/test_ppo_release_robot_asset.py::test_release_robot_asset_resolves_and_verifies_sha256`; none yet: release-notes wording check.
- **Reopen:** New material evidence or an explicit author ruling.

### D-054: The PPO velocity-delta reading stays after the plausibility check
- **Date:** 2026-10-01
- **Question:** Under the correct velocity-delta reading (#9995, D-016) the
  published PPO fell from about 50 % to 18 % success in rehearsal 1. Revert
  #9995?
- **Choice:** Keep #9995. The checkpoints were trained with velocity-delta
  semantics; 18 % is an honest out-of-distribution result and is disclosed.
  0.0.7's roughly 50 % came from a misread action contract.
- **Reason:** The rehearsal plausibility check confirmed the training
  contract. Reverting would restore a reading the policy was never trained on.
- **Decided by:** orchestrator (delegated), 2026-10-01 01:03, after the
  plausibility review. Private label: D-059.
- **Alternatives:** revert #9995 in train 1 (`git revert -m 1` of its
  train merge); keep the old reading for the old checkpoint only.
- **Evidence:** rehearsal plausibility report (archived with the 0.0.8 review
  reports); #9995 verdicts; plaus_report.md, private-ops review archive.
- **Implemented in:** #9995 (merged in train 1, #10046). D-053 later replaced
  the published checkpoint in the `ppo` row.
- **Enforced by:** the D-016 tests
  (`tests/baselines/test_ppo_action_semantics.py::test_signed_delta_precedes_release_clipping`,
  `tests/baselines/test_ppo_action_semantics.py::test_release_checkpoints_fail_closed_without_delta_declaration`).
- **Reopen:** New material evidence or an explicit author ruling.

### D-055: Pedestrian obstacle force holds pedestrians back in narrow passages: disclose and flag rows
- **Date:** 2026-09-30
- **Question:** With correct walls, the released obstacle force stops a lone
  pedestrian in front of a 1.2 m door. Does that distort 0.0.8 results?
- **Choice:** Verdict B: some scenarios are affected, the headline results stay
  interpretable. Disclose in the release notes and the thesis with the text
  on #10061. Flag the rows of classic_doorway_{low,medium,high},
  francis2023_narrow_hallway, the three-width doorway slice and the 2.0 m
  narrow-doorway probe (disclosure only). Draw no doorway- or hallway-specific
  planner claims and no door-width effects from them. Refit the obstacle force
  in the next release.
- **Reason:** Measured on dev seeds 1001-1005 at 1.0x, 0.3x and zero obstacle
  force: overall success for goal, ORCA and social_force moves by at most 2.5
  points at 0.3x and their ranking is unchanged; dropping the doorway
  scenarios changes any arm's overall success by at most 5 points. The
  distortion is local, so under D-001 it is disclosed, not fixed. In the
  slice the stand-off point moves with door width, so the slice cannot carry a
  width claim.
- **Decided by:** orchestrator (delegated). Private label: D-058.
- **Alternatives:** recalibrate the obstacle force before 0.0.8
  (changes every number); drop the affected scenarios.
- **Evidence:** #10061 comment of 2026-09-30 22:58
  (https://github.com/ll7/robot_sf_ll7/issues/10061#issuecomment-5921162115);
  obstacle-force investigation report; obst_report.md, private-ops review archive.
- **Implemented in:** disclosure (release notes draft; thesis intake diss#3027).
  Fix: #10061, #10074 (milestone 0.1.0).
- **Enforced by:** none yet: a release-notes presence check that lists the
  flagged scenario ids, plus a thesis check that no prose draws a door-width or
  doorway-ranking claim from those rows.
- **Reopen:** New material evidence or an explicit author ruling.

### D-056: The three-width doorway slice stays in 0.0.8, disclosed as measured against a queue
- **Date:** 2026-10-01
- **Question:** In the unchanged 0.0.8 model a lone pedestrian walking at an
  opening up to about 3.0 m wide stops about 2.25 m in front of it and never
  passes. In the slice (2.2 / 2.8 / 3.6 m) pedestrians queue instead of
  flowing through. Keep the slice, or drop it?
- **Choice:** Keep it in 0.0.8 with a disclosure that pedestrians queue in
  front of openings up to about 3 m wide; robot results there are measured
  against that queue. Improve in the next release. Next-release acceptance: a
  lone pedestrian passes every opening of 1.0 m and wider without stopping, and
  pedestrians flow through all three slice widths on dev seeds; then the slice
  is re-run.
- **Reason:** Even with queuing it shows how planners handle doors of different
  widths; dropping it would mean reopening finished tooling. The behaviour is
  the same for every planner and dates back to 0.0.7, so under D-001 it is a
  disclosure, not a blocker. Mechanism: when a wall segment's projection
  misses, it acts from its nearest endpoint; several polygon-post corner terms
  10/(d+0.57)^3 all push back and reach the desired-force cap v0/tau = 1.30 at
  about 2.5 m.
- **Decided by:** author: "keep and improve for 0.0.9" (chat, 2026-10-01
  13:03), on the orchestrator's recommendation to keep and disclose.
- **Alternatives:** drop the slice from 0.0.8.
- **Evidence:** #10074 comments 5932010889 and 5932040814; #10061 comment
  5932010502; doorway-check lane report; diss#3027 comment 5932041275; dooracc_report.md, private-ops review archive.
- **Implemented in:** release-notes draft; thesis intake diss#3027. Slice
  admission itself: D-075.
- **Enforced by:** none yet: a release-notes presence check for the queue
  sentence; the next-release acceptance lives in the 0.1.0 ledger.
- **Reopen:** New material evidence or an explicit author ruling.

### D-057: Crowd walking speed (0.65 m/s) and pedestrian body size are disclosed, not changed, for 0.0.8
- **Date:** 2026-10-01
- **Question:** Every crowd pedestrian gets spawn speed 0.5 m/s x 1.3 =
  0.65 m/s, which is both its desired speed and its hard cap, with no spread.
  Pedestrians are rigid discs of radius 0.40 m (0.35 m in the force kernel).
  Change for 0.0.8?
- **Choice:** No change in 0.0.8. The release notes and the thesis appendix
  state that crowd pedestrians walk at one shared 0.65 m/s (about half the
  measured adult free speed of 1.29-1.34 m/s) with no spread, so results
  describe slow, uniform crowds, and that pedestrians are discs 0.7-0.8 m wide.
  The literature-based speed model and a smaller radius go to the next release
  through the joint calibration.
- **Reason:** Either change alters every number and would restart the release
  path just before the freeze (D-001). Speed, radius and wall law are coupled
  (the wall stop sits where the wall force equals v0/tau), so they must be
  changed together, not one at a time.
- **Decided by:** orchestrator (delegated) for the 0.0.8 no-change and disclosure;
  author for moving the literature-based speed model to the next release
  (0.1.0 D-005; author direction, 2026-10-01 13:14).
- **Alternatives:** switch on the existing `typical` speed tier for
  0.0.8; shrink the radius for 0.0.8.
- **Evidence:** #10074 comment 5932171567 (speed source check); pedestrian
  validation baseline (draft #10075); diss research note
  `2026-10-01_chatgpt_pedestrian_dynamics.md`; pedval_report.md, private-ops review archive.
- **Implemented in:** release-notes draft; thesis appendix (planned).
- **Enforced by:** none yet: a release-notes presence check. D-039 describes scripted input speed; the crowd path derives both its desired speed and cap from the same 0.5 × 1.3 value, so there is no headroom above that target.
- **Reopen:** New material evidence or an explicit author ruling.

### D-058: Pedestrians do not steer around the robot and collisions are not attributed: disclose for 0.0.8
- **Date:** 2026-10-01
- **Question:** Pedestrian-robot repulsion is radial and collisions have no attribution. Change the model for 0.0.8?
- **Choice:** Disclose in 0.0.8 ("Simulated pedestrians slow down near the
  robot but do not steer around it, and collisions are not attributed").
  Narrow the thesis claims that pedestrians react to or avoid the robot. Fix in
  the next release (activation distance and attribution).
- **Reason:** Changing pedestrian-robot interaction changes every number. The 2.0 m centre-cutoff premise in #10065 and the plausibility report is refuted by the current call path: `PedRobotForce.__call__` adds force radius and robot radius to the configured activation threshold (2.0 + 0.35 + 1.0 = 3.35 m). The force is radial, so it has lateral components off-axis; it lacks an anticipatory side-selection mechanism. These distinctions must be preserved in the disclosure and next-release sensitivity design.
- **Decided by:** orchestrator (delegated) for the 0.0.8 disclosure, from the
  rehearsal plausibility check; author for the next-release direction (#10065
  comment, 2026-10-01 06:28).
- **Alternatives:** change the force before the freeze.
- **Evidence:** #10065 body and comments; diss#3027; plaus_report.md, private-ops review archive.
- **Implemented in:** disclosure; fix #10065 (milestone 0.1.0).
- **Enforced by:** none yet: release-notes presence check; thesis check
  for "pedestrians avoid the robot" wording.
- **Reopen:** New material evidence or an explicit author ruling.

### D-059: The robot cannot reverse in 0.0.8, and the release notes say so
- **Date:** 2026-10-01
- **Question:** `allow_backwards` is false in both drive models and the 0.0.8
  release does not override it. Enable reverse?
- **Choice:** Keep reverse off in 0.0.8 and state it in the release notes.
  Limited reverse is next-release scope (#10068; see the 0.1.0 ledger).
- **Reason:** Turning reverse on changes every planner's behaviour and every
  number, and most planners only sample forward moves, so the setting alone
  changes little without planner work.
- **Decided by:** author: "I agree, file it as 0.0.9 issue" (chat, 2026-10-01
  04:10), on the orchestrator's recommendation.
- **Alternatives:** enable reverse for 0.0.8; disallow reverse
  permanently.
- **Evidence:** #10068 body.
- **Implemented in:** release-notes draft.
- **Enforced by:** none yet: a test that the 0.0.8 release robot config
  resolves `allow_backwards: false`, plus a release-notes presence check.
- **Reopen:** New material evidence or an explicit author ruling.

### D-060: Occupancy-grid half-cell offset is disclosed for 0.0.8 and fixed with the PPO retrain
- **Date:** 2026-10-01
- **Question:** `rasterize_circle_fast` tests cell corners instead of cell
  centres, so each pedestrian appears about half a cell (+0.08 to +0.14 m)
  towards +x/+y in the grid. PPO and guarded PPO read the whole grid. Fix for
  0.0.8?
- **Choice:** Disclose as a known limitation in 0.0.8; fix in the next release
  together with the PPO retrain.
- **Reason:** It is a fixed systematic bias that the PPO policies were trained
  and evaluated with, so it does not change reported 0.0.8 numbers by itself.
  Fixing it without retraining would create a distribution shift for the
  learned arms. The retrained `ppo` arm (D-053) was also trained on the
  shifted grid, so the reasoning still holds.
- **Decided by:** orchestrator (delegated), 2026-10-01.
- **Alternatives:** fix the rasteriser for 0.0.8.
- **Evidence:** #10082 (found during the thesis filmstrip capture at
  `7ecb1ce5`, dev seed 1006); metgeo_report.md, private-ops review archive.
- **Implemented in:** disclosure; fix #10082 (milestone 0.1.0, blocks #10071).
- **Enforced by:** none yet: release-notes presence check. The fix needs the
  centroid regression test sketched in #10082.
- **Reopen:** New material evidence or an explicit author ruling.

### D-061: Time-to-collision is disclosed as centre-based; no metric-geometry finding blocks 0.0.8
- **Date:** 2026-10-01
- **Question:** A threshold check found metrics measured from the robot centre
  (D1-D6 in #10079). Do any change reported 0.0.8 numbers?
- **Choice:** None blocks 0.0.8. Wall and agent collisions in release rows come
  from the simulator's footprint check; a cross-check fails closed on
  disagreement (0 disagreements in 20 dev episodes). Time to collision ignores
  both radii and overstates the time by about 0.7 s at 2 m/s; it appears only
  in `metrics.time_to_collision_min` and `time_to_collision_min_mean` in
  `campaign_summary.json`, not in SNQI, tables or comparators. Disclose that
  caveat in the release notes. All fixes go to #10079.
- **Reason:** Under D-001 only measurable changes to reported numbers block.
- **Decided by:** orchestrator (delegated), from the metric-geometry lane
  (2026-10-01 12:34).
- **Alternatives:** treat D1/D2 as freeze blockers.
- **Evidence:** #10079; diss research note
  `2026-10-01_chatgpt_safety_metric_thresholds.md`; metgeo_report.md, private-ops review archive.
- **Implemented in:** release-notes draft.
- **Enforced by:** none yet: release-notes TTC caveat check and a release collision/footprint cross-check regression. The report describes 0/20 disagreements; no exact regression node proving that claim was located.
- **Reopen:** New material evidence or an explicit author ruling.

### D-062: 0.0.7 and 0.0.8 are different benchmarks; compare them only as distributions and disclose why
- **Date:** 2026-10-01
- **Question:** Rehearsal 1 showed large gains for several arms. How are
  0.0.7-vs-0.0.8 differences to be read?
- **Choice:** Disclose that 0.0.8 gains come mainly from new algorithm configs
  that fix 0.0.7 plant mismatches; that 0.0.7 and 0.0.8 are different
  benchmarks (#9856, #9762, #9725, #10026); that the 0.0.7 predictive_mppi
  baseline is invalid; that the #9764 planner-side kernel had no effect; that
  robot-radius alignment is incomplete (#4856). H400 budgets are fair. The
  comparator runs in distribution mode, not paired by seed.
- **Reason:** Different seeds, worlds and planner contracts make paired
  comparison meaningless; a reader must not attribute gains to planners alone.
- **Decided by:** orchestrator (delegated), from the plausibility check.
  Private label: D-061. Distribution-only comparison follows from the
  author's fresh-seed decision (D-049).
- **Alternatives:** a paired per-seed comparison.
- **Evidence:** rehearsal plausibility report; #10058 (distribution comparator); plaus_report.md, private-ops review archive.
- **Implemented in:** release-notes draft; #10058 (open).
- **Enforced by:** (open PR #10058) `tests/analysis/test_compare_release_distributions.py::test_definition_changed_never_differenced_and_success_support`, `tests/analysis/test_compare_release_distributions.py::test_schema_mixture_and_reversed_versions_refused`; none yet: disclosure presence check.
- **Reopen:** New material evidence or an explicit author ruling.

### D-063: `groups` keeps the large-crowd formula; small-crowd truncation is documented
- **Date:** 2026-09-30
- **Question:** D-035 fixed the meaning of `simulation_config.groups` (fraction
  of pedestrians in groups). For small crowds, integer allocation truncates the
  realised fraction. Allocate exactly?
- **Choice:** Keep the formula; document the small-crowd truncation (realised
  0.17 / 0.36 / 0.44 in the cases measured). Exact allocation is #10040.
- **Reason:** Exact allocation changes crowd composition and every affected
  number; the effect is small.
- **Decided by:** orchestrator (delegated). Private label: D-051.
- **Alternatives:** exact allocation for 0.0.8.
- **Evidence:** #10024 review rounds; #10040; rr10024b_report.md, private-ops review archive.
- **Implemented in:** #10024 (merged via the combo branch in #10080).
- **Enforced by:** none yet for the documented truncation values.
- **Reopen:** New material evidence or an explicit author ruling.

### D-064: Historical configs reproduce main exactly; 0.0.8 keeps authored budgets
- **Date:** 2026-09-30
- **Question:** An earlier delegated ruling said a historical fixed 600-step
  horizon extends authored scenario limits. A refute review measured that this
  premise was false: on main, `horizon: 600` only capped the runner loop, and
  each scenario stopped at its authored limit (effective = min(authored, 600)).
  The extension changed 1,505 published 0.0.7 rows under the same episode id.
- **Choice:** Historical configs reproduce main exactly under a policy named
  `legacy_runner_cap` (authored simulator limit, runner cap 600; provenance
  records authored, runner and effective budgets). They are admitted by an
  exact content registry (config SHA-256 to version and policy), with no config
  edits and no test-only injection. Only configs that declare a 0.0.8+
  protocol carry authored-budget authority; reserved
  `metadata.campaign_horizon.*` keys on inputs are refused, as is a passed
  horizon that differs from the scheduled budget. Release acceptance accepts
  per-scenario authored budgets. The v4 tuning config's effective budgets were
  500/500/400/400, equal to 0.0.8's for those families, so there is no tuning
  mismatch and nothing to disclose.
- **Reason:** Historical rows and ids must not change; the earlier premise was
  never measured.
- **Decided by:** orchestrator (delegated), correcting its own earlier ruling;
  the #9999 PR contract records author domain approval of the corrected rule.
  Private labels: "corrected D-050" and "D-054" (both used in the public #9999
  body; they collide with ledger D-050).
- **Alternatives:** `legacy_fixed_extends_authored`; editing
  historical configs; a test-only injection.
- **Evidence:** #9999 review comments RR9999d (P1 witness) and RR9999E (37
  historical configs raised, 88 re-hashed); #9999 body; rr9999e_report.md, private-ops review archive.
- **Implemented in:** #9999, still open; verified at its fetched head, not yet effective on main.
- **Enforced by:** (open PR #9999) `tests/benchmark/test_campaign_horizon_compatibility.py::test_every_tracked_campaign_preserves_main_admission_and_simulator_limits`, `tests/benchmark/test_scheduled_campaign_compatibility.py::test_historical_schedule_preserves_all_main_scenario_bytes`, `tests/benchmark/test_campaign_horizon_authority.py::test_historical_runner_cap_matches_main_oracle`.
- **Reopen:** New material evidence or an explicit author ruling.

### D-065: A non-release sampler test checks safety, not goal arrival, under its authored 400-step budget (re-scoped by D-050)
- **Date:** 2026-10-01
- **Question:** In merge train 2, `socnav_sampling` on crowding / dev seed 1004
  hit its 400-step limit with zero collisions instead of reaching the goal. The
  test still assumed 600 steps.
- **Choice:** (as issued) The authored per-scenario budget (400 steps) governs.
  The test asserts zero wall and total collisions and records goal arrival
  without requiring it, for this cell only. STOP instead of loosening if more
  than 50 consecutive steps have |v| < 0.05 m/s or the last 150 steps make no
  net progress. Bisect which constituent caused the slowdown.
- **Re-scope (D-050):** The cell is the non-release
  `classic_interactions_francis2023_goal_zone_entry_v1.yaml` /
  `socnav_sampling_bounded_v2.yaml` test cell. The ruling makes no release
  claim. In the release configuration the same scenario/seed is a pedestrian
  collision at step 32.
- **Reason:** The ruling as issued said "the per-scenario budget is the 0.0.8
  rule", which assumed a release cell. The single-cell bisect pointed at the
  combo merge and a #10008 forecast ablation; the distribution review refuted
  #10008 as the release-level cause.
- **Decided by:** orchestrator (delegated), 2026-10-01 12:04; re-scoped by
  the orchestrator's D-050 ruling the same afternoon.
- **Alternatives:** keep the unconditional goal assertion; extend the
  test to 600 steps.
- **Evidence:** #10080 body ("Earlier non-release sampler ruling and cell
  proof"); commit `ea0f729e`; refute review rr10080 finding F1; rr10080_report.md, private-ops review archive.
- **Implemented in:** #10080 (merged).
- **Enforced by:**
  `tests/benchmark/test_issue_9727_socnav_sampling.py::test_non_release_goal_zone_entry_v1_bounded_v2_replay_safety`.
- **Reopen:** New material evidence or an explicit author ruling.

### D-066: Metric v2 curvature is arc-length mean absolute curvature
- **Date:** 2026-09-30
- **Question:** The v1-style curvature (time mean of |v x a| / |v|^3) exploded
  near standstill (rehearsal p99 9.2e10; auditor false positives up to 2.1e13).
- **Choice:** `curvature_mean` = sum |wrap(dphi)| / max(sum ds, 1 m) over steps
  with ds >= 1e-3 m, in rad/m.
- **Reason:** Bounded, physically meaningful, and independent of speed near
  standstill.
- **Decided by:** authority disputed: private notes attribute a delegated orchestrator ruling; #10054 calls private label D-055 an author decision. The implemented formula is verified; author attribution remains unresolved pending confirmation.
- **Alternatives:** keep the time-mean formula with a speed floor.
- **Evidence:** #10054 body and review; #10065 item 3; curvfix_report.md, private-ops review archive.
- **Implemented in:** #10054 (merged in #10080).
- **Enforced by:** `tests/benchmark/test_curvature_v2.py::test_creep_segment_is_bounded`, `tests/benchmark/test_curvature_v2.py::test_stop_and_rotate_then_leave_counts_one_turn`, `tests/benchmark/test_curvature_v2.py::test_timestep_does_not_change_path_geometry`.
- **Reopen:** New material evidence or an explicit author ruling.

### D-067: SNQI v2 reporting rules: declared grid, matched seed draws, per-difference tie tolerance
- **Date:** 2026-09-30
- **Question:** The external SNQI v2 review (F1, F6, F8) found that reports
  accept undeclared or non-rectangular grids, compute ranking stability on
  different draws from the confidence intervals, and apply a tie tolerance
  globally.
- **Choice:** A non-rectangular or undeclared grid raises; ranking stability
  uses the same seed draws as the confidence intervals; the flip tie tolerance
  applies per difference.
- **Reason:** Each gap lets a report show a stability that was not measured.
  No finding changed rehearsal rankings (Spearman 1.0 in all 64 variants).
- **Decided by:** orchestrator (delegated). Private label: D-056.
- **Alternatives:** defer to 0.1.0.
- **Evidence:** SNQI v2 review verification report; snqiv_report.md, private-ops review archive.
- **Implemented in:** No implementing PR was located in the reviewed GitHub inventory; record as delegated intention, not completed enforcement.
- **Enforced by:** none yet: reject undeclared/non-rectangular report grids, reuse confidence-interval seed draws for stability, and apply tie tolerance per difference.
- **Reopen:** New material evidence or an explicit author ruling.

### D-068: SNQI v2 calibration uses development seeds 1001/1002
- **Date:** 2026-09-30
- **Question:** D-048 names development calibration seeds 101-102. Which seeds
  does calibration use now?
- **Choice:** Dev seeds 1001 and 1002, with 1003 held apart for the N2
  diagnostic. The calibration receipt carries the sealed-seed digest, and
  `validate_evaluation_seeds` rejects 1001-1030 and 101/102 as evaluation
  seeds. The file
  `configs/benchmarks/snqi_v2/calibration.dev1001_1002_scheduled_acquisition.yaml`
  now names both the actual seeds and scheduled acquisition purpose. #10045
  renamed it from `calibration.dev101_102.yaml`.
- **Reason:** One development band (1001-1030) for all development work.
- **Decided by:** orchestrator (delegated).
- **Alternatives:** 101/102.
- **Amends:** D-048 ("seeds 101-102").
- **Evidence:** private orchestration notes; #10045; rr10045b_report.md, private-ops review archive.
- **Implemented in:** #10045 (refresh implemented; PR open).
- **Enforced by:**
  `tests/unit/benchmark/test_snqi_v2.py::test_development_calibration_matches_candidate_and_preserves_frozen_007`
  (lines 79–94), which loads the new filename and asserts development seeds
  1001/1002 and parity with the authored candidate.
- **Reopen:** New material evidence or an explicit author ruling.

### D-069: 0.0.8 release rows are untraced; traces come from rehearsal 2
- **Date:** 2026-09-30
- **Question:** Tracing every release row needed about 72 GiB of memory and a
  3.7 GB archive. Where do diagnostic and illustration traces come from?
- **Choice:** Release rows are untraced. Traces come from rehearsal 2 (dev
  seed 1001) at the freeze commit. No sealed-seed trace run.
- **Reason:** Cost, and no sealed-seed step outside the campaign.
- **Decided by:** authority disputed: private notes attribute a delegated
  orchestrator ruling; #10047's body says "Author directed" for private
  D-057 untraced release rows. No author ruling on traces was found in the
  chat extract. The policy is recorded; author attribution remains unresolved
  pending confirmation of the original attribution. D-081 now records the
  orchestrator's explicit delegated approval of untraced release rows for the
  0.0.7-vs-0.0.8 comparison at #10047 head 31d09675.
- **Alternatives:** trace all release rows.
- **Evidence:** #10047 body and review RR10047; rr10047_report.md, private-ops review archive.
- **Implemented in:** #10047 (open).
- **Enforced by:** (open PR #10047) `tests/analysis/test_compare_release_0_0_7_to_0_0_8.py::test_small_real_trace_read_discards_series` verifies reading untraced rows; none yet: campaign-level prohibition of sealed diagnostic trace runs. Trace-schema tests do not enforce the untraced-release policy.
- **Reopen:** New material evidence or an explicit author ruling.

### D-070: The freeze commit is the train-2 head on main, after rehearsal 2 passes on it
- **Date:** 2026-09-30
- **Question:** D-004 named `release/0.0.8-freeze` at `5c27a404`. It predates
  #9926 (93ba0d75), which changes results.
- **Choice:** The freeze commit is the train-2 merge head on main, once
  rehearsal 2 (full pipeline, dev seeds 1001-1003) passes on it. Before
  minting, check that the freeze sha contains 93ba0d75 and every train
  constituent. Rehearsal 2 must run the release packaging tool (not only the
  camera-ready tool) so the publication bundle is exercised, and the thesis
  intake must run on that bundle before the sealed campaign.
- **Reason:** The first freeze came before the rehearsal proved the pipeline;
  rehearsal 1 found 8 pipeline defects no single review caught.
- **Decided by:** author (merge trains and rehearsal gate, 2026-09-30);
  orchestrator for the bundle requirement (intake-prep finding).
- **Alternatives:** keep `5c27a404`.
- **Supersedes:** D-004's freeze commit (the pinned-commit rule stays).
- **Evidence:** release tracker #10013; private notes; runbook_report.md, private-ops review archive.
- **Implemented in:** process; post-freeze runbook `docs/release/0.0.8/runbook.md`.
- **Enforced by:** none yet in this repository: a mint-side ancestry check
  for 93ba0d75 and each constituent.
- **Reopen:** New material evidence or an explicit author ruling.

**Rehearsal status, 2026-10-03 (CHAIN-3):** The original development projection
was refused before execution on `f52de283e3b60ec85910fd761e432d07ab748158`.
D-086 then authorised the distinct, permanently non-releasable identity.
The complete shared pipeline passed on tooling source `b8d5e970ab2428285aeb532b0d8e73b4601a33a6`
in [#10103](https://github.com/ll7/robot_sf_ll7/pull/10103): public generation
and verification, checkpoint staging, exact-source smoke (672 cells, Slurm
job 16798), and the prescribed 14 arms × 48 scenarios × seeds
1001–1003 campaign (2,016 cells, Slurm job 16800, 32 CPUs). Both jobs
exited 0. All 14 native arms completed without missing/unexpected identities,
exclusions or forbidden runtime statuses. The common runner's publication
export, `publication_preflight.py`, checksums, manifest/metadata roles and
commit/SNQI reconciliation passed. Development calibration diagnostics used
1,344 fit rows (1001/1002) and 672 held-apart rows (1003), covering all five
authored budget classes. D-062's `compare_release_distributions.py` passed in
diagnostic mode against the pinned published 0.0.7 archive; release mode
refused this identity with exit 2. No held-out reset or step occurred.

The dissertation intake archive is
`chain3_final_campaign_b8d5e970ab24_dev1001_1003_publication_bundle.tar.gz`
(8,206,322 bytes; SHA-256
`a280b1b7e3b60076b6b8d251bd30d5e7fca8ed74ddc490e40bf6092917007611`). Its copy's digest matches the exporter output.
All commands, exit codes and output paths are retained in the CHAIN3 lane
report and command journal for orchestrator handoff. The artifact remains
`release_kind: development_rehearsal` with `release_eligible: false`.
Packaging is verified; dissertation intake is pending the orchestrator's
copy and dry run (diss#3026). The main-head rehearsal/freeze decision remains
open; this branch evidence does not name a freeze SHA or grant release status.

### D-071: Public merge trains land with merge commits after a full suite on the exact train head
- **Date:** 2026-09-30 (trains); 2026-10-01 (full-suite rule)
- **Question:** GitHub's merge queue is unavailable for a user-owned repo, and
  PR CI skips slow tests, so train 1 turned main red after merging.
- **Choice:** Approved PRs land in manual merge trains built with
  `git merge --no-ff` of exact reviewed heads, merged with a merge commit so
  constituents auto-close. Before merging a train, run the full suite
  including slow tests on the exact train head on the workstations (never
  Slurm), and classify every failure.
- **Reason:** Serial merges invalidated each other; push CI on main runs slow
  tests that PR CI skips (#10048).
- **Decided by:** author (trains, 2026-09-30; workstation full suite, part of
  "yes, do this as proposed", 2026-10-01). Private label: D-053.
- **Alternatives:** serial merges; squash merges; suite runs on Slurm.
- **Evidence:** #10046, #10069 (main repair), #10080; #10070; train2_report.md, private-ops review archive.
- **Implemented in:** process.
- **Enforced by:** none yet (#10048, #10070).
- **Reopen:** New material evidence or an explicit author ruling.

### D-072: The sealed campaign runs on one node
- **Date:** 2026-09-30
- **Question:** guarded_ppo episodes are not bit-reproducible across hosts
  (8 of 20 reruns differ, #10052).
- **Choice:** Run the campaign on one node, record its CPU model in the
  receipt, and resume only on the same node.
- **Reason:** Resuming on another host would mix rows from different numerics.
- **Decided by:** orchestrator (delegated), from the adapter-review
  verification.
- **Alternatives:** spread across nodes.
- **Evidence:** #10052; advv_report.md, private-ops review archive.
- **Implemented in:** runbook; private launcher.
- **Enforced by:** none yet: a receipt check that every row carries the same
  CPU model.
- **Reopen:** New material evidence or an explicit author ruling.

### D-073: #10063 and #10064 block the freeze
- **Date:** 2026-10-01
- **Question:** The plausibility check found a robot spawn inside a pedestrian
  lane in francis2023_pedestrian_overtaking (11/30 dev seeds collide within 20
  steps) and a no-admissible-command fallback that brakes to zero and cannot
  reach its escape (risk_dwa, predictive_mppi, guarded PPO guard). Block?
- **Choice:** Both block the freeze: each changes reported numbers.
- **Reason:** D-001: measurable P1s block.
- **Decided by:** orchestrator (delegated). Private label: D-060.
- **Alternatives:** disclose only.
- **Evidence:** #10063, #10064; #10066 diagnostic gate (1,440 paired dev
  episodes; predictive_mppi 105 -> 149 successes); rr10066_report.md, private-ops review archive.
- **Implemented in:** #10066 (merged in #10080, closes #10064); #10067 (open,
  train 2 part 2, for #10063).
- **Enforced by:** `tests/planner/test_no_admissible_recovery.py::test_infeasible_progress_escape_is_selectable`, `tests/planner/test_no_admissible_recovery.py::test_mppi_infeasible_escape_beats_zero_with_real_clearance_costs`; (open PR #10067) `tests/validation/test_release_spawn_goal_overlap.py::test_overtaking_lane_cannot_intersect_full_robot_spawn_rectangle`.
- **Reopen:** New material evidence or an explicit author ruling.

### D-074: The release campaign template binds the sealed seeds
- **Date:** 2026-10-01
- **Question:** #9999 adds the authored-budget release template
  ([authored candidate on the open #9999 head](https://github.com/ll7/robot_sf_ll7/blob/9c9c53c9d7e05541f231b909ffce6d1c666fce5a/configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml)) with seed set `paper_eval_s30`
  (111-140), and #10039's sealed allowlist names only the other template. The
  strict guard would refuse the release run on release day.
- **Choice:** In train 2 part 2: point the template the campaign uses at the
  sealed seed set, add it to the allowlist (or derive the allowlist from the
  release template), retire the unused template, and add a test that the
  release template resolves to `EVAL_SEEDS_0_0_8` and passes the sealed guard
  while the retired band is refused.
- **Reason:** Fail-closed seam between two independently reviewed PRs.
- **Decided by:** orchestrator (delegated), from the release-notes lane.
- **Alternatives:** none recorded.
- **Evidence:** #10085; relnotes_report.md, private-ops review archive; statically verified candidate seed policy at the #9999 head and canonical-path allowlist in `robot_sf/benchmark/release_protocol.py` on main.
- **Implemented in:** Train 2 / [#9999](https://github.com/ll7/robot_sf_ll7/pull/9999), under D-083, merged at `879f75b69eb93ca16006f2019ee5c85c7aa724dd`. [#10085](https://github.com/ll7/robot_sf_ll7/issues/10085) was closed on 2026-10-02 after the orchestrator accepted CHAIN-2 verification against main `f52de283e3b60ec85910fd761e432d07ab748158`. The development packaging rehearsal remains a separate D-070 / CHAIN-3 process requirement.
- **Enforced by:** `tests/benchmark/test_release_campaign_authority.py::test_d083_authored_template_resolves_exact_sealed_seed_file` and `tests/benchmark/test_release_campaign_authority.py::test_d083_materialized_selected_template_passes_guard_without_execution` passed on that main SHA: exact sealed-file transport, public identity materialization and guard admission with recording stubs, retired-band and retired-template refusal. Environment construction, reset and step are blocked by the witness fixture.
- **Reopen:** New material evidence or an explicit author ruling.

### D-075: Release acceptance admits the doorway slice as its own bound release kind
- **Date:** 2026-10-01
- **Question:** Release acceptance requires exactly 20,160 cells and 48
  scenarios, so it refuses the 1,260-cell doorway slice that D-049 makes part
  of the sealed evaluation.
- **Choice:** Recognise a bound `benchmark-doorway-width-slice.v1` kind: widths
  2.2 / 2.8 / 3.6 m, the same 14-arm roster and sealed seeds, 1,260 cells,
  H400, its own denominator. Main acceptance stays exactly as strict.
- **Reason:** Both datasets come from one freeze and must pass the same strict
  runner.
- **Decided by:** orchestrator (delegated).
- **Alternatives:** relax the main count; run the slice outside the
  strict runner.
- **Evidence:** #10078; #10081; review rr10081 (FIX: a wrong-seed witness
  survives a mutant; retarget onto the train head); rr10081_report.md, private-ops review archive.
- **Implemented in:** [#10081](https://github.com/ll7/robot_sf_ll7/pull/10081), merged in train 2 at `879f75b69eb93ca16006f2019ee5c85c7aa724dd`.
- **Enforced by:** `tests/benchmark/test_doorway_release_acceptance.py::test_bound_doorway_slice_accepted` (lines 83–89), `tests/benchmark/test_doorway_release_acceptance.py::test_main_refuses_slice_denominator` and `tests/benchmark/test_doorway_release_acceptance.py::test_slice_refuses_consistent_wrong_seed_inventory`. The complete static doorway witness file passed on main `f52de283e3b60ec85910fd761e432d07ab748158` on 2026-10-02, including the consistent wrong-seed refusal. No sealed environment was reset or stepped.
- **Reopen:** New material evidence or an explicit author ruling.

### D-076: Release notes live in the repository and cite decisions by content
- **Date:** 2026-10-01
- **Question:** Where do the 0.0.8 release notes live, and how do they refer to
  decisions?
- **Choice:** `docs/release/0.0.8/release_notes.md` in the repository plus a
  short GitHub release body. The notes state decisions by content, not by
  ledger number.
- **Reason:** Versioned with the release; ledger numbers are still moving.
- **Decided by:** orchestrator (delegated).
- **Alternatives:** GitHub release body only.
- **Evidence:** release-notes lane report; relnotes_report.md, private-ops review archive.
- **Implemented in:** not yet (draft outside the repository).
- **Enforced by:** none yet: verify the versioned release-notes file exists and the release body points to it.
- **Reopen:** New material evidence or an explicit author ruling.

### D-077: Corrected wall order: fix the builders, replay with unchanged configs, mark old bundles superseded
- **Date:** 2026-09-30
- **Question:** The emergent-phenomena builders passed walls in the wrong
  coordinate order, so the face-validity and lane-formation campaigns ran
  without their walls (#10056).
- **Choice:** Fix the conversion, replay every affected campaign with its
  original configs, seeds, metrics and preregistered thresholds, keep the old
  bundles byte-identical with superseded notices, and change no force
  parameter.
- **Reason:** Thesis evidence must stand on correct geometry, and old bytes must
  stay comparable.
- **Decided by:** orchestrator (delegated).
- **Alternatives:** re-run with new thresholds; delete old bundles.
- **Evidence:** #10056; #10057 body (428 corrected runs + 428 same-runtime
  replays); diss#3028; rr10057_report.md, private-ops review archive.
- **Implemented in:** #10057 (merged in #10080).
- **Enforced by:** `tests/research/test_wall_segment_order.py::test_emergent_simulator_has_intended_walls`, `tests/research/test_wall_segment_order.py::test_generated_simulator_has_intended_walls`, `tests/research/test_wall_segment_order.py::test_force_microbenchmark_preserves_sampled_endpoints` (six parametrized real-simulator cases).
- **Reopen:** New material evidence or an explicit author ruling.

### D-078: Hybrid planners are labelled v4 in the thesis
- **Date:** 2026-10-01
- **Question:** 0.0.8 runs the v4 hybrids. Should the thesis keep v3 labels?
- **Choice:** Rename to v4 at intake.
- **Reason:** The text must match the release. The intake tooling refuses a
  partly renamed planner list.
- **Decided by:** author; expressly confirmed on 2026-10-01 evening. The earlier orchestrator default is now an author ruling.
- **Alternatives:** keep v3 labels.
- **Evidence:** Author records-lane instruction, 2026-10-01 evening; diss#3026 intake contract.
- **Implemented in:** diss#3026 (merged); rename itself at intake.
- **Enforced by:** none yet in this repository: verify the thesis planner list against the admitted v4 release roster during intake (diss#3026).
- **Reopen:** New material evidence or an explicit author ruling.

### D-079: Thesis basis is 0.0.8 until the stop-or-continue check
- **Date:** 2026-10-01 evening
- **Question:** Which release is the thesis written on?
- **Choice:** Write on 0.0.8. Switch to 0.1.0 only if it ships before submission would happen anyway; decide at the stop-or-continue check.
- **Decided by:** author, explicit evening ruling.
- **Reason:** Secure the existing result set without delaying submission for a new benchmark generation.
- **Alternatives:** Assume the next release will be the thesis basis; delay submission until it ships.
- **Evidence:** Author records-lane instruction, 2026-10-01 evening; 0.1.0 D-002.
- **Enforced by:** none yet: thesis intake records the selected release and the stop-or-continue ruling.
- **Reopen:** New material evidence or an explicit author ruling at that check.

### D-080: Decision records have one home, structured entries, a generated overview and a catcher
- **Date:** 2026-10-01 evening
- **Question:** How should later decision-record tooling prevent records being lost or duplicated?
- **Choice:** One home per record type, machine-readable entries, a generated overview, and a catcher check. Implementation is a separate later lane.
- **Decided by:** author, explicit evening approval.
- **Reason:** Private working labels and stale registers currently obscure the effective decisions.
- **Alternatives:** Continue copying independent decision lists across chats, PR bodies and ledgers.
- **Evidence:** Author records-lane instruction, 2026-10-01 evening; the private-label reconciliation below.
- **Enforced by:** none yet: the later records-tooling lane implements the catcher and overview consistency checks.
- **Reopen:** New material evidence or an explicit author ruling.

## Append-only amendments

| Original | Effective later record | Amendment |
|---|---|---|
| D-004 | D-070 | The old freeze commit is superseded by the train-2 head after rehearsal 2, packaging and intake pass. No final freeze SHA is asserted here. |
| D-017 | D-053 | The author replaces the plain PPO arm; the older supplementary-only choice remains historical. #10077 is still open. |
| D-048 | D-068 | Calibration uses 1001/1002. #10045 renamed the file to `calibration.dev1001_1002_scheduled_acquisition.yaml`; `test_development_calibration_matches_candidate_and_preserves_frozen_007` in `tests/unit/benchmark/test_snqi_v2.py:79–94` enforces the filename and seed policy. |
| D-049 | D-051, D-052 | Existing tests may execute retired seeds; refusal witnesses must stop before reset/step. Sealed seeds remain forbidden outside the sealed campaign. |
| D-039 | D-057 | Scripted input speed and crowd-derived desired speed are distinct paths; crowd target and cap are both 0.65 m/s. |
| D-038, D-040 | 0.1.0 D-001 | Historical references to the planned 0.0.9 release now mean 0.1.0. |

### D-081: Benchmark-domain approval for the release comparator at #10047 head 31d09675
- **Date:** 2026-10-01
- **Question:** Are D4, D6, untraced release rows and the publication-field
  overlay in #10047 approved at the reviewed head?
- **Choice:** APPROVED for #10047 at
  `31d09675bfcd7bdf91ee5c7ca6c27e82279867a5`, based on the rr10047c
  benchmark-domain review (section C):
  - **D4, the complete pinned-payload row binding:** approved. It binds
    configuration identity.
  - **D6, schema v2 plus reset angular velocity:** approved. It is metadata
    only, and trace steps, metrics and outcomes are byte-identical.
  - **D-057 (private label = public D-069), untraced release rows:** approved
    for the 0.0.7-vs-0.0.8 comparison. Recorded cost: release rows carry no
    trace-only fields (`progress_at_timeout`, `robot_force_samples`). Any
    analysis needing them must use the separate dev-seed trace slice.
  - **The release_tag/doi overlay:** approved as identity-only. It does not
    verify release identity.
  Conditions for the release step, owned by the release chain and not
  blocking this merge: (1) bind the successor rows to the published 0.0.8
  bundle SHA256SUMS (rr10047c P3-b); (2) exercise parsing of the real
  resolved-identity file at the release gate (P3-c).
- **Decided by:** orchestrator (delegated), 2026-10-01 17:08 UTC.
- **Reason:** The domain review accepts configuration binding and the
  metadata-only changes within their stated boundaries. Untraced rows save
  tracing cost while giving up trace-only fields; the identity-only overlay
  does not establish published release identity. The two release-step
  conditions preserve that distinction.
- **Alternatives:** Leave benchmark-domain approval pending; require every
  release row to carry trace-only fields; treat the publication-field overlay
  as verification of release identity. None was adopted by this ruling.
- **Evidence:** rr10047c review, private-ops review archive, section C and
  findings P3-b/P3-c; [delegated benchmark-domain ruling for #10047](https://github.com/ll7/robot_sf_ll7/pull/10047#issuecomment-5936508831).
- **Enforced by:** The strict v2 PR contract's `domain_approval` for #10047.
  none yet: the 0.0.8 runbook release-identity admission gate must bind successor rows to the
  published bundle SHA256SUMS and exercise parsing of the real
  resolved-identity file before the release step. The ruling also requires
  the held-out guard on the development trace script and the string-valued
  extra-key overlay test killing M3 in #10047's fix round before MERGE;
  approval is not a claim that those fixes or the release gates have run.
- **Reopen:** New material evidence, a changed implementation head, or an
  explicit author/orchestrator ruling.

### D-082: The release-distribution comparison uses a scenario-conditional primary estimand
- **Date:** 2026-10-01
- **Question:** What resampling, multiplicity and conditioning rules should
  #10058 use for the 0.0.7-vs-0.0.8 release comparison?
- **Choice:** The comparison asks whether the release changed outcomes on
  this fixed benchmark suite. The primary estimand is therefore
  scenario-conditional: the 48 scenarios are fixed, seeds are resampled
  within each scenario, and the two releases are resampled independently
  because their seed sets are disjoint. Report the paired joint scenario
  resample (one set of scenario indices for both releases, matched by
  scenario_id) as a sensitivity column.
  The 28 predeclared primaries form their own family, corrected with Holm at
  alpha 0.05. Unit and arm cells form a separate exploratory family with BH
  at q = 0.05, labelled as exploratory. Success-conditioned metrics carry a
  visible "conditional on success; estimands differ when success rates
  differ" marker in the md and csv outputs. After the fix round,
  `domain_approval` becomes approved at the new head, given a delta review.
- **Decided by:** orchestrator (delegated), 2026-10-01 17:08 UTC.
- **Reason:** The current independent two-stage scenario resample counts
  between-scenario variance twice and is not acceptable as the primary;
  rr10047c's null split-half widths are reported as 1.4-5.4x. The fixed-suite
  question motivates the scenario-conditional primary, with the paired
  joint resample retained to show sensitivity to scenario sampling. Holm
  protects the predeclared primary family; BH results remain exploratory.
- **Alternatives:** The current independent two-stage scenario resample as
  primary; the paired joint scenario resample as primary. Both were rejected.
- **Evidence:** rr10047c review, private-ops review archive, P2-A and
  P3-a/P3-b/P3-c; [delegated comparison-method ruling for #10058](https://github.com/ll7/robot_sf_ll7/pull/10058#issuecomment-5936509251).
- **Enforced by:** none yet: pending in #10058's fix round in
  `tests/analysis/test_compare_release_distributions.py`: update
  `test_two_stage_bootstrap_retains_between_scenario_variation` to pin the
  new primary; add the null split-half known-difference-zero regression with
  a width bound, the Holm-28-primary/separate-BH-exploratory-family check,
  the M5 BH step-up monotonicity test, the M6 test that `changed` requires
  the difference CI to exclude 0, and the md/csv conditional-on-success
  marker check. These named intended checks are pending, not enforcement
  already present on main; new node IDs are not yet committed. The strict
  v2 PR contract must require benchmark `domain_approval` at the new head
  after the fix round and delta review.
- **Reopen:** New material evidence, a changed estimand or implementation,
  or an explicit author/orchestrator ruling.

### D-083: The authored-budget candidate is the single authoritative 0.0.8 campaign
- **Date:** 2026-10-02
- **Question:** Which candidate must the release identity and sealed guard select for #10085?
- **Choice:** Select `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml` as the single authoritative 0.0.8 release campaign template, consistent with D-064's authored budgets. Bind its named seed policy to the sealed seed-set file, the current release matrix and approved 14-arm release bindings. The alternate `paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate.yaml` is retired from release admission and retained for historical static references. The generic campaign template is also no longer a canonical sealed execution path. Identity materialization must select and hash the authored template; no inline seed list replaces the sealed file.
- **Decided by:** orchestrator, delegated direction for #9999, 2026-10-02.
- **Reason:** D-074 identified the seam: the authored candidate resolved the retired seed band while the sealed allowlist selected another template. One source-bound authority closes that seam without changing authored-budget policy or historical artifacts.
- **Alternatives:** Keep two executable candidates; return to the retired seed band; select the fixed-H600 candidate. None is adopted.
- **Evidence:** [#10085](https://github.com/ll7/robot_sf_ll7/issues/10085), implemented in [#9999](https://github.com/ll7/robot_sf_ll7/pull/9999); the issue was closed after accepted CHAIN-2 verification on 2026-10-02. The D-086 development pipeline completed in [#10103](https://github.com/ll7/robot_sf_ll7/pull/10103), with full 2,016-cell packaging and diagnostic comparison recorded under D-070. Dissertation intake and the main freeze decision remain open; the diagnostic bundle cannot grant release status.
- **Enforced by:** `tests/benchmark/test_release_campaign_authority.py::test_d083_authored_template_resolves_exact_sealed_seed_file` and `tests/benchmark/test_release_campaign_authority.py::test_d083_materialized_selected_template_passes_guard_without_execution`: real seed transport, public identity materialization, source-bound guard admission with recording workers, and retired-band/canonical-path refusal without reset or step.
- **Reopen:** New material evidence about release input binding or an explicit author/orchestrator ruling.

### D-084: Pedestrian overtaking uses a 600-step authored release budget
- **Date:** 2026-10-02
- **Question:** Can `francis2023_pedestrian_overtaking` retain a 400-step budget while requiring a real overtake?
- **Choice:** Give this scenario a 600-step budget in the authoritative 0.0.8 schedule and authored scenario. No other scheduled budget changes. A current-protocol schedule below its scenario's authored limit is refused before execution; historical schedules retain their original behavior.
- **Decided by:** author ruling, 2026-10-02, relayed by the orchestrator for #9999 and #10067.
- **Reason:** The required 0.7 speed cap needs 456–490 steps. Raising the cap to finish within 400 steps does not preserve a real overtake; the release budget must represent the intended interaction, consistent with D-064.
- **Alternatives:** Keep H400 and truncate; raise the cap and lose the intended overtake. Neither is adopted.
- **Evidence:** Opus review P1-1 of [#10067](https://github.com/ll7/robot_sf_ll7/pull/10067), author ruling 2026-10-02. #9999 includes the authored single-scenario H600 change, schedule and loader floor guard so its release path is self-consistent. #10067 makes the identical source change and owns removal of the release-matrix override.
- **Enforced by:** `tests/benchmark/test_campaign_horizon_contracts.py::test_d084_selected_release_schedules_overtaking_at_600` resolves the real authoritative campaign through runner binding; `tests/benchmark/test_campaign_horizon_contracts.py::test_schedule_below_authored_limit_is_refused` checks current-protocol refusal and historical preservation without episode execution.
- **Reopen:** New material evidence about the intended overtaking interaction or an explicit author/orchestrator ruling.


### D-085: Guarded-PPO overtaking is reported outside its trained speed range
- **Date:** 2026-10-02
- **Question:** How should the guarded_ppo / francis2023_pedestrian_overtaking
  release cell be interpreted under the 0.7 m/s scenario speed cap?
- **Choice:** Report the cell with the explicit caveat that the policy runs
  outside its trained speed range (policy v_max 2.0 m/s; scenario cap 0.7 m/s).
  Keep the cap and pedestrian overtake; authorise 600 steps in the source scenario.
- **Reason:** Refute traces show about 94% speed-clamp saturation, terminal
  misses of 0.5–1 m and an inactive guard. This cell measures cap/budget
  sensitivity and must not be presented as a planner-quality regression or
  parked-pedestrian obstruction.
- **Decided by:** author, explicit 2026-10-02 ruling in the OVTFIX2 task.
- **Alternatives:** change the cap (refuted: late or absent overtakes), omit
  the cell, or report it without the speed-envelope caveat.
- **Evidence:** PR #10067 refute review of 9a5f187c; author ruling, 2026-10-02.
- **Implemented in:** PR #10067; existing scenario metadata.plausibility.notes
  for this planner/scenario cell. Authored horizon schedule and scheduled
  feasibility guard belong to PR #9999.
- **Enforced by:** tests/validation/test_release_spawn_goal_overlap.py::test_guarded_ppo_overtaking_cell_preserves_speed_envelope_caveat;
  tests/validation/test_release_spawn_goal_overlap.py::test_overtaking_budget_is_authored_and_release_inherits_it.
- **Reopen:** changed policy training envelope, scenario speed cap or explicit author ruling.
### D-086: Development packaging identities are permanently non-releasable
- **Date:** 2026-10-02
- **Question:** How can D-070 rehearse the full release path without running held-out seeds?
- **Choice:** Add the explicit public resolver opt-in `generate --development-rehearsal --development-seeds 1001,1002,1003`. Only unique development seeds 1001–1030 are admitted; retired 111–140 and the sealed tuple are refused before execution. Materialize the same D-083 template, matrix, planners, budgets and source closure with a digest-covered `release_kind: development_rehearsal` marker and diagnostic coordinates. The common runner, Slurm wrapper, exporter, bundle validators and D-062 comparator remain in use. The preparatory smoke uses the same 14 × 48 matrix on seed 1001; the campaign uses seeds 1001–1003. Results and bundles retain `release_eligible: false`; the comparator requires diagnostic mode.
- **Decided by:** orchestrator, explicit CHAIN-3 decision, 2026-10-02, following merged records PR [#10102](https://github.com/ll7/robot_sf_ll7/pull/10102).
- **Reason:** The sealed-only resolver prevented the development full-pipeline rehearsal. A distinct diagnostic identity provides executable packaging evidence while preserving sealed admission.
- **Alternatives:** Relax the sealed gate or run held-out seeds. Neither is authorised.
- **Evidence:** CHAIN-3's original exit-2 refusal and subsequent complete shared pipeline are recorded under D-070 and [#10103](https://github.com/ll7/robot_sf_ll7/pull/10103), source `b8d5e970ab2428285aeb532b0d8e73b4601a33a6`. Public CLI/recording-worker witnesses use real source-bound D-083 inputs without construction/reset/step. The real relative-root comparator invocation exposed a path-normalization defect; the same source-bound witness fails before the correction and passes with absolute and relative roots afterward.
- **Enforced by:** `tests/benchmark/test_release_development_rehearsal.py::test_public_development_identity_round_trip`, `test_public_rehearsal_refuses_non_development_inventory`, `test_rehearsal_marker_cannot_be_stripped`, `test_rehearsal_cannot_acquire_release_status`, `test_rehearsal_publication_contract_refuses_release`, `test_shared_exporter_preserves_non_release_marker`, `test_development_smoke_keeps_shared_admission_and_digest_checks` and `test_development_comparator_uses_same_pinned_runtime`. The parameterized release-boundary witness covers sealed/full acceptance, mint preflight, DOI metadata/binding, tag tooling, comparator release mode and release runtime-smoke admission. Fast release-boundary/export coverage is enforced by `tests/benchmark/test_development_rehearsal_release_boundaries.py::test_development_release_boundaries_and_publication_bytes` (real public identity and exporter, no steps). Shared orchestration is additionally enforced by `tests/tools/test_run_benchmark_release.py::test_development_identity_uses_shared_runner_without_release_success`, the detached-source and full development-grid witnesses in `tests/analysis/test_development_pinned_runtime.py`, and the positive receipt witness in `tests/benchmark/test_development_rehearsal_smoke_admission.py`.
- **Reopen:** Changed diagnostic scope or explicit author/orchestrator ruling. Rehearsal output can never become release evidence by removing a marker.

## Private working labels

These private labels were never public ledger IDs. Read each citation in its PR context; a collision does not supersede the public entry of the same number. Bodies of the 100 most recently updated PRs were scanned, including every PR citing the labels named in this catch-up request.

| Private label | Public ledger home | Citation / interpretation |
|---|---|---|
| D-050 (initial horizon extension) | not adopted | #9999; refuted premise, replaced by the corrected rule. Public D-050 is the crowd-distribution acceptance. |
| D-050 (corrected horizon), D-054 | D-064 | #9999; reproduce historical main limits, authored authority only for current protocols. |
| D-051 | D-063 | Group-allocation truncation; private notes, #10024 review context. |
| D-052 | unresolved | unresolved: no source defines private D-052; the draft's duplicate claim is unverified |
| D-053 | D-071 | Manual merge trains and exact-head full suite. |
| D-055 | D-066 | #10054 implements curvature; #10058 cites its changed-definition exclusion. Formula verified, author/delegated attribution unresolved. |
| D-056 | D-067 | SNQI-v2 reporting rules; implementing PR not located. |
| D-057 | D-069 | #10047; untraced release rows and rehearsal diagnostic traces. |
| D-058 | D-055 | Obstacle-force disclosure. |
| D-059 | D-054 | Keep velocity-delta PPO semantics. |
| D-060 | D-073 | Overtaking and no-command recovery block the freeze. |
| D-061 | D-062 | Different benchmarks; distribution-only comparison. |
| D-062 | D-053 | #10077 and superseded #10072; replace the plain PPO arm. |

## Enforcement gaps in the catch-up entries

Fields marked `none yet:` identify the intended checks. A cited open-PR test is not yet effective on main, and a structural ledger test does not enforce any runtime or disclosure decision.
