# Issue #8068: observation-only one-frame goal-posterior baseline

This note records the implementation boundary for the `H=1` (one decision-point
frame) heading baseline. It is an implementation-integrity and synthetic-smoke
record, not evidence of calibrated human intention prediction or planner value.

## Public actor interface

`update_heading_goal_posterior` accepts only:

- a tracked pedestrian ID, global-frame current position, and global-frame current
  velocity;
- a `GoalCandidateSet` whose provenance, frame, role, availability, identity, and
  optional prior weights have been validated; and
- a `HeadingGoalPosteriorConfig` containing heading concentration, speed threshold,
  prior floor, unknown-hypothesis prior/likelihood, and stationary policy.

The result is a `GoalBeliefV1` with `source=observation_only`. It carries candidate
probabilities, explicit unknown mass, derived entropy, one observed history row, and
blockers for stationary motion, unavailable/non-point candidates, and unestimated
arrival/change probabilities. The one-frame likelihood uses only circular heading
alignment; it does not use distance-to-goal, route truth, force, acceleration, or
simulator state.

Point candidates are `active_waypoint`, `final_destination`, or `route_endpoint`.
`open_ray`, `unknown`, unavailable, and non-point hypotheses are retained as
unknown mass rather than being assigned invented point likelihoods. Same-ray
near/far endpoints therefore remain observationally ambiguous when their priors
match.

## Oracle boundary and wiring

`planner_oracle_goal_posterior_channel_from_state` is the explicitly named
upper-bound helper for the historical issue #4164 smoke. It reads true simulator
goal columns only in that compatibility/evaluation path and emits
`source=simulator_upper_bound`, `oracle_only=true`, and
`candidate_source=oracle_true_goal_identity`. The old
`planner_goal_posterior_channel_from_state` name remains only as a deprecated
compatibility wrapper with the same labels.

`RobotEnv` no longer constructs candidates from `states[:, 4:6]`. Because no public
candidate provider is configured in `EnvSettings` yet, an enabled environment
channel reports `candidate_provider_not_configured`; the default remains disabled
and unchanged. The hybrid-rule planner has an opt-in `goal_posterior_actor_only`
guard that rejects oracle channels before they can influence an actor evaluation.

## Reproducible smoke

Run the config-first fixture from the repository root:

```bash
scripts/dev/run_worktree_shared_venv.sh -- python \
  scripts/benchmark/run_goal_posterior_actor_smoke_issue_8068.py \
  --config configs/benchmarks/issue_8068_goal_posterior_actor_smoke.yaml \
  --output output/benchmarks/issue_8068_goal_posterior_actor_smoke.json
```

The fixture covers aligned east/west/north/south candidates, a stationary prior,
same-ray near/far endpoints, a candidate-set misspecification with dominant unknown
mass, no public candidates, and rotated copies. The compact report includes candidate
definitions and digests, probability vectors, entropy, unknown mass, blockers, source,
mode, and belief digests. It records no hidden true-goal identity.

The tracked receipt is
[`evidence/issue_8068_goal_posterior_actor_smoke_receipt.v1.json`](evidence/issue_8068_goal_posterior_actor_smoke_receipt.v1.json).

## Evidence boundary and next research directions

The smoke result is classified as `implementation_integrity_smoke` only. It supports
the API, numerical stability, frame invariance, explicit uncertainty, and leakage
boundary. It does not support calibration, prediction accuracy, planner-performance
improvement, or a benchmark ranking.

The next bounded research directions are public map/route/open-ray candidate
generation, an inverse-force versus heading comparator, and a stateful hierarchical
posterior with change detection. Each direction needs its own candidate provenance,
comparator, and evidence tier before benchmark claims are considered.
