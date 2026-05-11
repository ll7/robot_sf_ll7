# Issue #1059 Deferred Planner-Improvement Program

Issue: [#1059](https://github.com/ll7/robot_sf_ll7/issues/1059)

Status date: 2026-05-09

## Goal

Preserve the planner-improvement program as evidence-gated work instead of starting from aggregate
h500 scores alone. The first planner child must name exact scenarios and seeds, compare against a
strict incumbent safety envelope, require activation diagnostics and rejection reasons, and avoid
treating full h500 promotion as complete until a targeted slice has passed.

## Evidence That Unblocked A Child

The trace-backed mechanism evidence now exists:

- [Issue #1049 H500 Mechanism Pilot](issue_1049_h500_mechanism_pilot.md) retains fixed-h100 versus
  h500 traces for three ORCA cells.
- [Issue #1056 H500 Failure Classification](issue_1056_h500_failure_classification.md) defines the
  routing vocabulary that separates reporting follow-up, planner follow-up, and scenario-contract
  blockers.
- [Issue #1055 Exposure-Aware H500 Tables](issue_1055_exposure_aware_h500_tables.md) records the
  exposure-aware interpretation table used to avoid treating h500 success as a single winner signal.
- [representative_trace_summary.csv](evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv)
  is the compact retained seed-level table.

The child-promoting cell is `classic_merging_low` seed `111` under h500. In #1049 and #1056 it is
classified as `safety_regressed_long_horizon`: fixed h100 times out without collision, while h500
continues into collision after force exposure begins. That makes it a planner-follow-up candidate
only under a strict safety envelope, not a clean h500 success claim.

## First Child Program

The first child task is already present and completed:

- [#1034](https://github.com/ll7/robot_sf_ll7/issues/1034) / PR
  [#1036](https://github.com/ll7/robot_sf_ll7/pull/1036): continuous-collision-checked corridor
  recovery candidate for the classic-merging blocker slice.

#1034 named the required target seeds:

- `classic_merging_low` seeds `111` and `113`
- `classic_merging_medium` seeds `111`, `112`, and `113`

It also required:

- config-gated implementation,
- continuous static-collision checking or a proven equivalent,
- activation diagnostics and rejection behavior,
- targeted h500 step diagnostics,
- `nominal_sanity` and `stress_slice` h500 gates,
- no fallback or degraded execution counted as success.

PR #1036 reported targeted recovery evidence:

- `classic_merging_low` seed `111`: route-complete success at step `481`, no collisions.
- `classic_merging_low` seed `113`: route-complete success at step `315`, no collisions.
- `classic_merging_medium` seeds `111` and `112`: route-complete success, no collisions.
- `classic_merging_medium` seed `113`: safe h500 timeout, no collisions.
- h500 `nominal_sanity`: 18 episodes, success `1.0`, collision `0.0`, execution mode `adapter`.
- h500 `stress_slice`: 24 episodes, success `1.0`, collision `0.0`, execution mode `adapter`.

This satisfies the #1059 requirement to promote a narrow planner-improvement child from trace
evidence and to require targeted proof before broader promotion.

## Strict Incumbent Envelope

The strict h500 incumbent for comparison remains
`scenario_adaptive_hybrid_orca_v2_collision_guard`, with
`scenario_adaptive_hybrid_orca_v1` as the raw-success comparator.

The relevant h500 strict envelope is:

- `nominal_sanity`: success at least `0.80`, collision at most `0.02`.
- Scenario-stratified guard: classic collision at most `0.07`, Francis collision at most `0.05`.
- Full h500 promotion claims must evaluate the configured gates in
  [configs/policy_search/promotion_gates.yaml](../../configs/policy_search/promotion_gates.yaml).

## Remaining Boundary

PR #1036 is targeted recovery evidence, not full-matrix benchmark-strengthening evidence. It did
not run or claim a full `full_matrix_h500` promotion-scale campaign for
`hybrid_rule_v3_fast_progress_static_escape_continuous`.

Follow-up:

- [#1113](https://github.com/ll7/robot_sf_ll7/issues/1113) runs the full h500 promotion matrix for
  the continuous corridor recovery candidate and decides whether it remains targeted-only or earns
  broader promotion consideration.

Until #1113 is complete, the program boundary is:

- planner work may cite #1034/#1036 as targeted classic-merging recovery evidence,
- paper or benchmark text must not describe the continuous candidate as a full-matrix h500
  promotion,
- `classic_merging_medium` seed `113` remains an explicit safe-timeout limitation,
- fallback or degraded rows remain non-success evidence.

## Validation

This note is a documentation and routing update. Validation should check that the cited notes,
configs, issue, and PR exist, and then run the normal docs/change readiness gate before PR handoff.
