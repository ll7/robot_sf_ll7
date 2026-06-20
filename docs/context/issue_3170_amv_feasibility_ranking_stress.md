# Issue #3170 AMV Feasibility Ranking Stress Synthesis (2026-06-20)

Issue: [#3170](https://github.com/ll7/robot_sf_ll7/issues/3170)
Follow-up: [#3181](https://github.com/ll7/robot_sf_ll7/issues/3181)
Status: diagnostic-only stress-slice synthesis, not benchmark-strength or paper-facing evidence.

## Question

Does the current AMV/AMMV feasibility ranking hold across paired multi-seed,
multi-scenario evidence, or is it an artifact of a narrow scenario/seed slice?

## Decision

No general AMV feasibility ranking claim is justified from the currently
tracked evidence.

The available evidence separates into two incompatible scopes:

- `docs/context/issue_2446_amv_feasibility_ranking.md` records a diagnostic
  actuation-aware feasibility ordering from one `classic_cross_trap_high`
  seed-101 slice.
- `docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json`
  records the broadest existing paired AMMV/default slice, with 15 paired rows
  across five scenario families and three seeds per family, but every episode,
  frame, selected action, event, and AMMV force-vector delta is exactly zero.

That combination satisfies the issue stop rule: stop as diagnostic-only and
record the next execution prerequisite instead of claiming a ranking.

## Evidence Summary

| Evidence surface | Scope | Result | Claim boundary |
|---|---:|---|---|
| Issue #2446 actuation-aware feasibility ranking | 1 scenario x 1 seed x 2 candidates | command clipping improved by -0.0875, success delta 0.0 | diagnostic direction only |
| Issue #2432 AMMV/default trace selection | 1 scenario x 3 seeds x 2 arms | 0 non-identical pairs | diagnostic frame-delta screening |
| Issue #2434 AMMV/default scenario sweep | 5 scenarios x 3 seeds x 2 arms | 0 non-identical pairs, max metric/frame/action/event/force-vector delta 0.0 | diagnostic multi-scenario screening |

For the #2434 zero-event result, the exact one-sided zero-event 95 percent
upper bound on the non-identical pair rate is about 0.181, with the simple
rule-of-three bound at 0.2. The observed evidence therefore says "no difference
was seen in this compact slice," not "differences are impossible."

## Mechanism And Disagreement Classification

Mechanism fields available in the #2434 compact evidence include:

- episode metrics: status, success, collision counts, clearance, average speed,
  force metrics, and numeric metric deltas;
- frame-level deltas: robot, pedestrians, selected action, planner event, and
  AMMV pedestrian force vectors.

All reported deltas are zero. This means the AMMV/default pathway did not expose
a discriminating mechanism signal in the tracked compact evidence.

The only disagreement is across evidence surfaces:

- #2446 has a nonzero synthetic actuation feasibility direction but is too
  narrow for generalization.
- #2434 has broader scenario/seed pairing but no observed mechanism or outcome
  difference.

## Successor Update

Issue [#3181](https://github.com/ll7/robot_sf_ll7/issues/3181) now has a small direct successor
slice:
[issue_3181_amv_feasibility_ranking.md](issue_3181_amv_feasibility_ranking.md). The local
2-scenario x 2-seed synthetic diagnostic run found a bounded command-clipping direction for
`actuation_aware_hybrid_rule_v0` versus `hybrid_rule_v3_fast_progress`, with no fallback/degraded
rows. The result remains diagnostic-only: success stayed zero, one paired row collided in both
variants, and one row traded clipping improvement against worse final progress.

## Next Step

Issue [#3181](https://github.com/ll7/robot_sf_ll7/issues/3181) tracks the
smallest direct successor and should now be interpreted through the #3181 note above rather than as
an unrun gap.

#2446 remains a one-slice diagnostic predecessor; #3181 adds a small paired diagnostic direction,
but still no planner-ranking or paper-facing AMV claim.

## Evidence Artifact

Compact synthesis artifact:
[issue_3170_amv_feasibility_ranking_stress_2026-06-20/summary.json](evidence/issue_3170_amv_feasibility_ranking_stress_2026-06-20/summary.json)
