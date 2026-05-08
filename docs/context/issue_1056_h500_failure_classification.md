# Issue 1056 H500 Failure Classification

Date: 2026-05-07

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1056>

## Goal

Classify fixed-horizon versus h500 outcomes by scenario, time budget, planner mechanism, and
reporting risk so follow-up work is driven by observed evidence instead of aggregate score deltas.

## Evidence Sources

Primary inputs:

* `docs/context/issue_1045_h500_solvability_mechanisms.md`
* `docs/context/issue_1049_h500_mechanism_pilot.md`
* `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv`
* `docs/scenario_certification.md`

This classification uses retained issue #1049 traces for observed seed-level claims. Broader
aggregate labels from issue #1045 remain useful for selecting cells, but they are not treated as
causal proof without trace or video evidence.

## Classification Vocabulary

| Class | Evidence requirement | Interpretation | Follow-up owner |
|---|---|---|---|
| `time_budget_clean_relief` | Fixed horizon times out or remains unfinished; h500 succeeds without collision, near-miss, or force/comfort exposure increase. | Fixed horizon hid a route/time-budget artifact. | Reporting: separate strict-time benchmark from long-horizon sensitivity. |
| `exposure_enabled_completion` | Fixed horizon is unfinished; h500 succeeds while force/comfort exposure, near-miss count, or minimum pedestrian clearance worsens. | Longer horizon enables completion by spending more time in interaction pressure. | Reporting: exposure-aware h500 table; planner follow-up only if repeated traces show avoidable behavior. |
| `safety_regressed_long_horizon` | Fixed horizon is unfinished or clean; h500 introduces collision or sustained force/comfort exposure. | Longer horizon reveals unsafe behavior that strict timeout can hide. | Planner or scenario follow-up depending on scenario certificate and recurrence. |
| `persistent_low_progress_timeout` | Both fixed and h500 remain unfinished, with no collision and little progress. | Planner cannot make progress under current route/scenario interaction. | Planner follow-up when scenario is certified eligible; scenario follow-up if route/certification is suspect. |
| `scenario_contract_blocker` | Failure aligns with invalid, geometrically infeasible, kinodynamically infeasible, dynamically overconstrained, or route-clearance-warning evidence. | Do not attribute failure to planner mechanism yet. | Scenario certification or benchmark-contract follow-up. |
| `unsupported_wait_then_go_hypothesis` | Aggregate h500 success improves, but traces/videos do not show intentional waiting/yielding followed by progress. | Waiting remains a hypothesis, not paper language. | Evidence follow-up: videos or richer planner-decision traces. |

## Representative Classification

| Scenario | Planner | Seed | #1045 aggregate mechanism | #1049 observed class | Observed evidence | Follow-up boundary |
|---|---|---:|---|---|---|---|
| `classic_bottleneck_low` | ORCA | 111 | `budget_limited_clean_completion` | `time_budget_clean_relief` | Fixed h100 records 100 steps without route completion; h500 succeeds at step 102 with zero force-exposure, comfort-exposure, near-miss, or collision events. | Reporting follow-up only. This supports a strict-time artifact claim, not a planner-defect issue. |
| `classic_t_intersection_medium` | ORCA | 111 | `exposure_enabled_completion` | `exposure_enabled_completion` | Fixed h100 times out with 9 force-exposure steps and comfort exposure sum 3.0; h500 succeeds at step 182 with 50 force-exposure steps, comfort exposure sum 16.667, and min pedestrian distance dropping from 2.235 m to 1.413 m. | Use in exposure-aware tables. Do not call it a near-miss-timing example because discrete `near_misses` remains zero. |
| `classic_merging_low` | ORCA | 111 | `safety_regressed_completion` | `safety_regressed_long_horizon` | Fixed h100 times out without collision; h500 reaches collision at step 272 after force exposure starts at step 259. | Planner follow-up is justified only after checking recurrence and current scenario certification. This seed is not a clean h500 completion. |

## Planner vs Scenario vs Reporting Boundaries

Use these routing rules for h500 follow-ups:

* Create a reporting/table issue when h500 changes interpretation without showing a repeated
  fixable planner behavior. `time_budget_clean_relief` and one-off exposure pressure belong here.
* Create a planner issue when a certified-eligible scenario repeatedly shows the same fixable
  mechanism: low-progress timeout, collision after specific exposure phase, route-local minimum, or
  missing recovery.
* Create a scenario/certification issue when the failure overlaps route-clearance warnings,
  infeasible geometry, dynamic overconstraint, or certification gaps.
* Keep unsupported causal language out of paper text. In particular, do not write that h500 wins are
  mostly wait-then-go behavior until traces or videos show intentional yielding followed by progress.

## Scenario Certification Interaction

`scenario_cert.v1` answers whether the scenario is eligible, stress-only, or excluded. It does not
prove planner solvability. H500 classification should therefore be layered on top of certification:

1. Excluded or unresolved-certification scenarios cannot support planner-failure attribution.
2. Eligible or `hard_but_solvable` scenarios may support planner follow-ups when the failure class
   recurs across seeds or planners.
3. `knife_edge` scenarios should be reported as stress evidence unless a benchmark issue explicitly
   promotes them into a headline surface.

## Reusable Reporting Language

Recommended wording:

> H500 separates strict time-budget artifacts from longer-horizon interaction costs. Some fixed
> failures become clean completions, but other h500 successes carry higher exposure or comfort
> pressure, and some longer runs reveal collisions that fixed horizons hide as timeouts.

Avoid:

> H500 mainly shows planners waiting for pedestrians to pass.

That claim is unsupported by the retained #1049 traces.

## Validation

Validation performed for this classification:

* Links checked against `docs/context/issue_1045_h500_solvability_mechanisms.md`,
  `docs/context/issue_1049_h500_mechanism_pilot.md`, and `docs/scenario_certification.md`.
* Classification rows are derived from
  `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv`.

No code classifier was introduced, so no new tests are required beyond docs/link checks and
`rtk git diff --check`.
