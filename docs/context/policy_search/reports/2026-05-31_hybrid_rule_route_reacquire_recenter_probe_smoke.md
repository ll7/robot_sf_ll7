# Candidate Report: hybrid_rule_route_reacquire_recenter_probe (smoke)

## Decision

pass

## Hypothesis

Combining the moderate 2.4 m/s progress envelope with route-corridor subgoal recovery, static recenter probes, and a narrow rotate-in-place recovery path should test whether existing hybrid-rule hooks can escape h500 route-local-minimum blockers without weakening hard static or dynamic safety filters.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_route_reacquire_recenter_probe/smoke/issue1905_smoke/summary.json`
- Git commit: `629c8402560999df3138e802b5e803f25bae5992`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8613 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |

## Family Override Runs

- `nominal`: success `1.0000`, collision `0.0000`
