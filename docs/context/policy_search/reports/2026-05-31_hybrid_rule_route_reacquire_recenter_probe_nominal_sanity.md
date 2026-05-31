# Candidate Report: hybrid_rule_route_reacquire_recenter_probe (nominal_sanity)

## Decision

revise

## Hypothesis

Combining the moderate 2.4 m/s progress envelope with route-corridor subgoal recovery, static recenter probes, and a narrow rotate-in-place recovery path should test whether existing hybrid-rule hooks can escape h500 route-local-minimum blockers without weakening hard static or dynamic safety filters.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_route_reacquire_recenter_probe/nominal_sanity/issue1905_nominal/summary.json`
- Git commit: `629c8402560999df3138e802b5e803f25bae5992`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.0000 | 0.2222 | 4.1389 | 1.6586 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0833 | 0.0000 | 0.3333 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `3`
- `timeout_low_progress`: `11`

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.2411 | n/a |
| orca | +0.0378 | -0.0355 | n/a |
| ppo | -0.0260 | -0.0993 | n/a |

## Family Override Runs

- `classic`: success `0.0833`, collision `0.0000`
- `francis2023`: success `0.0000`, collision `0.0000`
- `nominal`: success `1.0000`, collision `0.0000`
