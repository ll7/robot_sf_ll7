# Candidate Report: adaptive_proxemic_selector_v1 (smoke)

## Decision

pass

## Hypothesis

A neutral-default local-context selector should avoid using the open proxemic profile unless sparse low-progress context suggests over-conservatism, while routing near-human, dense, or constrained passage contexts to the conservative profile. This row is diagnostic only until comfort and near-miss tradeoffs are reviewed.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `adaptive_proxemic_selector_v1`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/adaptive_proxemic_selector_v1/smoke/latest/summary.json`
- Git commit: `c260ea6db03f6da47f3fc78263162dd7ebc6de88`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8941 |

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
