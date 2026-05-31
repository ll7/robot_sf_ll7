# Candidate Report: adaptive_proxemic_selector_v1 (nominal_sanity)

## Decision

revise

## Hypothesis

A neutral-default local-context selector should avoid using the open proxemic profile unless sparse low-progress context suggests over-conservatism, while routing near-human, dense, or constrained passage contexts to the conservative profile. This row is diagnostic only until comfort and near-miss tradeoffs are reviewed.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `adaptive_proxemic_selector_v1`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/adaptive_proxemic_selector_v1/nominal_sanity/latest/summary.json`
- Git commit: `c260ea6db03f6da47f3fc78263162dd7ebc6de88`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.0000 | 0.2222 | 4.1717 | 1.6058 |

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
