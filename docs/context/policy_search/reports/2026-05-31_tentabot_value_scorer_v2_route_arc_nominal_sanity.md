# Candidate Report: tentabot_value_scorer_v2_route_arc (nominal_sanity)

## Decision

revise

## Hypothesis

A clean-room Tentabot-style primitive value scorer can keep the v1 static safety gate while making route-arc progress a first-class value term for all accepted candidates. This tests whether route-local progress can reduce low-progress timeouts without buying that gain through more static collisions.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/tentabot_value_scorer_v2_route_arc/nominal_sanity/issue1877_v2/summary.json`
- Git commit: `8f5b3d04c16bbda05473bebb14832311e64dc0ce`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.1111 | 0.2778 | 4.3251 | 1.6371 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0833 | 0.1667 | 0.4167 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `4`
- `static_collision`: `2`
- `timeout_low_progress`: `8`

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.1300 | n/a |
| orca | +0.0378 | +0.0756 | n/a |
| ppo | -0.0260 | +0.0118 | n/a |
