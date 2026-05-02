# Candidate Report: hybrid_rule_v3_waypoint2_mild_comfort (nominal_sanity)

## Decision

revise

## Hypothesis

A mild dynamic-clearance increase on top of the waypoint2 candidate may reduce intrusive near misses without the static collision regression seen in the stronger comfort variant.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_mild_comfort_nominal/summary.json`
- Git commit: `1ca3d03cc3228c08ec1cbd61efc1675fd28b5f5a`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2778 | 0.0556 | 0.0556 | 3.9394 | 1.7285 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.1667 | 0.0833 | 0.0833 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `static_collision`: `1`
- `timeout_low_progress`: `12`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2636 | -0.1855 | +0.0000 |
| orca | +0.0934 | +0.0201 | -4.2774 |
| ppo | +0.0296 | -0.0437 | -3.4694 |
