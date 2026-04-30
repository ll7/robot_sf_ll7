# Candidate Report: hybrid_rule_v3_static_margin0_waypoint3 (nominal_sanity)

## Decision

revise

## Hypothesis

A 3.0 m waypoint handoff distance may further reduce route-local stalls in long crossing/corridor scenes, but may be brittle around doorway waypoints.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_static_margin0_waypoint3_nominal/summary.json`
- Git commit: `1ca3d03cc3228c08ec1cbd61efc1675fd28b5f5a`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2778 | 0.0556 | 0.1111 | 3.6750 | 1.7100 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.1667 | 0.0833 | 0.1667 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `static_collision`: `1`
- `timeout_low_progress`: `12`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2636 | -0.1855 | +0.0000 |
| orca | +0.0934 | +0.0201 | -4.2219 |
| ppo | +0.0296 | -0.0437 | -3.4139 |
