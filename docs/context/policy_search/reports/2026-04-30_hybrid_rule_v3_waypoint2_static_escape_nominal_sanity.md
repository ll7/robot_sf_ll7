# Candidate Report: hybrid_rule_v3_waypoint2_static_escape (nominal_sanity)

## Decision

revise

## Hypothesis

Allowing only slow, non-worsening commands when the robot is already inside the conservative occupancy-grid static clearance band may reduce static deadlock timeouts without weakening occupied-cell collision rejection.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_static_escape_nominal/summary.json`
- Git commit: `a819e4071ba5fdc1177b96462c1c191edd175502`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2778 | 0.0556 | 0.0556 | 4.0204 | 1.7688 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.1667 | 0.0833 | 0.0833 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `1`
- `static_collision`: `1`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2636 | -0.1855 | +0.0000 |
| orca | +0.0934 | +0.0201 | -4.2774 |
| ppo | +0.0296 | -0.0437 | -3.4694 |
