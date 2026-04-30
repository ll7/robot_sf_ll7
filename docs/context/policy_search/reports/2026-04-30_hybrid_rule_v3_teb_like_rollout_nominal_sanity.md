# Candidate Report: hybrid_rule_v3_teb_like_rollout (nominal_sanity)

## Decision

revise

## Hypothesis

Adding an occupancy-grid route-guide candidate to the safety-filtered DWA scorer should recover progress in static/corridor local minima without giving back the collision improvement from the static footprint filter.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_teb_like_rollout_nominal_final/summary.json`
- Git commit: `93edf63efb9a5d91095387f157bc11ae072dbd74`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2778 | 0.0000 | 0.1667 | 3.8516 | 1.6830 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.1667 | 0.0000 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `1`
- `timeout_low_progress`: `12`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2636 | -0.2411 | +0.0000 |
| orca | +0.0934 | -0.0355 | -4.1663 |
| ppo | +0.0296 | -0.0993 | -3.3583 |
