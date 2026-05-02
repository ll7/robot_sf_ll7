# Candidate Report: hybrid_rule_v3_progress_2p4 (nominal_sanity)

## Decision

revise

## Hypothesis

A moderate 2.4 m/s speed envelope with stronger progress pressure should recover long-route nominal-sanity timeouts without the near-miss and doorway regressions observed at 3.0 m/s.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_progress_2p4_nominal/summary.json`
- Git commit: `309a143d4052de3f1bd8cc0b11ffa155f786a017`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.0000 | 0.1667 | 3.7714 | 1.6308 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0833 | 0.0000 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `2`
- `timeout_low_progress`: `12`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.2411 | +0.0000 |
| orca | +0.0378 | -0.0355 | -4.1663 |
| ppo | -0.0260 | -0.0993 | -3.3583 |
