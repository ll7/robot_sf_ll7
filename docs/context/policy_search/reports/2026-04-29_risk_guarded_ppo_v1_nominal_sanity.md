# Candidate Report: risk_guarded_ppo_v1 (nominal_sanity)

## Decision

revise

## Hypothesis

The existing PPO success signal can be preserved if unsafe short-horizon actions are vetoed and replaced with safer local controls.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/risk_guarded_ppo_v1/nominal_sanity/latest/summary.json`
- Git commit: `8c04f1023f7a201e539fc9529cfbcfa97a362ebf`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.2778 | 0.0556 | 5.0715 | 1.3903 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.4167 | 0.0833 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `static_collision`: `5`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | +0.0367 | +0.0000 |
| orca | -0.0177 | +0.2423 | -4.2774 |
| ppo | -0.0815 | +0.1785 | -3.4694 |
