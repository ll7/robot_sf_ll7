# Candidate Report: hybrid_rule_v3_static_margin0 (stress_slice)

## Decision

tracked

## Hypothesis

Enforcing static hard clearance over the full rollout while using no extra static margin beyond the reported robot radius should preserve static safety without the doorway freezing caused by a 5 cm buffer in tight passages.


## Evaluation Scope

- Stage: `stress_slice`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/stress_slice_matrix.yaml`
- Seed manifest: `configs/policy_search/stress_slice_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_static_margin0_stress/summary.json`
- Git commit: `309a143d4052de3f1bd8cc0b11ffa155f786a017`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.2917 | 0.0000 | 0.2500 | 4.7441 | 1.6522 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.3333 | 0.0000 | 0.2500 |
| francis2023 | 12 | 0.2500 | 0.0000 | 0.2500 |

## Failure Taxonomy

- `near_miss_intrusive`: `6`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2775 | -0.2411 | +0.0000 |
| orca | +0.1073 | -0.0355 | -4.0830 |
| ppo | +0.0435 | -0.0993 | -3.2750 |
