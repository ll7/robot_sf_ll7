# Candidate Report: hybrid_rule_v3_fast_progress_static_escape (smoke)

## Decision

pass

## Hypothesis

Remaining fast-progress failures enter the conservative static-clearance band but remain above the robot footprint. Reusing the slow static-escape gate on the fast-progress incumbent, with a bounded 5 cm tolerance while staying above 1.0 m clearance, may recover crossing stalls without the unsafe fully relaxed static margin.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_smoke_h500_w1/summary.json`
- Git commit: `8e5ceb432d67fe46e58f2349959bfbe8520dad88`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8604 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | +0.0000 |
| orca | +0.8156 | -0.0355 | -4.3330 |
| ppo | +0.7518 | -0.0993 | -3.5250 |
