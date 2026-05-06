# Candidate Report: hybrid_rule_v3_waypoint2_route_lookahead8_static05 (full_matrix)

## Decision

tracked

## Hypothesis

The 8-cell route lookahead may become acceptable if paired with a 5 cm static hard margin, trading some doorway progress for collision-free behavior.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_lookahead8_static05/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Durable summary evidence: `not promoted`; the `output/` path is retained only as regeneration context.
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2639 | 0.0139 | 0.3333 | 3.4935 | 1.5057 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1739 | 0.0145 | 0.3623 |
| francis2023 | 75 | 0.3467 | 0.0133 | 0.3067 |

## Failure Taxonomy

- `near_miss_intrusive`: `38`
- `overconservative_stop`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `65`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2497 | -0.2272 | +0.0000 |
| orca | +0.0795 | -0.0216 | n/a |
| ppo | +0.0157 | -0.0854 | n/a |

Near-miss deltas are `n/a` for baselines that only have `near_misses_mean` count-style
reference values; the candidate aggregate reports an episode-level near-miss rate.
