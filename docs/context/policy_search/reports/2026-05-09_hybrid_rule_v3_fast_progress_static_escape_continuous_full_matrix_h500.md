# Candidate Report: hybrid_rule_v3_fast_progress_static_escape_continuous (full_matrix_h500)

## Decision

tracked

## Hypothesis

Environment-bound continuous static checks should let the fast-progress static-escape planner distinguish real obstacle contact from conservative occupancy-grid clearance bands, while the corridor-subgoal sequence primitive verifies turn-then-forward recovery maneuvers before selection. A larger hard dynamic safety margin preserves pedestrian safety after exact static clearance admits more motion through narrow route slices.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `docs/context/evidence/issue_1113_continuous_h500_2026-05-10/continuous_candidate_summary.json`
- Report naming pattern: `YYYY-MM-DD_<candidate>_<stage>.md` (date format: `YYYY-MM-DD`)
- Git commit: `e75cb985c1ab896f988f7c60db7d3692b480edca`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.9167 | 0.0139 | 0.3958 | 2.9479 | 1.4194 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.9275 | 0.0145 | 0.4783 |
| francis2023 | 75 | 0.9067 | 0.0133 | 0.3200 |

## Failure Taxonomy

- `near_miss_intrusive`: `5`
- `overconservative_stop`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `4`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9025 | -0.2272 | n/a |
| orca | +0.7323 | -0.0216 | n/a |
| ppo | +0.6685 | -0.0854 | n/a |

## Family Override Runs

- `classic`: success `0.9275`, collision `0.0145`
- `francis2023`: success `0.9167`, collision `0.0139`
- `francis2023__francis2023_perpendicular_traffic`: success `0.6667`, collision `0.0000`
