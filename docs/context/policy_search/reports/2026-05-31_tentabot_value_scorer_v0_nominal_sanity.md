# Candidate Report: tentabot_value_scorer_v0 (nominal_sanity)

## Decision

revise

## Hypothesis

A clean-room Tentabot-style primitive value scorer can reuse Robot SF's hybrid-rule candidate lattice and auditable route, clearance, pedestrian TTC, smoothness, and command-bound features without importing upstream Tentabot code, ROS/Gazebo dependencies, source assets, or hidden future state. The v0 scorer uses hand-scored teacher weights as a supervised spike baseline before any learned model is considered.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/tentabot_value_scorer_v0/nominal_sanity/issue_tentabot_nominal_h120/summary.json`
- Git commit: `fd66bceb514418f6250bbdc601962c2478dd9b99`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.1111 | 0.2222 | 4.3433 | 1.6152 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0833 | 0.1667 | 0.3333 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `3`
- `static_collision`: `2`
- `timeout_low_progress`: `9`

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.1300 | n/a |
| orca | +0.0378 | +0.0756 | n/a |
| ppo | -0.0260 | +0.0118 | n/a |
