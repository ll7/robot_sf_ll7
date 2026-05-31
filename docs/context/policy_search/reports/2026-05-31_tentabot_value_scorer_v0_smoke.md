# Candidate Report: tentabot_value_scorer_v0 (smoke)

## Decision

pass

## Hypothesis

A clean-room Tentabot-style primitive value scorer can reuse Robot SF's hybrid-rule candidate lattice and auditable route, clearance, pedestrian TTC, smoothness, and command-bound features without importing upstream Tentabot code, ROS/Gazebo dependencies, source assets, or hidden future state. The v0 scorer uses hand-scored teacher weights as a supervised spike baseline before any learned model is considered.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/tentabot_value_scorer_v0/smoke/issue1832_final_h80/summary.json`
- Git commit: `39e6417e783504cb6873b4b9bfae73f458f0a2f5`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8190 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
