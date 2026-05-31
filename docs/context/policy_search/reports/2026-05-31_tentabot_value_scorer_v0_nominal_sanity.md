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
- Summary JSON: `output/policy_search/tentabot_value_scorer_v0/nominal_sanity/issue1826_safety_retune_final_h120/summary.json`
- Git commit: `30570b7ee4af32fb6d185cdd822f744fb5a9c284`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.0556 | 0.1667 | 4.2641 | 1.6149 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0833 | 0.0833 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `2`
- `static_collision`: `1`
- `timeout_low_progress`: `11`

## Issue #1826 Comparison (2026-05-31)

The safety retune lowers the hand-scored Tentabot-style scorer's speed/progress pressure and raises
static-clearance, dynamic-clearance, TTC, and hard-safety margins. Against the predecessor
2026-05-31 nominal-sanity report, success is unchanged at `0.2222`, collision rate drops from
`0.1111` to `0.0556`, and near-miss rate drops from `0.2222` to `0.1667`. The tradeoff is more
low-progress timeouts (`9` to `11`), so this remains a `revise` result rather than a promotion.

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.1855 | n/a |
| orca | +0.0378 | +0.0201 | n/a |
| ppo | -0.0260 | -0.0437 | n/a |
