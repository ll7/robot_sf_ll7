# Candidate Report: shielded_ppo_issue1474_collision20_v1 (smoke)

## Decision

pass

## Launch Packet Gate

fail

The generic policy-search smoke decision is `pass` because the single episode had no collision or
near miss. The issue #1396/#1474 launch packet is stricter: smoke requires success `1.0`, collision
`0.0`, and guard fallback rate at most `0.60`. This run had success `0.0`, so nominal-sanity
escalation is blocked.

## Hypothesis

Replacing the historical BR-06 v3 PPO checkpoint inside the frozen risk_guarded_ppo_v1 runtime guard with the issue #1474 collision-20 repair checkpoint should reduce unsafe PPO proposals while preserving enough learned progress to pass the shielded-PPO smoke and nominal-sanity stop gates.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/shielded_ppo_issue1474_collision20_v1/smoke/issue1474_collision20_post_training_smoke/summary.json`
- Git commit: `dac66644c040ca3d25e334bce9815b926ff4be5f`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | n/a | 0.0000 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 0.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `overconservative_stop`: `1`

## Guard Diagnostics

The smoke JSONL reports `shield_decision_count=80`, `shield_intervention_count=0`,
`shield_override_count=0`, and `decision_counts.goal_reached=80`. The guarded wrapper selected zero
motion throughout the episode, producing a timeout rather than a useful smoke success.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | -0.2411 | n/a |
| orca | -0.1844 | -0.0355 | n/a |
| ppo | -0.2482 | -0.0993 | n/a |
