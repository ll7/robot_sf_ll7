# Candidate Report: shielded_ppo_issue1474_collision20_v1 (smoke)

## Decision

pass

## Hypothesis

Replacing the historical BR-06 v3 PPO checkpoint inside the frozen risk_guarded_ppo_v1 runtime guard with the issue #1474 collision-20 repair checkpoint should reduce unsafe PPO proposals while preserving enough learned progress to pass the shielded-PPO smoke and nominal-sanity stop gates. The first post-training smoke failed the launch-packet success gate with an overconservative zero-motion timeout; issue #2006 repaired the handoff, and this #2029 SLURM replay checks whether that repair survives the compute-node smoke path. The row remains prototype-gated until nominal-sanity escalation is explicitly approved and interpreted.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/shielded_ppo_issue1474_collision20_v1/smoke/issue2029_shielded_ppo_smoke_replay/summary.json`
- Git commit: `2b86ef4cf5c10553ae2a0cf32c91a243164c44d2`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.9998 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
