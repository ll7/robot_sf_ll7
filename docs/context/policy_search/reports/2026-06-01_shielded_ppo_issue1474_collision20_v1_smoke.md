# Candidate Report: shielded_ppo_issue1474_collision20_v1 (smoke)

## Decision

pass

## Hypothesis

Replacing the historical BR-06 v3 PPO checkpoint inside the frozen risk_guarded_ppo_v1 runtime guard with the issue #1474 collision-20 repair checkpoint should reduce unsafe PPO proposals while preserving enough learned progress to pass the shielded-PPO smoke and nominal-sanity stop gates.

Issue #2006 root cause: the candidate previously inherited the guarded-PPO default
`sensor_fusion_state` observation mode, but the runtime guard consumes SocNav goal fields. The guard
therefore parsed missing goal fields as zeros, classified every step as `goal_reached`, and selected
zero motion. The repaired candidate sets `params.observation_mode: socnav_state`.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/issue_2006_policy_smoke_h80_after/summary.json`
- Compact evidence: `docs/context/evidence/issue_2006_shielded_ppo_smoke_after_summary.json`
- Git commit: `3cc4a21e6e151e9830e89d07dff95bdbc5337536`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8866 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Guard Diagnostics

Before the issue #2006 repair, the smoke JSONL reported `shield_decision_count=80`,
`shield_intervention_count=0`, `shield_override_count=0`, and
`decision_counts.goal_reached=80`. After the observation-contract repair, the local smoke reported
`shield_decision_count=75`, `shield_intervention_count=10`, `shield_override_count=10`,
`decision_counts.ppo_clear=65`, and `decision_counts.fallback_safe=10`. Treat this as
single-episode launch-packet smoke evidence, not nominal, stress, or full-matrix evidence.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
