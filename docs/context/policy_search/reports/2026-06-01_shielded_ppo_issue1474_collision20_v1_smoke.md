# Candidate Report: shielded_ppo_issue1474_collision20_v1 (smoke)

## Decision

pass

## Launch Packet Gate

pass

The issue #1396/#1474 launch packet requires smoke success `1.0`, collision `0.0`, and
guard fallback rate at most `0.60`. The issue #2006 local repair smoke observed success `1.0`,
collision `0.0`, near miss `0.0`, shield intervention rate `0.0`, and shield override rate `0.0`.
All `71` shield decisions were `ppo_clear`.

## Repair Note

Issue #2006 fixed the handoff by running this guarded-PPO candidate with the SocNav observation
contract used by the #1474 training checkpoint. The guard also now tracks `goal.current` before
switching to `goal.next` and honors array-shaped pedestrian counts so padded observation rows do
not become synthetic near-field blockers.

## Hypothesis

Replacing the historical BR-06 v3 PPO checkpoint inside the frozen risk_guarded_ppo_v1 runtime guard with the issue #1474 collision-20 repair checkpoint should reduce unsafe PPO proposals while preserving enough learned progress to pass the shielded-PPO smoke and nominal-sanity stop gates. The first post-training smoke failed the launch-packet success gate with an overconservative zero-motion timeout; issue #2006 repaired the local smoke handoff, but the row stays prototype-gated until SLURM smoke replay and nominal-sanity escalation are explicitly approved.

Issue #2006 root cause: the candidate previously inherited the guarded-PPO default
`sensor_fusion_state` observation mode, but the runtime guard consumes SocNav goal fields. The guard
therefore parsed missing goal fields as zeros, classified every step as `goal_reached`, and selected
zero motion. The repaired candidate sets `params.observation_mode: socnav_state`.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search_issue2006_local/summary.json`
- Compact evidence: `docs/context/evidence/issue_2006_shielded_ppo_smoke_after_summary.json`
- Git commit: `647e7726ae2d2951736037ba39a069c6effb1c23`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.9997 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Guard Diagnostics

Before the issue #2006 repair, the smoke JSONL reported `shield_decision_count=80`,
`shield_intervention_count=0`, `shield_override_count=0`, and
`decision_counts.goal_reached=80`. After the observation-contract and guard-extraction repair, the
local smoke reported `shield_decision_count=71`, `shield_intervention_count=0`,
`shield_override_count=0`, and `decision_counts.ppo_clear=71`. Treat this as single-episode
launch-packet smoke evidence, not nominal, stress, or full-matrix evidence.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
