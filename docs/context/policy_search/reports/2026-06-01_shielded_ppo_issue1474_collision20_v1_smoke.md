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


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search_issue2006_local/summary.json`
- Git commit: `b6f17e45de2d12303a692353d615bf98397caa07`

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

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
