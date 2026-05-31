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
- Summary JSON: `output/policy_search/tentabot_value_scorer_v0/nominal_sanity/issue1832_final/summary.json`
- Git commit: `39e6417e783504cb6873b4b9bfae73f458f0a2f5`

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

## Issue #1832 Progress-Recovery Probe

Issue #1832 tested three bounded recovery directions after the #1826 safety retune:

- Baseline reproduction at `issue1832_baseline`: 4/18 success, 1/18 collision, 3/18 near-miss,
  11 low-progress timeouts.
- Progress-pressure config retune at `issue1832_progress_recovery`: low-progress timeouts improved
  to 8, but collisions increased to 3/18 and near-miss episodes increased to 4/18. Rejected.
- Static recovery-only config retune at `issue1832_static_recovery_only`: low-progress timeouts
  improved to 10, but collisions increased to 2/18. Rejected.

The retained planner change treats negative goal-distance progress as stalled for corridor-subgoal
activation, which is the safer interpretation for recovery gating. On this nominal slice it did not
change aggregate outcomes relative to the #1831/#1826 baseline, so the candidate remains
`revise`; this is not progress-improvement evidence.

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.1855 | n/a |
| orca | +0.0378 | +0.0201 | n/a |
| ppo | -0.0260 | -0.0437 | n/a |
