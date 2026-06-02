# Candidate Report: shielded_ppo_issue1474_collision20_v1 (nominal_sanity)

## Decision

revise

## Evidence Status

invalid-degraded

## Hypothesis

Replacing the historical BR-06 v3 PPO checkpoint inside the frozen risk_guarded_ppo_v1 runtime guard with the issue #1474 collision-20 repair checkpoint should reduce unsafe PPO proposals while preserving enough learned progress to pass the shielded-PPO smoke and nominal-sanity stop gates. The first post-training smoke failed the launch-packet success gate with an overconservative zero-motion timeout; issue #2006 repaired the handoff, and #2029 replayed the smoke gate successfully on SLURM.

This nominal-sanity attempt is invalid as guarded-PPO evidence: job `12701_0` used `workers=2`, and all 18 rows recorded `algorithm_metadata.status=fallback` with `fallback_reason=model_load_failed`. The SLURM stderr explains the failure as CUDA reinitialization in forked subprocesses, so the aggregate below is fallback-to-goal behavior, not the trained PPO checkpoint.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/shielded_ppo_issue1474_collision20_v1/nominal_sanity/issue1474_shielded_ppo_nominal_sanity/summary.json`
- Git commit: `c397e26c22b32b50c481e7e9658e08ae6561a160`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.1667 | 0.3889 | 4.8775 | 1.2658 |

Do not use these aggregate values as #1474 nominal-sanity evidence.

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.2500 | 0.3333 |
| francis2023 | 3 | 0.0000 | 0.0000 | 1.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `6`
- `static_collision`: `3`
- `timeout_low_progress`: `6`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | -0.0744 | n/a |
| orca | -0.0177 | +0.1312 | n/a |
| ppo | -0.0815 | +0.0674 | n/a |
