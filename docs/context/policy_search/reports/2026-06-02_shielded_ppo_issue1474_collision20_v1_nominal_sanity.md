# Candidate Report: shielded_ppo_issue1474_collision20_v1 (nominal_sanity)

## Decision

revise

## Evidence Status

valid-nonfallback

## Hypothesis

Replacing the historical BR-06 v3 PPO checkpoint inside the frozen risk_guarded_ppo_v1 runtime guard with the issue #1474 collision-20 repair checkpoint should reduce unsafe PPO proposals while preserving enough learned progress to pass the shielded-PPO smoke and nominal-sanity stop gates. Issue #2006 repaired the local smoke handoff, and issue #2029 replayed the smoke gate successfully on SLURM. This nominal-sanity replay is valid guarded-PPO evidence, but it does not pass the issue #1396 launch-packet nominal stop gate.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/shielded_ppo_issue1474_collision20_v1/nominal_sanity/issue1474_shielded_ppo_nominal_sanity_w1b/summary.json`
- Git commit: `c73760c8b07613d4dff6fb1c1e4823573e209f43`
- SLURM job: `12703_0`

## Evidence Boundary

- Job `12703_0` ran with `workers=1`; all 18 rows recorded
  `algorithm_metadata.status=ok` and `fallback_reason=none`.
- The launch-packet nominal-sanity gate fails on success rate:
  `0.1667 < 0.2778`.
- The collision stop gate passes: `0.0000 <= 0.0556`.
- Guard diagnostics recorded 2020 shield decisions, 771 interventions/overrides, and 0 hard
  guard violations.
- Do not escalate this candidate to `stress_slice` or full-matrix evaluation without a repair or
  redesigned learned-progress mechanism.

## Discarded Attempt

Job `12701_0` used `workers=2` and is invalid as guarded-PPO evidence: all 18 rows recorded
`algorithm_metadata.status=fallback` with `fallback_reason=model_load_failed`. SLURM stderr traced
the failure to CUDA reinitialization in forked subprocesses, so those aggregate values represented
fallback-to-goal behavior rather than the trained PPO checkpoint.

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.0000 | 0.3333 | 4.3454 | 1.3222 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.0000 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 1.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `6`
- `timeout_low_progress`: `9`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | -0.2411 | n/a |
| orca | -0.0177 | -0.0355 | n/a |
| ppo | -0.0815 | -0.0993 | n/a |
