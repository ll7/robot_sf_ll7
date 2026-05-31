# Candidate Report: hybrid_rule_v3_progress_2p4_static_escape_probe (nominal_sanity)

## Decision

revise

## Hypothesis

Keeping the 2.4 m/s progress envelope while enabling the static-escape and recenter probes from the faster static-escape candidate may convert some low-progress stalls without adopting the 3.0 m/s fast-progress speed envelope.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_progress_2p4_static_escape_probe_nominal_issue1834_final/summary.json`
- Git commit: `fd66bceb514418f6250bbdc601962c2478dd9b99`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.0000 | 0.2778 | 4.1995 | 1.6116 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0833 | 0.0000 | 0.4167 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `4`
- `timeout_low_progress`: `10`

## Issue 1834 Interpretation

This probe is diagnostic-only and should not be promoted from the nominal-sanity result. It preserves
the original `hybrid_rule_v3_progress_2p4` nominal success rate recorded in
`docs/context/policy_search/experiment_ledger.md` (`0.2222`) and keeps collision rate at `0.0000`,
but it increases nominal near-miss rate from the ledger value (`0.1667`) to `0.2778` while
`timeout_low_progress` remains the dominant failure mode. The bounded static-escape/recenter
mechanism is therefore wired and runnable, but it has not yet repaired the 2.4 m/s progress
regression.

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.2411 | n/a |
| orca | +0.0378 | -0.0355 | n/a |
| ppo | -0.0260 | -0.0993 | n/a |

## Family Override Runs

- `classic`: success `0.0833`, collision `0.0000`
- `francis2023`: success `0.0000`, collision `0.0000`
- `nominal`: success `1.0000`, collision `0.0000`
