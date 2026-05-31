# Candidate Report: mpc_clearance_guarded_v1 (nominal_sanity)

## Decision

revise

## Hypothesis

A hard first-step static-clearance guard should reduce the known static-collision failure mode in the NMPC clearance sampler while exposing whether the added safety simply trades collisions for low-progress timeouts.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `nmpc_social`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/mpc_clearance_guarded_v1/nominal_sanity/issue1842_nominal/summary.json`
- Git commit: `7da2d414d7d62f181c0df7910a5dddd41dbada25`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.0000 | 0.2222 | 4.3392 | 1.4439 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.0000 | 0.3333 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `4`
- `timeout_low_progress`: `11`

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | -0.2411 | n/a |
| orca | -0.0177 | -0.0355 | n/a |
| ppo | -0.0815 | -0.0993 | n/a |
