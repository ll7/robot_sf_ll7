# Candidate Report: actuation_aware_hybrid_rule_v0 (amv_actuation_smoke)

## Decision

pass

## Hypothesis

A deterministic hybrid-rule local scorer can penalize command candidates that would be clipped by the synthetic AMV actuation envelope, exposing whether local command smoothing reduces synthetic actuation stress. This is diagnostic-only synthetic evidence, not calibrated hardware or paper-facing AMV performance evidence.


## Evaluation Scope

- Stage: `amv_actuation_smoke`
- Algorithm: `actuation_aware_hybrid_rule_v0`
- Scenario matrix: `configs/scenarios/sets/classic_cross_trap_subset.yaml`
- Scenario filter: `classic_cross_trap_high`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/actuation_aware_hybrid_rule_v0/amv_actuation_smoke/latest/summary.json`
- Git commit: `946a91b7ce8a077f74fd0300e7a71e3f0d008a45`
- Synthetic actuation profile: `amv-actuation-stress-v0` (`synthetic-only`, diagnostic-only)

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | 2.3627 | 1.6670 |

## Synthetic Actuation Diagnostics

| Command Clip | Yaw Saturation | Signed Braking Peak |
|---:|---:|---:|
| 0.1875 | 0.0000 | -2.5000 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 1 | 0.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `timeout_low_progress`: `1`

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | -0.2411 | n/a |
| orca | -0.1844 | -0.0355 | n/a |
| ppo | -0.2482 | -0.0993 | n/a |
