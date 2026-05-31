# Candidate Report: adaptive_proxemic_selector_v0 (smoke)

## Decision

pass

## Hypothesis

A deterministic local-context selector can switch among the existing conservative, neutral, and open proxemic profile parameterizations to expose when each profile would be chosen. This row is diagnostic only: progress gains are not success evidence unless near-miss and comfort exposure tradeoffs are reviewed separately.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `adaptive_proxemic_selector_v0`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/adaptive_proxemic_selector_v0/smoke/issue1805_local/summary.json`
- Git commit: `c29e2bbb7d4bf014eb74405b4afdf6be1ad3338b`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8941 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Claim Boundary

This smoke run only proves that the diagnostic selector is wired into the policy-search runner and
can complete the planner sanity scenario. It is not benchmark-strengthening evidence for proxemic
comfort, near-miss behavior, or general navigation quality; broader scenario-family evidence is
still required before making performance claims.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
