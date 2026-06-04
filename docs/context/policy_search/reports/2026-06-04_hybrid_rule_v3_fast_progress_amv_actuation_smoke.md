# Candidate Report: hybrid_rule_v3_fast_progress (amv_actuation_smoke)

## Decision

pass

## Hypothesis

Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters.


## Evaluation Scope

- Stage: `amv_actuation_smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/sets/classic_cross_trap_subset.yaml`
- Scenario filter: `classic_cross_trap_high`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/issue2224/hybrid_rule_v3_fast_progress/amv_actuation_smoke/summary.json`
- Git commit: `28e1a502c8bc07ad1510428ee5a58900fd0279af`
- Synthetic actuation profile: `amv-actuation-stress-v0` (`synthetic-only`, diagnostic-only)

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | 2.1571 | 1.6623 |

## Synthetic Actuation Diagnostics

| Command Clip | Yaw Saturation | Signed Braking Peak |
|---:|---:|---:|
| 0.2750 | 0.0000 | -2.5000 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 1 | 0.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `timeout_low_progress`: `1`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | -0.2411 | n/a |
| orca | -0.1844 | -0.0355 | n/a |
| ppo | -0.2482 | -0.0993 | n/a |
