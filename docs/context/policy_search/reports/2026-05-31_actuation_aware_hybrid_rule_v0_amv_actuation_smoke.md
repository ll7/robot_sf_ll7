# Candidate Report: actuation_aware_hybrid_rule_v0 (amv_actuation_smoke)

## Decision

excluded

## Hypothesis

A deterministic hybrid-rule local scorer can penalize command candidates that would be clipped by the synthetic AMV actuation envelope, exposing whether local command smoothing reduces synthetic actuation stress. This is diagnostic-only synthetic evidence, not calibrated hardware or paper-facing AMV performance evidence.


## Evaluation Scope

- Stage: `amv_actuation_smoke`
- Algorithm: `actuation_aware_hybrid_rule_v0`
- Scenario matrix: `configs/scenarios/sets/classic_cross_trap_subset.yaml`
- Scenario filter: `classic_cross_trap_high`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/actuation_aware_hybrid_rule_v0/amv_actuation_smoke/latest/summary.json`
- Git commit: `f81ff02484d7c8dff60fd3295eb73add433194c6`
- Synthetic actuation profile: `amv-actuation-stress-v0` (`synthetic-only`, diagnostic-only)

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 1.0000 | 0.0000 | 1.0513 | 0.0000 |

## Evidence-Adjusted Results

| Episodes | Excluded | Success | Collision | Near Miss |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0.0000 | 0.0000 | 0.0000 |

Raw aggregate results above still include excluded rows; evidence-adjusted results only remove rows with explicit exclusion metadata.

## Synthetic Actuation Diagnostics

| Command Clip | Yaw Saturation | Signed Braking Peak |
|---:|---:|---:|
| 0.0000 | 0.0000 | 0.0000 |

## Scenario Exclusions

| Scenario | Seed | Status | Reason | Evidence |
|---|---:|---|---|---|
| classic_cross_trap_high | 111 | impossible | initial_robot_pedestrian_overlap | first_step_collision_with_zero_progress; min_clearance_m=-0.3487; nearest_pedestrian_distance_m=1.0117; candidate_collision_radius_m=1.4500; all_first_step_candidates_rejected_for_dynamic_collision |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 1 | 0.0000 | 1.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | +0.7589 | n/a |
| orca | -0.1844 | +0.9645 | n/a |
| ppo | -0.2482 | +0.9007 | n/a |
