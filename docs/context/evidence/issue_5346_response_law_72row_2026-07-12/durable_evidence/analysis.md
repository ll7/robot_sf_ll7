<!-- AI-GENERATED (robot_sf#5378, 2026-07-12) - NEEDS-REVIEW -->
# Issue #3574 Heterogeneous Population Ablation Report

This report compiles supplied paired episode records into the issue #3574 integration schema.

## Claim Boundary

- Evidence status: `diagnostic-only` until a separately reviewed campaign provides attributable
  paired records and appropriate confidence bounds.
- This report does not run a benchmark and does not establish a heterogeneous-population effect,
  planner rank-stability result, realism claim, or sim-to-real claim.
- Any ranking text below describes only the supplied records; it is not an empirical conclusion by
  itself.

## Executive Summary
We compared planners across two paired population arms:
1. **Heterogeneous**: A mixture of cautious (25%), standard (50%), and hurried (25%) pedestrians.
2. **Mean-Matched Homogeneous**: A homogeneous population with speed/radius set to the weighted means of the mixture.

We evaluated the safety metric **clearance_m** (distance between robot and nearest pedestrian) using **mean** and **CVaR (alpha=0.2)** (tail safety of the 20% closest encounters).

## Per-Archetype Trace Metrics

The JSON summary contains per-archetype reports for every metric required by the paired manifest:
`clearance_m, near_field_exposure_s`. These reports preserve both arms for
each scenario/seed/planner triplet and are only rendered after the fail-closed integration-readiness
check passes.

## Rank-Order Sensitivity Analysis
We ran paired bootstrap resampling (1000 iterations) over seeds to compute the probability of one planner beating another under both arms.

### Rankings (metric: clearance_m, higher is safer)

### Arm: heterogeneous/response_law_fraction_0
- **Rank Order**: social_force, goal, orca
- **Observed Means**:
  - goal: 0.9105 m
  - orca: 0.8836 m
  - social_force: 1.0953 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats orca) = 62.90%
  - P(goal beats social_force) = 30.40%
  - P(orca beats goal) = 37.10%
  - P(orca beats social_force) = 30.40%
  - P(social_force beats goal) = 69.60%
  - P(social_force beats orca) = 69.60%

### Arm: heterogeneous/response_law_fraction_0.1
- **Rank Order**: social_force, goal, orca
- **Observed Means**:
  - goal: 0.9065 m
  - orca: 0.8833 m
  - social_force: 1.1042 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats orca) = 62.90%
  - P(goal beats social_force) = 30.40%
  - P(orca beats goal) = 37.10%
  - P(orca beats social_force) = 30.40%
  - P(social_force beats goal) = 69.60%
  - P(social_force beats orca) = 69.60%

### Arm: heterogeneous/response_law_fraction_0.25
- **Rank Order**: orca, social_force, goal
- **Observed Means**:
  - goal: 1.0121 m
  - orca: 1.0640 m
  - social_force: 1.0352 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats orca) = 29.90%
  - P(goal beats social_force) = 42.00%
  - P(orca beats goal) = 70.10%
  - P(orca beats social_force) = 64.00%
  - P(social_force beats goal) = 58.00%
  - P(social_force beats orca) = 36.00%

### Arm: heterogeneous/response_law_fraction_0.5
- **Rank Order**: social_force, orca, goal
- **Observed Means**:
  - goal: 0.8947 m
  - orca: 0.9408 m
  - social_force: 1.0085 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats orca) = 29.90%
  - P(goal beats social_force) = 4.90%
  - P(orca beats goal) = 70.10%
  - P(orca beats social_force) = 30.40%
  - P(social_force beats goal) = 95.10%
  - P(social_force beats orca) = 69.60%

### Arm: mean_matched_homogeneous/response_law_fraction_0
- **Rank Order**: social_force, orca, goal
- **Observed Means**:
  - goal: 0.7216 m
  - orca: 0.8257 m
  - social_force: 1.1179 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats orca) = 4.90%
  - P(goal beats social_force) = 4.90%
  - P(orca beats goal) = 95.10%
  - P(orca beats social_force) = 4.90%
  - P(social_force beats goal) = 95.10%
  - P(social_force beats orca) = 95.10%

### Arm: mean_matched_homogeneous/response_law_fraction_0.1
- **Rank Order**: social_force, orca, goal
- **Observed Means**:
  - goal: 0.7206 m
  - orca: 0.8255 m
  - social_force: 1.1201 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats orca) = 4.90%
  - P(goal beats social_force) = 4.90%
  - P(orca beats goal) = 95.10%
  - P(orca beats social_force) = 4.90%
  - P(social_force beats goal) = 95.10%
  - P(social_force beats orca) = 95.10%

### Arm: mean_matched_homogeneous/response_law_fraction_0.25
- **Rank Order**: social_force, orca, goal
- **Observed Means**:
  - goal: 0.7870 m
  - orca: 0.8861 m
  - social_force: 1.0296 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats orca) = 4.90%
  - P(goal beats social_force) = 4.90%
  - P(orca beats goal) = 95.10%
  - P(orca beats social_force) = 4.90%
  - P(social_force beats goal) = 95.10%
  - P(social_force beats orca) = 95.10%

### Arm: mean_matched_homogeneous/response_law_fraction_0.5
- **Rank Order**: social_force, orca, goal
- **Observed Means**:
  - goal: 0.7383 m
  - orca: 0.8468 m
  - social_force: 1.0091 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats orca) = 4.90%
  - P(goal beats social_force) = 4.90%
  - P(orca beats goal) = 95.10%
  - P(orca beats social_force) = 4.90%
  - P(social_force beats goal) = 95.10%
  - P(social_force beats orca) = 95.10%

### Rank Reversals
- **WARNING**: Reversal detected! Heterogeneous ranking ['social_force', 'goal', 'orca'] differs from homogeneous ['social_force', 'orca', 'goal']
- **WARNING**: Reversal detected! Heterogeneous ranking ['social_force', 'goal', 'orca'] differs from homogeneous ['social_force', 'orca', 'goal']
- **WARNING**: Reversal detected! Heterogeneous ranking ['orca', 'social_force', 'goal'] differs from homogeneous ['social_force', 'orca', 'goal']

## Detailed Ablation Results
Below is the table of the clearance metrics per seed and arm:

| Scenario | Seed | Planner | Arm | Mean Clearance (m) | CVaR Clearance (m) |
|---|---|---|---|---|---|
| issue_3574_classic_crossing_density_002 | 101 | goal | heterogeneous | 17.4056 | 4.6504 |
| issue_3574_classic_crossing_density_002 | 101 | goal | mean_matched_homogeneous | 16.5518 | 4.1272 |
| issue_3574_classic_crossing_density_002 | 102 | goal | heterogeneous | 12.7987 | 2.5869 |
| issue_3574_classic_crossing_density_002 | 102 | goal | mean_matched_homogeneous | 12.7987 | 2.5814 |
| issue_3574_classic_crossing_density_002 | 103 | goal | heterogeneous | 12.4586 | 1.8018 |
| issue_3574_classic_crossing_density_002 | 103 | goal | mean_matched_homogeneous | 11.2162 | 2.2426 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | heterogeneous | 13.4341 | 5.6200 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | mean_matched_homogeneous | 12.3858 | 8.2404 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | heterogeneous | 12.7980 | 2.5865 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | mean_matched_homogeneous | 12.7980 | 2.5809 |
| issue_3574_classic_crossing_density_002 | 103 | social_force | heterogeneous | 11.6950 | 2.0991 |
| issue_3574_classic_crossing_density_002 | 103 | social_force | mean_matched_homogeneous | 11.6613 | 2.3263 |
| issue_3574_classic_crossing_density_002 | 101 | orca | heterogeneous | 16.4180 | 6.4812 |
| issue_3574_classic_crossing_density_002 | 101 | orca | mean_matched_homogeneous | 15.5428 | 7.5605 |
| issue_3574_classic_crossing_density_002 | 102 | orca | heterogeneous | 12.7981 | 2.5865 |
| issue_3574_classic_crossing_density_002 | 102 | orca | mean_matched_homogeneous | 12.7980 | 2.5810 |
| issue_3574_classic_crossing_density_002 | 103 | orca | heterogeneous | 11.0159 | 4.3400 |
| issue_3574_classic_crossing_density_002 | 103 | orca | mean_matched_homogeneous | 10.4197 | 4.8108 |
| issue_3574_classic_crossing_density_002 | 101 | goal | heterogeneous | 17.4056 | 4.6504 |
| issue_3574_classic_crossing_density_002 | 101 | goal | mean_matched_homogeneous | 16.5518 | 4.1272 |
| issue_3574_classic_crossing_density_002 | 102 | goal | heterogeneous | 12.7987 | 2.5869 |
| issue_3574_classic_crossing_density_002 | 102 | goal | mean_matched_homogeneous | 12.7987 | 2.5814 |
| issue_3574_classic_crossing_density_002 | 103 | goal | heterogeneous | 12.4572 | 1.7965 |
| issue_3574_classic_crossing_density_002 | 103 | goal | mean_matched_homogeneous | 11.2175 | 2.2454 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | heterogeneous | 13.5069 | 5.5948 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | mean_matched_homogeneous | 12.3865 | 8.2442 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | heterogeneous | 12.7980 | 2.5865 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | mean_matched_homogeneous | 12.7980 | 2.5809 |
| issue_3574_classic_crossing_density_002 | 103 | social_force | heterogeneous | 11.7011 | 2.1018 |
| issue_3574_classic_crossing_density_002 | 103 | social_force | mean_matched_homogeneous | 11.6662 | 2.3310 |
| issue_3574_classic_crossing_density_002 | 101 | orca | heterogeneous | 16.4180 | 6.4812 |
| issue_3574_classic_crossing_density_002 | 101 | orca | mean_matched_homogeneous | 15.5428 | 7.5605 |
| issue_3574_classic_crossing_density_002 | 102 | orca | heterogeneous | 12.7981 | 2.5865 |
| issue_3574_classic_crossing_density_002 | 102 | orca | mean_matched_homogeneous | 12.7980 | 2.5810 |
| issue_3574_classic_crossing_density_002 | 103 | orca | heterogeneous | 11.0188 | 4.3400 |
| issue_3574_classic_crossing_density_002 | 103 | orca | mean_matched_homogeneous | 10.4220 | 4.8108 |
| issue_3574_classic_crossing_density_002 | 101 | goal | heterogeneous | 19.2069 | 4.8908 |
| issue_3574_classic_crossing_density_002 | 101 | goal | mean_matched_homogeneous | 18.2303 | 4.4345 |
| issue_3574_classic_crossing_density_002 | 102 | goal | heterogeneous | 12.7987 | 2.5869 |
| issue_3574_classic_crossing_density_002 | 102 | goal | mean_matched_homogeneous | 12.7987 | 2.5814 |
| issue_3574_classic_crossing_density_002 | 103 | goal | heterogeneous | 12.4572 | 1.7965 |
| issue_3574_classic_crossing_density_002 | 103 | goal | mean_matched_homogeneous | 11.2175 | 2.2454 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | heterogeneous | 13.4154 | 5.3467 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | mean_matched_homogeneous | 13.0989 | 4.2746 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | heterogeneous | 12.7980 | 2.5865 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | mean_matched_homogeneous | 12.7980 | 2.5809 |
| issue_3574_classic_crossing_density_002 | 103 | social_force | heterogeneous | 11.7011 | 2.1018 |
| issue_3574_classic_crossing_density_002 | 103 | social_force | mean_matched_homogeneous | 11.6662 | 2.3310 |
| issue_3574_classic_crossing_density_002 | 101 | orca | heterogeneous | 22.0755 | 5.1467 |
| issue_3574_classic_crossing_density_002 | 101 | orca | mean_matched_homogeneous | 21.2077 | 4.9317 |
| issue_3574_classic_crossing_density_002 | 102 | orca | heterogeneous | 12.7981 | 2.5865 |
| issue_3574_classic_crossing_density_002 | 102 | orca | mean_matched_homogeneous | 12.7980 | 2.5810 |
| issue_3574_classic_crossing_density_002 | 103 | orca | heterogeneous | 11.0063 | 4.3138 |
| issue_3574_classic_crossing_density_002 | 103 | orca | mean_matched_homogeneous | 10.4180 | 4.7827 |
| issue_3574_classic_crossing_density_002 | 101 | goal | heterogeneous | 19.2069 | 4.8908 |
| issue_3574_classic_crossing_density_002 | 101 | goal | mean_matched_homogeneous | 18.2303 | 4.4345 |
| issue_3574_classic_crossing_density_002 | 102 | goal | heterogeneous | 12.7983 | 2.5855 |
| issue_3574_classic_crossing_density_002 | 102 | goal | mean_matched_homogeneous | 12.7983 | 2.5799 |
| issue_3574_classic_crossing_density_002 | 103 | goal | heterogeneous | 12.4040 | 1.5837 |
| issue_3574_classic_crossing_density_002 | 103 | goal | mean_matched_homogeneous | 11.1789 | 2.2252 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | heterogeneous | 13.4153 | 5.3467 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | mean_matched_homogeneous | 13.1091 | 4.2708 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | heterogeneous | 12.7977 | 2.5851 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | mean_matched_homogeneous | 12.7977 | 2.5795 |
| issue_3574_classic_crossing_density_002 | 103 | social_force | heterogeneous | 11.6412 | 1.9994 |
| issue_3574_classic_crossing_density_002 | 103 | social_force | mean_matched_homogeneous | 11.6100 | 2.2351 |
| issue_3574_classic_crossing_density_002 | 101 | orca | heterogeneous | 22.0755 | 5.1467 |
| issue_3574_classic_crossing_density_002 | 101 | orca | mean_matched_homogeneous | 21.2077 | 4.9317 |
| issue_3574_classic_crossing_density_002 | 102 | orca | heterogeneous | 12.7977 | 2.5851 |
| issue_3574_classic_crossing_density_002 | 102 | orca | mean_matched_homogeneous | 12.7977 | 2.5795 |
| issue_3574_classic_crossing_density_002 | 103 | orca | heterogeneous | 11.2424 | 3.7013 |
| issue_3574_classic_crossing_density_002 | 103 | orca | mean_matched_homogeneous | 10.3853 | 4.7774 |

## Non-Reactive Mixture Sweeps Caveats
- Since this is a CPU-level smoke validation run on a small slice, rank sensitivity estimates carry higher uncertainty.
- In full runs, a larger sample of seeds and scenarios is required to establish statistical significance.
