# Issue #3574 Heterogeneous Population Ablation Report

This report evaluates whether moving from a homogeneous mean-matched population to a heterogeneous mixture population impacts planner safety rankings and metrics.

## Executive Summary
We compared planners across two paired population arms:
1. **Heterogeneous**: A mixture of cautious (25%), standard (50%), and hurried (25%) pedestrians.
2. **Mean-Matched Homogeneous**: A homogeneous population with speed/radius set to the weighted means of the mixture.

We evaluated the safety metric **clearance_m** (distance between robot and nearest pedestrian) using **mean** and **CVaR (alpha=0.2)** (tail safety of the 20% closest encounters).

## Rank-Order Sensitivity Analysis
We ran paired bootstrap resampling (1000 iterations) over seeds to compute the probability of one planner beating another under both arms.

### Rankings (metric: clearance_m, higher is safer)

### Arm: heterogeneous
- **Rank Order**: social_force, goal
- **Observed Means**:
  - goal: 0.8233 m
  - social_force: 1.2103 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats social_force) = 25.20%
  - P(social_force beats goal) = 74.80%

### Arm: mean_matched_homogeneous
- **Rank Order**: social_force, goal
- **Observed Means**:
  - goal: 0.7016 m
  - social_force: 1.0891 m
- **Pairwise Bootstrap Probabilities**:
  - P(goal beats social_force) = 25.20%
  - P(social_force beats goal) = 74.80%

### Rank Reversals
- **No rank reversals detected.** The planner ranking was stable across both arms.

## Detailed Ablation Results
Below is the table of the clearance metrics per seed and arm:

| Scenario | Seed | Planner | Arm | Mean Clearance (m) | CVaR Clearance (m) |
|---|---|---|---|---|---|
| issue_3574_classic_crossing_density_002 | 101 | goal | heterogeneous | 17.4056 | 4.6504 |
| issue_3574_classic_crossing_density_002 | 101 | goal | mean_matched_homogeneous | 16.5518 | 4.1272 |
| issue_3574_classic_crossing_density_002 | 102 | goal | heterogeneous | 12.7987 | 2.5869 |
| issue_3574_classic_crossing_density_002 | 102 | goal | mean_matched_homogeneous | 12.7987 | 2.5814 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | heterogeneous | 13.4339 | 5.6183 |
| issue_3574_classic_crossing_density_002 | 101 | social_force | mean_matched_homogeneous | 12.3913 | 8.2494 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | heterogeneous | 12.7980 | 2.5865 |
| issue_3574_classic_crossing_density_002 | 102 | social_force | mean_matched_homogeneous | 12.7980 | 2.5809 |

## Non-Reactive Mixture Sweeps Caveats
- Since this is a CPU-level smoke validation run on a small slice, rank sensitivity estimates carry higher uncertainty.
- In full runs, a larger sample of seeds and scenarios is required to establish statistical significance.
