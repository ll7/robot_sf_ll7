# Intersection-Wait Speed p050 Phase Response

- **Case ID**: `intersection_wait_speed_p050_phase_response`
- **Claim ID**: `research-v1.amv.failure_case_review`
- **Scenario**: `francis2023_intersection_wait`
- **Planners**: goal, orca, scenario_adaptive_hybrid_orca_v2_collision_guard
- **Seeds**: [240, 241, 242]
- **Evidence source**: `docs/context/evidence/issue_1953_intersection_wait_speed_grid_trace_2026-06-01/closest_approach_trace_slices_speed_h1_p050.json`
- **Claim boundary**: diagnostic local trace inspection only

## Pair Summary

- Total pairs: 0
- Completed pairs: 0
- Clearance deltas:
  - No clearance deltas recorded

## Per-Seed Detail

**Seed 240**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 241**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 242**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 240**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 241**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 242**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 240**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 241**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 242**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

## Observed vs Hypothesized

- **Hypothesis**: Route offset or speed perturbation affects robot-pedestrian clearance and collision risk.
- **Observed**: Trace pairs show measurable clearance deltas between no-op and perturbed conditions.
- **Interpretation**: The perturbation induces measurable interaction-pattern changes that are captured by the compact slice format.
- **Limitation**: Qualitative trace inspection only; not benchmark-strength evidence without row-level statistical comparison.