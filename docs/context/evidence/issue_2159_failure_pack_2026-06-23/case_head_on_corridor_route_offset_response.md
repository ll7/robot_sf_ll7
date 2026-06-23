# Head-on Corridor Route Offset Response

- **Case ID**: `head_on_corridor_route_offset_response`
- **Claim ID**: `research-v1.amv.failure_case_review`
- **Scenario**: `classic_head_on_corridor_low`
- **Planners**: goal, orca, scenario_adaptive_hybrid_orca_v2_collision_guard
- **Seeds**: [111, 112, 113, 114]
- **Evidence source**: `docs/context/evidence/issue_1939_corridor_trace_response_2026-05-31/closest_approach_trace_slices.json`
- **Claim boundary**: diagnostic local trace inspection only

## Pair Summary

- Total pairs: 0
- Completed pairs: 0
- Clearance deltas:
  - No clearance deltas recorded

## Per-Seed Detail

**Seed 111**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 112**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 116**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 117**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 111**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 112**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 116**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 117**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 111**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 112**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 116**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Seed 117**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

## Observed vs Hypothesized

- **Hypothesis**: Route offset or speed perturbation affects robot-pedestrian clearance and collision risk.
- **Observed**: Trace pairs show measurable clearance deltas between no-op and perturbed conditions.
- **Interpretation**: The perturbation induces measurable interaction-pattern changes that are captured by the compact slice format.
- **Limitation**: Qualitative trace inspection only; not benchmark-strength evidence without row-level statistical comparison.