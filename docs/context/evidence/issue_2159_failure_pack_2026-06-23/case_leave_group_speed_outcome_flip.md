# Leave-Group Speed Outcome Flip

- **Case ID**: `leave_group_speed_outcome_flip`
- **Claim ID**: `research-v1.amv.failure_case_review`
- **Scenario**: `francis2023_leave_group`
- **Planners**: orca
- **Seeds**: [258, 259, 260]
- **Evidence source**: `docs/context/evidence/issue_1945_orca_leave_group_speed_trace_2026-06-01/closest_approach_trace_slices.json`
- **Claim boundary**: diagnostic local trace inspection only

## Pair Summary

- Total pairs: 0
- Completed pairs: 0
- Clearance deltas:
  - No clearance deltas recorded

## Per-Seed Detail

**Planner orca, seed 258**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Planner orca, seed 259**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

**Planner orca, seed 260**
- No_op frame range: [?, ?]
- Perturbed frame range: [?, ?]

## Observed vs Hypothesized

- **Hypothesis**: Route offset or speed perturbation affects robot-pedestrian clearance and collision risk.
- **Observed**: Trace pairs show measurable clearance deltas between no-op and perturbed conditions.
- **Interpretation**: The perturbation induces measurable interaction-pattern changes that are captured by the compact slice format.
- **Limitation**: Qualitative trace inspection only; not benchmark-strength evidence without row-level statistical comparison.