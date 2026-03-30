# Issue 596 Atomic Scenario Suite Proposal

## Goal

Issue 596 now covers a full-breadth atomic scenario package rather than only a small gate proposal.

The implementation is split into:

- a runnable full suite: `configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml`
- a runnable verified-simple subset: `configs/scenarios/sets/verified_simple_subset_v1.yaml`
- validation-only fixtures: `configs/scenarios/sets/atomic_navigation_validation_fixtures_v1.yaml`
- a compact review matrix: `docs/context/issue_596_atomic_scenario_matrix.md`

The verified-simple subset remains a necessary-but-not-sufficient promotion gate, but it is now a
strict subset of a broader atomic suite that covers the full prompt breadth in one pass.

## Implemented Structure

### Full runnable suite

The full suite covers:

- kinematic and frame consistency:
  - `empty_map_8_directions_*`
  - `goal_behind_robot`
  - `small_angle_precision`
- static obstacle interaction:
  - `single_obstacle_circle`
  - `single_obstacle_rectangle`
  - `line_wall_detour`
  - `narrow_passage`
- topological navigation:
  - `corner_90_turn`
  - `u_trap_local_minimum`
  - `corridor_following`
- minimal dynamic interaction:
  - `single_ped_crossing_orthogonal`
  - `head_on_interaction`
  - `overtaking_interaction`
- robustness:
  - `start_near_obstacle`
  - `goal_very_close`
  - `symmetry_ambiguous_choice`

### Verified-simple subset

The subset keeps the low-ambiguity cases intended for cheap viability screening:

- representative empty-map direction cases,
- `goal_behind_robot`,
- `single_obstacle_circle`,
- `line_wall_detour`,
- `narrow_passage`,
- `single_ped_crossing_orthogonal`,
- `head_on_interaction`,
- `overtaking_interaction`

### Validation-only fixtures

Invalid geometry is kept out of benchmark runs:

- `goal_inside_obstacle_invalid`

This fixture is used to assert fail-closed behavior rather than generate benchmark metrics.

## Config Contract

Each runnable scenario now carries scenario-intent metadata directly in the YAML:

- `purpose`
- `expected_behavior`
- `expected_pass_criteria`
- `failure_modes`
- `primary_capability`
- `target_failure_mode`
- `determinism`

Parameterized families also add:

- `variation_family`
- `variation_value`
- `variation_rationale`

This metadata is treated as part of the contract, not just documentation. The suite tests verify
that the fields are present, that the direction family is represented explicitly, and that the
atomic SVG maps pass the repository verifier.

## Why This Fits Issue 596

This structure gives issue 596 two things at once:

- a broad, interpretable atomic benchmark surface for planner debugging and screening,
- a smaller strict subset that can be used as the first promotion gate before running broader
  benchmark evidence.

It also keeps the invalid-case policy clean:

- benchmark-strength evidence stays runnable,
- invalid geometry stays in validation fixtures,
- failures are easier to interpret and document.

## Next Use

The next benchmark-facing step after this configuration work is calibration:

1. run stable baselines and testing-only planners on `verified_simple_subset_v1`,
2. compare success, collision, runtime, and contradiction signals,
3. set a promotion threshold only after that evidence exists.

## Validation Surfaces

The atomic suite is now intended to be checked in three layers:

1. manifest and metadata contract validation via `tests/test_atomic_navigation_minimal_suite.py`
2. atomic SVG validation via `scripts/validation/verify_maps.py`
3. targeted environment smoke runs for representative static and dynamic scenarios
