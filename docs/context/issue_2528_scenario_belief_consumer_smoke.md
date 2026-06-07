# Issue #2528 ScenarioBelief Consumer Smoke (2026-06-07)

Status: diagnostic smoke evidence, not benchmark evidence.

Related surfaces:

- Issue: https://github.com/ll7/robot_sf_ll7/issues/2528
- Parent issue: https://github.com/ll7/robot_sf_ll7/issues/2521
- Predecessor uncertainty contract: `docs/context/issue_2478_uncertainty_scenario_belief.md`
- Implementation: `robot_sf/representation/scenario_belief.py` (`ScenarioBelief.to_uncertainty_report`)
- Tests:
  - `tests/representation/test_scenario_belief.py` (`test_to_uncertainty_report_preserves_covariance_and_class_probabilities`, `test_to_socnav_struct_fails_closed_for_uncertainty_consumption`)
  - `tests/representation/test_scenario_belief_uncertainty_manifest.py` (`test_uncertainty_manifest_consumer_boundary_uncertainty_report_exists`, `test_uncertainty_manifest_keeps_non_benchmark_claim_boundary`)
- Uncertainty manifest: `configs/representation/scenario_belief_uncertainty_issue_2478.yaml`

## Result

Issue #2528 adds a minimal ScenarioBelief consumer smoke for local planner observation projection:

- `ScenarioBelief.to_uncertainty_report()` — a diagnostic method that preserves covariance_xy,
  class_probabilities, position_confidence, velocity_confidence, and existence_probability for
  all visibility-filtered, distance-sorted, capped agents. This is the first projection that
  reads uncertainty fields directly rather than dropping them.
- `to_socnav_struct()` remains the legacy projection, with tests that explicitly prove it drops
  all uncertainty fields (fail-closed for uncertainty consumption).
- The uncertainty manifest records `to_uncertainty_report` as a consumer_boundary entry with
  parent issue 2528.

## Claim Boundary

This is diagnostic smoke evidence only. It proves that a ScenarioBelief can be produced, one
consumer projection (`to_uncertainty_report`) preserves covariance and class-probability fields,
and the legacy projection (`to_socnav_struct`) is fail-closed for uncertainty consumption. It
does not claim planner benefit, benchmark improvement, perception realism, or SNQI movement.

## Validation

```bash
uv run pytest tests/representation/test_scenario_belief.py \
  tests/representation/test_scenario_belief_uncertainty_manifest.py -q
uv run ruff check robot_sf/representation tests/representation/test_scenario_belief.py \
  tests/representation/test_scenario_belief_uncertainty_manifest.py
uv run ruff format --check robot_sf/representation tests/representation/test_scenario_belief.py \
  tests/representation/test_scenario_belief_uncertainty_manifest.py
```

## Follow-Up

The next step would be a planner-facing projection that converts covariance into a planner-compatible
cost or constraint (e.g., covariance-weighted risk field, expanded safety radius, or uncertainty-aware
collision-checking tolerance).
