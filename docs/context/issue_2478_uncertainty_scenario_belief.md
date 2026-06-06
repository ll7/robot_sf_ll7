# Issue #2478 Uncertainty-Aware ScenarioBelief Contract (2026-06-06)

Status: implemented interface contract, not benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2478
- Parent roadmap issue: https://github.com/ll7/robot_sf_ll7/issues/2469
- Predecessor contract: `docs/context/issue_2477_scenario_belief_contract.md`
- Uncertainty manifest: `configs/representation/scenario_belief_uncertainty_issue_2478.yaml`
- Implementation: `robot_sf/representation/scenario_belief.py`
- Tests: `tests/representation/test_scenario_belief_uncertainty_manifest.py`

## Result

Issue #2478 asked for an uncertainty-aware scenario belief interface covering object class, pose,
velocity, and covariance. The existing `ScenarioBelief` implementation already carried
`Estimate2D.mean_xy`, `Estimate2D.covariance_xy`, and `Estimate2D.confidence`; this pass makes the
units and object-class semantics explicit and executable.

The contract now records:

- object class through `EntityBelief.entity_type` plus `class_probabilities`;
- pose estimates in map-frame meters with meter-squared covariance;
- velocity estimates in map-frame meters per second with squared-meter-per-second covariance;
- confidence and existence probability as diagnostic unitless probabilities;
- the consumer boundary where `ScenarioBelief.to_socnav_struct()` drops uncertainty to preserve the
  existing policy-observation key layout.

## Claim Boundary

This is interface and fixture evidence only. It does not prove planner benefit, real-sensor
calibration, perception realism, or benchmark performance. Current adapter covariance is synthetic:
oracle values are near-zero variance and visibility-limited values increase variance when an agent
is not visible.

## Validation

Targeted validation:

```bash
uv run pytest tests/representation/test_scenario_belief.py \
  tests/representation/test_scenario_belief_contract_manifest.py \
  tests/representation/test_scenario_belief_uncertainty_manifest.py -q
uv run ruff check robot_sf/representation tests/representation/test_scenario_belief_uncertainty_manifest.py
uv run ruff format --check robot_sf/representation tests/representation/test_scenario_belief_uncertainty_manifest.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

## Follow-Up Boundary

Heading uncertainty is still a known gap because heading remains a scalar legacy field. The next
useful spike is a planner- or analysis-facing consumer that reads covariance directly; until then,
uncertainty-aware planner improvement should remain unclaimed.
