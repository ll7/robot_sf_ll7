# Issue #2475 Probabilistic Prediction Interface

Status: interface contract implemented; no prediction-quality claim, June 6, 2026.

Related issue: [#2475](https://github.com/ll7/robot_sf_ll7/issues/2475)
Parent issue: [#2469](https://github.com/ll7/robot_sf_ll7/issues/2469)

## Summary

Issue #2475 adds a minimal, typed contract for probabilistic pedestrian predictions:
`TrajectoryDistribution`, `ProbabilisticPrediction`, and the runtime-checkable
`ProbabilisticPredictor` protocol in `robot_sf/nav/predictive_types.py`.

The contract gives planners a shared shape for future trajectory distributions and confidence:

```yaml
future:
  trajectory_distribution:
    mean: "(T, 2) float32 future positions"
    std: "(T, 2) float32 optional diagonal uncertainty"
    covariance: "(T, 2, 2) float32 optional full covariance"
  confidence: "scalar in [0, 1]"
```

## Ownership Boundary

- The predictor owns trajectory means, uncertainty fields, confidence, horizon, timestep, timestamp,
  sample count, and metadata.
- Planner adapters consume this contract but do not infer prediction accuracy from its presence.
- SocNav-structured observations are the first input boundary. Raw model `(state, mask)` adapters
  are a follow-up, not part of this slice.
- Existing deterministic predictors may use `confidence=1.0`, `sample_count=1`, and omit uncertainty
  fields to represent deterministic output.

## Fixture And Smoke Contract

`tests/planner/test_probabilistic_prediction_interface.py` defines a dummy predictor and compact
SocNav-style observation fixture. The tests prove the interface is importable, runtime-checkable,
can bundle multiple pedestrian trajectories, and keeps per-pedestrian arrays independent.

This is a schema/interface smoke only. It does not exercise planner quality, predictor calibration,
or benchmark outcomes.

## Claim Boundary

This slice establishes interface readiness. It does not claim:

- probabilistic predictions are accurate or calibrated;
- any planner benefits from consuming the interface;
- heuristic probabilistic search from Issue #591 is paper-grade uncertainty evidence;
- observation or sensor uncertainty is solved beyond the adjacent ScenarioBelief boundary from
  `docs/context/issue_1966_scenario_belief_interface.md`.

Any future planner-quality claim must provide benchmark evidence under the repository fallback and
artifact-provenance policies.

## Validation

Targeted validation for this slice:

```bash
uv run ruff check robot_sf/nav/predictive_types.py robot_sf/nav/__init__.py tests/planner/test_probabilistic_prediction_interface.py
uv run ruff format --check robot_sf/nav/predictive_types.py robot_sf/nav/__init__.py tests/planner/test_probabilistic_prediction_interface.py
uv run pytest tests/planner/test_probabilistic_prediction_interface.py -q
git diff --check
```

## Follow-Up Risks

- Runtime integration remains future work: `PredictionPlannerAdapter` does not yet implement
  `ProbabilisticPredictor`.
- Input-boundary expansion to raw model tensors may be useful, but should not blur the SocNav
  observation ownership contract.
- Sample-based distributions may need a batched representation if future predictors expose
  Monte-Carlo samples directly.
