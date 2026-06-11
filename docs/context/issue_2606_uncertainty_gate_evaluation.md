# Issue #2606 Uncertainty Gate Evaluation (2026-06-11)

Status: diagnostic only, not benchmark evidence.

Related surfaces:

- Issue: https://github.com/ll7/robot_sf_ll7/issues/2606
- Predecessors:
  - [issue_2528_scenario_belief_consumer_smoke.md](issue_2528_scenario_belief_consumer_smoke.md)
  - [issue_2565_uncertainty_gating_smoke.md](issue_2565_uncertainty_gating_smoke.md)
  - [issue_2538_scenario_belief_planner_projection.md](issue_2538_scenario_belief_planner_projection.md)
- Report source: `robot_sf/representation/scenario_belief.py` (`ScenarioBelief.to_uncertainty_report`)
- Adapter: `robot_sf/planner/scenario_belief_adapter.py` (`project_scenario_belief_for_planner`)
- Planner consumer: `robot_sf/planner/stream_gap.py` (`StreamGapPlannerAdapter._uncertainty_keep_mask`)
- Tests: `tests/planner/test_scenario_belief_uncertainty_gate.py`
- Evidence summary: [evidence/issue_2606_uncertainty_gate/summary.json](evidence/issue_2606_uncertainty_gate/summary.json)

## Question

Can `ScenarioBelief.to_uncertainty_report()` feed one planner input/projection without
silently dropping covariance, confidence, class_probabilities, or existence metadata?

## Result

Yes. Evidence shows the full chain preserves all four target fields:

1. **to_uncertainty_report()** produces agent rows with all 8 metadata fields:
   `entity_id`, `class_probabilities`, `position_covariance_xy`, `velocity_covariance_xy`,
   `position_confidence`, `velocity_confidence`, `existence_probability`, `visibility_state`.

2. **project_scenario_belief_for_planner()** copies all 8 fields into the sidecar
   without transformation. The sidecar matches the report exactly.

3. **stream_gap planner** consumes 4 of 8 fields for gating decisions:
   `existence_probability`, `position_confidence`, `class_probabilities.pedestrian`,
   `position_covariance_xy` (as variance via trace/2).

4. **3 fields are preserved but not consumed** by the stream_gap gate:
   `velocity_covariance_xy`, `velocity_confidence`, `visibility_state`. These are
   available for future planner extensions.

5. The gate can change the planner's commit/wait decision based on uncertainty:
   a low-confidence blocker is dropped (v > 0) while a high-confidence blocker is
   kept (v == 0).

## Decision: usable

"Usable" means interface-usable: the fields reach the planner input and affect the
opt-in gate. It does NOT mean benchmark-proven, safety-improved, or paper-facing.

## Claim Boundary

This is diagnostic-only evidence. It proves field preservation through the full
ScenarioBelief-to-planner pipeline and that the stream_gap uncertainty gate reads
and acts on the four consumed fields. It does not prove benchmark improvement,
safety improvement, planner performance, perception calibration, or paper-facing
result.

## Validation

```bash
uv run pytest tests/planner/test_scenario_belief_uncertainty_gate.py -v
uv run ruff check tests/planner/test_scenario_belief_uncertainty_gate.py
uv run ruff format --check tests/planner/test_scenario_belief_uncertainty_gate.py
```

## Follow-Up

- If velocity covariance should affect planning, extend `_uncertainty_row_metrics`
  to consume `velocity_covariance_xy` and `velocity_confidence`.
- Connect a runtime ScenarioBelief producer to the planner projection path.
- Calibrate uncertainty thresholds for non-diagnostic use.
