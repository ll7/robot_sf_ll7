# Issue #2565 Uncertainty-Aware Stream-Gap Gating Smoke (2026-06-07)

Status: diagnostic smoke evidence, not benchmark evidence.

Related surfaces:

- Issue: https://github.com/ll7/robot_sf_ll7/issues/2565
- Predecessor: [issue_2528_scenario_belief_consumer_smoke.md](issue_2528_scenario_belief_consumer_smoke.md)
- Implementation: `robot_sf/planner/stream_gap.py`
- Tests: `tests/planner/test_stream_gap_planner.py`
- Evidence summary: [evidence/issue_2565_uncertainty_gating_smoke/summary.json](evidence/issue_2565_uncertainty_gating_smoke/summary.json)

## Result

Issue #2565 adds an opt-in `stream_gap` planner-input gate that consumes a
ScenarioBelief-style uncertainty sidecar under `observation["pedestrians"]["uncertainty"]`.
The gate uses existence probability, position confidence, pedestrian class probability, and
average position covariance variance to decide whether a pedestrian row should remain in the
deterministic stream-gap corridor check.

Default behavior is unchanged because `uncertainty_gating_enabled` defaults to `False`.
When enabled, missing, malformed, or length-mismatched uncertainty metadata fails closed by keeping
all deterministic pedestrian rows. The diagnostic `last_uncertainty_gate` payload records whether
the gate was disabled, empty, applied, or failed closed.

## Claim Boundary

This is planning-relevance smoke evidence only. It proves that uncertainty metadata derived from
the ScenarioBelief uncertainty-report shape can change one local-planner input decision on a single
fixture. It does not prove benchmark improvement, safety improvement, perception realism,
calibration quality, or paper-facing performance.

## Validation

```bash
uv run pytest tests/planner/test_stream_gap_planner.py -k uncertainty -q
uv run pytest tests/planner/test_stream_gap_planner.py -q
uv run ruff check robot_sf/planner/stream_gap.py tests/planner/test_stream_gap_planner.py
uv run ruff format --check robot_sf/planner/stream_gap.py tests/planner/test_stream_gap_planner.py
```

## Follow-Up

The next step is an end-to-end bridge from a runtime ScenarioBelief producer or observation builder
into the optional sidecar. Until that exists, this remains a planner-unit smoke rather than a full
runtime perception-to-planner proof.
