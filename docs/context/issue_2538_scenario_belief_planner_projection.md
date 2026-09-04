# Issue #2538 ScenarioBelief Planner Projection Smoke (2026-06-07)

Status: diagnostic interface smoke evidence, not benchmark evidence.

Related surfaces:

- Issue: https://github.com/ll7/robot_sf_ll7/issues/2538
- Predecessors:
  - [issue_2528_scenario_belief_consumer_smoke.md](issue_2528_scenario_belief_consumer_smoke.md)
  - [issue_2565_uncertainty_gating_smoke.md](issue_2565_uncertainty_gating_smoke.md)
- Implementation: `robot_sf/planner/scenario_belief_adapter.py`
- Planner consumer: `robot_sf/planner/stream_gap.py`
- Tests: `tests/planner/test_stream_gap_planner.py`
- Evidence summary:
  [evidence/issue_2538_scenario_belief_planner_projection/summary.json](evidence/issue_2538_scenario_belief_planner_projection/summary.json)

## Result

Issue #2538 adds a planner-facing ScenarioBelief projection helper:

- `project_scenario_belief_for_planner(..., planner_key="stream_gap")` returns the legacy
  `to_socnav_struct()` observation plus a `pedestrians.uncertainty` sidecar copied from
  `ScenarioBelief.to_uncertainty_report()`.
- The projection records a deterministic `uncertainty_compatibility` payload with schema
  `scenario-belief-planner-projection.v1`.
- Unsupported planner keys fail closed: the helper returns the legacy observation without the
  uncertainty sidecar and records `status: fail_closed` with
  `reason: unsupported_uncertainty_planner`.
- The stream-gap planner remains opt-in for uncertainty consumption. Missing or malformed
  uncertainty metadata still keeps deterministic pedestrian rows.

The additive issue #8050 diagnostic seam also provides
`project_belief_aware_planner_input(...)` for the explicitly named
`BeliefGuidedLocalPlanner`. It retains every canonical `ScenarioBelief.agents` entry in an
immutable, ID-keyed `tracks` mapping, including entries absent from the visible legacy rows, and
reports distinct `no_belief`, `empty_belief`, `unsupported_planner`, `invalid_belief`, and
`projected` statuses. Serialization is versioned and deterministic; legacy observations and the
existing stream-gap path are unchanged.

The current `ScenarioBelief` owner does not expose retirement generations. The new diagnostic
therefore reports `identity_generation_available: false`, marks its entity-ID token as not
reuse-safe, and requires stateful consumers to reset at an externally supplied lifecycle boundary.
It does not infer retirement, reuse, or benchmark/safety benefit.

## Claim Boundary

This proves only that ScenarioBelief uncertainty metadata can reach one planner-compatible local
observation shape and can be consumed by the existing stream-gap uncertainty gate on a fixture. It
does not prove better navigation, safety, SNQI, planner performance, perception calibration, or
benchmark movement.

## Validation

```bash
uv run pytest tests/planner/test_stream_gap_planner.py -k "uncertainty or scenario_belief" -q
uv run ruff check robot_sf/planner/scenario_belief_adapter.py tests/planner/test_stream_gap_planner.py
uv run ruff format --check robot_sf/planner/scenario_belief_adapter.py tests/planner/test_stream_gap_planner.py
```

## Follow-Up

The next useful step is a runtime observation-builder path that produces a ScenarioBelief during an
environment step and routes this projection into a planner selection or smoke command. A canonical
track-generation/retirement owner is also required before a stateful planner can claim reuse-safe
identity semantics. Until those gates exist, this remains a unit-level planner interface smoke.
