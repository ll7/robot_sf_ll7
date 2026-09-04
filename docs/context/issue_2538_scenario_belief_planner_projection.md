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
`project_belief_aware_planner_input(...)` for the currently supported projection target
`BeliefGuidedLocalPlanner`. It retains every `ScenarioBelief.agents` entry in an immutable,
entity-ID-keyed `tracks` mapping, including entries absent from the visible legacy rows, and
reports distinct `no_belief`, `empty_belief`, `projection_target_not_supported`, `invalid_belief`,
and `projected` statuses. Serialization is versioned and deterministic; legacy observations and
the existing stream-gap path are unchanged.

`track_id` is the entity identifier supplied by one `ScenarioBelief` snapshot, not a
visible-observation row number. The current representation does not expose retirement generations,
so the diagnostic reports `identity_generation_available: false`,
`identity_reuse_safe: false`, `retired_track_count: null`, and
`stateful_identity_admitted: false`. Stateful consumers must reset at an externally supplied
lifecycle boundary. No generation or continuity token is fabricated, and no retirement, reuse, or
benchmark/safety benefit is inferred. Aggregate confidence and the radius covariance block are
explicitly labelled as adapter-derived and unavailable-as-modelled in the diagnostics.

## Claim Boundary

Safe claim: this is a deterministic, entity-ID-keyed projection of one `ScenarioBelief` snapshot
that prevents joining uncertainty metadata by visible-row position. It proves only that
ScenarioBelief uncertainty metadata can reach one planner-compatible local observation shape and
can be consumed by the existing stream-gap uncertainty gate on a fixture. It does not prove
cross-lifecycle identity continuity, better navigation, safety, SNQI, planner performance,
perception calibration, or benchmark movement.

## Validation

```bash
uv run pytest tests/planner/test_scenario_belief_track_projection.py \
  tests/planner/test_scenario_belief_uncertainty_gate.py \
  tests/planner/test_stream_gap_planner.py \
  tests/representation/test_scenario_belief.py -q
uv run ruff check robot_sf/planner/scenario_belief_adapter.py \
  tests/planner/test_scenario_belief_track_projection.py
uv run ruff format --check robot_sf/planner/scenario_belief_adapter.py \
  tests/planner/test_scenario_belief_track_projection.py
```

## Follow-Up

The next useful step is a runtime observation-builder path that produces a ScenarioBelief during an
environment step and routes this projection into a planner selection or smoke command. A
track-generation/retirement owner is required before a stateful planner can claim reuse-safe
identity semantics. Until those gates exist, this remains a unit-level planner interface smoke;
#8050 stays open with `implementation_admitted: false`, and no downstream planner work is part of
this slice.
