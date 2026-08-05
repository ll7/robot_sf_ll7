# Issue #6790 Worked-Example Process Trace

`worked_example_process_trace.v1` builds deterministic, renderer-neutral process diagnostics from
admitted `simulation_trace_export.v1` traces. It is analysis-workbench evidence only: it does not
change canonical benchmark metrics, SNQI, planner behavior, or figure admission.

## Public Surfaces

- Schema: `robot_sf/analysis_workbench/schemas/worked_example_process_trace.v1.json`
- Builder and coordinate orchestration: `robot_sf/analysis_workbench/interaction_coordinates.py`
- Pair compatibility: `robot_sf/analysis_workbench/event_alignment.py`
- Stall/reversal phase summaries: `robot_sf/analysis_workbench/episode_phases.py`
- Proxy clearance and TCPA/CPA diagnostics: `robot_sf/analysis_workbench/safety_surrogates.py`
- CLI: `scripts/analysis/build_worked_example_process_trace.py`
- Tests: `tests/analysis_workbench/test_worked_example_process_trace.py`

## Reproduction Command

```bash
uv run python scripts/analysis/build_worked_example_process_trace.py \
  --input tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json \
  --route-id fixture-route \
  --route-provenance-id fixture-route.v1 \
  --route-start 0 0 \
  --route-end 2 0 \
  --conflict-zone-id fixture-zone \
  --conflict-provenance-id fixture-zone.v1 \
  --conflict-center 1 0 \
  --conflict-radius-m 0.25 \
  --out output/worked_example_process_trace_fixture.json
```

Route and conflict projections are available only when the caller provides provenance-bound geometry.
Missing actor bindings, velocities, radii, route provenance, zone provenance, or source-coordinate
support are reported as unavailable or not observed instead of zero-filled.
