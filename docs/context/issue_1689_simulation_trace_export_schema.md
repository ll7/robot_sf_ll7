# Issue #1689 Simulation Trace Export Schema

Date: 2026-05-30

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1689>
- <https://github.com/ll7/robot_sf_ll7/issues/1646>

## Scope

This note records the first analysis-workbench input contract. It defines a tiny
`simulation_trace_export.v1` schema, a typed loader, and a reviewable fixture for future workbench
consumers. It does not add a frontend, renderer, benchmark metric, or paper-facing claim.

## Contract

- Schema: `robot_sf/analysis_workbench/schemas/simulation_trace_export.v1.json`
- Loader: `robot_sf/analysis_workbench/simulation_trace_export.py`
- Fixture: `tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json`
- Test: `tests/analysis_workbench/test_simulation_trace_export.py`

The trace contains source metadata, a strict `analysis_workbench_only` evidence boundary, frame
units, robot state, pedestrian state, and a small planner action/event block. The typed loader also
checks that frame steps and times are strictly increasing so playback consumers can rely on
deterministic ordering.

## Evidence Boundary

`simulation_trace_export.v1` is analysis and visualization input only. A valid trace export is not
benchmark evidence, not a success claim, and not a replacement for schema-checked benchmark episode
records or durable campaign summaries. If a payload changes `evidence_boundary` to
`benchmark_evidence`, validation fails closed.

## Validation

```bash
uv run pytest -q tests/analysis_workbench/test_simulation_trace_export.py
```

Expected result: `3 passed`.
