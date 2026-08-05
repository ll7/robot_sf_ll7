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
  --route-registry-checksum 36f22d682d4df40aa1dbc9f0a548e940a633814e72c0c780bfd1c404ce118bdf \
  --route-start 0 0 \
  --route-end 2 0 \
  --conflict-zone-id fixture-zone \
  --conflict-provenance-id fixture-zone.v1 \
  --conflict-registry-checksum 3b9765014ad30f57a74a8e585e708cccdcadaba03024c3e96cab0ef524f5a097 \
  --conflict-center 1 0 \
  --conflict-radius-m 0.25 \
  --out output/worked_example_process_trace_fixture.json
```

Add `--encounter-report path/to/near_miss_encounter.v1.json` when binding the focal actor
and interval to canonical near-miss encounter output. The report must validate against the
canonical `near_miss_encounter.v1` schema and its provenance `input_checksums` must include
the SHA-256 of the input trace file.

Add `--pair-input path/to/other_trace.json --pair-comparison-grain ...` when building pair
compatibility; the comparison grain is required whenever a pair input is present.

Route and conflict projections are available only when the caller provides world-frame,
provenance- and checksum-bound finite geometry. Missing actor intervals, velocities, radii,
route/zone registry evidence, exact collision telemetry, or source-coordinate support are
reported as unavailable or not observed instead of zero-filled.

Pair compatibility requires an explicit comparison grain:

- `matched_planner_pair`: different planners on the same seed/realization, required
  initial-state equality, and equal map/horizon/config metadata.
- `matched_realization_pair`: the same planner on different seeds/realizations; start/spawn
  may differ and `shared_prefix=false` is allowed, with equal map/horizon/config metadata.

Event alignment selects the first available anchor from the declared fallback order and writes
`anchor_time_s` plus per-frame `tau_s = t - anchor_time_s`. Trace end is a boundary fallback,
not a standalone visual anchor without alignment.
