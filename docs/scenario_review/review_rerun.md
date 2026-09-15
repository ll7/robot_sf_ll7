# SREV-19 review rerun

Record synchronized inspections over saved simulation traces, so
researchers can replay a bounded episode without waiting for the whole
workbench. See the [`docs/glossary.md`](../glossary.md) for project terms
such as review bundle and component result.

## Evidence boundary

Diagnostic tooling only. Recording launches nothing — no simulation,
training, remote scheduling, or scientific publication — and admits no
scientific claim, benchmark result, or planner/simulator behavior change.
The Rerun recording is an optional prototype stream: core inspection (the
offline timeline plus the measured report) always works without the optional
`rerun-sdk` package, and simulation time stays the telemetry authority.

## Interfaces

- **Input**: a `component-request.v1` request for component
  `srev19-review-rerun`. Sources cite trace geometry by identity; the citable
  format is `simulation_trace_export.v1` (validated through
  `robot_sf.analysis_workbench.simulation_trace_export`). Source URIs resolve
  relative to the request file's directory. Other formats are skipped with a
  diagnostic and never block supported sources.
- **Config**: a `recording` mapping with `mode` (`auto`, `json`, or
  `rerun`; default `auto`). `json` writes only the offline timeline.
  `rerun` requires the optional SDK and is unavailable without it. `auto`
  writes the timeline plus the report and adds the recording whenever the
  SDK is installed, otherwise diagnosing the skipped stream.
- **`run(request)`**: returns a `component-result.v1` result with status
  `complete` (artifacts per-trace `inspection-timeline.json`, optional
  per-trace `inspection-recording.rrd`, `prototype-report.json`, plus
  `component-descriptor.json`), `unavailable` (unsupported component,
  capability, evidence, version, or an explicit rerun request without the
  SDK, with an actionable reason), or `failed` (corrupt config, missing
  evidence, or output collision; failed outputs never carry artifacts).
- **Output**: per-trace timelines with step/time-stamped robot and
  pedestrian geometry plus event identity, and a measured prototype report
  (frame/pedestrian-point counts, artifact bytes and digests, and an
  environment-labeled encode time). `descriptor()` exposes the versioned
  capability descriptor.

Unknown required versions fail; output collisions and unsafe paths are
rejected. The report copies the source identities accepted by the v1 request
schema; that schema carries no source-byte hash. Repeated fixture runs
compare equal timeline bytes and report digests.

## Usage

```bash
uv run pytest tests/render/test_review_rerun.py -q
uv run python -m robot_sf.render.review_rerun \
  --input tests/fixtures/scenario_review/review_rerun/request.json \
  --config tests/fixtures/scenario_review/review_rerun/config.json \
  --output output/scenario_review/srev-19-smoke
```

`run(request)` records offline; `descriptor()` lists what this component
provides. Outputs are written atomically into the component-owned output
directory, which must not exist beforehand. Generated output stays under
ignored `output/` and is disposable. No generated videos or recordings are
tracked in Git.
