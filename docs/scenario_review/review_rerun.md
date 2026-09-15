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
`rerun-sdk` package, and simulation time stays the telemetry authority. Every
published timeline wraps and validates the canonical `simulation_timeline.v1`
projection, carries the complete validated trace identity, and is marked
`evidence_boundary: analysis_workbench_only`, `diagnostic_only: true`, and
`admission: not_evaluated`.

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
- **Source identities**: every source `artifact_id` must be a single relative
  filename component and unique within the request. Traversal, absolute, or
  duplicate IDs fail before output staging. A CLI `--config` document must be
  a JSON object; non-object config is rejected rather than ignored.
- **`run(request)`**: returns a `component-result.v1` result with status
  `complete` (artifacts per-trace `inspection-timeline.json`, optional
  per-trace `inspection-recording.rrd`, `prototype-report.json`, plus
  `component-descriptor.json`), `unavailable` (unsupported component,
  capability, evidence, version, or an explicit rerun request without the
  SDK, with an actionable reason), or `failed` (corrupt config, missing
  evidence, or output collision; failed outputs never carry artifacts).
- **Output**: per-trace timelines with step/time-stamped robot and
  pedestrian geometry plus event identity, source trace metadata, source-byte
  and configuration digests, and the canonical timeline wrapper. The measured
  prototype report includes the same diagnostic boundary and provenance
  fields, frame/pedestrian-point counts, artifact bytes and digests, and an
  environment-labeled encode time. `descriptor()` exposes the versioned
  capability descriptor.

Unknown required versions fail; duplicate or unsafe artifact IDs, output
collisions, source digest mismatches, and unsafe paths are rejected. Every
published artifact URI is checked against the requested output directory and
its final bytes are re-hashed before a complete result is returned. Rerun
geometry is keyed by source actor identity and explicitly clears actors that
disappear from a frame. Per-frame metadata includes actor state and the
pedestrian count/IDs, including an explicit zero-pedestrian frame. An
installed SDK that fails during initialization, logging, or saving fails the
requested recording without publishing a partial directory. Repeated fixture
runs compare equal timeline bytes and report digests.

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
