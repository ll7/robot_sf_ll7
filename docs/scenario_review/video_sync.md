# Video sync (SREV-03)

Command-line glossary: SREV is the scenario-review portfolio; a media mapping
(`media-mapping.v1`) is the explicit frame-index to simulation-time map;
a component request (`component-request.v1`) is the fixture envelope that
invokes one workbench component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.render.video_sync` maps capture frames to simulation time
without advancing or mutating simulation state and without guessing alignment.
Inputs are an explicit capture-frames document (frame indexes with presentation
timestamps) and a sim-stamps document (step, time, episode/reset ids). Skipped
render steps, nonuniform sampling, and reset boundaries are recorded explicitly;
out-of-range frames resolve to `partial`, never to invented stamps.

## Command

```bash
uv run python -m robot_sf.render.video_sync \
  --input tests/fixtures/scenario_review/video_sync/request.json \
  --config tests/fixtures/scenario_review/video_sync/config.json \
  --output output/scenario_review/srev-03-smoke
```

The checked-in fixture deliberately exercises a skipped frame and nonuniform
sampling, so its expected verdict is `partial` with the `skipped_frames:3` and
`nonuniform_sampling` codes — asserting those proves the detectors work.
`--output` must not exist. Exit code is 0 only on `complete`.

Use `--descriptor` to print the shared `component-descriptor.v1` document
without reading inputs or creating an output directory:

```bash
uv run python -m robot_sf.render.video_sync --descriptor
```

## Output

- `media-mapping.json`: entries with frame index, video PTS, sim time/step,
  episode/reset ids, and camera identity, plus skipped-frame and explicit
  unavailable-frame lists, ordered episode/reset boundaries, and first/last
  captured records. An unavailable record has no simulation step or episode/
  reset anchor; in particular, frames after the final simulation stamp are not
  assigned the last stale stamp.
- `missing-capability-report.json`: skipped optional streams and diagnostics.
- The printed result is the shared `component-result.v1` envelope and carries
  `complete`, `partial`, `unavailable`, or `failed` with stable reason codes.
  Only `complete` results list envelope artifacts; mapping files stay on disk
  either way.

Presentation settings default to 1920x1080 at 30 fps and original speed (the
`test` preset uses 320x180 at 10 fps). A supplied presentation object must be
an object with positive integer dimensions and finite positive numeric `fps`
and `speed`; invalid, non-finite, or wrong-type values fail closed. Simulation
stamps are consumed in source order. Repeated or regressing times are retained
as boundary diagnostics and make mapping unavailable rather than selecting an
ambiguous episode/reset anchor.

Optional source identity can be supplied in request config under
`source_metadata`, keyed by source artifact ID. The mapping and result retain
the selected URI, format, declared schema, source commit, config identity,
computed source SHA-256, declared-vs-computed integrity status, and
`admission: not_evaluated`. Integrity is diagnostic metadata only and does not
admit scientific evidence.

## Unavailable reasons and limits

- Missing families, corrupt sources, malformed rows, invalid presentation
  values, ambiguous timelines, and out-of-range frames make the result
  `partial` or `failed`, never silently complete. Malformed or non-strict JSON
  input is reported as a contract-valid failed result by the CLI.
- No simulator, planner, or training imports: the module is observational by
  construction (asserted in tests). Capture hooks in `sim_view.py` /
  `jsonl_recording.py` were not needed and not touched.
- Evidence boundary: the map proves explicit correspondence between two provided
  timestamp series, not that either series is scientifically admissible. No
  benchmark, safety, or paper-facing claim follows from a mapping.
