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

## Output

- `media-mapping.json`: entries with frame index, video PTS, sim time/step,
  episode/reset ids, and camera identity, plus skipped-frame list, reset list,
  and first/last frames.
- `missing-capability-report.json`: skipped optional streams and diagnostics.
- The printed result envelope carries `complete`, `partial`, `unavailable`, or
  `failed` with stable reason codes. Only `complete` results list envelope
  artifacts; mapping files stay on disk either way.

## Unavailable reasons and limits

- Missing families, corrupt sources, malformed rows, and out-of-range frames make
  the result `partial` or `failed`, never silently complete.
- No simulator, planner, or training imports: the module is observational by
  construction (asserted in tests). Capture hooks in `sim_view.py` /
  `jsonl_recording.py` were not needed and not touched.
- Evidence boundary: the map proves explicit correspondence between two provided
  timestamp series, not that either series is scientifically admissible. No
  benchmark, safety, or paper-facing claim follows from a mapping.
