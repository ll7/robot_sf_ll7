# SREV-16 synchronized review panels

SREV-16 builds the offline, read-only scene/video/metric/event panel model used
by a scenario-review session. It is a renderer-neutral extension of the SREV-15
workbench: every panel consumes one shared simulation-time cursor and an
episode-scoped selection context. It does not write annotations, start a
simulation, call an AI provider, or fetch network assets.

## Contract and command

The Python entry point is:

```python
from robot_sf.render.review_panels import run

result = run(component_request)
```

`build_panel_model(request, base=...)` exposes the same renderer-neutral model
without reserving or writing an output directory. It still performs local media
trust and existence admission, so a missing or unsafe mapped media source is
diagnostic/partial in both APIs.

`component_request` is the shared `component-request.v1` envelope. The
component id is `srev16-review-panels`; `descriptor()` and
`descriptor_document()` expose its `component-descriptor.v1` contract. The
standalone CLI is:

```bash
uv run python -m robot_sf.render.review_panels \
  --input tests/fixtures/scenario_review/review_panels/request.json \
  --config tests/fixtures/scenario_review/review_panels/config.json \
  --output output/scenario_review/srev-16-smoke
```

The output directory must be new. A complete invocation writes
`review-panels.v1.json`, `review-panels.v1.html`, the local
`components/review_panels/review_panels.js` module, and a
`missing-capability-report.json`. Partial output remains diagnostic-only and
is listed in `provenance.emitted_artifacts`; the shared result envelope lists
artifacts only for a complete result. Generated output belongs under ignored
`output/`, not Git.

## Source-time rules

Simulation time is authoritative. Scene, metric, and event rows must carry an
explicit finite `time_s` (or a documented source-time alias). Video rows must
carry an explicit `source_time_s`/`source_t_s` mapping. A nominal `fps`, video
presentation timestamp, frame index, or `dt` is never converted into source
time by this component.

Each stream reports its `resolution_s`, source range, gaps, temporal error, and
missing samples. Cursor selection uses the nearest sample within that declared
resolution; ties choose the earlier sample, and no interpolation or stale-tail
holding is performed. Nonzero origins, unequal rates, gaps, and unequal stream
durations therefore remain visible. The `time` block records the rule as
`nearest_sample_within_declared_resolution; no_interpolation`.

`ReviewContext` and `SourceTimeCursor` carry the episode, execution, interval,
actor, cursor, and monotonically increasing `context_revision`. Browser seeks,
event jumps, interval changes, and keyboard stepping dispatch one shared cursor
update; panels only render the resulting snapshot, so panel-to-panel feedback
loops cannot move the cursor.

## Panel contents

- **Scene** preserves recorded actor geometry and the supplied coordinate
  frame. A missing scene or geometry is `unavailable`, not synthesized.
- **Video** preserves mapped frame index, presentation timestamp, camera
  identity, and source-time mapping. A media file without an explicit mapping
  remains unavailable.
- **Metrics** retain source values, units, selected visibility, per-stream
  resolution, and missingness. A small default set is selected when present;
  requested-but-unrecorded metrics remain toggleable unavailable rows. Simple
  speed, angular velocity, and goal-distance values may be derived only from
  explicit recorded scene fields and are marked `derived`.
- **Events** retain episode-scoped IDs, intervals, actor IDs, categories,
  metric values, and declared precursor/recovery links. Clicking an event seeks
  to its recorded start time.
- **Goal geometry** exposes the actual supplied goal point and completion
  boundary with source and coordinate-frame metadata. Canonical
  `threejs-viewer.v1` map goal zones are preserved as the completion boundary
  and provide a representative first-zone centroid when no explicit point is
  recorded. Each field is independently `available` or `unavailable`.

Every source retains declared and computed identity/integrity, source URI,
format, and `admission: not_evaluated`. Integrity and diagnostic availability
do not grant benchmark or scientific admission.

## Browser controls

The local web module provides play/pause, previous/next scene step, speed
selection, interval selection, metric toggles, event seeks, and a shared
scrubber. Space, Left/Right arrows, and `+`/`-` work when focus is not in a
text-entry control. Playback advances the source-time cursor from an injectable
wall-clock/request-animation-frame delta multiplied by the selected speed; it
clamps to the selected interval and pauses at its end. The controller exposes a
deterministic `tick(now_ms)` method for headless runtime probes, and cancels
scheduled handles and its per-controller keyboard listener on pause/unmount.

The scene panel mounts an offline canvas that consumes the existing
`threejs-viewer.v1` map/frame contract. When a local media URI is declared, the
video panel mounts an HTML `<video>` element and applies only the explicit
source-time-to-media-time mapping; remote media and nominal FPS alignment are
not used. Local media referenced by a generated HTML request is materialized
under that output's `media/` directory and the model receives a safe relative
URI. Missing files and absolute, traversal, scheme-relative, or network media
paths emit diagnostics, scrub the video URI, and downgrade the result; the
offline browser policy rejects the same paths unless an explicit media scheme is
supplied.
Metric traces expose one button per recorded sample with its exact source time,
and hiding a metric removes its trace controls. The module has no Three.js, CDN,
unpkg, or other remote asset import; source bytes remain read-only.

## Fixture and evidence boundary

`tests/fixtures/scenario_review/review_panels/` contains a positive fixture with
nonzero time origin, unequal rates/durations, a gap, missing metric sample,
explicit video mapping, event intervals, source identity, goal geometry, and a
derived metric. Focused tests also exercise missing mappings, unavailable
identity/geometry, unsafe paths, incompatible versions, output collisions,
cancellation, and strict CLI handling.

This is diagnostic review tooling only. A complete component result is not
benchmark evidence, a scientific claim, a planner result, or proof of causal
failure.
