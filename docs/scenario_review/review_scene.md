# SREV-09 review scene

Render recorded scenario states through canonical plotting primitives, so
researchers get numbered scene frames plus vector/raster exports and a
per-frame source map without waiting for the whole workbench. See the
[`docs/glossary.md`](../glossary.md) for project terms such as review
bundle and component result.

## Evidence boundary

Diagnostic tooling only. Rendering launches nothing — no simulation,
training, remote scheduling, or scientific publication — and admits no
scientific claim, benchmark result, or planner/simulator behavior change.
Figures are drawn with Matplotlib primitives over recorded trace geometry
only. Required geometry missing from a frame fails closed; schema-optional
radii fall back to documented defaults recorded in the source map.

## Interfaces

- **Input**: a `component-request.v1` request for component
  `srev09-review-scene`. Sources cite trace geometry by identity; the
  citable format is `simulation_trace_export.v1` (validated through
  `robot_sf.analysis_workbench.simulation_trace_export`). Source URIs resolve
  relative to the request file's directory. Other formats are skipped with a
  diagnostic and never block supported sources.
- **Config**: a `scene` mapping with optional `frame_indices` (distinct
  non-negative trace indices; default renders every frame), `formats` (a
  non-empty subset of SVG/PDF/PNG), and a `figure` preset (`width_in`,
  `height_in`, `dpi`; default is a small 3.2x2.4 figure at 80 dpi).
- **`run(request)`**: returns a `component-result.v1` result with status
  `complete` (artifacts per-frame `*-scene_*.svg/pdf/png` plus
  `frame-source-map.json` and `component-descriptor.json`), `partial` (extra
  sources skipped), `unavailable` (unsupported component, capability,
  version, or evidence, with an actionable reason), or `failed` (corrupt
  config, missing geometry, out-of-range indices, or output collision;
  failed outputs never carry artifacts).
- **Output**: numbered scene frames with robot/heading/pedestrian geometry
  in metres/radians, and a source map binding every file to its trace step,
  source time, actor states, and units. `descriptor()` exposes the versioned
  capability descriptor.

Unknown required versions fail; output collisions, absolute paths, traversal,
and symlink escapes are rejected. The source map records the SHA-256 of the
exact source bytes parsed by the canonical trace owner, along with the trace
schema, embedded source identity, evidence boundary, coordinate frame, and
units. `embedded-observed` is intentionally not an external identity
attestation: the current v1 request schema carries no source-byte declaration.
Any available declaration that cannot be checked or that disagrees with the
observed bytes is diagnosed and the source is not rendered. Malformed JSON
object shapes at the CLI/API boundary return a schema-valid failed result.
The CLI result envelope includes `schema_version: component-result.v1`.
Adding independent source hash, commit, or configuration declarations remains
a shared-contract-owner follow-up for SREV-01/#9270; this leaf does not widen
that owner and therefore does not claim those declarations are verified.
SVG output carries fixture coordinates as figure text; PNG dimensions follow
the figure preset. Repeated fixture runs compare equal source-map bytes.

## Usage

```bash
uv run pytest tests/render/test_review_scene.py -q
uv run python -m robot_sf.render.review_scene \
  --input tests/fixtures/scenario_review/review_scene/request.json \
  --config tests/fixtures/scenario_review/review_scene/config.json \
  --output output/scenario_review/srev-09-smoke
```

`run(request)` renders the scenes offline; `descriptor()` lists what this
component provides. Outputs are written atomically into the component-owned
output directory, which must not exist beforehand. Generated output stays
under ignored `output/` and is disposable. No generated media is tracked in
Git.
