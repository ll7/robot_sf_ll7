# Review camera and presentation layout (SREV-11)

Command-line glossary: SREV is the scenario-review portfolio; a review camera specification
(`camera-layout-spec.v1`) defines renderer-neutral full-scene, follow, and event camera tracks
plus presentation layout; a component request (`component-request.v1`) is the fixture envelope
that invokes one workbench component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.render.review_camera` computes deterministic camera tracks and presentation layout
specifications from scenario geometry and trajectory data:
- **Aspect ratio validation**: ensures presentation dimensions (e.g. 1920x1080 or 320x180) are finite and positive.
- **Finite and in-bounds camera coordinates**: guarantees that all centers, extents, and bounds are finite numbers within the scenario world.
- **Full-scene static camera**: always available whenever geometry is declared; centers on scenario geometry and fits presentation aspect ratio with margins.
- **Follow camera track**: follows actor trajectories with configurable context retention margin (e.g. 5.0m) and in-bounds clamping. Follow mode requires stated trajectory data; if absent, it reports `unavailable` with an explicit reason.
- **Event camera**: centers on scenario event positions with context retention. Event mode requires stated event or trajectory data; if absent, it reports `unavailable`.
- **Cropped content disclosure**: any view that crops scenario geometry explicitly discloses `is_cropped: true`, references the full-scene geometry bounds, and supplies an overview inset specification.
- **Pixel-metric freedom**: simulation and presentation measurements remain strictly in world units (meters, seconds, radians). No metrics are derived from screen pixels.

## Command

```bash
uv run python -m robot_sf.render.review_camera \
  --input tests/fixtures/scenario_review/review_camera/request.json \
  --config tests/fixtures/scenario_review/review_camera/config.json \
  --output output/scenario_review/srev-11-smoke
```

Inspect capability descriptor:

```bash
uv run python -m robot_sf.render.review_camera --descriptor
```

## Output

- `camera-layout-spec.v1.json`: complete camera specification containing:
  - `active_camera_mode`: `"full_scene"`, `"follow"`, or `"event"`.
  - `presentation`: dimensions (`width`, `height`, `aspect_ratio`), `fps`, `speed`.
  - `geometry_bounds`: scenario bounding box `(min_x, max_x, min_y, max_y)`.
  - `cameras`: specifications for available cameras (`full_scene`, `follow`, `event`).
  - `layout`: viewport and optional overview inset specification when cropped.
  - `tracks`: per-timestep / keyframe camera tracks.
  - `disclosures`: explicit disclosures on units, coordinate bounds, and context retention.
- `visualization-spec.v1.json`: compatible `visualization-spec.v1` document linking camera and layout to declared sources.
- Component result JSON printed to stdout: status (`complete`, `partial`, `unavailable`, `failed`), written artifacts, diagnostics, and provenance.

## Unavailable reasons and limits

- If follow mode or `follow-camera` capability is requested but no trajectory data is declared, the component reports `unavailable` with reason `follow mode requires stated trajectory data`.
- If event mode or `event-camera` capability is requested but no event or trajectory data is declared, the component reports `unavailable` with reason `event mode requires stated trajectory or event data`.
- If unsupported capabilities are requested, the component reports `unavailable` with `missing capabilities: <list>`.
- Non-finite geometry, dimensions, or coordinates fail closed (`failed` status) with descriptive reasons.
- Output directory collisions are rejected to prevent overwriting prior runs or durable data.
- Evidence boundary: camera and layout specifications provide diagnostic visualization tooling only. No benchmark admission, planner change, or scientific claim follows from camera specification.
