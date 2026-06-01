# Trajectory Debug Visualization

Related issue: `ll7/robot_sf_ll7#1244`

`scripts/tools/rerun_debug_export.py` converts one episode JSONL file into a compact debug timeline
for inspecting planner, pedestrian, and adversarial-failure behavior.

The default export is dependency-free JSON:

```bash
uv run python scripts/tools/rerun_debug_export.py \
  --source output/benchmarks/example/episodes.jsonl \
  --output output/debug/example_timeline.json \
  --format json
```

The JSON payload uses `schema_version: robot-sf-debug-timeline.v1` and records:

- episode id, scenario id, seed, status, and terminal event,
- per-frame robot pose,
- per-frame pedestrian poses,
- selected/proposed action when present,
- TTC, PET, and clearance annotations when present in frame or episode metrics.

For typed `simulation_trace_export.v1` artifacts, render a static Markdown inspection report with:

```bash
uv run python scripts/tools/render_trace_report.py \
  --trace tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json \
  --output output/debug/trace_report/report.md
```

Qualitative frame-range annotations for trace fixtures use `trace_annotation_set.v1`. They anchor
review comments to inclusive trace steps, optional planner event IDs, and robot or pedestrian
entities while preserving a strict `analysis_workbench_qualitative_only` evidence boundary. The
fixture added for issue #1962 is:

```text
tests/fixtures/analysis_workbench/trace_annotation_set_v1/issue_1962_planner_sanity_open_annotations.json
```

An optional Rerun output path is available when `rerun-sdk` is installed in the active environment:

```bash
uv run python scripts/tools/rerun_debug_export.py \
  --source output/benchmarks/example/episodes.jsonl \
  --output output/debug/example_timeline.rrd \
  --format rerun
```

If `rerun-sdk` is absent, the command fails closed with an install hint instead of adding a required
dependency to the repository.

## Three.js Trace Viewer

For browser-based qualitative review of `simulation_trace_export.v1` fixtures, use the static
Three.js trace viewer:

```bash
uv run python -m robot_sf.render.trace_viewer \
  tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json \
  --output-dir output/trace_viewer
```

Open `output/trace_viewer/index.html` in a browser. The viewer provides:

- Timeline scrub with frame-by-frame robot pose/heading and pedestrian positions.
- HUD overlay showing trace ID, scenario, planner, seed, step, time, planner action
  (linear/angular velocity), event type, and event ID when available.
- Trajectory polyline rendered from cumulative robot positions.
- Optional annotation embedding when paired with a `trace_annotation_set.v1` fixture:

```bash
uv run python -m robot_sf.render.trace_viewer \
  tests/fixtures/analysis_workbench/simulation_trace_export_v1/planner_sanity_open_episode_0000.json \
  --annotations tests/fixtures/analysis_workbench/trace_annotation_set_v1/issue_1962_planner_sanity_open_annotations.json \
  --output-dir output/trace_viewer
```

The viewer reuses the same `robot_sf/render/web_assets/` static assets (Three.js CDN, dark theme)
as the JSONL-playback viewer. Map bounds are auto-computed from trace positions; no SVG map
geometry is available in this mode.

### Annotation Span Markers

When paired with a `trace_annotation_set.v1` fixture, the timeline displays colored span markers
for each annotation's frame range (`frame_start` to `frame_end`). The HUD overlays show matching
annotation summaries when the scrubber lands on an annotated frame. Annotation anchors carry
`event_ids`, `entities`, and `details` for the frontend to surface during interactive review.

```bash
uv run python -m robot_sf.render.trace_viewer \
  tests/fixtures/analysis_workbench/simulation_trace_export_v1/planner_sanity_open_episode_0000.json \
  --annotations tests/fixtures/analysis_workbench/trace_annotation_set_v1/issue_1962_planner_sanity_open_annotations.json \
  --output-dir output/trace_viewer
```

The HUD also displays the source fixture path and a `[diagnostic-only]` badge. Annotation summaries
appear inline in the HUD bar for every frame within an annotation span.

### Source Fixture Path

The viewer HUD shows the source fixture path (`src=`) when available, making it easy to identify
which tracked fixture produced the current view.

### Diagnostic-Only Status

The scene metadata carries `diagnostic_only: true` and the HUD displays a `[diagnostic-only]`
badge. The limitations list also states that the viewer is diagnostic-only and not benchmark
evidence. These diagnostics identify the artifact as qualitative review tooling and prevent
mistaking it for benchmark evidence.

Programmatic usage:

```python
from pathlib import Path
from robot_sf.analysis_workbench.simulation_trace_export import load_simulation_trace_export
from robot_sf.render.trace_viewer import export_trace_viewer

trace = load_simulation_trace_export(Path("path/to/trace.json"))
result = export_trace_viewer(trace, "output/trace_viewer")
print(f"Viewer files in {result.output_dir}")
```

## Evidence Boundary

Debug timelines and trace viewer exports are diagnostic visualization artifacts. They do not replace
episode JSONL, publication bundles, release manifests, benchmark summaries, or paper-facing evidence
contracts. Keep generated viewer and timeline files under ignored `output/` unless a compact review
fixture is explicitly needed.
