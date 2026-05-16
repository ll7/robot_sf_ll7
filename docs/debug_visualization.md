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

An optional Rerun output path is available when `rerun-sdk` is installed in the active environment:

```bash
uv run python scripts/tools/rerun_debug_export.py \
  --source output/benchmarks/example/episodes.jsonl \
  --output output/debug/example_timeline.rrd \
  --format rerun
```

If `rerun-sdk` is absent, the command fails closed with an install hint instead of adding a required
dependency to the repository.

## Evidence Boundary

Debug timelines are diagnostic visualization artifacts. They do not replace episode JSONL,
publication bundles, release manifests, benchmark summaries, or paper-facing evidence contracts.
Keep generated timeline files under ignored `output/` unless a compact review fixture is explicitly
needed.
