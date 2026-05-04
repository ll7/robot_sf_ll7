# Issue #956 CARLA T0 Export Manifest Reader

## Goal

Issue #956 adds a CARLA-free helper for reading the `manifest.json` emitted by the T0 export record
writer. This gives downstream local tooling a validated entry point before it decides whether to
load individual payload files.

## Decision

`robot_sf_carla_bridge.read_export_manifest(path)` reads UTF-8 JSON and validates:

- manifest object shape,
- `schema_version == "carla-replay-export-manifest.v1"`,
- `exports` as a list,
- each entry's non-empty `scenario_id` and relative payload `path` strings.

The helper returns the manifest dictionary unchanged after validation and deliberately does not load
or validate each referenced payload file.

## Boundary

This is local-file-only T0 tooling. It does not add CARLA replay, durable artifact upload, payload
hydration, or any manifest writer format change.

## Validation

- RED: `rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_read_export_manifest_round_trips_writer_output tests/carla_bridge/test_t0_export.py::test_read_export_manifest_rejects_invalid_manifest_shape -q`
  failed with `ImportError: cannot import name 'read_export_manifest'`.
- GREEN: the same targeted command passed after implementing and exporting the helper.
