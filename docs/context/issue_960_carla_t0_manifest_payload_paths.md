# Issue #960 CARLA T0 Manifest Payload Paths

## Goal

Issue #960 adds a CARLA-free helper for turning a validated T0 export manifest into local payload
file paths. Downstream tooling can now discover an export batch's payload files without redoing
manifest path handling.

## Decision

`robot_sf_carla_bridge.resolve_export_manifest_payload_paths(path)` reads the manifest through
`read_export_manifest(...)` and returns records with:

- `scenario_id`
- `path`: a `Path` resolved relative to the manifest directory

The helper rejects absolute payload paths and `..` parent-directory escapes so manifest entries stay
scoped to the local export batch.

## Boundary

The helper does not load or validate referenced payload files, invoke CARLA, replay scenarios,
upload artifacts, or change the writer manifest format.

## Validation

- RED: `rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_resolve_export_manifest_payload_paths_returns_local_batch_paths tests/carla_bridge/test_t0_export.py::test_resolve_export_manifest_payload_paths_rejects_unsafe_paths -q`
  failed with `ImportError: cannot import name 'resolve_export_manifest_payload_paths'`.
- GREEN: the same targeted command passed after adding and exporting the helper.
