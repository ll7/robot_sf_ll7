# Issue #962 CARLA T0 Manifest Payload Loader

## Goal

Issue #962 adds a CARLA-free helper for loading every payload JSON file referenced by a validated
T0 export manifest.

## Decision

`robot_sf_carla_bridge.load_export_manifest_payloads(path)` composes existing helpers:

1. `resolve_export_manifest_payload_paths(...)` validates the manifest and resolves safe local paths.
2. `read_export_payload(...)` loads and schema-validates each payload file.

The helper preserves manifest order and returns records with:

- `scenario_id`
- `path`
- `payload`

## Boundary

This remains local T0 export tooling. It does not invoke CARLA, replay scenarios, upload artifacts,
change manifest/payload writer formats, or make benchmark parity claims.

## Validation

- RED: `rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_load_export_manifest_payloads_preserves_manifest_order tests/carla_bridge/test_t0_export.py::test_load_export_manifest_payloads_fails_for_missing_payload -q`
  failed with `ImportError: cannot import name 'load_export_manifest_payloads'`.
- GREEN: the same targeted command passed after adding and exporting the helper.
