# Issue #958 CARLA T0 Manifest Validation CLI

## Goal

Issue #958 adds a thin CARLA-free CLI for validating a local T0 export `manifest.json` before
downstream scripts consume an export batch.

## Decision

`robot_sf_carla_bridge.cli.validate_t0_manifest_main(argv)` accepts:

- `--manifest`: path to the export manifest JSON.

The wrapper calls `read_export_manifest(...)`, prints a concise export count, and returns
process-style exit code `0` when validation succeeds. The installable project script is:

- `robot-sf-validate-carla-t0-manifest`

## Boundary

The validator intentionally does not load referenced payload files, invoke CARLA, replay scenarios,
upload artifacts, or change the manifest writer format. Invalid manifests raise through the reader,
which gives normal console-script execution a non-zero process exit.

## Validation

- RED: `rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_manifest_main_reads_manifest_and_prints_count tests/carla_bridge/test_t0_export_cli.py::test_export_t0_cli_is_packaged_as_project_script -q`
  failed with missing `validate_t0_manifest_main` and missing project script metadata.
- GREEN: the same targeted command passed after adding the CLI wrapper and script entry.
