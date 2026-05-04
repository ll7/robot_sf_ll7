# Issue #952 CARLA T0 Export CLI

## Goal

Issue #952 adds a thin, CARLA-free command-line wrapper around the existing T0 scenario-file
exporter and deterministic record writer.

## Decision

The CLI lives at `robot_sf_carla_bridge.cli.export_t0_scenarios_main(argv)`. It accepts:

- `--scenario-file`: scenario manifest YAML path.
- `--output-dir`: local directory for exported JSON records and `manifest.json`.
- `--robot-sf-commit`: required provenance identifier.
- `--created-by`: optional provenance producer, defaulting to `robot_sf_carla_bridge.cli`.
- `--certificate-generator`: optional provenance generator, defaulting to `scenario_cert.v1`.

The wrapper builds a provenance mapping, calls
`build_export_payloads_from_scenario_file(...)`, writes records with `write_export_records(...)`,
prints the generated manifest path, and returns process-style exit code `0`.

## Boundary

This is intentionally not a packaged `console_scripts` entry point yet. It also does not import
CARLA, perform replay, upload durable artifacts, or change the export schema. The implementation is
stacked on issue #950's local record writer and keeps runtime behavior limited to local JSON export.

## Validation

- RED: `rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py -q` failed with
  `ModuleNotFoundError: No module named 'robot_sf_carla_bridge.cli'`.
- GREEN: `rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py -q` passed after adding the
  CLI wrapper.
