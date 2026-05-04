# Issue #954 CARLA T0 Export CLI Packaging

## Goal

Issue #954 exposes the CARLA-free export CLI from issue #952 as an installable project script
without changing the CLI argument contract or adding CARLA runtime behavior.

## Decision

The project script is:

- `robot-sf-export-carla-t0 = "robot_sf_carla_bridge.cli:export_t0_scenarios_main"`

The stacked base already includes `robot_sf_carla_bridge` in Hatch wheel and sdist metadata, so this
slice adds the script entry and a packaging-surface regression test that reads `pyproject.toml`.

## Boundary

This does not publish a package artifact, add CARLA replay, upload exports, or change the neutral
`carla-replay-export.v1` schema. It only makes the existing import-safe CLI reachable after package
installation.

## Validation

- RED: `rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py -q` failed with
  `KeyError: 'robot-sf-export-carla-t0'`.
- GREEN: `rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py -q` passed after adding the
  project script entry.
