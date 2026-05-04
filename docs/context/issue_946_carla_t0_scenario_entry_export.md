# Issue #946 CARLA T0 Scenario Entry Export

## Goal

Issue #946 adds a CARLA-free helper that exports one scenario-loader entry to a
`carla-replay-export.v1` neutral payload. It is a child of issue #872 and is stacked on the T0
schema, builder, read-helper, and `MapDefinition` adapter work.

## Decision

`build_export_payload_from_scenario_entry(...)`:

- builds the Robot-SF runtime config through `build_robot_config_from_scenario(...)`,
- certifies the scenario through `scenario_cert.v1`,
- selects the loaded map definition,
- derives robot radius, kinematics, timestep, horizon, scenario id, and map id,
- delegates geometry serialization to `build_export_payload_from_map_definition(...)`.

Excluded or unsupported certificates still fail closed through the map-definition adapter.

## Boundary

This remains T0 neutral export construction only. It does not import CARLA, run CARLA, batch-export
scenario files, convert coordinates into a CARLA world, or claim simulator-transfer evidence.

## Validation

Proof command for the implementation branch:

```bash
uv run pytest tests/carla_bridge/test_t0_export.py -q
```

The RED run failed with two missing-import failures for
`build_export_payload_from_scenario_entry`. After implementation, the focused CARLA-free test file
passed with `13 passed`.
