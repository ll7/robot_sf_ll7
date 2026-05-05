# Issue #978 CARLA Availability JSON Schema

## Scope

Issue #978 adds `robot_sf_carla_bridge/schemas/carla_availability.v1.json` and a
`load_availability_schema()` helper. The schema validates the versioned availability metadata
emitted by `check_carla_availability()` and `robot-sf-check-carla --json`.

The schema covers both available and not-available states and preserves the existing metadata
contract:

- `schema_version`
- `available`
- `status`
- `reason`
- `dependency`

## Boundary

This change only adds schema validation for local setup metadata. It does not install CARLA, start
CARLA, run replay, change dependency metadata, or compare simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_missing_carla_availability_validates_against_schema tests/carla_bridge/test_t0_export.py::test_importable_carla_availability_validates_against_schema -q
```

Failed before implementation because `load_availability_schema` was absent.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_missing_carla_availability_validates_against_schema tests/carla_bridge/test_t0_export.py::test_importable_carla_availability_validates_against_schema -q
```

Passed after adding the schema resource and loader: `2 passed`.
