# Issue #980 CARLA Availability Schema CLI

## Scope

Issue #980 extends `robot-sf-check-carla` with `--schema`, which prints the
`carla-availability.v1` JSON Schema as deterministic JSON. The command uses
`load_availability_schema()` and returns `0` without probing for CARLA availability.

Existing modes remain intact:

- text status output,
- `--json` status output,
- `--require` fail-closed status checks.

## Boundary

This change only exposes the local metadata schema. It does not install CARLA, start CARLA, run
replay, change dependency metadata, or compare simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_schema -q
```

Failed before implementation because `--schema` was not recognized.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_schema tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_json_status tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_require_fails_when_unavailable -q
```

Passed after adding schema mode: `3 passed`.
