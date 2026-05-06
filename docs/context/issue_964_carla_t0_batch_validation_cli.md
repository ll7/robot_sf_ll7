# Issue #964 CARLA T0 Batch Validation CLI

## Scope

Issue #964 adds a CARLA-free command-line entry point for validating a complete local T0 export
batch. The CLI accepts a manifest path, loads every referenced payload through the issue #962 batch
loader, and reports the validated payload count.

The change is intentionally stacked on the issue #962 loader. Missing payload files, unsafe manifest
paths, invalid manifest JSON, and invalid payload JSON are handled by the existing loader and schema
validation path instead of a separate CLI-only parser.

## Boundary

This note does not claim CARLA replay parity or execute CARLA. The validation only proves that a
local Robot-SF T0 export batch is internally readable and schema-valid through the CARLA-free bridge
helpers.

Out of scope:

- CARLA simulator startup or replay execution.
- Durable artifact upload or hydration.
- Writer format changes.
- Benchmark-strength parity claims.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_loads_payloads_and_prints_count tests/carla_bridge/test_t0_export_cli.py::test_export_t0_cli_is_packaged_as_project_script -q
```

Failed before implementation because `validate_t0_export_batch_main` and the
`robot-sf-validate-carla-t0-batch` project script were absent.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_loads_payloads_and_prints_count tests/carla_bridge/test_t0_export_cli.py::test_export_t0_cli_is_packaged_as_project_script -q
```

Passed after adding the CLI wrapper and project script metadata: `2 passed`.
