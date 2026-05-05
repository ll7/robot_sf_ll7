# Issue #988 CARLA T0 Batch Summary Version

## Scope

Issue #988 adds `schema_version: carla-replay-export-batch-validation-summary.v1` to
`robot-sf-validate-carla-t0-batch --json` output. Existing summary fields remain unchanged:

- `manifest`
- `payload_count`
- `scenario_ids`

Text output remains unchanged and still reports the number of validated payloads.

## Boundary

This change does not add the packaged batch-summary JSON Schema, change manifest or payload
validation semantics, load CARLA, start CARLA, run replay, or compare simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_prints_json_summary -q
```

Failed before implementation because the JSON summary lacked `schema_version`.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_prints_json_summary tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_loads_payloads_and_prints_count -q
```

Passed after adding the version marker: `2 passed in 12.66s`.
