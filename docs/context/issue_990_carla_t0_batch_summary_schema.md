# Issue #990 CARLA T0 Batch Summary Schema

## Scope

Issue #990 adds a packaged JSON Schema for
`carla-replay-export-batch-validation-summary.v1` and exposes it through
`load_batch_validation_summary_schema()`. The schema captures the current
`robot-sf-validate-carla-t0-batch --json` contract:

- `schema_version`
- `manifest`
- `payload_count`
- `scenario_ids`

The CLI uses the shared `BATCH_VALIDATION_SUMMARY_SCHEMA_VERSION` constant so emitted JSON and the
schema resource stay aligned.

## Boundary

This change does not change summary contents beyond #988, add a batch-validator `--schema` flag,
change manifest or payload validation semantics, load CARLA, start CARLA, run replay, or compare
simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_json_summary_validates_against_schema -q
```

Failed before implementation because `load_batch_validation_summary_schema` was not exported by
`robot_sf_carla_bridge`.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_json_summary_validates_against_schema tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_prints_json_summary -q
```

Passed after adding the summary schema resource and loader: `2 passed in 12.69s`.
