# Issue #986 CARLA T0 Manifest Schema CLI

## Scope

Issue #986 extends `robot-sf-validate-carla-t0-manifest` with `--schema`, which prints the packaged
`carla-replay-export-manifest.v1` JSON Schema as deterministic JSON. The command returns `0`
without requiring `--manifest`, reading an export manifest, loading payload files, installing CARLA,
or running replay.

Existing manifest validation behavior remains intact: normal validation mode still requires
`--manifest` and reports the export count for that manifest.

## Boundary

This change only exposes the existing local manifest schema through the CLI. It does not change the
schema contents, change payload schema CLI behavior, load referenced payload files, install CARLA,
start CARLA, run replay, or compare simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_manifest_main_prints_schema -q
```

Failed before implementation because `--schema` still required `--manifest`.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_manifest_main_prints_schema tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_manifest_main_reads_manifest_and_prints_count -q
```

Passed after adding schema mode: `2 passed in 12.65s`.
