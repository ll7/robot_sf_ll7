# Compute-window schema/reader inventory

`compute_window_schema_reader_inventory.py` preserves active compute-window output read paths. It
consumes a `compute-window-schema-reader-inventory.v1` packet and emits deterministic JSON/table:

```bash
uv run python scripts/tools/compute_window_schema_reader_inventory.py \
  --packet tests/fixtures/compute_window_schema_reader_inventory/active_roles.json --root . --format table
```

Each role binds exact schema/version, reader symbol/source SHA-256, dependencies, compatibility, units/display metadata, representative bytes, a validation command, and exact source material.

The capsule covers episode, Parquet metadata, trace, snapshot, dashboard-input, and migrated trace-series outputs. `reader.execution: python_path` executes a digest-bound reader; incompatible roles are explicitly `reader_unavailable`.

The checker requires exact representative schema versions and does not convert historical data. Missing bindings, version/digest drift, stale adapters, unit/display drift, unsafe paths, missing
commands, and ambiguous roles fail closed. Statuses are `readable_verified`,
`readable_with_declared_adapter`, `schema_only`, `reader_unavailable`, `unversioned`, and `conflict`;
exit codes are 0 for valid packets, 2 for blocked packets, and 3 for malformed JSON.
