# Compute-window schema/reader inventory

`compute_window_schema_reader_inventory.py` preserves the read path for active compute-window
outputs. It consumes a JSON packet with schema `compute-window-schema-reader-inventory.v1`, one
record per unambiguous output role, and emits deterministic JSON or a compact table:

```bash
uv run python scripts/tools/compute_window_schema_reader_inventory.py \
  --packet <packet.json> --root <source-root> --check --format json
uv run python scripts/tools/compute_window_schema_reader_inventory.py \
  --packet <packet.json> --root <source-root> --format table
```

Each role binds the exact schema/version, reader symbol and source SHA-256, dependencies,
compatibility mode, schema-bound units/display metadata, representative bytes, a validation
command, and content-addressed source material. Existing owners remain authoritative: result-store
readers, case-workbench schemas, trace-series adapters/export validators, and the compute-window
readiness dashboard are referenced by packet entries; this tool does not copy or change them.

The checker reads lightweight JSON/JSONL fixtures, including Parquet-like metadata, trace,
snapshot, and migrated records, and requires each representative object or row to carry the exact
declared schema version. It also checks that an available reader symbol occurs in its digest-bound
source. It does not import heavy optional engines or convert historical data. Missing
schema/version, reader or dependency, digest/version mismatch, stale adapter, unit/display drift,
absolute paths, missing check commands, and ambiguous roles fail closed. Statuses are
`readable_verified`, `readable_with_declared_adapter`, `schema_only`, `reader_unavailable`,
`unversioned`, and `conflict`; exit codes are 0 for a valid packet, 2 for a blocked packet, and 3
for malformed packet JSON.
