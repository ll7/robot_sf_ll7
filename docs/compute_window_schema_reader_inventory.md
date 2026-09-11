# Compute-window schema/reader inventory

`compute_window_schema_reader_inventory.py` preserves the read path for active compute-window outputs. It consumes a JSON packet with schema `compute-window-schema-reader-inventory.v1`, one record per
unambiguous output role, and emits deterministic JSON or a compact table:

```bash
uv run python scripts/tools/compute_window_schema_reader_inventory.py \
  --packet tests/fixtures/compute_window_schema_reader_inventory/active_roles.json \
  --root . --check --format json
uv run python scripts/tools/compute_window_schema_reader_inventory.py --packet <packet.json> --root <source-root> --format table
```

Each role binds exact schema/version, reader symbol/source SHA-256, dependencies, compatibility,
schema-bound units/display metadata, representative bytes, validation command, and source material.
Existing result-store readers, case-workbench schemas, trace-series adapters/export validators, and
the compute-window readiness dashboard remain authoritative; this tool does not copy or change them.

The checked-in capsule covers episode, Parquet metadata, trace, snapshot, dashboard-input, and migrated trace-series outputs; its fixture test is the inventory smoke check.

The checker reads lightweight JSON/JSONL fixtures, including Parquet-like metadata, trace, snapshot, and migrated records, and requires each representative object or row to carry the exact
declared schema version. It also checks that an available reader symbol occurs in its digest-bound
source. It does not import heavy optional engines or convert historical data. Missing
schema/version, reader or dependency, digest/version mismatch, stale adapter, unit/display drift,
absolute paths, missing check commands, and ambiguous roles fail closed. Statuses are
`readable_verified`, `readable_with_declared_adapter`, `schema_only`, `reader_unavailable`,
`unversioned`, and `conflict`; exit codes are 0 for a valid packet, 2 for a blocked packet, and 3
for malformed packet JSON.
