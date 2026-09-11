# Compute-window schema/reader inventory

`compute_window_schema_reader_inventory.py` validates a versioned role packet and emits deterministic JSON/table output; run it with `--packet ... --root . --format table`.

Roles bind exact schema/version, source SHA-256, reader, dependencies, compatibility, units/display metadata, representative bytes, and check command. JSON sources use `$id` plus version constraints; Python sources use top-level schema constants. Digest-bound `python_path` readers have a two-second child deadline, and missing or mismatched inputs fail closed with six explicit statuses.
