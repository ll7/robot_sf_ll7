# Post-Access Compute and Artifact Handoff Protocol

Plain-language summary: defines the deterministic post-access execution and artifact handoff contract
generated when compute-window access concludes. Tool owner: [generate_post_access_handoff.py](../../scripts/tools/generate_post_access_handoff.py).

Status: operational protocol. Governed by [artifact_retention_and_cleanup.md](artifact_retention_and_cleanup.md).

## 1. Schema and Formats

- **Output Schema**: `robot_sf.post_access_handoff.v1`.
- **CLI Commands**:
  - `uv run python scripts/tools/generate_post_access_handoff.py --check --inventory <INVENTORY_JSON> --format json`
  - `uv run python scripts/tools/generate_post_access_handoff.py --inventory <INVENTORY_JSON> --format markdown --output <REPORT_MD>`
- **Workload States**: `completed_validated`, `completed_unvalidated`, `running`, `queued`, `failed`, `cancelled`, `harvest_blocked`, `transfer_blocked`, `not_submitted`, and `safe_to_defer`.

## 2. Integrity and Privacy Guarantees

- **Incomplete Workload Enforcement**: Every incomplete row requires an actionable `next_command` or explicit blocker reason.
- **Fail-Closed Relational Integrity**: Detects and rejects contradictory statuses, orphan scheduler jobs, orphan artifacts, and duplicate identities.
- **Public Redaction**: Automatically scans and redacts private absolute paths, private cluster hostnames and IP addresses, credentials, and signed query parameters.
