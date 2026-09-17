# Durable benchmark-audit records

BA-03 provides an offline, portable record store for the benchmark auditor. The
canonical record is versioned NDJSON; SQLite is only a disposable projection
that can be deleted and rebuilt. Raw traces and videos stay in their artifact
store and are referenced by `SourceRef` identity and digest.

## Python launch route

The store works without a browser, network, simulator, or Codex client:

```python
from pathlib import Path

from robot_sf.analysis_workbench import Annotation, AuditStore, TimeInterval

root = Path("audit-data")
with AuditStore(root) as store:
    annotation = Annotation(
        annotation_id="annotation-1",
        episode_id="episode-identity-from-audit-contract",
        classification="interesting_valid",
        mode="quick",
        interval=TimeInterval(1.0, 1.5),
    )
    receipt = store.save(annotation, operation_id="client-uuid-1", expected_revision=0)
```

The save receipt is acknowledged only after the canonical journal line is
flushed and synced. Use the returned per-record revision for the next compare-
and-swap update. A stale revision raises `AuditConflictError`; reusing an
operation ID with the same request is idempotent, while reusing it for another
request raises `OperationConflictError`.

## Records and evidence identity

`EpisodeRef` requires campaign and source digests plus an execution ID. Planner,
scenario, and seed are lookup fields and never form the complete identity. A
rerun, changed configuration, checkpoint, or environment therefore receives a
different episode identity. `Reference` accepts image-space points from video;
image references retain source-image coordinates, timestamp, and optional seek
identity. Use the closed `ImageDisplayTransform` contract for source size, crop,
and display size when a UI resize/crop is involved; it provides exact
source/display round trips. World-space points from video require an explicit
`{"kind": ..., "version": ..., "parameters": {...}}` calibration mapping.
Opaque calibration scalars are rejected. Existing
SREV `SourceRef`, trace/timeline, and annotation contracts remain the owners of
those semantics; this package stores links and review metadata around them.
An `EpisodeRef` with a `SourceRef.sha256` must use the same source digest. An
annotation's hash-bound `source_identity` and optional `source_revision` are
validated against its `source_ref`; explicit `verified`, `stale`, `mutated`,
and `unavailable` provenance statuses prevent a changed or missing source from
being silently rebound.

Quick and one-click annotations do not require a cause, confidence, or detailed
evidence. Their `review_scope` remains `interval` unless a caller explicitly
creates a `full_episode` review. Author kind is one of `human`, `detector`, or
`agent`; an agent record cannot impersonate a human review.

Signals use `flagged`, `clear`, `unavailable`, or `error`. Missing optional media
is recorded as missingness, not silently treated as corruption.

## Findings and related cases

`Finding` keeps candidate members, confirmed members, and negative controls in
separate collections. `audit_findings.confirm_member` is an explicit action;
similarity and agent suggestions never promote membership or causal status.
Finding states are `proposed`, `under_investigation`, `supported`, `refuted`,
and `resolved`.

`find_similar_cases` supports named modes for same-scenario/cross-planner,
same-planner/cross-seed, symptom, anomaly signature, geometry, outcome, metric,
and existing-finding retrieval. Each result includes matched features,
missingness, compatibility (`compatible`, `unknown`, or `incompatible`), and
human-readable reasons. Scores are retrieval priorities, not probabilities or
evidence of a shared cause.

## Recovery, backup, and relocation

The journal is `audit.ndjson`; the projection is `audit.sqlite3`. If a process
stops after the journal sync but before SQLite projection, opening the store
replays the journal and advances the projection checkpoint. A partial final
NDJSON line is discarded during recovery; a malformed complete line fails
closed. Tombstones remain in history, and `store.undo(...)` writes a new
revision rather than erasing history.

Portable backups contain the canonical journal and a manifest, but not raw
media or the disposable index by default:

```python
backup = store.export("audit-backup.zip")
restored = AuditStore.restore(backup, "relocated-audit")
restored.close()
```

The manifest includes a journal digest and restore rejects tampered archives,
unsafe archive paths, unsupported schema versions, and obvious credential
fields. Relocation does not rewrite source identities; callers must preserve or
rehydrate the referenced artifact root and verify each `SourceRef` digest.
`store.migrate_schema(...)` writes a versioned migrated copy without mutating
the original store. Existing SQLite files are never required for restore.

`store.save(...)` requires a compare-and-swap revision for an existing record;
unconditional replacement is available only through the explicitly named
`store.force_save(...)`. The compatibility `write_ndjson(...)` helper accepts
only `audit.ndjson` and routes through `AuditStore`, so a record-only file cannot
be mistaken for the canonical transaction journal. Repeating an identical
`write_ndjson(...)` call is an idempotent no-op; changed payloads use a new
operation ID and the current revisions as compare-and-swap expectations.

## Offline command line

The bounded, machine-readable command line uses exit code `0` for success and
`2` for contract, collision, or storage errors. It never fetches raw media or
contacts a service:

```bash
python -m robot_sf.analysis_workbench.audit_store inspect audit-data
python -m robot_sf.analysis_workbench.audit_store rebuild audit-data
python -m robot_sf.analysis_workbench.audit_store export audit-data audit-backup.zip
python -m robot_sf.analysis_workbench.audit_store restore audit-backup.zip relocated-audit
python -m robot_sf.analysis_workbench.audit_store restore audit-backup.zip audit-data --overwrite
```

`export` and `restore` refuse existing destinations unless `--overwrite` is
explicit. Restore validates and rebuilds in a same-parent staging directory,
then swaps it atomically; a malformed or digest-valid-but-unreadable backup
leaves an existing destination untouched. `inspect` and `rebuild` report JSON
checkpoint and record metadata on standard output; errors are JSON on standard
error.

There is intentionally no hosted multi-user database, raw-trace format,
simulator replay, automatic benchmark regrading, or network/GitHub side effect
in this package. Those operations belong to the later audit service slices.
