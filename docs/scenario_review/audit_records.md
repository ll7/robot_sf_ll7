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
world-space points from video require an explicit calibration mapping. Existing
SREV `SourceRef`, trace/timeline, and annotation contracts remain the owners of
those semantics; this package stores links and review metadata around them.

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

There is intentionally no hosted multi-user database, raw-trace format,
simulator replay, automatic benchmark regrading, or network/GitHub side effect
in this package. Those operations belong to the later audit service slices.
