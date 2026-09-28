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

## Detector-rule proposals

`DetectorRuleProposal` is the separate BA V1 governance record for a candidate
detector change. It is not a `Finding`, detector implementation, registry entry,
or benchmark result. The record retains a closed proposal kind (`new_rule`,
`threshold_change`, `parameter_change`, or `cohort_change`), target detector ID,
bounded declarative `candidate_rule` data, detector-registry version/digest,
campaign/source provenance, evidence IDs, rationale, proposer/author identity,
and a bounded metadata mapping. Candidate data is strict JSON and rejects
callable, module, command, shell, Python, expression, and other execution keys.

The V1 lifecycle is `proposed`, `approved`, `rejected`, or `withdrawn`.
Non-proposed states require a human decision identity, timestamp, and reason;
the original proposer remains unchanged when a human updates the record. Every
V1 proposal has `activation_status: inactive`, including approved proposals.
Candidate-rule keys use a bounded lower-`snake_case` declarative grammar; they
cannot name callbacks, commands, modules, code, or other execution mechanisms.
The proposal origin—candidate rule, proposer, rationale/metadata, registry
version/digest, campaign/source identity and revision, and evidence IDs—is
immutable across compare-and-swap, force-save, replay, and projection rebuild;
only lifecycle, decision, current-author, and update-audit fields may change.
Record IDs are type-stable, so a proposal cannot be replaced by another audit
record type at the same durable ID.
The published JSON Schema is a wire-level structural precheck; typed
reconstruction remains authoritative for recursive depth/node bounds and
canonical normalization. Schema-only consumers must route records through
`record_from_dict`/`deserialize_record` before persistence or treat them as
untrusted input.
Saving the record uses the same canonical journal, typed reconstruction, and
compare-and-swap revision contract as the other audit records. No proposal is
loaded into the BA-01 default detector registry or applied automatically.
The BA-05 service exposes a source/context/actor-bound agent proposal write and
source-bound reads through the same projection. Human approve/reject/withdraw
decisions require an identified human session and an expected revision; MCP
exposes the proposal write/read path but never decision authority. Every
decision remains `activation_status: inactive`, and no service path mutates the
BA-01 detector registry.

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
