"""Crash-safe canonical storage and rebuildable SQLite projection for BA-03.

``AuditStore`` is intentionally a small local database rather than a second
domain model.  The NDJSON journal is authoritative; SQLite contains only an
index/projection and a checkpoint.  A process lock plus a thread lock means
browser and agent clients cannot append independently or lose a revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.audit_contracts import (
    AUDIT_EXPORT_SCHEMA_VERSION,
    AUDIT_JOURNAL_SCHEMA_VERSION,
    AUDIT_RECORD_SCHEMA_VERSION,
    DETECTOR_RULE_PROPOSAL_IMMUTABLE_FIELDS,
    AuditContractError,
    canonical_json,
    record_from_dict,
    record_id,
    record_to_dict,
    record_type,
)
from robot_sf.common.optional_import import try_import

fcntl = try_import("fcntl")  # Platform-dependent stdlib module (Unix); None on Windows.


class AuditStoreError(RuntimeError):
    """Base class for durable-store failures."""


class AuditConflictError(AuditStoreError):
    """Raised when an expected record revision is stale."""

    def __init__(self, record_id: str, expected: int | None, actual: int):
        """Build a conflict with both caller and current revisions."""

        self.record_id = record_id
        self.expected_revision = expected
        self.actual_revision = actual
        super().__init__(
            f"expected_revision is required for existing record {record_id!r}; "
            f"current revision is {actual}"
            if expected is None
            else f"revision conflict for {record_id!r}: expected {expected!r}, current {actual}"
        )


RevisionConflictError = AuditConflictError
ExpectedRevisionRequiredError = AuditConflictError


class OperationConflictError(AuditStoreError):
    """Raised when an operation ID is reused for a different request."""


class AuditCorruptionError(AuditStoreError):
    """Raised when a non-trailing canonical journal line is invalid."""


class ProjectionError(AuditStoreError):
    """Raised when SQLite cannot be updated; canonical data remains recoverable."""


class AuditExportError(AuditStoreError):
    """Raised when an export is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class StoredRecord:
    """One projected record and its durable revision metadata."""

    record_id: str
    record_type: str
    revision: int
    deleted: bool
    record: Any | None
    operation_id: str
    committed_at: str
    global_revision: int

    @property
    def payload(self) -> Any | None:
        """Alias used by callers that prefer payload terminology."""

        return self.record

    @property
    def value(self) -> Any | None:
        """Return the typed value represented by this projected row."""

        return self.record

    @property
    def is_tombstone(self) -> bool:
        """Return whether this projected row is a deletion tombstone."""

        return self.deleted

    def to_dict(self) -> dict[str, Any]:
        """Return record and durable revision metadata as JSON-compatible data.

        Returns:
            A mapping suitable for diagnostics or an export manifest.
        """

        payload: dict[str, Any] = {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "revision": self.revision,
            "deleted": self.deleted,
            "operation_id": self.operation_id,
            "committed_at": self.committed_at,
            "global_revision": self.global_revision,
        }
        payload["record"] = record_to_dict(self.record) if self.record is not None else None
        return payload

    def __getattr__(self, name: str) -> Any:
        """Delegate record fields while preserving revision metadata.

        Returns:
            The requested field from the typed record.
        """

        record = object.__getattribute__(self, "record")
        if record is not None:
            return getattr(record, name)
        raise AttributeError(name)


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Receipt returned only after the canonical line is fsynced."""

    operation_id: str
    record_id: str
    record_type: str
    revision: int
    global_revision: int
    committed: bool = True
    replayed: bool = False
    deleted: bool = False
    checkpoint_revision: int = 0

    @property
    def record_revision(self) -> int:
        """Return the per-record revision number."""

        return self.revision

    @property
    def revision_id(self) -> int:
        """Return the per-record revision under the explicit ID alias."""

        return self.revision


@dataclass(frozen=True, slots=True)
class BatchCommitResult:
    """Receipt for an atomic multi-record transaction."""

    operation_id: str
    changes: tuple[CommitResult, ...]
    global_revision: int
    committed: bool = True
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class ProjectionCheckpoint:
    """SQLite position proved to include canonical journal data."""

    revision: int
    offset: int
    digest: str


@dataclass(frozen=True, slots=True)
class _Change:
    record_id: str
    record_type: str
    revision: int
    deleted: bool
    record: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _JournalTransaction:
    operation_id: str
    global_revision: int
    committed_at: str
    expected_revisions: Mapping[str, int | None]
    changes: tuple[_Change, ...]
    request_digest: str
    actor: Mapping[str, Any]
    unconditional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_JOURNAL_SCHEMA_VERSION,
            "kind": "transaction",
            "operation_id": self.operation_id,
            "global_revision": self.global_revision,
            "committed_at": self.committed_at,
            "expected_revisions": dict(self.expected_revisions),
            "changes": [
                {
                    "record_id": change.record_id,
                    "record_type": change.record_type,
                    "revision": change.revision,
                    "deleted": change.deleted,
                    "record": change.record,
                }
                for change in self.changes
            ],
            "request_digest": self.request_digest,
            "actor": dict(self.actor),
            "unconditional": self.unconditional,
        }


class _PathLock:
    """Thread/process lock for one canonical store path."""

    _locks: dict[str, threading.RLock] = {}
    _guard = threading.Lock()

    def __init__(self, path: Path):
        key = str(path.resolve())
        with self._guard:
            self._thread_lock = self._locks.setdefault(key, threading.RLock())
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> _PathLock:
        self._thread_lock.acquire()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
        self._thread_lock.release()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_json_constant(token: str) -> Any:
    raise AuditExportError(f"non-finite JSON constant is not allowed: {token}")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditStoreError(f"{name} must be a mapping")
    return dict(value)


class AuditStore:
    """Single-writer canonical audit store with a disposable SQLite index.

    Args:
        root: Directory containing ``audit.ndjson`` and ``audit.sqlite3``.
        projection_failure_hook: Optional test/integration hook called after a
            journal fsync and before projection.  Raising from the hook leaves
            a recoverable canonical transaction, modelling a process crash.
    """

    JOURNAL_FILENAME = "audit.ndjson"
    PROJECTION_FILENAME = "audit.sqlite3"
    LOCK_FILENAME = "audit.lock"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        root: str | Path,
        *,
        projection_failure_hook: Callable[[Mapping[str, Any]], None] | None = None,
        fail_after_journal_once: bool = False,
        recover_incomplete: bool = True,
    ) -> None:
        """Open or create a canonical store at ``root``.

        Args:
            root: Directory containing canonical and projected artifacts.
            projection_failure_hook: Optional fault-injection callback.
            fail_after_journal_once: Simulate one crash before projection.
            recover_incomplete: Recover a partial final journal line when true.
        """

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.canonical_path = self.root / self.JOURNAL_FILENAME
        self.projection_path = self.root / self.PROJECTION_FILENAME
        self.lock_path = self.root / self.LOCK_FILENAME
        self._projection_failure_hook = projection_failure_hook
        self._fail_after_journal_once = fail_after_journal_once
        self._recover_incomplete = recover_incomplete
        self._closed = False
        self._ensure_header()
        self._connection = self._open_connection()
        with _PathLock(self.lock_path):
            self._read_journal(recover=self._recover_incomplete)
            self._ensure_projection_locked()

    def __enter__(self) -> AuditStore:
        """Return this store for context-manager use.

        Returns:
            The open store.
        """

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close the projection connection at context exit."""

        self.close()

    @property
    def journal_path(self) -> Path:
        """Alias for the canonical NDJSON path."""

        return self.canonical_path

    @property
    def sqlite_path(self) -> Path:
        """Alias for the rebuildable projection path."""

        return self.projection_path

    def close(self) -> None:
        """Close the SQLite projection connection."""

        if not self._closed:
            self._connection.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise AuditStoreError("audit store is closed")

    def _ensure_header(self) -> None:
        if self.canonical_path.exists() and self.canonical_path.stat().st_size:
            return
        header = {
            "schema_version": AUDIT_JOURNAL_SCHEMA_VERSION,
            "kind": "header",
            "store_id": uuid.uuid4().hex,
            "created_at": _now(),
        }
        self.canonical_path.parent.mkdir(parents=True, exist_ok=True)
        with _PathLock(self.lock_path):
            if self.canonical_path.exists() and self.canonical_path.stat().st_size:
                return
            with self.canonical_path.open("wb") as handle:
                data = (canonical_json(header) + "\n").encode("utf-8")
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.projection_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                record_type TEXT NOT NULL,
                revision INTEGER NOT NULL,
                deleted INTEGER NOT NULL,
                payload TEXT,
                operation_id TEXT NOT NULL,
                committed_at TEXT NOT NULL,
                global_revision INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS operations (
                operation_id TEXT PRIMARY KEY,
                request_digest TEXT NOT NULL,
                result TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS records_type_idx ON records(record_type);
            """
        )
        connection.commit()
        return connection

    def _read_journal(  # noqa: C901, PLR0912, PLR0915
        self, *, recover: bool = False
    ) -> tuple[dict[str, Any], list[_JournalTransaction], int, bytes]:
        """Read canonical transactions and truncate only an incomplete tail.

        Returns:
            Header, parsed transactions, valid byte offset, and raw bytes.
        """

        try:
            raw = self.canonical_path.read_bytes()
        except OSError as exc:
            raise AuditCorruptionError(f"cannot read canonical journal: {exc}") from exc
        if not raw:
            raise AuditCorruptionError("canonical journal is empty")
        header: dict[str, Any] | None = None
        transactions: list[_JournalTransaction] = []
        operation_ids: set[str] = set()
        latest_revisions: dict[str, int] = {}
        latest_record_types: dict[str, str] = {}
        proposal_origins: dict[str, dict[str, Any]] = {}
        valid_offset = 0
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines):
            is_complete = line.endswith(b"\n")
            content = line[:-1] if is_complete else line
            try:
                payload = json.loads(content.decode("utf-8"), parse_constant=_reject_json_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, AuditExportError) as exc:
                if index == len(lines) - 1 and not is_complete and recover:
                    break
                raise AuditCorruptionError(f"invalid canonical line {index + 1}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise AuditCorruptionError(f"canonical line {index + 1} is not an object")
            if index == 0:
                if (
                    payload.get("kind") != "header"
                    or payload.get("schema_version") != AUDIT_JOURNAL_SCHEMA_VERSION
                ):
                    raise AuditCorruptionError("canonical journal header is invalid")
                unknown_header = set(payload) - {
                    "schema_version",
                    "kind",
                    "store_id",
                    "created_at",
                }
                if unknown_header:
                    raise AuditCorruptionError(
                        "canonical journal header contains unknown fields: "
                        + ", ".join(sorted(unknown_header))
                    )
                header = dict(payload)
            else:
                try:
                    transaction = self._transaction_from_dict(payload)
                except (KeyError, TypeError, ValueError, AuditStoreError) as exc:
                    raise AuditCorruptionError(
                        f"invalid canonical transaction {index + 1}: {exc}"
                    ) from exc
                if transaction.operation_id in operation_ids:
                    raise AuditCorruptionError(
                        f"duplicate operation ID in canonical journal: {transaction.operation_id}"
                    )
                operation_ids.add(transaction.operation_id)
                for change in transaction.changes:
                    previous_revision = latest_revisions.get(change.record_id)
                    if previous_revision is not None and change.revision <= previous_revision:
                        raise AuditCorruptionError(
                            "canonical record revisions are not increasing for "
                            f"{change.record_id!r}"
                        )
                    latest_revisions[change.record_id] = change.revision
                    previous_type = latest_record_types.get(change.record_id)
                    if previous_type is not None and previous_type != change.record_type:
                        raise AuditCorruptionError(
                            "canonical record type changed for "
                            f"{change.record_id!r}: {previous_type!r} -> "
                            f"{change.record_type!r}"
                        )
                    if not change.deleted:
                        latest_record_types[change.record_id] = change.record_type
                    elif previous_type is None and change.record_type != "unknown":
                        raise AuditCorruptionError(
                            "first canonical tombstone must use record type 'unknown' for "
                            f"{change.record_id!r}"
                        )
                    try:
                        self._validate_proposal_origin(
                            proposal_origins,
                            change.record_id,
                            change.record,
                        )
                    except AuditStoreError as exc:
                        raise AuditCorruptionError(
                            f"invalid detector rule proposal history at canonical line "
                            f"{index + 1}: {exc}"
                        ) from exc
                transactions.append(transaction)
            valid_offset += len(line)
            if index == len(lines) - 1 and not is_complete and recover:
                # A crash can finish the JSON bytes before the final newline.
                # Complete that record rather than concatenating the next
                # transaction onto it.
                with self.canonical_path.open("ab") as handle:
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                raw += b"\n"
                valid_offset += 1
        if valid_offset < len(raw) and recover:
            with self.canonical_path.open("r+b") as handle:
                handle.truncate(valid_offset)
                handle.flush()
                os.fsync(handle.fileno())
            raw = raw[:valid_offset]
        if header is None:
            raise AuditCorruptionError("canonical journal has no header")
        previous = 0
        for transaction in transactions:
            if transaction.global_revision <= previous:
                raise AuditCorruptionError("canonical transaction revisions are not increasing")
            previous = transaction.global_revision
        return header, transactions, valid_offset, raw

    def _transaction_from_dict(  # noqa: C901, PLR0912, PLR0915
        self, payload: Mapping[str, Any]
    ) -> _JournalTransaction:
        unknown = set(payload) - {
            "schema_version",
            "kind",
            "operation_id",
            "global_revision",
            "committed_at",
            "expected_revisions",
            "changes",
            "request_digest",
            "actor",
            "unconditional",
        }
        if unknown:
            raise AuditStoreError(
                "transaction contains unknown fields: " + ", ".join(sorted(unknown))
            )
        if payload.get("kind") != "transaction":
            raise AuditStoreError("unexpected canonical entry kind")
        if payload.get("schema_version") != AUDIT_JOURNAL_SCHEMA_VERSION:
            raise AuditStoreError("unsupported journal transaction schema")
        operation_id = payload.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise AuditStoreError("transaction operation_id is required")
        actor = payload.get("actor", {})
        if not isinstance(actor, Mapping):
            raise AuditStoreError("actor must be a mapping")
        actor = self._actor(actor)
        unconditional = payload.get("unconditional", False)
        if not isinstance(unconditional, bool):
            raise AuditStoreError("transaction unconditional flag is invalid")
        changes: list[_Change] = []
        raw_changes = payload.get("changes")
        if (
            not isinstance(raw_changes, Sequence)
            or isinstance(raw_changes, (str, bytes))
            or not raw_changes
        ):
            raise AuditStoreError("transaction changes are required")
        seen_record_ids: set[str] = set()
        for raw_change in raw_changes:
            if not isinstance(raw_change, Mapping):
                raise AuditStoreError("transaction change is not an object")
            unknown_change = set(raw_change) - {
                "record_id",
                "record_type",
                "revision",
                "deleted",
                "record",
            }
            if unknown_change:
                raise AuditStoreError(
                    "transaction change contains unknown fields: "
                    + ", ".join(sorted(unknown_change))
                )
            rid = raw_change.get("record_id")
            kind = raw_change.get("record_type")
            revision = raw_change.get("revision")
            deleted = raw_change.get("deleted")
            if not isinstance(rid, str) or not rid.strip() or not isinstance(kind, str):
                raise AuditStoreError("transaction change identity is invalid")
            if rid in seen_record_ids:
                raise AuditStoreError(f"duplicate record ID in transaction: {rid}")
            seen_record_ids.add(rid)
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision < 1
                or not isinstance(deleted, bool)
            ):
                raise AuditStoreError("transaction change revision/deleted fields are invalid")
            record_payload = raw_change.get("record")
            if deleted:
                if record_payload is not None:
                    raise AuditStoreError("tombstones cannot carry record payloads")
            else:
                if not isinstance(record_payload, Mapping):
                    raise AuditStoreError("live changes require record payloads")
                try:
                    typed = record_from_dict(record_payload)
                except (AuditContractError, KeyError, TypeError, ValueError) as exc:
                    raise AuditStoreError(f"invalid record payload: {exc}") from exc
                if record_id(typed) != rid or record_type(typed) != kind:
                    raise AuditStoreError("record payload identity disagrees with transaction")
                self._validate_record_actor(typed, actor["kind"])
                record_payload = record_to_dict(typed)
            changes.append(
                _Change(
                    record_id=rid,
                    record_type=kind,
                    revision=revision,
                    deleted=deleted,
                    record=dict(record_payload) if record_payload is not None else None,
                )
            )
        expected = payload.get("expected_revisions", {})
        if not isinstance(expected, Mapping):
            raise AuditStoreError("expected_revisions must be a mapping")
        expected_revisions: dict[str, int | None] = {}
        for key, value in expected.items():
            if (
                not isinstance(key, str)
                or not key.strip()
                or (
                    value is not None
                    and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
                )
            ):
                raise AuditStoreError("expected revision is invalid")
            expected_revisions[key] = value
        global_revision = payload.get("global_revision")
        if (
            isinstance(global_revision, bool)
            or not isinstance(global_revision, int)
            or global_revision < 1
        ):
            raise AuditStoreError("global_revision is invalid")
        committed_at = payload.get("committed_at", "")
        if not isinstance(committed_at, str) or not committed_at:
            raise AuditStoreError("committed_at is invalid")
        request_digest = payload.get("request_digest")
        if (
            not isinstance(request_digest, str)
            or len(request_digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in request_digest)
        ):
            raise AuditStoreError("request_digest is invalid")
        expected_digest = self._request_digest(
            operation_id,
            tuple(
                (
                    change.record_id,
                    "tombstone" if change.deleted else change.record_type,
                    change.record,
                    change.deleted,
                )
                for change in changes
            ),
            expected_revisions,
            actor,
            unconditional=unconditional,
        )
        if request_digest.lower() != expected_digest and "unconditional" not in payload:
            # The first BA-03 journal omitted the explicit flag while still
            # including it in the request digest.  Accept that historical
            # force-save encoding and normalize it on the next write.
            legacy_digest = self._request_digest(
                operation_id,
                tuple(
                    (
                        change.record_id,
                        "tombstone" if change.deleted else change.record_type,
                        change.record,
                        change.deleted,
                    )
                    for change in changes
                ),
                expected_revisions,
                actor,
                unconditional=True,
            )
            if request_digest.lower() == legacy_digest:
                unconditional = True
                expected_digest = legacy_digest
        if request_digest.lower() != expected_digest:
            raise AuditStoreError("transaction request_digest does not match canonical payload")
        return _JournalTransaction(
            operation_id=operation_id,
            global_revision=global_revision,
            committed_at=committed_at,
            expected_revisions=expected_revisions,
            changes=tuple(changes),
            request_digest=request_digest.lower(),
            actor=actor,
            unconditional=unconditional,
        )

    def _state(
        self, transactions: Sequence[_JournalTransaction]
    ) -> tuple[
        dict[str, StoredRecord], dict[str, list[StoredRecord]], dict[str, _JournalTransaction]
    ]:
        state: dict[str, StoredRecord] = {}
        history: dict[str, list[StoredRecord]] = {}
        operations: dict[str, _JournalTransaction] = {}
        for transaction in transactions:
            operations[transaction.operation_id] = transaction
            for change in transaction.changes:
                stored = StoredRecord(
                    record_id=change.record_id,
                    record_type=change.record_type,
                    revision=change.revision,
                    deleted=change.deleted,
                    record=(record_from_dict(change.record) if change.record is not None else None),
                    operation_id=transaction.operation_id,
                    committed_at=transaction.committed_at,
                    global_revision=transaction.global_revision,
                )
                state[change.record_id] = stored
                history.setdefault(change.record_id, []).append(stored)
        return state, history, operations

    def _checkpoint_locked(self) -> ProjectionCheckpoint:
        row = self._connection.execute(
            "SELECT key, value FROM metadata WHERE key IN ('journal_revision', 'journal_offset', 'journal_digest')"
        ).fetchall()
        values = {item["key"]: item["value"] for item in row}
        return ProjectionCheckpoint(
            revision=int(values.get("journal_revision", "0")),
            offset=int(values.get("journal_offset", "0")),
            digest=values.get("journal_digest", ""),
        )

    def checkpoint(self) -> ProjectionCheckpoint:
        """Return the SQLite projection checkpoint."""

        self._ensure_open()
        with _PathLock(self.lock_path):
            self._ensure_projection_locked()
            return self._checkpoint_locked()

    def _ensure_projection_locked(self) -> None:
        # A caller may intentionally remove the disposable index while this
        # process is alive.  SQLite keeps the unlinked inode open, so reopen
        # before rebuilding or reads would continue to see the old projection.
        if not self.projection_path.exists():
            self._connection.close()
            self._connection = self._open_connection()
        _header, transactions, offset, raw = self._read_journal(recover=True)
        expected_revision = transactions[-1].global_revision if transactions else 0
        expected_digest = _digest_bytes(raw)
        checkpoint = self._checkpoint_locked()
        if (
            checkpoint.revision != expected_revision
            or checkpoint.offset != offset
            or checkpoint.digest != expected_digest
        ):
            self._rebuild_projection_locked(transactions, offset, expected_digest)

    def _rebuild_projection_locked(
        self,
        transactions: Sequence[_JournalTransaction],
        offset: int,
        digest: str,
    ) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute("DELETE FROM records")
            self._connection.execute("DELETE FROM operations")
            self._apply_transactions_locked(transactions)
            revision = transactions[-1].global_revision if transactions else 0
            self._set_metadata_locked(
                journal_revision=str(revision), journal_offset=str(offset), journal_digest=digest
            )
            self._connection.commit()
        except Exception as exc:
            self._connection.rollback()
            raise ProjectionError(f"cannot rebuild SQLite projection: {exc}") from exc

    def _set_metadata_locked(self, **values: str) -> None:
        for key, value in values.items():
            self._connection.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def _apply_transactions_locked(self, transactions: Sequence[_JournalTransaction]) -> None:
        for transaction in transactions:
            for change in transaction.changes:
                payload = canonical_json(change.record) if change.record is not None else None
                self._connection.execute(
                    "INSERT INTO records(record_id, record_type, revision, deleted, payload, operation_id, committed_at, global_revision) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(record_id) DO UPDATE SET record_type=excluded.record_type, revision=excluded.revision, "
                    "deleted=excluded.deleted, payload=excluded.payload, operation_id=excluded.operation_id, "
                    "committed_at=excluded.committed_at, global_revision=excluded.global_revision",
                    (
                        change.record_id,
                        change.record_type,
                        change.revision,
                        int(change.deleted),
                        payload,
                        transaction.operation_id,
                        transaction.committed_at,
                        transaction.global_revision,
                    ),
                )
            result = {
                "operation_id": transaction.operation_id,
                "global_revision": transaction.global_revision,
                "changes": [
                    {
                        "record_id": change.record_id,
                        "record_type": change.record_type,
                        "revision": change.revision,
                        "deleted": change.deleted,
                    }
                    for change in transaction.changes
                ],
            }
            self._connection.execute(
                "INSERT OR REPLACE INTO operations(operation_id, request_digest, result) VALUES(?, ?, ?)",
                (transaction.operation_id, transaction.request_digest, canonical_json(result)),
            )

    def _project_transactions_locked(
        self,
        transactions: Sequence[_JournalTransaction],
        offset: int,
        raw: bytes,
    ) -> None:
        if self._projection_failure_hook is not None:
            self._projection_failure_hook(transactions[-1].to_dict() if transactions else {})
        if self._fail_after_journal_once:
            self._fail_after_journal_once = False
            raise ProjectionError("simulated crash after canonical journal commit")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute("DELETE FROM records")
            self._connection.execute("DELETE FROM operations")
            self._apply_transactions_locked(transactions)
            revision = transactions[-1].global_revision if transactions else 0
            self._set_metadata_locked(
                journal_revision=str(revision),
                journal_offset=str(offset),
                journal_digest=_digest_bytes(raw),
            )
            self._connection.commit()
        except ProjectionError:
            self._connection.rollback()
            raise
        except Exception as exc:
            self._connection.rollback()
            raise ProjectionError(f"cannot update SQLite projection: {exc}") from exc

    def _append_transaction_locked(
        self, transaction: _JournalTransaction
    ) -> tuple[list[_JournalTransaction], int, bytes]:
        with self.canonical_path.open("ab") as handle:
            data = (canonical_json(transaction.to_dict()) + "\n").encode("utf-8")
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _header, transactions, offset, raw = self._read_journal(recover=False)
        return transactions, offset, raw

    @staticmethod
    def _validate_identifier(value: Any, *, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AuditStoreError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _validate_revision(value: Any, *, name: str) -> int | None:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise AuditStoreError(f"{name} must be a non-negative integer or null")
        return value

    @classmethod
    def _validate_expected_revisions(
        cls, value: Mapping[str, int | None], *, name: str = "expected_revisions"
    ) -> dict[str, int | None]:
        if not isinstance(value, Mapping):
            raise AuditStoreError(f"{name} must be a mapping")
        result: dict[str, int | None] = {}
        for key, revision in value.items():
            cls._validate_identifier(key, name=f"{name} key")
            result[key] = cls._validate_revision(revision, name=f"{name}[{key!r}]")
        return result

    @staticmethod
    def _actor(actor: str | Mapping[str, Any], actor_id: str = "") -> dict[str, Any]:
        if isinstance(actor, Mapping):
            result = dict(actor)
        else:
            result = {"kind": actor, "id": actor_id}
        if result.get("kind") not in {"human", "detector", "agent"}:
            raise AuditStoreError("actor kind must be human, detector, or agent")
        if "id" in result and not isinstance(result["id"], str):
            raise AuditStoreError("actor id must be a string")
        return result

    @staticmethod
    def _validate_record_actor(record: Any, actor_kind: str) -> None:
        """Require the transaction actor to equal any declared record actor."""

        for field_name in ("author_kind", "actor_kind"):
            declared_kind = getattr(record, field_name, None)
            if declared_kind is not None and actor_kind != declared_kind:
                raise AuditStoreError(f"transaction actor kind does not match record {field_name}")

    @staticmethod
    def _proposal_origin_payload(record: Any) -> dict[str, Any] | None:
        """Return the immutable portion of a detector-rule proposal."""

        if isinstance(record, Mapping):
            if record.get("record_type") != "detector_rule_proposal":
                return None
            payload = record
        else:
            try:
                if record_type(record) != "detector_rule_proposal":
                    return None
            except AuditContractError:
                return None
            payload = record_to_dict(record)
        return {
            field_name: payload.get(field_name)
            for field_name in DETECTOR_RULE_PROPOSAL_IMMUTABLE_FIELDS
        }

    @classmethod
    def _validate_proposal_origin(
        cls,
        origins: dict[str, dict[str, Any]],
        rid: str,
        record: Any,
    ) -> None:
        """Reject changes to proposal origin fields across every journal revision."""

        candidate = cls._proposal_origin_payload(record)
        if candidate is None:
            return
        previous = origins.get(rid)
        if previous is None:
            origins[rid] = candidate
            return
        if canonical_json(previous) == canonical_json(candidate):
            return
        changed = [
            field_name
            for field_name in DETECTOR_RULE_PROPOSAL_IMMUTABLE_FIELDS
            if canonical_json(previous.get(field_name)) != canonical_json(candidate.get(field_name))
        ]
        raise AuditStoreError(
            "detector rule proposal immutable fields changed: " + ", ".join(changed)
        )

    @staticmethod
    def _request_digest(
        operation_id: str,
        changes: Sequence[tuple[str, str, dict[str, Any] | None, bool]],
        expected: Mapping[str, int | None],
        actor: Mapping[str, Any],
        *,
        unconditional: bool = False,
    ) -> str:
        payload = {
            "operation_id": operation_id,
            "changes": [
                {
                    "record_id": rid,
                    "record_type": kind,
                    "record": record,
                    "deleted": deleted,
                }
                for rid, kind, record, deleted in changes
            ],
            "expected_revisions": dict(expected),
            "actor": dict(actor),
            "unconditional": unconditional,
        }
        return _digest_bytes(canonical_json(payload).encode("utf-8"))

    def commit(
        self,
        records: Sequence[Any],
        *,
        operation_id: str,
        expected_revisions: Mapping[str, int | None] | None = None,
        expected_revision: int | None = None,
        actor: str | Mapping[str, Any] = "human",
        actor_id: str = "",
    ) -> CommitResult | BatchCommitResult:
        """Commit records using compare-and-swap semantics.

        Returns:
            A scalar or batch commit receipt.
        """

        return self._commit(
            records,
            operation_id=operation_id,
            expected_revisions=expected_revisions,
            expected_revision=expected_revision,
            actor=actor,
            actor_id=actor_id,
        )

    def _commit(  # noqa: C901, PLR0912
        self,
        records: Sequence[Any],
        *,
        operation_id: str,
        expected_revisions: Mapping[str, int | None] | None = None,
        expected_revision: int | None = None,
        actor: str | Mapping[str, Any] = "human",
        actor_id: str = "",
        _allow_unconditional: bool = False,
    ) -> CommitResult | BatchCommitResult:
        """Commit one or more records as one serialized transaction.

        Returns:
            A scalar receipt for one record or a batch receipt for many.
        """

        self._ensure_open()
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise AuditStoreError("operation_id must be a non-empty string")
        record_list = list(records)
        if not record_list:
            raise AuditStoreError("at least one record is required")
        actor_payload = self._actor(actor, actor_id)
        if expected_revisions is None:
            expected_revisions = {}
        else:
            expected_revisions = self._validate_expected_revisions(expected_revisions)
        if expected_revision is not None:
            if len(record_list) != 1:
                raise AuditStoreError("expected_revision is only valid for a single-record commit")
            self._validate_revision(expected_revision, name="expected_revision")
            expected_revisions[record_id(record_list[0])] = expected_revision

        changes_input: list[tuple[str, str, dict[str, Any] | None, bool]] = []
        for record in record_list:
            try:
                rid = record_id(record)
                kind = record_type(record)
                payload = record_to_dict(record)
                # Reconstruct and reserialize every payload before computing
                # the request digest.  This closes the boundary for callers
                # that mutate a frozen dataclass's nested mapping or pass a
                # value whose constructor did not normalize it.
                canonical_record = record_from_dict(payload)
                payload = record_to_dict(canonical_record)
            except (AuditContractError, TypeError, ValueError) as exc:
                raise AuditStoreError(f"record cannot be committed: {exc}") from exc
            self._validate_record_actor(canonical_record, actor_payload["kind"])
            changes_input.append((rid, kind, payload, False))
        request_digest = self._request_digest(
            operation_id,
            changes_input,
            expected_revisions,
            actor_payload,
            unconditional=_allow_unconditional,
        )
        with _PathLock(self.lock_path):
            _header, transactions, _offset, _raw = self._read_journal(recover=True)
            _state, _history, operations = self._state(transactions)
            for rid, kind, payload, _deleted in changes_input:
                previous_type = next(
                    (
                        stored.record_type
                        for stored in reversed(_history.get(rid, ()))
                        if not stored.deleted
                    ),
                    None,
                )
                if previous_type is not None and previous_type != kind:
                    raise AuditStoreError(
                        f"record type is immutable for {rid!r}: {previous_type!r} -> {kind!r}"
                    )
                if kind != "detector_rule_proposal":
                    continue
                origins: dict[str, dict[str, Any]] = {}
                for stored in _history.get(rid, ()):
                    self._validate_proposal_origin(origins, rid, stored.record)
                self._validate_proposal_origin(origins, rid, payload)
            previous_operation = operations.get(operation_id)
            if previous_operation is not None:
                if previous_operation.request_digest != request_digest:
                    raise OperationConflictError(
                        f"operation ID {operation_id!r} was already used for a different request"
                    )
                # A previous call may have synced the journal and then lost
                # the process before projection.  Make a replay self-healing
                # before acknowledging the idempotent retry.
                self._ensure_projection_locked()
                return self._commit_result_from_transaction(previous_operation, replayed=True)
            ids = [rid for rid, _kind, _payload, _deleted in changes_input]
            if len(ids) != len(set(ids)):
                raise AuditStoreError("one transaction cannot contain duplicate record IDs")
            for rid, _kind, _payload, _deleted in changes_input:
                current = _state.get(rid)
                current_revision = current.revision if current is not None else 0
                expected = expected_revisions.get(rid)
                if (
                    current is not None
                    and not _allow_unconditional
                    and (rid not in expected_revisions or expected is None)
                ):
                    raise ExpectedRevisionRequiredError(rid, None, current_revision)
                if expected is not None and expected != current_revision:
                    raise AuditConflictError(rid, expected, current_revision)
            global_revision = transactions[-1].global_revision + 1 if transactions else 1
            changes = tuple(
                _Change(
                    record_id=rid,
                    record_type=kind,
                    revision=(_state[rid].revision + 1 if rid in _state else 1),
                    deleted=False,
                    record=payload,
                )
                for rid, kind, payload, _deleted in changes_input
            )
            transaction = _JournalTransaction(
                operation_id=operation_id,
                global_revision=global_revision,
                committed_at=_now(),
                expected_revisions=expected_revisions,
                changes=changes,
                request_digest=request_digest,
                actor=actor_payload,
                unconditional=_allow_unconditional,
            )
            updated, offset, raw = self._append_transaction_locked(transaction)
            self._project_transactions_locked(updated, offset, raw)
            return self._commit_result_from_transaction(transaction)

    def save(
        self,
        record: Any,
        *,
        operation_id: str,
        expected_revision: int | None = None,
        actor: str | Mapping[str, Any] = "human",
        actor_id: str = "",
    ) -> CommitResult:
        """Persist one record with an optional compare-and-swap revision.

        Returns:
            Durable commit receipt.
        """

        result = self.commit(
            [record],
            operation_id=operation_id,
            expected_revision=expected_revision,
            actor=actor,
            actor_id=actor_id,
        )
        if not isinstance(result, CommitResult):  # pragma: no cover - one change always scalar.
            raise AuditStoreError("single-record commit returned a batch receipt")
        return result

    put = save
    append = save
    save_record = save

    def force_save(
        self,
        record: Any,
        *,
        operation_id: str,
        actor: str | Mapping[str, Any] = "human",
        actor_id: str = "",
    ) -> CommitResult:
        """Persist a record without CAS only through an explicit force operation.

        Returns:
            The durable commit receipt.
        """

        result = self._commit(
            [record],
            operation_id=operation_id,
            actor=actor,
            actor_id=actor_id,
            _allow_unconditional=True,
        )
        if not isinstance(result, CommitResult):  # pragma: no cover - one change is scalar.
            raise AuditStoreError("force_save returned a batch receipt")
        return result

    force_update = force_save

    def _commit_result_from_transaction(
        self, transaction: _JournalTransaction, *, replayed: bool = False
    ) -> CommitResult | BatchCommitResult:
        results = tuple(
            CommitResult(
                operation_id=transaction.operation_id,
                record_id=change.record_id,
                record_type=change.record_type,
                revision=change.revision,
                global_revision=transaction.global_revision,
                replayed=replayed,
                deleted=change.deleted,
                checkpoint_revision=transaction.global_revision,
            )
            for change in transaction.changes
        )
        if len(results) == 1:
            return results[0]
        return BatchCommitResult(
            operation_id=transaction.operation_id,
            changes=results,
            global_revision=transaction.global_revision,
            replayed=replayed,
        )

    def _ensure_projection(self) -> None:
        with _PathLock(self.lock_path):
            self._ensure_projection_locked()

    def get(self, rid: str, *, include_deleted: bool = False) -> StoredRecord | None:
        """Read a projected record, optionally including its latest tombstone.

        Returns:
            The latest projected row, or ``None`` when absent/deleted.
        """

        self._ensure_open()
        self._ensure_projection()
        row = self._connection.execute(
            "SELECT record_id, record_type, revision, deleted, payload, operation_id, committed_at, global_revision "
            "FROM records WHERE record_id = ?",
            (rid,),
        ).fetchone()
        if row is None or (row["deleted"] and not include_deleted):
            return None
        payload = json.loads(row["payload"]) if row["payload"] is not None else None
        return StoredRecord(
            record_id=row["record_id"],
            record_type=row["record_type"],
            revision=int(row["revision"]),
            deleted=bool(row["deleted"]),
            record=(record_from_dict(payload) if payload is not None else None),
            operation_id=row["operation_id"],
            committed_at=row["committed_at"],
            global_revision=int(row["global_revision"]),
        )

    load = get
    get_record = get

    def get_revision(self, rid: str) -> int:
        """Return the latest revision or zero for an unseen ID."""

        stored = self.get(rid, include_deleted=True)
        return stored.revision if stored is not None else 0

    def list_records(
        self, *, record_type: str | None = None, include_deleted: bool = False
    ) -> list[StoredRecord]:
        """List the latest projected records in deterministic ID order.

        Returns:
            Stored rows matching the optional type/deletion filters.
        """

        self._ensure_open()
        self._ensure_projection()
        query = (
            "SELECT record_id FROM records WHERE deleted = 0 ORDER BY record_id"
            if not include_deleted
            else "SELECT record_id FROM records ORDER BY record_id"
        )
        ids = [row["record_id"] for row in self._connection.execute(query)]
        result = [
            item
            for rid in ids
            if (item := self.get(rid, include_deleted=include_deleted)) is not None
        ]
        if record_type is not None:
            result = [item for item in result if item.record_type == record_type]
        return result

    records = list_records

    def history(self, rid: str) -> list[StoredRecord]:
        """Return all revisions, including tombstones, in commit order."""

        self._ensure_open()
        with _PathLock(self.lock_path):
            _header, transactions, _offset, _raw = self._read_journal(recover=True)
            _state, history, _operations = self._state(transactions)
            return list(history.get(rid, ()))

    revisions = history

    def delete(
        self,
        rid: str,
        *,
        operation_id: str,
        expected_revision: int | None = None,
        actor: str | Mapping[str, Any] = "human",
        actor_id: str = "",
    ) -> CommitResult:
        """Append a tombstone without erasing the prior record history.

        Returns:
            Durable tombstone receipt.
        """

        self._validate_identifier(rid, name="record_id")
        self._validate_identifier(operation_id, name="operation_id")
        self._validate_revision(expected_revision, name="expected_revision")
        self._ensure_open()
        with _PathLock(self.lock_path):
            _header, transactions, _offset, _raw = self._read_journal(recover=True)
            state, _history, operations = self._state(transactions)
            actor_payload = self._actor(actor, actor_id)
            expected = {} if expected_revision is None else {rid: expected_revision}
            changes_input = [(rid, "tombstone", None, True)]
            request_digest = self._request_digest(
                operation_id, changes_input, expected, actor_payload
            )
            previous_operation = operations.get(operation_id)
            if previous_operation is not None:
                if previous_operation.request_digest != request_digest:
                    raise OperationConflictError(
                        f"operation ID {operation_id!r} was already used for a different request"
                    )
                self._ensure_projection_locked()
                result = self._commit_result_from_transaction(previous_operation, replayed=True)
                if not isinstance(result, CommitResult):
                    raise AuditStoreError("delete replay returned a batch receipt")
                return result
            current = state.get(rid)
            current_revision = current.revision if current is not None else 0
            if current is not None and expected_revision is None:
                raise ExpectedRevisionRequiredError(rid, None, current_revision)
            if expected_revision is not None and expected_revision != current_revision:
                raise AuditConflictError(rid, expected_revision, current_revision)
            transaction = _JournalTransaction(
                operation_id=operation_id,
                global_revision=transactions[-1].global_revision + 1 if transactions else 1,
                committed_at=_now(),
                expected_revisions=expected,
                changes=(
                    _Change(
                        record_id=rid,
                        record_type=current.record_type if current is not None else "unknown",
                        revision=current_revision + 1,
                        deleted=True,
                        record=None,
                    ),
                ),
                request_digest=request_digest,
                actor=actor_payload,
                unconditional=False,
            )
            updated, offset, raw = self._append_transaction_locked(transaction)
            self._project_transactions_locked(updated, offset, raw)
            result = self._commit_result_from_transaction(transaction)
            if not isinstance(result, CommitResult):
                raise AuditStoreError("delete returned a batch receipt")
            return result

    tombstone = delete
    delete_record = delete

    def undo(
        self,
        rid: str,
        *,
        operation_id: str,
        expected_revision: int | None = None,
        actor: str | Mapping[str, Any] = "human",
        actor_id: str = "",
    ) -> CommitResult:
        """Restore the immediately preceding revision as a new revision.

        Returns:
            Durable undo receipt.
        """

        revisions = self.history(rid)
        if not revisions:
            raise AuditStoreError(f"cannot undo unknown record {rid!r}")
        current = revisions[-1]
        if expected_revision is not None and expected_revision != current.revision:
            raise AuditConflictError(rid, expected_revision, current.revision)
        previous = revisions[-2] if len(revisions) >= 2 else None
        if previous is None or previous.deleted or previous.record is None:
            return self.delete(
                rid,
                operation_id=operation_id,
                expected_revision=current.revision,
                actor=actor,
                actor_id=actor_id,
            )
        return self.save(
            previous.record,
            operation_id=operation_id,
            expected_revision=current.revision,
            actor=actor,
            actor_id=actor_id,
        )

    undo_record = undo

    def rebuild_projection(self) -> ProjectionCheckpoint:
        """Delete/rebuild SQLite from canonical NDJSON.

        Returns:
            The new projection checkpoint.
        """

        self._ensure_open()
        with _PathLock(self.lock_path):
            if not self.projection_path.exists():
                self._connection.close()
                self._connection = self._open_connection()
            _header, transactions, offset, raw = self._read_journal(recover=True)
            self._rebuild_projection_locked(transactions, offset, _digest_bytes(raw))
            return self._checkpoint_locked()

    rebuild_index = rebuild_projection

    def export(  # noqa: C901, PLR0912, PLR0915
        self,
        destination: str | Path,
        *,
        include_projection: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """Create a portable directory or zip backup of committed audit data.

        Existing destinations require explicit ``overwrite=True``.  The
        complete artifact is staged beside the destination and swapped into
        place only after all canonical bytes have been written.

        Returns:
            The created directory or archive path.
        """

        self._ensure_open()
        destination = Path(destination)
        self._validate_export_destination(destination)
        if destination.exists() and not overwrite:
            raise AuditExportError(f"export destination exists; pass overwrite=True: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.is_symlink():
            raise AuditExportError("export destination must not be a symbolic link")
        if (
            destination.exists()
            and destination.suffix.lower() == ".zip"
            and not destination.is_file()
        ):
            raise AuditExportError("zip export destination is not a regular file")
        if (
            destination.exists()
            and destination.suffix.lower() != ".zip"
            and not destination.is_dir()
        ):
            raise AuditExportError("directory export destination is not a directory")
        with _PathLock(self.lock_path):
            self._ensure_projection_locked()
            header, transactions, _offset, raw = self._read_journal(recover=True)
            self._assert_export_safe(transactions)
            manifest = self._manifest(header, transactions, raw)
            if destination.suffix.lower() == ".zip":
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.export-",
                    suffix=".tmp",
                    dir=destination.parent,
                )
                os.close(fd)
                temporary = Path(temporary_name)
                try:
                    with zipfile.ZipFile(
                        temporary, "w", compression=zipfile.ZIP_DEFLATED
                    ) as archive:
                        archive.writestr(self.JOURNAL_FILENAME, raw)
                        archive.writestr(self.MANIFEST_FILENAME, canonical_json(manifest) + "\n")
                        if include_projection:
                            archive.writestr(
                                self.PROJECTION_FILENAME, self.projection_path.read_bytes()
                            )
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                return destination
            staging = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.export-", dir=destination.parent)
            )
            old_destination: Path | None = None
            try:
                self._atomic_write(staging / self.JOURNAL_FILENAME, raw)
                self._atomic_write(
                    staging / self.MANIFEST_FILENAME,
                    (canonical_json(manifest) + "\n").encode(),
                )
                if include_projection:
                    shutil.copy2(self.projection_path, staging / self.PROJECTION_FILENAME)
                if destination.exists():
                    old_destination = Path(
                        tempfile.mkdtemp(prefix=f".{destination.name}.old-", dir=destination.parent)
                    )
                    old_destination.rmdir()
                    os.replace(destination, old_destination)
                try:
                    os.replace(staging, destination)
                    staging = Path()
                except Exception:
                    if old_destination is not None and old_destination.exists():
                        os.replace(old_destination, destination)
                        old_destination = None
                    raise
                if old_destination is not None and old_destination.exists():
                    shutil.rmtree(old_destination)
                    old_destination = None
            finally:
                if staging != Path() and staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
                if old_destination is not None and old_destination.exists():
                    if not destination.exists():
                        os.replace(old_destination, destination)
                    else:
                        shutil.rmtree(old_destination, ignore_errors=True)
            return destination

    def _validate_export_destination(self, destination: Path) -> None:
        """Reject any destination whose path overlaps the source store."""

        source_root = self.root.resolve()
        target = destination.resolve(strict=False)
        if target == source_root or source_root in target.parents or target in source_root.parents:
            raise AuditExportError(
                "export destination overlaps the source store; choose a disjoint destination"
            )

    backup = export
    export_backup = export

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    @staticmethod
    def _manifest(
        header: Mapping[str, Any], transactions: Sequence[_JournalTransaction], raw: bytes
    ) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_EXPORT_SCHEMA_VERSION,
            "journal_schema_version": AUDIT_JOURNAL_SCHEMA_VERSION,
            "store_id": header.get("store_id", ""),
            "journal_digest": _digest_bytes(raw),
            "transaction_count": len(transactions),
            "global_revision": transactions[-1].global_revision if transactions else 0,
            "created_at": _now(),
            "portable": True,
            "raw_artifacts_included": False,
        }

    @staticmethod
    def _assert_export_safe(transactions: Sequence[_JournalTransaction]) -> None:
        """Fail closed for obvious credential fields in small export records."""

        sensitive = {"password", "passwd", "secret", "token", "access_token", "private_key"}

        def visit(value: Any, path: str = "") -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    key_text = str(key).lower()
                    if key_text in sensitive:
                        raise AuditExportError(f"export refused sensitive field at {path}/{key}")
                    visit(item, f"{path}/{key}")
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for index, item in enumerate(value):
                    visit(item, f"{path}/{index}")

        for transaction in transactions:
            visit(transaction.to_dict())

    @classmethod
    def restore(  # noqa: C901, PLR0912, PLR0915
        cls,
        backup: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> AuditStore:
        """Restore a portable backup into a new location and rebuild SQLite.

        Returns:
            An open store rooted at ``destination``.
        """

        backup_path = Path(backup)
        destination = Path(destination)
        if destination.exists() and not overwrite:
            raise AuditExportError(
                f"restore destination exists; pass --overwrite explicitly: {destination}"
            )
        if destination.exists() and not destination.is_dir():
            raise AuditExportError(f"restore destination is not a directory: {destination}")
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=parent))
        old_destination: Path | None = None
        staged_store: AuditStore | None = None
        try:
            if backup_path.is_dir():
                source_journal = backup_path / cls.JOURNAL_FILENAME
                source_manifest = backup_path / cls.MANIFEST_FILENAME
                if not source_journal.exists() or not source_manifest.exists():
                    raise AuditExportError(
                        "backup directory is missing canonical journal or manifest"
                    )
                journal_bytes = source_journal.read_bytes()
                manifest = json.loads(
                    source_manifest.read_text(encoding="utf-8"),
                    parse_constant=_reject_json_constant,
                )
            elif backup_path.is_file() and zipfile.is_zipfile(backup_path):
                with zipfile.ZipFile(backup_path) as archive:
                    names = set(archive.namelist())
                    if cls.JOURNAL_FILENAME not in names or cls.MANIFEST_FILENAME not in names:
                        raise AuditExportError(
                            "backup archive is missing canonical journal or manifest"
                        )
                    if any(name.startswith("/") or ".." in Path(name).parts for name in names):
                        raise AuditExportError("backup archive contains unsafe paths")
                    journal_bytes = archive.read(cls.JOURNAL_FILENAME)
                    manifest = json.loads(
                        archive.read(cls.MANIFEST_FILENAME), parse_constant=_reject_json_constant
                    )
            else:
                raise AuditExportError(f"unsupported backup path: {backup_path}")
            if (
                not isinstance(manifest, Mapping)
                or manifest.get("schema_version") != AUDIT_EXPORT_SCHEMA_VERSION
            ):
                raise AuditExportError("unsupported or malformed backup manifest")
            if manifest.get("journal_digest") != _digest_bytes(journal_bytes):
                raise AuditExportError("backup journal digest does not match manifest")

            # Validate and rebuild in a same-parent staging directory.  The
            # destination is untouched until this constructor has replayed the
            # complete journal and proved its SQLite checkpoint.
            cls._atomic_write(staging / cls.JOURNAL_FILENAME, journal_bytes)
            staged_store = cls(staging, recover_incomplete=False)
            _header, staged_transactions, _offset, _raw = staged_store._read_journal(recover=False)
            cls._assert_export_safe(staged_transactions)
            staged_store.close()
            staged_store = None

            if destination.exists():
                old_destination = Path(
                    tempfile.mkdtemp(prefix=f".{destination.name}.old-", dir=parent)
                )
                old_destination.rmdir()
                os.replace(destination, old_destination)
            try:
                os.replace(staging, destination)
            except Exception:
                if old_destination is not None and old_destination.exists():
                    os.replace(old_destination, destination)
                    old_destination = None
                raise
            staging = Path()
            try:
                restored = cls(destination)
            except Exception:
                if destination.exists():
                    shutil.rmtree(destination)
                if old_destination is not None and old_destination.exists():
                    os.replace(old_destination, destination)
                    old_destination = None
                raise
            if old_destination is not None and old_destination.exists():
                shutil.rmtree(old_destination)
                old_destination = None
            return restored
        except json.JSONDecodeError as exc:
            raise AuditExportError(f"backup manifest is invalid JSON: {exc}") from exc
        finally:
            if staged_store is not None:
                staged_store.close()
            if staging != Path() and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if old_destination is not None and old_destination.exists():
                # The only remaining old directory is a failed swap; restore
                # it before cleaning up a stale staging artifact.
                if not destination.exists():
                    os.replace(old_destination, destination)
                else:
                    shutil.rmtree(old_destination, ignore_errors=True)

    restore_backup = restore

    def migrate_schema(
        self,
        destination: str | Path,
        *,
        target_version: str = AUDIT_RECORD_SCHEMA_VERSION,
    ) -> Path:
        """Export a migrated canonical journal without changing this store.

        Returns:
            The migrated file, directory, or archive path.
        """

        if target_version != AUDIT_RECORD_SCHEMA_VERSION:
            raise AuditContractError(f"unsupported audit schema target: {target_version}")
        destination = Path(destination)
        with _PathLock(self.lock_path):
            header, transactions, _offset, _raw = self._read_journal(recover=True)
            migrated_lines = [
                {
                    "schema_version": AUDIT_JOURNAL_SCHEMA_VERSION,
                    "kind": "header",
                    "store_id": header.get("store_id", ""),
                    "created_at": header.get("created_at", _now()),
                }
            ]
            for transaction in transactions:
                migrated_lines.append(transaction.to_dict())
            if destination.suffix.lower() == ".zip" or destination.is_dir():
                temporary_root = Path(tempfile.mkdtemp(prefix="audit-migrate-"))
                try:
                    data = (
                        "\n".join(canonical_json(line) for line in migrated_lines).encode() + b"\n"
                    )
                    self._atomic_write(temporary_root / self.JOURNAL_FILENAME, data)
                    manifest = self._manifest(header, transactions, data)
                    self._atomic_write(
                        temporary_root / self.MANIFEST_FILENAME,
                        (canonical_json(manifest) + "\n").encode(),
                    )
                    if destination.suffix.lower() == ".zip":
                        return self._zip_directory(temporary_root, destination)
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        temporary_root / self.JOURNAL_FILENAME, destination / self.JOURNAL_FILENAME
                    )
                    shutil.copy2(
                        temporary_root / self.MANIFEST_FILENAME,
                        destination / self.MANIFEST_FILENAME,
                    )
                    return destination
                finally:
                    shutil.rmtree(temporary_root, ignore_errors=True)
            output = destination
            output.parent.mkdir(parents=True, exist_ok=True)
            data = "\n".join(canonical_json(line) for line in migrated_lines).encode() + b"\n"
            self._atomic_write(output, data)
            return output

    migrate = migrate_schema

    @classmethod
    def _zip_directory(cls, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for child in sorted(source.iterdir()):
                    archive.write(child, child.name)
            os.replace(temporary, destination)
            return destination
        finally:
            temporary.unlink(missing_ok=True)

    def journal(self) -> list[dict[str, Any]]:
        """Return parsed transaction mappings for diagnostics and replay tools."""

        with _PathLock(self.lock_path):
            _header, transactions, _offset, _raw = self._read_journal(recover=True)
            return [transaction.to_dict() for transaction in transactions]


def _checkpoint_dict(checkpoint: ProjectionCheckpoint) -> dict[str, Any]:
    return {
        "revision": checkpoint.revision,
        "offset": checkpoint.offset,
        "digest": checkpoint.digest,
    }


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and move portable BA-03 audit stores")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="report the current projection")
    inspect.add_argument("root", type=Path)
    inspect.add_argument("--include-deleted", action="store_true")

    rebuild = commands.add_parser("rebuild", help="rebuild SQLite from the canonical journal")
    rebuild.add_argument("root", type=Path)

    export = commands.add_parser("export", help="create a portable backup")
    export.add_argument("root", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument("--include-projection", action="store_true")
    export.add_argument("--overwrite", action="store_true")

    restore = commands.add_parser("restore", help="validate and restore a portable backup")
    restore.add_argument("backup", type=Path)
    restore.add_argument("destination", type=Path)
    restore.add_argument("--overwrite", action="store_true")
    return parser


def _cli_export(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.destination
    with AuditStore(args.root) as store:
        output = store.export(
            destination,
            include_projection=args.include_projection,
            overwrite=args.overwrite,
        )
    return {"backup": str(output), "include_projection": bool(args.include_projection)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run bounded offline audit-store commands with JSON output and exit codes.

    Returns:
        Zero on success, or two for a bounded contract/storage error.
    """

    parser = _cli_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            with AuditStore(args.root) as store:
                records = store.list_records(include_deleted=args.include_deleted)
                result = {
                    "root": str(args.root),
                    "checkpoint": _checkpoint_dict(store.checkpoint()),
                    "record_count": len(records),
                    "records": [
                        {
                            "record_id": item.record_id,
                            "record_type": item.record_type,
                            "revision": item.revision,
                            "deleted": item.deleted,
                        }
                        for item in records
                    ],
                }
        elif args.command == "rebuild":
            with AuditStore(args.root) as store:
                result = {"root": str(args.root), **_checkpoint_dict(store.rebuild_projection())}
        elif args.command == "export":
            result = _cli_export(args)
        elif args.command == "restore":
            if args.destination.exists() and not args.overwrite:
                raise AuditExportError(
                    f"restore destination exists; pass --overwrite explicitly: {args.destination}"
                )
            with AuditStore.restore(
                args.backup, args.destination, overwrite=args.overwrite
            ) as store:
                result = {
                    "root": str(args.destination),
                    "checkpoint": _checkpoint_dict(store.checkpoint()),
                }
        else:  # pragma: no cover - argparse enforces the command choices.
            raise AuditStoreError(f"unknown command: {args.command}")
    except (AuditContractError, AuditStoreError, OSError, ValueError) as exc:
        json.dump(
            {"error": str(exc), "error_type": type(exc).__name__},
            sys.stderr,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        sys.stderr.write("\n")
        return 2
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


CanonicalAuditStore = AuditStore
AuditWriter = AuditStore


__all__ = [
    "AuditConflictError",
    "AuditCorruptionError",
    "AuditExportError",
    "AuditStore",
    "AuditStoreError",
    "AuditWriter",
    "BatchCommitResult",
    "CanonicalAuditStore",
    "CommitResult",
    "ExpectedRevisionRequiredError",
    "OperationConflictError",
    "ProjectionCheckpoint",
    "ProjectionError",
    "RevisionConflictError",
    "StoredRecord",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised by offline CLI smoke tests.
    raise SystemExit(main())
