"""Crash-safe authority state for analysis-workbench sessions.

The BA-03 :class:`~audit_store.AuditStore` owns findings, annotations, and
action records.  Session credentials, policy bindings, budget reservations,
and operation lifecycle are a different mutable domain and must not be
represented as audit records.  ``AuditAuthorityStore`` is the small durable
boundary for that domain.

The journal stores complete, strictly-JSON state snapshots.  Each transaction
is hash chained and anchored by a sidecar checkpoint.  A process lock and a
fresh read under that lock serialize mutations across service instances.  A
partial or rolled-back journal therefore fails closed rather than being
silently repaired.  Session token material is never accepted in state; the
service stores only a salted verifier.

The durable lock currently has a POSIX locking contract: platforms without
``fcntl.flock`` are rejected explicitly at store construction instead of
silently falling back to a process-local lock that could corrupt a journal
across processes.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.audit_contracts import canonical_json

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None

AUTHORITY_SCHEMA_VERSION = "audit-authority.v1"
CODEX_AUTHORITY_SCHEMA_VERSION = "audit-codex-authority.v1"
MAX_AUTHORITY_BYTES = 16 * 1024 * 1024
MAX_STATE_BYTES = 8 * 1024 * 1024
_CODEX_SESSION_FIELDS = {
    "schema_version",
    "codex_session_id",
    "audit_session_id",
    "actor",
    "policy_id",
    "policy_revision",
    "policy_digest",
    "context",
    "source_ref",
    "source_digest",
    "source_revision",
    "route",
    "evidence",
    "provider_session_id",
    "status",
    "created_at",
    "updated_at",
    "last_operation_id",
}
_CODEX_OPERATION_FIELDS = {
    "schema_version",
    "operation_id",
    "action",
    "codex_session_id",
    "audit_session_id",
    "actor",
    "request_digest",
    "policy_digest",
    "context_revision",
    "source_revision",
    "source_digest",
    "route_digest",
    "status",
    "result_status",
    "reason",
    "result_digest",
    "reservation_id",
    "reservation_operation_id",
    "provider_session_id",
    "result",
    "created_at",
    "finished_at",
}


class AuditAuthorityError(RuntimeError):
    """Base class for durable authority failures."""


class AuthorityPlatformError(AuditAuthorityError):
    """Raised when the platform cannot provide the durable lock contract."""


class AuthorityCorruptionError(AuditAuthorityError):
    """Raised when authority bytes are partial, tampered, or rolled back."""


class AuthorityOperationConflict(AuditAuthorityError):
    """Raised when a transaction or external operation ID is reused."""


class AuthorityValidationError(AuditAuthorityError, ValueError):
    """Raised when a state mutation would violate the authority schema."""


@dataclass(frozen=True, slots=True)
class AuthorityMutation:
    """Result of one serialized authority transaction."""

    operation_id: str
    sequence: int
    value: Any
    state: Mapping[str, Any]
    replayed: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_json_constant(token: str) -> Any:
    raise AuthorityCorruptionError(f"non-finite JSON constant is not allowed: {token}")


def _nonempty(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuthorityValidationError(f"{name} must be a non-empty string")
    return value


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


class _PathLock:
    """Thread/process lock for one authority journal path."""

    _locks: dict[str, threading.RLock] = {}
    _guard = threading.Lock()

    def __init__(self, path: Path):
        key = str(path.resolve())
        with self._guard:
            self._thread_lock = self._locks.setdefault(key, threading.RLock())
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> _PathLock:
        """Return this process lock while the filesystem lock is held."""

        if fcntl is None:
            raise AuthorityPlatformError(
                "durable authority requires POSIX fcntl.flock process locking"
            )
        self._thread_lock.acquire()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Release both filesystem and process locks."""

        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        try:
            self._handle.close()
            self._handle = None
        finally:
            self._thread_lock.release()


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "sessions": {},
        "reservations": {},
        "operations": {},
    }


def _validate_codex_extension(value: Any) -> dict[str, Any]:  # noqa: C901
    """Validate the bounded Codex extension without importing service types.

    The authority journal is deliberately lower level than ``AuditService``.
    It therefore validates the version, closed field sets, and strict JSON
    shape here, while the service validates actor/policy/context/source
    relationships before exposing a record to a client.

    Returns:
        A detached strict-JSON extension mapping.
    """

    if not isinstance(value, Mapping) or set(value) != {"codex"}:
        raise AuthorityCorruptionError("authority extensions are invalid")
    codex = value["codex"]
    if not isinstance(codex, Mapping) or set(codex) != {"schema_version", "sessions", "operations"}:
        raise AuthorityCorruptionError("authority Codex extension schema is invalid")
    if codex.get("schema_version") != CODEX_AUTHORITY_SCHEMA_VERSION:
        raise AuthorityCorruptionError("authority Codex extension version is invalid")
    for name, fields, status_values in (
        ("sessions", _CODEX_SESSION_FIELDS, None),
        ("operations", _CODEX_OPERATION_FIELDS, {"inflight", "finished"}),
    ):
        records = codex.get(name)
        if not isinstance(records, Mapping):
            raise AuthorityCorruptionError(f"authority Codex {name} are invalid")
        if any(not isinstance(key, str) or not key.strip() for key in records):
            raise AuthorityCorruptionError(f"authority Codex {name} have invalid keys")
        for key, raw in records.items():
            if not isinstance(raw, Mapping) or set(raw) != fields:
                raise AuthorityCorruptionError(f"authority Codex {name} record is malformed")
            if raw.get("schema_version") != CODEX_AUTHORITY_SCHEMA_VERSION:
                raise AuthorityCorruptionError(f"authority Codex {name} record version is invalid")
            identity = "codex_session_id" if name == "sessions" else "operation_id"
            if raw.get(identity) != key:
                raise AuthorityCorruptionError(f"authority Codex {name} identity is inconsistent")
            if status_values is not None and raw.get("status") not in status_values:
                raise AuthorityCorruptionError("authority Codex operation status is invalid")
            if name == "operations":
                result_status = raw.get("result_status")
                if raw.get("status") == "inflight" and result_status != "":
                    raise AuthorityCorruptionError("inflight Codex result status is invalid")
                if raw.get("status") == "finished" and not isinstance(result_status, str):
                    raise AuthorityCorruptionError("finished Codex result status is invalid")
    try:
        return _strict_json(dict(value), name="authority extensions")
    except AuthorityValidationError as exc:
        raise AuthorityCorruptionError(str(exc)) from exc


def _strict_json(value: Any, *, name: str, limit: int = MAX_STATE_BYTES) -> Any:
    try:
        encoded = canonical_json(value).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuthorityValidationError(f"{name} is not strict JSON: {exc}") from exc
    if len(encoded) > limit:
        raise AuthorityValidationError(f"{name} exceeds {limit} bytes")
    return copy.deepcopy(value)


class AuditAuthorityStore:
    """Single-writer, hash-chained authority journal.

    ``root`` may be the same directory used by ``AuditStore``.  The authority
    journal intentionally has separate filenames and state vocabulary, while
    every session/budget/cancel/operation transition is committed as one
    snapshot transaction within this journal.
    """

    JOURNAL_FILENAME = "audit-authority.ndjson"
    LOCK_FILENAME = "audit-authority.lock"
    CHECKPOINT_FILENAME = "audit-authority.checkpoint.json"

    def __init__(
        self,
        root: str | Path,
        *,
        fail_after_journal_once: bool = False,
        mutation_hook: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        """Open or create the authority journal rooted at ``root``."""

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.root / self.JOURNAL_FILENAME
        self.lock_path = self.root / self.LOCK_FILENAME
        self.checkpoint_path = self.root / self.CHECKPOINT_FILENAME
        self._fail_after_journal_once = fail_after_journal_once
        self._mutation_hook = mutation_hook
        with _PathLock(self.lock_path):
            self._ensure_header_locked()
            self._read_locked()

    def __enter__(self) -> AuditAuthorityStore:
        """Return this authority store."""

        return self

    def __exit__(self, *_args: Any) -> None:
        """Close the authority store at context exit."""

        self.close()

    def close(self) -> None:
        """Keep API parity with ``AuditStore``; the journal has no open handle."""

    @staticmethod
    def new_token_verifier(token: str) -> dict[str, str]:
        """Return a salted verifier without retaining the opaque token."""

        token = _nonempty(token, name="session token")
        salt = secrets.token_bytes(16)
        digest = hashlib.sha256(salt + token.encode("utf-8")).hexdigest()
        return {
            "algorithm": "sha256-salt-v1",
            "salt": salt.hex(),
            "digest": digest,
        }

    @staticmethod
    def verify_token(record: Mapping[str, Any], token: str) -> bool:
        """Verify a token against a persisted verifier without exposing it.

        Returns:
            ``True`` only when the token matches the salted verifier.
        """

        if not isinstance(token, str) or not token.strip():
            return False
        verifier = record.get("token_verifier")
        if not isinstance(verifier, Mapping):
            return False
        if verifier.get("algorithm") != "sha256-salt-v1":
            return False
        salt_hex = verifier.get("salt")
        expected = verifier.get("digest")
        if not isinstance(salt_hex, str) or not isinstance(expected, str):
            return False
        try:
            salt = bytes.fromhex(salt_hex)
        except ValueError:
            return False
        actual = hashlib.sha256(salt + token.encode("utf-8")).hexdigest()
        return hmac.compare_digest(actual, expected)

    def _ensure_header_locked(self) -> None:
        if self.journal_path.exists():
            try:
                if self.journal_path.stat().st_size:
                    return
            except OSError as exc:
                raise AuthorityCorruptionError("authority journal cannot be inspected") from exc
            raise AuthorityCorruptionError("authority journal is empty")
        header = {
            "schema_version": AUTHORITY_SCHEMA_VERSION,
            "kind": "header",
            "store_id": secrets.token_hex(16),
            "created_at": _now(),
        }
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("wb") as handle:
            handle.write((canonical_json(header) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        if not self.checkpoint_path.exists():
            self._write_checkpoint_locked(sequence=0, digest=_digest(header))

    def _write_checkpoint_locked(self, *, sequence: int, digest: str) -> None:
        payload = {
            "schema_version": AUTHORITY_SCHEMA_VERSION,
            "sequence": sequence,
            "digest": digest,
        }
        temporary = self.checkpoint_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            handle.write((canonical_json(payload) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.checkpoint_path)
        try:
            directory_fd = os.open(self.checkpoint_path.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):  # pragma: no cover - non-POSIX fallback.
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _read_checkpoint_locked(self) -> tuple[int, str]:
        if not self.checkpoint_path.exists():
            raise AuthorityCorruptionError("authority checkpoint is missing")
        try:
            raw = self.checkpoint_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuthorityCorruptionError(f"authority checkpoint is unreadable: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise AuthorityCorruptionError("authority checkpoint is not an object")
        if set(payload) != {"schema_version", "sequence", "digest"}:
            raise AuthorityCorruptionError("authority checkpoint fields are invalid")
        sequence = payload.get("sequence")
        digest = payload.get("digest")
        if (
            payload.get("schema_version") != AUTHORITY_SCHEMA_VERSION
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or not _is_sha256_hex(digest)
        ):
            raise AuthorityCorruptionError("authority checkpoint values are invalid")
        return sequence, digest

    @staticmethod
    def _validate_state(state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise AuthorityCorruptionError("authority state is not an object")
        required = {"schema_version", "sessions", "reservations", "operations"}
        allowed = required | {"extensions"}
        if not set(state).issubset(allowed) or not required.issubset(state):
            raise AuthorityCorruptionError("authority state schema is invalid")
        if state.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
            raise AuthorityCorruptionError("authority state schema is invalid")
        for name in ("sessions", "reservations", "operations"):
            value = state.get(name)
            if not isinstance(value, Mapping):
                raise AuthorityCorruptionError(f"authority state {name} is invalid")
            if any(not isinstance(key, str) or not key.strip() for key in value):
                raise AuthorityCorruptionError(f"authority state {name} has invalid keys")
        if "extensions" in state:
            _validate_codex_extension(state["extensions"])
        try:
            copied = _strict_json(dict(state), name="authority state")
        except AuthorityValidationError as exc:
            raise AuthorityCorruptionError(str(exc)) from exc
        return copied

    def _read_locked(  # noqa: C901, PLR0912
        self,
    ) -> tuple[dict[str, Any], int, str, dict[str, dict[str, Any]]]:
        try:
            raw = self.journal_path.read_bytes()
        except OSError as exc:
            raise AuthorityCorruptionError(f"authority journal is unreadable: {exc}") from exc
        if not raw or len(raw) > MAX_AUTHORITY_BYTES:
            raise AuthorityCorruptionError("authority journal is empty or too large")
        lines = raw.splitlines(keepends=True)
        if not lines or not lines[0].endswith(b"\n"):
            raise AuthorityCorruptionError("authority journal header is incomplete")
        try:
            header = json.loads(lines[0][:-1].decode("utf-8"), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuthorityCorruptionError(f"authority journal header is invalid: {exc}") from exc
        if (
            not isinstance(header, Mapping)
            or set(header) != {"schema_version", "kind", "store_id", "created_at"}
            or header.get("schema_version") != AUTHORITY_SCHEMA_VERSION
            or header.get("kind") != "header"
            or not isinstance(header.get("store_id"), str)
            or not isinstance(header.get("created_at"), str)
        ):
            raise AuthorityCorruptionError("authority journal header is invalid")
        state = _empty_state()
        sequence = 0
        previous_digest = _digest(header)
        transactions: dict[str, dict[str, Any]] = {}
        for index, line in enumerate(lines[1:], start=2):
            if not line.endswith(b"\n"):
                raise AuthorityCorruptionError(f"authority journal line {index} is incomplete")
            try:
                transaction = json.loads(
                    line[:-1].decode("utf-8"), parse_constant=_reject_json_constant
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AuthorityCorruptionError(
                    f"authority journal line {index} is invalid: {exc}"
                ) from exc
            if not isinstance(transaction, Mapping):
                raise AuthorityCorruptionError(f"authority journal line {index} is not an object")
            allowed = {
                "schema_version",
                "kind",
                "sequence",
                "operation_id",
                "request_digest",
                "prev_digest",
                "state",
                "value",
                "committed_at",
                "digest",
            }
            if set(transaction) != allowed or transaction.get("kind") != "transaction":
                raise AuthorityCorruptionError(f"authority transaction {index} fields are invalid")
            current_sequence = transaction.get("sequence")
            operation_id = transaction.get("operation_id")
            request_digest = transaction.get("request_digest")
            prev_digest = transaction.get("prev_digest")
            digest = transaction.get("digest")
            if (
                transaction.get("schema_version") != AUTHORITY_SCHEMA_VERSION
                or isinstance(current_sequence, bool)
                or not isinstance(current_sequence, int)
                or current_sequence != sequence + 1
                or not isinstance(operation_id, str)
                or not operation_id.strip()
                or not _is_sha256_hex(request_digest)
                or prev_digest != previous_digest
                or not _is_sha256_hex(prev_digest)
                or not _is_sha256_hex(digest)
                or operation_id in transactions
                or not isinstance(transaction.get("committed_at"), str)
            ):
                raise AuthorityCorruptionError(f"authority transaction {index} identity is invalid")
            unsigned = {key: value for key, value in transaction.items() if key != "digest"}
            expected_digest = _digest(unsigned)
            if not hmac.compare_digest(expected_digest, digest):
                raise AuthorityCorruptionError(f"authority transaction {index} digest mismatch")
            state = self._validate_state(transaction["state"])
            try:
                _strict_json(transaction.get("value"), name="authority transaction value")
            except AuthorityValidationError as exc:
                raise AuthorityCorruptionError(str(exc)) from exc
            transaction_copy = dict(transaction)
            transactions[operation_id] = transaction_copy
            sequence = current_sequence
            previous_digest = digest
        checkpoint_sequence, checkpoint_digest = self._read_checkpoint_locked()
        if checkpoint_sequence > sequence:
            raise AuthorityCorruptionError("authority journal was rolled back")
        if checkpoint_sequence == sequence and checkpoint_digest != previous_digest:
            raise AuthorityCorruptionError("authority checkpoint does not match journal")
        if checkpoint_sequence < sequence:
            # A crash after journal fsync and before checkpoint promotion leaves
            # a valid, newer journal.  Advance the checkpoint only after all
            # bytes and hashes have been checked.
            self._write_checkpoint_locked(sequence=sequence, digest=previous_digest)
        return state, sequence, previous_digest, transactions

    def snapshot(self) -> dict[str, Any]:
        """Read and validate the latest state under the process lock.

        Returns:
            A detached copy of the latest authority state.
        """

        with _PathLock(self.lock_path):
            state, _sequence, _digest_value, _transactions = self._read_locked()
            return copy.deepcopy(state)

    def snapshot_with_sequence(self) -> tuple[dict[str, Any], int]:
        """Return the latest state and its monotonic CAS sequence.

        Returns:
            A detached state snapshot and the sequence that can be supplied to
            :meth:`mutate` as ``expected_sequence``.
        """

        with _PathLock(self.lock_path):
            state, sequence, _digest_value, _transactions = self._read_locked()
            return copy.deepcopy(state), sequence

    def mutate(
        self,
        operation_id: str,
        request_digest: str,
        mutator: Callable[[dict[str, Any]], Any],
        *,
        expected_sequence: int | None = None,
    ) -> AuthorityMutation:
        """Apply one serialized state transition or replay its transaction.

        Returns:
            The committed state and compact mutation result.
        """

        operation_id = _nonempty(operation_id, name="authority operation_id")
        request_digest = _nonempty(request_digest, name="authority request_digest")
        if not _is_sha256_hex(request_digest):
            raise AuthorityValidationError("authority request_digest must be sha256 hex")
        with _PathLock(self.lock_path):
            state, sequence, previous_digest, transactions = self._read_locked()
            previous = transactions.get(operation_id)
            if previous is not None:
                if previous.get("request_digest") != request_digest:
                    raise AuthorityOperationConflict(
                        f"authority operation ID {operation_id!r} was reused for another request"
                    )
                return AuthorityMutation(
                    operation_id=operation_id,
                    sequence=int(previous["sequence"]),
                    value=copy.deepcopy(previous.get("value")),
                    state=copy.deepcopy(previous["state"]),
                    replayed=True,
                )
            if expected_sequence is not None and expected_sequence != sequence:
                raise AuthorityOperationConflict(
                    f"authority sequence conflict: expected {expected_sequence}, current {sequence}"
                )
            candidate = copy.deepcopy(state)
            value = mutator(candidate)
            candidate = self._validate_state(candidate)
            value = _strict_json(value, name="authority mutation value", limit=MAX_STATE_BYTES)
            transaction = {
                "schema_version": AUTHORITY_SCHEMA_VERSION,
                "kind": "transaction",
                "sequence": sequence + 1,
                "operation_id": operation_id,
                "request_digest": request_digest,
                "prev_digest": previous_digest,
                "state": candidate,
                "value": value,
                "committed_at": _now(),
            }
            transaction["digest"] = _digest(transaction)
            with self.journal_path.open("ab") as handle:
                handle.write((canonical_json(transaction) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            if self._mutation_hook is not None:
                self._mutation_hook(transaction)
            if self._fail_after_journal_once:
                self._fail_after_journal_once = False
                raise AuditAuthorityError("simulated crash after authority journal commit")
            self._write_checkpoint_locked(sequence=sequence + 1, digest=transaction["digest"])
            return AuthorityMutation(
                operation_id=operation_id,
                sequence=sequence + 1,
                value=copy.deepcopy(value),
                state=copy.deepcopy(candidate),
                replayed=False,
            )

    compare_and_swap = mutate
    transaction = mutate


DurableAuthorityStore = AuditAuthorityStore


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "CODEX_AUTHORITY_SCHEMA_VERSION",
    "AuditAuthorityError",
    "AuditAuthorityStore",
    "AuthorityCorruptionError",
    "AuthorityMutation",
    "AuthorityOperationConflict",
    "AuthorityPlatformError",
    "AuthorityValidationError",
    "DurableAuthorityStore",
]
