"""Revision-bound BA-02/BA-04 adapters for the local audit service.

The read adapter intentionally exposes only reads. ``QueueNextAdapter`` is a
separate, narrowly-scoped mutation owner for BA-02 ``select_next``. It holds a
stable filesystem lock while the queue mutation and the service's authority
callback run. The lock serializes competing clients, but it is not an atomic
transaction across the queue and authority stores; the service's durable
inflight operation remains the crash/retry boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from robot_sf.analysis_workbench.audit_contracts import (
    ReviewPacket,
    ReviewRecord,
    canonical_json,
    record_from_dict,
    record_to_dict,
)
from robot_sf.analysis_workbench.audit_coverage import AuditHealthReport
from robot_sf.analysis_workbench.audit_queue import (
    AuditQueue,
    QueueConflictError,
    QueueOperationConflictError,
    QueueSelectionContext,
    QueueStateError,
    SelectionExplanation,
    SelectionResult,
)
from robot_sf.analysis_workbench.audit_scan import AuditScanReport
from robot_sf.analysis_workbench.audit_service import (
    AuditContextConflict,
    AuditNextAmbiguous,
    AuditNextBlocked,
    AuditSelectionContext,
    CapabilityResult,
    CapabilityUnavailable,
    _episode_ref_matches_row,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - the queue itself is POSIX-bound.
    fcntl = None


_NEXT_COORDINATION_SCHEMA = "audit-queue-next.v1"
_NEXT_COORDINATION_MAX_BYTES = 4 * 1024 * 1024
_NEXT_COORDINATION_MAX_COMPLETED = 512


def coverage_reviews_for_scan(
    scan: AuditScanReport, reviews: tuple[ReviewRecord, ...]
) -> tuple[ReviewRecord, ...]:
    """Project BA-02 episode IDs onto BA-04's literal inventory IDs.

    This is a read-only projection of trusted typed evidence.  A generated
    EpisodeRef ID earns credit only when its *full* identity matches exactly
    one readable inventory row and no other ref claims that row.  Ambiguous
    generated IDs are omitted, including a collision with a literal row ID.
    BA-04 still checks the receipt's source and scan tokens independently.

    Returns:
        Typed reviews with only unambiguous generated IDs projected.
    """

    if not isinstance(scan, AuditScanReport) or any(
        not isinstance(review, ReviewRecord) for review in reviews
    ):
        raise ValueError("coverage projection requires typed scan and reviews")
    row_counts = Counter(item.episode_id for item in scan.inventory)
    ref_ids = {ref.episode_id for ref in scan.episode_refs}
    candidate_rows: dict[str, str] = {}
    claimed_rows: dict[str, int] = {}
    for ref in scan.episode_refs:
        if (
            ref.campaign_digest != scan.audit.campaign_digest
            or ref.source_digest != scan.audit.source_digest
            or ref.source != scan.audit.source
        ):
            continue
        matches = [
            item.episode_id
            for item in scan.inventory
            if item.readable and _episode_ref_matches_row(ref, item.row)
        ]
        if len(matches) != 1 or row_counts[matches[0]] != 1:
            continue
        candidate_rows[ref.episode_id] = matches[0]
        claimed_rows[matches[0]] = claimed_rows.get(matches[0], 0) + 1

    projected: list[ReviewRecord] = []
    for review in reviews:
        if review.episode_id not in ref_ids:
            projected.append(review)
            continue
        row_id = candidate_rows.get(review.episode_id)
        if row_id is None or claimed_rows[row_id] != 1:
            continue
        projected.append(replace(review, episode_id=row_id))
    return tuple(projected)


def _next_coordination_path(state_path: Path) -> Path:
    return state_path.with_name(f"{state_path.name}.service-next.json")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")


def _result_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _empty_coordination() -> dict[str, Any]:
    return {"schema_version": _NEXT_COORDINATION_SCHEMA, "active": None, "completed": {}}


def _read_coordination(path: Path) -> dict[str, Any]:  # noqa: C901
    """Read the queue-side lease without treating corruption as empty state.

    Returns:
        The validated coordination mapping.
    """

    if path.is_symlink():
        raise CapabilityUnavailable("BA-02 service Next coordination path must not be a symlink")
    if not path.exists():
        return _empty_coordination()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CapabilityUnavailable("BA-02 service Next coordination state is unavailable") from exc
    if len(raw) > _NEXT_COORDINATION_MAX_BYTES:
        raise CapabilityUnavailable("BA-02 service Next coordination state exceeds its bound")
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapabilityUnavailable("BA-02 service Next coordination state is malformed") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "active", "completed"}:
        raise CapabilityUnavailable("BA-02 service Next coordination schema is invalid")
    if value.get("schema_version") != _NEXT_COORDINATION_SCHEMA:
        raise CapabilityUnavailable("unsupported BA-02 service Next coordination schema")
    active = value.get("active")
    if active is not None and not isinstance(active, dict):
        raise CapabilityUnavailable("BA-02 service Next active lease is malformed")
    completed = value.get("completed")
    if not isinstance(completed, dict) or len(completed) > _NEXT_COORDINATION_MAX_COMPLETED:
        raise CapabilityUnavailable("BA-02 service Next replay index is malformed")
    if any(
        not isinstance(key, str) or not isinstance(item, dict) for key, item in completed.items()
    ):
        raise CapabilityUnavailable("BA-02 service Next replay entry is malformed")
    return value


def _write_coordination(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically persist the queue-side lease and replay envelope."""

    if path.is_symlink():
        raise CapabilityUnavailable("BA-02 service Next coordination path must not be a symlink")
    payload = canonical_json(value).encode("utf-8") + b"\n"
    if len(payload) > _NEXT_COORDINATION_MAX_BYTES:
        raise CapabilityUnavailable("BA-02 service Next coordination state exceeds its bound")
    temporary: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except OSError as exc:
        raise CapabilityUnavailable(
            "BA-02 service Next coordination state cannot be committed"
        ) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


@dataclass(frozen=True)
class AuditSourceBinding:
    """The admitted campaign/source identity shared by queue and coverage."""

    campaign_id: str
    campaign_digest: str
    source_digest: str
    source_revision: int | str
    source_aliases: tuple[str, ...] = ()

    def check_context(self, context: AuditSelectionContext) -> None:
        """Reject reads from a different or stale source selection."""

        if context.campaign_id != self.campaign_id:
            raise AuditContextConflict("audit adapter campaign changed")
        if context.source_revision != self.source_revision:
            raise AuditContextConflict("audit adapter source revision changed")
        if context.source_identity not in (self.source_digest, *self.source_aliases):
            raise AuditContextConflict("audit adapter source identity changed")


class _NextLockState:
    """In-process ownership state for one durable queue lock path."""

    __slots__ = ("depth", "mutex", "owner")

    def __init__(self) -> None:
        self.mutex = threading.RLock()
        self.owner: int | None = None
        self.depth = 0


_NEXT_LOCK_STATES_GUARD = threading.Lock()
_NEXT_LOCK_STATES: dict[str, _NextLockState] = {}


def _open_next_lock_handle(lock_path: Path, flags: int) -> Any:
    """Open the durable Next lock and close a raw descriptor on wrapper failure.

    Returns:
        The open text handle for the durable lock file.
    """

    descriptor: int | None = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, flags, 0o600)
        handle = os.fdopen(descriptor, "a+", encoding="utf-8")
        descriptor = None
        return handle
    except OSError as exc:
        raise CapabilityUnavailable("BA-02 service Next lock is unavailable") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


@contextmanager
def _next_lock(path: Path):
    """Hold the durable cross-service lock for one queue state path.

    The queue callback contract permits a same-thread service operation to
    re-enter the transaction (for example, an authority context CAS from a
    ``Next`` callback).  ``flock`` on a newly opened descriptor is not
    reentrant, so keep an in-process owner/depth guard around the durable lock:
    nested calls reuse the outer descriptor while other threads still wait on
    the per-path mutex and other processes still wait on ``flock``.
    """

    if fcntl is None:
        raise CapabilityUnavailable("BA-02 service Next requires POSIX file locking")
    lock_path = path.with_name(f"{path.name}.service-next.lock")
    lock_key = os.path.abspath(os.fspath(lock_path))
    with _NEXT_LOCK_STATES_GUARD:
        lock_state = _NEXT_LOCK_STATES.setdefault(lock_key, _NextLockState())
    lock_state.mutex.acquire()
    owner = threading.get_ident()
    if lock_state.owner == owner:
        lock_state.depth += 1
        try:
            yield
        finally:
            lock_state.depth -= 1
            lock_state.mutex.release()
        return
    lock_state.owner = owner
    lock_state.depth = 1
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    handle = None
    locked = False
    try:
        handle = _open_next_lock_handle(lock_path, flags)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            raise CapabilityUnavailable("BA-02 service Next lock cannot be acquired") from exc
        yield
    finally:
        try:
            if locked:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError as exc:
                    raise CapabilityUnavailable(
                        "BA-02 service Next lock cannot be released"
                    ) from exc
        finally:
            try:
                if handle is not None:
                    handle.close()
            finally:
                lock_state.owner = None
                lock_state.depth = 0
                lock_state.mutex.release()


@dataclass(frozen=True)
class QueueNextResult:
    """One durable BA-02 packet selection and its queue revisions."""

    selection_result: SelectionResult
    state_revision: int
    input_revision: int
    input_identity: str

    @property
    def packet(self) -> ReviewPacket:
        """Return the selected BA-03 packet."""

        return self.selection_result.packet

    @property
    def context(self) -> QueueSelectionContext:
        """Return BA-02's versioned selection context."""

        return self.selection_result.context

    @property
    def selection(self) -> QueueSelectionContext:
        """Compatibility alias for the queue selection context."""

        return self.selection_result.context

    def to_dict(self) -> dict[str, object]:
        """Return a bounded queue/service handoff envelope."""

        return {
            **self.selection_result.to_dict(),
            "queue_state_revision": self.state_revision,
            "queue_input_revision": self.input_revision,
            "queue_input_identity": self.input_identity,
        }


@dataclass(frozen=True)
class QueueReadAdapter:
    """Project a fresh BA-02 queue snapshot without advancing selection."""

    binding: AuditSourceBinding
    queue_provider: Callable[[], AuditQueue]

    def read(self, *, context: AuditSelectionContext, limit: int) -> CapabilityResult:
        """Return rank/reasons/current packet for the bound source revision."""

        self.binding.check_context(context)
        queue = self.queue_provider()
        if not isinstance(queue, AuditQueue):
            raise CapabilityUnavailable("BA-02 queue provider is unavailable")
        if (
            queue.dataset.campaign_digest != self.binding.campaign_digest
            or queue.dataset.source_digest != self.binding.source_digest
            or queue.stale_inputs
        ):
            raise AuditContextConflict("BA-02 queue input identity changed")
        ranked = queue.rank_candidates()[:limit]
        packet = queue.current_packet
        return CapabilityResult(
            "ba-02.queue",
            "complete",
            {
                "campaign_digest": queue.dataset.campaign_digest,
                "source_digest": queue.dataset.source_digest,
                "source_revision": self.binding.source_revision,
                "input_revision": queue.input_revision,
                "state_revision": queue.state_revision,
                "input_identity": queue.dataset.identity,
                "ranked": [candidate.to_dict() for candidate in ranked],
                "current_packet": record_to_dict(packet) if packet is not None else None,
            },
            provider="audit_queue.AuditQueue",
        )


@dataclass(frozen=True)
class QueueNextAdapter:
    """Own the bounded BA-05 call into BA-02 ``AuditQueue.select_next``.

    ``before_select`` and ``after_select`` are deliberately callbacks instead
    of another persistence implementation. The service uses them to perform a
    fresh authority/context CAS while this adapter's durable lock is held.
    """

    binding: AuditSourceBinding
    queue_provider: Callable[[], AuditQueue]

    def _queue(self, context: AuditSelectionContext) -> AuditQueue:
        self.binding.check_context(context)
        try:
            queue = self.queue_provider()
        except ValueError as exc:
            raise CapabilityUnavailable(f"BA-02 queue state is unavailable: {exc}") from exc
        if not isinstance(queue, AuditQueue):
            raise CapabilityUnavailable("BA-02 queue provider is unavailable")
        if queue.state_path is None:
            raise CapabilityUnavailable("BA-02 service Next requires queue state persistence")
        if (
            queue.dataset.campaign_digest != self.binding.campaign_digest
            or queue.dataset.source_digest != self.binding.source_digest
            or queue.stale_inputs
        ):
            raise AuditContextConflict("BA-02 queue input identity changed")
        return queue

    def _stable_queue(self, context: AuditSelectionContext) -> AuditQueue:
        """Retry only the short BA-02 pending-file boundary before locking.

        Returns:
            A queue loaded after the transient pending mutation boundary.
        """

        deadline = time.monotonic() + 1.0
        while True:
            try:
                return self._queue(context)
            except CapabilityUnavailable as exc:
                if "pending queue mutation" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def _validate_selection(  # noqa: C901
        self,
        queue: AuditQueue,
        selection: SelectionResult,
        *,
        expected_state_revision: int | None,
        expected_input_revision: int | None,
        require_current: bool = True,
    ) -> QueueNextResult:
        if selection.context.campaign_digest != self.binding.campaign_digest:
            raise AuditContextConflict("BA-02 selection campaign identity changed")
        if selection.context.source_digest != self.binding.source_digest:
            raise AuditContextConflict("BA-02 selection source identity changed")
        if selection.context.input_revision != queue.input_revision:
            raise AuditContextConflict("BA-02 selection input revision changed")
        if selection.context.input_identity != queue.dataset.identity:
            raise AuditContextConflict("BA-02 selection input identity changed")
        if require_current:
            if (
                queue.current_packet is None
                or queue.current_packet.packet_id != selection.packet.packet_id
            ):
                raise AuditContextConflict("BA-02 current packet does not match selection")
            if (
                not queue.state.selection_history
                or queue.state.selection_history[-1] != selection.context
            ):
                raise AuditContextConflict("BA-02 selection history does not match packet")
        elif not any(item == selection.context for item in queue.state.selection_history):
            raise CapabilityUnavailable("BA-02 exact selection history is unavailable")
        if (
            expected_state_revision is not None
            and queue.state_revision != expected_state_revision + 1
        ):
            raise AuditContextConflict("BA-02 queue state revision changed")
        if expected_input_revision is not None and queue.input_revision != expected_input_revision:
            raise AuditContextConflict("BA-02 queue input revision changed")
        return QueueNextResult(
            selection_result=selection,
            state_revision=queue.state_revision,
            input_revision=queue.input_revision,
            input_identity=queue.dataset.identity,
        )

    @staticmethod
    def _queue_identity(queue: AuditQueue) -> str:
        """Return the stable queue-state identity used by the sidecar lease."""

        if queue.state_path is None:  # pragma: no cover - guarded by ``_queue``.
            raise CapabilityUnavailable("BA-02 service Next requires queue state persistence")
        return str(queue.state_path.absolute())

    def _lease_record(
        self,
        queue: AuditQueue,
        *,
        context: AuditSelectionContext,
        operation_id: str,
        session_id: str,
        expected_state_revision: int | None,
        expected_input_revision: int | None,
    ) -> dict[str, Any]:
        return {
            "operation_id": operation_id,
            "session_id": session_id,
            "queue_path": self._queue_identity(queue),
            "campaign_id": context.campaign_id,
            "campaign_digest": self.binding.campaign_digest,
            "source_digest": self.binding.source_digest,
            "source_revision": self.binding.source_revision,
            "context": context.to_dict(),
            "expected_state_revision": expected_state_revision,
            "expected_input_revision": expected_input_revision,
            "phase": "preflight",
        }

    @staticmethod
    def _same_lease(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return all(
            left.get(key) == right.get(key)
            for key in (
                "operation_id",
                "session_id",
                "queue_path",
                "campaign_id",
                "campaign_digest",
                "source_digest",
                "source_revision",
                "context",
            )
        )

    @staticmethod
    def _clear_active(coordination: dict[str, Any], *, path: Path) -> None:
        coordination["active"] = None
        _write_coordination(path, coordination)

    def _claim_lease(
        self,
        queue: AuditQueue,
        *,
        context: AuditSelectionContext,
        operation_id: str,
        session_id: str,
        expected_state_revision: int | None,
        expected_input_revision: int | None,
    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
        path = _next_coordination_path(queue.state_path)
        coordination = _read_coordination(path)
        candidate = self._lease_record(
            queue,
            context=context,
            operation_id=operation_id,
            session_id=session_id,
            expected_state_revision=expected_state_revision,
            expected_input_revision=expected_input_revision,
        )
        active = coordination.get("active")
        if active is not None:
            if not isinstance(active, Mapping):
                raise CapabilityUnavailable("BA-02 service Next active lease is malformed")
            if not self._same_lease(active, candidate):
                if active.get("phase") == "preflight":
                    raise AuditContextConflict("another BA-02 queue Next operation is in preflight")
                if active.get("phase") == "queue_committed":
                    raise AuditNextBlocked(
                        "another BA-02 queue Next operation owns the committed queue lease"
                    )
                raise AuditNextAmbiguous(
                    "BA-02 queue Next has an unresolved operation; reconciliation is unavailable"
                )
            if active.get("phase") != "preflight":
                raise AuditNextAmbiguous(
                    "BA-02 queue commit is unresolved; reconciliation is unavailable"
                )
            return dict(active), coordination, path
        coordination["active"] = candidate
        _write_coordination(path, coordination)
        return candidate, coordination, path

    def _replay_result(  # noqa: C901, PLR0912, PLR0915
        self,
        queue: AuditQueue,
        record: Mapping[str, Any],
        *,
        context: AuditSelectionContext,
        expected_result_digest: str = "",
    ) -> QueueNextResult:
        """Reconstruct exactly the envelope recorded for one operation.

        Returns:
            The exact packet, selection context, explanation, and revisions.
        """

        if record.get("queue_path") != self._queue_identity(queue):
            raise AuditContextConflict("BA-02 replay queue identity changed")
        if record.get("campaign_id") != context.campaign_id:
            raise AuditContextConflict("BA-02 replay campaign identity changed")
        if record.get("campaign_digest") != self.binding.campaign_digest:
            raise AuditContextConflict("BA-02 replay campaign digest changed")
        if record.get("source_digest") != self.binding.source_digest:
            raise AuditContextConflict("BA-02 replay source identity changed")
        if record.get("source_revision") != self.binding.source_revision:
            raise AuditContextConflict("BA-02 replay source revision changed")
        stored_context = record.get("context")
        if not isinstance(stored_context, Mapping):
            raise CapabilityUnavailable("BA-02 replay context binding is unavailable")
        if stored_context.get("campaign_id") != context.campaign_id:
            raise AuditContextConflict("BA-02 replay context campaign changed")
        if stored_context.get("source_identity") not in {
            self.binding.source_digest,
            *self.binding.source_aliases,
        }:
            raise AuditContextConflict("BA-02 replay context source changed")
        if stored_context.get("source_revision") != self.binding.source_revision:
            raise AuditContextConflict("BA-02 replay context revision changed")
        envelope = record.get("result")
        if not isinstance(envelope, Mapping):
            raise CapabilityUnavailable("BA-02 exact replay envelope is unavailable")
        digest = record.get("result_digest")
        if not isinstance(digest, str) or _result_digest(envelope) != digest:
            raise CapabilityUnavailable("BA-02 exact replay envelope digest is invalid")
        if expected_result_digest and digest != expected_result_digest:
            raise CapabilityUnavailable("BA-02 replay result digest does not match authority")
        packet_payload = envelope.get("packet")
        selection_payload = envelope.get("selection")
        if not isinstance(packet_payload, Mapping) or not isinstance(selection_payload, Mapping):
            raise CapabilityUnavailable("BA-02 exact replay envelope is malformed")
        try:
            packet = record_from_dict(packet_payload)
            selection = QueueSelectionContext.from_mapping(selection_payload)
            explanation_payload = envelope.get("explanation")
            explanation = (
                SelectionExplanation(**dict(explanation_payload))
                if isinstance(explanation_payload, Mapping)
                else None
            )
        except (TypeError, KeyError, ValueError) as exc:
            raise CapabilityUnavailable("BA-02 exact replay envelope is malformed") from exc
        if not isinstance(packet, ReviewPacket):
            raise CapabilityUnavailable("BA-02 exact replay packet is malformed")
        if selection.packet_id != packet.packet_id:
            raise AuditContextConflict("BA-02 replay packet and selection diverged")
        stored_revision = stored_context.get("context_revision")
        if isinstance(stored_revision, bool) or not isinstance(stored_revision, int):
            raise CapabilityUnavailable("BA-02 replay context revision is malformed")
        if context.context_revision == stored_revision:
            if context.to_dict() != dict(stored_context):
                raise AuditContextConflict("BA-02 replay context binding changed")
        elif (
            context.context_revision < stored_revision
            or context.reference_id != packet.packet_id
            or context.episode_id != selection.primary_episode_id
        ):
            raise AuditContextConflict("BA-02 replay context does not identify the selected packet")
        if envelope.get("queue_input_revision") != queue.input_revision:
            raise AuditContextConflict("BA-02 replay queue input revision changed")
        if envelope.get("queue_input_identity") != queue.dataset.identity:
            raise AuditContextConflict("BA-02 replay queue input identity changed")
        snapshot = queue.state.packet_payloads.get(packet.packet_id)
        if not isinstance(snapshot, Mapping) or canonical_json(snapshot) != canonical_json(
            packet_payload
        ):
            raise AuditContextConflict("BA-02 replay packet snapshot changed")
        state_revision = envelope.get("queue_state_revision")
        if isinstance(state_revision, bool) or not isinstance(state_revision, int):
            raise CapabilityUnavailable("BA-02 replay queue revision is malformed")
        if state_revision > queue.state_revision:
            raise AuditContextConflict("BA-02 replay queue state revision is unavailable")
        history = queue.state.selection_history
        if not any(item.selection_id == selection.selection_id for item in history):
            raise CapabilityUnavailable("BA-02 exact selection history is unavailable")
        if not any(item == selection for item in history):
            raise AuditContextConflict("BA-02 replay selection context changed")
        return QueueNextResult(
            selection_result=SelectionResult(
                packet=packet, context=selection, explanation=explanation
            ),
            state_revision=state_revision,
            input_revision=envelope["queue_input_revision"],
            input_identity=envelope["queue_input_identity"],
        )

    def transact(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        *,
        context: AuditSelectionContext,
        before_select: Callable[[AuditQueue], None] | None = None,
        after_select: Callable[[QueueNextResult], object] | None = None,
        expected_state_revision: int | None = None,
        expected_input_revision: int | None = None,
        force_current: bool = False,
        lock_held: bool = False,
        operation_id: str | None = None,
        session_id: str | None = None,
    ) -> object:
        """Select once, then invoke the service commit callback under the lock.

        Returns:
            The callback value, a selected queue result, or an unavailable
            capability envelope when the queue is empty.
        """

        if expected_state_revision is not None and (
            isinstance(expected_state_revision, bool)
            or not isinstance(expected_state_revision, int)
            or expected_state_revision < 0
        ):
            raise AuditContextConflict("BA-02 expected queue state revision is invalid")
        if expected_input_revision is not None and (
            isinstance(expected_input_revision, bool)
            or not isinstance(expected_input_revision, int)
            or expected_input_revision < 0
        ):
            raise AuditContextConflict("BA-02 expected queue input revision is invalid")
        queue = self._stable_queue(context)
        if expected_state_revision is not None and queue.state_revision != expected_state_revision:
            raise AuditContextConflict(
                "BA-02 queue state revision CAS failed",
                expected=expected_state_revision,
                actual=queue.state_revision,
            )
        if expected_input_revision is not None and queue.input_revision != expected_input_revision:
            raise AuditContextConflict(
                "BA-02 queue input revision CAS failed",
                expected=expected_input_revision,
                actual=queue.input_revision,
            )
        lease_operation_id = operation_id or f"adapter-next-{uuid.uuid4().hex}"
        lease_session_id = session_id or ""
        lock = nullcontext() if lock_held else _next_lock(queue.state_path)
        with lock:
            # Recheck identities after waiting for another service instance.
            if not lock_held:
                queue = self._stable_queue(context)
            if (
                expected_state_revision is not None
                and queue.state_revision != expected_state_revision
            ):
                raise AuditContextConflict("BA-02 queue state revision CAS failed")
            if (
                expected_input_revision is not None
                and queue.input_revision != expected_input_revision
            ):
                raise AuditContextConflict("BA-02 queue input revision CAS failed")
            lease, coordination, coordination_path = self._claim_lease(
                queue,
                context=context,
                operation_id=lease_operation_id,
                session_id=lease_session_id,
                expected_state_revision=expected_state_revision,
                expected_input_revision=expected_input_revision,
            )
            try:
                if before_select is not None:
                    before_select(queue)
            except Exception:
                # A preflight failure is known to precede queue mutation.  If
                # the lease cannot be cleared, retain the conservative block.
                try:
                    self._clear_active(coordination, path=coordination_path)
                except CapabilityUnavailable as exc:
                    raise AuditNextAmbiguous(
                        "BA-02 preflight failed and its lease cannot be cleared"
                    ) from exc
                raise
            try:
                selection = queue.select_next(force_current=force_current)
            except (QueueConflictError, QueueOperationConflictError, QueueStateError) as exc:
                # These BA-02 errors are raised before a successful selection
                # envelope exists; release the preflight lease only if that
                # release is itself durable.
                try:
                    self._clear_active(coordination, path=coordination_path)
                except CapabilityUnavailable as clear_exc:
                    raise AuditNextAmbiguous(
                        "BA-02 queue selection failed and its lease cannot be cleared"
                    ) from clear_exc
                if isinstance(exc, QueueConflictError):
                    raise AuditContextConflict("BA-02 queue state revision changed") from exc
                if isinstance(exc, QueueOperationConflictError):
                    raise AuditContextConflict("BA-02 queue operation identity conflicts") from exc
                raise CapabilityUnavailable(f"BA-02 queue state is unavailable: {exc}") from exc
            if selection is None:
                try:
                    self._clear_active(coordination, path=coordination_path)
                except CapabilityUnavailable as exc:
                    raise AuditNextAmbiguous(
                        "BA-02 empty selection could not clear its durable lease"
                    ) from exc
                return CapabilityResult(
                    "ba-02.queue.next",
                    "unavailable",
                    reason="BA-02 queue has no selectable packet",
                )
            try:
                selected = self._validate_selection(
                    queue,
                    selection,
                    expected_state_revision=expected_state_revision,
                    expected_input_revision=expected_input_revision,
                )
            except Exception as exc:
                raise AuditNextAmbiguous(
                    "BA-02 queue commit cannot be validated; reconciliation is unavailable"
                ) from exc
            lease = dict(lease)
            envelope = selected.to_dict()
            lease.update(
                {
                    "phase": "queue_committed",
                    "result": envelope,
                    "result_digest": _result_digest(envelope),
                }
            )
            coordination["active"] = lease
            try:
                _write_coordination(coordination_path, coordination)
            except CapabilityUnavailable as exc:
                raise AuditNextAmbiguous(
                    "BA-02 queue commit is durable but its replay envelope is not"
                    "; reconciliation is unavailable"
                ) from exc
            if after_select is not None:
                try:
                    result = after_select(selected)
                except Exception as exc:
                    raise AuditNextAmbiguous(
                        "BA-02 queue commit succeeded but authority commit is unresolved"
                        "; reconciliation is unavailable"
                    ) from exc
            else:
                result = selected
            # A service callback means the separate BA-05 authority operation
            # still needs to publish its terminal receipt.  Keep the
            # queue_committed lease active until the service reads this exact
            # envelope back after that receipt is durable.  Direct adapter
            # callers have no outer receipt boundary, so retain the historical
            # immediate finalization for that path.
            if not (after_select is not None and operation_id is not None):
                completed = dict(coordination.get("completed", {}))
                if operation_id is not None:
                    completed[operation_id] = lease
                    while len(completed) > _NEXT_COORDINATION_MAX_COMPLETED:
                        completed.pop(next(iter(completed)))
                coordination["completed"] = completed
                coordination["active"] = None
                try:
                    _write_coordination(coordination_path, coordination)
                except CapabilityUnavailable as exc:
                    raise AuditNextAmbiguous(
                        "BA-02 queue and authority commit may differ; replay finalization"
                        " is unavailable"
                    ) from exc
            return result

    @contextmanager
    def transaction_lock(self, *, context: AuditSelectionContext) -> Iterator[None]:
        """Hold the durable Next lock across admission and queue commit."""

        queue = self._stable_queue(context)
        with _next_lock(queue.state_path):
            yield

    def record_human_review(  # noqa: PLR0913
        self,
        *,
        context: AuditSelectionContext,
        before_review: Callable[[AuditQueue], None],
        expected_state_revision: int,
        expected_input_revision: int,
        operation_id: str,
        author_id: str,
        outcome: str,
        notes: str = "",
        annotation_ids: tuple[str, ...] = (),
    ) -> ReviewRecord:
        """Commit one explicit full-human review of the current packet.

        This uses the same durable coordination lock as Next. BA-02 remains
        the owner of queue state and the BA-03 store remains the record owner.

        Returns:
            The BA-02 durable review receipt.
        """

        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (expected_state_revision, expected_input_revision)
        ):
            raise AuditContextConflict("BA-02 review expected revisions are invalid")
        queue = self._stable_queue(context)
        with _next_lock(queue.state_path):
            queue = self._stable_queue(context)
            coordination = _read_coordination(_next_coordination_path(queue.state_path))
            if coordination["active"] is not None:
                raise AuditNextAmbiguous("BA-02 Next has an unresolved durable lease")
            before_review(queue)
            if (
                queue.state_revision != expected_state_revision
                or queue.input_revision != expected_input_revision
            ):
                raise AuditContextConflict("BA-02 review queue revision CAS failed")
            if not (
                queue.dataset._identity_admitted
                and queue.dataset.source_identity
                and queue.dataset.scan_identity
            ):
                raise CapabilityUnavailable("BA-02 review lacks typed BA-01 scan admission")
            packet = queue.current_packet
            if (
                packet is None
                or packet.packet_id != context.reference_id
                or packet.primary.episode_id != context.episode_id
            ):
                raise AuditContextConflict("BA-02 review is not bound to the current packet")
            try:
                return queue.record_review(
                    context.episode_id,
                    scope="full_episode",
                    outcome=outcome,
                    author_kind="human",
                    author_id=author_id,
                    notes=notes,
                    annotation_ids=annotation_ids,
                    operation_id=operation_id,
                )
            except (QueueConflictError, QueueOperationConflictError, QueueStateError) as exc:
                raise AuditContextConflict("BA-02 review queue commit conflicted") from exc

    def select_next(
        self,
        *,
        context: AuditSelectionContext,
        expected_state_revision: int | None = None,
        expected_input_revision: int | None = None,
        force_current: bool = False,
    ) -> CapabilityResult:
        """Select one packet without exposing queue state mutation details.

        Returns:
            A capability envelope containing a :class:`QueueNextResult`.
        """

        result = self.transact(
            context=context,
            expected_state_revision=expected_state_revision,
            expected_input_revision=expected_input_revision,
            force_current=force_current,
        )
        if isinstance(result, QueueNextResult):
            return CapabilityResult(
                "ba-02.queue.next",
                "complete",
                result,
                provider="audit_queue.AuditQueue.select_next",
            )
        return (
            result
            if isinstance(result, CapabilityResult)
            else CapabilityResult(
                "ba-02.queue.next", "failed", reason="BA-02 queue returned an invalid result"
            )
        )

    def current(self, *, context: AuditSelectionContext) -> QueueNextResult | None:
        """Read the durable current packet for a safe service replay.

        Returns:
            The current packet and selection context, or ``None`` when empty.
        """

        queue = self._stable_queue(context)
        packet = queue.current_packet
        if packet is None:
            return None
        selection = next(
            (
                item
                for item in reversed(queue.state.selection_history)
                if item.packet_id == packet.packet_id
            ),
            None,
        )
        if selection is None:
            raise CapabilityUnavailable("BA-02 selection history is unavailable")
        return self._validate_selection(
            queue,
            SelectionResult(packet=packet, context=selection),
            expected_state_revision=None,
            expected_input_revision=None,
        )

    def replay(  # noqa: C901
        self,
        *,
        operation_id: str,
        session_id: str,
        context: AuditSelectionContext,
        expected_result_digest: str = "",
    ) -> QueueNextResult:
        """Return the exact durable queue envelope for one service operation."""

        queue = self._stable_queue(context)
        lock = _next_lock(queue.state_path)
        with lock:
            queue = self._stable_queue(context)
            path = _next_coordination_path(queue.state_path)
            coordination = _read_coordination(path)
            record = coordination.get("completed", {}).get(operation_id)
            active = coordination.get("active")
            if record is None and isinstance(active, Mapping):
                if (
                    active.get("operation_id") == operation_id
                    and active.get("session_id") == session_id
                    and active.get("phase") == "queue_committed"
                ):
                    record = active
                elif active is not None:
                    raise AuditNextAmbiguous(
                        "BA-02 queue Next has an unresolved operation; reconciliation is unavailable"
                    )
            if not isinstance(record, Mapping):
                raise CapabilityUnavailable("BA-02 exact replay envelope is unavailable")
            if record.get("operation_id") != operation_id:
                raise AuditContextConflict("BA-02 replay operation identity changed")
            if record.get("session_id") != session_id:
                raise AuditContextConflict("BA-02 replay session identity changed")
            stored_context = record.get("context")
            if not isinstance(stored_context, Mapping):
                raise CapabilityUnavailable("BA-02 replay context binding is unavailable")
            try:
                immutable_context = AuditSelectionContext.from_mapping(stored_context)
            except (TypeError, KeyError, ValueError) as exc:
                raise CapabilityUnavailable("BA-02 replay context binding is malformed") from exc
            # The sidecar context is the operation's immutable queue binding.
            # The caller's current context is used only to bind the source
            # while loading the queue; it must not invalidate finalization
            # after an unrelated context CAS.
            self.binding.check_context(immutable_context)
            result = self._replay_result(
                queue,
                record,
                context=immutable_context,
                expected_result_digest=expected_result_digest,
            )
            if active is record or (
                isinstance(active, Mapping)
                and active.get("operation_id") == operation_id
                and active.get("phase") == "queue_committed"
            ):
                completed = dict(coordination.get("completed", {}))
                completed[operation_id] = dict(record)
                while len(completed) > _NEXT_COORDINATION_MAX_COMPLETED:
                    completed.pop(next(iter(completed)))
                coordination["completed"] = completed
                coordination["active"] = None
                _write_coordination(path, coordination)
            return result


@dataclass(frozen=True)
class CoverageReadAdapter:
    """Project a freshly computed BA-04 report under the same source binding."""

    binding: AuditSourceBinding
    report_provider: Callable[[], AuditHealthReport]
    # BA-04's provenance revision may be the source commit, while BA-05's
    # integer source_revision is a separate service CAS dimension.  The
    # trusted launcher supplies the former when it differs.
    report_source_revision: str | None = None

    def read(self, *, context: AuditSelectionContext) -> CapabilityResult:
        """Return deficits and evidence status; never infer missing inputs as complete."""

        self.binding.check_context(context)
        report = self.report_provider()
        if not isinstance(report, AuditHealthReport):
            raise CapabilityUnavailable("BA-04 report provider is unavailable")
        identity = report.identity
        if (
            identity.campaign_digest != self.binding.campaign_digest
            or identity.source_digest != self.binding.source_digest
            or identity.source_revision
            != (
                str(self.binding.source_revision)
                if self.report_source_revision is None
                else self.report_source_revision
            )
        ):
            raise AuditContextConflict("BA-04 report input identity changed")
        return CapabilityResult(
            "ba-04.coverage",
            "complete",
            report.to_dict(),
            provider="audit_coverage.evaluate_coverage",
        )
