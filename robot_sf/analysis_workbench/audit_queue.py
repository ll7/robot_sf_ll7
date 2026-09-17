"""Explainable, offline review-queue selection for the benchmark auditor.

The queue is a policy layer around the small BA-03 records.  In particular,
``ReviewPacket`` remains the closed BA-03 contract: ranking components,
coverage state, and random state live in :class:`QueueSelectionContext`, not in
``selection_reasons`` or an ad-hoc extension of the packet schema.

This module does not run a simulator, a shell command, a browser, a network
client, or an agent.  It consumes typed ``Signal`` values when BA-01 is
available and treats absent scan/coverage producers as explicit unavailable
inputs.  Scores are uncalibrated priorities.  They must never be read as
probabilities, defect rates, or scientific-example selection scores.
"""

# The queue contract has a deliberately broad, dataclass-heavy public surface;
# each individual record carries field-level semantics in its class docstring.
# Keep the lint gate focused on executable safety and formatting for this leaf.
# ruff: noqa: D102,D103,D105,D107,DOC201

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:  # ``fcntl`` is the fail-closed POSIX lock owner for queue state files.
    import fcntl
except ImportError:  # pragma: no cover - the supported runner is POSIX.
    fcntl = None

from robot_sf.analysis_workbench.audit_contracts import (
    ActionRecord,
    AuditContractError,
    EpisodeRef,
    Finding,
    ReviewPacket,
    ReviewRecord,
    Signal,
    finding_from_dict,
    record_from_dict,
    record_to_dict,
    signal_from_dict,
)
from robot_sf.analysis_workbench.audit_similarity import (
    INCOMPATIBLE,
    SIMILARITY_MODE_SAME_SCENARIO,
    UNKNOWN_COMPATIBILITY,
    compatible_case_similarity,
)
from robot_sf.analysis_workbench.event_alignment import (
    PAIR_COMPATIBILITY_PROFILE_VERSION,
    unavailable_pair_compatibility,
)

if TYPE_CHECKING:
    from robot_sf.analysis_workbench.audit_store import AuditStore

try:  # The import is deliberately optional: the queue has no portfolio dependency at runtime.
    from robot_sf.benchmark.case_portfolio import GRAINS as CASE_PORTFOLIO_GRAINS
except ImportError:  # pragma: no cover - minimal installations may omit benchmark extras.
    CASE_PORTFOLIO_GRAINS = frozenset(
        {"episode", "matched_planner_pair", "matched_seed_pair", "cell", "cross_cell"}
    )


QUEUE_SCHEMA_VERSION = "audit-queue.v1"
QUEUE_INPUT_SCHEMA_VERSION = "audit-queue-input.v1"
QUEUE_SELECTION_SCHEMA_VERSION = "audit-queue-selection.v1"
QUEUE_POLICY_FIXED = "fixed"
QUEUE_POLICY_ACTIVE = "active"
DEFAULT_POLICY_VERSION = "audit-queue.active.v1"
FIXED_POLICY_VERSION = "audit-queue.fixed.v1"

PRIORITY_BENCHMARK_CONFIG = "benchmark_config_defect"
PRIORITY_RELEASE_MANUSCRIPT = "release_manuscript_impact"
PRIORITY_UNEXPLAINED = "unexplained_behavior"
PRIORITY_STATISTICAL = "statistical_unusualness"
PRIORITY_PLANNER_DISAGREEMENT = "planner_disagreement"
PRIORITY_SAFETY = "safety_severity"
PRIORITY_BANDS = (
    PRIORITY_BENCHMARK_CONFIG,
    PRIORITY_RELEASE_MANUSCRIPT,
    PRIORITY_UNEXPLAINED,
    PRIORITY_STATISTICAL,
    PRIORITY_PLANNER_DISAGREEMENT,
    PRIORITY_SAFETY,
)
PRIORITY_BAND_RANK = {name: index for index, name in enumerate(PRIORITY_BANDS)}

# Public aliases make the accepted ordering convenient to inspect in a CLI or
# a notebook without requiring callers to duplicate the policy vocabulary.
BAND_BENCHMARK_CONFIG_DEFECT = PRIORITY_BENCHMARK_CONFIG
BAND_RELEASE_MANUSCRIPT_IMPACT = PRIORITY_RELEASE_MANUSCRIPT
BAND_UNEXPLAINED_BEHAVIOR = PRIORITY_UNEXPLAINED
BAND_STATISTICAL_UNUSUALNESS = PRIORITY_STATISTICAL
BAND_PLANNER_DISAGREEMENT = PRIORITY_PLANNER_DISAGREEMENT
BAND_SAFETY_SEVERITY = PRIORITY_SAFETY

CONTROL_STATUS_UNAVAILABLE = "unavailable"
CONTROL_STATUS_UNDER_REVIEW = "under_review"
CONTROL_STATUS_MET = "met"
CONTROL_STATUS_ERROR = "error"
CONTROL_STATUSES = frozenset(
    {
        CONTROL_STATUS_UNAVAILABLE,
        CONTROL_STATUS_UNDER_REVIEW,
        CONTROL_STATUS_MET,
        CONTROL_STATUS_ERROR,
    }
)
CONTROL_OUTCOMES = frozenset(
    {"success", "failure", "failed", "pass", "fail", "collision", "timeout"}
)
_MAX_INPUT_BYTES = 8 * 1024 * 1024
_MAX_JSON_DEPTH = 40
_MAX_JSON_NODES = 100_000
_MAX_STRING_BYTES = 512 * 1024

FIXED_WEIGHTS: dict[str, float] = {
    "signal_strength": 0.40,
    "benchmark_config_defect": 0.20,
    "release_impact": 0.20,
    "behavior_unexplained": 0.15,
    "statistical_unusualness": 0.10,
    "planner_disagreement": 0.10,
    "safety_severity": 0.05,
}
ACTIVE_WEIGHTS: dict[str, float] = {
    **FIXED_WEIGHTS,
    "coverage_gain": 0.20,
    "hypothesis_gain": 0.18,
    "unresolved_request": 0.15,
    "novelty": 0.16,
    "redundancy_penalty": 0.16,
    "age": 0.10,
    "defer_pressure": 0.05,
}


class AuditQueueError(ValueError):
    """Base class for malformed queue inputs or unsupported policy operations."""


class QueueInputError(AuditQueueError):
    """Raised when an input row cannot be safely admitted to the queue."""


class QueueStateError(AuditQueueError):
    """Raised when a persisted queue state is malformed or cannot be resumed."""


class QueueConflictError(AuditQueueError):
    """Raised when a queue state compare-and-swap revision is stale."""


class QueueOperationConflictError(AuditQueueError):
    """Raised when an operation ID is reused for a different queue mutation."""


# Compatibility aliases used by integrations that name conflicts after the
# BA-03 store's exceptions.
AuditQueueConflictError = QueueConflictError
AuditQueueOperationConflictError = QueueOperationConflictError


def _utc_now() -> str:
    """Return a UTC timestamp for provenance (never used as a rank tie-breaker)."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _text(value: Any, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise QueueInputError(f"{name} must be a non-empty string")
    if len(value.encode("utf-8")) > _MAX_STRING_BYTES:
        raise QueueInputError(f"{name} exceeds the input size limit")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueueInputError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError) as exc:
        raise QueueInputError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise QueueInputError(f"{name} must be a finite number")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QueueInputError(f"{name} must be a non-negative integer")
    return value


def _bounded_unit(value: Any, *, name: str, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = _finite(value, name=name)
    except QueueInputError:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        raise
    return min(1.0, max(0.0, number))


def _tuple_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QueueInputError(f"{name} must be a sequence of strings")
    result = tuple(_text(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    return tuple(dict.fromkeys(result))


def _sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    """Normalize a bounded JSON array while rejecting scalar/mapping surprises."""

    if value is None or isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QueueInputError(f"{name} must be a sequence")
    return tuple(value)


def _metadata_strings(
    metadata: Mapping[str, Any],
    name: str,
    *,
    allow_string: bool = False,
) -> tuple[str, ...]:
    """Validate one metadata collection before ranking can inspect it.

    Metadata is intentionally extensible, but collections consumed by the
    queue must not be allowed to fail later with a raw ``TypeError`` (for
    example, a mapping inside ``finding_ids`` cannot be hashed).  A scalar
    string is accepted only for the human-oriented symptom/tag conveniences;
    ID collections remain explicitly sequence-shaped.
    """

    if name not in metadata or metadata[name] is None:
        return ()
    value = metadata[name]
    if isinstance(value, str):
        if not allow_string:
            raise QueueInputError(f"candidate metadata {name} must be a sequence of strings")
        return (_text(value, name=f"candidate metadata {name}"),)
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise QueueInputError(f"candidate metadata {name} must be a sequence of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_text(item, name=f"candidate metadata {name}[{index}]"))
    return tuple(dict.fromkeys(result))


def _metadata_tags(metadata: Mapping[str, Any]) -> None:
    """Validate the tag container accepted by feature-signature extraction."""

    if "tags" not in metadata or metadata["tags"] is None:
        return
    value = metadata["tags"]
    if isinstance(value, Mapping):
        for key in value:
            _text(key, name="candidate metadata tags key")
        return
    _metadata_strings(metadata, "tags", allow_string=True)


def _metadata_alignment(metadata: Mapping[str, Any]) -> None:
    """Require alignment aliases to remain structured mappings."""

    for name in ("alignment", "event_alignment", "review_alignment", "pair_compatibility"):
        if (
            name in metadata
            and metadata[name] is not None
            and not isinstance(metadata[name], Mapping)
        ):
            raise QueueInputError(f"candidate metadata {name} must be a mapping")


def _metadata_boolean(metadata: Mapping[str, Any], name: str) -> None:
    if name in metadata and metadata[name] is not None and not isinstance(metadata[name], bool):
        raise QueueInputError(f"candidate metadata {name} must be boolean")


def _metadata_validate(metadata: Mapping[str, Any]) -> None:
    """Validate the extensible metadata fields consumed by queue policies."""

    _metadata_strings(metadata, "finding_ids")
    _metadata_strings(metadata, "feature_signatures")
    for name in ("geometry_signature", "geometry", "symptoms", "symptom"):
        _metadata_strings(metadata, name, allow_string=True)
    for name in (
        "competing_hypotheses",
        "hypothesis_evidence",
        "unresolved_requests",
        "evidence_requests",
    ):
        _metadata_strings(metadata, name)
    for name in ("review_scope", "initial_state_identity", "trace_content_sha256"):
        if name in metadata and metadata[name] is not None:
            _text(metadata[name], name=f"candidate metadata {name}")
    _metadata_tags(metadata)
    _metadata_alignment(metadata)
    if (
        "trace" in metadata
        and metadata["trace"] is not None
        and not isinstance(metadata["trace"], Mapping)
    ):
        raise QueueInputError("candidate metadata trace must be a mapping")
    for name in ("control_eligible", "ordinary", "contradictory_evidence", "contradicts_finding"):
        _metadata_boolean(metadata, name)
    for name in ("more_evidence_requested",):
        if name in metadata and metadata[name] is not None:
            _bounded_unit(metadata[name], name=f"candidate metadata {name}")


def _annotation_ids(value: Any) -> tuple[str, ...]:
    """Normalize annotation IDs without turning a scalar into characters."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise QueueInputError("annotation_ids must be a sequence of non-empty strings")
    return tuple(
        dict.fromkeys(
            _text(item, name=f"annotation_ids[{index}]") for index, item in enumerate(value)
        )
    )


def _strict_json(  # noqa: C901
    value: Any, *, path: str = "$", depth: int = 0, nodes: list[int] | None = None
) -> None:
    """Reject values that could evade bounded, offline JSON processing."""

    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > _MAX_JSON_NODES:
        raise QueueInputError("queue input exceeds the JSON node limit")
    if depth > _MAX_JSON_DEPTH:
        raise QueueInputError(f"{path} exceeds the JSON nesting limit")
    if isinstance(value, float) and not math.isfinite(value):
        raise QueueInputError(f"{path} contains a non-finite number")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > _MAX_STRING_BYTES:
            raise QueueInputError(f"{path} exceeds the string size limit")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise QueueInputError(f"{path} contains a non-string field name")
            _strict_json(item, path=f"{path}.{key}", depth=depth + 1, nodes=nodes)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _strict_json(item, path=f"{path}[{index}]", depth=depth + 1, nodes=nodes)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise QueueInputError(f"{path} contains unsupported value {type(value).__name__}")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QueueInputError(f"{name} must be a mapping")
    result = dict(value)
    _strict_json(result, path=name)
    return result


@contextmanager
def _state_lock(path: Path):
    """Hold the stable sibling lock used by every state-file CAS.

    The lock file is deliberately never removed: all writers derive the same
    sibling identity from the state path, including writers in other
    processes.  Falling back to an unlocked write would turn the revision
    check into a time-of-check/time-of-use race, so platforms without POSIX
    advisory locking fail closed.
    """

    if fcntl is None:
        raise QueueStateError("cross-process queue state locking is unavailable")
    lock_path = path.with_name(f"{path.name}.lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    except OSError as exc:
        raise QueueStateError(f"cannot open queue state lock {lock_path}: {exc}") from exc
    locked = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            raise QueueStateError(f"cannot lock queue state {path}: {exc}") from exc
        yield
    finally:
        try:
            if locked:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise QueueStateError(f"cannot unlock queue state {path}: {exc}") from exc
        finally:
            handle.close()


def _canonical(value: Any) -> str:
    """Return strict canonical JSON used for IDs, operation digests, and state."""

    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise QueueInputError(f"value is not strict JSON: {exc}") from exc
    return text


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finding_digest(findings: Sequence[Finding]) -> str:
    """Hash full finding payloads so membership/hypothesis edits become stale."""

    return _sha256(
        [record_to_dict(finding) for finding in sorted(findings, key=lambda item: item.finding_id)]
    )


def _reject_json_constant(token: str) -> Any:
    raise QueueInputError(f"invalid JSON constant: {token}")


def _read_json(path: str | Path) -> Any:
    """Read one bounded JSON document without following executable content."""

    source = Path(path)
    if not source.is_file():
        raise QueueInputError(f"input is not a regular file: {source}")
    if source.stat().st_size > _MAX_INPUT_BYTES:
        raise QueueInputError(f"input exceeds {_MAX_INPUT_BYTES} bytes")
    try:
        value = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise QueueInputError(f"cannot read queue JSON: {exc}") from exc
    _strict_json(value)
    return value


@dataclass(frozen=True, slots=True)
class CoverageDeficit:
    """Typed local input for BA-04 coverage gaps.

    BA-02 records the deficit it consumed but does not calculate protocol
    coverage.  ``source_revision`` and ``protocol_revision`` make a stale
    coverage artifact visible instead of silently reusing it.
    """

    stratum_id: str
    observed: int
    target: int
    source_revision: str = ""
    protocol_revision: str = ""
    reason: str = ""
    status: str = CONTROL_STATUS_UNDER_REVIEW
    source_id: str = ""
    protocol_id: str = ""

    def __post_init__(self) -> None:
        _text(self.stratum_id, name="coverage.stratum_id")
        observed = _nonnegative_int(self.observed, name="coverage.observed")
        target = _nonnegative_int(self.target, name="coverage.target")
        _text(self.status, name="coverage.status")
        if self.status not in CONTROL_STATUSES:
            raise QueueInputError(f"coverage.status must be one of {sorted(CONTROL_STATUSES)}")
        source_revision = _text(
            self.source_revision, name="coverage.source_revision", allow_empty=True
        )
        protocol_revision = _text(
            self.protocol_revision, name="coverage.protocol_revision", allow_empty=True
        )
        reason = _text(self.reason, name="coverage.reason", allow_empty=True)
        source_id = _text(self.source_id, name="coverage.source_id", allow_empty=True)
        protocol_id = _text(self.protocol_id, name="coverage.protocol_id", allow_empty=True)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "protocol_revision", protocol_revision)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "protocol_id", protocol_id)

    @property
    def gap(self) -> int:
        """Return the remaining target count (never negative)."""

        return max(self.target - self.observed, 0)

    @property
    def deficit(self) -> float:
        """Return a bounded coverage priority, not a coverage probability."""

        return min(1.0, self.gap / max(self.target, 1))

    @property
    def is_under_review(self) -> bool:
        return self.status == CONTROL_STATUS_UNDER_REVIEW and self.gap > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUEUE_SELECTION_SCHEMA_VERSION,
            "stratum_id": self.stratum_id,
            "observed": self.observed,
            "target": self.target,
            "source_revision": self.source_revision,
            "protocol_revision": self.protocol_revision,
            "source_id": self.source_id,
            "protocol_id": self.protocol_id,
            "reason": self.reason,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CoverageDeficit:
        payload = _mapping(value, name="coverage")
        return cls(
            stratum_id=payload.get("stratum_id", payload.get("id", "")),
            observed=payload.get("observed", 0),
            target=payload.get("target", 0),
            source_revision=payload.get("source_revision", ""),
            protocol_revision=payload.get("protocol_revision", ""),
            reason=payload.get("reason", ""),
            status=payload.get("status", CONTROL_STATUS_UNDER_REVIEW),
            source_id=payload.get("source_id", payload.get("source", "")),
            protocol_id=payload.get("protocol_id", payload.get("protocol", "")),
        )


@dataclass(frozen=True, slots=True)
class ScanSummary:
    """Small typed identity/accounting handle for a future BA-01 scan."""

    summary_id: str
    revision: int = 0
    campaign_digest: str = ""
    source_digest: str = ""
    accounting: Mapping[str, Any] = field(default_factory=dict)
    detector_accounting: Mapping[str, Any] = field(default_factory=dict)
    status: str = "available"
    missingness: tuple[str, ...] = ()
    source_revision: str = ""
    source_id: str = ""

    def __post_init__(self) -> None:
        _text(self.summary_id, name="scan_summary.summary_id")
        revision = _nonnegative_int(self.revision, name="scan_summary.revision")
        for name, value in (
            ("campaign_digest", self.campaign_digest),
            ("source_digest", self.source_digest),
        ):
            _text(value, name=f"scan_summary.{name}", allow_empty=True)
        accounting = _mapping(self.accounting, name="scan_summary.accounting")
        detector_accounting = _mapping(
            self.detector_accounting, name="scan_summary.detector_accounting"
        )
        for detector_id, details in detector_accounting.items():
            _text(detector_id, name="scan_summary.detector_accounting detector")
            if not isinstance(details, Mapping):
                raise QueueInputError("scan_summary.detector_accounting entries must be mappings")
            _strict_json(details, path=f"scan_summary.detector_accounting.{detector_id}")
        _text(self.status, name="scan_summary.status")
        if self.status not in {"available", CONTROL_STATUS_UNAVAILABLE, "error"}:
            raise QueueInputError("scan_summary.status must be available, unavailable, or error")
        source_revision = _text(
            self.source_revision, name="scan_summary.source_revision", allow_empty=True
        )
        source_id = _text(self.source_id, name="scan_summary.source_id", allow_empty=True)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "accounting", accounting)
        object.__setattr__(self, "detector_accounting", detector_accounting)
        object.__setattr__(
            self, "missingness", _tuple_strings(self.missingness, name="scan_summary.missingness")
        )
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_id", source_id)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ScanSummary:
        payload = _mapping(value, name="scan_summary")
        source_digest = payload.get("source_digest", "")
        source_id = payload.get("source_id", "")
        if not source_digest and isinstance(source_id, str) and len(source_id) == 64:
            source_digest = source_id
        return cls(
            summary_id=payload.get(
                "summary_id", payload.get("scan_id", payload.get("identity", ""))
            ),
            revision=payload.get("revision", payload.get("input_revision", 0)),
            campaign_digest=payload.get("campaign_digest", payload.get("campaign_id", "")),
            source_digest=source_digest,
            accounting=payload.get("accounting", {}),
            detector_accounting=payload.get("detector_accounting", payload.get("detectors", {})),
            status=payload.get("status", "available"),
            missingness=payload.get("missingness", ()),
            source_revision=payload.get("source_revision", ""),
            source_id=source_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUEUE_SELECTION_SCHEMA_VERSION,
            "summary_id": self.summary_id,
            "revision": self.revision,
            "campaign_digest": self.campaign_digest,
            "source_digest": self.source_digest,
            "accounting": dict(self.accounting),
            "detector_accounting": dict(self.detector_accounting),
            "status": self.status,
            "missingness": list(self.missingness),
            "source_revision": self.source_revision,
            "source_id": self.source_id,
        }


def _episode_from_mapping(value: Mapping[str, Any]) -> EpisodeRef:
    payload = dict(value)
    nested = payload.get("episode") or payload.get("episode_ref") or payload.get("primary")
    if isinstance(nested, EpisodeRef):
        return nested
    if isinstance(nested, Mapping):
        payload = {**dict(nested), **payload}
    try:
        return EpisodeRef(
            campaign_digest=payload.get("campaign_digest", payload.get("campaign_id", "")),
            source_digest=payload.get("source_digest", payload.get("source_id", "")),
            execution_id=payload.get("execution_id", ""),
            planner_id=payload.get("planner_id", ""),
            scenario_id=payload.get("scenario_id", ""),
            seed=payload.get("seed"),
            attempt=payload.get("attempt", 0),
            config_digest=payload.get("config_digest", ""),
            checkpoint_digest=payload.get("checkpoint_digest", ""),
            environment_digest=payload.get("environment_digest", ""),
            source=payload.get("source"),
            episode_id=payload.get("episode_id", ""),
        )
    except (AuditContractError, TypeError, ValueError) as exc:
        raise QueueInputError(f"invalid episode identity: {exc}") from exc


@dataclass(frozen=True, slots=True)
class QueueCandidate:
    """One queue candidate and its optional detector/trace context."""

    episode: EpisodeRef | Mapping[str, Any]
    signals: tuple[Signal | Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trace_available: bool | None = None
    trace_identity: str = ""
    stratum_id: str = ""
    outcome: str = ""

    def __post_init__(self) -> None:
        episode = (
            self.episode
            if isinstance(self.episode, EpisodeRef)
            else _episode_from_mapping(self.episode)
        )
        signals: list[Signal] = []
        signal_ids: set[str] = set()
        for index, signal in enumerate(_sequence(self.signals, name="candidate.signals")):
            try:
                item = signal if isinstance(signal, Signal) else signal_from_dict(signal)
            except (AuditContractError, KeyError, TypeError, ValueError) as exc:
                raise QueueInputError(
                    f"invalid signal at candidate {episode.episode_id}[{index}]: {exc}"
                ) from exc
            if item.signal_id in signal_ids:
                raise QueueInputError(
                    f"duplicate signal_id at candidate {episode.episode_id}: {item.signal_id}"
                )
            signal_ids.add(item.signal_id)
            signals.append(item)
        metadata = _mapping(self.metadata, name=f"candidate[{episode.episode_id}].metadata")
        _metadata_validate(metadata)
        trace_available = self.trace_available
        trace = metadata.get("trace")
        if trace_available is None and "trace_available" in metadata:
            trace_available = metadata["trace_available"]
        if trace_available is None and isinstance(trace, Mapping) and "available" in trace:
            trace_available = trace["available"]
        if trace_available is not None and not isinstance(trace_available, bool):
            raise QueueInputError("candidate.trace_available must be boolean or null")
        trace_identity = _text(
            self.trace_identity, name="candidate.trace_identity", allow_empty=True
        )
        if not trace_identity and "trace_identity" in metadata:
            trace_identity = _text(
                metadata["trace_identity"],
                name="candidate.metadata.trace_identity",
                allow_empty=True,
            )
        if not trace_identity and isinstance(trace, Mapping):
            trace_identity = _text(
                trace.get("identity", trace.get("trace_id", "")),
                name="candidate.trace.identity",
                allow_empty=True,
            )
        stratum_id = self.stratum_id or metadata.get("stratum_id", "")
        outcome = self.outcome or metadata.get("outcome", metadata.get("status", ""))
        if stratum_id:
            _text(stratum_id, name="candidate.stratum_id")
        _text(outcome, name="candidate.outcome", allow_empty=True)
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "signals", tuple(signals))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "trace_available", trace_available)
        object.__setattr__(self, "trace_identity", trace_identity)
        object.__setattr__(self, "stratum_id", stratum_id)
        object.__setattr__(self, "outcome", outcome)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> QueueCandidate:
        payload = _mapping(value, name="candidate")
        nested_episode = payload.get("episode") or payload.get("episode_ref")
        if nested_episode is None:
            nested_episode = payload
        metadata = (
            dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata", {}), Mapping)
            else payload.get("metadata", {})
        )
        # Top-level descriptive fields are accepted as input conveniences but
        # are retained under metadata so ranking remains explainable.
        for key in (
            "priority_band",
            "priority",
            "benchmark_config_defect",
            "release_manuscript_impact",
            "unexplained_behavior",
            "statistical_unusualness",
            "planner_disagreement",
            "safety_severity",
            "geometry_signature",
            "symptoms",
            "tags",
            "control_eligible",
            "ordinary",
            "more_evidence_requested",
            "competing_hypotheses",
            "contradictory_evidence",
            "finding_ids",
            "review_scope",
        ):
            if key in payload and key not in metadata:
                metadata[key] = payload[key]
        return cls(
            episode=nested_episode,
            signals=_sequence(payload.get("signals", ()), name="candidate.signals"),
            metadata=metadata,
            trace_available=payload.get("trace_available"),
            trace_identity=payload.get("trace_identity", ""),
            stratum_id=payload.get("stratum_id", ""),
            outcome=payload.get("outcome", ""),
        )

    @property
    def episode_id(self) -> str:
        return self.episode.episode_id

    @property
    def case_id(self) -> str:
        return self.episode_id

    @property
    def planner_id(self) -> str:
        return self.episode.planner_id

    @property
    def scenario_id(self) -> str:
        return self.episode.scenario_id

    @property
    def effective_stratum_id(self) -> str:
        if self.stratum_id:
            return self.stratum_id
        outcome = self.outcome or str(
            self.metadata.get("outcome", self.metadata.get("status", "unknown"))
        )
        return f"{self.planner_id}|{self.scenario_id}|{outcome}"

    @property
    def control_eligible(self) -> bool:
        explicit = self.metadata.get("control_eligible")
        if explicit is not None:
            if not isinstance(explicit, bool):
                raise QueueInputError(
                    f"candidate {self.episode_id} control_eligible must be boolean"
                )
            eligible = explicit
        else:
            ordinary = self.metadata.get("ordinary")
            if ordinary is not None:
                if not isinstance(ordinary, bool):
                    raise QueueInputError(f"candidate {self.episode_id} ordinary must be boolean")
                eligible = ordinary
            else:
                # A missing declaration is not an implicit control.  The control
                # stream must be explicitly detector-independent so a future detector
                # cannot silently reclassify ordinary rows or make a prevalence claim.
                eligible = False
        if not eligible:
            return False
        outcome = self.outcome or str(self.metadata.get("outcome", ""))
        if outcome.strip().lower() not in CONTROL_OUTCOMES:
            return False
        return True

    @property
    def feature_signatures(self) -> tuple[str, ...]:
        """Return stable descriptive signatures for active redundancy accounting."""

        values: list[str] = []
        geometry = self.metadata.get("geometry_signature", self.metadata.get("geometry", ()))
        if isinstance(geometry, str):
            geometry = (geometry,)
        if isinstance(geometry, Sequence) and not isinstance(geometry, (str, bytes)):
            values.extend(f"geometry:{item}" for item in geometry if item)
        symptoms = self.metadata.get("symptoms", self.metadata.get("symptom", ()))
        if isinstance(symptoms, str):
            symptoms = (symptoms,)
        if isinstance(symptoms, Sequence) and not isinstance(symptoms, (str, bytes)):
            values.extend(f"symptom:{item}" for item in symptoms if item)
        tags = self.metadata.get("tags", ())
        if isinstance(tags, Mapping):
            tags = tuple(key for key, item in tags.items() if item)
        if isinstance(tags, str):
            tags = (tags,)
        if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
            values.extend(f"tag:{item}" for item in tags if item)
        outcome = self.outcome or str(self.metadata.get("outcome", ""))
        if outcome:
            values.append(f"outcome:{outcome}")
        values.extend(
            f"signal:{signal.reason_code or signal.detector_id}"
            for signal in self.signals
            if signal.status == "flagged"
        )
        if self.planner_id:
            values.append(f"planner:{self.planner_id}")
        return tuple(dict.fromkeys(str(value) for value in values if value))

    @property
    def has_flagged_signal(self) -> bool:
        return any(signal.status == "flagged" for signal in self.signals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": record_to_dict(self.episode),
            "signals": [record_to_dict(signal) for signal in self.signals],
            "metadata": dict(self.metadata),
            "trace_available": self.trace_available,
            "trace_identity": self.trace_identity,
            "stratum_id": self.stratum_id,
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class QueueDataset:
    """Bounded queue input assembled from BA-03 records and producer summaries."""

    candidates: tuple[QueueCandidate | Mapping[str, Any] | EpisodeRef, ...]
    signals: tuple[Signal | Mapping[str, Any], ...] = ()
    findings: tuple[Finding | Mapping[str, Any], ...] = ()
    review_records: tuple[ReviewRecord | Mapping[str, Any], ...] = ()
    coverage_deficits: tuple[CoverageDeficit | Mapping[str, Any], ...] = ()
    scan_summary: ScanSummary | Mapping[str, Any] | None = None
    campaign_digest: str = ""
    source_digest: str = ""
    protocol_version: str = ""
    protocol_digest: str = ""
    input_revision: int = 0
    accounting: Mapping[str, Any] = field(default_factory=dict)
    missingness: tuple[str, ...] = ()
    review_context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        candidates: list[QueueCandidate] = []
        seen: set[str] = set()
        for index, candidate in enumerate(_sequence(self.candidates, name="queue.candidates")):
            if isinstance(candidate, QueueCandidate):
                item = candidate
            elif isinstance(candidate, EpisodeRef):
                item = QueueCandidate(candidate)
            else:
                try:
                    item = QueueCandidate.from_mapping(candidate)
                except (QueueInputError, TypeError, ValueError) as exc:
                    raise QueueInputError(
                        f"invalid queue candidate at index {index}: {exc}"
                    ) from exc
            if item.episode_id in seen:
                raise QueueInputError(f"duplicate queue candidate episode_id: {item.episode_id}")
            seen.add(item.episode_id)
            candidates.append(item)
        global_signals: list[Signal] = []
        global_signal_ids: set[str] = set()
        for index, signal in enumerate(_sequence(self.signals, name="queue.signals")):
            try:
                item = signal if isinstance(signal, Signal) else signal_from_dict(signal)
            except (AuditContractError, KeyError, TypeError, ValueError) as exc:
                raise QueueInputError(f"invalid global signal at index {index}: {exc}") from exc
            if item.signal_id in global_signal_ids:
                raise QueueInputError(f"duplicate global signal_id: {item.signal_id}")
            global_signal_ids.add(item.signal_id)
            global_signals.append(item)
        signal_by_id = {signal.signal_id: signal for signal in global_signals}
        for candidate in candidates:
            for signal in candidate.signals:
                existing = signal_by_id.get(signal.signal_id)
                if existing is not None and existing != signal:
                    raise QueueInputError(
                        "conflicting signal payload for signal_id "
                        f"{signal.signal_id!r} between global and local queue inputs"
                    )
                signal_by_id.setdefault(signal.signal_id, signal)
        findings: list[Finding] = []
        for index, finding in enumerate(_sequence(self.findings, name="queue.findings")):
            if isinstance(finding, Finding):
                findings.append(finding)
            elif isinstance(finding, Mapping):
                try:
                    findings.append(finding_from_dict(finding))
                except (AuditContractError, KeyError, TypeError, ValueError) as exc:
                    raise QueueInputError(f"invalid finding at index {index}: {exc}") from exc
            else:
                raise QueueInputError(f"invalid finding at index {index}")
        reviews: list[ReviewRecord] = []
        for index, review in enumerate(_sequence(self.review_records, name="queue.review_records")):
            if isinstance(review, ReviewRecord):
                reviews.append(review)
            elif isinstance(review, Mapping):
                try:
                    reviews.append(
                        ReviewRecord(
                            review_id=review["review_id"],
                            episode_id=review["episode_id"],
                            scope=review["scope"],
                            outcome=review.get("outcome", ""),
                            author_kind=review.get("author_kind", "human"),
                            author_id=review.get("author_id", ""),
                            source_revision=review.get("source_revision", 0),
                            annotation_ids=_annotation_ids(review.get("annotation_ids", ())),
                            created_at=review.get("created_at", _utc_now()),
                            notes=review.get("notes", ""),
                        )
                    )
                except (AuditContractError, KeyError, TypeError, ValueError) as exc:
                    raise QueueInputError(f"invalid review record at index {index}: {exc}") from exc
            else:
                raise QueueInputError(f"invalid review record at index {index}")
        deficits_list: list[CoverageDeficit] = []
        deficit_ids: set[str] = set()
        for index, deficit in enumerate(
            _sequence(self.coverage_deficits, name="queue.coverage_deficits")
        ):
            try:
                item = (
                    deficit
                    if isinstance(deficit, CoverageDeficit)
                    else CoverageDeficit.from_mapping(deficit)
                )
            except (QueueInputError, TypeError, ValueError) as exc:
                raise QueueInputError(f"invalid coverage deficit at index {index}: {exc}") from exc
            if item.stratum_id in deficit_ids:
                raise QueueInputError(f"duplicate coverage deficit stratum_id: {item.stratum_id}")
            deficit_ids.add(item.stratum_id)
            deficits_list.append(item)
        deficits = tuple(deficits_list)
        summary = self.scan_summary
        if summary is not None and not isinstance(summary, ScanSummary):
            summary = ScanSummary.from_mapping(summary)
        campaign_digest = self.campaign_digest or (summary.campaign_digest if summary else "")
        source_digest = self.source_digest or (summary.source_digest if summary else "")
        _text(campaign_digest, name="queue.campaign_digest", allow_empty=True)
        _text(source_digest, name="queue.source_digest", allow_empty=True)
        _text(self.protocol_version, name="queue.protocol_version", allow_empty=True)
        _text(self.protocol_digest, name="queue.protocol_digest", allow_empty=True)
        input_revision = _nonnegative_int(self.input_revision, name="queue.input_revision")
        if input_revision == 0 and summary is not None:
            input_revision = summary.revision
        supplied_accounting = _mapping(self.accounting, name="queue.accounting")
        accounting = dict(summary.accounting) if summary is not None else {}
        accounting.update(supplied_accounting)
        review_context = (
            None
            if self.review_context is None
            else _mapping(self.review_context, name="queue.review_context")
        )
        if review_context is not None:
            if isinstance(review_context.get("denominator"), int) and not isinstance(
                review_context["denominator"], bool
            ):
                accounting.setdefault("review_context_denominator", review_context["denominator"])
            selection_coverage = review_context.get("selection_coverage")
            if isinstance(selection_coverage, Mapping):
                accounting.setdefault("review_context_selection_coverage", dict(selection_coverage))
            campaign = review_context.get("campaign")
            if isinstance(campaign, Mapping):
                accounting.setdefault("review_context_campaign", dict(campaign))
        if summary is not None:
            accounting.setdefault("scan_summary_id", summary.summary_id)
            accounting.setdefault("scan_summary_revision", summary.revision)
            accounting.setdefault("detector_accounting", dict(summary.detector_accounting))
        if "expected_rows" not in accounting:
            accounting["expected_rows"] = len(candidates)
        if "indexed_rows" not in accounting:
            accounting["indexed_rows"] = len(candidates)
        missingness = list(_tuple_strings(self.missingness, name="queue.missingness"))
        if summary is not None and self.campaign_digest and summary.campaign_digest:
            if self.campaign_digest != summary.campaign_digest:
                missingness.append("scan-summary:campaign-identity-mismatch")
        if summary is not None and self.source_digest and summary.source_digest:
            if self.source_digest != summary.source_digest:
                missingness.append("scan-summary:source-identity-mismatch")
        if summary is not None:
            supplied_summary_id = supplied_accounting.get("scan_summary_id")
            if supplied_summary_id is not None and supplied_summary_id != summary.summary_id:
                missingness.append("scan-summary:accounting-id-mismatch")
            supplied_summary_revision = supplied_accounting.get("scan_summary_revision")
            if (
                supplied_summary_revision is not None
                and supplied_summary_revision != summary.revision
            ):
                missingness.append("scan-summary:accounting-revision-mismatch")
            if self.input_revision and self.input_revision != summary.revision:
                missingness.append("scan-summary:input-revision-mismatch")
            if summary.source_revision and summary.source_revision != summary.summary_id:
                missingness.append("scan-summary:source-revision-mismatch")
            if summary.source_id and source_digest and summary.source_id != source_digest:
                missingness.append("scan-summary:source-id-mismatch")
            supplied_source_revision = next(
                (
                    supplied_accounting[name]
                    for name in ("scan_source_revision", "source_revision")
                    if name in supplied_accounting
                ),
                None,
            )
            expected_source_revision = summary.source_revision or summary.summary_id
            if (
                supplied_source_revision is not None
                and supplied_source_revision != expected_source_revision
            ):
                missingness.append("scan-summary:accounting-source-revision-mismatch")
            supplied_source_id = next(
                (
                    supplied_accounting[name]
                    for name in ("scan_source_id", "source_id")
                    if name in supplied_accounting
                ),
                None,
            )
            expected_source_id = summary.source_id or source_digest
            if supplied_source_id is not None and expected_source_id:
                if supplied_source_id != expected_source_id:
                    missingness.append("scan-summary:accounting-source-id-mismatch")
        if campaign_digest:
            missingness.extend(
                f"candidate:{candidate.episode_id}:campaign-identity-mismatch"
                for candidate in candidates
                if candidate.episode.campaign_digest != campaign_digest
            )
        if source_digest:
            missingness.extend(
                f"candidate:{candidate.episode_id}:source-identity-mismatch"
                for candidate in candidates
                if candidate.episode.source_digest != source_digest
            )
        if not global_signals and not any(candidate.signals for candidate in candidates):
            missingness.append("ba-01-signals:unavailable")
        if summary is None:
            missingness.append("ba-01-scan-summary:unavailable")
        elif summary.status != "available":
            missingness.extend(
                f"ba-01-scan-summary:{item}" for item in summary.missingness or (summary.status,)
            )
        detector_accounting = accounting.get("detector_accounting", {})
        if not isinstance(detector_accounting, Mapping):
            raise QueueInputError("queue.accounting.detector_accounting must be a mapping")
        unavailable_detectors: list[str] = []
        for detector_id, details in detector_accounting.items():
            _text(detector_id, name="queue.accounting.detector_accounting detector")
            if not isinstance(details, Mapping):
                raise QueueInputError(
                    "queue.accounting.detector_accounting entries must be mappings"
                )
            _strict_json(details, path=f"queue.accounting.detector_accounting.{detector_id}")
            status = details.get("status")
            if status in {CONTROL_STATUS_UNAVAILABLE, CONTROL_STATUS_ERROR}:
                unavailable_detectors.append(detector_id)
                missingness.append(f"ba-01-detector:{detector_id}:{status}")
        declared_unavailable = accounting.get("unavailable_detectors", ())
        if isinstance(declared_unavailable, Sequence) and not isinstance(
            declared_unavailable, (str, bytes)
        ):
            for detector_id in declared_unavailable:
                if not isinstance(detector_id, str) or not detector_id:
                    raise QueueInputError(
                        "queue.accounting.unavailable_detectors must contain detector IDs"
                    )
                unavailable_detectors.append(detector_id)
                missingness.append(f"ba-01-detector:{detector_id}:unavailable")
        for signal in signal_by_id.values():
            if signal.status in {CONTROL_STATUS_UNAVAILABLE, CONTROL_STATUS_ERROR}:
                unavailable_detectors.append(signal.detector_id)
                missingness.append(f"ba-01-detector:{signal.detector_id}:{signal.status}")
        if unavailable_detectors:
            detector_ids = sorted(set(unavailable_detectors))
            declared_count = accounting.get("unavailable_detectors")
            if isinstance(declared_count, int) and not isinstance(declared_count, bool):
                # Preserve BA-01's denominator/count field and expose the
                # detector identities in a separate, unambiguous field.
                accounting["unavailable_detector_ids"] = detector_ids
            else:
                accounting["unavailable_detectors"] = detector_ids
        if not deficits:
            missingness.append("ba-04-coverage-deficits:unavailable")
        else:
            coverage_source_revisions = {
                item.source_revision for item in deficits if item.source_revision
            }
            coverage_protocol_revisions = {
                item.protocol_revision for item in deficits if item.protocol_revision
            }
            if len(coverage_source_revisions) > 1:
                missingness.append("coverage:source-revision-mismatch")
            if len(coverage_protocol_revisions) > 1:
                missingness.append("coverage:protocol-revision-mismatch")
            missingness.extend(
                f"coverage:{deficit.stratum_id}:revision-unavailable"
                for deficit in deficits
                if not deficit.source_revision or not deficit.protocol_revision
            )
            for deficit in deficits:
                if summary is not None and deficit.source_revision:
                    if deficit.source_revision != summary.summary_id:
                        missingness.append(
                            f"coverage:{deficit.stratum_id}:source-revision-mismatch"
                        )
                    if (
                        summary.source_revision
                        and deficit.source_revision != summary.source_revision
                    ):
                        missingness.append(
                            f"coverage:{deficit.stratum_id}:scan-source-revision-mismatch"
                        )
                if deficit.source_id:
                    expected_source = source_digest or (
                        summary.source_id if summary is not None else ""
                    )
                    if expected_source and deficit.source_id != expected_source:
                        missingness.append(f"coverage:{deficit.stratum_id}:source-id-mismatch")
                if self.protocol_version and deficit.protocol_revision:
                    if deficit.protocol_revision != self.protocol_version:
                        missingness.append(
                            f"coverage:{deficit.stratum_id}:protocol-revision-mismatch"
                        )
                if self.protocol_digest and deficit.protocol_id:
                    if deficit.protocol_id != self.protocol_digest:
                        missingness.append(f"coverage:{deficit.stratum_id}:protocol-id-mismatch")
        object.__setattr__(self, "candidates", tuple(candidates))
        object.__setattr__(self, "signals", tuple(global_signals))
        object.__setattr__(self, "findings", tuple(findings))
        object.__setattr__(self, "review_records", tuple(reviews))
        object.__setattr__(self, "coverage_deficits", deficits)
        object.__setattr__(self, "scan_summary", summary)
        object.__setattr__(self, "campaign_digest", campaign_digest)
        object.__setattr__(self, "source_digest", source_digest)
        object.__setattr__(self, "input_revision", input_revision)
        object.__setattr__(self, "accounting", accounting)
        object.__setattr__(self, "missingness", tuple(dict.fromkeys(missingness)))
        object.__setattr__(self, "review_context", review_context)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> QueueDataset:
        payload = _mapping(value, name="queue input")
        version = payload.get("schema_version")
        if version != QUEUE_INPUT_SCHEMA_VERSION:
            raise QueueInputError(
                "unsupported queue input schema_version: "
                f"{version!r}; expected {QUEUE_INPUT_SCHEMA_VERSION!r}"
            )
        candidates = payload.get("candidates", payload.get("episodes", ()))
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise QueueInputError("queue input candidates must be a sequence")
        return cls(
            candidates=tuple(candidates),
            signals=_sequence(payload.get("signals", ()), name="queue.signals"),
            findings=_sequence(payload.get("findings", ()), name="queue.findings"),
            review_records=_sequence(
                payload.get("review_records", payload.get("reviews", ())),
                name="queue.review_records",
            ),
            coverage_deficits=_sequence(
                payload.get("coverage_deficits", payload.get("coverage", ())),
                name="queue.coverage_deficits",
            ),
            scan_summary=payload.get("scan_summary"),
            campaign_digest=payload.get("campaign_digest", payload.get("campaign_id", "")),
            source_digest=payload.get("source_digest", payload.get("source_id", "")),
            protocol_version=payload.get("protocol_version", ""),
            protocol_digest=payload.get("protocol_digest", ""),
            input_revision=payload.get("input_revision", 0),
            accounting=payload.get("accounting", {}),
            missingness=payload.get("missingness", ()),
            review_context=payload.get("review_context"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact versioned root document accepted by ``from_mapping``."""

        return {
            "schema_version": QUEUE_INPUT_SCHEMA_VERSION,
            "campaign_digest": self.campaign_digest,
            "source_digest": self.source_digest,
            "protocol_version": self.protocol_version,
            "protocol_digest": self.protocol_digest,
            "input_revision": self.input_revision,
            "accounting": dict(self.accounting),
            "missingness": list(self.missingness),
            "scan_summary": self.scan_summary.to_dict() if self.scan_summary else None,
            "coverage_deficits": [item.to_dict() for item in self.coverage_deficits],
            "signals": [record_to_dict(item) for item in self.signals],
            "findings": [record_to_dict(item) for item in self.findings],
            "review_records": [record_to_dict(item) for item in self.review_records],
            "review_context": self.review_context,
            "candidates": [item.to_dict() for item in self.candidates],
        }

    @property
    def identity(self) -> str:
        """Return a stable input/index identity used by save-resume checks."""

        return _sha256(
            {
                "campaign_digest": self.campaign_digest,
                "source_digest": self.source_digest,
                "protocol_version": self.protocol_version,
                "protocol_digest": self.protocol_digest,
                "input_revision": self.input_revision,
                "accounting": dict(self.accounting),
                "missingness": list(self.missingness),
                "review_context": self.review_context,
                "scan_summary": self.scan_summary.to_dict() if self.scan_summary else None,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "signals": [record_to_dict(signal) for signal in self.signals],
                "findings": [record_to_dict(finding) for finding in self.findings],
                "reviews": [record_to_dict(review) for review in self.review_records],
                "coverage": [deficit.to_dict() for deficit in self.coverage_deficits],
            }
        )

    @property
    def denominator_accounting(self) -> Mapping[str, Any]:
        return self.accounting

    def signals_for(self, episode_id: str) -> tuple[Signal, ...]:
        local = [
            signal
            for signal in self.signals
            if not signal.episode_id or signal.episode_id == episode_id
        ]
        for candidate in self.candidates:
            if candidate.episode_id == episode_id:
                local.extend(candidate.signals)
                break
        seen: set[str] = set()
        return tuple(
            signal
            for signal in local
            if not (signal.signal_id in seen or seen.add(signal.signal_id))
        )

    def candidate(self, episode_id: str) -> QueueCandidate | None:
        return next((item for item in self.candidates if item.episode_id == episode_id), None)


# The shorter names are useful at the API boundary and preserve the issue's
# terminology without adding another model.
ReviewCase = QueueCandidate
QueueEntry = QueueCandidate
EpisodeCandidate = QueueCandidate
QueueInput = QueueDataset


@dataclass(frozen=True, slots=True)
class QueuePolicy:
    """Versioned fixed or active ranking policy.

    ``control_schedule`` contains finite selection positions, not a permanent
    quota.  A caller can supply a schedule appropriate to its protocol and
    preserve it in the selection context.
    """

    mode: str = QUEUE_POLICY_ACTIVE
    policy_version: str = DEFAULT_POLICY_VERSION
    weights: Mapping[str, float] = field(default_factory=lambda: dict(ACTIVE_WEIGHTS))
    max_defer_count: int = 3
    max_age: int = 10
    control_schedule: tuple[int, ...] = (0, 3, 7, 15, 31)
    max_control_selections: int | None = None
    control_enabled: bool = True

    def __post_init__(self) -> None:  # noqa: C901
        _text(self.mode, name="policy.mode")
        if self.mode not in {QUEUE_POLICY_FIXED, QUEUE_POLICY_ACTIVE}:
            raise AuditQueueError("queue policy mode must be fixed or active")
        policy_version = _text(self.policy_version, name="policy_version")
        weights = _mapping(self.weights, name="policy.weights")
        if self.mode == QUEUE_POLICY_FIXED:
            if policy_version == DEFAULT_POLICY_VERSION:
                policy_version = FIXED_POLICY_VERSION
            if weights == ACTIVE_WEIGHTS:
                weights = dict(FIXED_WEIGHTS)
        normalized: dict[str, float] = {}
        for key, value in weights.items():
            if not isinstance(key, str) or not key:
                raise AuditQueueError("policy weight names must be non-empty strings")
            number = _finite(value, name=f"policy.weights.{key}")
            if number < 0:
                raise AuditQueueError("policy weights cannot be negative")
            normalized[key] = number
        max_defer_count = _nonnegative_int(self.max_defer_count, name="policy.max_defer_count")
        max_age = _nonnegative_int(self.max_age, name="policy.max_age")
        schedule = tuple(
            _nonnegative_int(item, name="policy.control_schedule") for item in self.control_schedule
        )
        if tuple(sorted(set(schedule))) != schedule:
            raise AuditQueueError("policy.control_schedule must be sorted and unique")
        if self.max_control_selections is not None:
            _nonnegative_int(self.max_control_selections, name="policy.max_control_selections")
        if not isinstance(self.control_enabled, bool):
            raise AuditQueueError("policy.control_enabled must be boolean")
        object.__setattr__(self, "weights", normalized)
        object.__setattr__(self, "policy_version", policy_version)
        object.__setattr__(self, "max_defer_count", max_defer_count)
        object.__setattr__(self, "max_age", max_age)
        object.__setattr__(self, "control_schedule", schedule)

    @classmethod
    def fixed(cls, **overrides: Any) -> QueuePolicy:
        values: dict[str, Any] = {
            "mode": QUEUE_POLICY_FIXED,
            "policy_version": FIXED_POLICY_VERSION,
            "weights": dict(FIXED_WEIGHTS),
            "control_enabled": False,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def active(cls, **overrides: Any) -> QueuePolicy:
        values: dict[str, Any] = {
            "mode": QUEUE_POLICY_ACTIVE,
            "policy_version": DEFAULT_POLICY_VERSION,
            "weights": dict(ACTIVE_WEIGHTS),
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUEUE_SELECTION_SCHEMA_VERSION,
            "mode": self.mode,
            "policy_version": self.policy_version,
            "weights": dict(self.weights),
            "max_defer_count": self.max_defer_count,
            "max_age": self.max_age,
            "control_schedule": list(self.control_schedule),
            "max_control_selections": self.max_control_selections,
            "control_enabled": self.control_enabled,
        }


def fixed_policy(**overrides: Any) -> QueuePolicy:
    return QueuePolicy.fixed(**overrides)


def active_policy(**overrides: Any) -> QueuePolicy:
    return QueuePolicy.active(**overrides)


class FixedPolicy(QueuePolicy):
    """Convenience policy class selecting the fixed mode."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mode", QUEUE_POLICY_FIXED)
        kwargs.setdefault("policy_version", FIXED_POLICY_VERSION)
        kwargs.setdefault("weights", dict(FIXED_WEIGHTS))
        kwargs.setdefault("control_enabled", False)
        super().__init__(**kwargs)


class ActivePolicy(QueuePolicy):
    """Convenience policy class selecting the active mode."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mode", QUEUE_POLICY_ACTIVE)
        kwargs.setdefault("policy_version", DEFAULT_POLICY_VERSION)
        kwargs.setdefault("weights", dict(ACTIVE_WEIGHTS))
        super().__init__(**kwargs)


def _coerce_priority_band(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return PRIORITY_BANDS[value] if 0 <= value < len(PRIORITY_BANDS) else None
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    aliases = {
        "benchmark_defect": PRIORITY_BENCHMARK_CONFIG,
        "configuration_defect": PRIORITY_BENCHMARK_CONFIG,
        "benchmark_configuration_defect": PRIORITY_BENCHMARK_CONFIG,
        "benchmark_config": PRIORITY_BENCHMARK_CONFIG,
        "release_impact": PRIORITY_RELEASE_MANUSCRIPT,
        "manuscript_impact": PRIORITY_RELEASE_MANUSCRIPT,
        "release_or_manuscript_impact": PRIORITY_RELEASE_MANUSCRIPT,
        "unexplained": PRIORITY_UNEXPLAINED,
        "behavior_unexplained": PRIORITY_UNEXPLAINED,
        "statistical": PRIORITY_STATISTICAL,
        "outlier": PRIORITY_STATISTICAL,
        "planner": PRIORITY_PLANNER_DISAGREEMENT,
        "disagreement": PRIORITY_PLANNER_DISAGREEMENT,
        "safety": PRIORITY_SAFETY,
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in PRIORITY_BAND_RANK else None


def _keyword_band(*values: Any) -> str | None:
    text = " ".join(str(value).lower() for value in values if value is not None)
    rules = (
        (
            PRIORITY_BENCHMARK_CONFIG,
            ("benchmark", "config", "configuration", "provenance", "integrity", "telemetry"),
        ),
        (PRIORITY_RELEASE_MANUSCRIPT, ("release", "manuscript", "claim", "figure")),
        (
            PRIORITY_UNEXPLAINED,
            ("unexplained", "behavior", "behaviour", "stuck", "oscillation", "contradiction"),
        ),
        (PRIORITY_STATISTICAL, ("statistical", "outlier", "unusual", "cohort")),
        (PRIORITY_PLANNER_DISAGREEMENT, ("planner", "disagreement", "divergence")),
        (PRIORITY_SAFETY, ("safety", "collision", "clearance", "ttc", "near_miss")),
    )
    for band, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return band
    return None


def _number(metadata: Mapping[str, Any], names: Sequence[str], *, default: float = 0.0) -> float:
    for name in names:
        if name in metadata:
            return _bounded_unit(metadata[name], name=f"candidate.{name}", default=default)
    return default


def _boolish(metadata: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        if name in metadata:
            value = metadata[name]
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            return _bounded_unit(value, name=f"candidate.{name}")
    return 0.0


def _candidate_band(
    candidate: QueueCandidate, *, signals: Sequence[Signal] | None = None
) -> tuple[str, tuple[str, ...]]:
    metadata = candidate.metadata
    candidate_signals = candidate.signals if signals is None else tuple(signals)
    explicit = _coerce_priority_band(metadata.get("priority_band", metadata.get("priority")))
    if explicit is not None:
        return explicit, (f"explicit priority band {explicit}",)
    possible: list[tuple[int, str, str]] = []
    for key, band in (
        ("benchmark_config_defect", PRIORITY_BENCHMARK_CONFIG),
        ("release_manuscript_impact", PRIORITY_RELEASE_MANUSCRIPT),
        ("unexplained_behavior", PRIORITY_UNEXPLAINED),
        ("statistical_unusualness", PRIORITY_STATISTICAL),
        ("planner_disagreement", PRIORITY_PLANNER_DISAGREEMENT),
        ("safety_severity", PRIORITY_SAFETY),
    ):
        value = _number(metadata, (key, band), default=0.0)
        if value > 0:
            possible.append((PRIORITY_BAND_RANK[band], band, f"{key}={value:.3f}"))
    for signal in candidate_signals:
        if signal.status != "flagged":
            continue
        band = _coerce_priority_band(signal.reason_code) or _keyword_band(
            signal.reason_code, signal.detector_id, signal.message
        )
        if band is None:
            band = _keyword_band(signal.detector_id, signal.message)
        if band is not None:
            possible.append(
                (PRIORITY_BAND_RANK[band], band, f"signal {signal.signal_id} maps to {band}")
            )
    metadata_band = _keyword_band(
        metadata.get("reason_code"), metadata.get("reason"), metadata.get("message")
    )
    if metadata_band is not None:
        possible.append(
            (
                PRIORITY_BAND_RANK[metadata_band],
                metadata_band,
                f"metadata reason maps to {metadata_band}",
            )
        )
    if not possible:
        return PRIORITY_SAFETY, (
            "no higher-priority band was supplied; defaulting to safety severity",
        )
    first = min(possible, key=lambda item: (item[0], item[1], item[2]))
    return first[1], tuple(item[2] for item in sorted(possible))


def _signal_strength(
    candidate: QueueCandidate, *, signals: Sequence[Signal] | None = None
) -> float:
    candidate_signals = candidate.signals if signals is None else tuple(signals)
    values: list[float] = []
    for signal in candidate_signals:
        if signal.status != "flagged":
            continue
        measured = signal.measured
        candidate_values = [
            measured.get(name)
            for name in ("severity", "score", "priority", "magnitude", "normalized_severity")
            if isinstance(measured, Mapping) and name in measured
        ]
        values.extend(_bounded_unit(value, name="signal.measured") for value in candidate_values)
        if not candidate_values:
            values.append(1.0)
    return max(values, default=0.0)


def _fixed_components(
    candidate: QueueCandidate, *, signals: Sequence[Signal] | None = None
) -> dict[str, float]:
    metadata = candidate.metadata
    return {
        "signal_strength": _signal_strength(candidate, signals=signals),
        "benchmark_config_defect": _number(
            metadata, ("benchmark_config_defect", PRIORITY_BENCHMARK_CONFIG)
        ),
        "release_impact": _number(
            metadata, ("release_manuscript_impact", "release_impact", "manuscript_impact")
        ),
        "behavior_unexplained": _number(
            metadata, ("unexplained_behavior", "unexplained", "behavior_unexplained")
        ),
        "statistical_unusualness": _number(
            metadata, ("statistical_unusualness", "statistical", "outlier")
        ),
        "planner_disagreement": _number(
            metadata, ("planner_disagreement", "disagreement", "planner_divergence")
        ),
        "safety_severity": _number(metadata, ("safety_severity", "safety", "collision_severity")),
    }


def _weighted_score(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    active = [
        (key, value, weights.get(key, 0.0))
        for key, value in components.items()
        if weights.get(key, 0.0) > 0
    ]
    total_weight = sum(weight for _key, _value, weight in active)
    if total_weight <= 0:
        return 0.0
    return min(1.0, max(0.0, sum(value * weight for _key, value, weight in active) / total_weight))


@dataclass(frozen=True, slots=True)
class SelectionExplanation:
    """Visible score components and reasons for one candidate."""

    priority_band: str
    band_rank: int
    score: float
    components: Mapping[str, float] = field(default_factory=dict)
    weighted_contributions: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    selection_kind: str = "anomaly"
    starvation_override: bool = False

    def __post_init__(self) -> None:
        if self.priority_band not in PRIORITY_BAND_RANK:
            raise AuditQueueError(f"unknown priority band: {self.priority_band}")
        _text(self.selection_kind, name="selection.selection_kind")
        if self.band_rank != PRIORITY_BAND_RANK[self.priority_band]:
            raise AuditQueueError(
                "priority band rank does not match the accepted lexicographic ordering"
            )
        score = _bounded_unit(self.score, name="selection.score")
        components = {
            str(key): _bounded_unit(value, name=f"selection.components.{key}")
            for key, value in self.components.items()
        }
        contributions = {
            str(key): _finite(value, name=f"selection.contributions.{key}")
            for key, value in self.weighted_contributions.items()
        }
        if self.selection_kind not in {"anomaly", "control", "pinned"}:
            raise AuditQueueError("selection.selection_kind must be anomaly, control, or pinned")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "weighted_contributions", contributions)
        object.__setattr__(self, "reasons", _tuple_strings(self.reasons, name="selection.reasons"))

    @property
    def priority_score(self) -> float:
        return self.score

    @property
    def component_contributions(self) -> Mapping[str, float]:
        return self.weighted_contributions

    @property
    def priority_band_rank(self) -> int:
        return self.band_rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority_band": self.priority_band,
            "band_rank": self.band_rank,
            "score": self.score,
            "components": dict(self.components),
            "weighted_contributions": dict(self.weighted_contributions),
            "reasons": list(self.reasons),
            "selection_kind": self.selection_kind,
            "starvation_override": self.starvation_override,
        }


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """Candidate plus its deterministic explanation."""

    candidate: QueueCandidate
    explanation: SelectionExplanation

    @property
    def episode_id(self) -> str:
        return self.candidate.episode_id

    @property
    def score(self) -> float:
        return self.explanation.score

    @property
    def priority_band(self) -> str:
        return self.explanation.priority_band

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "candidate": self.candidate.to_dict(),
            "explanation": self.explanation.to_dict(),
        }


def _encode_rng(random_state: object) -> str:
    try:
        return json.dumps(random_state, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise QueueStateError(f"cannot encode RNG state: {exc}") from exc


def _decode_rng(value: str) -> tuple[Any, ...]:
    if not isinstance(value, str) or not value:
        raise QueueStateError("rng_state must be a serialized random.Random state")
    try:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, QueueInputError) as exc:
        raise QueueStateError(f"invalid rng_state: {exc}") from exc
    _strict_json(parsed, path="rng_state")

    def as_tuple(item: Any) -> Any:
        return tuple(as_tuple(value) for value in item) if isinstance(item, list) else item

    state = as_tuple(parsed)
    probe = random.Random()
    try:
        probe.setstate(state)
    except (TypeError, ValueError) as exc:
        raise QueueStateError(f"invalid rng_state structure: {exc}") from exc
    return state


@dataclass(frozen=True, slots=True)
class QueueSelectionContext:
    """Structured queue provenance kept separate from BA-03 ``ReviewPacket``."""

    selection_id: str
    packet_id: str
    primary_episode_id: str
    policy_mode: str
    policy_version: str
    input_revision: int
    input_identity: str
    campaign_digest: str = ""
    source_digest: str = ""
    execution_id: str = ""
    config_digest: str = ""
    checkpoint_digest: str = ""
    environment_digest: str = ""
    seed: int | str | None = None
    protocol_version: str = ""
    protocol_digest: str = ""
    scan_summary_id: str = ""
    scan_summary_revision: int = 0
    coverage_revision: str = ""
    stratum_id: str = ""
    review_scope: str = "episode"
    selection_kind: str = "anomaly"
    control_reason: str = ""
    priority_band: str = PRIORITY_SAFETY
    band_rank: int = PRIORITY_BAND_RANK[PRIORITY_SAFETY]
    priority_score: float = 0.0
    components: Mapping[str, float] = field(default_factory=dict)
    weighted_contributions: Mapping[str, float] = field(default_factory=dict)
    selection_reasons: tuple[str, ...] = ()
    missingness: tuple[str, ...] = ()
    rng_state: str = ""
    accounting: Mapping[str, Any] = field(default_factory=dict)
    selection_index: int = 0
    stale_inputs: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_now)
    policy_identity: str = ""

    def __post_init__(self) -> None:
        _text(self.selection_id, name="selection.selection_id")
        _text(self.packet_id, name="selection.packet_id")
        _text(self.primary_episode_id, name="selection.primary_episode_id")
        _text(self.policy_mode, name="selection.policy_mode")
        if self.policy_mode not in {QUEUE_POLICY_FIXED, QUEUE_POLICY_ACTIVE}:
            raise QueueStateError("selection.policy_mode must be fixed or active")
        _text(self.policy_version, name="selection.policy_version")
        _text(self.policy_identity, name="selection.policy_identity", allow_empty=True)
        input_revision = _nonnegative_int(self.input_revision, name="selection.input_revision")
        _text(self.input_identity, name="selection.input_identity")
        for name, value in (
            ("campaign_digest", self.campaign_digest),
            ("source_digest", self.source_digest),
            ("execution_id", self.execution_id),
            ("config_digest", self.config_digest),
            ("checkpoint_digest", self.checkpoint_digest),
            ("environment_digest", self.environment_digest),
            ("protocol_version", self.protocol_version),
            ("protocol_digest", self.protocol_digest),
            ("scan_summary_id", self.scan_summary_id),
            ("coverage_revision", self.coverage_revision),
            ("stratum_id", self.stratum_id),
            ("control_reason", self.control_reason),
            ("created_at", self.created_at),
        ):
            _text(value, name=f"selection.{name}", allow_empty=True)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, (int, str))
        ):
            raise QueueStateError("selection.seed must be an integer, string, or null")
        scan_revision = _nonnegative_int(
            self.scan_summary_revision, name="selection.scan_summary_revision"
        )
        selection_index = _nonnegative_int(self.selection_index, name="selection.selection_index")
        _text(self.selection_kind, name="selection.selection_kind")
        _text(self.review_scope, name="selection.review_scope")
        if self.review_scope not in CASE_PORTFOLIO_GRAINS:
            raise QueueStateError(f"unsupported review scope: {self.review_scope}")
        if self.selection_kind not in {"anomaly", "control", "pinned"}:
            raise QueueStateError("selection.selection_kind must be anomaly, control, or pinned")
        if (
            self.priority_band not in PRIORITY_BAND_RANK
            or self.band_rank != PRIORITY_BAND_RANK[self.priority_band]
        ):
            raise QueueStateError("selection priority band/rank is inconsistent")
        components = _mapping(self.components, name="selection.components")
        weighted_contributions = _mapping(
            self.weighted_contributions, name="selection.weighted_contributions"
        )
        object.__setattr__(self, "input_revision", input_revision)
        object.__setattr__(self, "scan_summary_revision", scan_revision)
        object.__setattr__(self, "selection_index", selection_index)
        object.__setattr__(
            self,
            "priority_score",
            _bounded_unit(self.priority_score, name="selection.priority_score"),
        )
        object.__setattr__(
            self,
            "components",
            {
                key: _bounded_unit(value, name=f"selection.components.{key}")
                for key, value in components.items()
            },
        )
        object.__setattr__(
            self,
            "weighted_contributions",
            {
                key: _finite(value, name=f"selection.weighted_contributions.{key}")
                for key, value in weighted_contributions.items()
            },
        )
        object.__setattr__(
            self,
            "selection_reasons",
            _tuple_strings(self.selection_reasons, name="selection.selection_reasons"),
        )
        object.__setattr__(
            self, "missingness", _tuple_strings(self.missingness, name="selection.missingness")
        )
        object.__setattr__(
            self, "stale_inputs", _tuple_strings(self.stale_inputs, name="selection.stale_inputs")
        )
        object.__setattr__(
            self, "accounting", _mapping(self.accounting, name="selection.accounting")
        )
        _text(self.rng_state, name="selection.rng_state", allow_empty=True)

    @property
    def score(self) -> float:
        return self.priority_score

    @property
    def reasons(self) -> tuple[str, ...]:
        return self.selection_reasons

    @property
    def cohort(self) -> str:
        """Return the stable control/anomaly stratum used as the cohort key."""

        return self.stratum_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUEUE_SELECTION_SCHEMA_VERSION,
            "selection_id": self.selection_id,
            "packet_id": self.packet_id,
            "primary_episode_id": self.primary_episode_id,
            "policy_mode": self.policy_mode,
            "policy_version": self.policy_version,
            "policy_identity": self.policy_identity,
            "input_revision": self.input_revision,
            "input_identity": self.input_identity,
            "campaign_digest": self.campaign_digest,
            "source_digest": self.source_digest,
            "execution_id": self.execution_id,
            "config_digest": self.config_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "environment_digest": self.environment_digest,
            "seed": self.seed,
            "protocol_version": self.protocol_version,
            "protocol_digest": self.protocol_digest,
            "scan_summary_id": self.scan_summary_id,
            "scan_summary_revision": self.scan_summary_revision,
            "coverage_revision": self.coverage_revision,
            "stratum_id": self.stratum_id,
            "review_scope": self.review_scope,
            "selection_kind": self.selection_kind,
            "control_reason": self.control_reason,
            "priority_band": self.priority_band,
            "band_rank": self.band_rank,
            "priority_score": self.priority_score,
            "components": dict(self.components),
            "weighted_contributions": dict(self.weighted_contributions),
            "selection_reasons": list(self.selection_reasons),
            "missingness": list(self.missingness),
            "rng_state": self.rng_state,
            "accounting": dict(self.accounting),
            "selection_index": self.selection_index,
            "stale_inputs": list(self.stale_inputs),
            "created_at": self.created_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> QueueSelectionContext:
        payload = _mapping(value, name="selection context")
        return cls(
            selection_id=payload["selection_id"],
            packet_id=payload["packet_id"],
            primary_episode_id=payload["primary_episode_id"],
            policy_mode=payload.get("policy_mode", QUEUE_POLICY_ACTIVE),
            policy_version=payload.get("policy_version", DEFAULT_POLICY_VERSION),
            policy_identity=payload.get("policy_identity", ""),
            input_revision=payload.get("input_revision", 0),
            input_identity=payload.get("input_identity", "unknown"),
            campaign_digest=payload.get("campaign_digest", ""),
            source_digest=payload.get("source_digest", ""),
            execution_id=payload.get("execution_id", ""),
            config_digest=payload.get("config_digest", ""),
            checkpoint_digest=payload.get("checkpoint_digest", ""),
            environment_digest=payload.get("environment_digest", ""),
            seed=payload.get("seed"),
            protocol_version=payload.get("protocol_version", ""),
            protocol_digest=payload.get("protocol_digest", ""),
            scan_summary_id=payload.get("scan_summary_id", ""),
            scan_summary_revision=payload.get("scan_summary_revision", 0),
            coverage_revision=payload.get("coverage_revision", ""),
            stratum_id=payload.get("stratum_id", ""),
            review_scope=payload.get("review_scope", "episode"),
            selection_kind=payload.get("selection_kind", "anomaly"),
            control_reason=payload.get("control_reason", ""),
            priority_band=payload.get("priority_band", PRIORITY_SAFETY),
            band_rank=payload.get("band_rank", PRIORITY_BAND_RANK[PRIORITY_SAFETY]),
            priority_score=payload.get("priority_score", payload.get("score", 0.0)),
            components=payload.get("components", {}),
            weighted_contributions=payload.get(
                "weighted_contributions", payload.get("component_contributions", {})
            ),
            selection_reasons=payload.get("selection_reasons", payload.get("reasons", ())),
            missingness=payload.get("missingness", ()),
            rng_state=payload.get("rng_state", ""),
            accounting=payload.get("accounting", {}),
            selection_index=payload.get("selection_index", 0),
            stale_inputs=payload.get("stale_inputs", ()),
            created_at=payload.get("created_at", _utc_now()),
        )


QueueSelectionRecord = QueueSelectionContext
SelectionContext = QueueSelectionContext


@dataclass(frozen=True, slots=True)
class QueueState:
    """Lossless, JSON-only state needed to resume queue selection."""

    state_revision: int = 0
    input_revision: int = 0
    input_identity: str = ""
    rng_state: str = ""
    selection_index: int = 0
    current_packet_id: str = ""
    previous_packet_ids: tuple[str, ...] = ()
    packet_payloads: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    selection_history: tuple[QueueSelectionContext, ...] = ()
    presented_counts: Mapping[str, int] = field(default_factory=dict)
    defer_counts: Mapping[str, int] = field(default_factory=dict)
    defer_until: Mapping[str, int] = field(default_factory=dict)
    skip_until: Mapping[str, int] = field(default_factory=dict)
    pinned_ids: tuple[str, ...] = ()
    more_evidence_requests: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    control_draws: int = 0
    control_schedule_cursor: int = 0
    operations: Mapping[str, str] = field(default_factory=dict)
    policy_identity: str = ""
    candidate_first_seen: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        for name, value in (
            ("state_revision", self.state_revision),
            ("input_revision", self.input_revision),
            ("selection_index", self.selection_index),
            ("control_draws", self.control_draws),
            ("control_schedule_cursor", self.control_schedule_cursor),
        ):
            _nonnegative_int(value, name=f"state.{name}")
        _text(self.input_identity, name="state.input_identity", allow_empty=True)
        _text(self.policy_identity, name="state.policy_identity", allow_empty=True)
        _text(self.rng_state, name="state.rng_state", allow_empty=True)
        _text(self.current_packet_id, name="state.current_packet_id", allow_empty=True)
        if not isinstance(self.packet_payloads, Mapping):
            raise QueueStateError("state.packet_payloads must be a mapping")
        if not isinstance(self.operations, Mapping):
            raise QueueStateError("state.operations must be a mapping")
        object.__setattr__(
            self,
            "previous_packet_ids",
            _tuple_strings(self.previous_packet_ids, name="state.previous_packet_ids"),
        )
        object.__setattr__(
            self, "pinned_ids", _tuple_strings(self.pinned_ids, name="state.pinned_ids")
        )
        packet_payloads: dict[str, Mapping[str, Any]] = {}
        for key, value in self.packet_payloads.items():
            try:
                packet_key = _text(key, name="state.packet_payloads key")
                payload = _mapping(value, name=f"state.packet_payloads.{packet_key}")
                packet = record_from_dict(payload)
            except (AuditContractError, KeyError, TypeError, ValueError) as exc:
                raise QueueStateError(f"corrupt persisted packet snapshot {key!r}: {exc}") from exc
            if not isinstance(packet, ReviewPacket):
                raise QueueStateError(f"packet snapshot {packet_key} is not a review_packet record")
            if packet.packet_id != packet_key:
                raise QueueStateError(
                    f"packet snapshot key {packet_key} does not match packet_id {packet.packet_id}"
                )
            packet_payloads[packet_key] = payload
        if self.current_packet_id and self.current_packet_id not in packet_payloads:
            raise QueueStateError(f"current packet snapshot is missing: {self.current_packet_id}")
        missing_history = [
            packet_id
            for packet_id in (*self.previous_packet_ids,)
            if packet_id not in packet_payloads
        ]
        if missing_history:
            raise QueueStateError(
                "previous packet snapshot is missing: " + ", ".join(missing_history)
            )
        object.__setattr__(self, "packet_payloads", packet_payloads)
        history = tuple(
            item
            if isinstance(item, QueueSelectionContext)
            else QueueSelectionContext.from_mapping(item)
            for item in _sequence(self.selection_history, name="state.selection_history")
        )
        for item in history:
            if item.packet_id not in packet_payloads:
                raise QueueStateError(
                    f"selection history references missing packet snapshot: {item.packet_id}"
                )
            try:
                packet = record_from_dict(packet_payloads[item.packet_id])
            except (AuditContractError, KeyError, TypeError, ValueError) as exc:
                raise QueueStateError(
                    f"corrupt historical packet snapshot {item.packet_id}: {exc}"
                ) from exc
            if (
                not isinstance(packet, ReviewPacket)
                or packet.primary.episode_id != item.primary_episode_id
            ):
                raise QueueStateError(
                    f"selection history primary does not match packet snapshot: {item.packet_id}"
                )
        object.__setattr__(self, "selection_history", history)
        object.__setattr__(
            self,
            "candidate_first_seen",
            self._counts(self.candidate_first_seen, "candidate_first_seen"),
        )
        if any(
            first_seen > self.selection_index for first_seen in self.candidate_first_seen.values()
        ):
            raise QueueStateError(
                "state.candidate_first_seen cannot be later than state.selection_index"
            )
        object.__setattr__(
            self, "presented_counts", self._counts(self.presented_counts, "presented_counts")
        )
        object.__setattr__(self, "defer_counts", self._counts(self.defer_counts, "defer_counts"))
        object.__setattr__(self, "defer_until", self._counts(self.defer_until, "defer_until"))
        object.__setattr__(self, "skip_until", self._counts(self.skip_until, "skip_until"))
        requests: dict[str, tuple[str, ...]] = {}
        if not isinstance(self.more_evidence_requests, Mapping):
            raise QueueStateError("state.more_evidence_requests must be a mapping")
        for key, values in self.more_evidence_requests.items():
            requests[_text(key, name="state.request key")] = _tuple_strings(
                values, name=f"state.requests.{key}"
            )
        object.__setattr__(self, "more_evidence_requests", requests)
        object.__setattr__(
            self,
            "operations",
            {
                _text(key, name="state.operations key"): _text(value, name="state.operation digest")
                for key, value in self.operations.items()
            },
        )

    @staticmethod
    def _counts(value: Mapping[str, int], name: str) -> dict[str, int]:
        if not isinstance(value, Mapping):
            raise QueueStateError(f"state.{name} must be a mapping")
        result: dict[str, int] = {}
        for key, item in value.items():
            result[_text(key, name=f"state.{name} key")] = _nonnegative_int(
                item, name=f"state.{name}.{key}"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "state_revision": self.state_revision,
            "input_revision": self.input_revision,
            "input_identity": self.input_identity,
            "policy_identity": self.policy_identity,
            "rng_state": self.rng_state,
            "selection_index": self.selection_index,
            "current_packet_id": self.current_packet_id,
            "previous_packet_ids": list(self.previous_packet_ids),
            "packet_payloads": {key: dict(value) for key, value in self.packet_payloads.items()},
            "selection_history": [item.to_dict() for item in self.selection_history],
            "candidate_first_seen": dict(self.candidate_first_seen),
            "presented_counts": dict(self.presented_counts),
            "defer_counts": dict(self.defer_counts),
            "defer_until": dict(self.defer_until),
            "skip_until": dict(self.skip_until),
            "pinned_ids": list(self.pinned_ids),
            "more_evidence_requests": {
                key: list(value) for key, value in self.more_evidence_requests.items()
            },
            "control_draws": self.control_draws,
            "control_schedule_cursor": self.control_schedule_cursor,
            "operations": dict(self.operations),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> QueueState:
        payload = _mapping(value, name="queue state")
        version = payload.get("schema_version")
        if version != QUEUE_SCHEMA_VERSION:
            raise QueueStateError(f"unsupported queue state schema: {version!r}")
        return cls(
            state_revision=payload.get("state_revision", 0),
            input_revision=payload.get("input_revision", 0),
            input_identity=payload.get("input_identity", ""),
            policy_identity=payload.get("policy_identity", ""),
            rng_state=payload.get("rng_state", ""),
            selection_index=payload.get("selection_index", 0),
            current_packet_id=payload.get("current_packet_id", ""),
            previous_packet_ids=payload.get("previous_packet_ids", ()),
            packet_payloads=payload.get("packet_payloads", {}),
            selection_history=payload.get("selection_history", ()),
            candidate_first_seen=payload.get("candidate_first_seen", {}),
            presented_counts=payload.get("presented_counts", {}),
            defer_counts=payload.get("defer_counts", {}),
            defer_until=payload.get("defer_until", {}),
            skip_until=payload.get("skip_until", {}),
            pinned_ids=payload.get("pinned_ids", ()),
            more_evidence_requests=payload.get("more_evidence_requests", {}),
            control_draws=payload.get("control_draws", 0),
            control_schedule_cursor=payload.get("control_schedule_cursor", 0),
            operations=payload.get("operations", {}),
        )


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Packet plus its separate structured selection context."""

    packet: ReviewPacket
    context: QueueSelectionContext
    explanation: SelectionExplanation | None = None

    @property
    def review_packet(self) -> ReviewPacket:
        return self.packet

    @property
    def selection(self) -> QueueSelectionContext:
        return self.context

    def to_dict(self) -> dict[str, Any]:
        result = {
            "packet": record_to_dict(self.packet),
            "selection": self.context.to_dict(),
        }
        if self.explanation is not None:
            result["explanation"] = self.explanation.to_dict()
        return result


class AuditQueue:
    """Offline fixed/active queue with durable actions and resumable state."""

    def __init__(
        self,
        dataset: QueueDataset | Mapping[str, Any] | Iterable[Any],
        *,
        policy: QueuePolicy | None = None,
        seed: int = 0,
        rng: random.Random | None = None,
        store: AuditStore | None = None,
        state_path: str | Path | None = None,
        actor_id: str = "audit-queue",
    ) -> None:
        if isinstance(dataset, QueueDataset):
            normalized = dataset
        elif isinstance(dataset, Mapping):
            normalized = QueueDataset.from_mapping(dataset)
        else:
            normalized = QueueDataset(tuple(dataset))
        self.dataset = normalized
        self.policy = policy or QueuePolicy.active()
        self._policy_identity = _sha256(self.policy.to_dict())
        if rng is not None and not isinstance(rng, random.Random):
            raise AuditQueueError("rng must be random.Random")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AuditQueueError("seed must be an integer")
        self._rng = rng or random.Random(seed)
        self.store = store
        self.actor_id = _text(actor_id, name="actor_id")
        self.state_path = Path(state_path) if state_path is not None else None
        self._persisted_revision = 0
        self._stale_inputs: tuple[str, ...] = ()
        self.state = QueueState(
            input_revision=self.dataset.input_revision,
            input_identity=self.dataset.identity,
            policy_identity=self._policy_identity,
            rng_state=_encode_rng(self._rng.getstate()),
            candidate_first_seen={item.episode_id: 0 for item in self.dataset.candidates},
        )
        if self.state_path is not None and self.state_path.exists():
            self._load_state_from_disk()
        else:
            self._ensure_candidate_first_seen()

    @classmethod
    def from_input(cls, *args: Any, **kwargs: Any) -> AuditQueue:
        return cls(*args, **kwargs)

    @property
    def input_revision(self) -> int:
        return self.dataset.input_revision

    @property
    def state_revision(self) -> int:
        return self.state.state_revision

    @property
    def current_packet(self) -> ReviewPacket | None:
        payload = self.state.packet_payloads.get(self.state.current_packet_id)
        if payload is None:
            return None
        try:
            packet = record_from_dict(payload)
        except (AuditContractError, KeyError, TypeError, ValueError) as exc:
            raise QueueStateError(
                f"corrupt current packet snapshot {self.state.current_packet_id}: {exc}"
            ) from exc
        if not isinstance(packet, ReviewPacket):
            raise QueueStateError(
                f"current packet snapshot is not a review_packet record: {self.state.current_packet_id}"
            )
        return packet

    @property
    def stale_inputs(self) -> tuple[str, ...]:
        return self._stale_inputs

    def _current_primary_id(self) -> str:
        """Resolve the current packet's primary without confusing packet/case IDs."""

        packet_id = self.state.current_packet_id
        if not packet_id:
            return ""
        payload = self.state.packet_payloads.get(packet_id)
        if isinstance(payload, Mapping):
            primary = payload.get("primary")
            if isinstance(primary, Mapping):
                episode_id = primary.get("episode_id")
                if isinstance(episode_id, str):
                    return episode_id
        return packet_id

    def _load_state_from_disk(self) -> None:
        assert self.state_path is not None
        if self.state_path.is_symlink():
            raise QueueStateError(f"queue state path must not be a symlink: {self.state_path}")
        try:
            payload = _read_json(self.state_path)
            state = QueueState.from_mapping(payload)
            if state.rng_state:
                self._rng.setstate(_decode_rng(state.rng_state))
        except (OSError, QueueInputError, QueueStateError, TypeError, ValueError) as exc:
            raise QueueStateError(f"cannot resume queue state {self.state_path}: {exc}") from exc
        self.state = state
        self._ensure_candidate_first_seen(default_index=0)
        self._persisted_revision = state.state_revision
        self._stale_inputs = self._compare_input_identity(state)

    def _ensure_candidate_first_seen(self, *, default_index: int | None = None) -> None:
        first_seen = dict(self.state.candidate_first_seen)
        default = self.state.selection_index if default_index is None else default_index
        for candidate in self.dataset.candidates:
            first_seen.setdefault(candidate.episode_id, default)
        if first_seen != self.state.candidate_first_seen:
            self.state = replace(self.state, candidate_first_seen=first_seen)

    def _compare_input_identity(self, state: QueueState) -> tuple[str, ...]:  # noqa: C901
        stale: list[str] = []
        if state.policy_identity != self._policy_identity:
            stale.append("stale-policy-semantics")
        if state.input_identity and state.input_identity != self.dataset.identity:
            stale.append("stale-input-revision")
        if state.input_revision != self.dataset.input_revision:
            stale.append("stale-input-revision")
        old_context = state.selection_history[-1] if state.selection_history else None
        summary = self.dataset.scan_summary
        if old_context is not None:
            if (
                old_context.policy_mode != self.policy.mode
                or old_context.policy_version != self.policy.policy_version
            ):
                stale.append("stale-policy-version")
            if old_context.policy_identity != self._policy_identity:
                stale.append("stale-policy-semantics")
            if old_context.scan_summary_id and (
                summary is None or old_context.scan_summary_id != summary.summary_id
            ):
                stale.append("stale-scan-summary")
            if summary is not None and old_context.scan_summary_revision != summary.revision:
                stale.append("stale-scan-summary-revision")
            current_coverage_revision = (
                _sha256([item.to_dict() for item in self.dataset.coverage_deficits])
                if self.dataset.coverage_deficits
                else ""
            )
            if old_context.coverage_revision != current_coverage_revision:
                stale.append("stale-coverage-deficit")
        if old_context is not None:
            old_finding_digest = old_context.accounting.get("finding_revision_digest")
            if old_finding_digest and old_finding_digest != _finding_digest(self.dataset.findings):
                stale.append("stale-finding-update")
        return tuple(dict.fromkeys(stale))

    def update_dataset(
        self, dataset: QueueDataset | Mapping[str, Any] | Iterable[Any]
    ) -> tuple[str, ...]:
        """Replace producer inputs while preserving selection history and flagging staleness."""

        new_dataset = (
            dataset
            if isinstance(dataset, QueueDataset)
            else QueueDataset.from_mapping(dataset)
            if isinstance(dataset, Mapping)
            else QueueDataset(tuple(dataset))
        )
        old_identity = self.dataset.identity
        old_summary = self.dataset.scan_summary
        old_findings = _finding_digest(self.dataset.findings)
        self.dataset = new_dataset
        self._ensure_candidate_first_seen(default_index=self.state.selection_index)
        stale = list(self._stale_inputs)
        if old_identity != new_dataset.identity:
            stale.append("stale-input-revision")
        if old_summary != new_dataset.scan_summary:
            stale.append("stale-scan-summary")
            if (
                old_summary is not None
                and new_dataset.scan_summary is not None
                and old_summary.summary_id == new_dataset.scan_summary.summary_id
                and old_summary.revision != new_dataset.scan_summary.revision
            ):
                stale.append("stale-scan-summary-revision")
        new_findings = _finding_digest(new_dataset.findings)
        if old_findings != new_findings:
            stale.append("stale-finding-update")
        self._stale_inputs = tuple(dict.fromkeys(stale))
        return self._stale_inputs

    def _reviewed_ids(self) -> set[str]:
        # Only a human's full-episode receipt grants coverage credit.  A
        # detector/agent proposal or interval inspection remains useful
        # evidence for redundancy, but it must not make a control stratum look
        # reviewed.
        return {
            review.episode_id
            for review in self.dataset.review_records
            if review.author_kind == "human" and review.scope == "full_episode"
        }

    def _finding_for(self, candidate: QueueCandidate) -> tuple[Finding, ...]:
        ids = set(_metadata_strings(candidate.metadata, "finding_ids"))
        return tuple(
            finding
            for finding in self.dataset.findings
            if candidate.episode_id in finding.candidate_members
            or candidate.episode_id in finding.confirmed_members
            or finding.finding_id in ids
        )

    def _active_components(  # noqa: C901
        self, candidate: QueueCandidate, *, signals: Sequence[Signal] | None = None
    ) -> dict[str, float]:
        candidate_signals = candidate.signals if signals is None else tuple(signals)
        components = _fixed_components(candidate, signals=candidate_signals)
        reviewed_ids = self._reviewed_ids()
        candidate_signatures = set(candidate.feature_signatures)
        candidate_signatures.update(
            f"signal:{signal.reason_code or signal.detector_id}"
            for signal in candidate_signals
            if signal.status == "flagged"
        )
        reviewed_signatures: set[str] = set()
        for reviewed_id in reviewed_ids:
            reviewed = self.dataset.candidate(reviewed_id)
            if reviewed is not None:
                reviewed_signatures.update(reviewed.feature_signatures)
                reviewed_signatures.update(
                    f"signal:{signal.reason_code or signal.detector_id}"
                    for signal in self.dataset.signals_for(reviewed_id)
                    if signal.status == "flagged"
                )
        findings = self._finding_for(candidate)
        known_members = {
            member
            for finding in findings
            for member in (*finding.candidate_members, *finding.confirmed_members)
        }
        shared_reviewed = len(candidate_signatures & reviewed_signatures)
        redundancy = 0.0
        if candidate_signatures:
            redundancy = min(1.0, shared_reviewed / len(candidate_signatures))
        if known_members:
            redundancy = max(
                redundancy, min(1.0, len(known_members) / max(len(self.dataset.candidates), 1))
            )
        novel_override = _number(candidate.metadata, ("novelty", "novelty_score"))
        if novel_override:
            novelty = novel_override
        elif not candidate_signatures:
            novelty = 0.0
        else:
            novelty = 1.0 - redundancy
        deficits = {
            item.stratum_id: item for item in self.dataset.coverage_deficits if item.is_under_review
        }
        deficit = deficits.get(candidate.effective_stratum_id)
        coverage_gain = (
            deficit.deficit
            if deficit
            else _number(candidate.metadata, ("coverage_gain", "coverage_deficit"))
        )
        competing = candidate.metadata.get(
            "competing_hypotheses", candidate.metadata.get("hypothesis_evidence", ())
        )
        hypothesis_gain = _number(
            candidate.metadata, ("hypothesis_gain", "competing_hypothesis_gain")
        )
        if (
            hypothesis_gain == 0.0
            and isinstance(competing, Sequence)
            and not isinstance(competing, (str, bytes))
        ):
            hypothesis_gain = min(1.0, len(competing) / 3.0)
        if candidate.metadata.get("contradictory_evidence") or candidate.metadata.get(
            "contradicts_finding"
        ):
            hypothesis_gain = max(hypothesis_gain, 1.0)
        requests = candidate.metadata.get(
            "unresolved_requests", candidate.metadata.get("evidence_requests", ())
        )
        request_score = _number(
            candidate.metadata, ("unresolved_request", "more_evidence_requested")
        )
        if isinstance(requests, Sequence) and not isinstance(requests, (str, bytes)):
            request_score = max(request_score, min(1.0, len(requests) / 2.0))
        if candidate.episode_id in self.state.more_evidence_requests:
            request_score = 1.0
        for finding in findings:
            if finding.finding_id in self.state.more_evidence_requests:
                request_score = 1.0
            if finding.status in {"proposed", "under_investigation"} and finding.hypotheses:
                hypothesis_gain = max(hypothesis_gain, min(1.0, len(finding.hypotheses) / 3.0))
        first_seen = self.state.candidate_first_seen.get(
            candidate.episode_id, self.state.selection_index
        )
        age = min(
            1.0,
            max(self.state.selection_index - first_seen, 0) / max(self.policy.max_age, 1),
        )
        presented = self.state.presented_counts.get(candidate.episode_id, 0)
        age = max(age, min(1.0, presented / max(self.policy.max_age, 1)))
        defer_count = self.state.defer_counts.get(candidate.episode_id, 0)
        defer_pressure = (
            min(1.0, defer_count / max(self.policy.max_defer_count, 1))
            if self.policy.max_defer_count
            else 1.0
            if defer_count
            else 0.0
        )
        components.update(
            {
                "coverage_gain": coverage_gain,
                "hypothesis_gain": hypothesis_gain,
                "unresolved_request": request_score,
                "novelty": novelty,
                "redundancy_penalty": redundancy,
                "age": age,
                "defer_pressure": defer_pressure,
            }
        )
        return components

    def _explanation(
        self, candidate: QueueCandidate, *, selection_kind: str = "anomaly"
    ) -> SelectionExplanation:
        signals = self.dataset.signals_for(candidate.episode_id)
        band, band_reasons = _candidate_band(candidate, signals=signals)
        components = (
            _fixed_components(candidate, signals=signals)
            if self.policy.mode == QUEUE_POLICY_FIXED
            else self._active_components(candidate, signals=signals)
        )
        score_components = dict(components)
        if self.policy.mode == QUEUE_POLICY_ACTIVE:
            score_components["redundancy_penalty"] = -components["redundancy_penalty"]
        weights = self.policy.weights
        total_weight = sum(
            weight for key, weight in weights.items() if weight > 0 and key in score_components
        )
        contributions = {
            key: value * weights.get(key, 0.0) / total_weight
            for key, value in score_components.items()
            if weights.get(key, 0.0) > 0 and total_weight > 0
        }
        score = _weighted_score(score_components, weights)
        defer_count = self.state.defer_counts.get(candidate.episode_id, 0)
        first_seen = self.state.candidate_first_seen.get(
            candidate.episode_id, self.state.selection_index
        )
        age = self.state.selection_index - first_seen
        presented_count = self.state.presented_counts.get(candidate.episode_id, 0)
        starvation_limit = self.policy.max_age * (presented_count + 1)
        starvation = defer_count >= self.policy.max_defer_count or (age >= starvation_limit)
        reasons = [f"priority band: {band} (lexicographic rank {PRIORITY_BAND_RANK[band]})"]
        reasons.extend(band_reasons)
        reasons.extend(
            f"{key} contribution={value:.3f}" for key, value in contributions.items() if value
        )
        if self.policy.mode == QUEUE_POLICY_ACTIVE:
            reasons.append(
                "active score is a heuristic priority, not a probability or information-gain estimate"
            )
            if components.get("redundancy_penalty", 0.0):
                reasons.append(
                    "repeated evidence lowers priority but does not remove the candidate"
                )
            if components.get("novelty", 0.0) > 0.5:
                reasons.append(
                    "novel geometry/planner/outcome or contradictory evidence restores priority"
                )
        if starvation:
            reasons.append(
                "age/defer limit reached; starvation safeguard may lift the candidate across bands"
            )
        if selection_kind == "control":
            reasons.append(
                "control selection ignores detector flags and samples within an observed stratum"
            )
        if selection_kind == "pinned":
            reasons.append("pinned case takes precedence over automatic ranking")
        return SelectionExplanation(
            priority_band=band,
            band_rank=PRIORITY_BAND_RANK[band],
            score=score,
            components={key: max(0.0, value) for key, value in components.items()},
            weighted_contributions=contributions,
            reasons=tuple(reasons),
            selection_kind=selection_kind,
            starvation_override=starvation,
        )

    def _eligible(self, candidate: QueueCandidate, *, exclude_current: bool) -> bool:
        if exclude_current and candidate.episode_id == self._current_primary_id():
            return False
        if self.state.skip_until.get(candidate.episode_id, -1) > self.state.selection_index:
            return False
        if self.state.defer_until.get(candidate.episode_id, -1) > self.state.selection_index:
            # Explicit age/defer limits prevent a deferred case from being
            # hidden forever; it becomes eligible at its configured limit.
            if self.state.defer_counts.get(candidate.episode_id, 0) < self.policy.max_defer_count:
                return False
        return True

    def rank_candidates(self, *, include_current: bool = True) -> list[RankedCandidate]:
        """Return deterministic anomaly ranking with explanations."""

        candidates = [
            item
            for item in self.dataset.candidates
            if self._eligible(item, exclude_current=not include_current)
        ]
        ranked = [(item, self._explanation(item)) for item in candidates]
        ranked.sort(
            key=lambda pair: (
                0 if pair[1].starvation_override else 1,
                pair[1].band_rank,
                -pair[1].score,
                pair[0].episode_id,
            )
        )
        return [
            RankedCandidate(candidate=item, explanation=explanation) for item, explanation in ranked
        ]

    rank = rank_candidates

    def _control_candidates(self) -> list[QueueCandidate]:
        return [
            item
            for item in self.dataset.candidates
            if item.control_eligible and self._eligible(item, exclude_current=True)
        ]

    def _next_control_deficit(
        self, candidates: Sequence[QueueCandidate]
    ) -> tuple[CoverageDeficit | None, list[QueueCandidate]]:
        deficits = [item for item in self.dataset.coverage_deficits if item.is_under_review]
        deficits.sort(key=lambda item: (-item.deficit, item.stratum_id))
        for deficit in deficits:
            matches = sorted(
                (item for item in candidates if item.effective_stratum_id == deficit.stratum_id),
                key=lambda item: item.episode_id,
            )
            if matches:
                return deficit, matches
        return None, sorted(candidates, key=lambda item: item.episode_id)

    def _control_due(
        self, candidates: Sequence[QueueCandidate]
    ) -> tuple[bool, CoverageDeficit | None, list[QueueCandidate]]:
        if not self.policy.control_enabled or not candidates:
            return False, None, []
        if (
            self.policy.max_control_selections is not None
            and self.state.control_draws >= self.policy.max_control_selections
        ):
            return False, None, []
        deficit, matches = self._next_control_deficit(candidates)
        schedule_due = (
            self.state.control_schedule_cursor < len(self.policy.control_schedule)
            and self.state.selection_index
            >= self.policy.control_schedule[self.state.control_schedule_cursor]
        )
        # The finite schedule is primary.  A deficit can also trigger a draw
        # once no scheduled position was missed, which keeps a declared gap
        # visible without manufacturing a permanent quota.
        due = schedule_due or bool(deficit and self.state.control_draws == 0)
        return due, deficit, matches

    @staticmethod
    def _alignment_for(candidate: QueueCandidate) -> Mapping[str, Any] | None:
        for name in ("alignment", "event_alignment", "review_alignment", "pair_compatibility"):
            value = candidate.metadata.get(name)
            if isinstance(value, Mapping):
                return value
        return None

    @staticmethod
    def _sha256_text(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdefABCDEF" for char in value)
        )

    def _canonical_peer_alignment(  # noqa: C901, PLR0912, PLR0915
        self,
        primary: QueueCandidate,
        candidate: QueueCandidate,
        alignment: Mapping[str, Any] | None,
    ) -> tuple[bool, tuple[str, ...]]:
        """Require the owner-produced pair contract before exposing a peer.

        A caller-provided ``compatible: true`` bit is not evidence.  The
        queue admits a peer only when the canonical profile, provenance gate,
        source trace identity, and full initial-state equivalence record are
        present and bind to both queue candidates.
        """

        if alignment is None:
            return False, ("alignment-unavailable",)
        outer_alignment = alignment
        canonical = alignment
        nested_compatibility = alignment.get("compatibility")
        if isinstance(nested_compatibility, Mapping):
            canonical = nested_compatibility
            if alignment.get("schema_version") != "review-alignment.v1":
                reasons = ["alignment-wrapper-profile-mismatch"]
            else:
                reasons = []
                digest = alignment.get("alignment_sha256")
                if not self._sha256_text(digest):
                    reasons.append("alignment-content-identity-unavailable")
                else:
                    try:
                        expected_digest = _sha256(
                            {
                                "comparison_grain": alignment.get("comparison_grain"),
                                "tolerances": alignment.get("tolerances"),
                                "compatibility": nested_compatibility,
                                "anchor": alignment.get("anchor"),
                                "durations_s": alignment.get("durations_s"),
                                "interpretation": alignment.get("interpretation"),
                            }
                        )
                    except QueueInputError:
                        expected_digest = ""
                    if digest.lower() != expected_digest:
                        reasons.append("alignment-content-identity-mismatch")
        elif isinstance(alignment.get("pair_compatibility"), Mapping):
            canonical = alignment["pair_compatibility"]
            reasons = []
        else:
            reasons = []
        if canonical.get("profile_version") != PAIR_COMPATIBILITY_PROFILE_VERSION:
            reasons.append("alignment-profile-mismatch")
        if canonical.get("status") != "available":
            reasons.append("alignment-unavailable")
        grain = canonical.get("comparison_grain")
        if not isinstance(grain, Mapping) or grain.get("grain_id") != "matched_planner_pair":
            reasons.append("alignment-grain-mismatch")
        elif (
            grain.get("left_role") != "primary_trace"
            or grain.get("right_role") != "comparison_trace"
        ):
            reasons.append("alignment-role-mismatch")

        provenance = outer_alignment.get("provenance")
        gate = canonical.get("provenance_gate")
        if not isinstance(provenance, Mapping) and isinstance(gate, Mapping):
            # The lower-level event-alignment owner emits trace identities in
            # the gate rather than the review-alignment wrapper's provenance.
            provenance = {
                "left_trace_id": gate.get("left_trace_id"),
                "right_trace_id": gate.get("right_trace_id"),
            }
        if not isinstance(provenance, Mapping):
            reasons.append("alignment-provenance-unavailable")
        else:
            for name in ("left_artifact_id", "right_artifact_id"):
                # Artifact IDs are available in the review-alignment wrapper;
                # direct pair-compatibility records bind the same relation via
                # their canonical left/right trace IDs.
                if name not in provenance:
                    continue
                if not isinstance(provenance.get(name), str) or not provenance[name]:
                    reasons.append(f"alignment-{name}-unavailable")
            if isinstance(nested_compatibility, Mapping):
                for name in ("left_artifact_id", "right_artifact_id"):
                    if not isinstance(provenance.get(name), str) or not provenance[name]:
                        reasons.append(f"alignment-{name}-unavailable")
            if provenance.get("left_trace_id") != primary.trace_identity:
                reasons.append("alignment-left-trace-identity-mismatch")
            if provenance.get("right_trace_id") != candidate.trace_identity:
                reasons.append("alignment-right-trace-identity-mismatch")
        if isinstance(nested_compatibility, Mapping) and not isinstance(
            outer_alignment.get("provenance"), Mapping
        ):
            reasons.append("alignment-provenance-unavailable")

        if not primary.trace_identity:
            reasons.append("primary-trace-identity-unavailable")
        if not candidate.trace_identity:
            reasons.append("peer-trace-identity-unavailable")
        if primary.episode.seed != candidate.episode.seed:
            reasons.append("peer-seed-mismatch")

        checks = gate.get("checks") if isinstance(gate, Mapping) else None
        availability = gate.get("availability") if isinstance(gate, Mapping) else None
        if not isinstance(gate, Mapping) or gate.get("status") != "available":
            reasons.append("alignment-provenance-gate-unavailable")
        if isinstance(gate, Mapping) and gate.get("compatible") is not True:
            reasons.append("alignment-provenance-gate-incompatible")
        if isinstance(gate, Mapping) and gate.get("comparison_grain") != "matched_planner_pair":
            reasons.append("alignment-gate-grain-mismatch")
        if not isinstance(checks, Mapping):
            reasons.append("alignment-provenance-checks-unavailable")
        else:
            required_checks = (
                "scenario_id_equal",
                "coordinate_frame_equal",
                "units_equal",
                "seed_equal",
                "planner_id_different",
                "map_id_present",
                "map_id_equal",
                "horizon_present",
                "horizon_equal",
                "config_digest_present",
                "time_step_s_present",
                "time_step_s_equal",
            )
            reasons.extend(
                f"alignment-check-failed:{name}"
                for name in required_checks
                if checks.get(name) is not True
            )
        if not isinstance(availability, Mapping):
            reasons.append("alignment-provenance-availability-unavailable")
        else:
            for name in ("map_id", "horizon", "config_digest", "time_step_s"):
                entry = availability.get(name)
                if not isinstance(entry, Mapping) or entry.get("status") != "available":
                    reasons.append(f"alignment-source-field-unavailable:{name}")
                elif entry.get("left") is None or entry.get("right") is None:
                    reasons.append(f"alignment-source-field-missing:{name}")
        contracts = gate.get("time_step_contracts") if isinstance(gate, Mapping) else None
        if not isinstance(contracts, Mapping) or any(
            not isinstance(contracts.get(side), Mapping)
            or contracts[side].get("status") != "available"
            for side in ("left", "right")
        ):
            reasons.append("alignment-time-step-contract-unavailable")

        right_source = canonical.get("right_source_trace")
        if not isinstance(right_source, Mapping):
            reasons.append("peer-source-trace-unavailable")
        else:
            if right_source.get("status") != "available":
                reasons.append("peer-source-trace-unavailable")
            if right_source.get("schema_version") != "simulation_trace_export.v1":
                reasons.append("peer-source-trace-profile-mismatch")
            if right_source.get("trace_id") != candidate.trace_identity:
                reasons.append("peer-source-trace-identity-mismatch")
            if not self._sha256_text(right_source.get("content_sha256")):
                reasons.append("peer-source-content-identity-unavailable")
            receipt = right_source.get("content_receipt")
            if not isinstance(receipt, Mapping) or not isinstance(
                receipt.get("content_contract"), Mapping
            ):
                reasons.append("peer-source-content-receipt-unavailable")
            source = right_source.get("source")
            if not isinstance(source, Mapping):
                reasons.append("peer-source-identity-unavailable")
            else:
                expected = {
                    "episode_id": candidate.episode_id,
                    "scenario_id": candidate.scenario_id,
                    "planner_id": candidate.planner_id,
                    "seed": candidate.episode.seed,
                }
                for name, expected_value in expected.items():
                    if name == "episode_id":
                        accepted_episode_ids = {
                            candidate.episode_id,
                            candidate.episode.execution_id,
                        }
                        if source.get(name) not in accepted_episode_ids:
                            reasons.append(f"peer-source-{name}-mismatch")
                        continue
                    if source.get(name) != expected_value:
                        reasons.append(f"peer-source-{name}-mismatch")
            expected_content = candidate.metadata.get("trace_content_sha256")
            if expected_content is not None:
                if not self._sha256_text(expected_content):
                    reasons.append("peer-trace-content-identity-malformed")
                elif expected_content.lower() != str(right_source.get("content_sha256")).lower():
                    reasons.append("peer-trace-content-identity-mismatch")

        if isinstance(gate, Mapping):
            for name, source_name in (
                ("left_content_sha256", "primary-trace-content"),
                ("right_content_sha256", "peer-trace-content"),
            ):
                content_digest = gate.get(name)
                if not self._sha256_text(content_digest):
                    reasons.append(f"{source_name}-identity-unavailable")
                elif (
                    name == "right_content_sha256"
                    and isinstance(right_source, Mapping)
                    and self._sha256_text(right_source.get("content_sha256"))
                    and content_digest.lower() != right_source["content_sha256"].lower()
                ):
                    reasons.append("peer-trace-content-identity-mismatch")
                elif name == "left_content_sha256":
                    expected_primary_content = primary.metadata.get("trace_content_sha256")
                    if expected_primary_content is not None:
                        if not self._sha256_text(expected_primary_content):
                            reasons.append("primary-trace-content-identity-malformed")
                        elif content_digest.lower() != expected_primary_content.lower():
                            reasons.append("primary-trace-content-identity-mismatch")

        initial = canonical.get("initial_state_equivalence")
        if (
            not isinstance(initial, Mapping)
            or initial.get("status") != "available"
            or initial.get("equivalent") is not True
        ):
            reasons.append("initial-state-equivalence-unavailable")
        else:
            # These fields are the canonical event-alignment owner's complete
            # initial-state identity/equivalence receipt.  A bare equivalent
            # flag is deliberately insufficient.
            required_initial_fields = (
                "robot_position_delta_m",
                "robot_velocity_delta_mps",
                "robot_heading_delta_rad",
                "robot_radius_delta_m",
                "actor_id_sets_equal",
                "actor_position_delta_m",
                "actor_velocity_delta_mps",
                "actor_radius_delta_m",
                "max_actor_position_delta_m",
                "max_actor_velocity_delta_mps",
                "max_actor_radius_delta_m",
                "position_tolerance_m",
                "heading_tolerance_rad",
            )
            reasons.extend(
                f"initial-state-field-unavailable:{name}"
                for name in required_initial_fields
                if name not in initial
            )
            if initial.get("actor_id_sets_equal") is not True:
                reasons.append("initial-state-actor-identity-mismatch")
            numeric_fields = (
                "robot_position_delta_m",
                "robot_velocity_delta_mps",
                "robot_heading_delta_rad",
                "robot_radius_delta_m",
                "max_actor_position_delta_m",
                "max_actor_velocity_delta_mps",
                "max_actor_radius_delta_m",
                "position_tolerance_m",
                "heading_tolerance_rad",
            )
            for name in numeric_fields:
                value = initial.get(name)
                if value is None and name.startswith("max_actor_"):
                    continue
                try:
                    _finite(value, name=f"initial-state.{name}")
                except QueueInputError:
                    reasons.append(f"initial-state-field-malformed:{name}")
            for name, allow_none in (
                ("actor_position_delta_m", False),
                ("actor_velocity_delta_mps", False),
                ("actor_radius_delta_m", True),
            ):
                values = initial.get(name)
                if not isinstance(values, Mapping):
                    reasons.append(f"initial-state-field-malformed:{name}")
                    continue
                for actor_id, value in values.items():
                    if not isinstance(actor_id, str):
                        reasons.append(f"initial-state-field-malformed:{name}")
                        continue
                    if value is None:
                        if not allow_none:
                            reasons.append(f"initial-state-field-malformed:{name}")
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        reasons.append(f"initial-state-field-malformed:{name}")
                    else:
                        try:
                            _finite(value, name=f"initial-state.{name}.{actor_id}")
                        except QueueInputError:
                            reasons.append(f"initial-state-field-malformed:{name}")
            expected_initial = candidate.metadata.get("initial_state_identity")
            if expected_initial is not None:
                if not isinstance(expected_initial, str) or not expected_initial:
                    reasons.append("initial-state-identity-malformed")
                elif expected_initial != _sha256(initial):
                    reasons.append("initial-state-identity-mismatch")
        return not reasons, tuple(dict.fromkeys(reasons))

    def _peer_candidates(  # noqa: C901
        self, primary: QueueCandidate
    ) -> tuple[tuple[EpisodeRef, ...], tuple[str, ...]]:
        peers: list[EpisodeRef] = []
        missingness: list[str] = []
        for candidate in sorted(self.dataset.candidates, key=lambda item: item.episode_id):
            if candidate.episode_id == primary.episode_id:
                continue
            if candidate.scenario_id != primary.scenario_id:
                continue
            if candidate.trace_available is False:
                missingness.append(f"peer:{candidate.episode_id}:trace-unavailable")
                continue
            if candidate.trace_available is None and not candidate.trace_identity:
                # Do not present a peer as a matched trace when no trace
                # capability was declared.  The primary remains reviewable.
                missingness.append(f"peer:{candidate.episode_id}:trace-unavailable")
                continue
            similarity = compatible_case_similarity(
                primary.episode, candidate.episode, mode=SIMILARITY_MODE_SAME_SCENARIO
            )
            alignment = self._alignment_for(candidate)
            if alignment is None:
                alignment = unavailable_pair_compatibility(
                    "peer_alignment_unavailable", comparison_grain="matched_planner_pair"
                )
            if similarity.compatibility == INCOMPATIBLE:
                missingness.append(f"peer:{candidate.episode_id}:incompatible-identity")
                continue
            if similarity.compatibility == UNKNOWN_COMPATIBILITY:
                missingness.append(f"peer:{candidate.episode_id}:compatibility-unknown")
                continue
            compatible, alignment_reasons = self._canonical_peer_alignment(
                primary, candidate, alignment
            )
            if not compatible:
                missingness.extend(
                    f"peer:{candidate.episode_id}:{reason}" for reason in alignment_reasons
                )
                missingness.append(f"peer:{candidate.episode_id}:alignment-incompatible")
                continue
            if candidate.planner_id == primary.planner_id:
                missingness.append(f"peer:{candidate.episode_id}:same-planner")
                continue
            peers.append(candidate.episode)
        return tuple(peers), tuple(missingness)

    def _review_scope(self, primary: QueueCandidate, peers: Sequence[EpisodeRef]) -> str:
        declared = primary.metadata.get("review_scope")
        if isinstance(declared, str) and declared in CASE_PORTFOLIO_GRAINS:
            return declared
        return "matched_planner_pair" if peers else "episode"

    def _packet(
        self,
        candidate: QueueCandidate,
        explanation: SelectionExplanation,
        *,
        control_reason: str = "",
    ) -> tuple[ReviewPacket, tuple[str, ...]]:
        peers, peer_missingness = self._peer_candidates(candidate)
        signals = self.dataset.signals_for(candidate.episode_id)
        missingness = list(
            dict.fromkeys((*self.dataset.missingness, *peer_missingness, *self._stale_inputs))
        )
        if not any(
            key in candidate.metadata
            for key in ("release_manuscript_impact", "release_impact", "manuscript_impact")
        ):
            missingness.append("release-manuscript-impact:unknown")
        if candidate.trace_available is False or (
            candidate.trace_available is None and not candidate.trace_identity
        ):
            missingness.append(f"primary:{candidate.episode_id}:trace-unavailable")
        reasons = list(explanation.reasons)
        if control_reason:
            reasons.append(f"control reason: {control_reason}")
        if peer_missingness:
            reasons.append(
                "some peer traces were omitted because compatibility or trace capability was unavailable"
            )
        packet_reasons = tuple(dict.fromkeys(reasons))
        peer_alignment = [
            {
                "episode_id": item.episode_id,
                "trace_identity": item.trace_identity,
                "alignment": self._alignment_for(item),
            }
            for item in sorted(self.dataset.candidates, key=lambda value: value.episode_id)
            if item.episode_id != candidate.episode_id
            and item.scenario_id == candidate.scenario_id
            and self._alignment_for(item) is not None
        ]
        review_scope = self._review_scope(candidate, peers)
        packet_identity = {
            "campaign_digest": self.dataset.campaign_digest,
            "source_digest": self.dataset.source_digest,
            "primary": record_to_dict(candidate.episode),
            "peers": [record_to_dict(peer) for peer in peers],
            "signals": [record_to_dict(signal) for signal in signals],
            "selection_reasons": list(packet_reasons),
            "missingness": list(missingness),
            "primary_alignment": self._alignment_for(candidate),
            "peer_alignment": peer_alignment,
            "selection": explanation.to_dict(),
            "control_reason": control_reason,
            "policy_version": self.policy.policy_version,
            "input_revision": self.dataset.input_revision,
            "scope": review_scope,
        }
        packet_id = "review-packet-" + _sha256(packet_identity)
        packet = ReviewPacket(
            packet_id=packet_id,
            primary=candidate.episode,
            peers=peers,
            signals=signals,
            selection_reasons=packet_reasons,
            missingness=tuple(missingness),
            policy_version=self.policy.policy_version,
            input_revision=self.dataset.input_revision,
        )
        return packet, tuple(missingness)

    def _persist_packet(self, packet: ReviewPacket, *, operation_id: str) -> None:
        if self.store is None:
            return
        existing = self.store.get(packet.packet_id, include_deleted=True)
        if existing is not None:
            if existing.deleted or existing.record != packet:
                raise QueueConflictError(
                    f"packet ID collision with a different durable packet: {packet.packet_id}"
                )
            return
        self.store.save(
            packet,
            operation_id=operation_id,
            expected_revision=0,
            actor="agent",
            actor_id=self.actor_id,
        )

    def _persist_state(self, *, expected_revision: int | None = None) -> None:
        if self.state_path is None:
            return
        path = self.state_path
        if path.is_symlink():
            raise QueueStateError(f"queue state path must not be a symlink: {path}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise QueueStateError(
                f"cannot prepare queue state directory {path.parent}: {exc}"
            ) from exc
        expected = self._persisted_revision if expected_revision is None else expected_revision
        with _state_lock(path):
            if path.exists():
                try:
                    current = QueueState.from_mapping(_read_json(path)).state_revision
                except (
                    OSError,
                    QueueInputError,
                    QueueStateError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise QueueStateError(f"cannot compare queue state {path}: {exc}") from exc
                if current != expected:
                    raise QueueConflictError(
                        f"queue state revision conflict: expected {expected}, current {current}"
                    )
            elif expected not in {0, self._persisted_revision}:
                raise QueueConflictError(
                    f"queue state does not exist at expected revision {expected}"
                )
            payload = _canonical(self.state.to_dict()) + "\n"
            fd, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                self._persisted_revision = self.state.state_revision
            except OSError as exc:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise QueueStateError(f"cannot persist queue state: {exc}") from exc

    def save_state(self, *, expected_revision: int | None = None) -> int:
        """Persist lossless JSON state with compare-and-swap semantics."""

        self._persist_state(expected_revision=expected_revision)
        return self.state.state_revision

    save = save_state

    def _record_action(
        self,
        action_type: str,
        target_id: str,
        details: Mapping[str, Any],
        *,
        actor_kind: str = "human",
        actor_id: str | None = None,
        operation_id: str,
    ) -> ActionRecord:
        action_digest = _sha256(
            {
                "operation_id": operation_id,
                "action_type": action_type,
                "target_id": target_id,
                "details": dict(details),
                "actor_kind": actor_kind,
                "actor_id": actor_id or self.actor_id,
            }
        )
        action = ActionRecord(
            action_id="action-" + action_digest,
            action_type=action_type,
            actor_kind=actor_kind,
            actor_id=actor_id or self.actor_id,
            target_id=target_id,
            details=dict(details),
        )
        if self.store is not None:
            existing = self.store.get(action.action_id)
            if existing is not None and isinstance(existing.record, ActionRecord):
                return existing.record
            self.store.save(
                action,
                operation_id=operation_id,
                expected_revision=0,
                actor=actor_kind,
                actor_id=actor_id or self.actor_id,
            )
        return action

    def _advance_state(
        self, *, context: QueueSelectionContext, packet: ReviewPacket, operation_id: str
    ) -> None:
        previous = list(self.state.previous_packet_ids)
        if self.state.current_packet_id:
            previous.append(self.state.current_packet_id)
        packets = dict(self.state.packet_payloads)
        packets[packet.packet_id] = record_to_dict(packet)
        presented = dict(self.state.presented_counts)
        presented[context.primary_episode_id] = presented.get(context.primary_episode_id, 0) + 1
        operations = dict(self.state.operations)
        operations[operation_id] = _sha256(
            {"packet_id": packet.packet_id, "selection_id": context.selection_id}
        )
        cursor = self.state.control_schedule_cursor
        if (
            context.selection_kind == "control"
            and cursor < len(self.policy.control_schedule)
            and self.state.selection_index >= self.policy.control_schedule[cursor]
        ):
            cursor += 1
        control_draws = self.state.control_draws + (1 if context.selection_kind == "control" else 0)
        self.state = replace(
            self.state,
            state_revision=self.state.state_revision + 1,
            input_revision=self.dataset.input_revision,
            input_identity=self.dataset.identity,
            policy_identity=self._policy_identity,
            rng_state=_encode_rng(self._rng.getstate()),
            selection_index=self.state.selection_index + 1,
            current_packet_id=packet.packet_id,
            previous_packet_ids=tuple(previous),
            packet_payloads=packets,
            selection_history=(*self.state.selection_history, context),
            presented_counts=presented,
            control_draws=control_draws,
            control_schedule_cursor=cursor,
            operations=operations,
        )
        self._persist_state()

    def select_next(self, *, force_current: bool = False) -> SelectionResult | None:
        """Select and persist the next packet; no ``ReviewRecord`` is created."""

        eligible = [
            item
            for item in self.dataset.candidates
            if self._eligible(item, exclude_current=not force_current)
        ]
        if not eligible and not force_current:
            eligible = list(self.dataset.candidates)
        if not eligible:
            return None
        selection_kind = "anomaly"
        control_reason = ""
        candidate: QueueCandidate
        pinned = [item for item in eligible if item.episode_id in self.state.pinned_ids]
        if pinned:
            candidate = sorted(pinned, key=lambda item: item.episode_id)[0]
            selection_kind = "pinned"
            control_reason = "manual pin takes precedence over policy ranking"
        else:
            reviewed_ids = self._reviewed_ids()
            controls = [
                item
                for item in eligible
                if item.control_eligible and item.episode_id not in reviewed_ids
            ]
            due, deficit, control_pool = self._control_due(controls)
            if due and control_pool:
                candidate = control_pool[self._rng.randrange(len(control_pool))]
                selection_kind = "control"
                control_reason = (
                    f"under-reviewed stratum {deficit.stratum_id} has {deficit.gap} remaining target observations"
                    if deficit
                    else "finite control schedule position reached"
                )
            else:
                ranked = self.rank_candidates(include_current=force_current)
                candidate = (
                    ranked[0].candidate
                    if ranked
                    else sorted(eligible, key=lambda item: item.episode_id)[0]
                )
        explanation = self._explanation(candidate, selection_kind=selection_kind)
        packet, missingness = self._packet(candidate, explanation, control_reason=control_reason)
        context = QueueSelectionContext(
            selection_id="selection-"
            + _sha256(
                {
                    "packet_id": packet.packet_id,
                    "index": self.state.selection_index,
                    "input": self.dataset.identity,
                }
            ),
            packet_id=packet.packet_id,
            primary_episode_id=candidate.episode_id,
            policy_mode=self.policy.mode,
            policy_version=self.policy.policy_version,
            policy_identity=self._policy_identity,
            input_revision=self.dataset.input_revision,
            input_identity=self.dataset.identity,
            campaign_digest=candidate.episode.campaign_digest,
            source_digest=candidate.episode.source_digest,
            execution_id=candidate.episode.execution_id,
            config_digest=candidate.episode.config_digest,
            checkpoint_digest=candidate.episode.checkpoint_digest,
            environment_digest=candidate.episode.environment_digest,
            seed=candidate.episode.seed,
            protocol_version=self.dataset.protocol_version,
            protocol_digest=self.dataset.protocol_digest,
            scan_summary_id=self.dataset.scan_summary.summary_id
            if self.dataset.scan_summary
            else "",
            scan_summary_revision=self.dataset.scan_summary.revision
            if self.dataset.scan_summary
            else 0,
            coverage_revision=_sha256([item.to_dict() for item in self.dataset.coverage_deficits])
            if self.dataset.coverage_deficits
            else "",
            stratum_id=candidate.effective_stratum_id,
            review_scope=self._review_scope(candidate, packet.peers),
            selection_kind=selection_kind,
            control_reason=control_reason,
            priority_band=explanation.priority_band,
            band_rank=explanation.band_rank,
            priority_score=explanation.score,
            components=explanation.components,
            weighted_contributions=explanation.weighted_contributions,
            selection_reasons=packet.selection_reasons,
            missingness=missingness,
            rng_state=_encode_rng(self._rng.getstate()),
            accounting={
                **dict(self.dataset.accounting),
                "finding_revision_digest": _finding_digest(self.dataset.findings),
            },
            selection_index=self.state.selection_index,
            stale_inputs=self._stale_inputs,
        )
        operation_id = f"queue-select-{self.state.selection_index}-{packet.packet_id}"
        self._persist_packet(packet, operation_id=f"{operation_id}:packet")
        self._record_action(
            "queue_select",
            candidate.episode_id,
            {
                "packet_id": packet.packet_id,
                "selection_id": context.selection_id,
                "human_review_granted": False,
                "selection_kind": selection_kind,
            },
            actor_kind="agent",
            operation_id=f"{operation_id}:action",
        )
        self._advance_state(context=context, packet=packet, operation_id=operation_id)
        return SelectionResult(packet=packet, context=context, explanation=explanation)

    next = select_next
    select = select_next
    next_best_review = select_next

    def resume(self, *, reload: bool = True) -> SelectionResult | None:
        """Return the saved current packet or select the first packet if none exists."""

        if reload and self.state_path is not None and self.state_path.exists():
            self._load_state_from_disk()
        packet = self.current_packet
        if packet is None:
            if self.state.current_packet_id:
                raise QueueStateError(
                    "current packet snapshot is missing: " + self.state.current_packet_id
                )
            return self.select_next()
        context = next(
            (
                item
                for item in reversed(self.state.selection_history)
                if item.packet_id == packet.packet_id
            ),
            None,
        )
        if context is None:
            raise QueueStateError(
                "selection history is missing for current packet snapshot: " + packet.packet_id
            )
        if self._stale_inputs:
            missingness = tuple(dict.fromkeys((*packet.missingness, *self._stale_inputs)))
            packet = replace(
                packet,
                missingness=missingness,
                selection_reasons=(
                    *packet.selection_reasons,
                    "resumed packet has stale producer input; ranking must be refreshed",
                ),
            )
            context = replace(context, missingness=missingness, stale_inputs=self._stale_inputs)
        return SelectionResult(packet=packet, context=context)

    def _manual_mutation(  # noqa: C901, PLR0912
        self,
        action_type: str,
        target_id: str,
        details: Mapping[str, Any],
        *,
        actor_kind: str = "human",
        actor_id: str | None = None,
        operation_id: str | None = None,
    ) -> ActionRecord:
        action_type = _text(action_type, name="queue action_type")
        details = _mapping(details, name=f"queue action {action_type}.details")
        if target_id and not isinstance(target_id, str):
            raise QueueInputError("queue action target_id must be a string")
        if actor_id is not None:
            _text(actor_id, name="queue action actor_id")
        current_packet_id = self.state.current_packet_id
        target_was_current = not target_id or target_id == current_packet_id
        target = target_id or current_packet_id
        if target.startswith("review-packet-"):
            packet = self.state.packet_payloads.get(target)
            target = str(packet.get("primary", {}).get("episode_id", target)) if packet else target
        if not target:
            raise AuditQueueError(f"{action_type} requires a target episode or current packet")
        candidate_ids = {item.episode_id for item in self.dataset.candidates}
        finding_ids = {item.finding_id for item in self.dataset.findings}
        if action_type == "more_evidence":
            if target not in candidate_ids and target not in finding_ids:
                raise QueueInputError(
                    f"more_evidence target is not a known episode or finding: {target}"
                )
        elif target not in candidate_ids:
            raise QueueInputError(f"{action_type} target is not a known episode: {target}")
        target_was_current = target_was_current or target == self._current_primary_id()
        operation_id = operation_id or f"queue-{action_type}-{self.state.state_revision}-{target}"
        _text(operation_id, name="queue action operation_id")
        if action_type == "more_evidence":
            _text(details.get("request", ""), name="more_evidence.request")
        digest = _sha256(
            {
                "action_type": action_type,
                "target_id": target,
                "details": dict(details),
                "actor_kind": actor_kind,
                "actor_id": actor_id or self.actor_id,
            }
        )
        existing_digest = self.state.operations.get(operation_id)
        if existing_digest is not None:
            if existing_digest != _sha256({"packet_id": target, "selection_id": digest}):
                raise QueueOperationConflictError(
                    f"operation ID {operation_id!r} was reused for a different queue action"
                )
            return self._record_action(
                action_type,
                target,
                details,
                actor_kind=actor_kind,
                actor_id=actor_id,
                operation_id=operation_id,
            )
        action = self._record_action(
            action_type,
            target,
            details,
            actor_kind=actor_kind,
            actor_id=actor_id,
            operation_id=operation_id,
        )
        state_kwargs: dict[str, Any] = {"state_revision": self.state.state_revision + 1}
        operations = dict(self.state.operations)
        operations[operation_id] = _sha256({"packet_id": target, "selection_id": digest})
        state_kwargs["operations"] = operations
        if action_type == "pin":
            state_kwargs["pinned_ids"] = tuple(dict.fromkeys((*self.state.pinned_ids, target)))
        elif action_type == "unpin":
            state_kwargs["pinned_ids"] = tuple(
                item for item in self.state.pinned_ids if item != target
            )
        elif action_type in {"skip", "defer"}:
            current = dict(
                self.state.skip_until if action_type == "skip" else self.state.defer_until
            )
            current[target] = self.state.selection_index + 1
            state_kwargs["skip_until" if action_type == "skip" else "defer_until"] = current
            if action_type == "defer":
                counts = dict(self.state.defer_counts)
                counts[target] = counts.get(target, 0) + 1
                state_kwargs["defer_counts"] = counts
            if target_was_current:
                state_kwargs["current_packet_id"] = ""
        elif action_type == "more_evidence":
            request = details["request"]
            requests = {
                key: tuple(value) for key, value in self.state.more_evidence_requests.items()
            }
            requests[target] = tuple(dict.fromkeys((*requests.get(target, ()), request)))
            state_kwargs["more_evidence_requests"] = requests
        self.state = replace(self.state, **state_kwargs)
        self._persist_state()
        return action

    def pin(
        self, episode_id: str = "", *, actor_id: str | None = None, operation_id: str | None = None
    ) -> ActionRecord:
        return self._manual_mutation(
            "pin",
            episode_id,
            {"reason": "manual pin"},
            actor_id=actor_id,
            operation_id=operation_id,
        )

    def unpin(
        self, episode_id: str = "", *, actor_id: str | None = None, operation_id: str | None = None
    ) -> ActionRecord:
        return self._manual_mutation(
            "unpin",
            episode_id,
            {"reason": "manual unpin"},
            actor_id=actor_id,
            operation_id=operation_id,
        )

    def skip(
        self, episode_id: str = "", *, actor_id: str | None = None, operation_id: str | None = None
    ) -> ActionRecord:
        return self._manual_mutation(
            "skip",
            episode_id,
            {"reason": "manual skip"},
            actor_id=actor_id,
            operation_id=operation_id,
        )

    def defer(
        self, episode_id: str = "", *, actor_id: str | None = None, operation_id: str | None = None
    ) -> ActionRecord:
        return self._manual_mutation(
            "defer",
            episode_id,
            {"reason": "manual defer"},
            actor_id=actor_id,
            operation_id=operation_id,
        )

    def more_evidence(
        self,
        request: str,
        episode_id: str = "",
        *,
        finding_id: str = "",
        actor_id: str | None = None,
        operation_id: str | None = None,
    ) -> ActionRecord:
        target = finding_id or episode_id
        return self._manual_mutation(
            "more_evidence",
            target,
            {"request": request},
            actor_id=actor_id,
            operation_id=operation_id,
        )

    request_more_evidence = more_evidence

    def previous_packet(  # noqa: C901
        self, *, actor_id: str | None = None, operation_id: str | None = None
    ) -> SelectionResult | None:
        if operation_id is not None:
            _text(operation_id, name="previous_packet.operation_id")
            existing_digest = self.state.operations.get(operation_id)
            if existing_digest is not None:
                for saved_context in reversed(self.state.selection_history):
                    expected_digest = _sha256(
                        {
                            "packet_id": saved_context.packet_id,
                            "selection_id": saved_context.selection_id,
                        }
                    )
                    if expected_digest != existing_digest:
                        continue
                    saved_payload = self.state.packet_payloads.get(saved_context.packet_id)
                    if saved_payload is None:
                        raise QueueStateError(
                            f"previous action {operation_id!r} references a missing packet snapshot"
                        )
                    try:
                        saved_packet = record_from_dict(saved_payload)
                    except (AuditContractError, KeyError, TypeError, ValueError) as exc:
                        raise QueueStateError(
                            f"previous action {operation_id!r} references a corrupt packet snapshot"
                        ) from exc
                    if not isinstance(saved_packet, ReviewPacket):
                        raise QueueStateError(
                            f"previous action {operation_id!r} references a non-packet snapshot"
                        )
                    return SelectionResult(packet=saved_packet, context=saved_context)
                raise QueueOperationConflictError(
                    f"operation ID {operation_id!r} was reused for a different previous action"
                )
        if not self.state.previous_packet_ids:
            return self.resume(reload=False)
        prior_id = self.state.previous_packet_ids[-1]
        packet_payload = self.state.packet_payloads.get(prior_id)
        if packet_payload is None:
            raise QueueStateError(f"previous packet snapshot is missing: {prior_id}")
        try:
            packet = record_from_dict(packet_payload)
        except (AuditContractError, KeyError, TypeError, ValueError) as exc:
            raise QueueStateError(f"corrupt previous packet snapshot {prior_id}: {exc}") from exc
        if not isinstance(packet, ReviewPacket):
            raise QueueStateError(f"previous snapshot is not a review_packet record: {prior_id}")
        history = list(self.state.previous_packet_ids)
        history.pop()
        context = next(
            (item for item in reversed(self.state.selection_history) if item.packet_id == prior_id),
            None,
        )
        if context is None:
            return None
        operation_id = operation_id or f"queue-previous-{self.state.state_revision}-{prior_id}"
        _text(operation_id, name="previous_packet.operation_id")
        operation_digest = _sha256({"packet_id": prior_id, "selection_id": context.selection_id})
        existing_digest = self.state.operations.get(operation_id)
        if existing_digest is not None:
            if existing_digest != operation_digest:
                raise QueueOperationConflictError(
                    f"operation ID {operation_id!r} was reused for a different previous action"
                )
            return SelectionResult(packet=packet, context=context)
        self._record_action(
            "previous_packet",
            context.primary_episode_id,
            {"packet_id": prior_id},
            actor_kind="human",
            actor_id=actor_id,
            operation_id=operation_id,
        )
        operations = dict(self.state.operations)
        operations[operation_id] = operation_digest
        self.state = replace(
            self.state,
            state_revision=self.state.state_revision + 1,
            current_packet_id=prior_id,
            previous_packet_ids=tuple(history),
            operations=operations,
        )
        self._persist_state()
        return SelectionResult(packet=packet, context=context)

    previous = previous_packet

    def record_review(  # noqa: C901, PLR0913
        self,
        episode_id: str = "",
        *,
        scope: str = "full_episode",
        outcome: str = "",
        author_kind: str = "human",
        author_id: str = "",
        notes: str = "",
        annotation_ids: Sequence[str] = (),
        review_id: str | None = None,
        operation_id: str | None = None,
    ) -> ReviewRecord:
        """Persist explicit review credit; selection itself never calls this."""

        target = episode_id
        if target and not isinstance(target, str):
            raise QueueInputError("review target episode_id must be a string")
        if not target and self.state.current_packet_id:
            packet = self.current_packet
            target = packet.primary.episode_id if packet else ""
        if not target:
            raise AuditQueueError("record_review requires an episode or current packet")
        if target not in {item.episode_id for item in self.dataset.candidates}:
            raise QueueInputError(f"review target is not a known episode: {target}")
        annotation_ids = _annotation_ids(annotation_ids)
        if operation_id is None:
            operation_id = "queue-review-" + _sha256(
                {
                    "episode_id": target,
                    "scope": scope,
                    "outcome": outcome,
                    "author_kind": author_kind,
                    "author_id": author_id,
                    "notes": notes,
                    "annotation_ids": tuple(annotation_ids),
                }
            )
        _text(operation_id, name="review.operation_id")
        review_id = review_id or "review-" + _sha256(
            {
                "episode_id": target,
                "scope": scope,
                "author_kind": author_kind,
                "author_id": author_id,
                "operation_id": operation_id,
            }
        )
        try:
            review = ReviewRecord(
                review_id=review_id,
                episode_id=target,
                scope=scope,
                outcome=outcome,
                author_kind=author_kind,
                author_id=author_id,
                source_revision=self.dataset.input_revision,
                annotation_ids=annotation_ids,
                notes=notes,
            )
        except (AuditContractError, TypeError, ValueError) as exc:
            raise QueueInputError(f"invalid review record: {exc}") from exc
        review_action_operation = f"{operation_id}:action"
        review_operation_digest = _sha256(
            {
                "review_id": review.review_id,
                "episode_id": review.episode_id,
                "scope": review.scope,
                "outcome": review.outcome,
                "author_kind": review.author_kind,
                "author_id": review.author_id,
                "source_revision": review.source_revision,
                "annotation_ids": review.annotation_ids,
                "notes": review.notes,
            }
        )
        existing_operation = self.state.operations.get(review_action_operation)
        if existing_operation is not None:
            if existing_operation != review_operation_digest:
                raise QueueOperationConflictError(
                    f"operation ID {review_action_operation!r} was reused for a different review"
                )
            existing_dataset_review = next(
                (
                    item
                    for item in self.dataset.review_records
                    if item.review_id == review.review_id
                ),
                None,
            )
            if existing_dataset_review is not None:
                return existing_dataset_review
        if self.store is not None:
            existing = self.store.get(review.review_id)
            if existing is not None and isinstance(existing.record, ReviewRecord):
                review = existing.record
            else:
                self.store.save(
                    review,
                    operation_id=operation_id,
                    expected_revision=0,
                    actor=author_kind,
                    actor_id=author_id,
                )
        existing_dataset_review = next(
            (item for item in self.dataset.review_records if item.review_id == review.review_id),
            None,
        )
        if existing_dataset_review is not None and existing_dataset_review != review:
            raise QueueConflictError(
                f"review ID collision with a different durable review: {review.review_id}"
            )
        if existing_dataset_review is None:
            self.dataset = replace(
                self.dataset, review_records=(*self.dataset.review_records, review)
            )
        self._record_action(
            "record_review",
            target,
            {
                "review_id": review.review_id,
                "scope": scope,
                "human_review_granted": author_kind == "human",
            },
            actor_kind=author_kind,
            actor_id=author_id or self.actor_id,
            operation_id=review_action_operation,
        )
        operations = dict(self.state.operations)
        operations[review_action_operation] = review_operation_digest
        self.state = replace(
            self.state,
            state_revision=self.state.state_revision + 1,
            input_revision=self.dataset.input_revision,
            input_identity=self.dataset.identity,
            operations=operations,
        )
        self._persist_state()
        return review

    review = record_review


ReviewQueue = AuditQueue
NextBestReview = AuditQueue


def rank_candidates(
    candidates: QueueDataset | Mapping[str, Any] | Iterable[Any],
    *,
    policy: QueuePolicy | None = None,
    seed: int = 0,
) -> list[RankedCandidate]:
    """Convenience function for bounded, non-persisting ranking."""

    return AuditQueue(candidates, policy=policy, seed=seed).rank_candidates()


def select_next(
    candidates: QueueDataset | Mapping[str, Any] | Iterable[Any],
    *,
    policy: QueuePolicy | None = None,
    seed: int = 0,
) -> SelectionResult | None:
    """Convenience function selecting one packet without a browser or live agent."""

    return AuditQueue(candidates, policy=policy, seed=seed).select_next()


def load_queue_input(path: str | Path) -> QueueDataset:
    """Load the documented offline fixture/input format."""

    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise QueueInputError("queue input root must be a mapping")
    return QueueDataset.from_mapping(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline explainable benchmark-audit queue")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("rank", "select", "resume"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--input", required=True, help="bounded queue input JSON")
        sub.add_argument(
            "--policy",
            choices=(QUEUE_POLICY_FIXED, QUEUE_POLICY_ACTIVE),
            default=QUEUE_POLICY_ACTIVE,
        )
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--state", help="optional queue state JSON for select/resume")
        sub.add_argument("--limit", type=int, default=0)
        sub.add_argument("--output", help="optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded offline queue CLI."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        dataset = load_queue_input(args.input)
        policy = QueuePolicy.fixed() if args.policy == QUEUE_POLICY_FIXED else QueuePolicy.active()
        queue = AuditQueue(dataset, policy=policy, seed=args.seed, state_path=args.state)
        if args.command == "rank":
            ranked = queue.rank_candidates()
            if args.limit:
                ranked = ranked[: max(0, args.limit)]
            payload: Any = {
                "schema_version": QUEUE_SELECTION_SCHEMA_VERSION,
                "policy": policy.to_dict(),
                "input_revision": dataset.input_revision,
                "ranked": [item.to_dict() for item in ranked],
                "missingness": list(dataset.missingness),
            }
        elif args.command == "resume":
            result = queue.resume()
            payload = (
                result.to_dict()
                if result
                else {"schema_version": QUEUE_SELECTION_SCHEMA_VERSION, "status": "empty"}
            )
        else:
            result = queue.select_next()
            payload = (
                result.to_dict()
                if result
                else {"schema_version": QUEUE_SELECTION_SCHEMA_VERSION, "status": "empty"}
            )
        text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    except (AuditQueueError, AuditContractError, OSError, TypeError, ValueError) as exc:
        sys.stderr.write(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True) + "\n")
        return 2


__all__ = [
    "ACTIVE_WEIGHTS",
    "BAND_BENCHMARK_CONFIG_DEFECT",
    "BAND_PLANNER_DISAGREEMENT",
    "BAND_RELEASE_MANUSCRIPT_IMPACT",
    "BAND_SAFETY_SEVERITY",
    "BAND_STATISTICAL_UNUSUALNESS",
    "BAND_UNEXPLAINED_BEHAVIOR",
    "CONTROL_STATUSES",
    "DEFAULT_POLICY_VERSION",
    "FIXED_POLICY_VERSION",
    "FIXED_WEIGHTS",
    "PRIORITY_BANDS",
    "PRIORITY_BAND_RANK",
    "PRIORITY_BENCHMARK_CONFIG",
    "PRIORITY_PLANNER_DISAGREEMENT",
    "PRIORITY_RELEASE_MANUSCRIPT",
    "PRIORITY_SAFETY",
    "PRIORITY_STATISTICAL",
    "PRIORITY_UNEXPLAINED",
    "QUEUE_INPUT_SCHEMA_VERSION",
    "ActivePolicy",
    "AuditQueue",
    "AuditQueueConflictError",
    "AuditQueueError",
    "AuditQueueOperationConflictError",
    "CoverageDeficit",
    "EpisodeCandidate",
    "FixedPolicy",
    "NextBestReview",
    "QueueCandidate",
    "QueueConflictError",
    "QueueDataset",
    "QueueEntry",
    "QueueInput",
    "QueueInputError",
    "QueueOperationConflictError",
    "QueuePolicy",
    "QueueSelectionContext",
    "QueueSelectionRecord",
    "QueueState",
    "QueueStateError",
    "RankedCandidate",
    "ReviewCase",
    "ReviewQueue",
    "ScanSummary",
    "SelectionContext",
    "SelectionExplanation",
    "SelectionResult",
    "active_policy",
    "fixed_policy",
    "load_queue_input",
    "main",
    "rank_candidates",
    "select_next",
]


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests.
    raise SystemExit(main())
