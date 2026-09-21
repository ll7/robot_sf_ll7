# ruff: noqa: DOC201

"""Typed, policy-bound domain service for the benchmark-audit workbench.

This module is intentionally the narrow authority shared by local clients.  It
does not own a second database, renderer, queue implementation, or provider
adapter.  Campaign reads are delegated to :mod:`audit_scan`; durable writes
and operation receipts are delegated to :class:`~audit_store.AuditStore`.

The service is useful without a browser or network connection.  A session
binds an immutable selection context, source digest, actor kind, and finite
policy.  Every attempted operation gets a durable ``ActionRecord`` before and
after execution, including conflicts, failures, denials, and cancellation.

Authority ``v1`` records created before durable full-``SourceRef`` persistence
are intentionally incompatible: shape validation rejects them instead of
recovering with digest-only provenance.  Operators must migrate or provision a
new authority root before reconnecting such a store.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from robot_sf.analysis_workbench.audit_authority import (
    AUTHORITY_SCHEMA_VERSION,
    CODEX_AUTHORITY_SCHEMA_VERSION,
    AuditAuthorityError,
    AuditAuthorityStore,
    AuthorityOperationConflict,
)
from robot_sf.analysis_workbench.audit_contracts import (
    AUTHOR_KINDS,
    SOURCE_PROVENANCE_MUTATED,
    SOURCE_PROVENANCE_STALE,
    SOURCE_PROVENANCE_UNAVAILABLE,
    ActionRecord,
    Annotation,
    AuditContractError,
    EpisodeRef,
    Finding,
    Reference,
    Signal,
    canonical_json,
    record_from_dict,
    record_to_dict,
    utc_now,
)
from robot_sf.analysis_workbench.audit_findings import FindingStore, create_from_annotation
from robot_sf.analysis_workbench.audit_scan import (
    AuditScanError,
    AuditScanReport,
    _row_config_digest,
    campaign_identity_from_payload,
    scan_campaign,
)
from robot_sf.analysis_workbench.audit_similarity import (
    SimilarityResult,
    find_similar_cases,
)
from robot_sf.analysis_workbench.audit_store import (
    AuditStore,
    AuditStoreError,
    CommitResult,
    OperationConflictError,
    StoredRecord,
)
from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    EXPERIMENT_RECIPE_SCHEMA_VERSION,
    ComponentRequest,
    ExperimentRecipe,
    ReviewContractsValidationError,
    SourceRef,
    component_request_from_dict,
    experiment_recipe_from_dict,
)

if TYPE_CHECKING:
    from robot_sf.analysis_workbench.audit_native_diagnostic import (
        NativeDiagnosticAdmission,
        NativeDiagnosticRequest,
        NativeDiagnosticResult,
    )

AUDIT_SERVICE_SCHEMA_VERSION = "audit-service.v1"
AUDIT_SELECTION_CONTEXT_SCHEMA_VERSION = "audit-selection-context.v1"
AUDIT_POLICY_SCHEMA_VERSION = "audit-session-policy.v1"
AUDIT_OPERATION_SCHEMA_VERSION = "audit-operation.v1"
CODEX_ACTIVITY_SCHEMA_VERSION = "audit-codex-activity.v1"
CODEX_OPERATION_STATUSES = (
    "complete",
    "unavailable",
    "failed",
    "cancelled",
    "conflict",
    "denied",
)
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
_CODEX_ROUTE_FIELDS = {
    "schema_version",
    "route_id",
    "provider",
    "model_id",
    "client_version",
    "protocol",
    "discovered",
    "capability_digest",
    "source",
}
_CODEX_APP_SERVER_SCHEMA_DIGEST = "7b9e7d385fffef8d428cc5490b56ce9c393bd3ed7bc7ccd730956387e723ec05"

SERVICE_STATUSES = (
    "complete",
    "committed",
    "partial",
    "unavailable",
    "failed",
    "cancelled",
    "conflict",
    "denied",
)
ACTOR_KINDS = AUTHOR_KINDS
MAX_TEXT_CHARS = 4_096
MAX_REASON_CHARS = 1_024
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_OPERATION_DETAIL_BYTES = 64 * 1024
MAX_RESULT_BYTES = 256 * 1024
MAX_CONTEXT_NODES = 128
MAX_PAYLOAD_DEPTH = 24
NATIVE_DIAGNOSTIC_OPERATION_TYPE = "diagnostic.native"
NATIVE_DIAGNOSTIC_DEFAULT_COMPUTE_COST = 2.0

_T = TypeVar("_T")


class AuditServiceError(RuntimeError):
    """Base class for service-bound failures."""


class AuditValidationError(AuditServiceError, ValueError):
    """Raised when a service envelope is malformed or exceeds its bounds."""


class AuditContextConflict(AuditServiceError):
    """Raised when the UI selection or source revision is stale."""

    def __init__(self, reason: str, *, expected: Any = None, actual: Any = None):
        """Build a conflict with both revisions when they are available."""

        self.reason = reason
        self.expected = expected
        self.actual = actual
        super().__init__(reason)


class AuditNextAmbiguous(AuditServiceError):
    """Raised when BA-02 may have committed but authority reconciliation is unknown."""


class AuditNextBlocked(AuditServiceError):
    """Raised when another operation owns a durable BA-02 queue lease."""


class AuditPolicyError(AuditServiceError):
    """Raised when the server-side session policy denies an operation."""


class AuditBudgetExceeded(AuditPolicyError):
    """Raised when an aggregate session budget is exhausted."""


class AuditCancelled(AuditServiceError):
    """Raised when a cancelled or killed session receives new work."""


class CapabilityUnavailable(AuditServiceError):
    """Raised only for an optional sibling capability that is not installed."""


def _bounded_text(value: Any, *, name: str, limit: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditValidationError(f"{name} must be a non-empty string")
    if len(value) > limit:
        raise AuditValidationError(f"{name} exceeds {limit} characters")
    return value


def _optional_text(value: Any, *, name: str, limit: int = MAX_TEXT_CHARS) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise AuditValidationError(f"{name} must be a string")
    if len(value) > limit:
        raise AuditValidationError(f"{name} exceeds {limit} characters")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AuditValidationError(f"{name} must be a finite number")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditValidationError(f"{name} must be a non-negative integer")
    return value


def _strict_walk(
    value: Any, *, path: str = "$", depth: int = 0, nodes: list[int] | None = None
) -> None:
    """Reject opaque/non-finite payloads before they cross the service boundary."""

    counters = nodes if nodes is not None else [0]
    counters[0] += 1
    if counters[0] > MAX_CONTEXT_NODES:
        raise AuditValidationError("service payload contains too many values")
    if depth > MAX_PAYLOAD_DEPTH:
        raise AuditValidationError(f"{path} exceeds maximum nesting depth")
    if isinstance(value, float) and not math.isfinite(value):
        raise AuditValidationError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AuditValidationError(f"{path} contains a non-string key")
            _strict_walk(item, path=f"{path}.{key}", depth=depth + 1, nodes=counters)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _strict_walk(item, path=f"{path}[{index}]", depth=depth + 1, nodes=counters)


def _bounded_mapping(
    value: Mapping[str, Any], *, name: str, limit: int = MAX_RESULT_BYTES
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditValidationError(f"{name} must be a mapping")
    copied = dict(value)
    _strict_walk(copied, path=name)
    try:
        encoded = canonical_json(copied).encode("utf-8")
    except (AuditContractError, TypeError, ValueError) as exc:
        raise AuditValidationError(f"{name} is not strict JSON: {exc}") from exc
    if len(encoded) > limit:
        raise AuditValidationError(f"{name} exceeds {limit} bytes")
    return copied


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _operation_request_digest(value: Any, **metadata: Any) -> str:
    """Hash the caller-owned operation input for idempotency/CAS replay."""

    return _digest({"value": value, **metadata})


def _source_ref_to_dict(source: SourceRef | None) -> dict[str, str] | None:
    """Serialize the complete source identity used by a durable session."""

    if source is None:
        return None
    return {name: str(getattr(source, name)) for name in SourceRef.__dataclass_fields__}


def _context_request_value(
    context: AuditSelectionContext | Mapping[str, Any] | None,
) -> Any:
    """Normalize a request context without dropping its revision fields."""

    return context.to_dict() if isinstance(context, AuditSelectionContext) else context


def _episode_ref_matches_row(ref: EpisodeRef, row: Mapping[str, Any]) -> bool:
    """Check whether a retained inventory row has the ref's full identity.

    Scenario and execution IDs are useful descriptive fields, but neither is
    sufficient to join a row to an :class:`EpisodeRef`: a source may reuse an
    execution ID across planners, seeds, or attempts.  Keep this comparison in
    lockstep with ``audit_scan._episode_ref`` and require every identity field
    that contributes to the generated ID.  The caller handles multiple
    matches as ambiguous instead of selecting an arbitrary ref.
    """

    execution_id = row.get("execution_id") or row.get("run_id") or row.get("episode_id")
    if not isinstance(execution_id, str) or not execution_id.strip():
        return False
    planner_id = row.get("planner_id") or row.get("planner") or row.get("algo") or ""
    scenario_id = row.get("scenario_id") or row.get("scenario") or ""
    checkpoint_digest = row.get("checkpoint_digest") or row.get("checkpoint_hash") or ""
    environment_digest = row.get("environment_digest") or row.get("environment_hash") or ""
    config_digest = _row_config_digest(row)
    if not all(
        isinstance(value, str)
        for value in (planner_id, scenario_id, checkpoint_digest, environment_digest, config_digest)
    ):
        return False
    return (
        ref.execution_id == execution_id.strip()
        and ref.planner_id == planner_id
        and ref.scenario_id == scenario_id
        and ref.seed == row.get("seed")
        and ref.attempt == row.get("attempt", 0)
        and ref.config_digest == config_digest.lower()
        and ref.checkpoint_digest == checkpoint_digest.lower()
        and ref.environment_digest == environment_digest.lower()
    )


@dataclass(frozen=True, slots=True)
class AuditCursor:
    """Versioned source-time cursor carried by a selection context."""

    time_s: float = 0.0
    context_revision: int = 0
    source: str = "initial"
    interval_id: str = ""

    def __post_init__(self) -> None:
        """Validate finite time and cursor revision binding."""

        object.__setattr__(self, "time_s", _finite(self.time_s, name="cursor.time_s"))
        object.__setattr__(
            self,
            "context_revision",
            _nonnegative_int(self.context_revision, name="cursor.context_revision"),
        )
        object.__setattr__(self, "source", _bounded_text(self.source, name="cursor.source"))
        object.__setattr__(
            self, "interval_id", _optional_text(self.interval_id, name="cursor.interval_id")
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AuditCursor:
        """Build a cursor while rejecting unknown fields."""

        if not isinstance(payload, Mapping):
            raise AuditValidationError("cursor must be a mapping")
        allowed = {"time_s", "context_revision", "source", "interval_id"}
        unknown = set(payload) - allowed
        if unknown:
            raise AuditValidationError(
                f"cursor contains unknown fields: {', '.join(sorted(unknown))}"
            )
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation."""

        return {
            "time_s": self.time_s,
            "context_revision": self.context_revision,
            "source": self.source,
            "interval_id": self.interval_id or None,
        }


@dataclass(frozen=True, slots=True)
class AuditSelectionContext:
    """Immutable browser/CLI selection snapshot used as a write capability."""

    campaign_id: str
    execution_id: str = ""
    episode_id: str = ""
    interval_id: str = ""
    cursor: AuditCursor = field(default_factory=AuditCursor)
    reference_id: str = ""
    actor_id: str = ""
    source_identity: str = ""
    source_revision: int | str = 0
    context_revision: int = 0
    schema_version: str = AUDIT_SELECTION_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate context version, identity, and cursor CAS fields."""

        if self.schema_version != AUDIT_SELECTION_CONTEXT_SCHEMA_VERSION:
            raise AuditValidationError(
                f"unsupported selection context schema: {self.schema_version}"
            )
        values = {
            "campaign_id": self.campaign_id,
            "execution_id": self.execution_id,
            "episode_id": self.episode_id,
            "interval_id": self.interval_id,
            "reference_id": self.reference_id,
            "actor_id": self.actor_id,
            "source_identity": self.source_identity,
        }
        for name, raw_value in values.items():
            value = (
                _bounded_text(raw_value, name=f"context.{name}")
                if name == "campaign_id"
                else _optional_text(raw_value, name=f"context.{name}")
            )
            object.__setattr__(self, name, value)
        revision = _nonnegative_int(self.context_revision, name="context.context_revision")
        object.__setattr__(self, "context_revision", revision)
        cursor = (
            self.cursor
            if isinstance(self.cursor, AuditCursor)
            else AuditCursor.from_mapping(self.cursor)
        )
        if cursor.context_revision != revision:
            raise AuditContextConflict(
                "cursor.context_revision does not match context_revision",
                expected=revision,
                actual=cursor.context_revision,
            )
        object.__setattr__(self, "cursor", cursor)
        source_revision = self.source_revision
        if isinstance(source_revision, bool) or not isinstance(source_revision, (int, str)):
            raise AuditValidationError("context.source_revision must be an integer or token")
        if isinstance(source_revision, int):
            if source_revision < 0:
                raise AuditValidationError("context.source_revision must be non-negative")
        elif len(source_revision) > MAX_TEXT_CHARS:
            raise AuditValidationError("context.source_revision exceeds the text limit")
        elif source_revision and not source_revision.strip():
            raise AuditValidationError("context.source_revision token must not be blank")
        object.__setattr__(self, "source_revision", source_revision)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AuditSelectionContext:
        """Parse a closed, versioned context envelope."""

        if not isinstance(payload, Mapping):
            raise AuditValidationError("selection context must be a mapping")
        _strict_walk(payload, path="context")
        allowed = {
            "schema_version",
            "campaign_id",
            "execution_id",
            "episode_id",
            "interval_id",
            "cursor",
            "reference_id",
            "actor_id",
            "source_identity",
            "source_revision",
            "context_revision",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise AuditValidationError(
                f"selection context contains unknown fields: {', '.join(sorted(unknown))}"
            )
        if payload.get("schema_version") != AUDIT_SELECTION_CONTEXT_SCHEMA_VERSION:
            raise AuditValidationError("selection context schema_version is required and versioned")
        if "campaign_id" not in payload:
            raise AuditValidationError("selection context campaign_id is required")
        values = dict(payload)
        cursor = values.get("cursor", {})
        if not isinstance(cursor, Mapping):
            raise AuditValidationError("selection context cursor must be a mapping")
        values["cursor"] = AuditCursor.from_mapping(cursor)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe context snapshot."""

        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "execution_id": self.execution_id or None,
            "episode_id": self.episode_id or None,
            "interval_id": self.interval_id or None,
            "cursor": self.cursor.to_dict(),
            "reference_id": self.reference_id or None,
            "actor_id": self.actor_id or None,
            "source_identity": self.source_identity or None,
            "source_revision": self.source_revision,
            "context_revision": self.context_revision,
        }

    def next_revision(self, **changes: Any) -> AuditSelectionContext:
        """Return the next context revision with a matching cursor revision."""

        allowed_changes = {
            "campaign_id",
            "execution_id",
            "episode_id",
            "interval_id",
            "reference_id",
            "actor_id",
            "source_identity",
            "source_revision",
            "time_s",
            "cursor_source",
        }
        unknown = set(changes) - allowed_changes
        if unknown:
            raise AuditValidationError(
                f"selection context update contains unknown fields: {', '.join(sorted(unknown))}"
            )
        values = {
            "campaign_id": self.campaign_id,
            "execution_id": self.execution_id,
            "episode_id": self.episode_id,
            "interval_id": self.interval_id,
            "reference_id": self.reference_id,
            "actor_id": self.actor_id,
            "source_identity": self.source_identity,
            "source_revision": self.source_revision,
            **changes,
        }
        revision = self.context_revision + 1
        values["context_revision"] = revision
        values["cursor"] = AuditCursor(
            time_s=changes.get("time_s", self.cursor.time_s),
            context_revision=revision,
            source=changes.get("cursor_source", "context"),
            interval_id=values.get("interval_id", self.interval_id),
        )
        values.pop("time_s", None)
        values.pop("cursor_source", None)
        return type(self)(**values)


@dataclass(frozen=True, slots=True)
class ActorRef:
    """Actor identity kept separate from human review identity."""

    kind: str
    actor_id: str = ""

    def __post_init__(self) -> None:
        """Validate the closed actor-kind vocabulary."""

        if self.kind not in ACTOR_KINDS:
            raise AuditValidationError(f"actor kind must be one of {ACTOR_KINDS}")
        object.__setattr__(self, "actor_id", _optional_text(self.actor_id, name="actor_id"))

    @classmethod
    def from_value(
        cls, value: ActorRef | Mapping[str, Any] | str, *, actor_id: str = ""
    ) -> ActorRef:
        """Parse one actor without permitting kind aliases or impersonation."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value, actor_id)
        if not isinstance(value, Mapping):
            raise AuditValidationError("actor must be a kind or mapping")
        unknown = set(value) - {"kind", "id", "actor_id"}
        if unknown:
            raise AuditValidationError(
                f"actor contains unknown fields: {', '.join(sorted(unknown))}"
            )
        return cls(value.get("kind", ""), value.get("id", value.get("actor_id", "")))

    def to_dict(self) -> dict[str, str]:
        """Return the store-compatible actor mapping."""

        return {"kind": self.kind, "id": self.actor_id}


@dataclass(frozen=True, slots=True)
class SessionPolicy:
    """Finite server-enforced authority for one audit session."""

    allowed_roots: tuple[str, ...] = ()
    allowed_repositories: tuple[str, ...] = ()
    allowed_recipes: tuple[str, ...] = ()
    token_budget: int = 0
    compute_budget: float = 0.0
    issue_write_budget: int = 0
    kill_switch: bool = False
    policy_id: str = "audit-policy-local"
    policy_revision: int = 0
    schema_version: str = AUDIT_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate roots, allowlists, and finite aggregate budgets."""

        if self.schema_version != AUDIT_POLICY_SCHEMA_VERSION:
            raise AuditValidationError(f"unsupported policy schema: {self.schema_version}")
        for name in ("allowed_roots", "allowed_repositories", "allowed_recipes"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise AuditValidationError(f"{name} must contain non-empty strings")
            if len(values) != len(set(values)):
                raise AuditValidationError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self, "token_budget", _nonnegative_int(self.token_budget, name="token_budget")
        )
        compute = _finite(self.compute_budget, name="compute_budget")
        if compute < 0:
            raise AuditValidationError("compute_budget must be non-negative")
        object.__setattr__(self, "compute_budget", compute)
        object.__setattr__(
            self,
            "issue_write_budget",
            _nonnegative_int(self.issue_write_budget, name="issue_write_budget"),
        )
        if not isinstance(self.kill_switch, bool):
            raise AuditValidationError("kill_switch must be boolean")
        object.__setattr__(self, "policy_id", _bounded_text(self.policy_id, name="policy_id"))
        object.__setattr__(
            self, "policy_revision", _nonnegative_int(self.policy_revision, name="policy_revision")
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SessionPolicy:
        """Parse a closed policy mapping with finite budgets."""

        if not isinstance(payload, Mapping):
            raise AuditValidationError("session policy must be a mapping")
        allowed = {
            "schema_version",
            "allowed_roots",
            "allowed_repositories",
            "allowed_recipes",
            "token_budget",
            "compute_budget",
            "issue_write_budget",
            "kill_switch",
            "policy_id",
            "policy_revision",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise AuditValidationError(
                f"session policy contains unknown fields: {', '.join(sorted(unknown))}"
            )
        if "schema_version" not in payload:
            raise AuditValidationError("session policy schema_version is required and versioned")
        values = dict(payload)
        for key in ("allowed_roots", "allowed_repositories", "allowed_recipes"):
            if key in values:
                if not isinstance(values[key], (tuple, list)):
                    raise AuditValidationError(f"{key} must be an array")
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a redaction-free policy document (tokens are not stored here)."""

        return {
            "schema_version": self.schema_version,
            "allowed_roots": list(self.allowed_roots),
            "allowed_repositories": list(self.allowed_repositories),
            "allowed_recipes": list(self.allowed_recipes),
            "token_budget": self.token_budget,
            "compute_budget": self.compute_budget,
            "issue_write_budget": self.issue_write_budget,
            "kill_switch": self.kill_switch,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
        }

    def allows_path(self, path: str | Path) -> bool:
        """Return whether a resolved path is beneath an explicitly allowed root."""

        if not self.allowed_roots:
            return False
        try:
            candidate = Path(path).resolve(strict=False)
            return any(
                candidate == root or root in candidate.parents
                for root in (Path(item).resolve(strict=False) for item in self.allowed_roots)
            )
        except (OSError, RuntimeError, ValueError):
            return False

    def allows_repository(self, repository: str) -> bool:
        """Return whether a repository is explicitly allowlisted."""

        return repository in self.allowed_repositories

    def allows_recipe(self, recipe_id: str) -> bool:
        """Return whether a diagnostic recipe is explicitly allowlisted."""

        return recipe_id in self.allowed_recipes


@dataclass(frozen=True, slots=True)
class SessionUsage:
    """Immutable aggregate usage snapshot reported by a service session."""

    tokens: int = 0
    compute: float = 0.0
    issue_writes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return current aggregate usage."""

        return {"tokens": self.tokens, "compute": self.compute, "issue_writes": self.issue_writes}


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    """Opaque, process-owned reservation capability for provider metering."""

    reservation_id: str
    session_id: str
    operation_id: str
    tokens: int
    compute: float
    issue_writes: int


@dataclass(frozen=True, slots=True)
class CodexSessionSnapshot:
    """Bounded, token-free authority snapshot for one Codex session."""

    codex_session_id: str
    audit_session_id: str
    actor: Mapping[str, Any]
    policy_id: str
    policy_revision: int
    policy_digest: str
    context: Mapping[str, Any]
    source_ref: Mapping[str, Any] | None
    source_digest: str
    source_revision: int | str
    route: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    provider_session_id: str
    status: str
    created_at: str
    updated_at: str
    last_operation_id: str
    schema_version: str = CODEX_AUTHORITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the closed authority representation."""

        return {
            "schema_version": self.schema_version,
            "codex_session_id": self.codex_session_id,
            "audit_session_id": self.audit_session_id,
            "actor": dict(self.actor),
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "policy_digest": self.policy_digest,
            "context": dict(self.context),
            "source_ref": dict(self.source_ref) if self.source_ref is not None else None,
            "source_digest": self.source_digest,
            "source_revision": self.source_revision,
            "route": dict(self.route),
            "evidence": [dict(item) for item in self.evidence],
            "provider_session_id": self.provider_session_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_operation_id": self.last_operation_id,
        }


@dataclass(frozen=True, slots=True)
class CodexOperationSnapshot:
    """Bounded durable operation state; an inflight row is never replayed."""

    operation_id: str
    action: str
    codex_session_id: str | None
    audit_session_id: str
    actor: Mapping[str, Any]
    request_digest: str
    policy_digest: str
    context_revision: int
    source_revision: int | str
    source_digest: str
    route_digest: str
    status: str
    result_status: str
    reason: str
    result_digest: str
    reservation_id: str
    reservation_operation_id: str
    provider_session_id: str
    result: Mapping[str, Any]
    created_at: str
    finished_at: str
    schema_version: str = CODEX_AUTHORITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the closed authority representation."""

        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "action": self.action,
            "codex_session_id": self.codex_session_id,
            "audit_session_id": self.audit_session_id,
            "actor": dict(self.actor),
            "request_digest": self.request_digest,
            "policy_digest": self.policy_digest,
            "context_revision": self.context_revision,
            "source_revision": self.source_revision,
            "source_digest": self.source_digest,
            "route_digest": self.route_digest,
            "status": self.status,
            "result_status": self.result_status,
            "reason": self.reason,
            "result_digest": self.result_digest,
            "reservation_id": self.reservation_id,
            "reservation_operation_id": self.reservation_operation_id,
            "provider_session_id": self.provider_session_id,
            "result": dict(self.result),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True, slots=True)
class CodexOperationActivitySnapshot:
    """Public Codex operation fields safe for an activity read.

    The durable authority operation also carries request, policy, result
    digests, reservation capabilities, and provider-session identity.  Those
    fields are intentionally not part of this projection: a panel needs the
    operation's lifecycle and metering outcome, not authority internals.
    """

    operation_id: str
    action: str
    codex_session_id: str | None
    context_revision: int
    source_revision: int | str
    source_digest: str
    status: str
    result_status: str
    reason: str
    usage: Mapping[str, Any]
    message: str
    created_at: str
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the closed operation projection without authority secrets."""

        return {
            "schema_version": CODEX_ACTIVITY_SCHEMA_VERSION,
            "operation_id": self.operation_id,
            "action": self.action,
            "codex_session_id": self.codex_session_id,
            "context_revision": self.context_revision,
            "source_revision": self.source_revision,
            "source_digest": self.source_digest,
            "status": self.status,
            "result_status": self.result_status,
            "reason": self.reason,
            "usage": dict(self.usage),
            "message": self.message,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True, slots=True)
class CodexActivitySnapshot:
    """Authenticated, source-bound activity view for one audit session.

    ``session`` is the validated durable :class:`CodexSessionSnapshot` when
    one exists.  The public ``to_dict`` projection deliberately omits its
    provider-session and source-reference internals.  Activity events are not
    persisted by the current authority schema, so a fresh process returns an
    explicit unavailable scope instead of fabricating history.
    """

    session: CodexSessionSnapshot | None
    operation: CodexOperationActivitySnapshot | None
    usage: SessionUsage
    status: str
    reason: str
    evidence: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...] = ()
    activity_scope: str = "unavailable"
    events_reason: str = "Codex activity events are unavailable because they are not persisted"
    schema_version: str = CODEX_ACTIVITY_SCHEMA_VERSION

    @staticmethod
    def _public_session(snapshot: CodexSessionSnapshot) -> dict[str, Any]:
        """Return panel fields without provider or filesystem metadata."""

        route = {
            key: snapshot.route[key]
            for key in (
                "schema_version",
                "route_id",
                "provider",
                "model_id",
                "client_version",
                "protocol",
                "discovered",
                "capability_digest",
            )
            if key in snapshot.route
        }
        return {
            "schema_version": snapshot.schema_version,
            "codex_session_id": snapshot.codex_session_id,
            "audit_session_id": snapshot.audit_session_id,
            "context": dict(snapshot.context),
            "source_digest": snapshot.source_digest,
            "source_revision": snapshot.source_revision,
            "route": route,
            "evidence": [dict(item) for item in snapshot.evidence],
            "status": snapshot.status,
            "created_at": snapshot.created_at,
            "updated_at": snapshot.updated_at,
            "last_operation_id": snapshot.last_operation_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the closed browser/panel activity representation."""

        return {
            "schema_version": self.schema_version,
            "session": self._public_session(self.session) if self.session is not None else None,
            "operation": self.operation.to_dict() if self.operation is not None else None,
            "usage": self.usage.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "evidence": [dict(item) for item in self.evidence],
            "events": [dict(item) for item in self.events],
            "activity_scope": self.activity_scope,
            "events_reason": self.events_reason,
        }


@dataclass(frozen=True, slots=True)
class CodexOperationLease:
    """Admission/replay result for one durable Codex operation."""

    status: str
    operation: CodexOperationSnapshot | None = None
    reservation_id: str = ""
    reservation_operation_id: str = ""
    reserved_tokens: int = 0
    reserved_compute: float = 0.0
    reserved_issue_writes: int = 0
    session: CodexSessionSnapshot | None = None
    result: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    replayed: bool = False

    @property
    def value(self) -> CodexOperationLease:
        """Expose the lease as a service-style value for thin adapters."""

        return self

    @property
    def admitted(self) -> bool:
        """Return whether the caller owns the provider admission."""

        return self.status == "admitted"


@dataclass(frozen=True, slots=True)
class AuditSession:
    """Opaque service-owned session handle.

    The handle is deliberately frozen.  Service internals use
    :func:`object.__setattr__` while callers can only submit this exact handle
    back to the owning service; a copied/replaced dataclass is rejected by the
    identity check in :meth:`AuditService._session`.
    """

    session_id: str
    actor: ActorRef
    policy: SessionPolicy
    context: AuditSelectionContext
    source_digest: str
    source_revision: int | str
    session_token: str = field(repr=False)
    source_ref: SourceRef | None = None
    usage: SessionUsage = field(default_factory=SessionUsage)
    cancelled: bool = False
    cancel_reason: str = ""
    created_at: str = field(default_factory=utc_now)

    @property
    def active(self) -> bool:
        """Return whether new service work may start."""

        return not self.cancelled and not self.policy.kill_switch

    def to_dict(self, *, include_token: bool = False) -> dict[str, Any]:
        """Return session metadata without leaking a token by default."""

        result = {
            "schema_version": AUDIT_SERVICE_SCHEMA_VERSION,
            "session_id": self.session_id,
            "actor": self.actor.to_dict(),
            "policy": self.policy.to_dict(),
            "context": self.context.to_dict(),
            "source_digest": self.source_digest,
            "source_revision": self.source_revision,
            "source_ref": _source_ref_to_dict(self.source_ref),
            "usage": self.usage.to_dict(),
            "cancelled": self.cancelled,
            "cancel_reason": self.cancel_reason,
            "created_at": self.created_at,
        }
        if include_token:
            result["session_token"] = self.session_token
        return result


@dataclass(frozen=True, slots=True)
class OperationReceipt:
    """Durable before/after operation receipt."""

    operation_id: str
    operation_type: str
    status: str
    actor: ActorRef
    session_id: str
    context_revision: int
    source_revision: int | str
    source_digest: str
    before_action_id: str
    after_action_id: str
    reason: str = ""
    replayed: bool = False
    result_digest: str = ""
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the public receipt envelope."""

        if self.status not in SERVICE_STATUSES and self.status != "started":
            raise AuditValidationError(f"unknown operation status: {self.status}")
        _bounded_text(self.operation_id, name="operation_id")
        _bounded_text(self.operation_type, name="operation_type")
        _bounded_text(self.session_id, name="session_id")
        _nonnegative_int(self.context_revision, name="context_revision")
        _optional_text(self.reason, name="operation.reason", limit=MAX_REASON_CHARS)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned receipt mapping."""

        return {
            "schema_version": AUDIT_OPERATION_SCHEMA_VERSION,
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "status": self.status,
            "actor": self.actor.to_dict(),
            "session_id": self.session_id,
            "context_revision": self.context_revision,
            "source_revision": self.source_revision,
            "source_digest": self.source_digest,
            "before_action_id": self.before_action_id,
            "after_action_id": self.after_action_id,
            "reason": self.reason,
            "replayed": self.replayed,
            "result_digest": self.result_digest,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ServiceResult(Generic[_T]):  # noqa: UP046 - repository supports Python 3.11 generics.
    """Typed result shared by browser, CLI, MCP, and Codex callers."""

    status: str
    value: _T | None = None
    reason: str = ""
    operation: OperationReceipt | None = None
    context: AuditSelectionContext | None = None

    @property
    def ok(self) -> bool:
        """Return whether the operation completed or committed."""

        return self.status in {"complete", "committed"}

    @property
    def receipt(self) -> OperationReceipt | None:
        """Alias for callers that call receipts directly."""

        return self.operation

    def to_dict(self) -> dict[str, Any]:
        """Return a bounded JSON-safe result envelope."""

        value: Any = self.value
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        elif isinstance(value, tuple):
            value = [item.to_dict() if hasattr(item, "to_dict") else item for item in value]
        elif isinstance(value, Mapping):
            value = dict(value)
        payload = {
            "schema_version": AUDIT_SERVICE_SCHEMA_VERSION,
            "status": self.status,
            "reason": self.reason,
            "value": value,
            "operation": self.operation.to_dict() if self.operation is not None else None,
            "context": self.context.to_dict() if self.context is not None else None,
        }
        try:
            _bounded_mapping(payload, name="service_result")
        except AuditValidationError:
            payload["value"] = None
            payload["reason"] = self.reason or "result_omitted_because_it_exceeded_bound"
        return payload


@dataclass(frozen=True, slots=True)
class EpisodeView:
    """One admitted episode row with explicit missingness."""

    episode_id: str
    status: str
    row: Mapping[str, Any] | None = None
    reason: str = ""
    source_artifact_id: str = ""
    episode_ref: Any | None = None

    @property
    def metrics(self) -> Mapping[str, Any] | None:
        """Return metrics without inventing values for missing rows."""

        return (
            self.row.get("metrics")
            if isinstance(self.row, Mapping) and isinstance(self.row.get("metrics"), Mapping)
            else None
        )

    @property
    def events(self) -> Any:
        """Return retained event data, if present."""

        return self.row.get("events") if isinstance(self.row, Mapping) else None

    @property
    def geometry(self) -> Any:
        """Return retained geometry data, if present."""

        return self.row.get("geometry") if isinstance(self.row, Mapping) else None

    def to_dict(self) -> dict[str, Any]:
        """Return an episode envelope."""

        return {
            "episode_id": self.episode_id,
            "status": self.status,
            "reason": self.reason,
            "source_artifact_id": self.source_artifact_id,
            "row": dict(self.row) if self.row is not None else None,
            "episode_ref": record_to_dict(self.episode_ref)
            if self.episode_ref is not None
            else None,
        }


@dataclass(frozen=True, slots=True)
class CampaignView:
    """Strict BA-01 report plus admitted episode rows."""

    report: AuditScanReport
    episodes: tuple[EpisodeView, ...]
    source_digest: str
    source_revision: int | str

    @property
    def status(self) -> str:
        """Return complete/partial based on BA-01 coverage and provenance."""

        coverage = self.report.counts.get("coverage", {})
        return (
            "complete"
            if not any(
                coverage.get(name, 0) for name in ("missing", "duplicate", "invalid", "unsupported")
            )
            else "partial"
        )

    def episode(self, episode_id: str) -> EpisodeView | None:
        """Find one unique literal row or digest-bound queue identity.

        ``ReviewPacket.primary.episode_id`` is generated from the admitted
        :class:`EpisodeRef`, while a campaign row may declare a different
        literal episode ID.  Both must address the same read surface without
        changing the queue/context ID used for writes.  Ambiguous aliases fail
        closed instead of selecting the first row.

        Returns:
            The unique matching row, or ``None`` for missing/ambiguous IDs.
        """

        matches = [
            item
            for item in self.episodes
            if item.episode_id == episode_id
            or (
                isinstance(item.episode_ref, EpisodeRef)
                and item.episode_ref.episode_id == episode_id
            )
        ]
        return matches[0] if len(matches) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        """Return a report/row mapping."""

        return {
            "report": self.report.to_dict(),
            "episodes": [item.to_dict() for item in self.episodes],
            "source_digest": self.source_digest,
            "source_revision": self.source_revision,
        }


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    """Optional BA-02/BA-04 adapter result."""

    capability: str
    status: str
    value: Any = None
    reason: str = ""
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return an unavailable-capability envelope."""

        return {
            "capability": self.capability,
            "status": self.status,
            "value": self.value,
            "reason": self.reason,
            "provider": self.provider,
        }


@dataclass(frozen=True, slots=True)
class NativeDiagnosticBinding:
    """Launcher-owned native source, request, recipe, and trust admission.

    The binding is deliberately not accepted by the diagnostic operation.  It
    is configured when the service is launched and selected only after the
    authenticated session source identity and policy have been checked.
    """

    admission: NativeDiagnosticAdmission
    request: ComponentRequest
    recipe: ExperimentRecipe

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> NativeDiagnosticBinding:
        """Parse one trusted binding while retaining the closed contracts."""

        from robot_sf.analysis_workbench import audit_native_diagnostic as native  # noqa: PLC0415

        if not isinstance(payload, Mapping):
            raise AuditValidationError("native diagnostic binding must be a mapping")
        allowed = {"admission", "request", "recipe"}
        unknown = set(payload) - allowed
        if unknown:
            raise AuditValidationError(
                "native diagnostic binding contains unknown fields: " + ", ".join(sorted(unknown))
            )
        try:
            admission = payload.get("admission")
            admission = (
                admission
                if isinstance(admission, native.NativeDiagnosticAdmission)
                else native.NativeDiagnosticAdmission.from_mapping(admission)
            )
            request = payload.get("request")
            request = (
                deepcopy(request)
                if isinstance(request, ComponentRequest)
                else component_request_from_dict(request, source="native diagnostic binding")
            )
            recipe = payload.get("recipe")
            recipe = (
                deepcopy(recipe)
                if isinstance(recipe, ExperimentRecipe)
                else experiment_recipe_from_dict(recipe, source="native diagnostic binding")
            )
        except (
            native.NativeDiagnosticError,
            ReviewContractsValidationError,
            TypeError,
            ValueError,
        ) as exc:
            raise AuditValidationError(f"invalid native diagnostic binding: {exc}") from exc
        if len(request.sources) != 1:
            raise AuditValidationError("native diagnostic binding requires exactly one source")
        return cls(admission=admission, request=request, recipe=recipe)

    def to_dict(self) -> dict[str, Any]:
        """Return a bounded launcher configuration representation."""

        request = {
            "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
            "request_id": self.request.request_id,
            "component_id": self.request.component_id,
            "sources": [asdict(source) for source in self.request.sources],
            "output_directory": self.request.output_directory,
            "config": dict(self.request.config),
            "required_capabilities": list(self.request.required_capabilities),
        }
        recipe = dict(self.recipe.document)
        recipe.setdefault("schema_version", EXPERIMENT_RECIPE_SCHEMA_VERSION)
        recipe.setdefault("recipe_id", self.recipe.recipe_id)
        return {
            "admission": self.admission.to_dict(),
            "request": request,
            "recipe": recipe,
        }


@dataclass(frozen=True, slots=True)
class NativeDiagnosticServiceConfig:
    """Trusted BA-05 adapter configuration owned by the service launcher."""

    bindings: tuple[NativeDiagnosticBinding, ...] = ()
    max_timeout_s: float = 60.0
    compute_cost: float = NATIVE_DIAGNOSTIC_DEFAULT_COMPUTE_COST

    def __post_init__(self) -> None:
        """Validate finite adapter ceilings before a service accepts work."""

        bindings = tuple(deepcopy(self.bindings))
        if any(not isinstance(binding, NativeDiagnosticBinding) for binding in bindings):
            raise AuditValidationError("native diagnostic bindings are malformed")
        if len({binding.recipe.recipe_id for binding in bindings}) != len(bindings):
            raise AuditValidationError("native diagnostic recipe IDs must be unique")
        object.__setattr__(self, "bindings", bindings)
        timeout = _finite(self.max_timeout_s, name="native_diagnostic.max_timeout_s")
        if timeout <= 0.0 or timeout > 60.0:
            raise AuditValidationError("native diagnostic max timeout must be within (0, 60]")
        object.__setattr__(self, "max_timeout_s", timeout)
        compute = _finite(self.compute_cost, name="native_diagnostic.compute_cost")
        if compute <= 0.0:
            raise AuditValidationError("native diagnostic compute cost must be positive")
        object.__setattr__(self, "compute_cost", compute)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> NativeDiagnosticServiceConfig:
        """Parse one or more launcher-owned source/recipe bindings."""

        if not isinstance(payload, Mapping):
            raise AuditValidationError("native diagnostic service config must be a mapping")
        allowed = {"bindings", "admission", "request", "recipe", "max_timeout_s", "compute_cost"}
        unknown = set(payload) - allowed
        if unknown:
            raise AuditValidationError(
                "native diagnostic service config contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        raw_bindings = payload.get("bindings")
        if raw_bindings is None:
            if not {"admission", "request", "recipe"}.issubset(payload):
                raw_bindings = ()
            else:
                raw_bindings = ({key: payload[key] for key in ("admission", "request", "recipe")},)
        if not isinstance(raw_bindings, (tuple, list)):
            raise AuditValidationError("native diagnostic config bindings must be an array")
        bindings = tuple(cls._binding(item) for item in raw_bindings)
        return cls(
            bindings=bindings,
            max_timeout_s=payload.get("max_timeout_s", 60.0),
            compute_cost=payload.get("compute_cost", NATIVE_DIAGNOSTIC_DEFAULT_COMPUTE_COST),
        )

    @staticmethod
    def _binding(value: Any) -> NativeDiagnosticBinding:
        if isinstance(value, NativeDiagnosticBinding):
            return value
        if not isinstance(value, Mapping):
            raise AuditValidationError("native diagnostic binding must be a mapping")
        return NativeDiagnosticBinding.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        """Return the trusted configuration without caller/session tokens."""

        return {
            "bindings": [binding.to_dict() for binding in self.bindings],
            "max_timeout_s": self.max_timeout_s,
            "compute_cost": self.compute_cost,
        }


class UnavailableQueueAdapter:
    """Typed BA-02 placeholder used until the sibling owner is merged."""

    def read(self, *, context: AuditSelectionContext, limit: int) -> CapabilityResult:
        """Return explicit BA-02 unavailability without queue semantics."""

        del context, limit
        return CapabilityResult("ba-02.queue", "unavailable", reason="ba-02 adapter is not merged")


class UnavailableCoverageAdapter:
    """Typed BA-04 placeholder used until the sibling owner is merged."""

    def read(self, *, context: AuditSelectionContext) -> CapabilityResult:
        """Return explicit BA-04 unavailability without coverage semantics."""

        del context
        return CapabilityResult(
            "ba-04.coverage", "unavailable", reason="ba-04 adapter is not merged"
        )


class AuditQueueAdapter(Protocol):
    """Typed seam for the future BA-02 queue owner."""

    def read(self, *, context: AuditSelectionContext, limit: int) -> Any:
        """Return a bounded queue response."""


class AuditQueueNextAdapter(Protocol):
    """Typed seam for the bounded BA-02 ``select_next`` owner."""

    def transact(  # noqa: PLR0913
        self,
        *,
        context: AuditSelectionContext,
        before_select: Callable[[Any], None] | None = None,
        after_select: Callable[[Any], object] | None = None,
        expected_state_revision: int | None = None,
        expected_input_revision: int | None = None,
        force_current: bool = False,
        lock_held: bool = False,
        operation_id: str | None = None,
        session_id: str | None = None,
    ) -> Any:
        """Select once and run the service-side authority callback."""

    def transaction_lock(self, *, context: AuditSelectionContext) -> Any:
        """Hold the adapter's durable lock across operation admission."""

    def current(self, *, context: AuditSelectionContext) -> Any:
        """Return the durable current selection for replay reconstruction."""

    def replay(
        self,
        *,
        operation_id: str,
        session_id: str,
        context: AuditSelectionContext,
        expected_result_digest: str = "",
    ) -> Any:
        """Return the exact durable result envelope for one queue operation."""


class AuditCoverageAdapter(Protocol):
    """Typed seam for the future BA-04 coverage owner."""

    def read(self, *, context: AuditSelectionContext) -> Any:
        """Return a bounded coverage response."""


class AuditService:
    """Single local authority for audit reads, CAS writes, and policy checks."""

    def __init__(  # noqa: PLR0913
        self,
        store: AuditStore | str | Path,
        *,
        campaign_source: Mapping[str, Any] | str | Path | None = None,
        source_ref: SourceRef | Mapping[str, Any] | None = None,
        source_root: str | Path | None = None,
        queue_adapter: AuditQueueAdapter | None = None,
        next_adapter: AuditQueueNextAdapter | None = None,
        queue_next_adapter: AuditQueueNextAdapter | None = None,
        coverage_adapter: AuditCoverageAdapter | None = None,
        authority_store: AuditAuthorityStore | str | Path | None = None,
        trusted_policy: SessionPolicy | Mapping[str, Any] | None = None,
        native_diagnostic_config: NativeDiagnosticServiceConfig
        | Mapping[str, Any]
        | Sequence[NativeDiagnosticBinding | Mapping[str, Any]]
        | None = None,
        native_diagnostic_admission: NativeDiagnosticAdmission | Mapping[str, Any] | None = None,
        native_diagnostic_request: ComponentRequest | Mapping[str, Any] | None = None,
        native_diagnostic_recipe: ExperimentRecipe | Mapping[str, Any] | None = None,
        native_diagnostic_max_timeout_s: float | None = None,
        native_diagnostic_compute_cost: float | None = None,
        github_provider: Any | None = None,
        github_private_roots: Sequence[str | Path] = (),
        github_allowed_media_hosts: Sequence[str] = (),
    ) -> None:
        """Bind the service to BA-03 and the durable session authority.

        ``trusted_policy`` is the launcher policy used when reconnecting a
        persisted session.  A stored session policy is descriptive state, not
        a source of new authority; callers recovering after a process restart
        must either provide this policy here or to :meth:`reconnect_session`.
        """

        self._owned_store = not isinstance(store, AuditStore)
        self.store = store if isinstance(store, AuditStore) else AuditStore(store)
        # BA-03 remains the sole canonical store.  These adapters only project
        # finding identity and GitHub outbox rows into that same store; they do
        # not introduce a second state machine or database.
        self.finding_store = FindingStore(self.store)
        self.github_provider = github_provider
        self.github_private_roots = tuple(github_private_roots)
        self.github_allowed_media_hosts = tuple(github_allowed_media_hosts)
        self._github_outbox: Any | None = None
        self.campaign_source = campaign_source
        self.source_ref = self._coerce_source_ref(source_ref)
        self.source_root = (
            Path(source_root).resolve(strict=False) if source_root is not None else None
        )
        self.queue_adapter = queue_adapter
        self.next_adapter = next_adapter or queue_next_adapter
        self.queue_next_adapter = self.next_adapter
        self.coverage_adapter = coverage_adapter
        authority_root = self.store.root if isinstance(store, AuditStore) else Path(store)
        self._owned_authority = authority_store is None or not isinstance(
            authority_store, AuditAuthorityStore
        )
        self.authority = (
            authority_store
            if isinstance(authority_store, AuditAuthorityStore)
            else AuditAuthorityStore(authority_store or authority_root)
        )
        self._trusted_policy = (
            trusted_policy
            if isinstance(trusted_policy, SessionPolicy)
            else SessionPolicy.from_mapping(trusted_policy)
            if trusted_policy is not None
            else None
        )
        self.native_diagnostic_config = self._coerce_native_diagnostic_config(
            native_diagnostic_config,
            admission=native_diagnostic_admission,
            request=native_diagnostic_request,
            recipe=native_diagnostic_recipe,
            max_timeout_s=native_diagnostic_max_timeout_s,
            compute_cost=native_diagnostic_compute_cost,
        )
        self._sessions: dict[str, AuditSession] = {}
        self._reservations: dict[str, BudgetReservation] = {}
        self._native_cancel_events: dict[str, set[threading.Event]] = {}
        self._lock = threading.RLock()
        self._load_authority_state()

    @staticmethod
    def _coerce_native_diagnostic_config(
        config: NativeDiagnosticServiceConfig
        | Mapping[str, Any]
        | Sequence[NativeDiagnosticBinding | Mapping[str, Any]]
        | None,
        *,
        admission: NativeDiagnosticAdmission | Mapping[str, Any] | None,
        request: ComponentRequest | Mapping[str, Any] | None,
        recipe: ExperimentRecipe | Mapping[str, Any] | None,
        max_timeout_s: float | None,
        compute_cost: float | None,
    ) -> NativeDiagnosticServiceConfig:
        """Normalize only launcher-supplied native configuration.

        The separate keyword form is intentionally convenient for local
        launchers while the MCP/browser operation has no path, receipt,
        request, or recipe arguments at all.
        """

        explicit = (admission, request, recipe)
        if any(item is not None for item in explicit):
            if config is not None or not all(item is not None for item in explicit):
                raise AuditValidationError(
                    "native diagnostic config and explicit binding fields cannot be mixed"
                )
            binding = NativeDiagnosticBinding.from_mapping(
                {"admission": admission, "request": request, "recipe": recipe}
            )
            return NativeDiagnosticServiceConfig(
                bindings=(binding,),
                max_timeout_s=60.0 if max_timeout_s is None else max_timeout_s,
                compute_cost=(
                    NATIVE_DIAGNOSTIC_DEFAULT_COMPUTE_COST if compute_cost is None else compute_cost
                ),
            )
        if config is None:
            return NativeDiagnosticServiceConfig(
                max_timeout_s=60.0 if max_timeout_s is None else max_timeout_s,
                compute_cost=(
                    NATIVE_DIAGNOSTIC_DEFAULT_COMPUTE_COST if compute_cost is None else compute_cost
                ),
            )
        if isinstance(config, NativeDiagnosticServiceConfig):
            base = config
        elif isinstance(config, Mapping):
            base = NativeDiagnosticServiceConfig.from_mapping(config)
        elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            base = NativeDiagnosticServiceConfig(
                bindings=tuple(NativeDiagnosticServiceConfig._binding(item) for item in config)
            )
        else:
            raise AuditValidationError("native diagnostic config is malformed")
        if max_timeout_s is None and compute_cost is None:
            return base
        return replace(
            base,
            max_timeout_s=base.max_timeout_s if max_timeout_s is None else max_timeout_s,
            compute_cost=base.compute_cost if compute_cost is None else compute_cost,
        )

    def __enter__(self) -> AuditService:
        """Enter a service context and return the same authority."""

        return self

    def __exit__(self, *_args: Any) -> None:
        """Close an owned store."""

        self.close()

    def close(self) -> None:
        """Close an internally-created canonical store."""

        if self._owned_store:
            self.store.close()
        if self._owned_authority:
            self.authority.close()

    @staticmethod
    def _authority_value(value: Any) -> Any:
        """Convert a compact transition result to strict JSON values."""

        if hasattr(value, "to_dict"):
            return AuditService._authority_value(value.to_dict())
        if isinstance(value, Mapping):
            return {str(key): AuditService._authority_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [AuditService._authority_value(item) for item in value]
        return value

    @staticmethod
    def _authority_session_record(
        session: AuditSession, token_verifier: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Serialize session authority without including the opaque token."""

        return {
            "session_id": session.session_id,
            "actor": session.actor.to_dict(),
            "policy": session.policy.to_dict(),
            "context": session.context.to_dict(),
            "source_digest": session.source_digest,
            "source_revision": session.source_revision,
            "source_ref": _source_ref_to_dict(session.source_ref),
            "token_verifier": dict(token_verifier),
            "usage": session.usage.to_dict(),
            "cancelled": session.cancelled,
            "cancel_reason": session.cancel_reason,
            "created_at": session.created_at,
        }

    @staticmethod
    def _authority_reservation(record: Mapping[str, Any]) -> BudgetReservation:
        """Parse one persisted reservation and fail closed on malformed state."""

        if not isinstance(record, Mapping):
            raise AuditValidationError("authority reservation is malformed")
        try:
            reservation = BudgetReservation(
                reservation_id=_bounded_text(record["reservation_id"], name="reservation_id"),
                session_id=_bounded_text(record["session_id"], name="reservation.session_id"),
                operation_id=_bounded_text(record["operation_id"], name="reservation.operation_id"),
                tokens=_nonnegative_int(record["tokens"], name="reservation.tokens"),
                compute=_finite(record["compute"], name="reservation.compute"),
                issue_writes=_nonnegative_int(
                    record["issue_writes"], name="reservation.issue_writes"
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditValidationError("authority reservation is malformed") from exc
        if reservation.compute < 0 or reservation.reservation_id != record.get("reservation_id"):
            raise AuditValidationError("authority reservation is malformed")
        AuditService._reservation_send_lease(record)
        return reservation

    @staticmethod
    def _reservation_send_lease(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Validate and return the optional provider-send lease on a reservation."""

        lease = record.get("send_lease")
        if lease is None:
            return None
        if not isinstance(lease, Mapping) or set(lease) != {
            "operation_id",
            "context_revision",
            "source_revision",
            "source_digest",
        }:
            raise AuditValidationError("authority reservation send lease is malformed")
        try:
            _bounded_text(lease["operation_id"], name="reservation.send_lease.operation_id")
            _nonnegative_int(
                lease["context_revision"], name="reservation.send_lease.context_revision"
            )
            source_revision = lease["source_revision"]
            if isinstance(source_revision, bool) or not isinstance(source_revision, (int, str)):
                raise AuditValidationError("reservation.send_lease.source_revision is malformed")
            _optional_text(lease["source_digest"], name="reservation.send_lease.source_digest")
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditValidationError("authority reservation send lease is malformed") from exc
        return dict(lease)

    def _load_reservations_state(self, state: Mapping[str, Any]) -> None:
        """Refresh in-process capability indexes from the latest authority state."""

        reservations = state.get("reservations", {})
        if not isinstance(reservations, Mapping):
            raise AuditValidationError("authority reservations are malformed")
        parsed: dict[str, BudgetReservation] = {}
        for reservation_id, record in reservations.items():
            reservation = self._authority_reservation(record)
            if reservation_id != reservation.reservation_id:
                raise AuditValidationError("authority reservation identity is inconsistent")
            parsed[reservation_id] = reservation
        self._reservations = parsed

    def _session_from_authority_record(
        self,
        record: Mapping[str, Any],
        *,
        token: str,
        existing: AuditSession | None = None,
    ) -> AuditSession:
        """Hydrate a service handle after authenticating its persisted verifier."""

        if not AuditAuthorityStore.verify_token(record, token):
            raise AuditPolicyError("session token is invalid")
        try:
            sid = _bounded_text(record["session_id"], name="session_id")
            actor = ActorRef.from_value(record["actor"])
            policy = SessionPolicy.from_mapping(record["policy"])
            context = AuditSelectionContext.from_mapping(record["context"])
            source_digest = _optional_text(record["source_digest"], name="source_digest")
            source_revision = record["source_revision"]
            source_ref = self._coerce_source_ref(record["source_ref"])
            usage_mapping = record["usage"]
            if not isinstance(usage_mapping, Mapping):
                raise AuditValidationError("session usage is malformed")
            usage = SessionUsage(
                tokens=_nonnegative_int(usage_mapping.get("tokens"), name="usage.tokens"),
                compute=_finite(usage_mapping.get("compute"), name="usage.compute"),
                issue_writes=_nonnegative_int(
                    usage_mapping.get("issue_writes"), name="usage.issue_writes"
                ),
            )
            if usage.compute < 0:
                raise AuditValidationError("usage.compute must be non-negative")
            cancelled = record["cancelled"]
            if not isinstance(cancelled, bool):
                raise AuditValidationError("session cancelled state is malformed")
            cancel_reason = _optional_text(
                record.get("cancel_reason", ""), name="cancel_reason", limit=MAX_REASON_CHARS
            )
            created_at = _bounded_text(record["created_at"], name="created_at")
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditValidationError("persisted audit session is malformed") from exc
        if not isinstance(source_revision, (int, str)) or isinstance(source_revision, bool):
            raise AuditValidationError("session source revision is malformed")
        if isinstance(source_revision, int) and source_revision < 0:
            raise AuditValidationError("session source revision is malformed")
        hydrated = existing or AuditSession(
            session_id=sid,
            actor=actor,
            policy=policy,
            context=context,
            source_digest=source_digest,
            source_revision=source_revision,
            session_token=token,
            source_ref=source_ref,
            usage=usage,
            cancelled=cancelled,
            cancel_reason=cancel_reason,
            created_at=created_at,
        )
        if hydrated.session_id != sid:
            raise AuditPolicyError("session handle identity is invalid")
        object.__setattr__(hydrated, "actor", actor)
        object.__setattr__(hydrated, "policy", policy)
        object.__setattr__(hydrated, "context", context)
        object.__setattr__(hydrated, "source_digest", source_digest)
        object.__setattr__(hydrated, "source_revision", source_revision)
        object.__setattr__(hydrated, "source_ref", source_ref)
        object.__setattr__(hydrated, "usage", usage)
        object.__setattr__(hydrated, "cancelled", cancelled)
        object.__setattr__(hydrated, "cancel_reason", cancel_reason)
        object.__setattr__(hydrated, "created_at", created_at)
        return hydrated

    def _load_authority_state(self) -> None:
        """Load durable indexes and reconcile only safe reserve crashes."""

        state = self.authority.snapshot()
        self._validate_authority_state_shape(state)
        self._reconcile_inflight_reservations()
        state = self.authority.snapshot()
        self._validate_authority_state_shape(state)
        self._load_reservations_state(state)

    @staticmethod
    def _codex_extension(state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the optional Codex extension, treating old state as empty."""

        extensions = state.get("extensions", {})
        if not isinstance(extensions, Mapping):
            raise AuditValidationError("authority extensions are malformed")
        if not extensions:
            return {
                "schema_version": CODEX_AUTHORITY_SCHEMA_VERSION,
                "sessions": {},
                "operations": {},
            }
        if set(extensions) != {"codex"} or not isinstance(extensions.get("codex"), Mapping):
            raise AuditValidationError("authority extensions are malformed")
        extension = extensions["codex"]
        if set(extension) != {"schema_version", "sessions", "operations"}:
            raise AuditValidationError("authority Codex extension is malformed")
        if extension.get("schema_version") != CODEX_AUTHORITY_SCHEMA_VERSION:
            raise AuditValidationError("authority Codex extension version is unsupported")
        if not isinstance(extension.get("sessions"), Mapping) or not isinstance(
            extension.get("operations"), Mapping
        ):
            raise AuditValidationError("authority Codex indexes are malformed")
        return extension

    @staticmethod
    def _codex_owns_reservation(
        state: Mapping[str, Any], operation_id: str, reservation_id: str
    ) -> bool:
        """Return whether only the durable Codex finish path may settle a hold."""

        reserve_operation = state["operations"].get(operation_id)
        if (
            isinstance(reserve_operation, Mapping)
            and reserve_operation.get("operation_type") == "budget.reserve.codex"
        ):
            return True
        codex_operations = AuditService._codex_extension(state)["operations"]
        return any(
            isinstance(item, Mapping)
            and item.get("status") == "inflight"
            and item.get("reservation_id") == reservation_id
            for item in codex_operations.values()
        )

    @staticmethod
    def _codex_historical_context_valid(
        historical: AuditSelectionContext,
        current: AuditSelectionContext,
        *,
        source_digest: str,
        source_ref: Mapping[str, Any] | None,
    ) -> bool:
        """Return whether a Codex context is an older binding of this source."""

        if historical == current:
            return True
        if historical.context_revision >= current.context_revision:
            return False
        if (
            historical.campaign_id != current.campaign_id
            or historical.source_revision != current.source_revision
        ):
            return False
        identities = {source_digest}
        if current.source_identity:
            identities.add(current.source_identity)
        if isinstance(source_ref, Mapping):
            for field_name in ("artifact_id", "sha256"):
                value = source_ref.get(field_name)
                if isinstance(value, str) and value:
                    identities.add(value)
        return not historical.source_identity or historical.source_identity in identities

    @staticmethod
    def _digest_value(value: Any, *, name: str, optional: bool = False) -> str:
        """Validate a SHA-256 identity used by the Codex authority."""

        if optional and value == "":
            return ""
        if not isinstance(value, str) or len(value) != 64:
            raise AuditValidationError(f"{name} must be a sha256 digest")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise AuditValidationError(f"{name} must be a sha256 digest") from exc
        return value

    def _codex_session_snapshot(self, value: Mapping[str, Any]) -> CodexSessionSnapshot:
        """Parse and validate one authority-owned Codex session snapshot."""

        if not isinstance(value, Mapping) or set(value) != _CODEX_SESSION_FIELDS:
            raise AuditValidationError("authority Codex session snapshot is malformed")
        try:
            codex_session_id = _bounded_text(value["codex_session_id"], name="codex_session_id")
            audit_session_id = _bounded_text(value["audit_session_id"], name="audit_session_id")
            actor = dict(value["actor"])
            if not isinstance(value["actor"], Mapping):
                raise AuditValidationError("Codex session actor is malformed")
            policy_id = _bounded_text(value["policy_id"], name="policy_id")
            policy_revision = _nonnegative_int(value["policy_revision"], name="policy_revision")
            policy_digest = self._digest_value(value["policy_digest"], name="policy_digest")
            context = AuditSelectionContext.from_mapping(value["context"])
            source_ref = self._coerce_source_ref(value["source_ref"])
            source_digest = _optional_text(value["source_digest"], name="source_digest")
            source_revision = value["source_revision"]
            if isinstance(source_revision, bool) or not isinstance(source_revision, (int, str)):
                raise AuditValidationError("Codex session source_revision is malformed")
            if isinstance(source_revision, int) and source_revision < 0:
                raise AuditValidationError("Codex session source_revision is malformed")
            route = _bounded_mapping(value["route"], name="Codex session route")
            evidence_raw = value["evidence"]
            if not isinstance(evidence_raw, (tuple, list)):
                raise AuditValidationError("Codex session evidence is malformed")
            evidence = tuple(
                _bounded_mapping(item, name="Codex session evidence item") for item in evidence_raw
            )
            provider_session_id = _bounded_text(
                value["provider_session_id"], name="provider_session_id"
            )
            status = _bounded_text(value["status"], name="Codex session status")
            if status not in {"active", "cancelled"}:
                raise AuditValidationError("Codex session status is invalid")
            created_at = _bounded_text(value["created_at"], name="created_at")
            updated_at = _bounded_text(value["updated_at"], name="updated_at")
            last_operation_id = _bounded_text(value["last_operation_id"], name="last_operation_id")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AuditValidationError):
                raise
            raise AuditValidationError("authority Codex session snapshot is malformed") from exc
        if not isinstance(actor, Mapping):
            raise AuditValidationError("Codex session actor is malformed")
        ActorRef.from_value(actor)
        return CodexSessionSnapshot(
            codex_session_id=codex_session_id,
            audit_session_id=audit_session_id,
            actor=actor,
            policy_id=policy_id,
            policy_revision=policy_revision,
            policy_digest=policy_digest,
            context=context.to_dict(),
            source_ref=_source_ref_to_dict(source_ref),
            source_digest=source_digest,
            source_revision=source_revision,
            route=route,
            evidence=evidence,
            provider_session_id=provider_session_id,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            last_operation_id=last_operation_id,
        )

    def _codex_operation_snapshot(  # noqa: C901
        self, value: Mapping[str, Any]
    ) -> CodexOperationSnapshot:
        """Parse and validate one authority-owned Codex operation snapshot."""

        if not isinstance(value, Mapping) or set(value) != _CODEX_OPERATION_FIELDS:
            raise AuditValidationError("authority Codex operation snapshot is malformed")
        try:
            operation_id = _bounded_text(value["operation_id"], name="operation_id")
            action = _bounded_text(value["action"], name="action")
            codex_session_id = value["codex_session_id"]
            if codex_session_id is not None:
                codex_session_id = _bounded_text(codex_session_id, name="codex_session_id")
            audit_session_id = _bounded_text(value["audit_session_id"], name="audit_session_id")
            actor = value["actor"]
            if not isinstance(actor, Mapping):
                raise AuditValidationError("Codex operation actor is malformed")
            actor = dict(actor)
            ActorRef.from_value(actor)
            request_digest = self._digest_value(value["request_digest"], name="request_digest")
            policy_digest = self._digest_value(value["policy_digest"], name="policy_digest")
            context_revision = _nonnegative_int(value["context_revision"], name="context_revision")
            source_revision = value["source_revision"]
            if isinstance(source_revision, bool) or not isinstance(source_revision, (int, str)):
                raise AuditValidationError("Codex operation source_revision is malformed")
            if isinstance(source_revision, int) and source_revision < 0:
                raise AuditValidationError("Codex operation source_revision is malformed")
            source_digest = _optional_text(value["source_digest"], name="source_digest")
            route_digest = self._digest_value(value["route_digest"], name="route_digest")
            status = value["status"]
            if status not in {"inflight", "finished"}:
                raise AuditValidationError("Codex operation status is invalid")
            result_status = value["result_status"]
            if not isinstance(result_status, str) or len(result_status) > MAX_TEXT_CHARS:
                raise AuditValidationError("Codex operation result_status is malformed")
            if status == "inflight" and result_status:
                raise AuditValidationError("inflight Codex operation has a result status")
            if status == "finished" and result_status not in CODEX_OPERATION_STATUSES:
                raise AuditValidationError("finished Codex operation result status is invalid")
            reason = _optional_text(value["reason"], name="reason", limit=MAX_REASON_CHARS)
            result_digest = self._digest_value(
                value["result_digest"], name="result_digest", optional=True
            )
            reservation_id = _optional_text(value["reservation_id"], name="reservation_id")
            reservation_operation_id = _optional_text(
                value["reservation_operation_id"], name="reservation_operation_id"
            )
            provider_session_id = _optional_text(
                value["provider_session_id"], name="provider_session_id"
            )
            # Authority shape and journal hashes protect ordinary recovery,
            # but a trusted extension writer must not make an invalid nested
            # provider result appear as a completed public activity read.
            result = self._codex_result_mapping(value["result"])
            if "usage" in result:
                self._codex_usage(result["usage"])
            created_at = _bounded_text(value["created_at"], name="created_at")
            finished_at = _optional_text(value["finished_at"], name="finished_at")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AuditValidationError):
                raise
            raise AuditValidationError("authority Codex operation snapshot is malformed") from exc
        return CodexOperationSnapshot(
            operation_id=operation_id,
            action=action,
            codex_session_id=codex_session_id,
            audit_session_id=audit_session_id,
            actor=actor,
            request_digest=request_digest,
            policy_digest=policy_digest,
            context_revision=context_revision,
            source_revision=source_revision,
            source_digest=source_digest,
            route_digest=route_digest,
            status=status,
            result_status=result_status,
            reason=reason,
            result_digest=result_digest,
            reservation_id=reservation_id,
            reservation_operation_id=reservation_operation_id,
            provider_session_id=provider_session_id,
            result=result,
            created_at=created_at,
            finished_at=finished_at,
        )

    def _validate_authority_state_shape(  # noqa: C901, PLR0912, PLR0915
        self, state: Mapping[str, Any]
    ) -> None:
        """Reject tampered nested authority records before serving requests."""

        sessions = state.get("sessions")
        reservations = state.get("reservations")
        operations = state.get("operations")
        if (
            not isinstance(sessions, Mapping)
            or not isinstance(reservations, Mapping)
            or not isinstance(operations, Mapping)
        ):
            raise AuditValidationError("authority state indexes are malformed")
        session_fields = {
            "session_id",
            "actor",
            "policy",
            "context",
            "source_digest",
            "source_revision",
            "source_ref",
            "token_verifier",
            "usage",
            "cancelled",
            "cancel_reason",
            "created_at",
        }
        for sid, raw in sessions.items():
            if (
                not isinstance(raw, Mapping)
                or set(raw) != session_fields
                or raw.get("session_id") != sid
            ):
                raise AuditValidationError("authority session record is malformed")
            verifier = raw.get("token_verifier")
            if (
                not isinstance(verifier, Mapping)
                or set(verifier) != {"algorithm", "salt", "digest"}
                or verifier.get("algorithm") != "sha256-salt-v1"
                or not isinstance(verifier.get("salt"), str)
                or not isinstance(verifier.get("digest"), str)
                or len(verifier["digest"]) != 64
            ):
                raise AuditValidationError("authority session token verifier is malformed")
            try:
                bytes.fromhex(verifier["salt"])
                ActorRef.from_value(raw["actor"])
                SessionPolicy.from_mapping(raw["policy"])
                AuditSelectionContext.from_mapping(raw["context"])
                usage = raw["usage"]
                if not isinstance(usage, Mapping):
                    raise AuditValidationError("authority usage is malformed")
                _nonnegative_int(usage["tokens"], name="usage.tokens")
                compute = _finite(usage["compute"], name="usage.compute")
                _nonnegative_int(usage["issue_writes"], name="usage.issue_writes")
                if compute < 0 or not isinstance(raw["cancelled"], bool):
                    raise AuditValidationError("authority session state is malformed")
                _optional_text(raw["source_digest"], name="source_digest")
                if raw["source_ref"] is not None:
                    self._coerce_source_ref(raw["source_ref"])
                _optional_text(raw["cancel_reason"], name="cancel_reason", limit=MAX_REASON_CHARS)
                _bounded_text(raw["created_at"], name="created_at")
            except (KeyError, TypeError, ValueError) as exc:
                raise AuditValidationError("authority session record is malformed") from exc
        reservation_totals: dict[str, SessionUsage] = {}
        reservation_operations: dict[str, str] = {}
        unleased_reservations: set[str] = set()
        for reservation_id, raw in reservations.items():
            try:
                reservation = self._authority_reservation(raw)
            except AuditValidationError as exc:
                raise AuditValidationError("authority reservation is malformed") from exc
            if reservation_id != reservation.reservation_id:
                raise AuditValidationError("authority reservation identity is inconsistent")
            if reservation.session_id not in sessions:
                raise AuditValidationError("authority reservation references an unknown session")
            if reservation.operation_id in reservation_operations:
                raise AuditValidationError("authority operation has duplicate reservations")
            reservation_operations[reservation.operation_id] = reservation_id
            if self._reservation_send_lease(raw) is None:
                unleased_reservations.add(reservation_id)
            previous = reservation_totals.get(reservation.session_id, SessionUsage())
            reservation_totals[reservation.session_id] = SessionUsage(
                tokens=previous.tokens + reservation.tokens,
                compute=previous.compute + reservation.compute,
                issue_writes=previous.issue_writes + reservation.issue_writes,
            )

        operation_fields = {
            "operation_id",
            "operation_type",
            "session_id",
            "actor",
            "request_digest",
            "context_revision",
            "source_revision",
            "source_digest",
            "status",
            "result_status",
            "reason",
            "result_digest",
            "created_at",
        }
        for operation_id, raw in operations.items():
            if (
                not isinstance(raw, Mapping)
                or set(raw) != operation_fields
                or raw.get("operation_id") != operation_id
                or raw.get("status") not in {"inflight", "finished"}
                or raw.get("result_status", "") not in {"", *SERVICE_STATUSES}
                or (raw.get("status") == "inflight" and raw.get("result_status", "") != "")
                or (raw.get("status") == "finished" and raw.get("result_status", "") == "")
            ):
                raise AuditValidationError("authority operation record is malformed")
            try:
                _bounded_text(raw["operation_type"], name="operation_type")
                _bounded_text(raw["session_id"], name="session_id")
                ActorRef.from_value(raw["actor"])
                _bounded_text(raw["request_digest"], name="request_digest")
                _nonnegative_int(raw["context_revision"], name="context_revision")
                _optional_text(raw["source_digest"], name="source_digest")
                _optional_text(raw["reason"], name="operation.reason", limit=MAX_REASON_CHARS)
                _optional_text(raw["result_digest"], name="result_digest")
                _bounded_text(raw["created_at"], name="created_at")
            except (KeyError, TypeError, ValueError) as exc:
                raise AuditValidationError("authority operation record is malformed") from exc
            session = sessions.get(raw["session_id"])
            if not isinstance(session, Mapping) or raw["actor"] != session.get("actor"):
                raise AuditValidationError("authority operation actor is inconsistent")
            if operation_id in reservation_operations:
                if raw["operation_type"] not in {"budget.reserve", "budget.reserve.codex"}:
                    raise AuditValidationError("authority reservation operation type is invalid")
                if raw["status"] == "finished" and raw["result_status"] != "committed":
                    raise AuditValidationError("authority reservation result is inconsistent")

        for operation_id in reservation_operations:
            if operation_id not in operations:
                raise AuditValidationError("authority reservation operation is missing")

        extension = self._codex_extension(state)
        codex_reservation_ids: set[str] = set()
        for codex_session_id, raw in extension["sessions"].items():
            if codex_session_id != raw.get("codex_session_id"):
                raise AuditValidationError("authority Codex session identity is inconsistent")
            snapshot = self._codex_session_snapshot(raw)
            audit_session = sessions.get(snapshot.audit_session_id)
            if not isinstance(audit_session, Mapping):
                raise AuditValidationError(
                    "authority Codex session references an unknown audit session"
                )
            if snapshot.actor != audit_session.get("actor"):
                raise AuditValidationError("authority Codex session actor is inconsistent")
            stored_policy = SessionPolicy.from_mapping(audit_session["policy"])
            if (
                snapshot.policy_id != stored_policy.policy_id
                or snapshot.policy_revision != stored_policy.policy_revision
                or snapshot.policy_digest != _digest(stored_policy.to_dict())
            ):
                raise AuditValidationError("authority Codex session policy binding is inconsistent")
            snapshot_context = AuditSelectionContext.from_mapping(snapshot.context)
            current_context = AuditSelectionContext.from_mapping(audit_session["context"])
            if not self._codex_historical_context_valid(
                snapshot_context,
                current_context,
                source_digest=str(audit_session.get("source_digest", "")),
                source_ref=audit_session.get("source_ref"),
            ):
                raise AuditValidationError("authority Codex session context is inconsistent")
            if snapshot.source_digest != audit_session.get("source_digest"):
                raise AuditValidationError("authority Codex session source is inconsistent")
            if snapshot.source_revision != audit_session.get("source_revision"):
                raise AuditValidationError(
                    "authority Codex session source revision is inconsistent"
                )
            if snapshot.source_ref != audit_session.get("source_ref"):
                raise AuditValidationError(
                    "authority Codex session source reference is inconsistent"
                )
        for operation_id, raw in extension["operations"].items():
            if operation_id != raw.get("operation_id"):
                raise AuditValidationError("authority Codex operation identity is inconsistent")
            operation = self._codex_operation_snapshot(raw)
            audit_session = sessions.get(operation.audit_session_id)
            if not isinstance(audit_session, Mapping):
                raise AuditValidationError(
                    "authority Codex operation references an unknown audit session"
                )
            if operation.actor != audit_session.get("actor"):
                raise AuditValidationError("authority Codex operation actor is inconsistent")
            stored_policy = SessionPolicy.from_mapping(audit_session["policy"])
            if operation.policy_digest != _digest(stored_policy.to_dict()):
                raise AuditValidationError(
                    "authority Codex operation policy binding is inconsistent"
                )
            current_context = AuditSelectionContext.from_mapping(audit_session["context"])
            if operation.context_revision > current_context.context_revision:
                raise AuditValidationError("authority Codex operation context is inconsistent")
            if operation.source_digest != audit_session.get(
                "source_digest"
            ) or operation.source_revision != audit_session.get("source_revision"):
                raise AuditValidationError("authority Codex operation source is inconsistent")
            if (
                operation.codex_session_id is not None
                and operation.codex_session_id not in extension["sessions"]
            ):
                raise AuditValidationError(
                    "authority Codex operation references an unknown Codex session"
                )
            if operation.codex_session_id is not None:
                operation_session = self._codex_session_snapshot(
                    extension["sessions"][operation.codex_session_id]
                )
                if operation.context_revision != operation_session.context["context_revision"]:
                    raise AuditValidationError(
                        "authority Codex operation context binding is inconsistent"
                    )
            if operation.reservation_id:
                reservation = reservations.get(operation.reservation_id)
                if operation.status == "inflight" and not isinstance(reservation, Mapping):
                    raise AuditValidationError("inflight Codex operation reservation is missing")
                if operation.status == "finished" and isinstance(reservation, Mapping):
                    raise AuditValidationError("finished Codex operation retains a reservation")
                if operation.status == "inflight":
                    codex_reservation_ids.add(operation.reservation_id)

        for reservation_id, raw in reservations.items():
            reservation_operation = operations.get(raw["operation_id"])
            if not isinstance(reservation_operation, Mapping):
                raise AuditValidationError("authority reservation operation is missing")
            codex_typed = reservation_operation.get("operation_type") == "budget.reserve.codex"
            codex_linked = reservation_id in codex_reservation_ids
            if codex_typed != codex_linked:
                raise AuditValidationError("authority Codex reservation relation is inconsistent")

        for sid, raw in sessions.items():
            usage_mapping = raw["usage"]
            usage = SessionUsage(
                tokens=usage_mapping["tokens"],
                compute=usage_mapping["compute"],
                issue_writes=usage_mapping["issue_writes"],
            )
            reserved = reservation_totals.get(sid, SessionUsage())
            cancelled_unleased = any(
                reservation_id in unleased_reservations
                and isinstance(reservations.get(reservation_id), Mapping)
                and reservations[reservation_id].get("session_id") == sid
                and reservation_id not in codex_reservation_ids
                for reservation_id in reservations
            )
            if (
                (raw["cancelled"] and cancelled_unleased)
                or usage.tokens < reserved.tokens
                or usage.compute < reserved.compute
                or usage.issue_writes < reserved.issue_writes
                or usage.tokens > SessionPolicy.from_mapping(raw["policy"]).token_budget
                or usage.compute > SessionPolicy.from_mapping(raw["policy"]).compute_budget
                or usage.issue_writes > SessionPolicy.from_mapping(raw["policy"]).issue_write_budget
            ):
                raise AuditValidationError(
                    "authority session usage is inconsistent with reservations"
                )

    def _reconcile_inflight_reservations(self) -> None:
        """Release reservations whose internal reserve operation crashed.

        A reserve transition has no external side effect.  If its operation
        remains inflight after a process crash, releasing that hold is safe and
        prevents a permanent budget leak.  Settlement/consume/write operations
        remain inflight and are deliberately not guessed at: retries fail
        closed until an operator can reconcile the ambiguous external action.
        """

        state = self.authority.snapshot()
        operations = state.get("operations", {})
        reservations = state.get("reservations", {})
        if not isinstance(operations, Mapping) or not isinstance(reservations, Mapping):
            raise AuditValidationError("authority operation state is malformed")
        candidates = tuple(
            (operation_id, operation)
            for operation_id, operation in operations.items()
            if isinstance(operation, Mapping)
            and operation.get("status") == "inflight"
            and operation.get("operation_type") == "budget.reserve"
        )
        for operation_id, _operation in candidates:
            transaction_id = f"authority.reconcile.reserve:{operation_id}"
            request_digest = _digest({"kind": "reconcile.reserve", "operation_id": operation_id})

            def reconcile(
                latest: dict[str, Any],
                *,
                _operation_id: str = operation_id,
            ) -> dict[str, Any]:
                latest_operations = latest["operations"]
                current = latest_operations.get(_operation_id)
                if not isinstance(current, Mapping) or current.get("status") != "inflight":
                    return {"reconciled": False}
                released = 0
                for reservation_id, raw in tuple(latest["reservations"].items()):
                    if not isinstance(raw, Mapping) or raw.get("operation_id") != _operation_id:
                        continue
                    session_id = raw.get("session_id")
                    session = latest["sessions"].get(session_id)
                    if isinstance(session, Mapping) and isinstance(session.get("usage"), Mapping):
                        usage = session["usage"]
                        usage["tokens"] = max(0, int(usage.get("tokens", 0)) - int(raw["tokens"]))
                        usage["compute"] = max(
                            0.0, float(usage.get("compute", 0.0)) - float(raw["compute"])
                        )
                        usage["issue_writes"] = max(
                            0, int(usage.get("issue_writes", 0)) - int(raw["issue_writes"])
                        )
                    latest["reservations"].pop(reservation_id, None)
                    released += 1
                current = dict(current)
                current.update(
                    {
                        "status": "finished",
                        "result_status": "unavailable",
                        "reason": "reserve operation reconciled after crash",
                    }
                )
                latest_operations[_operation_id] = current
                return {"reconciled": True, "released": released}

            self.authority.mutate(transaction_id, request_digest, reconcile)

    def _refresh_session(
        self,
        session_id: str,
        *,
        token: str,
        existing: AuditSession | None = None,
    ) -> AuditSession:
        """Read the latest session state and authenticate its opaque token."""

        state = self.authority.snapshot()
        sessions = state.get("sessions", {})
        if not isinstance(sessions, Mapping) or session_id not in sessions:
            raise AuditPolicyError("unknown audit session")
        target = self._session_from_authority_record(
            sessions[session_id], token=token, existing=existing
        )
        self._sessions[session_id] = target
        self._load_reservations_state(state)
        return target

    def _mutate_session_authority(
        self,
        session: AuditSession,
        *,
        transaction_id: str,
        request_digest: str,
        callback: Callable[[AuditSession, dict[str, Any]], _T],
    ) -> tuple[_T, bool]:
        """Run one session/reservation transition under the durable lock."""

        token = session.session_token

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            raw = state["sessions"].get(session.session_id)
            if not isinstance(raw, Mapping):
                raise AuditPolicyError("unknown audit session")
            working = self._session_from_authority_record(raw, token=token)
            value = callback(working, state)
            state["sessions"][session.session_id] = self._authority_session_record(
                working, raw["token_verifier"]
            )
            return {
                "session": state["sessions"][session.session_id],
                "value": self._authority_value(value),
            }

        mutation = self.authority.mutate(transaction_id, request_digest, mutate)
        payload = mutation.value
        if not isinstance(payload, Mapping) or not isinstance(payload.get("session"), Mapping):
            raise AuditAuthorityError("authority mutation returned malformed session state")
        self._session_from_authority_record(payload["session"], token=token, existing=session)
        self._sessions[session.session_id] = session
        self._load_reservations_state(mutation.state)
        value = payload.get("value")
        return value, mutation.replayed

    @staticmethod
    def _policy_within_trusted(stored: SessionPolicy, trusted: SessionPolicy) -> bool:
        """Return whether stored authority stays within launcher ceilings."""

        return (
            stored.policy_id == trusted.policy_id
            and stored.policy_revision == trusted.policy_revision
            and stored.token_budget <= trusted.token_budget
            and stored.compute_budget <= trusted.compute_budget
            and stored.issue_write_budget <= trusted.issue_write_budget
            and set(stored.allowed_roots).issubset(trusted.allowed_roots)
            and set(stored.allowed_repositories).issubset(trusted.allowed_repositories)
            and set(stored.allowed_recipes).issubset(trusted.allowed_recipes)
            and (not stored.kill_switch or trusted.kill_switch)
        )

    def _authority_operation(self, operation_id: str) -> Mapping[str, Any] | None:
        state = self.authority.snapshot()
        operations = state.get("operations", {})
        if not isinstance(operations, Mapping):
            raise AuditAuthorityError("authority operation index is malformed")
        operation = operations.get(operation_id)
        return operation if isinstance(operation, Mapping) else None

    def _finish_authority_operation(
        self,
        session: AuditSession,
        *,
        operation_id: str,
        status: str,
        reason: str,
        result_digest: str,
    ) -> None:
        """Persist the terminal operation state after its side effect."""

        request = {
            "operation_id": operation_id,
            "status": status,
            "reason": reason,
            "result_digest": result_digest,
        }
        transaction_id = f"authority.operation.finish:{session.session_id}:{operation_id}"
        request_digest = _digest(request)

        def finish(state: dict[str, Any]) -> None:
            operation = state["operations"].get(operation_id)
            if not isinstance(operation, Mapping):
                raise AuthorityOperationConflict("operation authority is missing")
            if operation.get("session_id") != session.session_id:
                raise AuthorityOperationConflict("operation belongs to another session")
            if operation.get("status") == "finished":
                if (
                    operation.get("result_status") != status
                    or operation.get("result_digest", "") != result_digest
                ):
                    raise AuthorityOperationConflict("operation terminal state conflicts")
                return
            if operation.get("status") != "inflight":
                raise AuthorityOperationConflict("operation authority is not inflight")
            updated = dict(operation)
            updated.update(
                {
                    "status": "finished",
                    "result_status": status,
                    "reason": reason,
                    "result_digest": result_digest,
                }
            )
            state["operations"][operation_id] = updated

        self.authority.mutate(transaction_id, request_digest, finish)

    @staticmethod
    def _coerce_source_ref(value: SourceRef | Mapping[str, Any] | None) -> SourceRef | None:
        if value is None:
            return None
        if isinstance(value, SourceRef):
            if any(
                not isinstance(getattr(value, name), str) for name in SourceRef.__dataclass_fields__
            ):
                raise AuditValidationError("source_ref fields must be strings")
            return value
        if not isinstance(value, Mapping):
            raise AuditValidationError("source_ref must be a SourceRef or mapping")
        allowed = {item.name for item in SourceRef.__dataclass_fields__.values()}
        unknown = set(value) - allowed
        if unknown:
            raise AuditValidationError(
                f"source_ref contains unknown fields: {', '.join(sorted(unknown))}"
            )
        required = {"artifact_id", "uri", "format"}
        if not required.issubset(value):
            raise AuditValidationError("source_ref requires artifact_id, uri, and format")
        values = {name: value.get(name, "") for name in allowed}
        if any(not isinstance(item, str) for item in values.values()):
            raise AuditValidationError("source_ref fields must be strings")
        return SourceRef(**values)

    @staticmethod
    def _operation_id(value: str | None) -> str:
        return _bounded_text(value or f"audit-op-{uuid.uuid4().hex}", name="operation_id")

    @staticmethod
    def _contains_sensitive_text(value: Any, secret: str) -> bool:
        """Return whether a bounded JSON value contains an authority secret."""

        if not secret:
            return False
        stack: list[Any] = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, str):
                if secret in current:
                    return True
            elif isinstance(current, Mapping):
                stack.extend(current.keys())
                stack.extend(current.values())
            elif isinstance(current, (tuple, list)):
                stack.extend(current)
        return False

    @staticmethod
    def _bound_request_digest(session: AuditSession, request_digest: str) -> str:
        """Bind an operation request to the server's current source/context."""

        return _operation_request_digest(
            request_digest,
            bound_context=session.context.to_dict(),
            bound_source_digest=session.source_digest,
            bound_source_revision=session.source_revision,
        )

    def _lookup_session(self, session_id: str) -> AuditSession:
        """Resolve one server-owned handle without authenticating a caller."""

        if not isinstance(session_id, str) or not session_id.strip():
            raise AuditValidationError("session_id is required")
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise AuditPolicyError("unknown audit session") from exc

    def _session(self, session: AuditSession | str, *, token: str | None = None) -> AuditSession:
        """Authenticate a session handle or an ID/token pair.

        A raw ID is never sufficient for a gateway operation.  The only
        implicit authentication path is the exact immutable handle returned by
        :meth:`open_session` and owned by this service instance.
        """

        if isinstance(session, AuditSession):
            with self._lock:
                known = self._sessions.get(session.session_id)
                if known is not session:
                    raise AuditPolicyError("session is not owned by this service")
                supplied = session.session_token if token is None else token
                if not isinstance(supplied, str) or not supplied.strip():
                    raise AuditPolicyError("session token is required")
                return self._refresh_session(session.session_id, token=supplied, existing=session)
        if not isinstance(token, str) or not token.strip():
            raise AuditPolicyError("session token is required")
        if not isinstance(session, str):
            raise AuditValidationError("session must be an AuditSession or session ID")
        with self._lock:
            local = self._sessions.get(session)
        if local is None:
            # An ID/token pair on a fresh service is a reconnect, not an
            # implicit way to hydrate persisted authority.  Require the
            # trusted launcher ceiling and current source revalidation.
            return self.reconnect_session(session, token)
        return self._refresh_session(session, token=token, existing=local)

    def _source_input(  # noqa: C901
        self, session: AuditSession
    ) -> tuple[Mapping[str, Any] | str | Path, Path | None, str, int | str]:
        """Return a policy-checked source and its current byte digest/token."""

        source = self.campaign_source
        if source is None:
            if self.native_diagnostic_config.bindings and session.source_ref is not None:
                digest = self._native_source_digest(session)
                return (
                    {
                        "native_source_digest": digest,
                        "source_revision": session.source_revision,
                    },
                    None,
                    digest,
                    session.source_revision,
                )
            raise CapabilityUnavailable("campaign source is not configured")
        if isinstance(source, Mapping):
            copied = _bounded_mapping(source, name="campaign_source", limit=MAX_SOURCE_BYTES)
            payload_revision = copied.get(
                "source_revision", copied.get("revision", session.source_revision)
            )
            if isinstance(payload_revision, bool) or not isinstance(payload_revision, (int, str)):
                raise AuditValidationError("campaign source revision must be an integer or token")
            return copied, None, _digest(copied), payload_revision
        path = Path(source)
        if not path.is_absolute() and self.source_root is not None:
            path = self.source_root / path
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CapabilityUnavailable("campaign source is unavailable") from exc
        if not session.policy.allows_path(resolved):
            raise AuditPolicyError("campaign source is outside allowed roots")
        try:
            if not resolved.is_file():
                raise AuditPolicyError("campaign source is not a regular file")
            raw = resolved.read_bytes()
        except OSError as exc:
            raise CapabilityUnavailable("campaign source cannot be read") from exc
        if len(raw) > MAX_SOURCE_BYTES:
            raise AuditValidationError("campaign source exceeds size limit")
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        revision: int | str = session.source_revision
        if isinstance(payload, Mapping):
            candidate = payload.get("source_revision", payload.get("revision", revision))
            if isinstance(candidate, (int, str)) and not isinstance(candidate, bool):
                revision = candidate
        return resolved, resolved.parent, hashlib.sha256(raw).hexdigest(), revision

    def _native_source_digest(self, session: AuditSession) -> str:
        """Revalidate a launcher-owned native source for reconnect/CAS checks."""

        from robot_sf.analysis_workbench import audit_native_diagnostic as native  # noqa: PLC0415

        binding = self._native_binding_for_session(session)
        request = native.NativeDiagnosticRequest(
            request_id=binding.request.request_id,
            admission=binding.admission,
            request=binding.request,
            recipe=binding.recipe,
            intervention={
                "intervention_id": "source-integrity-guard",
                "factor": "robot_goal",
                "robot_goal": (0.0, 0.0),
                "activation_epsilon_m": 0.0,
            },
            timeout_s=self.native_diagnostic_config.max_timeout_s,
        )
        root_fd: int | None = None
        try:
            native._validate_request(request)
            _root, root_fd, _receipt, _receipt_digest, resolution = native._open_and_resolve_source(
                request
            )
            if resolution.receipt is None:
                raise native.NativeDiagnosticError(["native source receipt is unavailable"])
            return resolution.receipt.source.sha256
        except (
            native.NativeDiagnosticError,
            OSError,
            ReviewContractsValidationError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise CapabilityUnavailable(f"native diagnostic source is unavailable: {exc}") from exc
        finally:
            if root_fd is not None:
                try:
                    os.close(root_fd)
                except OSError:
                    pass

    def _bind_context(
        self, session: AuditSession, context: AuditSelectionContext | Mapping[str, Any] | None
    ) -> AuditSelectionContext:
        selected = (
            session.context
            if context is None
            else (
                context
                if isinstance(context, AuditSelectionContext)
                else AuditSelectionContext.from_mapping(context)
            )
        )
        if selected != session.context:
            raise AuditContextConflict(
                "selection context is stale",
                expected=session.context.context_revision,
                actual=selected.context_revision,
            )
        return selected

    def _source_identities(self, session: AuditSession) -> frozenset[str]:
        """Return the exact source identities accepted for this session."""

        identities = {session.source_digest}
        if session.context.source_identity:
            identities.add(session.context.source_identity)
        if session.source_ref is not None:
            identities.update(
                value
                for value in (session.source_ref.artifact_id, session.source_ref.sha256)
                if value
            )
        return frozenset(identities)

    def _source_ref_matches(self, session: AuditSession, source: SourceRef) -> bool:
        """Require every supplied source-reference identity to agree."""

        identities = {value for value in (source.artifact_id, source.sha256) if value}
        if identities and not identities.issubset(self._source_identities(session)):
            return False
        canonical = session.source_ref
        if canonical is None:
            return not source.sha256 or source.sha256 == session.source_digest
        for field_name in SourceRef.__dataclass_fields__:
            supplied = getattr(source, field_name)
            expected = getattr(canonical, field_name)
            if supplied and supplied != expected:
                # A canonical SourceRef may omit its checksum while the
                # session still has one from the admitted source bytes.
                if field_name == "sha256" and not expected and supplied == session.source_digest:
                    continue
                return False
        return True

    @staticmethod
    def _source_revision_token(value: int | str | None) -> str:
        """Return a comparable non-empty source-revision token."""

        if value in (None, 0, ""):
            return ""
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return ""
        return str(value).strip()

    def _saved_annotation_for_context(
        self,
        session: AuditSession,
        record: Annotation,
        *,
        episode_id: str,
        source_digest: str,
        expected_revision: str,
        digest_only: bool,
    ) -> bool:
        """Return whether one durable annotation belongs to the selection."""

        if record.episode_id != episode_id:
            return False
        if digest_only:
            if record.source_identity != source_digest or record.source_revision not in (
                0,
                "0",
            ):
                return False
        elif (
            not expected_revision
            or not record.source_identity
            or record.source_identity not in (self._source_identities(session))
        ):
            return False
        if not digest_only and self._source_revision_token(record.source_revision) != (
            expected_revision
        ):
            return False
        if record.source_ref is None:
            return True
        return bool(record.source_ref.sha256) and (
            record.source_ref.sha256 == source_digest
            and self._source_ref_matches(session, record.source_ref)
        )

    def _saved_finding_for_context(
        self,
        record: Finding,
        *,
        episode_id: str,
        expected_revision: str,
        digest_only: bool,
    ) -> bool:
        """Return whether one durable finding belongs to the selection."""

        members = (
            set(record.candidate_members)
            | set(record.confirmed_members)
            | set(record.negative_controls)
        )
        if episode_id not in members:
            return False
        if digest_only:
            return record.source_revision == "0"
        return (
            bool(expected_revision)
            and self._source_revision_token(record.source_revision) == expected_revision
        )

    def _saved_record_for_context(
        self,
        session: AuditSession,
        stored: StoredRecord,
        *,
        episode_id: str,
        episode_ref: EpisodeRef,
    ) -> bool:
        """Return whether one durable annotation/finding belongs to the selection.

        The store projection already enforces record identity and revision
        consistency.  This second, service-owned check is deliberately
        conservative: a record with missing provenance is excluded rather
        than being presented as current evidence after a process restart.
        Digest-only sessions accept only explicit zero-bound records after the
        selected generated reference proves the current source digest.
        """

        source_digest = session.source_digest
        if not source_digest or episode_ref.source_digest != source_digest:
            return False
        expected_revision = self._source_revision_token(session.source_revision)
        digest_only = session.source_revision in (0, "")
        record = stored.record
        if stored.record_type == "annotation" and isinstance(record, Annotation):
            return self._saved_annotation_for_context(
                session,
                record,
                episode_id=episode_id,
                source_digest=source_digest,
                expected_revision=expected_revision,
                digest_only=digest_only,
            )
        if stored.record_type == "finding" and isinstance(record, Finding):
            return self._saved_finding_for_context(
                record,
                episode_id=episode_id,
                expected_revision=expected_revision,
                digest_only=digest_only,
            )
        return False

    @staticmethod
    def _validate_annotation_provenance_status(
        session: AuditSession,
        annotation: Annotation,
    ) -> None:
        """Reject explicit stale/mutated provenance before finding derivation."""

        if annotation.provenance_status in {
            SOURCE_PROVENANCE_STALE,
            SOURCE_PROVENANCE_MUTATED,
        }:
            raise AuditContextConflict("annotation provenance status is not current")
        if annotation.provenance_status == SOURCE_PROVENANCE_UNAVAILABLE and (
            annotation.source_ref is not None or annotation.source_identity != session.source_digest
        ):
            raise AuditContextConflict("annotation provenance is unavailable for this source")

    @staticmethod
    def _validate_annotation_author(session: AuditSession, annotation: Annotation) -> None:
        """Require annotation authorship to match the authenticated actor."""

        if annotation.author_kind != session.actor.kind:
            raise AuditPolicyError("annotation author_kind does not match session actor")
        if annotation.author_id and annotation.author_id != session.actor.actor_id:
            raise AuditPolicyError("annotation author_id does not match session actor")

    def _validate_annotation_references(
        self,
        session: AuditSession,
        annotation: Annotation,
    ) -> None:
        """Require nested reference identity and revision to match the source."""

        expected_revision = (
            "0" if session.source_revision in (0, "") else str(session.source_revision)
        )
        for reference in annotation.references:
            nested_source = reference.source
            if nested_source is not None and not self._source_ref_matches(session, nested_source):
                raise AuditContextConflict(
                    "annotation reference source does not match session source"
                )
            if reference.source_revision and reference.source_revision != expected_revision:
                raise AuditContextConflict(
                    "annotation reference source revision does not match selection"
                )

    def _validate_annotation_for_finding(
        self,
        session: AuditSession,
        annotation: Annotation,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None,
    ) -> None:
        """Require an annotation to be current before deriving a finding."""

        selected = self._bind_context(session, context)
        if not selected.episode_id:
            raise AuditContextConflict("finding derivation requires a selected episode")
        campaign = self._scan(session)
        episode = campaign.episode(selected.episode_id)
        episode_ref = episode.episode_ref if episode is not None else None
        if (
            episode is None
            or not isinstance(episode_ref, EpisodeRef)
            or episode_ref.source_digest != session.source_digest
        ):
            raise CapabilityUnavailable("selected generated episode is unavailable or ambiguous")
        self._validate_annotation_provenance_status(session, annotation)
        self._validate_annotation_author(session, annotation)
        if not self._saved_annotation_for_context(
            session,
            annotation,
            episode_id=selected.episode_id,
            source_digest=session.source_digest,
            expected_revision=self._source_revision_token(session.source_revision),
            digest_only=session.source_revision in (0, ""),
        ):
            raise AuditContextConflict("annotation provenance does not match selected source")
        self._validate_annotation_references(session, annotation)

    def _read_saved_records_for_context(
        self,
        session: AuditSession,
        *,
        episode_id: str,
        episode_ref: EpisodeRef,
    ) -> tuple[StoredRecord, ...]:
        """Read current source-bound annotation/finding envelopes from BA-03."""

        records: list[StoredRecord] = []
        for record_type_name, record_class in (
            ("annotation", Annotation),
            ("finding", Finding),
        ):
            for stored in self.store.list_records(record_type=record_type_name):
                if not isinstance(stored, StoredRecord):
                    raise AuditStoreError("durable record projection is malformed")
                if stored.deleted or stored.record is None:
                    raise AuditStoreError("deleted durable record reached active projection")
                if stored.record_type != record_type_name or not isinstance(
                    stored.record, record_class
                ):
                    raise AuditStoreError("durable record type is inconsistent")
                if stored.revision <= 0 or stored.global_revision <= 0:
                    raise AuditStoreError("durable record revision is invalid")
                if not self._saved_record_for_context(
                    session,
                    stored,
                    episode_id=episode_id,
                    episode_ref=episode_ref,
                ):
                    continue
                # Do not let a caller-controlled record payload echo the
                # opaque service token through the public read envelope.
                if self._contains_sensitive_text(stored.to_dict(), session.session_token):
                    raise CapabilityUnavailable("durable record contains a session secret")
                records.append(stored)
        return tuple(sorted(records, key=lambda item: (item.record_id, item.record_type)))

    def _validate_context_source(
        self, session: AuditSession, selected: AuditSelectionContext
    ) -> None:
        """Keep a newly selected campaign/source bound to the opened session."""

        if selected.campaign_id != session.context.campaign_id:
            raise AuditContextConflict(
                "selection campaign does not match session source",
                expected=session.context.campaign_id,
                actual=selected.campaign_id,
            )
        identities = self._source_identities(session)
        if session.context.source_identity and not selected.source_identity:
            raise AuditContextConflict(
                "selection source identity cannot be cleared without source CAS",
                expected=session.context.source_identity,
                actual=selected.source_identity,
            )
        if selected.source_identity and selected.source_identity not in identities:
            raise AuditContextConflict(
                "selection source identity does not match session source",
                expected=session.source_digest,
                actual=selected.source_identity,
            )

    def _assert_source(
        self, session: AuditSession, *, expected_source_revision: int | str | None = None
    ) -> tuple[Path | None, str, int | str]:
        source, _root, digest, revision = self._source_input(session)
        del source
        if digest != session.source_digest:
            raise AuditContextConflict(
                "campaign source changed", expected=session.source_digest, actual=digest
            )
        if revision != session.source_revision:
            raise AuditContextConflict(
                "campaign source revision changed",
                expected=session.source_revision,
                actual=revision,
            )
        if (
            expected_source_revision is not None
            and expected_source_revision != session.source_revision
        ):
            raise AuditContextConflict(
                "source revision CAS failed",
                expected=session.source_revision,
                actual=expected_source_revision,
            )
        return _root, digest, revision

    def _current_source_ref(self, session: AuditSession) -> SourceRef | None:
        """Derive the complete current source identity for reconnect checks."""

        source, root, digest, _revision = self._source_input(session)
        if isinstance(source, Mapping):
            return self.source_ref
        return self._admit_source_reference(
            source,
            root=root or Path(source).parent,
            source_digest=digest,
        )

    def _admit_source_reference(
        self,
        source: Mapping[str, Any] | str | Path,
        *,
        root: Path,
        source_digest: str,
    ) -> SourceRef | None:
        """Bind a file source reference to the canonical scan contract.

        The scan owner is the authority for relative URI/path, format, schema,
        and byte-digest identity.  A configured reference may choose a stable
        artifact identifier or provenance metadata, but it cannot override the
        admitted path or the format/schema derived from the source bytes.
        """

        if isinstance(source, Mapping):
            # In-memory campaigns use the service's canonical digest convention
            # (without the scan owner's serialized trailing newline).  There is
            # no local URI/path to bind here; nested references still undergo
            # the field-by-field service check.
            return self.source_ref
        try:
            admitted_report = scan_campaign(
                source,
                root=root,
                source_ref=self.source_ref,
                # Admission needs only the BA-01 source identity contract;
                # detector execution belongs to an explicit campaign read.
                detector_ids=(),
            )
            admitted = admitted_report.audit.source
            if admitted is None:
                raise AuditContextConflict("scan contract did not produce a source reference")
            if admitted.sha256.casefold() != source_digest.casefold():
                raise AuditContextConflict(
                    "campaign source changed during source-reference admission",
                    expected=source_digest,
                    actual=admitted.sha256,
                )
            if self.source_ref is None:
                return admitted

            derived_report = scan_campaign(
                source,
                root=root,
                source_ref=None,
                # Keep canonical derivation detectorless and bounded to source
                # identity rather than running the default detector registry.
                detector_ids=(),
            )
            derived = derived_report.audit.source
            if derived is None:
                raise AuditContextConflict("scan contract did not derive a source reference")
            if derived.sha256.casefold() != source_digest.casefold():
                raise AuditContextConflict(
                    "campaign source changed during source-reference admission",
                    expected=source_digest,
                    actual=derived.sha256,
                )
            if (
                admitted.uri != derived.uri
                or admitted.format != derived.format
                or admitted.schema != derived.schema
                or admitted.sha256.casefold() != derived.sha256.casefold()
            ):
                expected_identity = {
                    name: getattr(derived, name) for name in SourceRef.__dataclass_fields__
                }
                actual_identity = {
                    name: getattr(admitted, name) for name in SourceRef.__dataclass_fields__
                }
                raise AuditContextConflict(
                    "configured source reference does not match admitted path or format",
                    expected=expected_identity,
                    actual=actual_identity,
                )
            return admitted
        except AuditScanError as exc:
            raise AuditContextConflict(
                f"configured source reference rejected by scan contract: {exc}"
            ) from exc

    def _check_session(self, session: AuditSession) -> None:
        if session.policy.kill_switch:
            raise AuditCancelled("session policy kill switch is active")
        if session.cancelled:
            raise AuditCancelled(session.cancel_reason or "session is cancelled")

    @staticmethod
    def _check_policy_scope(
        session: AuditSession, *, repository: str | None = None, recipe_id: str | None = None
    ) -> None:
        """Enforce repository/recipe allowlists at the trusted service boundary."""

        if repository is not None:
            repository = _bounded_text(repository, name="repository")
            if not session.policy.allows_repository(repository):
                raise AuditPolicyError("repository is not allowed by the session policy")
        if recipe_id is not None:
            recipe_id = _bounded_text(recipe_id, name="recipe_id")
            if not session.policy.allows_recipe(recipe_id):
                raise AuditPolicyError("recipe is not allowed by the session policy")

    def _charge(
        self, session: AuditSession, *, tokens: int = 0, compute: float = 0.0, issue_writes: int = 0
    ) -> None:
        """Atomically consume finite aggregate budgets before provider work."""

        tokens = _nonnegative_int(tokens, name="tokens")
        compute = _finite(compute, name="compute")
        issue_writes = _nonnegative_int(issue_writes, name="issue_writes")
        if compute < 0:
            raise AuditValidationError("compute must be non-negative")
        with self._lock:
            self._charge_locked(
                session,
                tokens=tokens,
                compute=compute,
                issue_writes=issue_writes,
            )

    @staticmethod
    def _charge_locked(
        session: AuditSession, *, tokens: int, compute: float, issue_writes: int
    ) -> None:
        """Apply a validated budget charge while the service lock is held."""

        usage = session.usage
        if usage.tokens + tokens > session.policy.token_budget:
            raise AuditBudgetExceeded("aggregate token budget exhausted")
        if usage.compute + compute > session.policy.compute_budget:
            raise AuditBudgetExceeded("aggregate compute budget exhausted")
        if usage.issue_writes + issue_writes > session.policy.issue_write_budget:
            raise AuditBudgetExceeded("aggregate issue-write budget exhausted")
        object.__setattr__(
            session,
            "usage",
            SessionUsage(
                tokens=usage.tokens + tokens,
                compute=usage.compute + compute,
                issue_writes=usage.issue_writes + issue_writes,
            ),
        )

    def _reserve_budget_locked(
        self,
        session: AuditSession,
        *,
        operation_id: str,
        tokens: int,
        compute: float,
        issue_writes: int,
    ) -> dict[str, Any]:
        """Charge and issue one opaque reservation while the lock is held."""

        self._charge_locked(
            session,
            tokens=tokens,
            compute=compute,
            issue_writes=issue_writes,
        )
        reservation_id = f"budget-reservation-{secrets.token_urlsafe(24)}"
        self._reservations[reservation_id] = BudgetReservation(
            reservation_id=reservation_id,
            session_id=session.session_id,
            operation_id=operation_id,
            tokens=tokens,
            compute=compute,
            issue_writes=issue_writes,
        )
        return {
            "reservation_id": reservation_id,
            "operation_id": operation_id,
            "reserved": {
                "tokens": tokens,
                "compute": compute,
                "issue_writes": issue_writes,
            },
            "usage": session.usage.to_dict(),
        }

    @staticmethod
    def _reservation_payload(
        session: AuditSession, reservation: BudgetReservation
    ) -> dict[str, Any]:
        """Reconstruct the in-process capability for an active reservation."""

        return {
            "reservation_id": reservation.reservation_id,
            "operation_id": reservation.operation_id,
            "reserved": {
                "tokens": reservation.tokens,
                "compute": reservation.compute,
                "issue_writes": reservation.issue_writes,
            },
            "usage": session.usage.to_dict(),
        }

    def consume_budget(
        self,
        session: AuditSession | str,
        *,
        tokens: int = 0,
        compute: float = 0.0,
        issue_writes: int = 0,
        token: str | None = None,
        operation_id: str | None = None,
    ) -> ServiceResult[dict[str, Any]]:
        """Consume policy budget through the same durable operation path."""

        target = self._session(session, token=token)
        operation_id = self._operation_id(operation_id)

        def consume() -> dict[str, Any]:
            checked_tokens = _nonnegative_int(tokens, name="tokens")
            checked_compute = _finite(compute, name="compute")
            checked_issue_writes = _nonnegative_int(issue_writes, name="issue_writes")
            if checked_compute < 0:
                raise AuditValidationError("compute must be non-negative")

            def charge(working: AuditSession, _state: dict[str, Any]) -> dict[str, Any]:
                self._check_session(working)
                self._charge_locked(
                    working,
                    tokens=checked_tokens,
                    compute=checked_compute,
                    issue_writes=checked_issue_writes,
                )
                return working.usage.to_dict()

            value, _replayed = self._mutate_session_authority(
                target,
                transaction_id=f"authority.budget.consume:{target.session_id}:{operation_id}",
                request_digest=_digest(
                    {
                        "operation_id": operation_id,
                        "tokens": checked_tokens,
                        "compute": checked_compute,
                        "issue_writes": checked_issue_writes,
                    }
                ),
                callback=charge,
            )
            if not isinstance(value, Mapping):
                raise AuditAuthorityError("budget consume returned malformed usage")
            return dict(value)

        result, _receipt = self._execute(
            target,
            operation_type="budget.consume",
            operation_id=operation_id,
            context=target.context,
            request_digest=_operation_request_digest(
                {
                    "tokens": tokens,
                    "compute": compute,
                    "issue_writes": issue_writes,
                }
            ),
            callback=consume,
        )
        return result

    def reserve_budget(
        self,
        session: AuditSession | str,
        *,
        tokens: int = 0,
        compute: float = 0.0,
        issue_writes: int = 0,
        token: str | None = None,
        operation_id: str | None = None,
        enforce_active: bool = True,
    ) -> ServiceResult[dict[str, Any]]:
        """Reserve finite budget and return an opaque settlement capability."""

        target = self._session(session, token=token)
        opid = self._operation_id(operation_id)
        if opid.startswith("codex-budget-reserve:"):
            raise AuditValidationError("Codex reservation operation IDs are service-owned")

        def reserve() -> dict[str, Any]:
            checked_tokens = _nonnegative_int(tokens, name="tokens")
            checked_compute = _finite(compute, name="compute")
            checked_issue_writes = _nonnegative_int(issue_writes, name="issue_writes")
            if checked_compute < 0:
                raise AuditValidationError("compute must be non-negative")

            def reserve_state(working: AuditSession, state: dict[str, Any]) -> dict[str, Any]:
                if enforce_active:
                    self._check_session(working)
                self._charge_locked(
                    working,
                    tokens=checked_tokens,
                    compute=checked_compute,
                    issue_writes=checked_issue_writes,
                )
                reservation_id = f"budget-reservation-{secrets.token_urlsafe(24)}"
                reservation = {
                    "reservation_id": reservation_id,
                    "session_id": working.session_id,
                    "operation_id": opid,
                    "tokens": checked_tokens,
                    "compute": checked_compute,
                    "issue_writes": checked_issue_writes,
                }
                state["reservations"][reservation_id] = reservation
                return {
                    "reservation_id": reservation_id,
                    "operation_id": opid,
                    "reserved": {
                        "tokens": checked_tokens,
                        "compute": checked_compute,
                        "issue_writes": checked_issue_writes,
                    },
                    "usage": working.usage.to_dict(),
                }

            value, _replayed = self._mutate_session_authority(
                target,
                transaction_id=f"authority.budget.reserve:{target.session_id}:{opid}",
                request_digest=_digest(
                    {
                        "operation_id": opid,
                        "tokens": checked_tokens,
                        "compute": checked_compute,
                        "issue_writes": checked_issue_writes,
                    }
                ),
                callback=reserve_state,
            )
            if not isinstance(value, Mapping):
                raise AuditAuthorityError("budget reservation returned malformed capability")
            return dict(value)

        result, receipt = self._execute(
            target,
            operation_type="budget.reserve",
            operation_id=opid,
            context=target.context,
            enforce_active=enforce_active,
            request_digest=_operation_request_digest(
                {
                    "tokens": tokens,
                    "compute": compute,
                    "issue_writes": issue_writes,
                    "enforce_active": enforce_active,
                }
            ),
            callback=reserve,
        )
        if result.status == "committed" and receipt.replayed and result.value is None:
            with self._lock:
                candidates = tuple(
                    item
                    for item in self._reservations.values()
                    if item.session_id == target.session_id and item.operation_id == opid
                )
            if len(candidates) == 1:
                return ServiceResult(
                    "committed",
                    self._reservation_payload(target, candidates[0]),
                    result.reason,
                    receipt,
                    result.context,
                )
            return ServiceResult(
                "unavailable",
                None,
                "budget reservation capability is no longer available for replay",
                receipt,
                result.context,
            )
        return result

    def recover_budget_reservation(
        self,
        session: AuditSession | str,
        *,
        operation_id: str,
        token: str | None = None,
    ) -> dict[str, Any] | None:
        """Recover one authenticated active reservation after a lost reply.

        This read-only capability lookup does not grant new budget or provider
        authority. It is used only to settle a known no-send failure; a missing
        or ambiguous match must never be guessed.
        """

        target = self._session(session, token=token)
        opid = self._operation_id(operation_id)
        with self._lock:
            candidates = tuple(
                item
                for item in self._reservations.values()
                if item.session_id == target.session_id and item.operation_id == opid
            )
        return self._reservation_payload(target, candidates[0]) if len(candidates) == 1 else None

    def settle_budget(  # noqa: C901, PLR0913
        self,
        session: AuditSession | str,
        *,
        reserved_tokens: int = 0,
        reserved_compute: float = 0.0,
        reserved_issue_writes: int = 0,
        actual_tokens: int | None,
        actual_compute: float | None,
        actual_issue_writes: int | None = 0,
        reservation_id: str | None = None,
        reservation_operation_id: str | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[dict[str, Any] | CapabilityResult]:
        """Atomically replace a provider reservation with metered usage.

        A missing or malformed provider meter releases the reservation and
        returns ``unavailable``.  Usage that would exceed policy is likewise
        released and returns ``denied``; no provider result is promoted to a
        successful operation without finite, policy-fitting accounting.
        """

        target = self._session(session, token=token)
        requested_opid = self._operation_id(operation_id)
        opid = requested_opid
        if reservation_id is None:
            with self._lock:
                candidates = tuple(
                    item
                    for item in self._reservations.values()
                    if item.session_id == target.session_id and item.operation_id == requested_opid
                )
            if len(candidates) == 1:
                reservation_id = candidates[0].reservation_id
                reservation_operation_id = requested_opid
                # A settlement must not reuse the durable reserve operation's
                # ID, whose receipt already belongs to budget.reserve.
                opid = f"{requested_opid}-settle"

        def settle() -> dict[str, Any] | CapabilityResult:  # noqa: C901
            if not reservation_id:
                raise AuditPolicyError("budget reservation capability is required")
            if reservation_operation_id is not None:
                _bounded_text(reservation_operation_id, name="reservation_operation_id")

            def settle_state(  # noqa: C901 - one extra fail-closed Codex authority guard
                working: AuditSession, state: dict[str, Any]
            ) -> dict[str, Any] | CapabilityResult:
                try:
                    reserved_t = _nonnegative_int(reserved_tokens, name="reserved_tokens")
                    reserved_c = _finite(reserved_compute, name="reserved_compute")
                    reserved_i = _nonnegative_int(
                        reserved_issue_writes, name="reserved_issue_writes"
                    )
                    if reserved_c < 0:
                        raise AuditValidationError("reserved_compute must be non-negative")
                except AuditValidationError as exc:
                    return CapabilityResult("budget.meter", "unavailable", reason=str(exc))
                raw = state["reservations"].get(reservation_id)
                if not isinstance(raw, Mapping) or raw.get("session_id") != working.session_id:
                    raise AuditPolicyError("budget reservation is not owned by this session")
                expected_owner = reservation_operation_id or requested_opid
                if expected_owner != raw.get("operation_id"):
                    raise AuditPolicyError("budget reservation owner does not match operation")
                if self._codex_owns_reservation(state, expected_owner, reservation_id):
                    raise AuditPolicyError(
                        "Codex reservation requires the durable Codex finish transition"
                    )
                if (
                    reserved_t != raw.get("tokens")
                    or reserved_c != raw.get("compute")
                    or reserved_i != raw.get("issue_writes")
                ):
                    raise AuditPolicyError("budget reservation amounts do not match capability")
                try:
                    actual_t = _nonnegative_int(actual_tokens, name="actual_tokens")
                    actual_c = _finite(actual_compute, name="actual_compute")
                    actual_i = _nonnegative_int(actual_issue_writes, name="actual_issue_writes")
                    if actual_c < 0:
                        raise AuditValidationError("actual_compute must be non-negative")
                except AuditValidationError as exc:
                    usage = working.usage
                    object.__setattr__(
                        working,
                        "usage",
                        SessionUsage(
                            tokens=max(0, usage.tokens - int(raw["tokens"])),
                            compute=max(0.0, usage.compute - float(raw["compute"])),
                            issue_writes=max(0, usage.issue_writes - int(raw["issue_writes"])),
                        ),
                    )
                    state["reservations"].pop(reservation_id, None)
                    return CapabilityResult("budget.meter", "unavailable", reason=str(exc))
                usage = working.usage
                if (
                    usage.tokens < int(raw["tokens"])
                    or usage.compute < float(raw["compute"])
                    or usage.issue_writes < int(raw["issue_writes"])
                ):
                    raise AuditPolicyError("budget reservation is not owned by this session")
                base = SessionUsage(
                    tokens=usage.tokens - int(raw["tokens"]),
                    compute=usage.compute - float(raw["compute"]),
                    issue_writes=usage.issue_writes - int(raw["issue_writes"]),
                )
                state["reservations"].pop(reservation_id, None)
                if (
                    base.tokens + actual_t > working.policy.token_budget
                    or base.compute + actual_c > working.policy.compute_budget
                    or base.issue_writes + actual_i > working.policy.issue_write_budget
                ):
                    object.__setattr__(working, "usage", base)
                    return CapabilityResult(
                        "budget.meter",
                        "denied",
                        reason="metered provider usage exceeds session budget",
                    )
                object.__setattr__(
                    working,
                    "usage",
                    SessionUsage(
                        tokens=base.tokens + actual_t,
                        compute=base.compute + actual_c,
                        issue_writes=base.issue_writes + actual_i,
                    ),
                )
                return {
                    "reserved": {
                        "tokens": int(raw["tokens"]),
                        "compute": float(raw["compute"]),
                        "issue_writes": int(raw["issue_writes"]),
                    },
                    "actual": {
                        "tokens": actual_t,
                        "compute": actual_c,
                        "issue_writes": actual_i,
                    },
                    "usage": working.usage.to_dict(),
                }

            value, _replayed = self._mutate_session_authority(
                target,
                transaction_id=f"authority.budget.settle:{target.session_id}:{opid}",
                request_digest=_digest(
                    {
                        "operation_id": opid,
                        "reservation_id": reservation_id,
                        "reservation_operation_id": reservation_operation_id,
                        "reserved_tokens": reserved_tokens,
                        "reserved_compute": reserved_compute,
                        "reserved_issue_writes": reserved_issue_writes,
                        "actual_tokens": actual_tokens,
                        "actual_compute": actual_compute,
                        "actual_issue_writes": actual_issue_writes,
                    }
                ),
                callback=settle_state,
            )
            if isinstance(value, Mapping):
                if value.get("capability") == "budget.meter":
                    return CapabilityResult(
                        str(value.get("capability")),
                        str(value.get("status", "unavailable")),
                        value.get("value"),
                        str(value.get("reason", "")),
                        str(value.get("provider", "")),
                    )
                return dict(value)
            raise AuditAuthorityError("budget settlement returned malformed result")

        result, _ = self._execute(
            target,
            operation_type="budget.settle",
            operation_id=opid,
            context=target.context,
            enforce_source=False,
            enforce_active=False,
            request_digest=_operation_request_digest(
                {
                    "reserved_tokens": reserved_tokens,
                    "reserved_compute": reserved_compute,
                    "reserved_issue_writes": reserved_issue_writes,
                    "actual_tokens": actual_tokens,
                    "actual_compute": actual_compute,
                    "actual_issue_writes": actual_issue_writes,
                    "reservation_id": reservation_id,
                    "reservation_operation_id": reservation_operation_id,
                }
            ),
            callback=settle,
        )
        return result

    def _release_reservation(
        self,
        session: AuditSession,
        *,
        reserved_tokens: int,
        reserved_compute: float,
        reserved_issue_writes: int,
    ) -> None:
        """Release a reservation while the service lock is held."""

        usage = session.usage
        if (
            isinstance(reserved_tokens, bool)
            or not isinstance(reserved_tokens, int)
            or reserved_tokens < 0
            or isinstance(reserved_issue_writes, bool)
            or not isinstance(reserved_issue_writes, int)
            or reserved_issue_writes < 0
            or isinstance(reserved_compute, bool)
            or not isinstance(reserved_compute, (int, float))
            or not math.isfinite(float(reserved_compute))
            or reserved_compute < 0
        ):
            return
        object.__setattr__(
            session,
            "usage",
            SessionUsage(
                tokens=max(0, usage.tokens - reserved_tokens),
                compute=max(0.0, usage.compute - float(reserved_compute)),
                issue_writes=max(0, usage.issue_writes - reserved_issue_writes),
            ),
        )

    def open_session(  # noqa: C901, PLR0912, PLR0915
        self,
        context: AuditSelectionContext | Mapping[str, Any],
        *,
        actor: ActorRef | Mapping[str, Any] | str = "agent",
        actor_id: str = "",
        policy: SessionPolicy | Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> AuditSession:
        """Open a policy-bound session after binding the current source bytes."""

        selected = (
            context
            if isinstance(context, AuditSelectionContext)
            else AuditSelectionContext.from_mapping(context)
        )
        actor_ref = ActorRef.from_value(actor, actor_id=actor_id)
        session_policy = (
            policy
            if isinstance(policy, SessionPolicy)
            else SessionPolicy.from_mapping(policy)
            if policy is not None
            else SessionPolicy()
        )
        sid = _bounded_text(session_id or f"audit-session-{uuid.uuid4().hex}", name="session_id")
        with self._lock:
            if sid in self._sessions:
                raise AuditPolicyError("session_id is already in use")
        # Bind source identity without requiring a full scan at session creation.
        source = self.campaign_source
        source_path_for_admission: Path | None = None
        if source is None:
            # A native-only launcher may omit BA-03's campaign source because
            # the trusted diagnostic binding owns its admitted bytes.  Carry
            # that immutable digest into the session/receipt; the native
            # preflight still reopens and verifies the receipt before work.
            source_digest = self.source_ref.sha256 if self.source_ref is not None else ""
            source_revision: int | str = selected.source_revision
        elif isinstance(source, Mapping):
            source_digest = _digest(source)
            try:
                source_campaign_id = campaign_identity_from_payload(source)
            except AuditScanError as exc:
                raise AuditContextConflict(f"campaign source identity is invalid: {exc}") from exc
            if source_campaign_id is not None and source_campaign_id != selected.campaign_id:
                raise AuditContextConflict(
                    "selection campaign does not match source",
                    expected=source_campaign_id,
                    actual=selected.campaign_id,
                )
            candidate = source.get(
                "source_revision", source.get("revision", selected.source_revision)
            )
            source_revision = (
                candidate
                if isinstance(candidate, (int, str)) and not isinstance(candidate, bool)
                else selected.source_revision
            )
        else:
            path = Path(source)
            if not path.is_absolute() and self.source_root is not None:
                path = self.source_root / path
            resolved = path.resolve(strict=False)
            if not session_policy.allows_path(resolved):
                raise AuditPolicyError("campaign source is outside allowed roots")
            source_path_for_admission = resolved
            try:
                raw = resolved.read_bytes()
            except OSError as exc:
                raise CapabilityUnavailable("campaign source is unavailable") from exc
            if len(raw) > MAX_SOURCE_BYTES:
                raise AuditValidationError("campaign source exceeds size limit")
            source_digest = hashlib.sha256(raw).hexdigest()
            # A missing source revision is represented explicitly as ``0``.  Do
            # not invent a monotonic value for an artifact that only has a
            # content digest; the digest still provides the source CAS guard.
            source_revision = selected.source_revision
            try:
                payload = json.loads(raw)
                if isinstance(payload, Mapping):
                    try:
                        source_campaign_id = campaign_identity_from_payload(payload)
                    except AuditScanError as exc:
                        raise AuditContextConflict(
                            f"campaign source identity is invalid: {exc}"
                        ) from exc
                    if (
                        source_campaign_id is not None
                        and source_campaign_id != selected.campaign_id
                    ):
                        raise AuditContextConflict(
                            "selection campaign does not match source",
                            expected=source_campaign_id,
                            actual=selected.campaign_id,
                        )
                    candidate = payload.get(
                        "source_revision", payload.get("revision", source_revision)
                    )
                    if isinstance(candidate, (int, str)) and not isinstance(candidate, bool):
                        source_revision = candidate
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if (
            source_digest
            and self.source_ref is not None
            and self.source_ref.sha256
            and self.source_ref.sha256.casefold() != source_digest.casefold()
        ):
            raise AuditContextConflict(
                "configured source reference digest does not match observed source",
                expected=source_digest,
                actual=self.source_ref.sha256,
            )
        if source_path_for_admission is not None:
            canonical_source_ref = self._admit_source_reference(
                source_path_for_admission,
                root=self.source_root or source_path_for_admission.parent,
                source_digest=source_digest,
            )
            if canonical_source_ref is not None:
                self.source_ref = canonical_source_ref
        if selected.source_identity:
            valid_identities = {source_digest}
            if self.source_ref is not None:
                valid_identities.update({self.source_ref.artifact_id, self.source_ref.sha256})
            if selected.source_identity not in valid_identities:
                raise AuditContextConflict("selection source identity does not match source")
        if selected.source_revision not in (0, "") and selected.source_revision != source_revision:
            raise AuditContextConflict(
                "selection source revision does not match source",
                expected=source_revision,
                actual=selected.source_revision,
            )
        if selected.source_revision in (0, "") and source_revision not in (0, ""):
            # Persist the source revision discovered at admission in the
            # initial context.  Otherwise the first context advance would
            # normalize this omitted value and make historical Codex context
            # snapshots appear to change source identity on reopen.
            selected = replace(selected, source_revision=source_revision)
        session = AuditSession(
            session_id=sid,
            actor=actor_ref,
            policy=session_policy,
            context=selected,
            source_digest=source_digest,
            source_revision=source_revision,
            session_token=secrets.token_urlsafe(32),
            source_ref=self.source_ref,
        )
        token_verifier = AuditAuthorityStore.new_token_verifier(session.session_token)
        transaction_id = f"authority.session.open:{sid}"
        request_digest = _digest(
            {
                "session_id": sid,
                "actor": actor_ref.to_dict(),
                "policy": session_policy.to_dict(),
                "context": selected.to_dict(),
                "source_digest": source_digest,
                "source_revision": source_revision,
            }
        )

        def create(authority_state: dict[str, Any]) -> dict[str, str]:
            if sid in authority_state["sessions"]:
                raise AuthorityOperationConflict("session_id is already in use")
            authority_state["sessions"][sid] = self._authority_session_record(
                session, token_verifier
            )
            return {"session_id": sid}

        try:
            mutation = self.authority.mutate(transaction_id, request_digest, create)
        except AuthorityOperationConflict as exc:
            raise AuditPolicyError(str(exc)) from exc
        if mutation.replayed:
            raise AuditPolicyError("session_id is already in use")
        with self._lock:
            self._sessions[sid] = session
            self._load_reservations_state(mutation.state)
        return session

    create_session = open_session

    def reconnect_session(
        self,
        session_id: str,
        token: str,
        *,
        policy: SessionPolicy | Mapping[str, Any] | None = None,
    ) -> AuditSession:
        """Recover a persisted session after reconnecting to this service.

        The caller must present the original opaque token and a trusted
        launcher policy (either supplied here or at service construction).
        Persisted policy bytes can narrow a session's authority but can never
        grant more authority during recovery.
        """

        sid = _bounded_text(session_id, name="session_id")
        if not isinstance(token, str) or not token.strip():
            raise AuditPolicyError("session token is required")
        trusted = (
            policy
            if isinstance(policy, SessionPolicy)
            else SessionPolicy.from_mapping(policy)
            if policy is not None
            else self._trusted_policy
        )
        if trusted is None:
            raise AuditPolicyError("trusted launcher policy is required for session recovery")
        state = self.authority.snapshot()
        sessions = state.get("sessions", {})
        if not isinstance(sessions, Mapping) or sid not in sessions:
            raise AuditPolicyError("unknown audit session")
        raw = sessions[sid]
        if not isinstance(raw, Mapping):
            raise AuditPolicyError("persisted audit session is malformed")
        try:
            stored_policy = SessionPolicy.from_mapping(raw["policy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditPolicyError("persisted audit session policy is malformed") from exc
        if not self._policy_within_trusted(stored_policy, trusted):
            raise AuditPolicyError("persisted session policy exceeds trusted launcher policy")
        target = self._session_from_authority_record(raw, token=token)
        # Rebind the current source bytes before making the recovered handle
        # available.  A missing, changed, or out-of-root source is not a
        # recoverable session; callers must open a new explicitly authorized
        # session instead.
        try:
            self._assert_source(target)
            current_source_ref = self._current_source_ref(target)
        except (AuditContextConflict, CapabilityUnavailable, AuditPolicyError) as exc:
            raise AuditPolicyError(
                f"persisted session source cannot be revalidated: {exc}"
            ) from exc
        if current_source_ref != target.source_ref:
            raise AuditPolicyError(
                "persisted session source reference cannot be revalidated: full identity changed"
            )
        # Keep scan/provenance helpers on the same identity that was admitted
        # and persisted for this session.  Evidence links use the session copy
        # directly, so a later service reconnect cannot silently relabel them.
        self.source_ref = current_source_ref
        with self._lock:
            self._sessions[sid] = target
            self._load_reservations_state(state)
        return target

    reopen_session = reconnect_session
    recover_session = reconnect_session
    resume_session = reconnect_session
    connect_session = reconnect_session
    reconnect = reconnect_session

    def get_session(self, session: AuditSession | str, *, token: str | None = None) -> AuditSession:
        """Return a service-owned session object."""

        return self._session(session, token=token)

    def read_context(
        self,
        session: AuditSession | str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[AuditSelectionContext]:
        """Read the immutable session selection through the operation authority."""

        target = self._session(session, token=token)
        result, _ = self._execute(
            target,
            operation_type="read.context",
            operation_id=self._operation_id(operation_id),
            context=context,
            request_digest=_operation_request_digest(
                _context_request_value(context), requested_context=target.context.to_dict()
            ),
            callback=lambda: target.context,
        )
        return result

    def read_saved_records(
        self,
        session: AuditSession | str,
        episode_id: str | None = None,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[tuple[StoredRecord, ...]]:
        """Read current selected annotation/finding records from BA-03.

        The caller must authenticate the session and provide the current
        selection context.  ``episode_id`` is optional only as a convenience;
        when supplied it must equal the selected context episode exactly.  A
        record is returned only when its generated episode membership,
        source identity, and source revision all bind to that session.  A
        digest-only session may return explicitly zero-bound records only
        when its generated reference proves the same source digest.  The
        returned :class:`StoredRecord` values preserve the canonical nested
        record and durable revision envelope.
        """

        target = self._session(session, token=token)
        requested_episode_id = (
            None if episode_id is None else _bounded_text(episode_id, name="episode_id")
        )

        def read() -> tuple[StoredRecord, ...]:
            selected = self._bind_context(target, context)
            selected_episode_id = selected.episode_id
            if not selected_episode_id:
                raise CapabilityUnavailable("saved record reads require a selected episode")
            if not any(self._source_identities(target)):
                raise CapabilityUnavailable("current source identity is unavailable")
            if requested_episode_id is not None and requested_episode_id != selected_episode_id:
                raise AuditContextConflict(
                    "requested episode does not match selected context",
                    expected=selected_episode_id,
                    actual=requested_episode_id,
                )
            campaign = self._scan(target)
            episode = campaign.episode(selected_episode_id)
            if (
                episode is None
                or episode.episode_ref is None
                or episode.episode_ref.episode_id != selected_episode_id
                or episode.episode_ref.source_digest != target.source_digest
            ):
                raise CapabilityUnavailable(
                    "selected generated episode is unavailable or ambiguous"
                )
            return self._read_saved_records_for_context(
                target,
                episode_id=selected_episode_id,
                episode_ref=episode.episode_ref,
            )

        result, _ = self._execute(
            target,
            operation_type="read.saved_records",
            operation_id=self._operation_id(operation_id),
            context=context,
            request_digest=_operation_request_digest(
                {
                    "episode_id": requested_episode_id,
                    "context": _context_request_value(context),
                }
            ),
            callback=read,
        )
        return result

    read_records = read_saved_records
    list_saved_records = read_saved_records

    def validate_session_token(self, session: AuditSession | str, token: str) -> bool:
        """Check a loopback/MCP token without exposing its value."""

        if isinstance(session, AuditSession):
            session_id = session.session_id
        else:
            session_id = session
        if not isinstance(session_id, str) or not session_id.strip():
            return False
        try:
            state = self.authority.snapshot()
            sessions = state.get("sessions", {})
            record = sessions.get(session_id) if isinstance(sessions, Mapping) else None
            return isinstance(record, Mapping) and AuditAuthorityStore.verify_token(record, token)
        except AuditAuthorityError:
            return False

    def update_context(  # noqa: C901
        self,
        session: AuditSession | str,
        context: AuditSelectionContext | Mapping[str, Any],
        *,
        expected_context_revision: int,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[AuditSelectionContext]:
        """CAS-update the session selection before a UI write can proceed."""

        target = self._session(session, token=token)
        selected = (
            context
            if isinstance(context, AuditSelectionContext)
            else AuditSelectionContext.from_mapping(context)
        )
        opid = self._operation_id(operation_id)

        def update() -> AuditSelectionContext:
            with self._lock:
                self._assert_source(target)

                def update_state(working: AuditSession, state: dict[str, Any]) -> dict[str, Any]:
                    self._check_session(working)
                    if any(
                        isinstance(raw, Mapping)
                        and raw.get("session_id") == working.session_id
                        and self._reservation_send_lease(raw) is not None
                        for raw in state["reservations"].values()
                    ):
                        raise AuditContextConflict("provider send is in flight")
                    if expected_context_revision != working.context.context_revision:
                        raise AuditContextConflict(
                            "context revision CAS failed",
                            expected=working.context.context_revision,
                            actual=expected_context_revision,
                        )
                    if selected.context_revision != expected_context_revision + 1:
                        raise AuditContextConflict(
                            "new context must advance exactly one revision",
                            expected=expected_context_revision + 1,
                            actual=selected.context_revision,
                        )
                    self._validate_context_source(working, selected)
                    updated = selected
                    if working.source_revision not in (0, ""):
                        if selected.source_revision in (0, ""):
                            updated = replace(selected, source_revision=working.source_revision)
                        elif selected.source_revision != working.source_revision:
                            raise AuditContextConflict(
                                "context source revision does not match session source",
                                expected=working.source_revision,
                                actual=selected.source_revision,
                            )
                    elif selected.source_revision != working.context.source_revision:
                        raise AuditContextConflict(
                            "context source revision changed without source CAS"
                        )
                    self._set_context_locked(working, updated)
                    return updated.to_dict()

                value, _replayed = self._mutate_session_authority(
                    target,
                    transaction_id=f"authority.context.update:{target.session_id}:{opid}",
                    request_digest=_digest(
                        {
                            "operation_id": opid,
                            "context": selected.to_dict(),
                            "expected_context_revision": expected_context_revision,
                        }
                    ),
                    callback=update_state,
                )
            if not isinstance(value, Mapping):
                raise AuditAuthorityError("context update returned malformed context")
            return AuditSelectionContext.from_mapping(value)

        result, _receipt = self._execute(
            target,
            operation_type="context.update",
            operation_id=opid,
            context=target.context,
            request_digest=_operation_request_digest(
                selected.to_dict(), expected_context_revision=expected_context_revision
            ),
            callback=update,
        )
        return result

    @staticmethod
    def _set_context(
        session: AuditSession, context: AuditSelectionContext
    ) -> AuditSelectionContext:
        object.__setattr__(session, "context", context)
        return context

    @staticmethod
    def _set_context_locked(
        session: AuditSession, context: AuditSelectionContext
    ) -> AuditSelectionContext:
        """Replace context for callers already holding the service lock."""

        object.__setattr__(session, "context", context)
        return context

    def kill_switch(
        self,
        session: AuditSession | str,
        *,
        reason: str = "kill switch activated",
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[None]:
        """Cancel a session and prevent all future operations."""

        target = self._session(session, token=token)
        opid = self._operation_id(operation_id)
        checked_reason = _optional_text(reason, name="cancel_reason", limit=MAX_REASON_CHARS)
        if self._contains_sensitive_text((opid, checked_reason), target.session_token):
            raise AuditPolicyError("cancel operation contains a sensitive value")

        def cancel() -> None:
            with self._lock:

                def cancel_state(working: AuditSession, state: dict[str, Any]) -> None:
                    for event in tuple(self._native_cancel_events.get(working.session_id, ())):
                        event.set()
                    object.__setattr__(working, "cancelled", True)
                    object.__setattr__(working, "cancel_reason", checked_reason)
                    usage = working.usage
                    codex_operations = self._codex_extension(state)["operations"]
                    codex_reservations = {
                        operation.get("reservation_id")
                        for operation in codex_operations.values()
                        if isinstance(operation, Mapping) and operation.get("status") == "inflight"
                    }
                    for reservation_id, raw in tuple(state["reservations"].items()):
                        if (
                            not isinstance(raw, Mapping)
                            or raw.get("session_id") != working.session_id
                        ):
                            continue
                        if reservation_id in codex_reservations:
                            # This Codex turn may already have reached its provider.
                            continue
                        # A send lease is the durable provider-boundary
                        # linearization point.  Keep its reservation and
                        # metering hold until the sender settles, even though
                        # the session is now cancelled.
                        if self._reservation_send_lease(raw) is not None:
                            continue
                        object.__setattr__(
                            working,
                            "usage",
                            SessionUsage(
                                tokens=max(0, usage.tokens - int(raw["tokens"])),
                                compute=max(0.0, usage.compute - float(raw["compute"])),
                                issue_writes=max(0, usage.issue_writes - int(raw["issue_writes"])),
                            ),
                        )
                        usage = working.usage
                        state["reservations"].pop(reservation_id, None)

                _value, _replayed = self._mutate_session_authority(
                    target,
                    transaction_id=f"authority.session.cancel:{target.session_id}:{opid}",
                    request_digest=_digest({"operation_id": opid, "reason": checked_reason}),
                    callback=cancel_state,
                )

        result, _receipt = self._execute(
            target,
            operation_type="session.kill_switch",
            operation_id=opid,
            context=target.context,
            enforce_source=False,
            enforce_active=False,
            request_digest=_operation_request_digest({"reason": checked_reason}),
            callback=cancel,
        )
        return result

    cancel_session = kill_switch
    cancel = kill_switch

    def _action_id(self, operation_id: str, phase: str) -> str:
        return f"audit-operation-{operation_id}-{phase}"

    def _record_action(
        self,
        session: AuditSession,
        action_id: str,
        operation_type: str,
        status: str,
        details: Mapping[str, Any],
    ) -> None:
        payload = _bounded_mapping(
            {
                "schema_version": AUDIT_OPERATION_SCHEMA_VERSION,
                "operation_id": action_id,
                "operation_type": operation_type,
                "phase": "before" if status == "started" else "after",
                "session_id": session.session_id,
                "source_digest": session.source_digest,
                "source_revision": session.source_revision,
                "context_revision": session.context.context_revision,
                "policy_id": session.policy.policy_id,
                "policy_revision": session.policy.policy_revision,
                "usage": session.usage.to_dict(),
                **dict(details),
            },
            name="operation_details",
            limit=MAX_OPERATION_DETAIL_BYTES,
        )
        record = ActionRecord(
            action_id=action_id,
            action_type=operation_type,
            actor_kind=session.actor.kind,
            actor_id=session.actor.actor_id,
            target_id=session.context.episode_id,
            status=status,
            details=payload,
        )
        if self.store.get(action_id, include_deleted=True) is None:
            self.store.save(
                record, operation_id=action_id, expected_revision=0, actor=session.actor.to_dict()
            )

    def _record_action_result(  # noqa: PLR0913
        self,
        session: AuditSession,
        operation_id: str,
        operation_type: str,
        status: str,
        *,
        reason: str = "",
        result: Any = None,
        replayed: bool = False,
        request_digest: str = "",
        context_revision: int | None = None,
    ) -> OperationReceipt:
        after_id = self._action_id(operation_id, "after")
        receipt_context_revision = (
            session.context.context_revision if context_revision is None else context_revision
        )
        result_payload = result.to_dict() if hasattr(result, "to_dict") else result
        try:
            result_digest = _digest(result_payload) if result_payload is not None else ""
        except (AuditContractError, TypeError, ValueError):
            result_digest = ""
        details = {
            "context_revision": receipt_context_revision,
            "operation_id": operation_id,
            "result_status": status,
            "reason": _optional_text(reason, name="operation.reason", limit=MAX_REASON_CHARS),
            "result_digest": result_digest,
        }
        if request_digest:
            details["request_digest"] = request_digest
        self._record_action(
            session,
            after_id,
            operation_type,
            status,
            details,
        )
        receipt = OperationReceipt(
            operation_id=operation_id,
            operation_type=operation_type,
            status=status,
            actor=session.actor,
            session_id=session.session_id,
            context_revision=receipt_context_revision,
            source_revision=session.source_revision,
            source_digest=session.source_digest,
            before_action_id=self._action_id(operation_id, "before"),
            after_action_id=after_id,
            reason=reason,
            replayed=replayed,
            result_digest=result_digest,
        )
        self._finish_authority_operation(
            session,
            operation_id=operation_id,
            status=status,
            reason=reason,
            result_digest=result_digest,
        )
        return receipt

    def _existing_receipt(  # noqa: C901, PLR0912, PLR0915
        self,
        session: AuditSession,
        operation_id: str,
        operation_type: str,
        *,
        request_digest: str = "",
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        expected_source_revision: int | str | None = None,
        enforce_source: bool = True,
        enforce_active: bool = True,
    ) -> OperationReceipt | None:
        after = self.store.get(self._action_id(operation_id, "after"))
        authority_operation = self._authority_operation(operation_id)
        if authority_operation is not None and authority_operation.get("status") == "inflight":
            raise OperationConflictError("operation ID is already in progress")
        if after is None or not isinstance(after.record, ActionRecord):
            if authority_operation is None:
                return None
            if authority_operation.get("session_id") != session.session_id:
                raise OperationConflictError("operation ID belongs to another audit session")
            if authority_operation.get("operation_type") != operation_type:
                raise OperationConflictError("operation ID belongs to another operation type")
            if authority_operation.get("actor") != session.actor.to_dict():
                raise OperationConflictError("operation ID belongs to another actor")
            if authority_operation.get("request_digest") != request_digest:
                raise OperationConflictError("operation ID is already bound to another request")
            operation_status = authority_operation.get("status")
            if operation_status == "inflight":
                raise OperationConflictError("operation ID is already in progress")
            if operation_status != "finished":
                raise OperationConflictError("operation authority state is invalid")
            if enforce_active:
                self._check_session(session)
            self._bind_context(session, context)
            if enforce_source:
                self._assert_source(session, expected_source_revision=expected_source_revision)
            if (
                operation_type != "queue.next"
                and authority_operation.get("context_revision") != session.context.context_revision
            ):
                raise OperationConflictError(
                    "operation replay is bound to a stale context revision"
                )
            if authority_operation.get("source_digest") != session.source_digest:
                raise OperationConflictError("operation replay is bound to a stale source digest")
            if authority_operation.get("source_revision") != session.source_revision:
                raise OperationConflictError("operation replay is bound to a stale source revision")
            status = str(authority_operation.get("result_status", "failed"))
            reason = str(authority_operation.get("reason", ""))
            if operation_type.startswith("read.") and status in {
                "complete",
                "committed",
                "partial",
            }:
                status = "unavailable"
                reason = "replayed read result is not durably retained"
            return OperationReceipt(
                operation_id=operation_id,
                operation_type=operation_type,
                status=status,
                actor=session.actor,
                session_id=session.session_id,
                context_revision=session.context.context_revision,
                source_revision=session.source_revision,
                source_digest=session.source_digest,
                before_action_id=self._action_id(operation_id, "before"),
                after_action_id=self._action_id(operation_id, "after"),
                reason=reason,
                replayed=True,
                result_digest=str(authority_operation.get("result_digest", "")),
                created_at=str(authority_operation.get("created_at", utc_now())),
            )
        if authority_operation is not None:
            if authority_operation.get("status") != "finished":
                raise OperationConflictError("operation authority state is invalid")
            if authority_operation.get("session_id") != session.session_id:
                raise OperationConflictError("operation ID belongs to another audit session")
            if authority_operation.get("operation_type") != operation_type:
                raise OperationConflictError("operation ID belongs to another operation type")
            if authority_operation.get("actor") != session.actor.to_dict():
                raise OperationConflictError("operation ID belongs to another actor")
            if authority_operation.get("request_digest") != request_digest:
                raise OperationConflictError("operation ID is already bound to another request")
        details = after.record.details
        if details.get("operation_id") not in {None, operation_id}:
            raise OperationConflictError("operation ID is already bound to another operation")
        if details.get("session_id") not in {None, session.session_id}:
            raise OperationConflictError("operation ID belongs to another audit session")
        if (
            after.record.actor_kind != session.actor.kind
            or after.record.actor_id != session.actor.actor_id
        ):
            raise OperationConflictError("operation ID belongs to another actor")
        if after.record.action_type != operation_type:
            raise OperationConflictError("operation ID belongs to another operation type")
        stored_request_digest = str(details.get("request_digest", ""))
        if request_digest and stored_request_digest != request_digest:
            raise OperationConflictError("operation ID is already bound to another request")
        if enforce_active:
            self._check_session(session)
        # A finished queue.next receipt is the terminal replay boundary for
        # the queue-side lease. Its request digest, session/actor identity,
        # source binding, and durable sidecar envelope are checked below and
        # bind the original selection. Rebinding the caller's mutable context
        # here would strand that lease when the caller retries with the
        # explicit context it supplied before the selection advanced it.
        # New queue.next operations still bind their requested context in
        # _execute before the adapter callback runs.
        if operation_type != "queue.next":
            self._bind_context(session, context)
        if enforce_source:
            self._assert_source(session, expected_source_revision=expected_source_revision)
        stored_context_revision = details.get("context_revision")
        if (
            operation_type != "queue.next"
            and stored_context_revision is not None
            and stored_context_revision != session.context.context_revision
        ):
            raise OperationConflictError("operation replay is bound to a stale context revision")
        stored_source_digest = str(details.get("source_digest", ""))
        if stored_source_digest and stored_source_digest != session.source_digest:
            raise OperationConflictError("operation replay is bound to a stale source digest")
        stored_source_revision = details.get("source_revision")
        if stored_source_revision is not None and stored_source_revision != session.source_revision:
            raise OperationConflictError("operation replay is bound to a stale source revision")
        status = str(details.get("result_status", after.record.status))
        reason = str(details.get("reason", ""))
        if authority_operation is not None:
            authority_status = str(authority_operation.get("result_status", ""))
            authority_digest = str(authority_operation.get("result_digest", ""))
            if authority_status != status or authority_digest != str(
                details.get("result_digest", "")
            ):
                raise OperationConflictError("operation authority and audit receipt disagree")
        if operation_type.startswith("read.") and status in {"complete", "committed", "partial"}:
            status = "unavailable"
            reason = "replayed read result is not durably retained"
        replay_context_revision = (
            stored_context_revision
            if isinstance(stored_context_revision, int)
            and not isinstance(stored_context_revision, bool)
            else session.context.context_revision
        )
        return OperationReceipt(
            operation_id=operation_id,
            operation_type=operation_type,
            status=status,
            actor=session.actor,
            session_id=session.session_id,
            context_revision=replay_context_revision,
            source_revision=session.source_revision,
            source_digest=session.source_digest,
            before_action_id=self._action_id(operation_id, "before"),
            after_action_id=after.record.action_id,
            reason=reason,
            replayed=True,
            result_digest=str(details.get("result_digest", "")),
            created_at=after.record.created_at,
        )

    def _admit_operation(  # noqa: C901
        self,
        session: AuditSession,
        *,
        operation_id: str,
        operation_type: str,
        context: AuditSelectionContext | Mapping[str, Any] | None,
        request_digest: str,
        expected_source_revision: int | str | None,
        enforce_source: bool,
        enforce_active: bool,
    ) -> OperationReceipt | None:
        """Atomically admit one operation or return its durable replay."""

        with self._lock:
            existing = self._existing_receipt(
                session,
                operation_id,
                operation_type,
                request_digest=request_digest,
                context=context,
                expected_source_revision=expected_source_revision,
                enforce_source=enforce_source,
                enforce_active=enforce_active,
            )
            if existing is not None:
                return existing
            before_id = self._action_id(operation_id, "before")
            if self.store.get(before_id, include_deleted=True) is not None:
                raise OperationConflictError("operation ID is already in progress")
            details = {"reason": "operation_started"}
            if request_digest:
                details["request_digest"] = request_digest
            transaction_id = f"authority.operation.begin:{session.session_id}:{operation_id}"
            begin_digest = _digest(
                {
                    "operation_id": operation_id,
                    "operation_type": operation_type,
                    "session_id": session.session_id,
                    "actor": session.actor.to_dict(),
                    "request_digest": request_digest,
                    "context_revision": session.context.context_revision,
                    "source_revision": session.source_revision,
                    "source_digest": session.source_digest,
                }
            )

            def begin(state: dict[str, Any]) -> dict[str, str]:
                current_session = state["sessions"].get(session.session_id)
                if not isinstance(current_session, Mapping):
                    raise AuthorityOperationConflict("operation session is no longer available")
                authoritative_handle = self._session_from_authority_record(
                    current_session, token=session.session_token
                )
                if enforce_active:
                    self._check_session(authoritative_handle)
                    if (
                        authoritative_handle.context != session.context
                        or authoritative_handle.source_digest != session.source_digest
                        or authoritative_handle.source_revision != session.source_revision
                    ):
                        raise AuditContextConflict("session handle is stale")
                previous = state["operations"].get(operation_id)
                if previous is not None:
                    if not isinstance(previous, Mapping):
                        raise AuthorityOperationConflict(
                            "operation ID is already bound to another request"
                        )
                    if (
                        previous.get("session_id") != session.session_id
                        or previous.get("actor") != session.actor.to_dict()
                    ):
                        # Operation IDs are global authority keys, not scoped
                        # to the begin transaction ID.  A second session must
                        # never inherit an existing owner's callback, even
                        # when its request digest and actor happen to match.
                        raise AuthorityOperationConflict(
                            "operation ID belongs to another audit session"
                        )
                    if (
                        previous.get("request_digest") != request_digest
                        or previous.get("operation_type") != operation_type
                    ):
                        raise AuthorityOperationConflict(
                            "operation ID is already bound to another request"
                        )
                    return {"status": str(previous.get("status", ""))}
                state["operations"][operation_id] = {
                    "operation_id": operation_id,
                    "operation_type": operation_type,
                    "session_id": session.session_id,
                    "actor": session.actor.to_dict(),
                    "request_digest": request_digest,
                    "context_revision": session.context.context_revision,
                    "source_revision": session.source_revision,
                    "source_digest": session.source_digest,
                    "status": "inflight",
                    "result_status": "",
                    "reason": "",
                    "result_digest": "",
                    "created_at": utc_now(),
                }
                return {"status": "inflight"}

            try:
                authority_mutation = self.authority.mutate(transaction_id, begin_digest, begin)
            except AuthorityOperationConflict as exc:
                raise OperationConflictError(str(exc)) from exc
            if authority_mutation.replayed:
                current = self._authority_operation(operation_id)
                if current is None:
                    raise OperationConflictError("operation ID is already in progress")
                if current.get("status") == "inflight":
                    # A replay of the begin transaction does not grant a
                    # second callback.  Its original snapshot is still
                    # inflight, so the other instance owns admission until it
                    # publishes a terminal receipt.
                    raise OperationConflictError("operation ID is already in progress")
                if current.get("status") != "finished":
                    raise OperationConflictError("operation authority state is invalid")
                replay = self._existing_receipt(
                    session,
                    operation_id,
                    operation_type,
                    request_digest=request_digest,
                    context=context,
                    expected_source_revision=expected_source_revision,
                    enforce_source=enforce_source,
                    enforce_active=enforce_active,
                )
                if replay is None:
                    raise OperationConflictError("operation replay receipt is unavailable")
                return replay
            self._record_action(session, before_id, operation_type, "started", details)
            return None

    def _operation_status_result(
        self,
        session: AuditSession,
        *,
        operation_id: str,
        operation_type: str,
        status: str,
        reason: str,
    ) -> tuple[ServiceResult[Any], OperationReceipt]:
        """Build a typed result when admission/replay validation fails."""

        receipt = OperationReceipt(
            operation_id=operation_id,
            operation_type=operation_type,
            status=status,
            actor=session.actor,
            session_id=session.session_id,
            context_revision=session.context.context_revision,
            source_revision=session.source_revision,
            source_digest=session.source_digest,
            before_action_id=self._action_id(operation_id, "before"),
            after_action_id=self._action_id(operation_id, "after"),
            reason=reason,
        )
        return ServiceResult(status, None, reason, receipt, session.context), receipt

    def _execute(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        session: AuditSession,
        *,
        operation_type: str,
        operation_id: str,
        context: AuditSelectionContext | Mapping[str, Any] | None,
        callback: Callable[[], _T],
        preflight: Callable[[], Any] | None = None,
        replay_preflight: Callable[[], Any] | None = None,
        expected_source_revision: int | str | None = None,
        tokens: int = 0,
        compute: float = 0.0,
        durable_charge: bool = False,
        request_digest: str = "",
        enforce_source: bool = True,
        enforce_active: bool = True,
        bind_request: bool = True,
        receipt_context_revision_provider: Callable[[], int | None] | None = None,
    ) -> tuple[ServiceResult[_T], OperationReceipt]:
        """Record before/after state around one authorized operation."""

        opid = self._operation_id(operation_id)
        if self._contains_sensitive_text(opid, session.session_token):
            return self._operation_status_result(
                session,
                operation_id=opid,
                operation_type=operation_type,
                status="conflict",
                reason="operation ID contains a sensitive value",
            )
        if bind_request:
            request_digest = self._bound_request_digest(session, request_digest)
        elif not request_digest:
            request_digest = _digest({"operation_type": operation_type})
        try:
            existing = self._admit_operation(
                session,
                operation_id=opid,
                operation_type=operation_type,
                context=context,
                request_digest=request_digest,
                expected_source_revision=expected_source_revision,
                enforce_source=enforce_source,
                enforce_active=enforce_active,
            )
        except OperationConflictError as exc:
            return self._operation_status_result(
                session,
                operation_id=opid,
                operation_type=operation_type,
                status="conflict",
                reason=str(exc),
            )
        except AuditContextConflict as exc:
            return self._operation_status_result(
                session,
                operation_id=opid,
                operation_type=operation_type,
                status="conflict",
                reason=str(exc),
            )
        except AuditCancelled as exc:
            return self._operation_status_result(
                session,
                operation_id=opid,
                operation_type=operation_type,
                status="cancelled",
                reason=str(exc),
            )
        except CapabilityUnavailable as exc:
            return self._operation_status_result(
                session,
                operation_id=opid,
                operation_type=operation_type,
                status="unavailable",
                reason=str(exc),
            )
        except AuditPolicyError as exc:
            return self._operation_status_result(
                session,
                operation_id=opid,
                operation_type=operation_type,
                status="denied",
                reason=str(exc),
            )
        except AuditAuthorityError as exc:
            return self._operation_status_result(
                session,
                operation_id=opid,
                operation_type=operation_type,
                status="failed",
                reason=str(exc),
            )
        if existing is not None:
            if replay_preflight is not None:
                try:
                    replay_preflight()
                except AuditContextConflict as exc:
                    return self._operation_status_result(
                        session,
                        operation_id=opid,
                        operation_type=operation_type,
                        status="conflict",
                        reason=str(exc),
                    )
                except AuditCancelled as exc:
                    return self._operation_status_result(
                        session,
                        operation_id=opid,
                        operation_type=operation_type,
                        status="cancelled",
                        reason=str(exc),
                    )
                except CapabilityUnavailable as exc:
                    return self._operation_status_result(
                        session,
                        operation_id=opid,
                        operation_type=operation_type,
                        status="unavailable",
                        reason=str(exc),
                    )
                except AuditPolicyError as exc:
                    return self._operation_status_result(
                        session,
                        operation_id=opid,
                        operation_type=operation_type,
                        status="denied",
                        reason=str(exc),
                    )
                except AuditAuthorityError as exc:
                    return self._operation_status_result(
                        session,
                        operation_id=opid,
                        operation_type=operation_type,
                        status="failed",
                        reason=str(exc),
                    )
                except (
                    AuditValidationError,
                    AuditScanError,
                    AuditStoreError,
                    AuditContractError,
                    ValueError,
                    TypeError,
                ) as exc:
                    return self._operation_status_result(
                        session,
                        operation_id=opid,
                        operation_type=operation_type,
                        status="failed",
                        reason=str(exc),
                    )
            return ServiceResult(
                existing.status, None, existing.reason, existing, session.context
            ), existing
        try:
            if enforce_active:
                self._check_session(session)
            selected = self._bind_context(session, context)
            if enforce_source:
                self._assert_source(session, expected_source_revision=expected_source_revision)
            if preflight is not None:
                preflight()
            if durable_charge:
                if tokens or compute:
                    checked_tokens = _nonnegative_int(tokens, name="tokens")
                    checked_compute = _finite(compute, name="compute")
                    if checked_compute < 0:
                        raise AuditValidationError("compute must be non-negative")

                    def charge_state(
                        working: AuditSession, _state: dict[str, Any]
                    ) -> dict[str, Any]:
                        self._check_session(working)
                        self._charge_locked(
                            working,
                            tokens=checked_tokens,
                            compute=checked_compute,
                            issue_writes=0,
                        )
                        return working.usage.to_dict()

                    self._mutate_session_authority(
                        session,
                        transaction_id=f"authority.operation.charge:{session.session_id}:{opid}",
                        request_digest=_digest(
                            {
                                "operation_id": opid,
                                "operation_type": operation_type,
                                "tokens": checked_tokens,
                                "compute": checked_compute,
                            }
                        ),
                        callback=charge_state,
                    )
            else:
                self._charge(session, tokens=tokens, compute=compute)
            value = callback()
            # A callback can return after another valid context CAS. Queue
            # Next supplies the revision committed by its own immutable
            # selection callback; never derive that terminal receipt from the
            # mutable session after the callback returns. Operations without
            # an explicit provider retain the ordinary current-context
            # receipt behavior.
            receipt_context_revision = (
                receipt_context_revision_provider()
                if receipt_context_revision_provider is not None
                else session.context.context_revision
            )
            if receipt_context_revision is None:
                receipt_context_revision = session.context.context_revision
            optional_status = getattr(value, "status", None)
            status = (
                "unavailable"
                if optional_status
                in {
                    "unavailable",
                    "missing",
                    "invalid",
                    "duplicate",
                    "unsupported",
                    "ambiguous",
                    "pending",
                }
                else optional_status
                if optional_status in {"partial", "failed", "cancelled", "denied", "conflict"}
                else "cancelled"
                if operation_type == "session.kill_switch"
                else "committed"
                if operation_type.startswith("write.")
                or operation_type.startswith("context.")
                or operation_type.startswith("budget.")
                or operation_type.startswith("github.")
                else "complete"
            )
            reason = (
                str(getattr(value, "reason", "")) if status not in {"complete", "committed"} else ""
            )
            if operation_type == "session.kill_switch" and not reason:
                reason = session.cancel_reason
            receipt = self._record_action_result(
                session,
                opid,
                operation_type,
                status,
                reason=reason,
                result=value,
                request_digest=request_digest,
                context_revision=receipt_context_revision,
            )
            return ServiceResult(status, value, reason, receipt, selected), receipt
        except AuditNextBlocked as exc:
            # A different queue operation owns the committed sidecar lease.
            # This caller did not cause the ambiguous queue side effect, so
            # its own authority operation must be terminalized while the
            # owner's lease remains untouched.
            receipt = self._record_action_result(
                session,
                opid,
                operation_type,
                "conflict",
                reason=str(exc),
                request_digest=request_digest,
            )
            return ServiceResult("conflict", None, str(exc), receipt, session.context), receipt
        except AuditNextAmbiguous as exc:
            # The queue side effect may already be durable.  Keep the
            # authority operation inflight so retry cannot silently advance a
            # second packet or terminalize an unresolved context race.
            receipt = self._operation_status_result(
                session,
                operation_id=opid,
                operation_type=operation_type,
                status="unavailable",
                reason=str(exc),
            )[1]
            return ServiceResult("unavailable", None, str(exc), receipt, session.context), receipt
        except AuditContextConflict as exc:
            receipt = self._record_action_result(
                session,
                opid,
                operation_type,
                "conflict",
                reason=str(exc),
                request_digest=request_digest,
            )
            return ServiceResult("conflict", None, str(exc), receipt, session.context), receipt
        except AuditCancelled as exc:
            receipt = self._record_action_result(
                session,
                opid,
                operation_type,
                "cancelled",
                reason=str(exc),
                request_digest=request_digest,
            )
            return ServiceResult("cancelled", None, str(exc), receipt, session.context), receipt
        except CapabilityUnavailable as exc:
            receipt = self._record_action_result(
                session,
                opid,
                operation_type,
                "unavailable",
                reason=str(exc),
                request_digest=request_digest,
            )
            return ServiceResult("unavailable", None, str(exc), receipt, session.context), receipt
        except AuditBudgetExceeded as exc:
            receipt = self._record_action_result(
                session,
                opid,
                operation_type,
                "denied",
                reason=str(exc),
                request_digest=request_digest,
            )
            return ServiceResult("denied", None, str(exc), receipt, session.context), receipt
        except AuditPolicyError as exc:
            receipt = self._record_action_result(
                session,
                opid,
                operation_type,
                "denied",
                reason=str(exc),
                request_digest=request_digest,
            )
            return ServiceResult("denied", None, str(exc), receipt, session.context), receipt
        except AuditAuthorityError as exc:
            receipt = self._operation_status_result(
                session,
                operation_id=opid,
                operation_type=operation_type,
                status="failed",
                reason=str(exc),
            )[1]
            return ServiceResult("failed", None, str(exc), receipt, session.context), receipt
        except (
            AuditValidationError,
            AuditScanError,
            AuditStoreError,
            AuditContractError,
            ValueError,
            TypeError,
        ) as exc:
            receipt = self._record_action_result(
                session,
                opid,
                operation_type,
                "failed",
                reason=str(exc),
                request_digest=request_digest,
            )
            return ServiceResult("failed", None, str(exc), receipt, session.context), receipt
        except Exception as exc:  # noqa: BLE001 - retain a durable failure receipt for unknown adapters.
            receipt = self._record_action_result(
                session,
                opid,
                operation_type,
                "failed",
                reason=str(exc),
                request_digest=request_digest,
            )
            return ServiceResult("failed", None, str(exc), receipt, session.context), receipt

    def _scan(
        self,
        session: AuditSession,
        *,
        config: Mapping[str, Any] | None = None,
        detector_ids: Sequence[str] | None = None,
    ) -> CampaignView:
        source, root, digest, revision = self._source_input(session)
        if digest != session.source_digest or revision != session.source_revision:
            raise AuditContextConflict("campaign source changed")
        report = scan_campaign(
            source,
            root=root,
            source_ref=session.source_ref,
            config=config,
            detector_ids=detector_ids,
        )

        def ref_for_row(row: Mapping[str, Any] | None) -> EpisodeRef | None:
            if row is None:
                return None
            matches = tuple(
                ref for ref in report.episode_refs if _episode_ref_matches_row(ref, row)
            )
            return matches[0] if len(matches) == 1 else None

        episodes = tuple(
            EpisodeView(
                item.episode_id,
                item.status,
                item.row,
                item.reason,
                item.source_artifact_id,
                ref_for_row(item.row),
            )
            for item in report.inventory
        )
        source_digest = str(report.provenance.get("source", {}).get("sha256_observed", digest))
        return CampaignView(report, episodes, source_digest, revision)

    def read_campaign(
        self,
        session: AuditSession | str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        detector_ids: Sequence[str] | None = None,
        repository: str | None = None,
        recipe_id: str | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[CampaignView]:
        """Read the campaign inventory and detector signals through BA-01."""

        target = self._session(session, token=token)

        def read() -> CampaignView:
            self._check_policy_scope(target, repository=repository, recipe_id=recipe_id)
            checked_config = (
                None
                if config is None
                else _bounded_mapping(config, name="scan_config", limit=MAX_OPERATION_DETAIL_BYTES)
            )
            checked_detectors = None
            if detector_ids is not None:
                if isinstance(detector_ids, (str, bytes)):
                    raise AuditValidationError("detector_ids must be an array")
                if len(detector_ids) > 128:
                    raise AuditValidationError("detector_ids exceeds the item limit")
                checked_detectors = tuple(
                    _bounded_text(item, name="detector_id", limit=256) for item in detector_ids
                )
            return self._scan(target, config=checked_config, detector_ids=checked_detectors)

        result, _ = self._execute(
            target,
            operation_type="read.campaign",
            operation_id=self._operation_id(operation_id),
            context=context,
            request_digest=_operation_request_digest(
                {
                    "context": _context_request_value(context),
                    "config": config,
                    "detector_ids": detector_ids,
                    "repository": repository,
                    "recipe_id": recipe_id,
                }
            ),
            callback=read,
        )
        return result

    inspect_campaign = read_campaign
    scan_campaign = read_campaign

    def read_episode(
        self,
        session: AuditSession | str,
        episode_id: str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[EpisodeView]:
        """Read one episode row without treating missingness as success."""

        target = self._session(session, token=token)
        episode_id = _bounded_text(episode_id, name="episode_id")
        selected = context if context is not None else target.context
        result, _ = self._execute(
            target,
            operation_type="read.episode",
            operation_id=self._operation_id(operation_id),
            context=selected,
            request_digest=_operation_request_digest(
                {"episode_id": episode_id, "context": _context_request_value(context)}
            ),
            callback=lambda: self._episode_from_campaign(target, episode_id),
        )
        if (
            result.status == "complete"
            and result.value is not None
            and result.value.status != "readable"
        ):
            return ServiceResult(
                "unavailable", result.value, result.value.reason, result.operation, result.context
            )
        return result

    def _episode_from_campaign(self, session: AuditSession, episode_id: str) -> EpisodeView:
        campaign = self._scan(session)
        episode = campaign.episode(episode_id)
        if episode is None:
            return EpisodeView(episode_id, "missing", None, "episode is not present")
        return episode

    def _read_field(
        self,
        session: AuditSession | str,
        episode_id: str,
        field_name: str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[Any]:
        target = self._session(session, token=token)
        episode_id = _bounded_text(episode_id, name="episode_id")
        result, _ = self._execute(
            target,
            operation_type=f"read.{field_name}",
            operation_id=self._operation_id(operation_id),
            context=context,
            request_digest=_operation_request_digest(
                {
                    "episode_id": episode_id,
                    "field": field_name,
                    "context": _context_request_value(context),
                }
            ),
            callback=lambda: self._field_from_episode(target, episode_id, field_name),
        )
        if result.status == "complete" and result.value is None:
            return ServiceResult(
                "unavailable",
                None,
                f"{field_name} is unavailable",
                result.operation,
                result.context,
            )
        return result

    def _field_from_episode(self, session: AuditSession, episode_id: str, field_name: str) -> Any:
        episode = self._episode_from_campaign(session, episode_id)
        if episode.status != "readable" or episode.row is None:
            return CapabilityResult(
                f"episode.{field_name}",
                "unavailable",
                reason=episode.reason or f"episode is {episode.status}",
            )
        value = episode.row.get(field_name)
        if value is None:
            return CapabilityResult(
                f"episode.{field_name}", "unavailable", reason=f"{field_name} is unavailable"
            )
        return value

    def read_metrics(
        self,
        session: AuditSession | str,
        episode_id: str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[Any]:
        """Read source-bound episode metrics."""

        return self._read_field(
            session,
            episode_id,
            "metrics",
            context=context,
            operation_id=operation_id,
            token=token,
        )

    def read_events(
        self,
        session: AuditSession | str,
        episode_id: str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[Any]:
        """Read retained episode events."""

        return self._read_field(
            session,
            episode_id,
            "events",
            context=context,
            operation_id=operation_id,
            token=token,
        )

    def read_geometry(
        self,
        session: AuditSession | str,
        episode_id: str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[Any]:
        """Read retained episode geometry."""

        return self._read_field(
            session,
            episode_id,
            "geometry",
            context=context,
            operation_id=operation_id,
            token=token,
        )

    def read_signals(
        self,
        session: AuditSession | str,
        episode_id: str | None = None,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[tuple[Signal, ...]]:
        """Read BA-01 detector signals without promoting candidates."""

        target = self._session(session, token=token)
        if episode_id is not None:
            episode_id = _bounded_text(episode_id, name="episode_id")

        def read() -> tuple[Signal, ...]:
            campaign = self._scan(target)
            if episode_id is None:
                return campaign.report.signals
            episode = campaign.episode(episode_id)
            if episode is None:
                return ()
            return tuple(
                item for item in campaign.report.signals if item.episode_id == episode.episode_id
            )

        result, _ = self._execute(
            target,
            operation_type="read.signals",
            operation_id=self._operation_id(operation_id),
            context=context,
            request_digest=_operation_request_digest(
                {
                    "episode_id": episode_id,
                    "context": _context_request_value(context),
                }
            ),
            callback=read,
        )
        return result

    def related_cases(
        self,
        session: AuditSession | str,
        episode_id: str,
        *,
        mode: str = "same_scenario_across_planners",
        limit: int = 20,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[tuple[SimilarityResult, ...]]:
        """Return explained BA-03 similarity results, never confirmed findings."""

        target = self._session(session, token=token)
        episode_id = _bounded_text(episode_id, name="episode_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 1_000:
            raise AuditValidationError("related-case limit must be between 0 and 1000")

        def lookup() -> tuple[SimilarityResult, ...]:
            campaign = self._scan(target)
            query = campaign.episode(episode_id)
            if query is None or query.row is None:
                return ()
            candidates = [
                item.row for item in campaign.episodes if item.row is not None and item is not query
            ]
            return tuple(find_similar_cases(query.row, candidates, mode=mode, limit=limit))

        result, _ = self._execute(
            target,
            operation_type="read.related_cases",
            operation_id=self._operation_id(operation_id),
            context=context,
            request_digest=_operation_request_digest(
                {
                    "episode_id": episode_id,
                    "mode": mode,
                    "limit": limit,
                    "context": _context_request_value(context),
                }
            ),
            callback=lookup,
        )
        return result

    find_related_cases = related_cases
    compatible_peers = related_cases

    def next(  # noqa: C901, PLR0915
        self,
        session: AuditSession | str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        expected_context_revision: int | None = None,
        expected_queue_state_revision: int | None = None,
        expected_queue_input_revision: int | None = None,
        force_current: bool = False,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[Any]:
        """Select BA-02's next packet and commit the BA-05 context exactly once.

        The queue and authority stores are intentionally separate. A durable
        inflight ``queue.next`` operation therefore fails closed on retry when
        a process dies between the queue commit and the authority context
        commit; it is never silently selected again.
        """

        # Admit the durable operation before the queue-side preflight refresh.
        # ``_session`` normally refreshes immediately; keeping this first
        # lookup side-effect free makes a later authority race remain inside
        # the queue commit's unresolved boundary instead of losing admission.
        if isinstance(session, AuditSession):
            with self._lock:
                known = self._sessions.get(session.session_id)
            if known is not session:
                raise AuditPolicyError("session is not owned by this service")
            supplied_token = session.session_token if token is None else token
            if not isinstance(supplied_token, str) or not supplied_token.strip():
                raise AuditPolicyError("session token is required")
            target = session
        else:
            target = self._session(session, token=token)
        requested_context = (
            target.context
            if context is None
            else context
            if isinstance(context, AuditSelectionContext)
            else AuditSelectionContext.from_mapping(context)
        )
        opid = self._operation_id(operation_id)
        request_digest = _operation_request_digest(
            {
                "context": _context_request_value(context),
                "expected_context_revision": expected_context_revision,
                "expected_queue_state_revision": expected_queue_state_revision,
                "expected_queue_input_revision": expected_queue_input_revision,
                "force_current": force_current,
            }
        )
        adapter = self.next_adapter or self.queue_next_adapter
        terminal_context_revision: int | None = None

        def before_select(_queue: Any) -> None:
            """Refresh authority while the adapter's durable Next lock is held."""

            current = self._refresh_session(
                target.session_id, token=target.session_token, existing=target
            )
            if expected_context_revision is not None and (
                isinstance(expected_context_revision, bool)
                or not isinstance(expected_context_revision, int)
                or expected_context_revision < 0
            ):
                raise AuditContextConflict("expected context revision is invalid")
            if (
                expected_context_revision is not None
                and current.context.context_revision != expected_context_revision
            ):
                raise AuditContextConflict(
                    "next context revision CAS failed",
                    expected=current.context.context_revision,
                    actual=expected_context_revision,
                )
            if current.context != requested_context:
                raise AuditContextConflict(
                    "next selection context is stale",
                    expected=current.context.context_revision,
                    actual=requested_context.context_revision,
                )
            self._check_session(current)
            self._assert_source(current)

        def commit_selection(selected: Any) -> Any:
            """CAS the service context to the packet selected by BA-02."""

            nonlocal terminal_context_revision

            selection = getattr(selected, "context", None)
            packet = getattr(selected, "packet", None)
            if selection is None or packet is None:
                raise AuditValidationError("BA-02 returned an invalid selection")
            if selection.packet_id != packet.packet_id:
                raise AuditContextConflict("BA-02 packet and selection context diverged")
            updated = requested_context.next_revision(
                execution_id=selection.execution_id,
                episode_id=selection.primary_episode_id,
                interval_id="",
                reference_id=selection.packet_id,
                source_identity=target.source_digest,
                source_revision=target.source_revision,
            )

            def update_state(working: AuditSession, _state: dict[str, Any]) -> dict[str, Any]:
                self._check_session(working)
                if working.context != requested_context:
                    raise AuditContextConflict(
                        "next context revision CAS failed",
                        expected=requested_context.context_revision,
                        actual=working.context.context_revision,
                    )
                self._validate_context_source(working, updated)
                self._set_context_locked(working, updated)
                return updated.to_dict()

            value, _replayed = self._mutate_session_authority(
                target,
                transaction_id=f"authority.queue.next.context:{target.session_id}:{opid}",
                request_digest=_digest(
                    {
                        "operation_id": opid,
                        "packet_id": selection.packet_id,
                        "selection_id": selection.selection_id,
                        "expected_context_revision": requested_context.context_revision,
                    }
                ),
                callback=update_state,
            )
            if not isinstance(value, Mapping):
                raise AuditAuthorityError("queue.next context update returned malformed context")
            # Keep the receipt binding in this per-operation immutable result
            # slot. A concurrent update_context may run before the adapter
            # callback returns, but it must not relabel this packet as the
            # later context revision.
            terminal_context_revision = updated.context_revision
            return selected

        def select() -> Any:
            if adapter is None:
                return CapabilityResult(
                    "ba-02.queue.next", "unavailable", reason="ba-02 Next adapter is not merged"
                )
            request: dict[str, Any] = {
                "context": requested_context,
                "before_select": before_select,
                "after_select": commit_selection,
                "expected_state_revision": expected_queue_state_revision,
                "expected_input_revision": expected_queue_input_revision,
                "force_current": force_current,
                "operation_id": opid,
                "session_id": target.session_id,
            }
            return adapter.transact(**request)

        result, _receipt = self._execute(
            target,
            operation_type="queue.next",
            operation_id=opid,
            context=requested_context,
            request_digest=request_digest,
            callback=select,
            bind_request=False,
            receipt_context_revision_provider=lambda: terminal_context_revision,
        )
        if (
            result.operation is not None
            and result.status in {"complete", "committed"}
            and adapter is not None
        ):
            # The queue sidecar remains active until this exact readback.  On
            # the first call this happens after _execute has durably written
            # the outer terminal receipt; on an exact operation replay it
            # also closes the window left by a crash after that receipt.
            exact = None
            replay_reader = getattr(adapter, "replay", None)
            if not callable(replay_reader):
                return ServiceResult(
                    "unavailable",
                    None,
                    "BA-02 queue Next adapter cannot finalize its durable lease",
                    result.operation,
                    target.context,
                )
            try:
                exact = replay_reader(
                    operation_id=opid,
                    session_id=target.session_id,
                    context=target.context,
                    expected_result_digest=result.operation.result_digest,
                )
            except (AuditContextConflict, CapabilityUnavailable, AuditServiceError, ValueError):
                exact = None
            if exact is not None:
                if target.context.context_revision != result.operation.context_revision:
                    return ServiceResult(
                        "unavailable",
                        None,
                        "BA-02 queue Next result is historical; current context changed",
                        result.operation,
                        target.context,
                    )
                return ServiceResult(
                    result.status,
                    exact,
                    result.reason,
                    result.operation,
                    target.context,
                )
            return ServiceResult(
                "unavailable",
                None,
                "BA-02 queue Next result cannot be reconciled from durable queue state",
                result.operation,
                target.context,
            )
        if result.value is not None and result.status in {"complete", "committed"}:
            return replace(result, context=target.context)
        return result

    next_packet = next  # noqa: A003
    select_next = next  # noqa: A003
    queue_next = next  # noqa: A003

    def read_queue(
        self,
        session: AuditSession | str,
        *,
        limit: int = 20,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[Any]:
        """Read BA-02 queue data or return explicit unavailable capability."""

        target = self._session(session, token=token)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 1_000:
            raise AuditValidationError("queue limit must be between 0 and 1000")
        callback = (
            (
                lambda: self.queue_adapter.read(
                    context=self._bind_context(target, context), limit=limit
                )
            )
            if self.queue_adapter is not None
            else (
                lambda: CapabilityResult(
                    "ba-02.queue", "unavailable", reason="ba-02 adapter is not merged"
                )
            )
        )
        result, _ = self._execute(
            target,
            operation_type="read.queue",
            operation_id=self._operation_id(operation_id),
            context=context,
            request_digest=_operation_request_digest(
                {"limit": limit, "context": _context_request_value(context)}
            ),
            callback=callback,
        )
        if self.queue_adapter is None and result.status == "complete":
            return ServiceResult(
                "unavailable", result.value, result.value.reason, result.operation, result.context
            )
        return result

    queue = read_queue

    def read_coverage(
        self,
        session: AuditSession | str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[Any]:
        """Read BA-04 coverage data or return explicit unavailable capability."""

        target = self._session(session, token=token)
        callback = (
            (lambda: self.coverage_adapter.read(context=self._bind_context(target, context)))
            if self.coverage_adapter is not None
            else (
                lambda: CapabilityResult(
                    "ba-04.coverage", "unavailable", reason="ba-04 adapter is not merged"
                )
            )
        )
        result, _ = self._execute(
            target,
            operation_type="read.coverage",
            operation_id=self._operation_id(operation_id),
            context=context,
            request_digest=_operation_request_digest({"context": _context_request_value(context)}),
            callback=callback,
        )
        if self.coverage_adapter is None and result.status == "complete":
            return ServiceResult(
                "unavailable", result.value, result.value.reason, result.operation, result.context
            )
        return result

    coverage = read_coverage

    def _native_binding_for_session(self, session: AuditSession) -> NativeDiagnosticBinding:
        """Select one trusted binding matching the immutable session source."""

        bindings = self.native_diagnostic_config.bindings
        if not bindings:
            raise CapabilityUnavailable("native diagnostic adapter is not configured")
        if session.source_ref is None:
            raise AuditContextConflict("native diagnostic requires an immutable session source_ref")
        source_matches = [
            binding
            for binding in bindings
            if len(binding.request.sources) == 1
            and self._source_ref_matches(session, binding.request.sources[0])
        ]
        if not source_matches:
            raise AuditContextConflict("native diagnostic source does not match session source_ref")
        if len(source_matches) > 1:
            raise AuditPolicyError("native diagnostic source binding is ambiguous")
        binding = source_matches[0]
        if not session.policy.allows_recipe(binding.recipe.recipe_id):
            raise AuditPolicyError("native diagnostic recipe is not allowed by the session policy")
        root = Path(binding.admission.source_root).resolve(strict=False)
        if not session.policy.allows_path(root):
            raise AuditPolicyError("native diagnostic source root is not allowed")
        return binding

    @staticmethod
    def _native_intervention(
        intervention: Mapping[str, Any] | None,
        *,
        intervention_id: str | None,
        robot_goal: Sequence[float] | None,
        activation_epsilon_m: float,
    ) -> dict[str, Any]:
        """Validate the only caller-owned native intervention fields."""

        if intervention is not None:
            if not isinstance(intervention, Mapping):
                raise AuditValidationError("native diagnostic intervention must be a mapping")
            allowed = {"intervention_id", "factor", "robot_goal", "activation_epsilon_m"}
            unknown = set(intervention) - allowed
            if unknown:
                raise AuditValidationError(
                    "native diagnostic intervention contains unknown fields: "
                    + ", ".join(sorted(unknown))
                )
            if intervention_id is not None or robot_goal is not None:
                raise AuditValidationError(
                    "native diagnostic intervention cannot mix mapping and separate fields"
                )
            values = dict(intervention)
            intervention_id = values.get("intervention_id")
            robot_goal = values.get("robot_goal")
            activation_epsilon_m = values.get("activation_epsilon_m", activation_epsilon_m)
            if values.get("factor", "robot_goal") != "robot_goal":
                raise AuditValidationError("only factor='robot_goal' is supported")
        if intervention_id is None:
            raise AuditValidationError("native diagnostic intervention_id is required")
        if robot_goal is None or isinstance(robot_goal, (str, bytes)):
            raise AuditValidationError("native diagnostic robot_goal is required")
        if len(robot_goal) != 2:
            raise AuditValidationError(
                "native diagnostic robot_goal must contain exactly two values"
            )
        checked_goal = tuple(
            _finite(value, name=f"native diagnostic robot_goal[{index}]")
            for index, value in enumerate(robot_goal)
        )
        checked_id = _bounded_text(intervention_id, name="native diagnostic intervention_id")
        epsilon = _finite(activation_epsilon_m, name="native diagnostic activation_epsilon_m")
        if epsilon < 0.0 or epsilon > 1.0:
            raise AuditValidationError("native diagnostic activation_epsilon_m must be within 0..1")
        return {
            "intervention_id": checked_id,
            "factor": "robot_goal",
            "robot_goal": checked_goal,
            "activation_epsilon_m": epsilon,
        }

    def _native_request(
        self,
        binding: NativeDiagnosticBinding,
        intervention: Mapping[str, Any],
        *,
        deadline_s: float,
    ) -> NativeDiagnosticRequest:
        """Construct a native request from trusted config and bounded input."""

        from robot_sf.analysis_workbench import audit_native_diagnostic as native  # noqa: PLC0415

        # The binding has already crossed the review-contract parser at
        # launcher configuration time.  Preserve those typed contracts here;
        # ``asdict`` is intentionally not round-tripped because the review
        # contract schemas carry their version markers outside the dataclasses.
        return native.NativeDiagnosticRequest(
            request_id=binding.request.request_id,
            admission=binding.admission,
            request=binding.request,
            recipe=binding.recipe,
            intervention=dict(intervention),
            timeout_s=deadline_s,
        )

    def _preflight_native_request(
        self,
        request: NativeDiagnosticRequest,
        selected: AuditSelectionContext,
    ) -> None:
        """Resolve source bytes and historical eligibility before child work."""

        from robot_sf.analysis_workbench import audit_native_diagnostic as native  # noqa: PLC0415

        try:
            native._validate_request(request)
            _root, root_fd, _receipt, _receipt_digest, resolution = native._open_and_resolve_source(
                request
            )
        except (
            native.NativeDiagnosticError,
            OSError,
            ReviewContractsValidationError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise CapabilityUnavailable(f"native diagnostic source is unavailable: {exc}") from exc
        try:
            try:
                source_document = native._source_document(
                    resolution.source_bytes or b"", resolution.receipt.source
                )
                original = source_document.original_record
                if selected.episode_id and original.get("episode_id") != selected.episode_id:
                    raise AuditContextConflict(
                        "native diagnostic source episode does not match selected context",
                        expected=selected.episode_id,
                        actual=original.get("episode_id"),
                    )
                execution_errors = native._record_execution_errors(
                    original,
                    source_document.runner_input,
                    source_commit=resolution.receipt.source.source_commit,
                    source_identity=source_document.identity,
                    expected_role=native.ROLE_HISTORICAL_ORIGINAL,
                    source_id=source_document.source_id,
                )
                if execution_errors:
                    reason = "; ".join(execution_errors)
                    if any("trace" in error or "metrics" in error for error in execution_errors):
                        raise CapabilityUnavailable(
                            "native diagnostic historical source is not eligible: " + reason
                        )
                    raise AuditValidationError(
                        "native diagnostic historical source is not eligible: " + reason
                    )
                original_goal = native._vector2(
                    source_document.runner_input["robot_goal"], name="runner_input.robot_goal"
                )
                intervention_goal = request.intervention["robot_goal"]
                if all(
                    math.isclose(
                        original_goal[index], intervention_goal[index], rel_tol=0.0, abs_tol=0.0
                    )
                    for index in range(2)
                ):
                    raise AuditValidationError(
                        "intervention_not_effective: intervention goal equals original goal"
                    )
            except AuditValidationError:
                raise
            except (
                native.NativeDiagnosticError,
                ReviewContractsValidationError,
                TypeError,
                ValueError,
            ) as exc:
                raise AuditValidationError(
                    f"native diagnostic source is ineligible: {exc}"
                ) from exc
        finally:
            try:
                os.close(root_fd)
            except OSError:
                pass

    def run_native_diagnostic(  # noqa: PLR0913
        self,
        session: AuditSession | str,
        *,
        intervention: Mapping[str, Any] | None = None,
        intervention_id: str | None = None,
        robot_goal: Sequence[float] | None = None,
        activation_epsilon_m: float = 1e-9,
        deadline_s: float = 30.0,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[NativeDiagnosticResult]:
        """Run one trusted, selected-context-bound native diagnostic pair.

        Source roots, receipts, recipes, runner inputs, planner identities, and
        checkpoints are launcher-owned configuration.  Callers can submit only
        one bounded goal intervention and a deadline under the adapter ceiling.
        """

        target = self._session(session, token=token)
        opid = self._operation_id(operation_id)
        checked_intervention = self._native_intervention(
            intervention,
            intervention_id=intervention_id,
            robot_goal=robot_goal,
            activation_epsilon_m=activation_epsilon_m,
        )
        checked_deadline = _finite(deadline_s, name="native diagnostic deadline_s")
        if (
            checked_deadline <= 0.0
            or checked_deadline > self.native_diagnostic_config.max_timeout_s
        ):
            raise AuditValidationError(
                "native diagnostic deadline_s must be within adapter max timeout "
                f"(0, {self.native_diagnostic_config.max_timeout_s:g}]"
            )
        binding_digest = _digest(self.native_diagnostic_config.to_dict())
        cancel_event = threading.Event()
        request_holder: dict[str, NativeDiagnosticRequest] = {}

        def preflight() -> None:
            selected = self._bind_context(target, context)
            if not selected.episode_id:
                raise AuditContextConflict("native diagnostic requires a selected episode")
            binding = self._native_binding_for_session(target)
            request_holder["request"] = self._native_request(
                binding,
                checked_intervention,
                deadline_s=checked_deadline,
            )
            self._preflight_native_request(request_holder["request"], selected)

        def execute() -> NativeDiagnosticResult:
            request = request_holder.get("request")
            if request is None:
                raise CapabilityUnavailable("native diagnostic preflight did not produce a request")
            from robot_sf.analysis_workbench import audit_native_diagnostic as native  # noqa: PLC0415

            result = native.run_native_diagnostic(request, cancel_event=cancel_event)
            if (
                not isinstance(result, native.NativeDiagnosticResult)
                or result.evidence_boundary != "diagnostic_only"
                or result.scientific_claim_allowed is not False
            ):
                raise AuditPolicyError("native diagnostic result did not preserve diagnostic_only")
            return result

        with self._lock:
            self._native_cancel_events.setdefault(target.session_id, set()).add(cancel_event)
        try:
            result, _ = self._execute(
                target,
                operation_type=NATIVE_DIAGNOSTIC_OPERATION_TYPE,
                operation_id=opid,
                context=context,
                preflight=preflight,
                replay_preflight=preflight,
                compute=self.native_diagnostic_config.compute_cost,
                durable_charge=True,
                request_digest=_operation_request_digest(
                    {
                        "context": _context_request_value(context),
                        "intervention": checked_intervention,
                        "deadline_s": checked_deadline,
                        "native_config_digest": binding_digest,
                    }
                ),
                callback=execute,
                enforce_source=False,
            )
            return result
        finally:
            with self._lock:
                events = self._native_cancel_events.get(target.session_id)
                if events is not None:
                    events.discard(cancel_event)
                    if not events:
                        self._native_cancel_events.pop(target.session_id, None)

    native_diagnostic = run_native_diagnostic
    run_native = run_native_diagnostic

    def materialize_selected(  # noqa: C901 - policy, capability, charge, and adapter ordering
        self,
        session: AuditSession | str,
        *,
        output_root: str | Path,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        output_directory: str | None = None,
        render_config: Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[Any]:
        """Materialize only the selected, source-bound episode as diagnostic media.

        The admitted operation is durable before rendering. A crash during the
        renderer leaves the operation inflight and cannot silently re-run it.
        The fixed compute unit is a conservative admission charge, not measured
        renderer work.
        """

        from robot_sf.analysis_workbench.audit_materialize import (  # noqa: PLC0415
            MaterializationValidationError,
            admit_output_root,
            admit_source_root,
            materialize_episode,
        )

        target = self._session(session, token=token)
        output = Path(output_root).resolve(strict=False)
        opid = self._operation_id(operation_id)

        def materialize() -> Any:  # noqa: C901 - ordered capability and adapter admission
            selected = self._bind_context(target, context)
            if not selected.episode_id:
                raise AuditContextConflict("materialization requires a selected episode")
            if not target.policy.allows_path(output):
                raise AuditPolicyError("materialization output root is not allowed")
            source_root = self.source_root
            if source_root is None:
                raise CapabilityUnavailable("materialization source root is not configured")
            if not target.policy.allows_path(source_root):
                raise AuditPolicyError("materialization source root is not allowed")
            try:
                source_capability = admit_source_root(
                    source_root,
                    allowed_roots=target.policy.allowed_roots,
                )
            except (MaterializationValidationError, OSError, RuntimeError, ValueError) as error:
                raise AuditPolicyError(
                    "materialization source root could not be safely admitted"
                ) from error
            try:
                episode = self._episode_from_campaign(target, selected.episode_id)
                if episode.status != "readable" or episode.row is None:
                    raise CapabilityUnavailable("selected episode has no readable source row")
                try:
                    output_capability = admit_output_root(
                        output,
                        allowed_roots=target.policy.allowed_roots,
                    )
                except (MaterializationValidationError, OSError, RuntimeError, ValueError) as error:
                    raise AuditPolicyError(
                        "materialization output root could not be safely admitted"
                    ) from error
                try:
                    charge = self.consume_budget(
                        target, compute=1.0, operation_id=f"{opid}-compute"
                    )
                    if charge.status == "cancelled":
                        raise AuditCancelled(charge.reason or "audit session was cancelled")
                    if charge.status == "conflict":
                        raise AuditContextConflict(
                            charge.reason or "materialization budget context changed"
                        )
                    if not charge.ok:
                        raise AuditBudgetExceeded(
                            charge.reason or "materialization compute charge failed"
                        )
                    self._check_session(self._session(target))
                    materialized = materialize_episode(
                        episode,
                        source_root=source_capability,
                        output_root=output_capability,
                        output_directory=output_directory,
                        render_config=render_config,
                        _trusted_scan_identity_defaults=episode.row.get(
                            "_audit_scan_identity_defaults"
                        ),
                    )
                    self._check_session(self._session(target))
                    return materialized
                finally:
                    output_capability.close()
            finally:
                source_capability.close()

        result, _ = self._execute(
            target,
            operation_type="materialize.selected",
            operation_id=opid,
            context=context,
            request_digest=_operation_request_digest(
                {
                    "context": _context_request_value(context),
                    "output_root": str(output),
                    "output_directory": output_directory,
                    "render_config": render_config,
                }
            ),
            callback=materialize,
        )
        return result

    def _validate_github_finding_scope(
        self,
        session: AuditSession,
        finding: Finding,
        *,
        evidence: Any,
    ) -> Any:
        """Validate canonical finding/evidence provenance before GitHub work.

        The GitHub renderer sanitizes public text, but it cannot decide whether
        a caller supplied a stale campaign or source identity.  Keep that
        admission on the authenticated service boundary and let the existing
        synchronizer own marker, outbox, claim, and provider reconciliation.
        """

        session_revision = session.source_revision
        finding_revision = finding.source_revision
        session_revision_known = session_revision not in (0, "")
        finding_revision_known = finding_revision not in (0, "", None)
        if session_revision_known != finding_revision_known or (
            session_revision_known and str(session_revision) != str(finding_revision)
        ):
            raise AuditContextConflict(
                "finding source revision does not match the authenticated session source",
                expected=session_revision,
                actual=finding_revision,
            )
        if evidence is None:
            return None
        from robot_sf.analysis_workbench.audit_github import (  # noqa: PLC0415
            FindingEvidence,
        )

        checked = FindingEvidence.from_value(evidence)
        if checked.campaign_id and checked.campaign_id != session.context.campaign_id:
            raise AuditContextConflict(
                "GitHub evidence campaign does not match the authenticated context",
                expected=session.context.campaign_id,
                actual=checked.campaign_id,
            )
        if checked.source_identity and checked.source_identity not in self._source_identities(
            session
        ):
            raise AuditContextConflict(
                "GitHub evidence source identity does not match the authenticated source"
            )
        if checked.source_revision:
            if not session_revision_known or str(checked.source_revision) != str(session_revision):
                raise AuditContextConflict(
                    "GitHub evidence source revision does not match the authenticated source",
                    expected=session_revision,
                    actual=checked.source_revision,
                )
        if checked.source_digest and checked.source_digest != session.source_digest:
            raise AuditContextConflict(
                "GitHub evidence source digest does not match the authenticated source",
                expected=session.source_digest,
                actual=checked.source_digest,
            )
        return checked

    def _github_sync(
        self,
        session: AuditSession,
        provider: Any,
    ) -> Any:
        """Build the existing GitHub synchronizer on the canonical store."""

        from robot_sf.analysis_workbench.audit_github import (  # noqa: PLC0415
            GitHubOutbox,
            GitHubSync,
        )

        if self._github_outbox is None:
            self._github_outbox = GitHubOutbox(self.store)
        private_roots = (*self.github_private_roots, *session.policy.allowed_roots)
        return GitHubSync(
            provider,
            self._github_outbox,
            finding_store=self.finding_store,
            private_roots=private_roots,
            allowed_media_hosts=self.github_allowed_media_hosts,
        )

    def _acquire_github_send_lease(
        self,
        session: AuditSession,
        *,
        reservation_id: str,
        reservation_operation_id: str,
        context: AuditSelectionContext,
        source_digest: str,
        source_revision: int | str,
        operation_id: str,
    ) -> ServiceResult[dict[str, Any]]:
        """Linearize an authorized provider send against durable session state.

        A local mutex cannot protect a second service process (or a separate
        kill-switch caller).  The lease is therefore written into the same
        authority transition as the reservation owner check.  A cancellation
        that wins this transition removes the reservation and no provider call
        is admitted; a cancellation after it leaves the paid reservation for
        settlement instead of converting an already-started send into a
        misleading denial.
        """

        target = self._session(session)

        def acquire() -> dict[str, Any]:
            # Re-read the external source immediately before the durable lease
            # transition.  The provider boundary still has no source CAS, so
            # the post-send guard below remains mandatory.
            self._assert_source(target, expected_source_revision=source_revision)

            def acquire_state(working: AuditSession, state: dict[str, Any]) -> dict[str, Any]:
                self._check_session(working)
                if working.context != context:
                    raise AuditContextConflict(
                        "selection context changed before provider send",
                        expected=context.context_revision,
                        actual=working.context.context_revision,
                    )
                if (
                    working.source_digest != source_digest
                    or working.source_revision != source_revision
                ):
                    raise AuditContextConflict("source binding changed before provider send")
                raw = state["reservations"].get(reservation_id)
                if not isinstance(raw, Mapping) or raw.get("session_id") != working.session_id:
                    raise AuditPolicyError("GitHub issue-write reservation is no longer owned")
                if raw.get("operation_id") != reservation_operation_id:
                    raise AuditPolicyError("GitHub issue-write reservation owner changed")
                if self._reservation_send_lease(raw) is not None:
                    raise AuditContextConflict("GitHub provider send is already in flight")
                lease = {
                    "operation_id": operation_id,
                    "context_revision": context.context_revision,
                    "source_revision": source_revision,
                    "source_digest": source_digest,
                }
                leased = dict(raw)
                leased["send_lease"] = lease
                state["reservations"][reservation_id] = leased
                return lease

            value, _replayed = self._mutate_session_authority(
                target,
                transaction_id=f"authority.budget.send:{target.session_id}:{operation_id}",
                request_digest=_digest(
                    {
                        "operation_id": operation_id,
                        "reservation_id": reservation_id,
                        "reservation_operation_id": reservation_operation_id,
                        "context_revision": context.context_revision,
                        "source_digest": source_digest,
                        "source_revision": source_revision,
                    }
                ),
                callback=acquire_state,
            )
            if not isinstance(value, Mapping):
                raise AuditAuthorityError("GitHub provider send lease is malformed")
            return dict(value)

        result, _receipt = self._execute(
            target,
            operation_type="budget.send",
            operation_id=operation_id,
            context=context,
            expected_source_revision=source_revision,
            request_digest=_operation_request_digest(
                {
                    "reservation_id": reservation_id,
                    "reservation_operation_id": reservation_operation_id,
                    "context": context.to_dict(),
                    "source_digest": source_digest,
                    "source_revision": source_revision,
                }
            ),
            callback=acquire,
        )
        return result

    def _github_post_send_conflict(
        self,
        session: AuditSession,
        *,
        context: AuditSelectionContext,
        source_digest: str,
        source_revision: int | str,
    ) -> str:
        """Return a reason when source/context changed across a provider send."""

        try:
            state = self.authority.snapshot()
            sessions = state.get("sessions", {})
            raw = sessions.get(session.session_id) if isinstance(sessions, Mapping) else None
            if not isinstance(raw, Mapping):
                return "session authority disappeared after provider send"
            current_context = AuditSelectionContext.from_mapping(raw["context"])
            if current_context != context:
                return "selection context changed after provider send"
            if (
                raw.get("source_digest") != source_digest
                or raw.get("source_revision") != source_revision
            ):
                return "source binding changed after provider send"
            _source, _root, current_digest, current_revision = self._source_input(session)
            if current_digest != source_digest or current_revision != source_revision:
                return "campaign source changed after provider send"
        except AuditServiceError as exc:
            return f"source/context could not be revalidated after provider send: {exc}"
        return ""

    @staticmethod
    def _github_remote_write_count(result: Any, *, provider_started: bool) -> int:
        """Meter the explicit provider-boundary outcome, not its terminal status."""

        if not provider_started:
            return 0
        if result is None:
            # An exception after the provider boundary is an ambiguous remote
            # outcome and must consume the reserved issue-write unit.
            return 1
        outcome = getattr(result, "remote_write", None)
        if outcome in {"applied", "ambiguous"}:
            return 1
        if outcome == "none":
            return 0
        status = getattr(result, "status", "")
        # Keep a conservative fallback for provider adapters returning an
        # older result shape.  A reconciled outcome means readback found the
        # prior mutation and therefore consumes no new write unit.
        return 1 if status in {"created", "updated", "ambiguous"} else 0

    def sync_finding(  # noqa: C901, PLR0913, PLR0915
        self,
        session: AuditSession | str,
        *,
        finding_id: str,
        repository: str,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        evidence: Any = None,
        expected_finding_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        retry_ambiguous: bool = False,
        worker_id: str = "",
        provider: Any | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[Any]:
        """Synchronize one canonical finding under session and write policy.

        The finding is loaded by ID from the canonical ``FindingStore`` after
        durable service admission.  The optional provider is an injected,
        provider-neutral test/live seam; no credentials or live client are
        constructed by this service.  ``retry_ambiguous=True`` creates a
        separate durable service receipt while reusing the same GitHub outbox
        operation ID, so marker reconciliation remains idempotent.
        """

        target = self._session(session, token=token)
        requested_operation_id = self._operation_id(operation_id)
        service_operation_id = requested_operation_id
        if retry_ambiguous:
            service_operation_id = _bounded_text(
                f"{requested_operation_id}:retry", name="operation_id"
            )
        checked_source_revision = (
            target.source_revision if expected_source_revision is None else expected_source_revision
        )
        request_digest = _operation_request_digest(
            {
                "finding_id": finding_id,
                "repository": repository,
                "evidence": evidence,
                "expected_finding_revision": expected_finding_revision,
                "expected_source_revision": checked_source_revision,
                "retry_ambiguous": retry_ambiguous,
                "worker_id": worker_id,
                "sync_operation_id": requested_operation_id,
            }
        )

        def sync() -> Any:  # noqa: C901, PLR0912, PLR0915
            """Admit policy, reserve one possible remote write, then sync."""

            checked_finding_id = _bounded_text(finding_id, name="finding_id")
            checked_repository = _bounded_text(repository, name="repository")
            self._check_policy_scope(target, repository=checked_repository)
            if not isinstance(retry_ambiguous, bool):
                raise AuditValidationError("retry_ambiguous must be boolean")
            if not isinstance(worker_id, str):
                raise AuditValidationError("worker_id must be text")
            if expected_finding_revision is None:
                raise AuditValidationError("expected_finding_revision is required")
            checked_finding_revision = _nonnegative_int(
                expected_finding_revision, name="expected_finding_revision"
            )
            stored = self.store.get(checked_finding_id)
            if stored is None or not isinstance(stored.record, Finding):
                raise CapabilityUnavailable("canonical finding is unavailable")
            if stored.revision != checked_finding_revision:
                raise AuditContextConflict(
                    "finding revision CAS failed",
                    expected=checked_finding_revision,
                    actual=stored.revision,
                )
            finding = stored.record
            checked_evidence = self._validate_github_finding_scope(
                target, finding, evidence=evidence
            )
            active_provider = provider if provider is not None else self.github_provider
            if active_provider is None:
                raise CapabilityUnavailable(
                    "GitHub provider is unavailable; live issue sync is disabled"
                )
            syncer = self._github_sync(target, active_provider)
            checked_worker_id = _optional_text(worker_id, name="worker_id", limit=512)
            operation_context = target.context
            operation_source_digest = target.source_digest
            operation_source_revision = target.source_revision
            budget_operation_id = f"{service_operation_id}:issue-budget"
            reservation = self.reserve_budget(
                target,
                issue_writes=1,
                operation_id=budget_operation_id,
            )
            if reservation.status != "committed" or not isinstance(reservation.value, Mapping):
                raise AuditBudgetExceeded(
                    reservation.reason or "GitHub issue-write budget is unavailable"
                )
            capability = reservation.value
            reservation_id = capability.get("reservation_id")
            if not isinstance(reservation_id, str) or not reservation_id:
                raise AuditAuthorityError("GitHub issue-write reservation is malformed")
            reserved = capability.get("reserved")
            if not isinstance(reserved, Mapping):
                raise AuditAuthorityError("GitHub issue-write reservation amounts are malformed")
            result: Any = None
            lease_acquired = False
            provider_started = False
            provider_result: Any = None
            primary_error: BaseException | None = None
            try:
                with self._lock:
                    lease = self._acquire_github_send_lease(
                        target,
                        reservation_id=reservation_id,
                        reservation_operation_id=budget_operation_id,
                        context=operation_context,
                        source_digest=operation_source_digest,
                        source_revision=operation_source_revision,
                        operation_id=f"{budget_operation_id}:send",
                    )
                    if lease.status != "committed":
                        reason = lease.reason or "GitHub provider send was not authorized"
                        if lease.status == "cancelled":
                            raise AuditCancelled(reason)
                        if lease.status == "conflict":
                            raise AuditContextConflict(reason)
                        if lease.status == "denied":
                            raise AuditBudgetExceeded(reason)
                        if lease.status == "unavailable":
                            raise CapabilityUnavailable(reason)
                        raise AuditAuthorityError(reason)
                    lease_acquired = True
                    # Narrow the source check to the provider boundary once
                    # more.  The durable lease protects session/context races;
                    # source bytes have no comparable cross-process CAS.
                    self._assert_source(target, expected_source_revision=operation_source_revision)
                    self._bind_context(target, operation_context)
                    provider_started = True
                    try:
                        provider_result = syncer.sync(
                            checked_repository,
                            finding,
                            operation_id=requested_operation_id,
                            evidence=checked_evidence,
                            expected_finding_revision=checked_finding_revision,
                            retry_ambiguous=retry_ambiguous,
                            worker_id=checked_worker_id,
                        )
                    except BaseException as exc:
                        post_send_reason = self._github_post_send_conflict(
                            target,
                            context=operation_context,
                            source_digest=operation_source_digest,
                            source_revision=operation_source_revision,
                        )
                        if post_send_reason:
                            raise AuditNextAmbiguous(
                                f"provider outcome is ambiguous: {post_send_reason}"
                            ) from exc
                        raise
                    result = provider_result
                    post_send_reason = self._github_post_send_conflict(
                        target,
                        context=operation_context,
                        source_digest=operation_source_digest,
                        source_revision=operation_source_revision,
                    )
                    if post_send_reason:
                        original_status = getattr(provider_result, "status", "")
                        post_status = "ambiguous" if original_status == "ambiguous" else "conflict"
                        result = replace(
                            provider_result,
                            status=post_status,
                            reason=(
                                f"{getattr(provider_result, 'reason', '')}; {post_send_reason}"
                            ).strip("; "),
                        )
                    return result
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                remote_write = self._github_remote_write_count(
                    provider_result,
                    provider_started=provider_started,
                )
                # A reservation can be revoked by a kill switch that wins
                # before lease acquisition. Preserve the primary cancelled /
                # conflict result instead of turning that safe no-send path
                # into a misleading settlement denial.
                settle_target = self._session(target)
                settled = self.settle_budget(
                    settle_target,
                    reserved_tokens=int(reserved.get("tokens", 0)),
                    reserved_compute=float(reserved.get("compute", 0.0)),
                    reserved_issue_writes=int(reserved.get("issue_writes", 1)),
                    actual_tokens=0,
                    actual_compute=0.0,
                    actual_issue_writes=remote_write,
                    reservation_id=reservation_id,
                    reservation_operation_id=budget_operation_id,
                    operation_id=f"{budget_operation_id}:settle",
                )
                if settled.status != "committed":
                    if lease_acquired or primary_error is None:
                        raise AuditBudgetExceeded(
                            settled.reason or "GitHub issue-write budget settlement failed"
                        )

        result, _receipt = self._execute(
            target,
            operation_type="github.sync_finding",
            operation_id=service_operation_id,
            context=context,
            expected_source_revision=checked_source_revision,
            request_digest=request_digest,
            callback=sync,
        )
        return result

    def evidence_refs(
        self,
        session: AuditSession | str,
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        token: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return compact source-bound evidence links for an agent session."""

        target = self._session(session, token=token)
        selected = self._bind_context(target, context)
        return self._codex_evidence_refs_for(target, selected)

    @staticmethod
    def _codex_evidence_refs_for(
        session: AuditSession, selected: AuditSelectionContext
    ) -> tuple[dict[str, Any], ...]:
        """Build evidence links without re-reading authority during a CAS callback."""

        return (
            {
                "kind": "campaign",
                "campaign_id": selected.campaign_id,
                "episode_id": selected.episode_id or None,
                "artifact_id": session.source_ref.artifact_id
                if session.source_ref is not None
                else "campaign-source",
                "source_digest": session.source_digest,
                "source_revision": session.source_revision,
                "locator": f"audit://{session.session_id}/campaign/{selected.campaign_id}",
            },
        )

    @staticmethod
    def _codex_contains_text(value: Any, secret: str) -> bool:
        """Return whether a bounded JSON value contains a sensitive string."""

        if not secret:
            return False
        stack: list[Any] = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, str):
                if secret in current:
                    return True
            elif isinstance(current, Mapping):
                stack.extend(current.keys())
                stack.extend(current.values())
            elif isinstance(current, (tuple, list)):
                stack.extend(current)
        return False

    @staticmethod
    def _codex_route_digest(route: Mapping[str, Any]) -> str:
        """Match the Codex route receipt's digest without importing the client."""

        return hashlib.sha256(
            json.dumps(route, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    @staticmethod
    def _codex_route_digest_candidates(route: Mapping[str, Any]) -> set[str]:
        """Return accepted route identity digests for generic and App Server receipts."""

        candidates = {AuditService._codex_route_digest(route)}
        if (
            route.get("source") == "live-app-server-capability"
            and route.get("protocol") == "app-server-v2"
        ):
            candidates.add(
                hashlib.sha256(
                    json.dumps(
                        {
                            "route_id": route.get("route_id"),
                            "provider": route.get("provider"),
                            "model_id": route.get("model_id"),
                            "client_version": route.get("client_version"),
                            "protocol": route.get("protocol"),
                            "schema_digest": _CODEX_APP_SERVER_SCHEMA_DIGEST,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
            )
        return candidates

    @staticmethod
    def _codex_forbidden_result_key(key: str) -> bool:
        """Return whether a result key could carry a secret or prompt."""

        normalized = key.casefold().replace("-", "_")
        return normalized in {
            "audit_token",
            "session_token",
            "token",
            "credential",
            "credentials",
            "password",
            "prompt",
            "environment",
            "env",
        }

    @classmethod
    def _codex_result_mapping(
        cls, value: Mapping[str, Any], *, forbidden_value: str = ""
    ) -> dict[str, Any]:
        """Validate a bounded provider result and reject secret-bearing keys."""

        copied = _bounded_mapping(value, name="Codex operation result")
        if cls._codex_contains_text(copied, forbidden_value):
            raise AuditValidationError("Codex operation result contains a sensitive value")
        stack: list[Any] = [copied]
        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                for key, item in current.items():
                    if cls._codex_forbidden_result_key(str(key)):
                        raise AuditValidationError(
                            f"Codex operation result contains forbidden field: {key}"
                        )
                    stack.append(item)
            elif isinstance(current, (tuple, list)):
                stack.extend(current)
        return copied

    @staticmethod
    def _codex_usage(value: Mapping[str, Any]) -> dict[str, Any]:
        """Validate provider usage required to settle an external turn."""

        if not isinstance(value, Mapping):
            raise AuditValidationError("Codex provider usage is required")
        tokens = _nonnegative_int(value.get("tokens"), name="usage.tokens")
        compute = _finite(value.get("compute"), name="usage.compute")
        issue_writes = _nonnegative_int(value.get("issue_writes", 0), name="usage.issue_writes")
        if compute < 0:
            raise AuditValidationError("usage.compute must be non-negative")
        return {"tokens": tokens, "compute": compute, "issue_writes": issue_writes}

    def _codex_assert_binding(
        self,
        session: AuditSession,
        context: AuditSelectionContext | Mapping[str, Any],
        *,
        allow_cancelled: bool,
        allow_source_stale: bool = False,
    ) -> tuple[AuditSelectionContext, SourceRef | None, str]:
        """Revalidate the exact audit/session/source binding for Codex work."""

        selected = self._bind_context(session, context)
        if allow_source_stale and not allow_cancelled:
            raise AuditValidationError("stale-source allowance is reserved for cancellation")
        if allow_cancelled:
            # Recovery reads and the explicit kill/cancel path may inspect a
            # session before cancellation.  The caller still controls which
            # mutation can use this escape hatch.
            pass
        else:
            self._check_session(session)
        if allow_source_stale:
            return selected, session.source_ref, _digest(session.policy.to_dict())
        self._assert_source(session)
        current_source_ref = self._current_source_ref(session)
        if _source_ref_to_dict(current_source_ref) != _source_ref_to_dict(session.source_ref):
            raise AuditContextConflict("current source reference does not match the audit session")
        return selected, current_source_ref, _digest(session.policy.to_dict())

    def _codex_snapshot_matches_audit(  # noqa: C901
        self,
        snapshot: CodexSessionSnapshot,
        session: AuditSession,
        *,
        context: AuditSelectionContext | None = None,
        expected_evidence: Sequence[Mapping[str, Any]] | None = None,
        allow_historical_context: bool = False,
    ) -> None:
        """Check every persisted identity field against current authority."""

        selected = context or session.context
        expected_source_ref = _source_ref_to_dict(session.source_ref)
        expected_policy_digest = _digest(session.policy.to_dict())
        if snapshot.audit_session_id != session.session_id:
            raise AuditPolicyError("Codex session belongs to another audit session")
        if snapshot.actor != session.actor.to_dict():
            raise AuditPolicyError("Codex session actor binding changed")
        if (
            snapshot.policy_id != session.policy.policy_id
            or snapshot.policy_revision != session.policy.policy_revision
            or snapshot.policy_digest != expected_policy_digest
        ):
            raise AuditPolicyError("Codex session policy binding changed")
        snapshot_context = AuditSelectionContext.from_mapping(snapshot.context)
        if allow_historical_context:
            historical_valid = self._codex_historical_context_valid(
                snapshot_context,
                session.context,
                source_digest=session.source_digest,
                source_ref=_source_ref_to_dict(session.source_ref),
            )
        else:
            historical_valid = snapshot_context == selected
        if not historical_valid or snapshot_context != selected:
            if not allow_historical_context or not historical_valid:
                raise AuditContextConflict("Codex session context binding changed")
        if (
            snapshot.source_digest != session.source_digest
            or snapshot.source_revision != session.source_revision
            or snapshot.source_ref != expected_source_ref
        ):
            raise AuditContextConflict("Codex session source binding changed")
        if set(snapshot.route) != _CODEX_ROUTE_FIELDS:
            raise AuditValidationError("Codex session route digest is malformed")
        route_digest = snapshot.route.get("capability_digest")
        if not isinstance(route_digest, str) or len(route_digest) != 64:
            raise AuditValidationError("Codex session route digest is malformed")
        route_without_digest = {
            key: value for key, value in snapshot.route.items() if key != "capability_digest"
        }
        if route_digest not in self._codex_route_digest_candidates(route_without_digest):
            raise AuditContextConflict("Codex route binding changed")
        expected_evidence = tuple(
            expected_evidence
            if expected_evidence is not None
            else self.evidence_refs(session, context=selected)
        )
        if len(snapshot.evidence) != len(expected_evidence):
            raise AuditContextConflict("Codex evidence references changed")
        for index, (item, expected) in enumerate(
            zip(snapshot.evidence, expected_evidence, strict=True), start=1
        ):
            normalized = {
                "evidence_id": f"evidence-{index}",
                "artifact_id": expected.get("artifact_id"),
                "locator": expected.get("locator"),
                "source_digest": expected.get("source_digest"),
                "source_revision": expected.get("source_revision"),
                "episode_id": expected.get("episode_id"),
                "context_revision": selected.context_revision,
            }
            if dict(item) != normalized:
                raise AuditContextConflict("Codex evidence reference changed")

    def _codex_lease_from_state(
        self,
        state: Mapping[str, Any],
        operation: CodexOperationSnapshot,
        *,
        status: str,
        reason: str = "",
        replayed: bool = False,
    ) -> CodexOperationLease:
        """Build a typed lease from an already validated authority state."""

        extension = self._codex_extension(state)
        session = None
        if operation.codex_session_id is not None:
            raw_session = extension["sessions"].get(operation.codex_session_id)
            if isinstance(raw_session, Mapping):
                session = self._codex_session_snapshot(raw_session)
        reservation = state.get("reservations", {}).get(operation.reservation_id)
        reservation_id = operation.reservation_id
        reservation_operation_id = operation.reservation_operation_id
        reserved_tokens = reserved_issue_writes = 0
        reserved_compute = 0.0
        if isinstance(reservation, Mapping):
            parsed = self._authority_reservation(reservation)
            reserved_tokens = parsed.tokens
            reserved_compute = parsed.compute
            reserved_issue_writes = parsed.issue_writes
        return CodexOperationLease(
            status=status,
            operation=operation,
            reservation_id=reservation_id,
            reservation_operation_id=reservation_operation_id,
            reserved_tokens=reserved_tokens,
            reserved_compute=reserved_compute,
            reserved_issue_writes=reserved_issue_writes,
            session=session,
            result=operation.result,
            reason=reason or operation.reason,
            replayed=replayed,
        )

    def _codex_assert_terminal_replay_binding(  # noqa: C901
        self,
        operation: CodexOperationSnapshot,
        target: AuditSession,
        extension: Mapping[str, Any],
        *,
        request_digest: str,
        session_snapshot: CodexSessionSnapshot | None,
    ) -> None:
        """Validate a historical terminal replay without reopening provider work."""

        self._assert_source(target)
        current_source_ref = self._current_source_ref(target)
        if _source_ref_to_dict(current_source_ref) != _source_ref_to_dict(target.source_ref):
            raise AuditContextConflict("current source reference does not match the audit session")
        policy_digest = _digest(target.policy.to_dict())
        if (
            operation.audit_session_id != target.session_id
            or operation.actor != target.actor.to_dict()
            or operation.request_digest != request_digest
            or operation.policy_digest != policy_digest
            or operation.source_revision != target.source_revision
            or operation.source_digest != target.source_digest
            or operation.context_revision > target.context.context_revision
        ):
            raise AuthorityOperationConflict("Codex terminal replay binding conflicts")
        stored_snapshot: CodexSessionSnapshot | None = None
        if operation.codex_session_id is not None:
            raw_snapshot = extension["sessions"].get(operation.codex_session_id)
            if not isinstance(raw_snapshot, Mapping):
                raise AuthorityOperationConflict("Codex terminal replay session is unavailable")
            stored_snapshot = self._codex_session_snapshot(raw_snapshot)
            historical_context = AuditSelectionContext.from_mapping(stored_snapshot.context)
            if operation.context_revision != historical_context.context_revision:
                raise AuthorityOperationConflict("Codex terminal replay context conflicts")
            if not self._codex_historical_context_valid(
                historical_context,
                target.context,
                source_digest=target.source_digest,
                source_ref=_source_ref_to_dict(target.source_ref),
            ):
                raise AuthorityOperationConflict("Codex terminal replay context is stale")
        if session_snapshot is not None:
            if stored_snapshot is None:
                raise AuthorityOperationConflict("Codex terminal replay session conflicts")
            snapshot_context = AuditSelectionContext.from_mapping(session_snapshot.context)
            self._codex_snapshot_matches_audit(
                session_snapshot,
                target,
                context=snapshot_context,
                expected_evidence=self._codex_evidence_refs_for(target, snapshot_context),
                allow_historical_context=True,
            )
            if (
                stored_snapshot is not None
                and session_snapshot.codex_session_id != stored_snapshot.codex_session_id
            ):
                raise AuthorityOperationConflict("Codex terminal replay session conflicts")
            for field_name in (
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
                "last_operation_id",
            ):
                if getattr(session_snapshot, field_name) != getattr(stored_snapshot, field_name):
                    raise AuthorityOperationConflict("Codex terminal replay session conflicts")

    def begin_codex_operation(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        audit_session: AuditSession | str,
        *,
        audit_token: str | None = None,
        operation_id: str,
        action: str,
        codex_session_id: str | None = None,
        request_digest: str,
        context: AuditSelectionContext | Mapping[str, Any],
        route_digest: str,
        reserved_tokens: int = 0,
        reserved_compute: float = 0.0,
        reserved_issue_writes: int = 0,
        allow_cancelled: bool = False,
        allow_source_stale: bool = False,
        allow_context_stale: bool = False,
        allow_reservation_collision: bool = False,
    ) -> CodexOperationLease:
        """Durably admit one Codex provider operation and its reservation.

        The operation and reservation are committed in one authority
        transaction.  A recovered ``inflight`` row is deliberately returned
        as ``ambiguous`` and never grants a second provider callback.
        """

        target = self._session(audit_session, token=audit_token)
        opid = _bounded_text(operation_id, name="operation_id")
        action = _bounded_text(action, name="action")
        if action not in {"start", "reconnect", "resume", "cancel"}:
            raise AuditValidationError("unsupported Codex operation action")
        if allow_cancelled and action != "cancel":
            raise AuditValidationError("allow_cancelled is reserved for Codex cancellation")
        if allow_source_stale and (not allow_cancelled or action != "cancel"):
            raise AuditValidationError("stale-source allowance is reserved for cancellation")
        if allow_context_stale and (not allow_cancelled or action != "cancel"):
            raise AuditValidationError("stale-context allowance is reserved for cancellation")
        if allow_reservation_collision and (not allow_cancelled or action != "cancel"):
            raise AuditValidationError(
                "reservation-collision allowance is reserved for cancellation"
            )
        if self._contains_sensitive_text((opid, codex_session_id), target.session_token):
            raise AuditPolicyError("Codex operation contains a sensitive value")
        request_digest = self._digest_value(request_digest, name="request_digest")
        route_digest = self._digest_value(route_digest, name="route_digest")
        reserved_tokens = _nonnegative_int(reserved_tokens, name="reserved_tokens")
        reserved_compute = _finite(reserved_compute, name="reserved_compute")
        reserved_issue_writes = _nonnegative_int(
            reserved_issue_writes, name="reserved_issue_writes"
        )
        if reserved_compute < 0:
            raise AuditValidationError("reserved_compute must be non-negative")
        selected, current_source_ref, policy_digest = self._codex_assert_binding(
            target,
            context,
            allow_cancelled=allow_cancelled,
            allow_source_stale=allow_source_stale,
        )
        if action == "start" and codex_session_id is not None:
            raise AuditValidationError("start cannot bind an existing Codex session")
        if action != "start" and not codex_session_id:
            raise AuditValidationError("Codex session ID is required for this action")

        state = self.authority.snapshot()
        extension = self._codex_extension(state)
        sessions = extension["sessions"]
        operations = extension["operations"]
        existing_session: CodexSessionSnapshot | None = None
        if codex_session_id is not None:
            raw_session = sessions.get(codex_session_id)
            if not isinstance(raw_session, Mapping):
                raise AuditPolicyError("unknown durable Codex session")
            existing_session = self._codex_session_snapshot(raw_session)
            if not allow_context_stale:
                self._codex_snapshot_matches_audit(existing_session, target, context=selected)
            if existing_session.status == "cancelled" and action != "cancel":
                raise AuditCancelled("Codex session is cancelled")
        previous = operations.get(opid)
        if previous is not None:
            if not isinstance(previous, Mapping):
                raise AuthorityOperationConflict("Codex operation is malformed")
            operation = self._codex_operation_snapshot(previous)
            if (
                operation.audit_session_id != target.session_id
                or operation.actor != target.actor.to_dict()
                or operation.request_digest != request_digest
                or operation.action != action
                or (codex_session_id is not None and operation.codex_session_id != codex_session_id)
                or operation.policy_digest != policy_digest
                or operation.context_revision != selected.context_revision
                or operation.source_revision != target.source_revision
                or operation.source_digest != target.source_digest
                or operation.route_digest != route_digest
            ):
                raise AuthorityOperationConflict("Codex operation ID is bound to another request")
            if operation.status == "inflight":
                return self._codex_lease_from_state(
                    state,
                    operation,
                    status="ambiguous",
                    reason="Codex provider operation is inflight after a process boundary",
                    replayed=True,
                )
            return self._codex_lease_from_state(state, operation, status="replay", replayed=True)
        # A different operation ID must not be used to retry an unknown paid
        # turn for the same session (including a start with no session ID).
        for raw in operations.values():
            if not isinstance(raw, Mapping) or raw.get("status") != "inflight":
                continue
            if action == "cancel" or raw.get("audit_session_id") != target.session_id:
                continue
            operation = self._codex_operation_snapshot(raw)
            return self._codex_lease_from_state(
                state,
                operation,
                status="ambiguous",
                reason="another Codex provider operation is inflight for this session",
                replayed=True,
            )

        reservation_operation_id = f"codex-budget-reserve:{opid}"
        reservation_operation_occupied = (
            reservation_operation_id in state.get("operations", {})
            or reservation_operation_id in operations
        )
        if reservation_operation_occupied:
            if not allow_reservation_collision:
                raise AuthorityOperationConflict(
                    "Codex reservation operation ID collides with an existing operation"
                )
            collision_digest = hashlib.sha256(
                f"{target.session_id}:{opid}:{request_digest}".encode()
            ).hexdigest()
            reservation_operation_id = (
                f"codex-budget-reserve:{opid}:cancel-conflict:{collision_digest}"
            )
            if (
                reservation_operation_id in state.get("operations", {})
                or reservation_operation_id in operations
            ):
                raise AuthorityOperationConflict(
                    "Codex cancellation reservation operation ID is unavailable"
                )
        reservation_id = f"codex-reservation-{secrets.token_urlsafe(24)}"
        operation = {
            "schema_version": CODEX_AUTHORITY_SCHEMA_VERSION,
            "operation_id": opid,
            "action": action,
            "codex_session_id": codex_session_id,
            "audit_session_id": target.session_id,
            "actor": target.actor.to_dict(),
            "request_digest": request_digest,
            "policy_digest": policy_digest,
            "context_revision": selected.context_revision,
            "source_revision": target.source_revision,
            "source_digest": target.source_digest,
            "route_digest": route_digest,
            "status": "inflight",
            "result_status": "",
            "reason": "",
            "result_digest": "",
            "reservation_id": reservation_id,
            "reservation_operation_id": reservation_operation_id,
            "provider_session_id": existing_session.provider_session_id
            if existing_session is not None
            else "",
            "result": {},
            "created_at": utc_now(),
            "finished_at": "",
        }
        reservation = {
            "reservation_id": reservation_id,
            "session_id": target.session_id,
            "operation_id": reservation_operation_id,
            "tokens": reserved_tokens,
            "compute": reserved_compute,
            "issue_writes": reserved_issue_writes,
        }
        reserve_operation = {
            "operation_id": reservation_operation_id,
            "operation_type": "budget.reserve.codex",
            "session_id": target.session_id,
            "actor": target.actor.to_dict(),
            "request_digest": _digest(operation),
            "context_revision": selected.context_revision,
            "source_revision": target.source_revision,
            "source_digest": target.source_digest,
            "status": "finished",
            "result_status": "committed",
            "reason": "Codex provider reservation admitted",
            "result_digest": "",
            "created_at": operation["created_at"],
        }
        transaction_id = f"authority.codex.begin:{target.session_id}:{opid}"
        begin_digest = _digest(
            {
                "operation": operation,
                "reservation": reservation,
                "allow_cancelled": allow_cancelled,
                "allow_source_stale": allow_source_stale,
                "allow_context_stale": allow_context_stale,
                "allow_reservation_collision": allow_reservation_collision,
                "source_ref": _source_ref_to_dict(current_source_ref),
            }
        )

        def matches_request(candidate: CodexOperationSnapshot) -> bool:
            return (
                candidate.audit_session_id == target.session_id
                and candidate.actor == target.actor.to_dict()
                and candidate.request_digest == request_digest
                and candidate.action == action
                and (codex_session_id is None or candidate.codex_session_id == codex_session_id)
                and candidate.policy_digest == policy_digest
                and candidate.context_revision == selected.context_revision
                and candidate.source_revision == target.source_revision
                and candidate.source_digest == target.source_digest
                and candidate.route_digest == route_digest
            )

        def begin(state_value: dict[str, Any]) -> dict[str, Any]:
            current = state_value["sessions"].get(target.session_id)
            if not isinstance(current, Mapping):
                raise AuthorityOperationConflict("Codex audit session is no longer available")
            authoritative = self._session_from_authority_record(current, token=target.session_token)
            self._codex_assert_binding(
                authoritative,
                selected,
                allow_cancelled=allow_cancelled,
                allow_source_stale=allow_source_stale,
            )
            if opid in state_value["operations"]:
                raise AuthorityOperationConflict("Codex operation ID collides with audit operation")
            ext = state_value.setdefault(
                "extensions",
                {
                    "codex": {
                        "schema_version": CODEX_AUTHORITY_SCHEMA_VERSION,
                        "sessions": {},
                        "operations": {},
                    }
                },
            )
            codex = ext["codex"]
            if reservation_operation_id in state_value["operations"]:
                raise AuthorityOperationConflict(
                    "Codex reservation operation ID collides with an existing operation"
                )
            if reservation_operation_id in codex["operations"]:
                raise AuthorityOperationConflict(
                    "Codex reservation operation ID collides with an existing operation"
                )
            existing = codex["operations"].get(opid)
            if existing is not None:
                if not isinstance(existing, Mapping):
                    raise AuthorityOperationConflict("Codex operation is malformed")
                existing_operation = self._codex_operation_snapshot(existing)
                if not matches_request(existing_operation):
                    raise AuthorityOperationConflict(
                        "Codex operation ID is bound to another request"
                    )
                return {"operation": existing, "admitted": False}
            usage = authoritative.usage
            if (
                usage.tokens + reserved_tokens > authoritative.policy.token_budget
                or usage.compute + reserved_compute > authoritative.policy.compute_budget
                or usage.issue_writes + reserved_issue_writes
                > authoritative.policy.issue_write_budget
            ):
                raise AuditBudgetExceeded("aggregate Codex reservation exceeds session budget")
            object.__setattr__(
                authoritative,
                "usage",
                SessionUsage(
                    tokens=usage.tokens + reserved_tokens,
                    compute=usage.compute + reserved_compute,
                    issue_writes=usage.issue_writes + reserved_issue_writes,
                ),
            )
            state_value["sessions"][target.session_id] = self._authority_session_record(
                authoritative, current["token_verifier"]
            )
            state_value["reservations"][reservation_id] = reservation
            state_value["operations"][reservation_operation_id] = reserve_operation
            codex["operations"][opid] = operation
            return {"operation": operation, "admitted": True}

        mutation = self.authority.mutate(transaction_id, begin_digest, begin)
        latest_state = mutation.state
        latest_session = latest_state.get("sessions", {}).get(target.session_id)
        if isinstance(latest_session, Mapping):
            self._session_from_authority_record(
                latest_session, token=target.session_token, existing=target
            )
            self._sessions[target.session_id] = target
        self._load_reservations_state(latest_state)
        latest_ext = self._codex_extension(latest_state)
        latest_raw = latest_ext["operations"].get(opid)
        if not isinstance(latest_raw, Mapping):
            raise AuditAuthorityError("Codex operation admission did not persist")
        parsed = self._codex_operation_snapshot(latest_raw)
        if parsed.status != "inflight":
            return self._codex_lease_from_state(
                latest_state, parsed, status="replay", replayed=True
            )
        if mutation.replayed or not (
            isinstance(mutation.value, Mapping) and mutation.value.get("admitted") is True
        ):
            return self._codex_lease_from_state(
                latest_state,
                parsed,
                status="ambiguous",
                reason="Codex provider operation admission was already committed",
                replayed=True,
            )
        return self._codex_lease_from_state(latest_state, parsed, status="admitted")

    def finish_codex_operation(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        audit_session: AuditSession | str,
        *,
        audit_token: str | None = None,
        operation_id: str,
        request_digest: str,
        result_status: str,
        result: Mapping[str, Any],
        provider_session_id: str,
        usage: Mapping[str, Any],
        session_snapshot: CodexSessionSnapshot | Mapping[str, Any] | None,
        route_digest: str | None = None,
        allow_source_stale: bool = False,
    ) -> CodexOperationLease:
        """Atomically settle a Codex reservation and publish its terminal result."""

        target = self._session(audit_session, token=audit_token)
        opid = _bounded_text(operation_id, name="operation_id")
        if self._contains_sensitive_text(opid, target.session_token):
            raise AuditPolicyError("Codex operation contains a sensitive value")
        request_digest = self._digest_value(request_digest, name="request_digest")
        if result_status not in CODEX_OPERATION_STATUSES:
            raise AuditValidationError("unsupported Codex result status")
        if allow_source_stale and result_status not in {"cancelled", "conflict"}:
            raise AuditValidationError("stale-source allowance is reserved for cancellation")
        provider_session_id = _optional_text(
            provider_session_id, name="provider_session_id", limit=MAX_TEXT_CHARS
        )
        checked_result = self._codex_result_mapping(result, forbidden_value=target.session_token)
        checked_usage = self._codex_usage(usage)
        snapshot = (
            self._codex_session_snapshot(session_snapshot)
            if isinstance(session_snapshot, Mapping)
            else session_snapshot
        )
        if snapshot is not None and not isinstance(snapshot, CodexSessionSnapshot):
            raise AuditValidationError("Codex session snapshot is malformed")
        if target.session_token in provider_session_id or (
            snapshot is not None
            and self._codex_contains_text(snapshot.to_dict(), target.session_token)
        ):
            raise AuditValidationError("Codex operation contains a sensitive value")
        state = self.authority.snapshot()
        extension = self._codex_extension(state)
        raw_operation = extension["operations"].get(opid)
        if not isinstance(raw_operation, Mapping):
            raise AuditPolicyError("unknown durable Codex operation")
        operation = self._codex_operation_snapshot(raw_operation)
        result_digest = _digest(checked_result)
        if operation.status == "finished":
            self._codex_assert_terminal_replay_binding(
                operation,
                target,
                extension,
                request_digest=request_digest,
                session_snapshot=snapshot,
            )
            if route_digest is not None and route_digest != operation.route_digest:
                raise AuthorityOperationConflict("Codex route binding conflicts at finish")
            status_conflicts = operation.result_status != result_status
            if operation.result_status == "denied" and result_status in {"complete", "committed"}:
                # The serialized finish callback may have denied a nominally
                # successful provider result after a concurrent budget write.
                status_conflicts = False
            if (
                status_conflicts
                or operation.result_digest != result_digest
                or operation.provider_session_id != provider_session_id
            ):
                raise AuthorityOperationConflict("Codex operation terminal result conflicts")
            return self._codex_lease_from_state(state, operation, status="replay", replayed=True)
        selected, _current_source_ref, policy_digest = self._codex_assert_binding(
            target,
            target.context,
            allow_cancelled=allow_source_stale or result_status == "cancelled",
            allow_source_stale=allow_source_stale,
        )
        if (
            operation.audit_session_id != target.session_id
            or operation.actor != target.actor.to_dict()
            or operation.request_digest != request_digest
            or operation.policy_digest != policy_digest
            or operation.context_revision != selected.context_revision
            or operation.source_revision != target.source_revision
            or operation.source_digest != target.source_digest
        ):
            raise AuthorityOperationConflict("Codex operation binding conflicts at finish")
        if route_digest is not None and route_digest != operation.route_digest:
            raise AuthorityOperationConflict("Codex route binding conflicts at finish")
        if operation.status != "inflight":
            raise AuthorityOperationConflict("Codex operation is not inflight")
        reservation = state.get("reservations", {}).get(operation.reservation_id)
        if not isinstance(reservation, Mapping):
            raise AuthorityOperationConflict("Codex operation reservation is unavailable")
        parsed_reservation = self._authority_reservation(reservation)
        if parsed_reservation.operation_id != operation.reservation_operation_id:
            raise AuthorityOperationConflict("Codex operation reservation owner conflicts")
        if snapshot is not None:
            self._codex_snapshot_matches_audit(snapshot, target, context=selected)
            if snapshot.codex_session_id != (
                operation.codex_session_id or snapshot.codex_session_id
            ):
                raise AuthorityOperationConflict("Codex session identity conflicts at finish")
            snapshot_route_digest = snapshot.route.get("capability_digest")
            if snapshot_route_digest != operation.route_digest:
                raise AuthorityOperationConflict("Codex route identity conflicts at finish")
            if snapshot.provider_session_id != provider_session_id:
                raise AuthorityOperationConflict("Codex provider session identity conflicts")
        elif result_status in {"complete", "cancelled"}:
            raise AuditValidationError("terminal Codex success requires a session snapshot")
        if result_status == "cancelled" and snapshot is not None and snapshot.status != "cancelled":
            raise AuditValidationError("cancelled Codex result requires a cancelled snapshot")

        finish_request = {
            "operation_id": opid,
            "request_digest": request_digest,
            "result_status": result_status,
            "result": checked_result,
            "result_digest": result_digest,
            "provider_session_id": provider_session_id,
            "usage": checked_usage,
            "session_snapshot": snapshot.to_dict() if snapshot is not None else None,
        }
        finish_digest = _digest(finish_request)
        transaction_id = f"authority.codex.finish:{target.session_id}:{opid}"

        def finish(state_value: dict[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912
            current_state = state_value["sessions"].get(target.session_id)
            if not isinstance(current_state, Mapping):
                raise AuthorityOperationConflict("Codex audit session is no longer available")
            authoritative = self._session_from_authority_record(
                current_state, token=target.session_token, existing=target
            )
            authoritative_selected, _, authoritative_policy_digest = self._codex_assert_binding(
                authoritative,
                authoritative.context,
                allow_cancelled=allow_source_stale or result_status == "cancelled",
                allow_source_stale=allow_source_stale,
            )
            ext = state_value["extensions"]["codex"]
            current_raw = ext["operations"].get(opid)
            if not isinstance(current_raw, Mapping):
                raise AuthorityOperationConflict("Codex operation is missing")
            current = self._codex_operation_snapshot(current_raw)
            if (
                current.audit_session_id != authoritative.session_id
                or current.actor != authoritative.actor.to_dict()
                or current.policy_digest != authoritative_policy_digest
                or current.context_revision != authoritative_selected.context_revision
                or current.source_revision != authoritative.source_revision
                or current.source_digest != authoritative.source_digest
            ):
                raise AuthorityOperationConflict("Codex operation binding changed before finish")
            if snapshot is not None:
                self._codex_snapshot_matches_audit(
                    snapshot,
                    authoritative,
                    context=authoritative_selected,
                    expected_evidence=self._codex_evidence_refs_for(
                        authoritative, authoritative_selected
                    ),
                )
                if snapshot.codex_session_id != (
                    current.codex_session_id or snapshot.codex_session_id
                ):
                    raise AuthorityOperationConflict("Codex session identity conflicts at finish")
                if snapshot.route.get("capability_digest") != current.route_digest:
                    raise AuthorityOperationConflict("Codex route identity conflicts at finish")
                if snapshot.provider_session_id != provider_session_id:
                    raise AuthorityOperationConflict("Codex provider session identity conflicts")
            elif result_status in {"complete", "cancelled"}:
                raise AuditValidationError("terminal Codex success requires a session snapshot")
            if result_status == "cancelled" and snapshot is not None:
                if snapshot.status != "cancelled":
                    raise AuditValidationError(
                        "cancelled Codex result requires a cancelled snapshot"
                    )
            if current.status == "finished":
                return {"operation": current.to_dict()}
            if current.request_digest != request_digest:
                raise AuthorityOperationConflict("Codex operation request conflicts at finish")
            raw_reservation = state_value["reservations"].get(current.reservation_id)
            if not isinstance(raw_reservation, Mapping):
                raise AuthorityOperationConflict("Codex operation reservation is missing")
            reservation_value = self._authority_reservation(raw_reservation)
            usage = authoritative.usage
            base = SessionUsage(
                tokens=usage.tokens - reservation_value.tokens,
                compute=usage.compute - reservation_value.compute,
                issue_writes=usage.issue_writes - reservation_value.issue_writes,
            )
            if min(base.tokens, base.compute, base.issue_writes) < 0:
                raise AuthorityOperationConflict("Codex reservation is not owned by session")
            actual = SessionUsage(
                tokens=checked_usage["tokens"],
                compute=checked_usage["compute"],
                issue_writes=checked_usage["issue_writes"],
            )
            fits = (
                base.tokens + actual.tokens <= authoritative.policy.token_budget
                and base.compute + actual.compute <= authoritative.policy.compute_budget
                and base.issue_writes + actual.issue_writes
                <= authoritative.policy.issue_write_budget
            )
            if not fits:
                # The provider may overshoot the conservative reservation.  A
                # terminal non-success result burns the reservation ceiling;
                # it never releases a possibly paid turn for retry.
                actual = SessionUsage(
                    reservation_value.tokens,
                    reservation_value.compute,
                    reservation_value.issue_writes,
                )
            terminal_result_status = (
                "denied"
                if not fits and result_status in {"complete", "committed"}
                else result_status
            )
            terminal_reason = str(checked_result.get("reason", ""))[:MAX_REASON_CHARS]
            if terminal_result_status == "denied" and not terminal_reason:
                terminal_reason = "Codex operation denied by the authoritative budget ceiling"
            object.__setattr__(
                authoritative,
                "usage",
                SessionUsage(
                    tokens=base.tokens + actual.tokens,
                    compute=base.compute + actual.compute,
                    issue_writes=base.issue_writes + actual.issue_writes,
                ),
            )
            state_value["reservations"].pop(current.reservation_id, None)
            state_value["sessions"][target.session_id] = self._authority_session_record(
                authoritative, current_state["token_verifier"]
            )
            if snapshot is not None:
                ext["sessions"][snapshot.codex_session_id] = snapshot.to_dict()
            updated = current.to_dict()
            updated.update(
                {
                    "status": "finished",
                    "result_status": terminal_result_status,
                    "reason": terminal_reason,
                    "result_digest": result_digest,
                    "provider_session_id": provider_session_id,
                    "result": checked_result,
                    "finished_at": utc_now(),
                    "codex_session_id": snapshot.codex_session_id
                    if snapshot is not None
                    else current.codex_session_id,
                }
            )
            ext["operations"][opid] = updated
            return {"operation": updated}

        mutation = self.authority.mutate(transaction_id, finish_digest, finish)
        latest_state = mutation.state
        latest_session = latest_state.get("sessions", {}).get(target.session_id)
        if isinstance(latest_session, Mapping):
            self._session_from_authority_record(
                latest_session, token=target.session_token, existing=target
            )
            self._sessions[target.session_id] = target
        self._load_reservations_state(latest_state)
        latest_raw = self._codex_extension(latest_state)["operations"].get(opid)
        if not isinstance(latest_raw, Mapping):
            raise AuditAuthorityError("Codex operation finish did not persist")
        parsed = self._codex_operation_snapshot(latest_raw)
        return self._codex_lease_from_state(
            latest_state,
            parsed,
            status="replay" if mutation.replayed or parsed.status == "finished" else "admitted",
            replayed=mutation.replayed,
        )

    def load_codex_session(
        self, codex_session_id: str, *, audit_token: str
    ) -> CodexSessionSnapshot:
        """Authenticate and hydrate a persisted Codex session after restart."""

        cid = _bounded_text(codex_session_id, name="codex_session_id")
        if not isinstance(audit_token, str) or not audit_token.strip():
            raise AuditPolicyError("audit token is required for Codex recovery")
        state = self.authority.snapshot()
        extension = self._codex_extension(state)
        raw = extension["sessions"].get(cid)
        if not isinstance(raw, Mapping):
            raise AuditPolicyError("unknown durable Codex session")
        snapshot = self._codex_session_snapshot(raw)
        audit = self.reconnect_session(snapshot.audit_session_id, audit_token)
        _selected, current_source_ref, _policy_digest = self._codex_assert_binding(
            audit, audit.context, allow_cancelled=True
        )
        historical_context = AuditSelectionContext.from_mapping(snapshot.context)
        self._codex_snapshot_matches_audit(
            snapshot,
            audit,
            context=historical_context,
            expected_evidence=self._codex_evidence_refs_for(audit, historical_context),
            allow_historical_context=True,
        )
        if _source_ref_to_dict(current_source_ref) != snapshot.source_ref:
            raise AuditContextConflict("Codex source reference changed during recovery")
        expected_evidence = self._codex_evidence_refs_for(audit, historical_context)
        if len(snapshot.evidence) != len(expected_evidence):
            raise AuditContextConflict("Codex evidence references changed during recovery")
        for index, (stored, expected) in enumerate(
            zip(snapshot.evidence, expected_evidence, strict=True), start=1
        ):
            normalized = {
                "evidence_id": f"evidence-{index}",
                "artifact_id": expected.get("artifact_id"),
                "locator": expected.get("locator"),
                "source_digest": expected.get("source_digest"),
                "source_revision": expected.get("source_revision"),
                "episode_id": expected.get("episode_id"),
                "context_revision": historical_context.context_revision,
            }
            if dict(stored) != normalized:
                raise AuditContextConflict("Codex evidence reference changed during recovery")
        return snapshot

    @staticmethod
    def _codex_activity_operation_snapshot(
        operation: CodexOperationSnapshot,
    ) -> CodexOperationActivitySnapshot:
        """Project one durable operation without authority-only fields."""

        usage: Mapping[str, Any] = {}
        raw_usage = operation.result.get("usage")
        if isinstance(raw_usage, Mapping):
            usage = AuditService._codex_usage(raw_usage)
        reason = {
            "complete": "",
            "unavailable": "Codex operation unavailable",
            "failed": "Codex operation failed",
            "cancelled": "Codex operation cancelled",
            "conflict": "Codex operation conflicted",
            "denied": "Codex operation denied",
            "inflight": "Codex operation is in flight",
        }.get(operation.result_status or operation.status, "Codex operation status unavailable")
        return CodexOperationActivitySnapshot(
            operation_id=operation.operation_id,
            action=operation.action,
            codex_session_id=operation.codex_session_id,
            context_revision=operation.context_revision,
            source_revision=operation.source_revision,
            source_digest=operation.source_digest,
            status=operation.status,
            result_status=operation.result_status,
            reason=reason,
            usage=usage,
            message="",
            created_at=operation.created_at,
            finished_at=operation.finished_at,
        )

    def _codex_activity_session(
        self,
        raw: Any,
        *,
        target: AuditSession,
        context: AuditSelectionContext,
    ) -> CodexSessionSnapshot:
        """Parse and authenticate one Codex session for an activity read."""

        if not isinstance(raw, Mapping):
            raise AuditPolicyError("unknown durable Codex session")
        snapshot = self._codex_session_snapshot(raw)
        if snapshot.audit_session_id != target.session_id:
            # Do not confirm that a caller-controlled ID exists in another
            # audit session.
            raise AuditPolicyError("unknown durable Codex session")
        self._codex_snapshot_matches_audit(snapshot, target, context=context)
        return snapshot

    def _codex_activity_operation(
        self,
        raw: Any,
        *,
        target: AuditSession,
        context: AuditSelectionContext,
        policy_digest: str,
        sessions: Mapping[str, Any],
    ) -> tuple[CodexOperationSnapshot, CodexSessionSnapshot | None]:
        """Parse and authenticate one Codex operation for an activity read."""

        if not isinstance(raw, Mapping):
            raise AuditPolicyError("unknown durable Codex operation")
        operation = self._codex_operation_snapshot(raw)
        if operation.audit_session_id != target.session_id:
            # Do not confirm that a caller-controlled ID exists in another
            # audit session.
            raise AuditPolicyError("unknown durable Codex operation")
        if operation.actor != target.actor.to_dict():
            raise AuditPolicyError("Codex operation actor binding changed")
        if operation.policy_digest != policy_digest:
            raise AuditPolicyError("Codex operation policy binding changed")
        if operation.context_revision != context.context_revision:
            raise AuditContextConflict("Codex operation context binding changed")
        if (
            operation.source_digest != target.source_digest
            or operation.source_revision != target.source_revision
        ):
            raise AuditContextConflict("Codex operation source binding changed")
        session = None
        if operation.codex_session_id is not None:
            session = self._codex_activity_session(
                sessions.get(operation.codex_session_id),
                target=target,
                context=context,
            )
            route_digest = session.route.get("capability_digest")
            if route_digest != operation.route_digest:
                raise AuditContextConflict("Codex operation route binding changed")
        return operation, session

    def read_codex_activity(  # noqa: C901
        self,
        audit_session: AuditSession | str,
        *,
        codex_session_id: str | None = None,
        operation_id: str | None = None,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        audit_token: str | None = None,
        token: str | None = None,
    ) -> CodexActivitySnapshot:
        """Read source-bound Codex status without exposing authority internals.

        The caller must authenticate the owning audit session.  A supplied
        Codex session or operation ID is filtered to that audit session and
        revalidated against its current source and exact selection context.
        The authority persists status, usage, route, and evidence references,
        but not the client event stream; reads therefore report events as
        unavailable unless a future schema adds a bounded durable event index.
        ``token`` is accepted as the service-wide spelling used by ordinary
        reads; ``audit_token`` remains the Codex lifecycle spelling.
        """

        if token is not None:
            if audit_token is not None and token != audit_token:
                raise AuditPolicyError("conflicting audit tokens")
            audit_token = token
        target = self._session(audit_session, token=audit_token)
        selected, _current_source_ref, policy_digest = self._codex_assert_binding(
            target,
            target.context if context is None else context,
            allow_cancelled=True,
        )
        requested_session_id = (
            None
            if codex_session_id is None
            else _bounded_text(codex_session_id, name="codex_session_id")
        )
        requested_operation_id = (
            None if operation_id is None else _bounded_text(operation_id, name="operation_id")
        )
        if self._contains_sensitive_text(
            (requested_session_id, requested_operation_id), target.session_token
        ):
            raise AuditPolicyError("Codex activity selector contains a sensitive value")

        state = self.authority.snapshot()
        extension = self._codex_extension(state)
        sessions = extension["sessions"]
        operations = extension["operations"]
        session_snapshot: CodexSessionSnapshot | None = None
        operation_snapshot: CodexOperationSnapshot | None = None

        if requested_session_id is not None:
            session_snapshot = self._codex_activity_session(
                sessions.get(requested_session_id),
                target=target,
                context=selected,
            )

        if requested_operation_id is not None:
            operation_snapshot, operation_session = self._codex_activity_operation(
                operations.get(requested_operation_id),
                target=target,
                context=selected,
                policy_digest=policy_digest,
                sessions=sessions,
            )
            if session_snapshot is not None:
                if operation_session is None or (
                    operation_session.codex_session_id != session_snapshot.codex_session_id
                ):
                    raise AuditContextConflict(
                        "Codex activity selectors refer to different sessions"
                    )
            else:
                session_snapshot = operation_session
        elif session_snapshot is not None and session_snapshot.last_operation_id:
            operation_snapshot, operation_session = self._codex_activity_operation(
                operations.get(session_snapshot.last_operation_id),
                target=target,
                context=selected,
                policy_digest=policy_digest,
                sessions=sessions,
            )
            if operation_session is None or (
                operation_session.codex_session_id != session_snapshot.codex_session_id
            ):
                raise AuditContextConflict("Codex session operation binding changed")

        if requested_session_id is None and requested_operation_id is None:
            candidate_operations = [
                raw
                for raw in operations.values()
                if isinstance(raw, Mapping) and raw.get("audit_session_id") == target.session_id
            ]
            if candidate_operations:
                latest = max(
                    candidate_operations,
                    key=lambda raw: str(raw.get("created_at", "")),
                )
                operation_snapshot, session_snapshot = self._codex_activity_operation(
                    latest,
                    target=target,
                    context=selected,
                    policy_digest=policy_digest,
                    sessions=sessions,
                )
            else:
                candidate_sessions = [
                    raw
                    for raw in sessions.values()
                    if isinstance(raw, Mapping) and raw.get("audit_session_id") == target.session_id
                ]
                if candidate_sessions:
                    latest = max(
                        candidate_sessions,
                        key=lambda raw: str(raw.get("updated_at", "")),
                    )
                    session_snapshot = self._codex_activity_session(
                        latest,
                        target=target,
                        context=selected,
                    )

        evidence = (
            tuple(dict(item) for item in session_snapshot.evidence)
            if session_snapshot is not None
            else tuple(
                {
                    "evidence_id": f"evidence-{index}",
                    **dict(item),
                    "context_revision": selected.context_revision,
                }
                for index, item in enumerate(
                    self._codex_evidence_refs_for(target, selected), start=1
                )
            )
        )
        projected_operation = (
            self._codex_activity_operation_snapshot(operation_snapshot)
            if operation_snapshot is not None
            else None
        )
        status = (
            projected_operation.result_status or projected_operation.status
            if projected_operation is not None
            else session_snapshot.status
            if session_snapshot is not None
            else "unavailable"
        )
        reason = projected_operation.reason if projected_operation is not None else ""
        return CodexActivitySnapshot(
            session=session_snapshot,
            operation=projected_operation,
            usage=target.usage,
            status=status,
            reason=reason,
            evidence=evidence,
        )

    def _write_record(  # noqa: C901
        self,
        session: AuditSession,
        record: Annotation | Reference | Finding,
        *,
        operation_type: str,
        operation_id: str | None,
        expected_revision: int | None,
        context: AuditSelectionContext | Mapping[str, Any] | None,
        expected_source_revision: int | str | None,
    ) -> ServiceResult[CommitResult]:
        self._assert_source(session, expected_source_revision=expected_source_revision)
        selected = self._bind_context(session, context)
        if isinstance(record, Annotation):
            if not selected.episode_id or record.episode_id != selected.episode_id:
                raise AuditContextConflict("annotation episode does not match selected context")
            if record.author_kind != session.actor.kind:
                raise AuditPolicyError("annotation author_kind does not match session actor")
            if record.author_id and record.author_id != session.actor.actor_id:
                raise AuditPolicyError("annotation author_id does not match session actor")
            for reference in record.references:
                nested_source = reference.source
                if nested_source is not None and not self._source_ref_matches(
                    session, nested_source
                ):
                    raise AuditContextConflict(
                        "annotation reference source does not match session source"
                    )
                if (
                    reference.source_revision
                    and session.source_revision not in (0, "")
                    and reference.source_revision != str(session.source_revision)
                ):
                    raise AuditContextConflict(
                        "annotation reference source revision does not match selection"
                    )
        elif session.actor.kind == "agent" and isinstance(record, Finding):
            if record.confirmed_members or record.status in {"supported", "resolved"}:
                raise AuditPolicyError("agent findings cannot silently confirm or resolve members")
        record_source_identity = getattr(record, "source_identity", "")
        if record_source_identity and record_source_identity not in self._source_identities(
            session
        ):
            raise AuditContextConflict("record source identity does not match session source")
        record_source = getattr(record, "source_ref", None) or getattr(record, "source", None)
        if record_source is not None and not self._source_ref_matches(session, record_source):
            raise AuditContextConflict("record source reference does not match session source")
        if (
            isinstance(record, Reference)
            and record.source_revision
            and session.source_revision not in (0, "")
            and record.source_revision != str(session.source_revision)
        ):
            raise AuditContextConflict("reference source revision does not match selection")
        record_source_revision = getattr(record, "source_revision", 0)
        if (
            record_source_revision not in (0, "", None)
            and session.source_revision not in (0, "")
            and str(record_source_revision) != str(session.source_revision)
        ):
            raise AuditContextConflict("record source revision does not match selection")
        return self._save_record_source_bound(
            session,
            record,
            operation_id=operation_id,
            expected_revision=expected_revision,
            expected_source_revision=expected_source_revision,
        )

    def _save_record_source_bound(
        self,
        session: AuditSession,
        record: Annotation | Reference | Finding,
        *,
        operation_id: str | None,
        expected_revision: int | None,
        expected_source_revision: int | str | None,
    ) -> CommitResult:
        """CAS-save a record with a source guard before and after the commit.

        The post-save source check is the service's linearization point: a
        source mutation observed before it compensates the store commit and
        returns ``conflict``.  The source is external to the audit-store lock,
        so a mutation after that check cannot be prevented or rolled back
        atomically.  Subsequent service reads recheck the source and fail
        closed instead of treating the committed record as current evidence.
        """

        record_id_value = getattr(
            record,
            "annotation_id",
            getattr(record, "reference_id", getattr(record, "finding_id", "")),
        )
        opid = self._operation_id(operation_id)
        with self._lock:
            self._assert_source(session, expected_source_revision=expected_source_revision)
            previous = self.store.get(record_id_value, include_deleted=True)
            current = self.store.get(record_id_value)
            expected = expected_revision
            if expected is None and current is None:
                expected = 0
            committed = self.store.save(
                record,
                operation_id=opid,
                expected_revision=expected,
                actor=session.actor.to_dict(),
            )
            try:
                self._assert_source(session, expected_source_revision=expected_source_revision)
            except AuditContextConflict as exc:
                rollback_id = f"{opid}-source-rollback-{uuid.uuid4().hex}"
                try:
                    if (
                        previous is not None
                        and not previous.deleted
                        and previous.record is not None
                    ):
                        self.store.save(
                            previous.record,
                            operation_id=rollback_id,
                            expected_revision=committed.revision,
                            actor=session.actor.to_dict(),
                        )
                    else:
                        self.store.delete(
                            record_id_value,
                            operation_id=rollback_id,
                            expected_revision=committed.revision,
                            actor=session.actor.to_dict(),
                        )
                except AuditStoreError as rollback_exc:
                    raise AuditContextConflict(
                        "campaign source changed during record commit and rollback failed"
                    ) from rollback_exc
                raise AuditContextConflict("campaign source changed during record commit") from exc
            return committed

    def write_annotation(
        self,
        session: AuditSession | str,
        annotation: Annotation | Mapping[str, Any],
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[CommitResult]:
        """Persist one annotation through BA-03 with context/source CAS."""

        target = self._session(session, token=token)
        value = annotation if isinstance(annotation, Annotation) else record_from_dict(annotation)
        if not isinstance(value, Annotation):
            raise AuditValidationError("annotation payload is not an Annotation record")
        opid = self._operation_id(operation_id)
        result, _ = self._execute(
            target,
            operation_type="write.annotation",
            operation_id=opid,
            context=context,
            expected_source_revision=expected_source_revision,
            request_digest=_operation_request_digest(
                record_to_dict(value),
                expected_revision=expected_revision,
                expected_source_revision=expected_source_revision,
            ),
            callback=lambda: self._write_record(
                target,
                value,
                operation_type="write.annotation",
                operation_id=opid,
                expected_revision=expected_revision,
                context=context,
                expected_source_revision=expected_source_revision,
            ),
        )
        return result

    save_annotation = write_annotation

    def write_reference(
        self,
        session: AuditSession | str,
        reference: Reference | Mapping[str, Any],
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[CommitResult]:
        """Persist one source-bound reference through BA-03."""

        target = self._session(session, token=token)
        value = reference if isinstance(reference, Reference) else record_from_dict(reference)
        if not isinstance(value, Reference):
            raise AuditValidationError("reference payload is not a Reference record")
        opid = self._operation_id(operation_id)
        result, _ = self._execute(
            target,
            operation_type="write.reference",
            operation_id=opid,
            context=context,
            expected_source_revision=expected_source_revision,
            request_digest=_operation_request_digest(
                record_to_dict(value),
                expected_revision=expected_revision,
                expected_source_revision=expected_source_revision,
            ),
            callback=lambda: self._write_record(
                target,
                value,
                operation_type="write.reference",
                operation_id=opid,
                expected_revision=expected_revision,
                context=context,
                expected_source_revision=expected_source_revision,
            ),
        )
        return result

    save_reference = write_reference

    def write_finding(
        self,
        session: AuditSession | str,
        finding: Finding | Mapping[str, Any],
        *,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[CommitResult]:
        """Persist a candidate finding through BA-03 without agent confirmation."""

        target = self._session(session, token=token)
        value = finding if isinstance(finding, Finding) else record_from_dict(finding)
        if not isinstance(value, Finding):
            raise AuditValidationError("finding payload is not a Finding record")
        opid = self._operation_id(operation_id)
        result, _ = self._execute(
            target,
            operation_type="write.finding",
            operation_id=opid,
            context=context,
            expected_source_revision=expected_source_revision,
            request_digest=_operation_request_digest(
                record_to_dict(value),
                expected_revision=expected_revision,
                expected_source_revision=expected_source_revision,
            ),
            callback=lambda: self._write_record(
                target,
                value,
                operation_type="write.finding",
                operation_id=opid,
                expected_revision=expected_revision,
                context=context,
                expected_source_revision=expected_source_revision,
            ),
        )
        return result

    save_finding = write_finding

    def finding_from_annotation(
        self,
        session: AuditSession | str,
        annotation: Annotation | Mapping[str, Any],
        *,
        finding_id: str,
        title: str | None = None,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        token: str | None = None,
    ) -> ServiceResult[CommitResult]:
        """Create a proposed/candidate finding from an annotation."""

        target = self._session(session, token=token)
        value = annotation if isinstance(annotation, Annotation) else record_from_dict(annotation)
        if not isinstance(value, Annotation):
            raise AuditValidationError("annotation payload is not an Annotation record")
        bounded_finding_id = _bounded_text(finding_id, name="finding_id")
        opid = self._operation_id(operation_id)
        request_digest = _operation_request_digest(
            {
                "annotation": record_to_dict(value),
                "finding_id": bounded_finding_id,
                "title": title,
            }
        )

        def create() -> CommitResult:
            self._validate_annotation_for_finding(target, value, context=context)
            finding = create_from_annotation(value, finding_id=bounded_finding_id, title=title)
            source_revision = self._source_revision_token(value.source_revision)
            if value.source_revision == 0 and value.source_identity == target.source_digest:
                source_revision = "0"
            if source_revision:
                finding = replace(finding, source_revision=source_revision)
            return self._write_record(
                target,
                finding,
                operation_type="write.finding",
                operation_id=opid,
                expected_revision=None,
                context=context,
                expected_source_revision=None,
            )

        result, _ = self._execute(
            target,
            operation_type="write.finding",
            operation_id=opid,
            context=context,
            request_digest=request_digest,
            callback=create,
        )
        return result

    def list_operation_records(
        self, session: AuditSession | str, *, token: str | None = None
    ) -> tuple[ActionRecord, ...]:
        """Return before/after operation records for this actor/session."""

        target = self._session(session, token=token)
        records = []
        for stored in self.store.list_records(record_type="action_record"):
            record = stored.record
            if (
                isinstance(record, ActionRecord)
                and record.actor_kind == target.actor.kind
                and record.details.get("session_id") == target.session_id
            ):
                records.append(record)
        return tuple(records)


__all__ = [
    "ACTOR_KINDS",
    "AUDIT_OPERATION_SCHEMA_VERSION",
    "AUDIT_POLICY_SCHEMA_VERSION",
    "AUDIT_SELECTION_CONTEXT_SCHEMA_VERSION",
    "AUDIT_SERVICE_SCHEMA_VERSION",
    "AUTHORITY_SCHEMA_VERSION",
    "CODEX_ACTIVITY_SCHEMA_VERSION",
    "CODEX_AUTHORITY_SCHEMA_VERSION",
    "CODEX_OPERATION_STATUSES",
    "NATIVE_DIAGNOSTIC_OPERATION_TYPE",
    "ActorRef",
    "AuditAuthorityError",
    "AuditAuthorityStore",
    "AuditBudgetExceeded",
    "AuditCancelled",
    "AuditContextConflict",
    "AuditCoverageAdapter",
    "AuditCursor",
    "AuditNextAmbiguous",
    "AuditNextBlocked",
    "AuditPolicyError",
    "AuditQueueAdapter",
    "AuditQueueNextAdapter",
    "AuditSelectionContext",
    "AuditService",
    "AuditServiceError",
    "AuditSession",
    "AuditValidationError",
    "BudgetReservation",
    "CampaignView",
    "CapabilityResult",
    "CodexActivitySnapshot",
    "CodexOperationActivitySnapshot",
    "CodexOperationLease",
    "CodexOperationSnapshot",
    "CodexSessionSnapshot",
    "EpisodeView",
    "NativeDiagnosticBinding",
    "NativeDiagnosticServiceConfig",
    "OperationReceipt",
    "ServiceResult",
    "SessionPolicy",
    "SessionUsage",
    "StoredRecord",
    "UnavailableCoverageAdapter",
    "UnavailableQueueAdapter",
]
