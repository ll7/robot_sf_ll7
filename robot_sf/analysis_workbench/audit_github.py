# ruff: noqa: C901, D102, D105, D107, DOC201, PLR0912, PLR0913, PLR0915

"""Finding-level GitHub synchronization with a durable, offline-testable seam.

The synchronizer is deliberately small and policy-neutral.  It renders one
finding into a stable marker, records the immutable publication intent in the
canonical :class:`~audit_store.AuditStore` as an outbox action, and delegates
transport to a provider protocol.  V1 publication creates an immutable issue
snapshot once and appends later finding revisions as marked comments.  The
historical body-block helpers remain only as a compatibility/testing seam and
are never selected by the append-only REST adapter.  Session policy,
issue-write budgets, credentials and live-provider construction remain
service-owned integration work.

The important safety properties are local and deterministic:

* a ``(repository, finding_id)`` claim serializes all operation IDs and each operation has one
  durable ``(repository, finding_id, operation_id)`` outbox entry; terminal entry and claim
  versions are committed together and replay repairs an older split boundary;
* a create is preceded by exact-marker search and an ambiguous create is
  reconciled before any retry;
* incomplete marker pagination never authorizes a remote mutation;
* append-only providers never rewrite issue bodies or comments, while the
  historical delimited-block helpers remain available only to explicit legacy
  fakes;
* canonical-linked remote writes require an explicit provider revision-reservation seam and
  otherwise fail closed before mutation;
* marker, finding, repository, source and response mismatches fail closed; and
* credentials, configured private roots and local-only media references never
  enter a public issue body.

No live GitHub client is constructed here.  Fake providers can exercise the
full state machine without credentials or network mutations.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from robot_sf.analysis_workbench.audit_contracts import (
    ActionRecord,
    AuditContractError,
    Finding,
    canonical_json,
    record_to_dict,
    utc_now,
)
from robot_sf.analysis_workbench.audit_store import (
    AuditConflictError,
    AuditStore,
    AuditStoreError,
    StoredRecord,
)

try:  # ``FindingStore`` is a narrow optional adapter, not a module dependency.
    from robot_sf.analysis_workbench.audit_findings import FindingStore
except ImportError:  # pragma: no cover - defensive for partial package imports.
    FindingStore = Any  # type: ignore[misc,assignment]


GITHUB_SYNC_SCHEMA_VERSION = "github-sync.v1"
GITHUB_OUTBOX_SCHEMA_VERSION = "github-outbox.v1"
GITHUB_LINK_SCHEMA_VERSION = "github-link.v1"
GITHUB_CLAIM_SCHEMA_VERSION = "github-claim.v1"
GITHUB_PUBLICATION_SCHEMA_VERSION = "github-publication.v1"

FINDING_MARKER_PREFIX = "<!-- robot_sf_audit_finding:v1 "
FINDING_MARKER_SUFFIX = " -->"
AUDITOR_BLOCK_START = "<!-- robot_sf_auditor_block:v1 -->"
AUDITOR_BLOCK_END = "<!-- /robot_sf_auditor_block:v1 -->"

_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_FINDING_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_MARKER_RE = re.compile(
    r"<!-- robot_sf_audit_finding:v1 repository=(?P<repository>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+) "
    r"finding_id=(?P<finding_id>[A-Za-z0-9_.:-]+) -->"
)
_MARKER_TOKEN_RE = re.compile(r"<!--\s*robot_sf_audit_finding:v1(?P<body>.*?)-->", re.DOTALL)
_MARKER_SHAPED_TEXT_RE = re.compile(r"<!--\s*robot_sf_audit_finding:v1", re.IGNORECASE)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{12,}\b"),
    re.compile(r"(?i)\bgithub_pat_[A-Za-z0-9_]{12,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s,;]+"),
    re.compile(
        r"(?i)\b(?:token|credential|oauth[_-]?token|client[_-]?secret|private[_-]?key|authorization)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
)
_SECRET_KEY_RE = re.compile(
    r"(?i)(?:^|_)(?:token|tokens|credential|credentials|password|passwd|secret|secrets|api_key|"
    r"api_keys|access_token|access_tokens|oauth_token|oauth_tokens|client_secret|private_key|"
    r"authorization|auth)(?:$|_)"
)
_LOCAL_MEDIA_KEYS = {
    "media",
    "media_url",
    "media_urls",
    "video",
    "video_url",
    "video_urls",
    "image",
    "image_url",
    "image_urls",
    "artifact_url",
    "artifact_urls",
    "local_media",
}
_OUTBOX_STATES = frozenset({"pending", "in_flight", "succeeded", "ambiguous", "conflict", "failed"})
_CLAIM_STATES = frozenset({"in_flight", "succeeded", "ambiguous", "conflict", "failed"})
_PUBLICATION_KINDS = frozenset({"legacy_body", "initial_issue", "revision_comment"})
_PUBLICATION_SCHEMA_VERSIONS = frozenset({GITHUB_PUBLICATION_SCHEMA_VERSION, "github-publication.legacy"})


class GitHubSyncError(RuntimeError):
    """Base class for fail-closed synchronization errors."""


class GitHubValidationError(GitHubSyncError, ValueError):
    """Raised for malformed findings, markers, provider responses or links."""


class GitHubConflictError(GitHubSyncError):
    """Raised when a remote or durable value changed outside the operation CAS."""


class GitHubPrivacyError(GitHubSyncError, ValueError):
    """Raised when evidence cannot be published safely."""


class GitHubTransportError(GitHubSyncError):
    """Raised when a provider response is malformed or transport is unavailable."""


class GitHubOutboxError(GitHubSyncError):
    """Raised when the durable outbox is malformed or has a stale CAS write."""


class GitHubAmbiguousCreate(GitHubTransportError):
    """A create may have reached GitHub, so a blind retry is forbidden."""


class GitHubIssueMissing(GitHubTransportError):
    """A previously claimed issue no longer exists at the provider."""


class GitHubProvider(Protocol):
    """Narrow transport seam implemented by a later live adapter or a fake.

    V1 providers expose ``append_auditor_comment`` and never rewrite an
    existing issue body.  The legacy body-CAS methods remain only for
    compatibility fakes and are not selected when append-only capability is
    present.  A provider without a native GitHub CAS cannot satisfy the
    historical body-replacement contract; that limitation is retained in the
    durable acceptance record instead of being hidden behind a GET-then-PATCH.
    """

    def search_issues(
        self, repository: str, *, marker: str
    ) -> SearchResult | Sequence[GitHubIssue | Mapping[str, Any]]:
        """Search one repository for the exact stable marker."""

    def create_issue(
        self,
        repository: str,
        *,
        title: str,
        body: str,
        labels: Sequence[str],
    ) -> GitHubIssue | Mapping[str, Any]:
        """Create one issue and return a complete remote snapshot."""

    def get_issue(self, repository: str, number: int) -> GitHubIssue | Mapping[str, Any]:
        """Read one issue by repository and number."""

    def append_auditor_comment(
        self,
        repository: str,
        number: int,
        *,
        body: str,
        request_digest: str,
    ) -> Any:
        """Append one idempotent, marker-bearing auditor comment.

        A conforming provider returns a mapping or value exposing ``comment``
        (with an ``id`` and ``body``), ``status`` and the immutable request
        marker.  Providers without this capability are outside the V1 path.
        """

    def update_issue(
        self, repository: str, number: int, *, body: str
    ) -> GitHubIssue | Mapping[str, Any]:
        """Legacy body write retained for create-only/simple fake providers."""

    def update_issue_if_unchanged(
        self,
        repository: str,
        number: int,
        *,
        body: str,
        expected_body_digest: str,
        expected_updated_at: str,
    ) -> GitHubIssue | Mapping[str, Any]:
        """Replace the body only when the provider snapshot is unchanged."""

    def create_issue_with_finding_revision(
        self,
        repository: str,
        *,
        finding_id: str,
        expected_finding_revision: int,
        title: str,
        body: str,
        labels: Sequence[str],
    ) -> GitHubIssue | Mapping[str, Any]:
        """Create under a provider/service-owned canonical revision reservation."""

    def update_issue_with_finding_revision(
        self,
        repository: str,
        number: int,
        *,
        finding_id: str,
        expected_finding_revision: int,
        body: str,
        expected_body_digest: str,
        expected_updated_at: str,
    ) -> GitHubIssue | Mapping[str, Any]:
        """Conditionally update under a canonical revision reservation."""


@dataclass(frozen=True, slots=True)
class GitHubIssue:
    """Validated remote issue snapshot used by the transport seam."""

    repository: str
    number: int
    url: str
    title: str
    body: str
    labels: tuple[str, ...] = ()
    state: str = "open"
    comments: tuple[Mapping[str, Any], ...] = ()
    updated_at: str = ""

    def __post_init__(self) -> None:
        _validate_repository(self.repository)
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number <= 0:
            raise GitHubValidationError("remote issue number must be a positive integer")
        if not isinstance(self.url, str) or not self.url.startswith(("https://", "http://")):
            raise GitHubValidationError("remote issue url must be an http(s) URL")
        if not isinstance(self.title, str) or not self.title.strip():
            raise GitHubValidationError("remote issue title must be non-empty")
        if not isinstance(self.body, str):
            raise GitHubValidationError("remote issue body must be text")
        if self.state not in {"open", "closed"}:
            raise GitHubValidationError("remote issue state must be open or closed")
        if any(not isinstance(label, str) or not label.strip() for label in self.labels):
            raise GitHubValidationError("remote issue labels must be non-empty strings")
        if any(not isinstance(comment, Mapping) for comment in self.comments):
            raise GitHubValidationError("remote issue comments must be mappings")
        if not isinstance(self.updated_at, str):
            raise GitHubValidationError("remote issue updated_at must be text")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GitHubIssue:
        """Build a strict issue snapshot from a provider response."""

        if not isinstance(payload, Mapping):
            raise GitHubValidationError("provider issue response must be an object")
        required = {"repository", "number", "url", "title", "body"}
        missing = sorted(required - set(payload))
        if missing:
            raise GitHubValidationError(f"provider issue response is missing: {', '.join(missing)}")
        labels = payload.get("labels", ())
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise GitHubValidationError("provider issue labels must be a list")
        comments = payload.get("comments", ())
        if not isinstance(comments, Sequence) or isinstance(comments, (str, bytes)):
            raise GitHubValidationError("provider issue comments must be a list")
        return cls(
            repository=payload["repository"],
            number=payload["number"],
            url=payload["url"],
            title=payload["title"],
            body=payload["body"],
            labels=tuple(labels),
            state=payload.get("state", "open"),
            comments=tuple(comments),
            updated_at=payload.get("updated_at", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable provider-neutral snapshot."""

        return {
            "repository": self.repository,
            "number": self.number,
            "url": self.url,
            "title": self.title,
            "body": self.body,
            "labels": list(self.labels),
            "state": self.state,
            "comments": [dict(comment) for comment in self.comments],
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Provider search result with an explicit completeness boundary."""

    issues: tuple[GitHubIssue, ...] = ()
    complete: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if any(not isinstance(issue, GitHubIssue) for issue in self.issues):
            raise GitHubValidationError("search issues must be validated GitHubIssue values")
        if not isinstance(self.complete, bool):
            raise GitHubValidationError("search completeness must be boolean")
        if not isinstance(self.reason, str):
            raise GitHubValidationError("search reason must be text")

    @classmethod
    def from_value(cls, value: Any) -> SearchResult:
        if isinstance(value, SearchResult):
            return value
        if isinstance(value, Mapping):
            raw_issues = value.get("issues")
            if not isinstance(raw_issues, Sequence) or isinstance(raw_issues, (str, bytes)):
                raise GitHubValidationError("provider search response issues must be a list")
            return cls(
                issues=tuple(_coerce_issue(item) for item in raw_issues),
                complete=value.get("complete", True),
                reason=value.get("reason", ""),
            )
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise GitHubValidationError("provider search response must be a list or object")
        return cls(issues=tuple(_coerce_issue(item) for item in value))


@dataclass(frozen=True, slots=True)
class FindingEvidence:
    """Source-bound evidence fields permitted in a public finding issue."""

    campaign_id: str = ""
    source_identity: str = ""
    source_revision: str = ""
    source_digest: str = ""
    representative_cases: tuple[str, ...] = ()
    reproduction_commands: tuple[str, ...] = ()
    artifact_urls: tuple[str, ...] = ()
    media_urls: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    competing_explanations: tuple[str, ...] = ()
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "campaign_id",
            "source_identity",
            "source_revision",
            "source_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise GitHubValidationError(f"evidence {name} must be text")
        for name in (
            "representative_cases",
            "reproduction_commands",
            "artifact_urls",
            "media_urls",
            "observations",
            "competing_explanations",
        ):
            values = getattr(self, name)
            if any(not isinstance(item, str) for item in values):
                raise GitHubValidationError(f"evidence {name} must contain text")
        if any(not isinstance(item, Mapping) for item in self.diagnostics):
            raise GitHubValidationError("evidence diagnostics must contain mappings")
        if not isinstance(self.metadata, Mapping):
            raise GitHubValidationError("evidence metadata must be a mapping")

    @classmethod
    def from_value(cls, value: FindingEvidence | Mapping[str, Any] | None) -> FindingEvidence:
        if value is None:
            return cls()
        if isinstance(value, FindingEvidence):
            return value
        if not isinstance(value, Mapping):
            raise GitHubValidationError("finding evidence must be a mapping")

        def _items(*names: str) -> tuple[str, ...]:
            for name in names:
                if name in value:
                    raw = value[name]
                    if isinstance(raw, str):
                        return (raw,)
                    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, str)):
                        raise GitHubValidationError(f"evidence {name} must be a list")
                    return tuple(raw)
            return ()

        campaign_id = value.get("campaign_id", value.get("campaign_digest", ""))
        source_identity = value.get("source_identity", value.get("source_id", ""))
        source_revision = value.get("source_revision", "")
        source_digest = value.get("source_digest", "")
        for name, item in (
            ("campaign_id", campaign_id),
            ("source_identity", source_identity),
            ("source_revision", source_revision),
            ("source_digest", source_digest),
        ):
            if item is None:
                item = ""
            if not isinstance(item, (str, int)) or isinstance(item, bool):
                raise GitHubValidationError(f"evidence {name} must be text or integer")
            if name == "campaign_id":
                campaign_id = str(item)
            elif name == "source_identity":
                source_identity = str(item)
            elif name == "source_revision":
                source_revision = str(item)
            else:
                source_digest = str(item)
        diagnostics = value.get("diagnostics", value.get("diagnostic_results", ()))
        if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
            raise GitHubValidationError("evidence diagnostics must be a list")
        metadata = value.get("metadata", {})
        return cls(
            campaign_id=campaign_id,
            source_identity=source_identity,
            source_revision=source_revision,
            source_digest=source_digest,
            representative_cases=_items("representative_cases", "cases", "episode_ids"),
            reproduction_commands=_items("reproduction_commands", "commands"),
            artifact_urls=_items("artifact_urls", "artifacts"),
            media_urls=_items("media_urls", "media"),
            observations=_items("observations"),
            competing_explanations=_items("competing_explanations", "hypotheses"),
            diagnostics=tuple(diagnostics),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class PublicDataFilter:
    """Redact private text and reject local-only public media references."""

    private_roots: tuple[Path, ...] = ()
    allowed_media_hosts: tuple[str, ...] = ()

    def __init__(
        self,
        private_roots: Sequence[str | Path] = (),
        allowed_media_hosts: Sequence[str] = (),
    ) -> None:
        roots = tuple(Path(root).expanduser().resolve(strict=False) for root in private_roots)
        hosts = tuple(
            str(host).lower().strip() for host in allowed_media_hosts if str(host).strip()
        )
        object.__setattr__(self, "private_roots", roots)
        object.__setattr__(self, "allowed_media_hosts", hosts)

    def text(self, value: Any) -> str:
        """Return text with configured roots and credential-shaped values redacted."""

        if isinstance(value, Path):
            value = str(value)
        if not isinstance(value, str):
            value = str(value)
        result = value
        for root in self.private_roots:
            root_text = str(root)
            result = result.replace(root_text, "[REDACTED_PRIVATE_PATH]")
        for pattern in _SECRET_PATTERNS:
            result = pattern.sub("[REDACTED_SECRET]", result)
        return _MARKER_SHAPED_TEXT_RE.sub("&lt;!-- robot_sf_audit_finding:v1", result)

    def value(self, value: Any, *, key: str = "") -> Any:
        """Recursively sanitize JSON-like evidence while checking media URLs."""

        normalized_key = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", key)
        normalized_key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", normalized_key)
        normalized_key = re.sub(r"[^A-Za-z0-9_]+", "_", normalized_key).lower()
        secret_key = bool(_SECRET_KEY_RE.search(normalized_key))
        if secret_key:
            return "[REDACTED_SECRET]"
        if isinstance(value, Mapping):
            return {
                self.text(name): self.value(item, key=str(name)) for name, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [self.value(item, key=normalized_key) for item in value]
        if isinstance(value, (str, Path)):
            text = self.text(value)
            if normalized_key in _LOCAL_MEDIA_KEYS or normalized_key.endswith("_media_url"):
                self._reject_local_media(text)
            return text
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        raise GitHubPrivacyError(f"unsupported evidence value type: {type(value).__name__}")

    def _reject_local_media(self, value: str) -> None:
        parsed = urlparse(value)
        if parsed.scheme in {"file", "data", "blob"}:
            raise GitHubPrivacyError("local-only media URL cannot be published")
        if parsed.scheme in {"http", "https"}:
            hostname = (parsed.hostname or "").lower()
            if hostname in {"localhost", "localhost.localdomain"}:
                raise GitHubPrivacyError("loopback media URL cannot be published")
            try:
                address = ipaddress.ip_address(hostname)
            except ValueError:
                address = None
            if address is not None and (
                address.is_loopback or address.is_private or address.is_link_local
            ):
                raise GitHubPrivacyError("private-network media URL cannot be published")
            if self.allowed_media_hosts and hostname not in self.allowed_media_hosts:
                raise GitHubPrivacyError(f"media host {hostname!r} is not authorized")
            return
        if parsed.scheme:
            raise GitHubPrivacyError("unsupported media URL scheme cannot be published")
        if value.startswith(("/", "./", "../", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise GitHubPrivacyError("local media path cannot be published")


@dataclass(frozen=True, slots=True)
class RenderedFinding:
    """Sanitized issue request and the digest used for block CAS."""

    repository: str
    finding_id: str
    marker: str
    title: str
    body: str
    auditor_block: str
    auditor_block_digest: str
    labels: tuple[str, ...]
    evidence: Mapping[str, Any]

    @property
    def request_digest(self) -> str:
        """Return the stable digest binding operation input to rendered bytes."""

        return _sha256(
            {
                "schema_version": GITHUB_SYNC_SCHEMA_VERSION,
                "repository": self.repository,
                "finding_id": self.finding_id,
                "marker": self.marker,
                "title": self.title,
                "body": self.body,
                "labels": self.labels,
            }
        )


@dataclass(frozen=True, slots=True)
class GitHubOutboxEntry:
    """Versioned durable request state persisted through ``AuditStore``."""

    repository: str
    finding_id: str
    operation_id: str
    request_digest: str
    title: str
    body: str
    marker: str
    labels: tuple[str, ...]
    auditor_block_digest: str
    expected_block_digest: str = ""
    finding_revision: int | None = None
    state: str = "pending"
    attempts: int = 0
    worker_id: str = ""
    issue: Mapping[str, Any] | None = None
    reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: str = GITHUB_OUTBOX_SCHEMA_VERSION
    publication_schema_version: str = "github-publication.legacy"
    publication_kind: str = "legacy_body"
    publication_revision: int | None = None
    publication_digest: str = ""
    publication_key: str = ""
    comment: Mapping[str, Any] | None = None
    request_operation_id: str = ""

    def __post_init__(self) -> None:
        _validate_repository(self.repository)
        _validate_finding_id(self.finding_id)
        _validate_operation_id(self.operation_id)
        if self.schema_version != GITHUB_OUTBOX_SCHEMA_VERSION:
            raise GitHubValidationError(f"unsupported outbox schema: {self.schema_version}")
        if self.publication_schema_version not in _PUBLICATION_SCHEMA_VERSIONS:
            raise GitHubValidationError(
                f"unsupported publication schema: {self.publication_schema_version}"
            )
        if self.publication_kind not in _PUBLICATION_KINDS:
            raise GitHubValidationError(f"unknown publication kind: {self.publication_kind}")
        if self.state not in _OUTBOX_STATES:
            raise GitHubValidationError(f"unknown outbox state: {self.state}")
        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts < 0
        ):
            raise GitHubValidationError("outbox attempts must be a non-negative integer")
        if self.finding_revision is not None and (
            isinstance(self.finding_revision, bool)
            or not isinstance(self.finding_revision, int)
            or self.finding_revision < 0
        ):
            raise GitHubValidationError("outbox finding_revision must be non-negative")
        if self.publication_revision is not None and (
            isinstance(self.publication_revision, bool)
            or not isinstance(self.publication_revision, int)
            or self.publication_revision < 0
        ):
            raise GitHubValidationError("publication_revision must be non-negative")
        for name in (
            "request_digest",
            "title",
            "body",
            "marker",
            "auditor_block_digest",
            "expected_block_digest",
            "worker_id",
            "reason",
            "created_at",
            "updated_at",
            "publication_schema_version",
            "publication_kind",
            "publication_digest",
            "publication_key",
            "request_operation_id",
        ):
            if not isinstance(getattr(self, name), str):
                raise GitHubValidationError(f"outbox {name} must be text")
        if any(not isinstance(label, str) for label in self.labels):
            raise GitHubValidationError("outbox labels must be text")
        if self.issue is not None and not isinstance(self.issue, Mapping):
            raise GitHubValidationError("outbox issue must be an object")
        if self.comment is not None and not isinstance(self.comment, Mapping):
            raise GitHubValidationError("outbox comment must be an object")
        if not re.fullmatch(r"[0-9a-f]{64}", self.request_digest):
            raise GitHubValidationError("outbox request_digest must be a SHA-256 hex digest")
        for name in ("auditor_block_digest", "expected_block_digest"):
            value = getattr(self, name)
            if value and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise GitHubValidationError(f"outbox {name} must be a SHA-256 hex digest")
        if self.publication_digest and not re.fullmatch(r"[0-9a-f]{64}", self.publication_digest):
            raise GitHubValidationError("publication_digest must be a SHA-256 hex digest")
        if self.publication_key and not re.fullmatch(r"[0-9a-f]{64}", self.publication_key):
            raise GitHubValidationError("publication_key must be a SHA-256 hex digest")
        if self.publication_schema_version == GITHUB_PUBLICATION_SCHEMA_VERSION:
            if self.publication_kind == "legacy_body":
                raise GitHubValidationError("V1 publication cannot use the legacy body kind")
            if self.publication_revision is None:
                raise GitHubValidationError("V1 publication requires a revision")
            if not self.publication_digest or not self.publication_key:
                raise GitHubValidationError("V1 publication requires digest and semantic key")
        if self.marker != finding_marker(self.repository, self.finding_id):
            raise GitHubValidationError("outbox marker does not match its finding key")

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the uniqueness tuple required by the sync contract."""

        return self.repository, self.finding_id, self.operation_id

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON representation."""

        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "finding_id": self.finding_id,
            "operation_id": self.operation_id,
            "request_digest": self.request_digest,
            "title": self.title,
            "body": self.body,
            "marker": self.marker,
            "labels": list(self.labels),
            "auditor_block_digest": self.auditor_block_digest,
            "expected_block_digest": self.expected_block_digest,
            "finding_revision": self.finding_revision,
            "state": self.state,
            "attempts": self.attempts,
            "worker_id": self.worker_id,
            "issue": dict(self.issue) if self.issue is not None else None,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "publication_schema_version": self.publication_schema_version,
            "publication_kind": self.publication_kind,
            "publication_revision": self.publication_revision,
            "publication_digest": self.publication_digest,
            "publication_key": self.publication_key,
            "comment": dict(self.comment) if self.comment is not None else None,
            "request_operation_id": self.request_operation_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GitHubOutboxEntry:
        """Parse and validate an outbox record from canonical storage."""

        if not isinstance(payload, Mapping):
            raise GitHubOutboxError("outbox payload must be an object")
        labels = payload.get("labels", ())
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise GitHubOutboxError("outbox labels must be a list")
        return cls(
            schema_version=payload.get("schema_version", ""),
            repository=payload.get("repository", ""),
            finding_id=payload.get("finding_id", ""),
            operation_id=payload.get("operation_id", ""),
            request_digest=payload.get("request_digest", ""),
            title=payload.get("title", ""),
            body=payload.get("body", ""),
            marker=payload.get("marker", ""),
            labels=tuple(labels),
            auditor_block_digest=payload.get("auditor_block_digest", ""),
            expected_block_digest=payload.get("expected_block_digest", ""),
            finding_revision=payload.get("finding_revision"),
            state=payload.get("state", ""),
            attempts=payload.get("attempts", 0),
            worker_id=payload.get("worker_id", ""),
            issue=payload.get("issue"),
            reason=payload.get("reason", ""),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            publication_schema_version=payload.get("publication_schema_version", "github-publication.legacy"),
            publication_kind=payload.get("publication_kind", "legacy_body"),
            publication_revision=payload.get("publication_revision"),
            publication_digest=payload.get("publication_digest", ""),
            publication_key=payload.get("publication_key", ""),
            comment=payload.get("comment"),
            request_operation_id=payload.get("request_operation_id", ""),
        )


@dataclass(frozen=True, slots=True)
class GitHubFindingClaim:
    """Durable finding-wide ownership of one remote issue identity."""

    repository: str
    finding_id: str
    operation_id: str
    request_digest: str
    marker: str
    state: str = "in_flight"
    worker_id: str = ""
    finding_revision: int | None = None
    issue: Mapping[str, Any] | None = None
    reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: str = GITHUB_CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_repository(self.repository)
        _validate_finding_id(self.finding_id)
        _validate_operation_id(self.operation_id)
        if self.schema_version != GITHUB_CLAIM_SCHEMA_VERSION:
            raise GitHubValidationError(f"unsupported finding claim schema: {self.schema_version}")
        if self.state not in _CLAIM_STATES:
            raise GitHubValidationError(f"unknown finding claim state: {self.state}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.request_digest):
            raise GitHubValidationError("finding claim request_digest must be a SHA-256 hex digest")
        if self.marker != finding_marker(self.repository, self.finding_id):
            raise GitHubValidationError("finding claim marker does not match its finding key")
        if self.finding_revision is not None and (
            isinstance(self.finding_revision, bool)
            or not isinstance(self.finding_revision, int)
            or self.finding_revision < 0
        ):
            raise GitHubValidationError("finding claim revision must be non-negative")
        for name in ("worker_id", "reason", "created_at", "updated_at"):
            if not isinstance(getattr(self, name), str):
                raise GitHubValidationError(f"finding claim {name} must be text")
        if self.issue is not None and not isinstance(self.issue, Mapping):
            raise GitHubValidationError("finding claim issue must be an object")

    @property
    def key(self) -> tuple[str, str]:
        """Return the finding-wide uniqueness tuple."""

        return self.repository, self.finding_id

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON representation."""

        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "finding_id": self.finding_id,
            "operation_id": self.operation_id,
            "request_digest": self.request_digest,
            "marker": self.marker,
            "state": self.state,
            "worker_id": self.worker_id,
            "finding_revision": self.finding_revision,
            "issue": dict(self.issue) if self.issue is not None else None,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GitHubFindingClaim:
        """Parse and validate a finding-wide claim from canonical storage."""

        if not isinstance(payload, Mapping):
            raise GitHubOutboxError("finding claim payload must be an object")
        return cls(
            schema_version=payload.get("schema_version", ""),
            repository=payload.get("repository", ""),
            finding_id=payload.get("finding_id", ""),
            operation_id=payload.get("operation_id", ""),
            request_digest=payload.get("request_digest", ""),
            marker=payload.get("marker", ""),
            state=payload.get("state", ""),
            worker_id=payload.get("worker_id", ""),
            finding_revision=payload.get("finding_revision"),
            issue=payload.get("issue"),
            reason=payload.get("reason", ""),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
        )


@dataclass(frozen=True, slots=True)
class GitHubSyncResult:
    """Observable result of one idempotent synchronization attempt."""

    status: str
    repository: str
    finding_id: str
    operation_id: str
    issue: GitHubIssue | None = None
    outbox: GitHubOutboxEntry | None = None
    finding: Finding | None = None
    reason: str = ""
    replayed: bool = False
    remote_write: str = "none"
    comment: Mapping[str, Any] | None = None
    publication_kind: str = "legacy_body"
    publication_revision: int | None = None
    publication_digest: str = ""

    @property
    def ok(self) -> bool:
        """Return whether the remote operation and durable handoff completed."""

        return self.status in {"created", "updated", "unchanged", "commented", "reconciled"}

    def to_dict(self) -> dict[str, Any]:
        """Return a bounded JSON-safe sync result for service/MCP callers."""

        return {
            "status": self.status,
            "repository": self.repository,
            "finding_id": self.finding_id,
            "operation_id": self.operation_id,
            "issue": self.issue.to_dict() if self.issue is not None else None,
            "outbox": self.outbox.to_dict() if self.outbox is not None else None,
            "finding": record_to_dict(self.finding) if self.finding is not None else None,
            "reason": self.reason,
            "replayed": self.replayed,
            "remote_write": self.remote_write,
            "comment": dict(self.comment) if self.comment is not None else None,
            "publication_kind": self.publication_kind,
            "publication_revision": self.publication_revision,
            "publication_digest": self.publication_digest,
        }


class GitHubOutbox:
    """Persist GitHub sync entries through the canonical audit store.

    The outbox uses ``ActionRecord`` only as a typed projection row; the
    canonical NDJSON journal and its compare-and-swap remain owned by
    :class:`AuditStore`.  It therefore does not create a second general store.
    """

    RECORD_PREFIX = "github-outbox:"
    ACTION_TYPE = "github.sync.outbox"
    CLAIM_RECORD_PREFIX = "github-finding-claim:"
    CLAIM_ACTION_TYPE = "github.sync.finding_claim"
    ACTOR = {"kind": "agent", "id": "github-sync"}

    def __init__(self, store: AuditStore | str | Path):
        self._owns_store = not isinstance(store, AuditStore)
        self.store = store if isinstance(store, AuditStore) else AuditStore(store)

    def __enter__(self) -> GitHubOutbox:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    @classmethod
    def record_id(cls, repository: str, finding_id: str, operation_id: str) -> str:
        """Return a collision-resistant projection ID for the uniqueness tuple."""

        repository = _validate_repository(repository)
        finding_id = _validate_finding_id(finding_id)
        operation_id = _validate_operation_id(operation_id)
        digest = _sha256(
            {"repository": repository, "finding_id": finding_id, "operation_id": operation_id}
        )
        return f"{cls.RECORD_PREFIX}{digest}"

    def _stored(self, repository: str, finding_id: str, operation_id: str) -> StoredRecord | None:
        stored = self.store.get(self.record_id(repository, finding_id, operation_id))
        if stored is None:
            return None
        if (
            not isinstance(stored.record, ActionRecord)
            or stored.record.action_type != self.ACTION_TYPE
        ):
            raise GitHubOutboxError("outbox projection row has the wrong action type")
        try:
            entry = GitHubOutboxEntry.from_dict(stored.record.details)
        except (GitHubSyncError, TypeError, ValueError) as exc:
            raise GitHubOutboxError(f"outbox projection is malformed: {exc}") from exc
        if entry.key != (repository, finding_id, operation_id):
            raise GitHubOutboxError("outbox projection key does not match its lookup key")
        return stored

    def get(self, repository: str, finding_id: str, operation_id: str) -> GitHubOutboxEntry | None:
        """Load one durable entry, or ``None`` when it has not been enqueued."""

        stored = self._stored(repository, finding_id, operation_id)
        return None if stored is None else GitHubOutboxEntry.from_dict(stored.record.details)

    def find_publication(
        self,
        repository: str,
        finding_id: str,
        publication_key: str,
    ) -> GitHubOutboxEntry | None:
        """Find a durable publication by semantic identity, not caller operation ID.

        This scan deliberately uses the canonical projection.  It is bounded by
        the local store and makes repeated UI/CLI/MCP requests for one finding
        revision converge on the first durable operation.
        """

        _validate_repository(repository)
        _validate_finding_id(finding_id)
        if not re.fullmatch(r"[0-9a-f]{64}", publication_key):
            raise GitHubValidationError("publication_key must be a SHA-256 hex digest")
        for stored in self.store.list_records():
            if not isinstance(stored.record, ActionRecord) or stored.record.action_type != self.ACTION_TYPE:
                continue
            try:
                entry = GitHubOutboxEntry.from_dict(stored.record.details)
            except (GitHubSyncError, TypeError, ValueError) as exc:
                raise GitHubOutboxError(f"outbox projection is malformed: {exc}") from exc
            if (
                entry.repository == repository
                and entry.finding_id == finding_id
                and entry.publication_key == publication_key
            ):
                return entry
        return None

    def _get_with_revision(
        self, repository: str, finding_id: str, operation_id: str
    ) -> tuple[GitHubOutboxEntry, int] | None:
        stored = self._stored(repository, finding_id, operation_id)
        if stored is None:
            return None
        return GitHubOutboxEntry.from_dict(stored.record.details), stored.revision

    def put(self, entry: GitHubOutboxEntry, *, expected_revision: int) -> GitHubOutboxEntry:
        """CAS-write an entry and reload its projected value."""

        action = ActionRecord(
            action_id=self.record_id(*entry.key),
            action_type=self.ACTION_TYPE,
            actor_kind="agent",
            actor_id="github-sync",
            target_id=entry.finding_id,
            status=entry.state,
            details=entry.to_dict(),
        )
        operation_id = "github-outbox-write-" + _sha256(
            {"record": action.action_id, "entry": entry.to_dict(), "expected": expected_revision}
        )
        try:
            self.store.save(
                action,
                operation_id=operation_id,
                expected_revision=expected_revision,
                actor=self.ACTOR,
            )
        except (AuditConflictError, AuditStoreError) as exc:
            raise GitHubOutboxError(f"outbox CAS write failed: {exc}") from exc
        loaded = self._get_with_revision(*entry.key)
        if loaded is None:  # pragma: no cover - canonical store writes are synchronous.
            raise GitHubOutboxError("outbox CAS write disappeared before reload")
        return loaded[0]

    def enqueue(self, entry: GitHubOutboxEntry) -> GitHubOutboxEntry:
        """Insert an entry exactly once, checking request-digest idempotency."""

        current = self._get_with_revision(*entry.key)
        if current is not None:
            existing, _revision = current
            if existing.request_digest != entry.request_digest:
                raise GitHubConflictError(
                    "operation ID is already bound to a different rendered finding request"
                )
            return existing
        try:
            return self.put(entry, expected_revision=0)
        except GitHubOutboxError:
            current = self._get_with_revision(*entry.key)
            if current is not None and current[0].request_digest == entry.request_digest:
                return current[0]
            raise

    def update(
        self,
        entry: GitHubOutboxEntry,
        *,
        expected_revision: int,
    ) -> GitHubOutboxEntry:
        """Update an existing entry with compare-and-swap semantics."""

        current = self._get_with_revision(*entry.key)
        if current is None:
            raise GitHubOutboxError("cannot update a missing outbox entry")
        if current[1] != expected_revision:
            raise GitHubOutboxError(
                f"outbox revision conflict: expected {expected_revision}, current {current[1]}"
            )
        if current[0].request_digest != entry.request_digest:
            raise GitHubConflictError("outbox request digest changed during update")
        return self.put(entry, expected_revision=expected_revision)

    @classmethod
    def claim_record_id(cls, repository: str, finding_id: str) -> str:
        """Return the stable record ID that serializes all operations for a finding."""

        repository = _validate_repository(repository)
        finding_id = _validate_finding_id(finding_id)
        digest = _sha256({"repository": repository, "finding_id": finding_id})
        return f"{cls.CLAIM_RECORD_PREFIX}{digest}"

    def _claim_stored(self, repository: str, finding_id: str) -> StoredRecord | None:
        stored = self.store.get(self.claim_record_id(repository, finding_id))
        if stored is None:
            return None
        if (
            not isinstance(stored.record, ActionRecord)
            or stored.record.action_type != self.CLAIM_ACTION_TYPE
        ):
            raise GitHubOutboxError("finding claim projection row has the wrong action type")
        try:
            claim = GitHubFindingClaim.from_dict(stored.record.details)
        except (GitHubSyncError, TypeError, ValueError) as exc:
            raise GitHubOutboxError(f"finding claim projection is malformed: {exc}") from exc
        if claim.key != (repository, finding_id):
            raise GitHubOutboxError("finding claim key does not match its lookup key")
        return stored

    def get_claim(self, repository: str, finding_id: str) -> GitHubFindingClaim | None:
        """Load the one durable finding-wide claim, if present."""

        stored = self._claim_stored(repository, finding_id)
        return None if stored is None else GitHubFindingClaim.from_dict(stored.record.details)

    def _claim_with_revision(
        self, repository: str, finding_id: str
    ) -> tuple[GitHubFindingClaim, int] | None:
        stored = self._claim_stored(repository, finding_id)
        if stored is None:
            return None
        return GitHubFindingClaim.from_dict(stored.record.details), stored.revision

    def put_claim(self, claim: GitHubFindingClaim, *, expected_revision: int) -> GitHubFindingClaim:
        """CAS-write a finding-wide claim and reload its projected value."""

        action = ActionRecord(
            action_id=self.claim_record_id(*claim.key),
            action_type=self.CLAIM_ACTION_TYPE,
            actor_kind="agent",
            actor_id="github-sync",
            target_id=claim.finding_id,
            status=claim.state,
            details=claim.to_dict(),
        )
        operation_id = "github-finding-claim-write-" + _sha256(
            {"record": action.action_id, "claim": claim.to_dict(), "expected": expected_revision}
        )
        try:
            self.store.save(
                action,
                operation_id=operation_id,
                expected_revision=expected_revision,
                actor=self.ACTOR,
            )
        except (AuditConflictError, AuditStoreError) as exc:
            raise GitHubOutboxError(f"finding claim CAS write failed: {exc}") from exc
        loaded = self._claim_with_revision(*claim.key)
        if loaded is None:  # pragma: no cover - canonical store writes are synchronous.
            raise GitHubOutboxError("finding claim CAS write disappeared before reload")
        return loaded[0]

    def enqueue_claim(self, claim: GitHubFindingClaim) -> GitHubFindingClaim:
        """Insert a finding-wide claim exactly once, tolerating a CAS race."""

        current = self._claim_with_revision(*claim.key)
        if current is not None:
            return current[0]
        try:
            return self.put_claim(claim, expected_revision=0)
        except GitHubOutboxError:
            current = self._claim_with_revision(*claim.key)
            if current is not None:
                return current[0]
            raise

    def update_claim(
        self,
        claim: GitHubFindingClaim,
        *,
        expected_revision: int,
    ) -> GitHubFindingClaim:
        """Update a finding-wide claim with compare-and-swap semantics."""

        current = self._claim_with_revision(*claim.key)
        if current is None:
            raise GitHubOutboxError("cannot update a missing finding claim")
        if current[1] != expected_revision:
            raise GitHubOutboxError(
                f"finding claim revision conflict: expected {expected_revision}, current {current[1]}"
            )
        return self.put_claim(claim, expected_revision=expected_revision)

    def complete_success(
        self,
        entry: GitHubOutboxEntry,
        claim: GitHubFindingClaim,
    ) -> tuple[GitHubOutboxEntry, GitHubFindingClaim]:
        """Commit terminal outbox and finding-claim state in one transaction.

        The final operation receipt and its finding-wide ownership are one
        durable version boundary.  A process can still stop after the
        journal fsync, but reopening the store observes both terminal rows;
        it cannot leave a succeeded operation paired with an in-flight claim.
        """

        if entry.state != "succeeded" or claim.state != "succeeded":
            raise GitHubOutboxError("complete_success requires succeeded entry and claim")
        if entry.repository != claim.repository or entry.finding_id != claim.finding_id:
            raise GitHubConflictError("outbox entry and finding claim identify different findings")
        for _attempt in range(3):
            current_entry = self._get_with_revision(*entry.key)
            current_claim = self._claim_with_revision(*claim.key)
            if current_entry is None or current_claim is None:
                raise GitHubOutboxError("cannot complete a missing outbox entry or finding claim")
            stored_entry, entry_revision = current_entry
            stored_claim, claim_revision = current_claim
            if stored_entry.request_digest != entry.request_digest:
                raise GitHubConflictError("outbox request digest changed before completion")
            if stored_claim.request_digest != claim.request_digest:
                if stored_claim.state == "succeeded":
                    raise GitHubConflictError("a newer finding claim already succeeded")
                raise GitHubConflictError("finding claim request digest changed before completion")
            if stored_entry.state == "succeeded" and stored_claim.state == "succeeded":
                return stored_entry, stored_claim
            final_entry = replace(
                entry,
                worker_id=entry.worker_id or stored_entry.worker_id,
            )
            final_claim = replace(
                claim,
                worker_id=claim.worker_id or stored_claim.worker_id,
            )
            entry_action = ActionRecord(
                action_id=self.record_id(*final_entry.key),
                action_type=self.ACTION_TYPE,
                actor_kind="agent",
                actor_id="github-sync",
                target_id=final_entry.finding_id,
                status=final_entry.state,
                details=final_entry.to_dict(),
                created_at=final_entry.updated_at,
            )
            claim_action = ActionRecord(
                action_id=self.claim_record_id(*final_claim.key),
                action_type=self.CLAIM_ACTION_TYPE,
                actor_kind="agent",
                actor_id="github-sync",
                target_id=final_claim.finding_id,
                status=final_claim.state,
                details=final_claim.to_dict(),
                created_at=final_claim.updated_at,
            )
            operation_id = "github-sync-complete-" + _sha256(
                {
                    "entry": entry_action.details,
                    "claim": claim_action.details,
                }
            )
            try:
                self.store.commit(
                    [entry_action, claim_action],
                    operation_id=operation_id,
                    expected_revisions={
                        entry_action.action_id: entry_revision,
                        claim_action.action_id: claim_revision,
                    },
                    actor=self.ACTOR,
                )
            except (AuditConflictError, AuditStoreError):
                continue
            loaded_entry = self._get_with_revision(*entry.key)
            loaded_claim = self._claim_with_revision(*claim.key)
            if loaded_entry is not None and loaded_claim is not None:
                return loaded_entry[0], loaded_claim[0]
        raise GitHubOutboxError("outbox and finding claim completion CAS failed")


class GitHubSync:
    """Coordinate immutable issue publication and durable append-only revisions."""

    def __init__(
        self,
        provider: GitHubProvider,
        outbox: GitHubOutbox,
        *,
        finding_store: FindingStore | None = None,
        private_roots: Sequence[str | Path] = (),
        allowed_media_hosts: Sequence[str] = (),
        append_only: bool | None = None,
    ) -> None:
        self.provider = provider
        self.outbox = outbox
        self.finding_store = finding_store
        self.data_filter = PublicDataFilter(private_roots, allowed_media_hosts)
        detected = callable(getattr(provider, "append_auditor_comment", None))
        self.append_only = detected if append_only is None else bool(append_only)

    def sync(
        self,
        repository: str,
        finding: Finding,
        *,
        operation_id: str,
        evidence: FindingEvidence | Mapping[str, Any] | None = None,
        expected_finding_revision: int | None = None,
        retry_ambiguous: bool = False,
        worker_id: str = "",
    ) -> GitHubSyncResult:
        """Synchronize one finding without issuing an unbounded or blind retry.

        ``retry_ambiguous`` is intentionally opt-in.  A worker that reopens an
        ``in_flight`` or ``ambiguous`` outbox entry first performs another
        complete exact-marker search; an incomplete search never authorizes a
        second create.
        """

        repository = _validate_repository(repository)
        if not isinstance(finding, Finding):
            raise GitHubValidationError("finding must be a Finding contract")
        operation_id = _validate_operation_id(operation_id)
        expected_finding_revision = self._preflight_finding_revision(
            finding, expected_finding_revision
        )
        worker_id = worker_id.strip() or f"worker-{uuid.uuid4().hex[:12]}"
        rendered = render_finding_issue(
            finding,
            repository=repository,
            evidence=evidence,
            data_filter=self.data_filter,
        )
        if self.append_only:
            return self._sync_append_only(
                repository,
                finding,
                rendered,
                operation_id=operation_id,
                expected_finding_revision=expected_finding_revision,
                retry_ambiguous=retry_ambiguous,
                worker_id=worker_id,
            )
        entry = GitHubOutboxEntry(
            repository=repository,
            finding_id=finding.finding_id,
            operation_id=operation_id,
            request_digest=rendered.request_digest,
            title=rendered.title,
            body=rendered.body,
            marker=rendered.marker,
            labels=rendered.labels,
            auditor_block_digest=rendered.auditor_block_digest,
            finding_revision=expected_finding_revision,
        )
        entry = self.outbox.enqueue(entry)
        # ``remote_write`` is an explicit provider-boundary outcome.  A
        # terminal ``conflict``/``reconciled`` status alone cannot tell the
        # service whether a reserved issue-write unit was consumed.
        remote_write = "none"

        if entry.state == "succeeded":
            issue = _issue_from_entry(entry)
            if issue is None:
                raise GitHubOutboxError("succeeded outbox entry has no valid remote issue")
            self._repair_succeeded_claim(entry, issue)
            updated_finding = self._persist_link(
                finding,
                issue,
                entry,
                expected_finding_revision=expected_finding_revision,
            )
            return GitHubSyncResult(
                "unchanged",
                repository,
                finding.finding_id,
                operation_id,
                issue,
                entry,
                updated_finding,
                "idempotent outbox replay",
                replayed=True,
            )

        linked = _linked_issue_identity(finding, repository, rendered.marker)
        search, search_error = self._search(repository, rendered.marker)
        if search_error is not None:
            entry = self._mark(entry, "failed", search_error)
            return self._result("failed", repository, finding, entry, reason=search_error)
        if search is None:  # pragma: no cover - guarded by _search.
            raise GitHubTransportError("provider search returned no result")
        if not search.complete:
            reason = (
                search.reason or "marker search is incomplete; synchronization is not authorized"
            )
            entry = self._mark(entry, "ambiguous", reason)
            return self._result("ambiguous", repository, finding, entry, reason=reason)
        remote = _select_marker_issue(search, repository=repository, finding_id=finding.finding_id)

        if linked is not None:
            if remote is not None and remote.number != linked["number"]:
                entry = self._mark(
                    entry, "conflict", "durable finding link conflicts with marker search"
                )
                return self._result("conflict", repository, finding, entry, reason=entry.reason)
            if remote is None:
                try:
                    remote = _coerce_issue(self.provider.get_issue(repository, linked["number"]))
                except Exception as exc:  # noqa: BLE001 - transport boundary is fail-closed.
                    reason = f"linked issue could not be read: {type(exc).__name__}: {exc}"
                    entry = self._mark(entry, "failed", reason)
                    return self._result("failed", repository, finding, entry, reason=reason)
                _validate_issue_for_finding(remote, repository, finding.finding_id, rendered.marker)

        claim: GitHubFindingClaim | None = None
        claim_expected_block_digest = (
            entry.expected_block_digest or _link_block_digest(finding.github_issue) or ""
        )
        if remote is None:
            if entry.state in {"in_flight", "ambiguous"} and not retry_ambiguous:
                reason = (
                    "create remains ambiguous; re-search succeeded but retry was not authorized"
                )
                entry = self._mark(entry, "ambiguous", reason)
                return self._result("ambiguous", repository, finding, entry, reason=reason)
            self._preflight_finding_revision(finding, expected_finding_revision)
            claim, claim_owned = self._acquire_finding_claim(
                rendered,
                operation_id=operation_id,
                finding_revision=expected_finding_revision,
                worker_id=worker_id,
                issue=None,
                retry_ambiguous=retry_ambiguous,
            )
            if not claim_owned:
                if claim.state == "succeeded" and claim.issue is not None:
                    if claim.request_digest != rendered.request_digest:
                        reason = "a newer succeeded finding claim has a different rendered request"
                        entry = self._mark(entry, "conflict", reason)
                        return self._result("conflict", repository, finding, entry, reason=reason)
                    try:
                        claimed_snapshot = _issue_from_claim(claim)
                        remote = self._refresh_claim_issue(claim, repository, finding.finding_id)
                        if claimed_snapshot is not None:
                            self._reject_untracked_snapshot_change(
                                claimed_snapshot,
                                remote,
                                expected_block_digest=claim_expected_block_digest,
                            )
                    except GitHubIssueMissing:
                        claim, claim_owned = self._recover_missing_claim(
                            claim,
                            rendered,
                            operation_id=operation_id,
                            finding_revision=expected_finding_revision,
                            worker_id=worker_id,
                        )
                    except GitHubConflictError as exc:
                        reason = str(exc)
                        self._mark_claim(claim, "conflict", reason)
                        entry = self._mark(entry, "conflict", reason)
                        return self._result("conflict", repository, finding, entry, reason=reason)
                    except GitHubTransportError as exc:
                        reason = str(exc)
                        self._mark_claim(claim, "ambiguous", reason)
                        entry = self._mark(entry, "ambiguous", reason)
                        return self._result("ambiguous", repository, finding, entry, reason=reason)
                    else:
                        status = "unchanged"
                if remote is None and not claim_owned:
                    status = "ambiguous" if claim.state == "ambiguous" else "pending"
                    reason = claim.reason or "another worker owns the finding-wide issue claim"
                    entry = self._mark(entry, status, reason)
                    return self._result(status, repository, finding, entry, reason=reason)
            if remote is None and claim_owned:
                claimed_issue = _issue_from_claim(claim)
                if claimed_issue is not None:
                    try:
                        remote = self._refresh_claim_issue(claim, repository, finding.finding_id)
                        self._reject_untracked_snapshot_change(
                            claimed_issue,
                            remote,
                            expected_block_digest=claim_expected_block_digest,
                        )
                    except GitHubIssueMissing:
                        claim, claim_owned = self._recover_missing_claim(
                            claim,
                            rendered,
                            operation_id=operation_id,
                            finding_revision=expected_finding_revision,
                            worker_id=worker_id,
                        )
                    except GitHubConflictError as exc:
                        reason = str(exc)
                        self._mark_claim(claim, "conflict", reason)
                        entry = self._mark(entry, "conflict", reason)
                        return self._result("conflict", repository, finding, entry, reason=reason)
                    except GitHubTransportError as exc:
                        reason = str(exc)
                        self._mark_claim(claim, "ambiguous", reason)
                        entry = self._mark(entry, "ambiguous", reason)
                        return self._result("ambiguous", repository, finding, entry, reason=reason)
                    else:
                        status = "unchanged"
                if remote is None and claim_owned:
                    self._preflight_finding_revision(finding, expected_finding_revision)
                    claimed = self._claim(entry, worker_id)
                    if claimed.worker_id != worker_id and claimed.state == "in_flight":
                        reason = "another worker owns the in-flight issue create"
                        return self._result("pending", repository, finding, claimed, reason=reason)
                    entry = claimed
                    if entry.state == "succeeded":
                        remote = _issue_from_entry(entry)
                        if remote is None:
                            raise GitHubOutboxError(
                                "succeeded outbox entry has no valid remote issue after claim"
                            )
                        status = "unchanged"
                    else:
                        try:
                            create_returned = False
                            remote = self._create_issue(
                                rendered,
                                finding_revision=expected_finding_revision,
                            )
                            create_returned = True
                            remote_write = "applied"
                            _validate_issue_for_finding(
                                remote, repository, finding.finding_id, rendered.marker
                            )
                        except GitHubConflictError as exc:
                            reason = str(exc)
                            if not create_returned:
                                remote_write = self._create_failure_write_outcome(
                                    rendered,
                                    conflict_hint=True,
                                )
                            entry = self._mark(entry, "conflict", reason)
                            self._mark_claim(claim, "conflict", reason)
                            return self._result(
                                "conflict",
                                repository,
                                finding,
                                entry,
                                reason=reason,
                                remote_write=remote_write,
                            )
                        except Exception as exc:  # noqa: BLE001 - create outcome is inherently ambiguous.
                            reconciled, reconcile_error = self._search(repository, rendered.marker)
                            if (
                                reconcile_error is None
                                and reconciled is not None
                                and reconciled.complete
                            ):
                                remote = _select_marker_issue(
                                    reconciled,
                                    repository=repository,
                                    finding_id=finding.finding_id,
                                )
                            else:
                                remote = None
                            if remote is None:
                                if reconciled is not None and not reconciled.complete:
                                    reconcile_reason = (
                                        reconciled.reason
                                        or "create reconciliation search is incomplete"
                                    )
                                elif reconcile_error is not None:
                                    reconcile_reason = reconcile_error
                                else:
                                    reconcile_reason = "no exact marker was confirmed"
                                reason = (
                                    f"create outcome is ambiguous: {type(exc).__name__}: {exc}; "
                                    f"{reconcile_reason}"
                                )
                                remote_write = "ambiguous"
                                entry = self._mark(entry, "ambiguous", reason)
                                self._mark_claim(claim, "ambiguous", reason)
                                return self._result(
                                    "ambiguous",
                                    repository,
                                    finding,
                                    entry,
                                    reason=reason,
                                    remote_write=remote_write,
                                )
                            remote_write = "applied"
                            status = "reconciled"
                        else:
                            status = "created"
                        try:
                            observed_digest = _sha256_text(_extract_auditor_block(remote.body))
                        except GitHubSyncError as exc:
                            reason = f"created issue auditor block is invalid: {exc}"
                            entry = self._mark(entry, "conflict", reason)
                            self._mark_claim(claim, "conflict", reason)
                            return self._result(
                                "conflict",
                                repository,
                                finding,
                                entry,
                                reason=reason,
                                remote_write=remote_write,
                            )
                        if observed_digest != rendered.auditor_block_digest:
                            reason = (
                                "remote auditor block changed during create reconciliation; "
                                "human conflict retained"
                            )
                            entry = self._mark(entry, "conflict", reason)
                            self._mark_claim(claim, "conflict", reason)
                            return self._result(
                                "conflict",
                                repository,
                                finding,
                                entry,
                                reason=reason,
                                remote_write=remote_write,
                            )
                        entry = replace(
                            entry,
                            issue=remote.to_dict(),
                            state="in_flight",
                            expected_block_digest=rendered.auditor_block_digest,
                            reason="",
                        )
                        entry = self._persist_entry(entry)
        else:
            status = "unchanged"
            self._preflight_finding_revision(finding, expected_finding_revision)
            claim, claim_owned = self._acquire_finding_claim(
                rendered,
                operation_id=operation_id,
                finding_revision=expected_finding_revision,
                worker_id=worker_id,
                issue=remote,
                retry_ambiguous=retry_ambiguous,
            )
            if not claim_owned:
                if claim.state == "succeeded":
                    if claim.request_digest != rendered.request_digest:
                        reason = "a newer succeeded finding claim has a different rendered request"
                        entry = self._mark(entry, "conflict", reason)
                        return self._result("conflict", repository, finding, entry, reason=reason)
                    status = "unchanged"
                    claim_status = "pending"
                else:
                    claim_status = "ambiguous" if claim.state == "ambiguous" else "pending"
                reason = claim.reason or "another worker owns the finding-wide issue claim"
                if claim.state != "succeeded":
                    entry = self._mark(entry, claim_status, reason)
                    return self._result(claim_status, repository, finding, entry, reason=reason)

        _validate_issue_for_finding(remote, repository, finding.finding_id, rendered.marker)
        expected_block_digest = (
            entry.expected_block_digest or _link_block_digest(finding.github_issue) or ""
        )
        if status != "created":
            try:
                remote = self._refresh_remote_before_update(
                    remote,
                    repository=repository,
                    finding_id=finding.finding_id,
                    marker=rendered.marker,
                    expected_block_digest=expected_block_digest,
                )
            except GitHubIssueMissing as exc:
                reason = str(exc)
                if claim is not None:
                    self._mark_claim(claim, "failed", reason)
                entry = self._mark(entry, "conflict", reason)
                return self._result("conflict", repository, finding, entry, reason=reason)
            except GitHubConflictError as exc:
                reason = str(exc)
                if claim is not None:
                    self._mark_claim(claim, "conflict", reason)
                entry = self._mark(entry, "conflict", reason)
                return self._result("conflict", repository, finding, entry, reason=reason)
            except GitHubValidationError as exc:
                reason = str(exc)
                if claim is not None:
                    self._mark_claim(claim, "conflict", reason)
                entry = self._mark(entry, "conflict", reason)
                return self._result("conflict", repository, finding, entry, reason=reason)
            except GitHubTransportError as exc:
                reason = str(exc)
                if claim is not None:
                    self._mark_claim(claim, "ambiguous", reason)
                entry = self._mark(entry, "ambiguous", reason)
                return self._result("ambiguous", repository, finding, entry, reason=reason)
        reconciled_update = False
        if entry.state in {"ambiguous", "in_flight"} and not expected_block_digest:
            try:
                observed_digest = _sha256_text(_extract_auditor_block(remote.body))
            except GitHubSyncError as exc:
                reason = f"ambiguous issue auditor block is invalid: {exc}"
                entry = self._mark(entry, "conflict", reason)
                if claim is not None:
                    self._mark_claim(claim, "conflict", reason)
                return self._result("conflict", repository, finding, entry, reason=reason)
            if observed_digest != rendered.auditor_block_digest:
                reason = (
                    "ambiguous issue auditor block changed before retry; human conflict retained"
                )
                entry = self._mark(entry, "conflict", reason)
                if claim is not None:
                    self._mark_claim(claim, "conflict", reason)
                return self._result("conflict", repository, finding, entry, reason=reason)
        try:
            target_body = _replace_auditor_block(
                remote,
                target_block=rendered.auditor_block,
                expected_block_digest=expected_block_digest,
            )
        except GitHubConflictError as exc:
            if entry.state in {"ambiguous", "in_flight"}:
                try:
                    already_applied = replace_auditor_block(
                        remote.body,
                        rendered.auditor_block,
                    )
                except GitHubSyncError:
                    already_applied = ""
                if already_applied == remote.body:
                    target_body = remote.body
                    reconciled_update = True
                    status = "reconciled"
                elif entry.state == "ambiguous" and not retry_ambiguous:
                    reason = "update remains ambiguous; retry was not authorized"
                    entry = self._mark(entry, "ambiguous", reason)
                    if claim is not None:
                        self._mark_claim(claim, "ambiguous", reason)
                    return self._result("ambiguous", repository, finding, entry, reason=reason)
                elif entry.state == "ambiguous" and not expected_block_digest:
                    reason = "ambiguous update has no prior auditor-block digest"
                    entry = self._mark(entry, "conflict", reason)
                    if claim is not None:
                        self._mark_claim(claim, "conflict", reason)
                    return self._result("conflict", repository, finding, entry, reason=reason)
                else:
                    entry = self._mark(entry, "conflict", str(exc))
                    if claim is not None:
                        self._mark_claim(claim, "conflict", str(exc))
                    return self._result("conflict", repository, finding, entry, reason=str(exc))
            else:
                entry = self._mark(entry, "conflict", str(exc))
                if claim is not None:
                    self._mark_claim(claim, "conflict", str(exc))
                return self._result("conflict", repository, finding, entry, reason=str(exc))
        changed = target_body != remote.body
        if changed:
            if claim is not None and not claim_owned:
                reason = "another succeeded finding claim owns this issue; refusing a remote update"
                entry = self._mark(entry, "pending", reason)
                return self._result("pending", repository, finding, entry, reason=reason)
            if entry.worker_id and entry.worker_id != worker_id and entry.state == "in_flight":
                reason = "another worker owns the in-flight issue update"
                entry = self._mark(entry, "in_flight", reason)
                return self._result("pending", repository, finding, entry, reason=reason)
            claimed = self._claim(entry, worker_id)
            if claimed.worker_id != worker_id and claimed.state == "in_flight":
                reason = "another worker owns the in-flight issue update"
                return self._result("pending", repository, finding, claimed, reason=reason)
            entry = claimed
            if entry.state == "succeeded":
                remote = _issue_from_entry(entry)
                if remote is None:
                    raise GitHubOutboxError(
                        "succeeded outbox entry has no valid remote issue before update"
                    )
                changed = False
            else:
                if not entry.expected_block_digest:
                    try:
                        expected_block_digest = _sha256_text(_extract_auditor_block(remote.body))
                    except GitHubSyncError as exc:
                        reason = f"current auditor block cannot be guarded: {exc}"
                        entry = self._mark(entry, "conflict", reason)
                        if claim is not None:
                            self._mark_claim(claim, "conflict", reason)
                        return self._result("conflict", repository, finding, entry, reason=reason)
                    entry = replace(
                        entry,
                        expected_block_digest=expected_block_digest,
                        updated_at=utc_now(),
                    )
                    entry = self._persist_entry(entry)
                try:
                    update_returned = False
                    updated_remote = self._update_issue(
                        remote,
                        rendered=rendered,
                        finding_revision=expected_finding_revision,
                        body=target_body,
                    )
                    update_returned = True
                    remote_write = "applied"
                    _validate_issue_for_finding(
                        updated_remote, repository, finding.finding_id, rendered.marker
                    )
                    if updated_remote.body != target_body:
                        raise GitHubValidationError(
                            "provider update response body does not match request"
                        )
                    if updated_remote.labels != remote.labels:
                        raise GitHubValidationError("provider update changed unrelated labels")
                    if (
                        updated_remote.number != remote.number
                        or updated_remote.url != remote.url
                        or updated_remote.title != remote.title
                        or updated_remote.state != remote.state
                        or updated_remote.comments != remote.comments
                    ):
                        raise GitHubValidationError(
                            "provider update changed unrelated issue fields"
                        )
                    remote = updated_remote
                    status = "updated"
                except (GitHubConflictError, GitHubIssueMissing) as exc:
                    reason = str(exc)
                    remote_write = (
                        "applied"
                        if update_returned
                        else self._update_failure_write_outcome(
                            remote,
                            target_body,
                            conflict_hint=True,
                        )
                    )
                    entry = self._mark(entry, "conflict", reason)
                    if claim is not None:
                        claim_state = (
                            "failed" if isinstance(exc, GitHubIssueMissing) else "conflict"
                        )
                        self._mark_claim(claim, claim_state, reason)
                    return self._result(
                        "conflict",
                        repository,
                        finding,
                        entry,
                        reason=reason,
                        remote_write=remote_write,
                    )
                except Exception as exc:  # noqa: BLE001 - update may have applied remotely.
                    reason = f"update outcome is ambiguous: {type(exc).__name__}: {exc}"
                    remote_write = (
                        "applied"
                        if update_returned
                        else self._update_failure_write_outcome(
                            remote,
                            target_body,
                            conflict_hint=False,
                        )
                    )
                    entry = self._mark(entry, "ambiguous", reason)
                    if claim is not None:
                        self._mark_claim(claim, "ambiguous", reason)
                    return self._result(
                        "ambiguous",
                        repository,
                        finding,
                        entry,
                        reason=reason,
                        remote_write=remote_write,
                    )
        elif status not in {"reconciled", "created"} and not reconciled_update:
            status = "unchanged"

        link = _build_link(remote, finding, rendered)
        updated_finding = self._persist_link(
            finding,
            remote,
            entry,
            expected_finding_revision=expected_finding_revision,
            link=link,
        )
        entry = replace(
            entry,
            state="succeeded",
            issue=remote.to_dict(),
            expected_block_digest=rendered.auditor_block_digest,
            auditor_block_digest=rendered.auditor_block_digest,
            worker_id=worker_id,
            reason="",
        )
        if claim is not None:
            claim = replace(
                claim,
                state="succeeded",
                issue=remote.to_dict(),
                worker_id=worker_id,
                reason="",
                updated_at=utc_now(),
            )
            entry, claim = self.outbox.complete_success(entry, claim)
        else:
            entry = self._persist_entry(entry)
        return GitHubSyncResult(
            status,
            repository,
            finding.finding_id,
            operation_id,
            remote,
            entry,
            updated_finding,
            remote_write=remote_write,
        )

    def _sync_append_only(
        self,
        repository: str,
        finding: Finding,
        rendered: RenderedFinding,
        *,
        operation_id: str,
        expected_finding_revision: int | None,
        retry_ambiguous: bool,
        worker_id: str,
    ) -> GitHubSyncResult:
        """Run the D1 create/link-once plus append-only revision protocol."""

        revision = 0 if expected_finding_revision is None else expected_finding_revision
        initial_body = _render_initial_issue_body(rendered, revision)
        initial_digest = _publication_digest(
            rendered,
            finding_revision=revision,
            publication_kind="initial_issue",
            body=initial_body,
        )
        initial_key = _publication_key(
            repository,
            finding.finding_id,
            publication_kind="initial_issue",
            finding_revision=revision,
            publication_digest=initial_digest,
        )
        entry = self.outbox.find_publication(repository, finding.finding_id, initial_key)
        if entry is None:
            entry = self.outbox.enqueue(
                self._publication_entry(
                    rendered,
                    operation_id=operation_id,
                    finding_revision=revision,
                    publication_kind="initial_issue",
                    publication_digest=initial_digest,
                    publication_key=initial_key,
                    body=initial_body,
                )
            )
        if entry.state == "succeeded" and entry.issue is not None:
            issue = _issue_from_entry(entry)
            if issue is None:
                raise GitHubOutboxError("succeeded append-only entry has an invalid issue")
            try:
                updated_finding = self._persist_link(
                    finding,
                    issue,
                    entry,
                    expected_finding_revision=expected_finding_revision,
                    link=_build_link(issue, finding, rendered, entry=entry),
                )
            except GitHubConflictError as exc:
                return self._append_result(
                    "conflict",
                    repository,
                    finding,
                    operation_id,
                    entry,
                    issue,
                    finding,
                    reason=f"durable finding link remains unresolved: {exc}",
                    replayed=True,
                )
            return self._append_result(
                "unchanged",
                repository,
                finding,
                operation_id,
                entry,
                issue,
                updated_finding,
                replayed=True,
            )

        search, search_error = self._search(repository, rendered.marker)
        if search_error is not None:
            entry = self._mark(entry, "failed", search_error)
            return self._append_result("failed", repository, finding, operation_id, entry, None, finding, reason=search_error)
        if search is None or not search.complete:
            reason = (search.reason if search is not None else "") or "marker search is incomplete"
            entry = self._mark(entry, "ambiguous", reason)
            return self._append_result("ambiguous", repository, finding, operation_id, entry, None, finding, reason=reason)
        remote = _select_marker_issue(search, repository=repository, finding_id=finding.finding_id)

        if remote is None:
            if entry.state in {"in_flight", "ambiguous"} and not retry_ambiguous:
                reason = "create remains ambiguous; exact marker is absent and retry was not authorized"
                entry = self._mark(entry, "ambiguous", reason)
                return self._append_result("ambiguous", repository, finding, operation_id, entry, None, finding, reason=reason)
            claim, claim_owned = self._acquire_finding_claim(
                rendered,
                operation_id=entry.operation_id,
                finding_revision=revision,
                worker_id=worker_id,
                issue=None,
                retry_ambiguous=retry_ambiguous,
            )
            if not claim_owned:
                if claim.state == "succeeded":
                    remote = self._refresh_claim_issue(claim, repository, finding.finding_id)
                else:
                    reason = claim.reason or "another worker owns the finding publication"
                    entry = self._mark(entry, "ambiguous" if claim.state == "ambiguous" else "pending", reason)
                    return self._append_result("ambiguous" if claim.state == "ambiguous" else "pending", repository, finding, operation_id, entry, None, finding, reason=reason)
            if remote is None and claim_owned:
                entry = self._claim(entry, worker_id)
                try:
                    remote = _coerce_issue(
                        self.provider.create_issue(
                            repository,
                            title=rendered.title,
                            body=initial_body,
                            labels=rendered.labels,
                        )
                    )
                    _validate_issue_for_finding(remote, repository, finding.finding_id, rendered.marker)
                    remote_write = "applied"
                    status = "created"
                except Exception as exc:  # noqa: BLE001 - remote create is ambiguous by definition.
                    reconciled, reconcile_error = self._search(repository, rendered.marker)
                    remote = (
                        _select_marker_issue(reconciled, repository=repository, finding_id=finding.finding_id)
                        if reconciled is not None and reconcile_error is None and reconciled.complete
                        else None
                    )
                    if remote is None:
                        reason = f"create outcome is ambiguous: {type(exc).__name__}: {exc}"
                        entry = self._mark(entry, "ambiguous", reason)
                        self._mark_claim(claim, "ambiguous", reason)
                        return self._append_result("ambiguous", repository, finding, operation_id, entry, None, finding, reason=reason, remote_write="ambiguous")
                    status = "reconciled"
                    remote_write = "none"
                entry = replace(entry, issue=remote.to_dict(), state="succeeded", reason="", updated_at=utc_now())
                # Persist the remote observation before attempting the optional
                # canonical FindingStore link CAS.  A link conflict must not
                # erase evidence that GitHub accepted the immutable issue.
                entry = self._persist_entry(entry)
                try:
                    updated_finding = self._persist_link(
                        finding,
                        remote,
                        entry,
                        expected_finding_revision=expected_finding_revision,
                        link=_build_link(remote, finding, rendered, entry=entry),
                    )
                except GitHubConflictError as exc:
                    reason = f"durable finding link remains unresolved: {exc}"
                    self._mark_claim(claim, "conflict", reason)
                    return self._append_result(
                        "conflict",
                        repository,
                        finding,
                        operation_id,
                        entry,
                        remote,
                        finding,
                        reason=reason,
                        remote_write=remote_write,
                    )
                succeeded_claim = replace(
                    claim,
                    state="succeeded",
                    issue=remote.to_dict(),
                    worker_id=worker_id,
                    reason="",
                    updated_at=utc_now(),
                )
                try:
                    entry, _ = self.outbox.complete_success(entry, succeeded_claim)
                except GitHubOutboxError:
                    # The remote issue and publication entry are already
                    # durable; preserve the observation if the paired claim
                    # boundary races with another worker.
                    entry = self._persist_entry(entry)
                return self._append_result(status, repository, finding, operation_id, entry, remote, updated_finding, remote_write=remote_write)

        # A marker found remotely settles the immutable initial publication,
        # even when the local outbox was lost or this is a migration from the
        # legacy body-CAS path.  Never rewrite the issue body.
        if entry.state != "succeeded" or entry.issue is None:
            entry = self._persist_entry(
                replace(
                    entry,
                    state="succeeded",
                    issue=remote.to_dict(),
                    reason="",
                    updated_at=utc_now(),
                )
            )
        # A crash may have left the finding-wide claim in flight even though
        # the exact immutable issue is now visible. Reconcile that local
        # ownership boundary before considering later revisions.
        self._repair_succeeded_claim(entry, remote)

        _validate_issue_for_finding(remote, repository, finding.finding_id, rendered.marker)
        comment_digest = _publication_digest(
            rendered, finding_revision=revision, publication_kind="revision_comment"
        )
        comment_key = _publication_key(
            repository,
            finding.finding_id,
            publication_kind="revision_comment",
            finding_revision=revision,
            publication_digest=comment_digest,
        )
        comment_entry = self.outbox.find_publication(repository, finding.finding_id, comment_key)
        comment_body = _render_publication_comment(finding, rendered, revision, comment_digest)
        if comment_entry is None:
            comment_entry = self.outbox.enqueue(
                self._publication_entry(
                    rendered,
                    operation_id=operation_id,
                    finding_revision=revision,
                    publication_kind="revision_comment",
                    publication_digest=comment_digest,
                    publication_key=comment_key,
                    body=comment_body,
                )
            )
        if comment_entry.state == "succeeded":
            try:
                updated_finding = self._persist_link(
                    finding,
                    remote,
                    comment_entry,
                    expected_finding_revision=expected_finding_revision,
                    link=_build_link(remote, finding, rendered, entry=comment_entry),
                )
            except GitHubConflictError as exc:
                return self._append_result(
                    "conflict",
                    repository,
                    finding,
                    operation_id,
                    comment_entry,
                    remote,
                    finding,
                    reason=f"durable finding link remains unresolved: {exc}",
                    replayed=True,
                )
            return self._append_result(
                "unchanged",
                repository,
                finding,
                operation_id,
                comment_entry,
                remote,
                updated_finding,
                replayed=True,
            )
        claimed_comment = self._claim(comment_entry, worker_id)
        if claimed_comment.worker_id != worker_id and claimed_comment.state == "in_flight":
            reason = "another worker owns the in-flight auditor comment"
            return self._append_result("pending", repository, finding, operation_id, claimed_comment, remote, finding, reason=reason)
        try:
            write = self.provider.append_auditor_comment(
                repository,
                remote.number,
                body=comment_body,
                request_digest=comment_digest,
            )
            comment, write_status = _coerce_comment_write(write, publication_digest=comment_digest)
            remote_write = "applied" if write_status == "created" else "none"
            status = "commented" if write_status == "created" else "reconciled"
        except Exception as exc:  # noqa: BLE001 - preserve unresolved comment ambiguity.
            comment = self._reconcile_comment(remote, comment_digest)
            if comment is None:
                reason = f"comment outcome is ambiguous: {type(exc).__name__}: {exc}"
                comment_entry = self._mark(claimed_comment, "ambiguous", reason)
                return self._append_result("ambiguous", repository, finding, operation_id, comment_entry, remote, finding, reason=reason, remote_write="ambiguous")
            write_status = "reconciled"
            remote_write = "none"
            status = "reconciled"
        comment_entry = replace(
            claimed_comment,
            state="succeeded",
            issue=remote.to_dict(),
            comment=comment,
            reason="",
            updated_at=utc_now(),
        )
        comment_entry = self._persist_entry(comment_entry)
        try:
            updated_finding = self._persist_link(
                finding,
                remote,
                comment_entry,
                expected_finding_revision=expected_finding_revision,
                link=_build_link(remote, finding, rendered, entry=comment_entry),
            )
        except GitHubConflictError as exc:
            reason = f"durable finding link remains unresolved: {exc}"
            return self._append_result(
                "conflict",
                repository,
                finding,
                operation_id,
                comment_entry,
                remote,
                finding,
                reason=reason,
                remote_write=remote_write,
            )
        return self._append_result(status, repository, finding, operation_id, comment_entry, remote, updated_finding, remote_write=remote_write)

    def _publication_entry(
        self,
        rendered: RenderedFinding,
        *,
        operation_id: str,
        finding_revision: int | None,
        publication_kind: str,
        publication_digest: str,
        publication_key: str,
        body: str,
    ) -> GitHubOutboxEntry:
        """Build a semantic publication entry before any remote mutation."""

        return GitHubOutboxEntry(
            repository=rendered.repository,
            finding_id=rendered.finding_id,
            operation_id=f"publication-{publication_key}",
            request_operation_id=operation_id,
            request_digest=_sha256(
                {
                    "schema_version": GITHUB_PUBLICATION_SCHEMA_VERSION,
                    "publication_key": publication_key,
                    "body": body,
                    "labels": rendered.labels,
                }
            ),
            title=rendered.title,
            body=body,
            marker=rendered.marker,
            labels=rendered.labels,
            auditor_block_digest=rendered.auditor_block_digest,
            finding_revision=finding_revision,
            publication_schema_version=GITHUB_PUBLICATION_SCHEMA_VERSION,
            publication_kind=publication_kind,
            publication_revision=finding_revision,
            publication_digest=publication_digest,
            publication_key=publication_key,
        )

    def _append_result(
        self,
        status: str,
        repository: str,
        finding: Finding,
        operation_id: str,
        entry: GitHubOutboxEntry,
        issue: GitHubIssue | None,
        updated_finding: Finding | None,
        *,
        reason: str = "",
        replayed: bool = False,
        remote_write: str = "none",
    ) -> GitHubSyncResult:
        return GitHubSyncResult(
            status=status,
            repository=repository,
            finding_id=finding.finding_id,
            operation_id=operation_id,
            issue=issue or _issue_from_entry(entry),
            outbox=entry,
            finding=updated_finding,
            reason=reason,
            replayed=replayed,
            remote_write=remote_write,
            comment=entry.comment,
            publication_kind=entry.publication_kind,
            publication_revision=entry.publication_revision,
            publication_digest=entry.publication_digest,
        )

    def _reconcile_comment(
        self, remote: GitHubIssue, publication_digest: str
    ) -> Mapping[str, Any] | None:
        """Read the complete issue comment collection after an ambiguous POST."""

        try:
            refreshed = _coerce_issue(self.provider.get_issue(remote.repository, remote.number))
        except Exception:  # noqa: BLE001 - unreadable remote state remains ambiguous.
            return None
        marker = _publication_request_marker(publication_digest)
        matches = []
        for comment in refreshed.comments:
            body = comment.get("body")
            if isinstance(body, str) and marker in body:
                matches.append(comment)
        if len(matches) != 1:
            return None
        return dict(matches[0])

    def _preflight_finding_revision(
        self,
        finding: Finding,
        expected_finding_revision: int | None,
    ) -> int | None:
        """Verify canonical identity and return the revision reserved by this call."""

        if expected_finding_revision is not None and (
            isinstance(expected_finding_revision, bool)
            or not isinstance(expected_finding_revision, int)
            or expected_finding_revision < 0
        ):
            raise GitHubValidationError("expected_finding_revision must be non-negative")
        if self.finding_store is None:
            if expected_finding_revision is not None:
                raise GitHubConflictError(
                    "expected_finding_revision requires a canonical FindingStore adapter"
                )
            return None
        store = getattr(self.finding_store, "store", None)
        if store is None:
            raise GitHubConflictError("finding-store adapter does not expose canonical AuditStore")
        stored = store.get(finding.finding_id)
        if stored is None or not isinstance(stored.record, Finding):
            raise GitHubConflictError("durable finding is missing or has the wrong record type")
        if expected_finding_revision is not None and stored.revision != expected_finding_revision:
            raise GitHubConflictError(
                "finding revision does not match canonical revision before remote mutation"
            )
        if not _same_finding_content(stored.record, finding):
            raise GitHubConflictError("caller finding is stale relative to canonical finding")
        return stored.revision

    def _repair_succeeded_claim(
        self,
        entry: GitHubOutboxEntry,
        issue: GitHubIssue,
    ) -> GitHubFindingClaim | None:
        """Settle a claim left in flight by a crash after outbox success."""

        claim = self.outbox.get_claim(entry.repository, entry.finding_id)
        if claim is None or claim.state == "succeeded":
            return claim
        if claim.operation_id != entry.operation_id:
            return claim
        return self._mark_claim(claim, "succeeded", issue=issue)

    def _acquire_finding_claim(
        self,
        rendered: RenderedFinding,
        *,
        operation_id: str,
        finding_revision: int | None,
        worker_id: str,
        issue: GitHubIssue | None,
        retry_ambiguous: bool,
    ) -> tuple[GitHubFindingClaim, bool]:
        """CAS-acquire the finding-wide remote-issue claim."""

        desired_issue = issue.to_dict() if issue is not None else None
        for _attempt in range(3):
            current = self.outbox._claim_with_revision(rendered.repository, rendered.finding_id)
            if current is None:
                desired = GitHubFindingClaim(
                    repository=rendered.repository,
                    finding_id=rendered.finding_id,
                    operation_id=operation_id,
                    request_digest=rendered.request_digest,
                    marker=rendered.marker,
                    state="in_flight",
                    worker_id=worker_id,
                    finding_revision=finding_revision,
                    issue=desired_issue,
                )
                try:
                    claimed = self.outbox.enqueue_claim(desired)
                    return claimed, (
                        claimed.operation_id == operation_id and claimed.state == "in_flight"
                    )
                except GitHubOutboxError:
                    continue
            current_claim, revision = current
            if current_claim.state == "in_flight" and current_claim.operation_id != operation_id:
                return current_claim, False
            if current_claim.state == "succeeded":
                # A different succeeded request is newer durable state.  An
                # older ambiguous operation must never roll it back; an
                # identical request may only observe/reconcile that issue.
                return current_claim, False
            if current_claim.state == "ambiguous" and not retry_ambiguous:
                return current_claim, False
            if desired_issue is None and current_claim.state != "failed":
                desired_issue = current_claim.issue
            desired = replace(
                current_claim,
                operation_id=operation_id,
                request_digest=rendered.request_digest,
                marker=rendered.marker,
                state="in_flight",
                worker_id=worker_id,
                finding_revision=finding_revision,
                issue=desired_issue,
                reason="",
                updated_at=utc_now(),
            )
            try:
                return self.outbox.update_claim(desired, expected_revision=revision), True
            except GitHubOutboxError:
                continue
        latest = self.outbox.get_claim(rendered.repository, rendered.finding_id)
        if latest is None:
            raise GitHubOutboxError("finding-wide claim disappeared during acquisition")
        return latest, False

    def _refresh_claim_issue(
        self,
        claim: GitHubFindingClaim,
        repository: str,
        finding_id: str,
    ) -> GitHubIssue:
        """Read a claimed issue again instead of creating from a stale snapshot."""

        issue = _issue_from_claim(claim)
        if issue is None:
            raise GitHubOutboxError("finding claim has no valid remote issue")
        try:
            refreshed = _coerce_issue(self.provider.get_issue(repository, issue.number))
        except LookupError as exc:
            raise GitHubIssueMissing(f"claimed issue was not found: {exc}") from exc
        except GitHubValidationError as exc:
            raise GitHubConflictError(f"claimed issue snapshot is malformed: {exc}") from exc
        except GitHubSyncError:
            raise
        except Exception as exc:
            raise GitHubTransportError(f"claimed issue could not be read: {exc}") from exc
        try:
            _validate_issue_for_finding(refreshed, repository, finding_id, claim.marker)
        except GitHubValidationError as exc:
            raise GitHubConflictError(f"claimed issue identity changed: {exc}") from exc
        return refreshed

    def _recover_missing_claim(
        self,
        claim: GitHubFindingClaim,
        rendered: RenderedFinding,
        *,
        operation_id: str,
        finding_revision: int | None,
        worker_id: str,
    ) -> tuple[GitHubFindingClaim, bool]:
        """Terminalize a deleted claim and CAS-acquire a replacement owner."""

        reason = "claimed remote issue is missing; recovering the finding claim"
        self._mark_claim(claim, "failed", reason, allow_terminalize=True)
        return self._acquire_finding_claim(
            rendered,
            operation_id=operation_id,
            finding_revision=finding_revision,
            worker_id=worker_id,
            issue=None,
            retry_ambiguous=True,
        )

    def _refresh_remote_before_update(
        self,
        remote: GitHubIssue,
        *,
        repository: str,
        finding_id: str,
        marker: str,
        expected_block_digest: str,
    ) -> GitHubIssue:
        """Re-read immediately before an update and reject an untracked body edit."""

        try:
            refreshed = _coerce_issue(self.provider.get_issue(repository, remote.number))
        except LookupError as exc:
            raise GitHubIssueMissing(f"remote issue was deleted before update: {exc}") from exc
        except GitHubSyncError:
            raise
        except Exception as exc:
            raise GitHubTransportError(
                f"remote issue could not be re-read before update: {exc}"
            ) from exc
        _validate_issue_for_finding(refreshed, repository, finding_id, marker)
        if refreshed.body != remote.body and not expected_block_digest:
            raise GitHubConflictError(
                "issue body changed after marker search; no prior auditor-block digest is available"
            )
        return refreshed

    @staticmethod
    def _reject_untracked_snapshot_change(
        before: GitHubIssue,
        after: GitHubIssue,
        *,
        expected_block_digest: str,
    ) -> None:
        """Reject a body edit observed between marker search and claim refresh."""

        if before.body != after.body and not expected_block_digest:
            raise GitHubConflictError(
                "issue body changed during claim refresh; no prior auditor-block digest is available"
            )

    def _create_issue(
        self,
        rendered: RenderedFinding,
        *,
        finding_revision: int | None,
    ) -> GitHubIssue:
        """Create through the guarded canonical-revision seam when required."""

        if self.finding_store is not None:
            if finding_revision is None:
                raise GitHubConflictError(
                    "canonical finding revision is unavailable for issue create"
                )
            guarded = getattr(self.provider, "create_issue_with_finding_revision", None)
            if not callable(guarded):
                raise GitHubConflictError(
                    "provider lacks canonical finding-revision reservation; refusing issue create"
                )
            response = guarded(
                rendered.repository,
                finding_id=rendered.finding_id,
                expected_finding_revision=finding_revision,
                title=rendered.title,
                body=rendered.body,
                labels=rendered.labels,
            )
        else:
            response = self.provider.create_issue(
                rendered.repository,
                title=rendered.title,
                body=rendered.body,
                labels=rendered.labels,
            )
        return _coerce_issue(response)

    def _update_issue(
        self,
        remote: GitHubIssue,
        *,
        rendered: RenderedFinding,
        finding_revision: int | None,
        body: str,
    ) -> GitHubIssue:
        """Update through an explicit body and canonical revision CAS seam."""

        expected_body_digest = _sha256_text(remote.body)
        if self.finding_store is not None:
            if finding_revision is None:
                raise GitHubConflictError(
                    "canonical finding revision is unavailable for issue update"
                )
            guarded = getattr(self.provider, "update_issue_with_finding_revision", None)
            if not callable(guarded):
                raise GitHubConflictError(
                    "provider lacks canonical finding-revision reservation; refusing issue update"
                )
            response = guarded(
                rendered.repository,
                remote.number,
                finding_id=rendered.finding_id,
                expected_finding_revision=finding_revision,
                body=body,
                expected_body_digest=expected_body_digest,
                expected_updated_at=remote.updated_at,
            )
        else:
            guarded = getattr(self.provider, "update_issue_if_unchanged", None)
            if not callable(guarded):
                raise GitHubConflictError(
                    "provider lacks issue CAS/ETag support; refusing human-editable issue update"
                )
            response = guarded(
                rendered.repository,
                remote.number,
                body=body,
                expected_body_digest=expected_body_digest,
                expected_updated_at=remote.updated_at,
            )
        return _coerce_issue(response)

    def _mark_claim(
        self,
        claim: GitHubFindingClaim,
        state: str,
        reason: str = "",
        *,
        issue: GitHubIssue | None = None,
        allow_terminalize: bool = False,
    ) -> GitHubFindingClaim:
        """Persist a claim transition without downgrading a completed claim."""

        current = self.outbox._claim_with_revision(*claim.key)
        if current is None:
            raise GitHubOutboxError("cannot mark a missing finding claim")
        current_claim, revision = current
        if current_claim.state == "succeeded" and state != "succeeded" and not allow_terminalize:
            return current_claim
        marked = replace(
            current_claim,
            state=state,
            reason=reason,
            issue=issue.to_dict() if issue is not None else current_claim.issue,
            updated_at=utc_now(),
        )
        try:
            return self.outbox.update_claim(marked, expected_revision=revision)
        except GitHubOutboxError:
            latest = self.outbox.get_claim(*claim.key)
            if latest is not None:
                return latest
            raise

    def _search(self, repository: str, marker: str) -> tuple[SearchResult | None, str | None]:
        try:
            return SearchResult.from_value(
                self.provider.search_issues(repository, marker=marker)
            ), None
        except GitHubValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider is an external boundary.
            return None, f"marker search failed: {type(exc).__name__}: {exc}"

    def _create_failure_write_outcome(
        self,
        rendered: RenderedFinding,
        *,
        conflict_hint: bool,
    ) -> str:
        """Classify a failed create using a read-only marker reconciliation.

        A provider-side conflict can be rejected before its remote mutation,
        while a transport failure can arrive after GitHub accepted the create.
        The terminal status alone cannot distinguish those cases.  A complete
        exact-marker readback proves that an issue exists; an unavailable or
        incomplete readback remains ambiguous and therefore consumes the
        reserved write.  This helper never issues a second create.
        """

        search, search_error = self._search(rendered.repository, rendered.marker)
        if search_error is not None or search is None or not search.complete:
            return "ambiguous"
        remote = _select_marker_issue(
            search,
            repository=rendered.repository,
            finding_id=rendered.finding_id,
        )
        if remote is not None:
            return "applied"
        return "none" if conflict_hint else "ambiguous"

    def _update_failure_write_outcome(
        self,
        remote: GitHubIssue,
        target_body: str,
        *,
        conflict_hint: bool,
    ) -> str:
        """Classify a failed update from a read-only issue snapshot.

        ``GitHubConflictError`` is normally a no-send CAS rejection, but a
        provider may apply the body and then lose its response.  A target body
        readback proves the mutation; an unchanged body proves a pre-send
        conflict.  Other changed/unreadable snapshots remain ambiguous.
        """

        try:
            observed = _coerce_issue(self.provider.get_issue(remote.repository, remote.number))
        except Exception:  # noqa: BLE001 - an unreadable post-send state is ambiguous.
            return "ambiguous"
        if observed.body == target_body:
            return "applied"
        if conflict_hint and observed.body == remote.body:
            return "none"
        return "ambiguous"

    def _claim(self, entry: GitHubOutboxEntry, worker_id: str) -> GitHubOutboxEntry:
        current = self.outbox._get_with_revision(*entry.key)
        if current is None:
            raise GitHubOutboxError("cannot claim a missing outbox entry")
        current_entry, revision = current
        if current_entry.state == "in_flight" and current_entry.worker_id not in {"", worker_id}:
            return current_entry
        if current_entry.state == "succeeded":
            return current_entry
        claimed = replace(
            current_entry,
            state="in_flight",
            attempts=current_entry.attempts + 1,
            worker_id=worker_id,
            reason="",
            updated_at=utc_now(),
        )
        return self.outbox.update(claimed, expected_revision=revision)

    def _mark(self, entry: GitHubOutboxEntry, state: str, reason: str) -> GitHubOutboxEntry:
        current = self.outbox._get_with_revision(*entry.key)
        if current is None:
            raise GitHubOutboxError("cannot mark a missing outbox entry")
        current_entry, revision = current
        if current_entry.state == "succeeded" and state != "succeeded":
            return current_entry
        marked = replace(current_entry, state=state, reason=reason, updated_at=utc_now())
        try:
            return self.outbox.update(marked, expected_revision=revision)
        except GitHubOutboxError:
            latest = self.outbox.get(*entry.key)
            if latest is not None:
                return latest
            raise

    def _persist_entry(self, entry: GitHubOutboxEntry) -> GitHubOutboxEntry:
        current = self.outbox._get_with_revision(*entry.key)
        if current is None:
            return self.outbox.enqueue(entry)
        current_entry, revision = current
        if current_entry.to_dict() == entry.to_dict():
            return current_entry
        return self.outbox.update(replace(entry, updated_at=utc_now()), expected_revision=revision)

    def _result(
        self,
        status: str,
        repository: str,
        finding: Finding,
        entry: GitHubOutboxEntry,
        *,
        reason: str = "",
        remote_write: str = "none",
    ) -> GitHubSyncResult:
        return GitHubSyncResult(
            status,
            repository,
            finding.finding_id,
            entry.operation_id,
            _issue_from_entry(entry),
            entry,
            finding,
            reason,
            False,
            remote_write,
        )

    def _persist_link(
        self,
        finding: Finding,
        remote: GitHubIssue,
        entry: GitHubOutboxEntry,
        *,
        expected_finding_revision: int | None,
        link: Mapping[str, Any] | None = None,
    ) -> Finding:
        """Persist the remote identity when the optional BA-03 adapter is supplied."""

        link = link or _build_link(remote, finding, _rendered_from_entry(entry))
        if self.finding_store is None:
            # Keep the link visible to the caller even when the optional
            # canonical FindingStore adapter is not wired yet. This is not a
            # durable canonical write; the caller must persist this value.
            return replace(finding, github_issue=link, updated_at=utc_now())
        store = getattr(self.finding_store, "store", None)
        if store is None:
            raise GitHubConflictError("finding-store adapter does not expose canonical AuditStore")
        stored = store.get(finding.finding_id)
        if stored is None or not isinstance(stored.record, Finding):
            raise GitHubConflictError("durable finding is missing or has the wrong record type")
        if stored.record != finding:
            # A replay commonly carries the caller's pre-link snapshot. It is
            # safe to return the canonical already-linked finding when every
            # scientific field matches and only link/timestamp metadata differ.
            if _same_finding_content(stored.record, finding) and stored.record.github_issue == link:
                return stored.record
            raise GitHubConflictError("caller finding is stale relative to canonical finding")
        if stored.record.github_issue == link:
            return stored.record
        expected = (
            stored.revision if expected_finding_revision is None else expected_finding_revision
        )
        if expected != stored.revision:
            raise GitHubConflictError("finding revision does not match canonical revision")
        updated = replace(finding, github_issue=link, updated_at=utc_now())
        try:
            self.finding_store.update(
                updated,
                operation_id=f"{entry.operation_id}:finding-link",
                expected_revision=expected,
                actor="agent",
            )
        except Exception as exc:
            raise GitHubConflictError(f"durable finding link CAS failed: {exc}") from exc
        return updated


def sync_finding(
    provider: GitHubProvider,
    outbox: GitHubOutbox,
    repository: str,
    finding: Finding,
    *,
    operation_id: str,
    evidence: FindingEvidence | Mapping[str, Any] | None = None,
    finding_store: FindingStore | None = None,
    private_roots: Sequence[str | Path] = (),
    allowed_media_hosts: Sequence[str] = (),
    expected_finding_revision: int | None = None,
    retry_ambiguous: bool = False,
    worker_id: str = "",
) -> GitHubSyncResult:
    """Convenience wrapper around :class:`GitHubSync.sync`."""

    return GitHubSync(
        provider,
        outbox,
        finding_store=finding_store,
        private_roots=private_roots,
        allowed_media_hosts=allowed_media_hosts,
    ).sync(
        repository,
        finding,
        operation_id=operation_id,
        evidence=evidence,
        expected_finding_revision=expected_finding_revision,
        retry_ambiguous=retry_ambiguous,
        worker_id=worker_id,
    )


def finding_marker(repository: str, finding_id: str) -> str:
    """Return the stable exact marker searched before an issue create."""

    repository = _validate_repository(repository)
    finding_id = _validate_finding_id(finding_id)
    return f"{FINDING_MARKER_PREFIX}repository={repository} finding_id={finding_id}{FINDING_MARKER_SUFFIX}"


def parse_finding_marker(value: str) -> tuple[str, str] | None:
    """Parse one exact marker; malformed or duplicate markers fail closed."""

    if not isinstance(value, str):
        raise GitHubValidationError("marker source must be text")
    tokens = list(_MARKER_TOKEN_RE.finditer(value))
    if not tokens:
        if "robot_sf_audit_finding:v1" in value:
            raise GitHubValidationError("finding marker is malformed")
        return None
    if len(tokens) != 1:
        raise GitHubValidationError("issue contains duplicate finding markers")
    match = _MARKER_RE.fullmatch(tokens[0].group(0))
    if match is None:
        raise GitHubValidationError("finding marker is malformed")
    return match.group("repository"), match.group("finding_id")


def render_finding_issue(
    finding: Finding,
    *,
    repository: str,
    evidence: FindingEvidence | Mapping[str, Any] | None = None,
    private_roots: Sequence[str | Path] = (),
    allowed_media_hosts: Sequence[str] = (),
    data_filter: PublicDataFilter | None = None,
) -> RenderedFinding:
    """Render a finding with source identity, separated counts and safe evidence."""

    if not isinstance(finding, Finding):
        raise GitHubValidationError("finding must be a Finding contract")
    repository = _validate_repository(repository)
    marker = finding_marker(repository, finding.finding_id)
    filter_ = data_filter or PublicDataFilter(private_roots, allowed_media_hosts)
    evidence_value = FindingEvidence.from_value(evidence)
    sanitized = filter_.value(
        {
            "campaign_id": evidence_value.campaign_id,
            "source_identity": evidence_value.source_identity,
            "source_revision": evidence_value.source_revision or finding.source_revision,
            "source_digest": evidence_value.source_digest,
            "representative_cases": evidence_value.representative_cases,
            "reproduction_commands": evidence_value.reproduction_commands,
            "artifact_urls": evidence_value.artifact_urls,
            "media_urls": evidence_value.media_urls,
            "observations": evidence_value.observations,
            "competing_explanations": evidence_value.competing_explanations,
            "diagnostics": evidence_value.diagnostics,
            "metadata": evidence_value.metadata,
        }
    )
    finding_evidence = filter_.value(finding.evidence)
    negative_evidence = filter_.value(finding.negative_evidence)
    finding_diagnostics = filter_.value(finding.diagnostic_results)
    # Counts are computed from the typed finding before redaction so candidates
    # and confirmations can never collapse into one undifferentiated count.
    lines = [
        "## Benchmark audit finding",
        "",
        f"**Finding ID:** `{finding.finding_id}`",
        f"**Status:** `{finding.status}`",
        f"**Campaign:** `{sanitized['campaign_id'] or 'unavailable'}`",
        f"**Source identity:** `{sanitized['source_identity'] or 'unavailable'}`",
        f"**Source revision:** `{sanitized['source_revision'] or 'unavailable'}`",
        f"**Source digest:** `{sanitized['source_digest'] or 'unavailable'}`",
        "",
        f"### {filter_.text(finding.title)}",
        "",
        f"Candidate episodes ({finding.candidate_count}): {_render_items(finding.candidate_members, filter_)}",
        f"Confirmed episodes ({finding.confirmed_count}): {_render_items(finding.confirmed_members, filter_)}",
        f"Negative controls ({finding.negative_control_count}): {_render_items(finding.negative_controls, filter_)}",
        "",
        "Observations:",
        _render_bullets(finding.observations, filter_),
        "",
        "Competing explanations:",
        _render_bullets((*finding.hypotheses, *sanitized["competing_explanations"]), filter_),
        "",
        "Evidence records:",
        _render_json_bullets(finding_evidence),
        "",
        "Negative evidence:",
        _render_json_bullets(negative_evidence),
        "",
        "Representative cases:",
        _render_bullets(sanitized["representative_cases"], filter_),
        "",
        "Reproduction commands (source-bound):",
        _render_code_bullets(sanitized["reproduction_commands"], filter_),
        "",
        "Diagnostics:",
        _render_json_bullets((*finding_diagnostics, *sanitized["diagnostics"])),
        "",
        "Authorized artifact references:",
        _render_bullets(sanitized["artifact_urls"], filter_),
        "",
        "Media references:",
        _render_bullets(sanitized["media_urls"], filter_),
    ]
    content = "\n".join(lines).rstrip()
    auditor_block = f"{AUDITOR_BLOCK_START}\n{content}\n{AUDITOR_BLOCK_END}"
    body = f"{marker}\n\n{auditor_block}\n"
    labels = _safe_labels(finding.tags, filter_)
    return RenderedFinding(
        repository=repository,
        finding_id=finding.finding_id,
        marker=marker,
        title=f"[Audit finding] {filter_.text(finding.title)}",
        body=body,
        auditor_block=auditor_block,
        auditor_block_digest=_sha256_text(auditor_block),
        labels=labels,
        evidence=sanitized,
    )


def replace_auditor_block(
    issue_body: str,
    target_block: str,
    *,
    expected_block_digest: str = "",
) -> str:
    """Replace only one auditor-owned block and preserve all other body text."""

    if not isinstance(issue_body, str) or not isinstance(target_block, str):
        raise GitHubValidationError("issue body and target block must be text")
    if target_block.count(AUDITOR_BLOCK_START) != 1 or target_block.count(AUDITOR_BLOCK_END) != 1:
        raise GitHubValidationError("target auditor block delimiters are malformed")
    current_block = _extract_auditor_block(issue_body)
    start = issue_body.index(AUDITOR_BLOCK_START)
    end = start + len(current_block)
    if expected_block_digest and _sha256_text(current_block) != expected_block_digest:
        raise GitHubConflictError("auditor-owned block changed remotely; human conflict retained")
    return issue_body[:start] + target_block + issue_body[end:]


def _replace_auditor_block(
    issue: GitHubIssue,
    *,
    target_block: str,
    expected_block_digest: str,
) -> str:
    return replace_auditor_block(
        issue.body,
        target_block,
        expected_block_digest=expected_block_digest,
    )


def _extract_auditor_block(issue_body: str) -> str:
    """Return one auditor block, rejecting missing, duplicate, or malformed delimiters."""

    if not isinstance(issue_body, str):
        raise GitHubValidationError("issue body must be text")
    start_count = issue_body.count(AUDITOR_BLOCK_START)
    end_count = issue_body.count(AUDITOR_BLOCK_END)
    if start_count != 1 or end_count != 1:
        raise GitHubConflictError("issue body does not contain exactly one auditor-owned block")
    start = issue_body.index(AUDITOR_BLOCK_START)
    try:
        end = issue_body.index(AUDITOR_BLOCK_END, start)
    except ValueError as exc:
        raise GitHubConflictError("issue body auditor block delimiters are out of order") from exc
    end += len(AUDITOR_BLOCK_END)
    if end < start or issue_body.find(AUDITOR_BLOCK_START, end) >= 0:
        raise GitHubConflictError("issue body contains overlapping auditor blocks")
    return issue_body[start:end]


def _validate_repository(repository: str) -> str:
    if not isinstance(repository, str) or not _REPOSITORY_RE.fullmatch(repository):
        raise GitHubValidationError("repository must use the owner/name form")
    return repository


def _validate_finding_id(finding_id: str) -> str:
    if not isinstance(finding_id, str) or not _FINDING_ID_RE.fullmatch(finding_id):
        raise GitHubValidationError(
            "finding_id must contain only marker-safe letters, digits, '.', '_', ':', or '-'"
        )
    return finding_id


def _validate_operation_id(operation_id: str) -> str:
    if not isinstance(operation_id, str) or not operation_id.strip() or len(operation_id) > 512:
        raise GitHubValidationError(
            "operation_id must be a non-empty value of at most 512 characters"
        )
    return operation_id


def _coerce_issue(value: GitHubIssue | Mapping[str, Any]) -> GitHubIssue:
    if isinstance(value, GitHubIssue):
        return value
    if isinstance(value, Mapping):
        return GitHubIssue.from_mapping(value)
    raise GitHubValidationError("provider returned a malformed issue")


def _sha256(value: Any) -> str:
    try:
        encoded = canonical_json(value).encode("utf-8")
    except (AuditContractError, TypeError, ValueError) as exc:
        raise GitHubValidationError(f"value cannot be digested: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _publication_digest(
    rendered: RenderedFinding,
    *,
    finding_revision: int,
    publication_kind: str,
    body: str | None = None,
) -> str:
    """Digest the exact source-bound revision payload being published."""

    return _sha256(
        {
            "schema_version": GITHUB_PUBLICATION_SCHEMA_VERSION,
            "repository": rendered.repository,
            "finding_id": rendered.finding_id,
            "finding_revision": finding_revision,
            "publication_kind": publication_kind,
            "rendered_request_digest": rendered.request_digest,
            "body": (
                rendered.body if body is None and publication_kind == "initial_issue" else body
                if body is not None
                else rendered.auditor_block
            ),
        }
    )


def _render_initial_issue_body(rendered: RenderedFinding, finding_revision: int) -> str:
    """Add immutable publication identity to the initial issue snapshot."""

    if isinstance(finding_revision, bool) or finding_revision < 0:
        raise GitHubValidationError("finding revision must be non-negative")
    # The digest is intentionally computed over the rendered body plus this
    # stable revision envelope by the caller.  The marker below carries the
    # revision identity for recovery without making the body mutable.
    return (
        f"<!-- robot_sf_audit_publication:v1 finding_id={rendered.finding_id} "
        f"revision={finding_revision} -->\n\n{rendered.body}"
    )


def _publication_key(
    repository: str,
    finding_id: str,
    *,
    publication_kind: str,
    finding_revision: int,
    publication_digest: str,
) -> str:
    """Return the semantic uniqueness key for one finding publication."""

    return _sha256(
        {
            "repository": repository,
            "finding_id": finding_id,
            "publication_kind": publication_kind,
            "finding_revision": finding_revision,
            "publication_digest": publication_digest,
        }
    )


def _publication_request_marker(publication_digest: str) -> str:
    """Build the stable marker understood by the REST comment adapter."""

    if not re.fullmatch(r"[0-9a-f]{64}", publication_digest):
        raise GitHubValidationError("publication digest must be a SHA-256 hex digest")
    return f"<!-- robot_sf_audit_request:v1 request_digest={publication_digest} -->"


def _render_publication_comment(
    finding: Finding,
    rendered: RenderedFinding,
    finding_revision: int,
    publication_digest: str,
) -> str:
    """Render a self-contained append-only revision comment."""

    publication_marker = (
        f"<!-- robot_sf_audit_publication:v1 finding_id={finding.finding_id} "
        f"revision={finding_revision} digest={publication_digest} -->"
    )
    return "\n".join(
        (
            publication_marker,
            "## Benchmark audit finding revision",
            "",
            f"**Finding ID:** `{finding.finding_id}`",
            f"**Published revision:** `{finding_revision}`",
            f"**Publication digest:** `{publication_digest}`",
            f"**Source revision:** `{finding.source_revision or 'unavailable'}`",
            f"**Candidate episodes:** `{finding.candidate_count}`",
            f"**Confirmed episodes:** `{finding.confirmed_count}`",
            "",
            rendered.auditor_block,
        )
    )


def _coerce_comment_write(
    value: Any,
    *,
    publication_digest: str,
) -> tuple[Mapping[str, Any], str]:
    """Validate the provider-neutral append-comment response."""

    if isinstance(value, Mapping):
        status = value.get("status", "created")
        comment = value.get("comment", value)
    else:
        status = getattr(value, "status", "created")
        comment = getattr(value, "comment", None)
    if status not in {"created", "reconciled", "unchanged"}:
        raise GitHubValidationError(f"unknown append-comment status: {status}")
    if not isinstance(comment, Mapping):
        raise GitHubValidationError("append-comment response has no comment object")
    comment_id = comment.get("id")
    body = comment.get("body")
    if isinstance(comment_id, bool) or not isinstance(comment_id, int) or comment_id <= 0:
        raise GitHubValidationError("append-comment response has an invalid comment id")
    if not isinstance(body, str) or _publication_request_marker(publication_digest) not in body:
        raise GitHubValidationError("append-comment response lost its immutable request marker")
    return dict(comment), str(status)


def _select_marker_issue(
    search: SearchResult,
    *,
    repository: str,
    finding_id: str,
) -> GitHubIssue | None:
    matches: list[GitHubIssue] = []
    for issue in search.issues:
        if issue.repository != repository:
            raise GitHubValidationError("provider returned an issue from the wrong repository")
        marker = parse_finding_marker(issue.body)
        if marker is None:
            continue
        marker_repository, marker_finding = marker
        if marker_repository != repository or marker_finding != finding_id:
            raise GitHubConflictError("search returned a conflicting finding marker")
        matches.append(issue)
    if len(matches) > 1:
        raise GitHubConflictError("more than one remote issue carries the finding marker")
    return matches[0] if matches else None


def _validate_issue_for_finding(
    issue: GitHubIssue,
    repository: str,
    finding_id: str,
    marker: str,
) -> None:
    if issue.repository != repository:
        raise GitHubValidationError("remote issue repository does not match configured repository")
    if parse_finding_marker(issue.body) != (repository, finding_id):
        raise GitHubValidationError("remote issue marker does not match finding identity")
    if marker not in issue.body:
        raise GitHubValidationError("remote issue is missing the exact stable finding marker")


def _linked_issue_identity(
    finding: Finding,
    repository: str,
    marker: str,
) -> dict[str, Any] | None:
    link = finding.github_issue
    if link is None:
        return None
    if not isinstance(link, Mapping):
        raise GitHubValidationError("finding github_issue link must be an object")
    link_repository = link.get("repository")
    number = link.get("number", link.get("issue_number"))
    if link.get("schema_version", GITHUB_LINK_SCHEMA_VERSION) != GITHUB_LINK_SCHEMA_VERSION:
        raise GitHubValidationError("finding github_issue link schema is unsupported")
    if link_repository != repository:
        raise GitHubConflictError("finding link repository does not match requested repository")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise GitHubValidationError("finding github_issue link number is invalid")
    if link.get("marker") != marker:
        raise GitHubConflictError("finding github_issue link marker does not match finding")
    return {"repository": repository, "number": number}


def _link_block_digest(link: Mapping[str, Any] | None) -> str:
    if not isinstance(link, Mapping):
        return ""
    value = link.get("auditor_block_digest", "")
    return value if isinstance(value, str) else ""


def _same_finding_content(left: Finding, right: Finding) -> bool:
    """Compare scientific finding fields while ignoring link/timestamp metadata."""

    left_value = replace(left, github_issue=None, updated_at=right.updated_at)
    right_value = replace(right, github_issue=None, updated_at=right.updated_at)
    return left_value == right_value


def _build_link(
    issue: GitHubIssue,
    finding: Finding,
    rendered: RenderedFinding,
    *,
    entry: GitHubOutboxEntry | None = None,
) -> dict[str, Any]:
    link = {
        "schema_version": GITHUB_LINK_SCHEMA_VERSION,
        "repository": issue.repository,
        "number": issue.number,
        "url": issue.url,
        "marker": rendered.marker,
        "source_revision": finding.source_revision,
        "candidate_count": finding.candidate_count,
        "confirmed_count": finding.confirmed_count,
        "auditor_block_digest": rendered.auditor_block_digest,
    }
    if entry is not None and entry.publication_schema_version == GITHUB_PUBLICATION_SCHEMA_VERSION:
        link.update(
            {
                "sync_mode": "append_only",
                "publication_schema_version": entry.publication_schema_version,
                "publication_kind": entry.publication_kind,
                "publication_revision": entry.publication_revision,
                "publication_digest": entry.publication_digest,
                "publication_key": entry.publication_key,
                "comment_id": (
                    entry.comment.get("id")
                    if isinstance(entry.comment, Mapping)
                    else None
                ),
            }
        )
    return link


def _issue_from_entry(entry: GitHubOutboxEntry) -> GitHubIssue | None:
    if entry.issue is None:
        return None
    try:
        return _coerce_issue(entry.issue)
    except GitHubSyncError:
        return None


def _issue_from_claim(claim: GitHubFindingClaim) -> GitHubIssue | None:
    if claim.issue is None:
        return None
    try:
        return _coerce_issue(claim.issue)
    except GitHubSyncError:
        return None


def _rendered_from_entry(entry: GitHubOutboxEntry) -> RenderedFinding:
    return RenderedFinding(
        repository=entry.repository,
        finding_id=entry.finding_id,
        marker=entry.marker,
        title=entry.title,
        body=entry.body,
        auditor_block=entry.body[
            entry.body.index(AUDITOR_BLOCK_START) : entry.body.index(AUDITOR_BLOCK_END)
            + len(AUDITOR_BLOCK_END)
        ],
        auditor_block_digest=entry.auditor_block_digest,
        labels=entry.labels,
        evidence={},
    )


def _render_items(items: Sequence[str], data_filter: PublicDataFilter) -> str:
    if not items:
        return "none"
    return ", ".join(f"`{data_filter.text(item)}`" for item in items)


def _render_bullets(items: Sequence[str], data_filter: PublicDataFilter) -> str:
    if not items:
        return "- none recorded"
    return "\n".join(f"- {data_filter.text(item)}" for item in items)


def _render_code_bullets(items: Sequence[str], data_filter: PublicDataFilter) -> str:
    if not items:
        return "- unavailable"
    return "\n".join(f"- `{data_filter.text(item)}`" for item in items)


def _render_json_bullets(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "- none recorded"
    return "\n".join(f"- `{canonical_json(item)}`" for item in items)


def _safe_labels(tags: Sequence[str], data_filter: PublicDataFilter) -> tuple[str, ...]:
    labels: list[str] = ["benchmark-audit"]
    for tag in tags:
        cleaned = data_filter.text(tag).strip()
        if cleaned and cleaned not in labels and "[REDACTED_" not in cleaned:
            labels.append(cleaned[:50])
    return tuple(labels)


__all__ = [
    "AUDITOR_BLOCK_END",
    "AUDITOR_BLOCK_START",
    "FINDING_MARKER_PREFIX",
    "FINDING_MARKER_SUFFIX",
    "GITHUB_CLAIM_SCHEMA_VERSION",
    "GITHUB_LINK_SCHEMA_VERSION",
    "GITHUB_OUTBOX_SCHEMA_VERSION",
    "GITHUB_PUBLICATION_SCHEMA_VERSION",
    "GITHUB_SYNC_SCHEMA_VERSION",
    "FindingEvidence",
    "GitHubAmbiguousCreate",
    "GitHubConflictError",
    "GitHubFindingClaim",
    "GitHubIssue",
    "GitHubIssueMissing",
    "GitHubOutbox",
    "GitHubOutboxEntry",
    "GitHubOutboxError",
    "GitHubPrivacyError",
    "GitHubProvider",
    "GitHubSync",
    "GitHubSyncError",
    "GitHubSyncResult",
    "GitHubTransportError",
    "GitHubValidationError",
    "PublicDataFilter",
    "RenderedFinding",
    "SearchResult",
    "finding_marker",
    "parse_finding_marker",
    "render_finding_issue",
    "replace_auditor_block",
    "sync_finding",
]
