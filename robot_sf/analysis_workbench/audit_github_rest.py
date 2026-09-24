# ruff: noqa: C901, D105, D107, DOC201, PLR0912, PLR0913, PLR0915

"""Bounded, injectable GitHub REST transport for audit synchronization.

This module is deliberately a transport foundation rather than a second
synchronizer.  It owns HTTP request construction, repository scope, response
validation, and complete pagination.  It exposes the existing
``GitHubIssue``/``SearchResult`` shapes for issue create/search/get, and a
separate append-only comment operation for later service integration.

Only ``GET`` and ``POST`` are implemented.  In particular, this adapter does
not implement read-then-``PATCH`` issue-body updates or pretend that GitHub's
ordinary issue endpoint provides a compare-and-swap primitive.  The V1
``GitHubSync`` path therefore creates the immutable issue snapshot once and
uses ``append_auditor_comment`` for later revisions.  The historical body-CAS
path remains available only to explicit compatibility fakes.

The HTTP client is injected.  No network client, credential lookup, retry, or
automatic mutation is constructed by this module.  A caller may provide an
HTTP implementation backed by a fake in tests or by an explicitly configured
client in a later integration.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse

from robot_sf.analysis_workbench.audit_github import (
    GitHubCapabilityUnavailable,
    GitHubIssue,
    GitHubIssueMissing,
    GitHubTransportError,
    GitHubValidationError,
    SearchResult,
)

DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_PER_PAGE = 100
DEFAULT_MAX_PAGES = 100
DEFAULT_MAX_RESPONSE_BYTES = 1 << 20
DEFAULT_MAX_TOTAL_BYTES = 16 << 20
GITHUB_SEARCH_RESULT_CAP = 1_000

AUDITOR_REQUEST_MARKER_PREFIX = "<!-- robot_sf_audit_request:v1 "
AUDITOR_REQUEST_MARKER_SUFFIX = " -->"

_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAFE_MARKER_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_MARKER_RE = re.compile(
    r"^<!-- robot_sf_audit_request:v1 "
    r"(?:(?:request_digest=(?P<digest>[0-9a-f]{64}))|(?:request_id=(?P<request_id>[A-Za-z0-9_.:-]{1,128}))) -->$"
)
_LINK_RE = re.compile(r"<(?P<url>[^>]+)>\s*;\s*rel=\"?(?P<rel>[^\";,\s]+)\"?")


class GitHubRestError(GitHubTransportError):
    """Base class for fail-closed REST adapter errors."""


class GitHubRestValidationError(GitHubRestError, GitHubValidationError):
    """Raised for an invalid request, repository, marker, or response shape."""


class GitHubRestPaginationError(GitHubRestError):
    """Raised when a bounded collection cannot be proven complete."""


class GitHubRestLimitError(GitHubRestPaginationError):
    """Raised when a page or collection exceeds its byte/page bound."""


class GitHubRestTimeoutError(GitHubRestError, TimeoutError):
    """A request timed out before its outcome was observed."""


class GitHubRestAmbiguousError(GitHubRestError):
    """A mutating request may have committed but cannot be safely reconciled."""


class GitHubRestHTTPError(GitHubRestError):
    """The provider returned a non-success HTTP status."""


class GitHubRestUnsupportedError(GitHubRestError, GitHubCapabilityUnavailable, NotImplementedError):
    """The requested operation is outside this append-only transport.

    The REST adapter raises this before constructing an HTTP request.  The
    synchronizer therefore reports an explicit unavailable capability with no
    remote-write charge; an exception raised after a ``POST`` remains an
    ambiguous provider outcome.
    """


class GitHubRestDuplicateMarkerError(GitHubRestError):
    """More than one exact marker was observed in a complete collection."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Small response shape accepted from an injected fake HTTP client."""

    status_code: int
    body: bytes | str | Mapping[str, Any] | Sequence[Any]
    headers: Mapping[str, str] = ()


class HttpTransport(Protocol):
    """Injected request callable used by :class:`GitHubRESTProvider`."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse | Any:
        """Issue one request and return an HTTP-like response."""


HttpRequest = Callable[..., HttpResponse | Any]


@dataclass(frozen=True, slots=True)
class GitHubCommentWrite:
    """Result of an idempotent auditor-comment append.

    ``comment`` is the validated provider payload.  ``status`` is ``created``
    for a directly observed POST response, ``reconciled`` when a timeout was
    settled by a complete comment listing, and ``unchanged`` when the exact
    immutable marker already existed.  This is deliberately not a remote
    issue mutation receipt: callers still need their own durable outbox.
    """

    comment: Mapping[str, Any]
    marker: str
    status: str

    def __post_init__(self) -> None:
        if self.status not in {"created", "reconciled", "unchanged"}:
            raise GitHubRestValidationError(f"unknown comment write status: {self.status}")
        if not isinstance(self.comment, Mapping):
            raise GitHubRestValidationError("comment write payload must be an object")
        if not isinstance(self.marker, str) or not self.marker:
            raise GitHubRestValidationError("comment write marker must be text")

    @property
    def reconciled(self) -> bool:
        """Whether the result was settled after a network-ambiguous POST."""

        return self.status == "reconciled"

    @property
    def id(self) -> int:
        """Return the provider comment ID."""

        return int(self.comment["id"])

    @property
    def body(self) -> str:
        """Return the complete stored comment body."""

        return str(self.comment["body"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result envelope."""

        return {
            "comment": dict(self.comment),
            "marker": self.marker,
            "status": self.status,
            "reconciled": self.reconciled,
        }


def auditor_request_marker(request_digest_or_id: str) -> str:
    """Build the immutable marker carried by one auditor-owned comment.

    SHA-256 request digests use ``request_digest=``.  A bounded operation ID
    may be used by a caller that has not yet rendered a digest and is encoded
    as ``request_id=``.  Both forms are exact-matchable and cannot contain
    whitespace or HTML delimiters.
    """

    if not isinstance(request_digest_or_id, str):
        raise GitHubRestValidationError("request marker identity must be text")
    if _DIGEST_RE.fullmatch(request_digest_or_id):
        field = f"request_digest={request_digest_or_id}"
    elif _SAFE_MARKER_VALUE_RE.fullmatch(request_digest_or_id):
        field = f"request_id={request_digest_or_id}"
    else:
        raise GitHubRestValidationError("request marker identity is malformed")
    return f"{AUDITOR_REQUEST_MARKER_PREFIX}{field}{AUDITOR_REQUEST_MARKER_SUFFIX}"


# Short aliases make the marker contract discoverable without exposing the
# implementation's field naming choice to callers.
build_auditor_request_marker = auditor_request_marker
request_marker = auditor_request_marker


def parse_auditor_request_marker(marker: str) -> str:
    """Validate and return the digest/ID embedded in an immutable marker."""

    if not isinstance(marker, str):
        raise GitHubRestValidationError("request marker must be text")
    match = _REQUEST_MARKER_RE.fullmatch(marker)
    if match is None:
        raise GitHubRestValidationError("request marker has an unsupported shape")
    return match.group("digest") or match.group("request_id")  # type: ignore[return-value]


def _validate_repository(repository: str, allowed: frozenset[str]) -> str:
    if not isinstance(repository, str) or _REPOSITORY_RE.fullmatch(repository) is None:
        raise GitHubRestValidationError("repository must be an owner/name pair")
    owner, name = repository.split("/", 1)
    if owner in {".", ".."} or name in {".", ".."}:
        raise GitHubRestValidationError("repository path segments are malformed")
    if repository not in allowed:
        raise GitHubRestValidationError("repository is outside the configured allowlist")
    return repository


def _validate_issue_number(number: int) -> int:
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise GitHubRestValidationError("issue number must be a positive integer")
    return number


def _header_value(headers: Mapping[str, Any], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value)
    return None


def _normalise_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        try:
            headers = dict(headers)
        except (TypeError, ValueError) as exc:
            raise GitHubRestValidationError("HTTP response headers must be a mapping") from exc
    return {str(key): str(value) for key, value in headers.items()}


def _body_as_bytes(body: Any, *, max_bytes: int) -> bytes:
    if isinstance(body, bytes):
        raw = body
    elif isinstance(body, str):
        raw = body.encode("utf-8")
    elif isinstance(body, (Mapping, list, tuple)):
        try:
            raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise GitHubRestValidationError("HTTP response body is not JSON serializable") from exc
    else:
        raise GitHubRestValidationError("HTTP response body must be JSON bytes or a JSON value")
    if len(raw) > max_bytes:
        raise GitHubRestLimitError("HTTP response exceeds the configured byte bound")
    return raw


def _decode_json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitHubRestValidationError("GitHub response body is not valid JSON") from exc


def _payload_repository(payload: Mapping[str, Any], fallback: str) -> str:
    raw = payload.get("repository", fallback)
    if isinstance(raw, Mapping):
        raw = raw.get("full_name", "")
    if not isinstance(raw, str) or raw != fallback:
        raise GitHubRestValidationError("provider response repository does not match request")
    return raw


def _normalise_labels(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise GitHubRestValidationError("provider issue labels must be a list")
    labels: list[str] = []
    for label in raw:
        if isinstance(label, Mapping):
            label = label.get("name")
        if not isinstance(label, str) or not label.strip():
            raise GitHubRestValidationError("provider issue labels must contain names")
        labels.append(label)
    return tuple(labels)


def _issue_from_payload(payload: Any, repository: str) -> GitHubIssue:
    if not isinstance(payload, Mapping):
        raise GitHubRestValidationError("provider issue response must be an object")
    _payload_repository(payload, repository)
    number = payload.get("number")
    title = payload.get("title")
    body = payload.get("body", "")
    url = payload.get("html_url", payload.get("url"))
    if body is None:
        body = ""
    if not isinstance(title, str) or not title.strip():
        raise GitHubRestValidationError("provider issue title must be non-empty text")
    if not isinstance(body, str):
        raise GitHubRestValidationError("provider issue body must be text or null")
    if not isinstance(url, str):
        raise GitHubRestValidationError("provider issue response is missing an HTTP URL")
    comments = payload.get("comments", ())
    # GitHub's issue, search, and create representations expose ``comments``
    # as an integer count.  The complete comment objects are a separate
    # collection endpoint and are populated by ``get_issue`` below.  Accept a
    # list as well for provider-neutral fakes and already-hydrated snapshots.
    if isinstance(comments, bool) or (
        not isinstance(comments, (int, Sequence)) or isinstance(comments, (str, bytes))
    ):
        raise GitHubRestValidationError("provider issue comments must be a count or list")
    if isinstance(comments, int):
        if comments < 0:
            raise GitHubRestValidationError("provider issue comment count must be non-negative")
        comments = ()
    try:
        return GitHubIssue(
            repository=repository,
            number=number,
            url=url,
            title=title,
            body=body,
            labels=_normalise_labels(payload.get("labels", ())),
            state=payload.get("state", "open"),
            comments=tuple(comments),
            updated_at=payload.get("updated_at", ""),
        )
    except (GitHubValidationError, TypeError, ValueError) as exc:
        raise GitHubRestValidationError(f"provider issue response is malformed: {exc}") from exc


def _comment_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GitHubRestValidationError("provider comment response must be an object")
    comment_id = payload.get("id")
    body = payload.get("body")
    if isinstance(comment_id, bool) or not isinstance(comment_id, int) or comment_id <= 0:
        raise GitHubRestValidationError("provider comment ID must be a positive integer")
    if not isinstance(body, str):
        raise GitHubRestValidationError("provider comment body must be text")
    return dict(payload)


def _link_next(headers: Mapping[str, str]) -> str | None:
    value = _header_value(headers, "link")
    if value is None or not value.strip():
        return None
    links = _LINK_RE.findall(value)
    if not links:
        raise GitHubRestPaginationError("provider Link header is malformed")
    next_urls = [url for url, rel in links if rel == "next"]
    if len(next_urls) > 1:
        raise GitHubRestPaginationError("provider returned duplicate next links")
    return next_urls[0] if next_urls else None


def _page_number(url: str) -> int | None:
    value = parse_qs(urlparse(url).query).get("page", [None])[0]
    if value is None:
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    return number if number > 0 else None


def _pagination_query_identity(url: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return stable collection query parameters with only ``page`` removed."""

    query = parse_qs(urlparse(url).query, keep_blank_values=True)
    query.pop("page", None)
    return tuple(sorted((key, tuple(values)) for key, values in query.items()))


class GitHubRESTProvider:
    """Allowlisted GitHub REST provider with injectable HTTP.

    The provider makes no request until one of its methods is called.  It
    never retries a mutating ``POST``.  If a comment POST times out, it lists
    all comments and returns ``reconciled`` only after exactly one immutable
    marker is observed; otherwise it raises ``GitHubRestAmbiguousError``.
    """

    def __init__(
        self,
        http: HttpTransport | HttpRequest,
        *,
        allowed_repositories: Sequence[str],
        token: str | None = None,
        api_base_url: str = DEFAULT_GITHUB_API_URL,
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
        per_page: int = DEFAULT_PER_PAGE,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_search_results: int = GITHUB_SEARCH_RESULT_CAP,
    ) -> None:
        if not isinstance(allowed_repositories, Sequence) or isinstance(
            allowed_repositories, (str, bytes)
        ):
            raise GitHubRestValidationError("allowed_repositories must be a sequence")
        allowed = frozenset(allowed_repositories)
        if not allowed:
            raise GitHubRestValidationError("allowed_repositories must not be empty")
        if any(
            _REPOSITORY_RE.fullmatch(repo) is None
            or any(segment in {".", ".."} for segment in repo.split("/", 1))
            for repo in allowed
        ):
            raise GitHubRestValidationError("allowed_repositories contains a malformed repository")
        if not isinstance(api_base_url, str):
            raise GitHubRestValidationError("api_base_url must be text")
        parsed_base = urlparse(api_base_url)
        if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
            raise GitHubRestValidationError("api_base_url must be an HTTP(S) origin")
        if (
            parsed_base.username
            or parsed_base.password
            or parsed_base.path not in {"", "/"}
            or parsed_base.query
            or parsed_base.fragment
        ):
            raise GitHubRestValidationError("api_base_url must not contain credentials or a query")
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
            raise GitHubRestValidationError("timeout_s must be positive")
        for name, value in (
            ("per_page", per_page),
            ("max_pages", max_pages),
            ("max_response_bytes", max_response_bytes),
            ("max_total_bytes", max_total_bytes),
            ("max_search_results", max_search_results),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GitHubRestValidationError(f"{name} must be a positive integer")
        if per_page > 100:
            raise GitHubRestValidationError("per_page cannot exceed GitHub's limit of 100")
        if max_search_results > GITHUB_SEARCH_RESULT_CAP:
            raise GitHubRestValidationError(
                "max_search_results cannot exceed GitHub's 1000-result cap"
            )
        if not callable(http) and not callable(getattr(http, "request", None)):
            raise GitHubRestValidationError("http must be callable or expose request()")
        if token is not None and not isinstance(token, str):
            raise GitHubRestValidationError("token must be text when supplied")

        self._http = http
        self.allowed_repositories = allowed
        self.token = token
        self.api_base_url = api_base_url.rstrip("/") + "/"
        self.timeout_s = float(timeout_s)
        self.per_page = per_page
        self.max_pages = max_pages
        self.max_response_bytes = max_response_bytes
        self.max_total_bytes = max_total_bytes
        self.max_search_results = max_search_results

    def search_issues(self, repository: str, *, marker: str) -> SearchResult:
        """Search all bounded pages and return only exact marker matches."""

        repository = _validate_repository(repository, self.allowed_repositories)
        if not isinstance(marker, str) or not marker:
            raise GitHubRestValidationError("issue marker must be non-empty text")
        query = f'repo:{repository} "{marker}"'
        pages = self._paginate(
            "/search/issues",
            params={"q": query, "per_page": str(self.per_page), "page": "1"},
            item_key="items",
            max_items=self.max_search_results,
            expected_total_key="total_count",
        )
        issues: list[GitHubIssue] = []
        for item in pages.items:
            issue = _issue_from_payload(item, repository)
            occurrences = issue.body.count(marker)
            if occurrences > 1:
                raise GitHubRestDuplicateMarkerError("exact issue marker occurs more than once")
            if occurrences == 1:
                issues.append(issue)
        if len(issues) > 1:
            raise GitHubRestDuplicateMarkerError("exact issue marker occurs in multiple issues")
        return SearchResult(tuple(issues), complete=True)

    def list_issues(self, repository: str, marker: str | None = None) -> tuple[GitHubIssue, ...]:
        """List every bounded issue page, optionally filtering one exact marker."""

        repository = _validate_repository(repository, self.allowed_repositories)
        if marker is not None and (not isinstance(marker, str) or not marker):
            raise GitHubRestValidationError("issue marker must be non-empty text")
        pages = self._paginate(
            f"/repos/{quote(repository.split('/')[0], safe='')}/{quote(repository.split('/')[1], safe='')}/issues",
            params={"state": "all", "per_page": str(self.per_page), "page": "1"},
            item_key=None,
            max_items=self.max_pages * self.per_page,
        )
        issues = tuple(_issue_from_payload(item, repository) for item in pages.items)
        if marker is None:
            return issues
        matches: list[GitHubIssue] = []
        for issue in issues:
            occurrences = issue.body.count(marker)
            if occurrences > 1:
                raise GitHubRestDuplicateMarkerError("exact issue marker occurs more than once")
            if occurrences == 1:
                matches.append(issue)
        if len(matches) > 1:
            raise GitHubRestDuplicateMarkerError("exact issue marker occurs in multiple issues")
        return tuple(matches)

    def get_issue(self, repository: str, number: int) -> GitHubIssue:
        """Read one issue and its complete bounded comment collection."""

        repository = _validate_repository(repository, self.allowed_repositories)
        number = _validate_issue_number(number)
        path = self._issue_path(repository, number)
        response = self._request("GET", path)
        if response.status_code == 404:
            raise GitHubIssueMissing(f"issue {repository}#{number} was not found")
        payload = self._expect_status(response, {200})
        issue = _issue_from_payload(payload, repository)
        comments = self.list_comments(repository, number)
        return replace(issue, comments=comments)

    def create_issue(
        self,
        repository: str,
        *,
        title: str,
        body: str,
        labels: Sequence[str],
    ) -> GitHubIssue:
        """Create one issue without retrying an ambiguous POST."""

        repository = _validate_repository(repository, self.allowed_repositories)
        if not isinstance(title, str) or not title.strip():
            raise GitHubRestValidationError("issue title must be non-empty text")
        if not isinstance(body, str):
            raise GitHubRestValidationError("issue body must be text")
        labels = self._validate_labels(labels)
        response = self._request(
            "POST",
            self._issues_path(repository),
            payload={"title": title, "body": body, "labels": list(labels)},
        )
        payload = self._expect_status(response, {201})
        issue = _issue_from_payload(payload, repository)
        if issue.body != body or issue.title != title or issue.labels != labels:
            raise GitHubRestValidationError("create response does not match the request")
        return issue

    def list_comments(self, repository: str, number: int) -> tuple[Mapping[str, Any], ...]:
        """Read every comment page, rejecting truncation and duplicate IDs."""

        repository = _validate_repository(repository, self.allowed_repositories)
        number = _validate_issue_number(number)
        pages = self._paginate(
            f"{self._issue_path(repository, number)}/comments",
            params={"per_page": str(self.per_page), "page": "1"},
            item_key=None,
            max_items=self.max_pages * self.per_page,
        )
        comments: list[Mapping[str, Any]] = []
        seen_ids: set[int] = set()
        for item in pages.items:
            comment = _comment_from_payload(item)
            comment_id = int(comment["id"])
            if comment_id in seen_ids:
                raise GitHubRestPaginationError("provider returned a duplicate comment ID")
            seen_ids.add(comment_id)
            comments.append(comment)
        return tuple(comments)

    def append_auditor_comment(
        self,
        repository: str,
        number: int,
        *,
        body: str,
        request_digest: str | None = None,
        request_marker: str | None = None,
        marker: str | None = None,
    ) -> GitHubCommentWrite:
        """Append one auditor comment, reconciling a timeout by exact marker.

        Existing comments are never edited.  A complete listing is required
        before POST so an already accepted request is returned idempotently.
        A timeout or ambiguous POST is never retried blindly.
        """

        repository = _validate_repository(repository, self.allowed_repositories)
        number = _validate_issue_number(number)
        if not isinstance(body, str):
            raise GitHubRestValidationError("comment body must be text")
        supplied_markers = [value for value in (request_marker, marker) if value is not None]
        if len(supplied_markers) > 1 and supplied_markers[0] != supplied_markers[1]:
            raise GitHubRestValidationError("request_marker and marker disagree")
        chosen_marker = supplied_markers[0] if supplied_markers else None
        if request_digest is not None:
            generated = auditor_request_marker(request_digest)
            if chosen_marker is not None and chosen_marker != generated:
                raise GitHubRestValidationError("request marker does not match request digest")
            chosen_marker = generated
        if chosen_marker is None:
            raise GitHubRestValidationError("request_digest or request_marker is required")
        parse_auditor_request_marker(chosen_marker)
        if body.count(chosen_marker) > 1:
            raise GitHubRestDuplicateMarkerError("comment request marker occurs more than once")
        comment_body = body if chosen_marker in body else f"{chosen_marker}\n\n{body}"
        if comment_body.count(chosen_marker) != 1:
            raise GitHubRestValidationError("comment body must contain one exact request marker")

        existing = self._find_comment_marker(self.list_comments(repository, number), chosen_marker)
        if existing is not None:
            return GitHubCommentWrite(existing, chosen_marker, "unchanged")

        try:
            response = self._request(
                "POST",
                f"{self._issue_path(repository, number)}/comments",
                payload={"body": comment_body},
            )
            payload = self._expect_status(response, {201})
            comment = _comment_from_payload(payload)
            if comment["body"].count(chosen_marker) != 1:
                raise GitHubRestValidationError(
                    "comment response lost the immutable request marker"
                )
            return GitHubCommentWrite(comment, chosen_marker, "created")
        except (GitHubRestTimeoutError, TimeoutError, ConnectionError, OSError) as exc:
            try:
                reconciled = self._find_comment_marker(
                    self.list_comments(repository, number), chosen_marker
                )
            except Exception as reconcile_exc:
                raise GitHubRestAmbiguousError(
                    "comment POST outcome is ambiguous and reconciliation failed"
                ) from reconcile_exc
            if reconciled is None:
                raise GitHubRestAmbiguousError(
                    "comment POST outcome is ambiguous; exact marker was not confirmed"
                ) from exc
            return GitHubCommentWrite(reconciled, chosen_marker, "reconciled")

    # Explicit aliases keep the comment operation easy to discover while
    # retaining one implementation and one idempotency contract.
    append_issue_comment = append_auditor_comment
    list_issue_comments = list_comments

    # Explicitly expose the missing update seam as a safe failure.  No PATCH
    # request is ever constructed by this provider.
    def update_issue(self, *_args: Any, **_kwargs: Any) -> None:
        """Reject unsafe issue-body replacement."""

        raise GitHubRestUnsupportedError(
            "issue-body PATCH/CAS is not implemented; use append_auditor_comment"
        )

    def update_issue_if_unchanged(self, *_args: Any, **_kwargs: Any) -> None:
        """Reject the unproven provider-side body CAS seam."""

        self.update_issue(*_args, **_kwargs)

    def create_issue_with_finding_revision(self, *_args: Any, **_kwargs: Any) -> None:
        """Reject canonical finding-revision reservation not owned by REST."""

        raise GitHubRestUnsupportedError(
            "canonical finding-revision reservation is not implemented by this adapter"
        )

    def update_issue_with_finding_revision(self, *_args: Any, **_kwargs: Any) -> None:
        """Reject combined finding revision and issue-body mutation."""

        raise GitHubRestUnsupportedError(
            "canonical finding-revision issue update is not implemented by this adapter"
        )

    def _issues_path(self, repository: str) -> str:
        owner, name = repository.split("/", 1)
        return f"/repos/{quote(owner, safe='')}/{quote(name, safe='')}/issues"

    def _issue_path(self, repository: str, number: int) -> str:
        return f"{self._issues_path(repository)}/{number}"

    @staticmethod
    def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise GitHubRestValidationError("issue labels must be a sequence")
        normalized = tuple(labels)
        if any(not isinstance(label, str) or not label.strip() for label in normalized):
            raise GitHubRestValidationError("issue labels must contain non-empty text")
        return normalized

    @staticmethod
    def _find_comment_marker(
        comments: Sequence[Mapping[str, Any]], marker: str
    ) -> Mapping[str, Any] | None:
        matches: list[Mapping[str, Any]] = []
        for comment in comments:
            body = comment.get("body")
            if not isinstance(body, str):
                raise GitHubRestValidationError("comment body must be text")
            occurrences = body.count(marker)
            if occurrences > 1:
                raise GitHubRestDuplicateMarkerError("exact comment marker occurs more than once")
            if occurrences == 1:
                matches.append(comment)
        if len(matches) > 1:
            raise GitHubRestDuplicateMarkerError("exact comment marker occurs in multiple comments")
        return matches[0] if matches else None

    @dataclass(frozen=True, slots=True)
    class _Pages:
        items: tuple[Any, ...]

    def _paginate(
        self,
        path: str,
        *,
        params: Mapping[str, str],
        item_key: str | None,
        max_items: int,
        expected_total_key: str | None = None,
    ) -> _Pages:
        current_url = self._build_url(path, params)
        first_url = current_url
        items: list[Any] = []
        total_bytes = 0
        total_count: int | None = None
        seen_urls: set[str] = set()
        for page_index in range(1, self.max_pages + 1):
            if current_url in seen_urls:
                raise GitHubRestPaginationError("provider pagination repeated a page URL")
            seen_urls.add(current_url)
            response = self._request("GET", current_url)
            payload = self._expect_status(response, {200})
            total_bytes += response.body_bytes
            if total_bytes > self.max_total_bytes:
                raise GitHubRestLimitError("paginated response exceeds the total byte bound")
            if item_key is None:
                if not isinstance(payload, list):
                    raise GitHubRestValidationError("paginated provider response must be a list")
                page_items = payload
            else:
                if not isinstance(payload, Mapping):
                    raise GitHubRestValidationError("paginated provider response must be an object")
                page_items = payload.get(item_key)
                if not isinstance(page_items, list):
                    raise GitHubRestValidationError("paginated provider items must be a list")
                if expected_total_key is not None:
                    candidate_total = payload.get(expected_total_key)
                    if (
                        isinstance(candidate_total, bool)
                        or not isinstance(candidate_total, int)
                        or candidate_total < 0
                    ):
                        raise GitHubRestValidationError("provider total_count must be non-negative")
                    if total_count is None:
                        total_count = candidate_total
                    elif candidate_total != total_count:
                        raise GitHubRestPaginationError(
                            "provider total_count changed mid-pagination"
                        )
                    if candidate_total > max_items:
                        raise GitHubRestLimitError(
                            "provider total_count exceeds the configured item bound"
                        )
                    if "incomplete_results" not in payload:
                        raise GitHubRestPaginationError(
                            "provider search omitted incomplete_results"
                        )
                    incomplete = payload["incomplete_results"]
                    if not isinstance(incomplete, bool):
                        raise GitHubRestValidationError(
                            "provider incomplete_results must be boolean"
                        )
                    if incomplete:
                        raise GitHubRestPaginationError("provider marked search results incomplete")
            if len(page_items) > self.per_page:
                raise GitHubRestPaginationError("provider page exceeds the configured page size")
            items.extend(page_items)
            if len(items) > max_items:
                raise GitHubRestLimitError("paginated collection exceeds the configured item bound")
            next_url = _link_next(response.headers)
            if next_url is None:
                if len(page_items) == self.per_page:
                    if expected_total_key is None or total_count != len(items):
                        raise GitHubRestPaginationError(
                            "full-sized final page has no proof that pagination is complete"
                        )
                if expected_total_key is not None and total_count != len(items):
                    raise GitHubRestPaginationError(
                        "provider omitted a next link before total_count was collected"
                    )
                return self._Pages(tuple(items))
            if page_index >= self.max_pages:
                raise GitHubRestLimitError("provider pagination exceeded max_pages")
            if not isinstance(next_url, str) or not next_url:
                raise GitHubRestPaginationError("provider next link is malformed")
            next_url = urljoin(current_url, next_url)
            next_parsed = urlparse(next_url)
            first_parsed = urlparse(first_url)
            if (
                next_parsed.scheme not in {"http", "https"}
                or next_parsed.scheme != first_parsed.scheme
                or next_parsed.netloc != first_parsed.netloc
            ):
                raise GitHubRestPaginationError(
                    "provider next link leaves the configured API origin"
                )
            if next_parsed.path != first_parsed.path:
                raise GitHubRestPaginationError("provider next link changes the collection path")
            if _pagination_query_identity(next_url) != _pagination_query_identity(first_url):
                raise GitHubRestPaginationError(
                    "provider next link changes collection query parameters"
                )
            next_page = _page_number(next_url)
            current_page = _page_number(current_url)
            if next_page is None or current_page is None or next_page != current_page + 1:
                raise GitHubRestPaginationError(
                    "provider next link does not advance to the immediate next page"
                )
            current_url = next_url
        raise GitHubRestLimitError("provider pagination exceeded max_pages")

    def _build_url(self, path: str, params: Mapping[str, str]) -> str:
        if not path.startswith("/"):
            raise GitHubRestValidationError("REST path must be absolute")
        url = urljoin(self.api_base_url, path.lstrip("/"))
        return f"{url}?{urlencode(params)}" if params else url

    def _request(
        self,
        method: str,
        url_or_path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> _Response:
        if method not in {"GET", "POST"}:
            raise GitHubRestUnsupportedError("only GET and POST are supported")
        url = self._build_url(url_or_path, {}) if url_or_path.startswith("/") else url_or_path
        parsed = urlparse(url)
        base = urlparse(self.api_base_url)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != base.netloc:
            raise GitHubRestValidationError("request URL leaves the configured API origin")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        body = None
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(body) > self.max_response_bytes:
                raise GitHubRestLimitError("request body exceeds the configured byte bound")
        try:
            if callable(self._http):
                raw = self._http(method, url, headers=headers, body=body, timeout=self.timeout_s)
            else:
                raw = self._http.request(
                    method, url, headers=headers, body=body, timeout=self.timeout_s
                )
        except (GitHubRestError, GitHubRestTimeoutError):
            raise
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise GitHubRestTimeoutError("GitHub HTTP request outcome was not observed") from exc
        return self._coerce_response(raw)

    def _coerce_response(self, raw: Any) -> _Response:
        if isinstance(raw, HttpResponse):
            status = raw.status_code
            headers = _normalise_headers(raw.headers)
            body = raw.body
        elif isinstance(raw, Mapping):
            status = raw.get("status_code", raw.get("status"))
            headers = _normalise_headers(raw.get("headers", {}))
            body = raw.get("body")
        else:
            status = getattr(raw, "status_code", getattr(raw, "status", None))
            headers = _normalise_headers(getattr(raw, "headers", {}))
            body = getattr(raw, "body", None)
            if body is None and callable(getattr(raw, "read", None)):
                try:
                    body = raw.read(self.max_response_bytes + 1)
                except TypeError:
                    body = raw.read()
            if body is None and callable(getattr(raw, "json", None)):
                body = raw.json()
        if isinstance(status, bool) or not isinstance(status, int) or status < 100:
            raise GitHubRestValidationError("HTTP response status is malformed")
        raw_body = _body_as_bytes(body, max_bytes=self.max_response_bytes)
        payload = _decode_json(raw_body)
        return _Response(status, headers, payload, len(raw_body))

    @staticmethod
    def _expect_status(response: _Response, expected: set[int]) -> Any:
        if response.status_code not in expected:
            if response.status_code == 404:
                raise GitHubIssueMissing("requested GitHub resource was not found")
            raise GitHubRestHTTPError(f"GitHub REST request returned HTTP {response.status_code}")
        return response.payload


@dataclass(frozen=True, slots=True)
class _Response:
    status_code: int
    headers: Mapping[str, str]
    payload: Any
    body_bytes: int


# Naming aliases keep the transport easy to discover for callers that use
# ``REST`` or ``Rest`` spelling.  They are the same implementation and do not
# create additional integration surfaces.
GitHubRestProvider = GitHubRESTProvider
GitHubRESTTransport = GitHubRESTProvider
GitHubRestTransport = GitHubRESTProvider


__all__ = [
    "AUDITOR_REQUEST_MARKER_PREFIX",
    "AUDITOR_REQUEST_MARKER_SUFFIX",
    "DEFAULT_GITHUB_API_URL",
    "GITHUB_SEARCH_RESULT_CAP",
    "GitHubCommentWrite",
    "GitHubRESTProvider",
    "GitHubRESTTransport",
    "GitHubRestAmbiguousError",
    "GitHubRestDuplicateMarkerError",
    "GitHubRestError",
    "GitHubRestHTTPError",
    "GitHubRestLimitError",
    "GitHubRestPaginationError",
    "GitHubRestProvider",
    "GitHubRestTimeoutError",
    "GitHubRestTransport",
    "GitHubRestUnsupportedError",
    "GitHubRestValidationError",
    "HttpRequest",
    "HttpResponse",
    "HttpTransport",
    "auditor_request_marker",
    "build_auditor_request_marker",
    "parse_auditor_request_marker",
    "request_marker",
]
