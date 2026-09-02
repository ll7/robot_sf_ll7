"""Sync a derived priority score into GitHub Project #5 items.

The score is intentionally simple and benchmark-oriented:

    score = improvement * success_probability * time_criticality * unlock_factor
            / effort_hours**alpha

This helper is the deterministic `gh` fallback for Project #5 score sync. It is
intentionally kept scriptable for local/manual batch routing even as
interactive issue/PR/project work moves toward GitHub MCP / app tools.

The helper reads issue-backed project items through an explicit cursor-paginated
Projects API query, applies defaults and clamping for missing or invalid inputs,
and writes the derived numeric score back to a `Priority Score` project field.

The autopilot's ``sync --only-empty`` mode fails closed and returns a
machine-readable blocked status when the GitHub token lacks ``read:project``.
Callers can continue with live-label queue ordering and recover score sync by
refreshing the token's Project scope. Other sync modes preserve their existing
exception behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.dev._gh_pagination import assert_not_truncated
from scripts.dev.github_quota import (
    DEFAULT_CORE_SAFETY_THRESHOLD,
    DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
    RateLimitSnapshot,
    graphql_budget_decision,
    parse_rate_limit_payload,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_ALPHA = 0.8
DEFAULT_IMPROVEMENT = 1.0
DEFAULT_SUCCESS_PROBABILITY = 0.7
DEFAULT_EFFORT_HOURS = 1.0
DEFAULT_TIME_CRITICALITY = 1.0
DEFAULT_UNLOCK_FACTOR = 1.0

MIN_EFFORT_HOURS = 0.1
MIN_SCORE_MULTIPLIER = 0.1
MAX_SCORE_MULTIPLIER = 10.0
MIN_TIME_CRITICALITY = MIN_SCORE_MULTIPLIER
MAX_TIME_CRITICALITY = MAX_SCORE_MULTIPLIER
MIN_UNLOCK_FACTOR = MIN_SCORE_MULTIPLIER
MAX_UNLOCK_FACTOR = MAX_SCORE_MULTIPLIER
MIN_IMPROVEMENT = MIN_SCORE_MULTIPLIER
MAX_IMPROVEMENT = MAX_SCORE_MULTIPLIER
MIN_SUCCESS_PROBABILITY = 0.0
MAX_SUCCESS_PROBABILITY = 1.0
SUCCESS_PROBABILITY_PERCENT_SCALE = 100.0

EFFORT_FIELD = "Expected Duration in Hours"
PRIORITY_SCORE_FIELD = "Priority Score"
DEFAULT_REPO = "ll7/robot_sf_ll7"
READY_LABEL = "state:ready"
AMBIGUOUS_OR_BLOCKING_LABELS = frozenset(
    {
        "blocked",
        "decision-required",
        "deferred",
        "duplicate",
        "invalid",
        "needs-triage",
        "state:blocked",
        "state:blocked-external-input",
        "state:hold",
        "state:parked",
        "state:review",
        "state:running",
        "state:working",
        "wontfix",
    }
)
TERMINAL_PROJECT_STATUSES = frozenset({"cancelled", "closed", "completed", "done"})
REQUIRED_NUMBER_FIELDS: tuple[str, ...] = (
    "Improvement",
    "Success Probability",
    "Time Criticality",
    "Unlock Factor",
    PRIORITY_SCORE_FIELD,
)
MISSING_PROJECT_SCOPE_RE = re.compile(
    r"missing required scopes?\s*\[(?P<scopes>[^\]]*\bread:project\b[^\]]*)\]",
    re.IGNORECASE,
)
PROJECT_ITEM_GRAPHQL_PAGE_SIZE = 100
PROJECT_ITEM_GRAPHQL_QUERY = """
query($projectId: ID!, $first: Int!, $after: String) {
  node(id: $projectId) {
    __typename
    ... on ProjectV2 {
      items(first: $first, after: $after) {
        nodes {
          id
          type
          content {
            __typename
            ... on Issue {
              number
              title
              repository {
                nameWithOwner
              }
            }
            ... on PullRequest {
              number
              title
              repository {
                nameWithOwner
              }
            }
          }
          fieldValues(first: 100) {
            nodes {
              __typename
              ... on ProjectV2ItemFieldNumberValue {
                number
                field {
                  ... on ProjectV2Field {
                    id
                    name
                  }
                }
              }
              ... on ProjectV2ItemFieldSingleSelectValue {
                name
                field {
                  ... on ProjectV2SingleSelectField {
                    id
                    name
                  }
                }
              }
              ... on ProjectV2ItemFieldTextValue {
                text
                field {
                  ... on ProjectV2Field {
                    id
                    name
                  }
                }
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
""".strip()


class MissingProjectScopeError(RuntimeError):
    """Raised when GitHub rejects a Project command for missing ``read:project``."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        details: str,
        required_scopes: Sequence[str],
    ) -> None:
        """Store the failed command, CLI details, and required scope names."""
        self.command = tuple(command)
        self.details = details
        self.required_scopes = tuple(required_scopes)
        super().__init__(
            "GitHub Project access requires scope(s) "
            + ", ".join(self.required_scopes)
            + ". Refresh the token before retrying Project #5 priority sync."
            + f"\n{details}"
        )


class ProjectQuotaBlockedError(RuntimeError):
    """Raised when a later Project API operation would cross quota safety margins."""

    def __init__(self, decision: dict[str, Any]) -> None:
        """Store the machine-readable quota decision for resumable callers."""
        self.decision = decision
        super().__init__(
            str(
                decision.get(
                    "message",
                    "Project API operation is blocked until the configured quota safety margin "
                    "is available.",
                )
            )
        )


@dataclass(frozen=True, slots=True)
class CachedProjectMetadata:
    """Validated local Project #5 identifiers used as read hints."""

    project_id: str
    fields: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    """Normalized score inputs for one project item."""

    improvement: float
    success_probability: float
    effort_hours: float
    time_criticality: float
    unlock_factor: float


@dataclass(frozen=True, slots=True)
class SyncPreview:
    """Summary of one computed score update."""

    issue_number: int
    title: str
    status: str
    old_score: float | None
    new_score: float
    inputs: ScoreInputs


@dataclass(frozen=True, slots=True)
class ProjectItemFetchStats:
    """Observable completeness summary for one cursor-paginated item read."""

    pages: int
    accumulated_items: int


@dataclass(frozen=True, slots=True)
class SyncOptions:
    """Configuration for one score synchronization pass."""

    owner: str
    project_number: int
    ensure_fields: bool
    limit: int
    alpha: float
    round_digits: int
    issue_number: int | None
    dry_run: bool
    skip_statuses: set[str]
    only_empty: bool = False
    min_graphql_remaining: int = DEFAULT_GRAPHQL_SAFETY_THRESHOLD
    cache_file: Path | None = Path(".github/cache/project5.json")
    repo: str = DEFAULT_REPO


def read_rate_limit() -> RateLimitSnapshot:
    """Read GitHub quota through the REST endpoint without spending GraphQL budget."""
    try:
        completed = subprocess.run(
            ["gh", "api", "rate_limit"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RateLimitSnapshot(status="unavailable", error=f"rate_limit request failed: {exc}")
    if completed.returncode != 0:
        return RateLimitSnapshot(
            status="unavailable",
            error=(
                completed.stderr.strip() or completed.stdout.strip() or "rate_limit request failed"
            ),
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return RateLimitSnapshot(status="malformed", error=f"invalid rate_limit JSON: {exc}")
    return parse_rate_limit_payload(payload)


def estimated_project_graphql_requests(options: SyncOptions) -> int:
    """Estimate the initial bounded Project API requests before any mutation.

    An unscoped cursor scan has no known page count until its first response;
    later pages are guarded dynamically by ``item_list_paginated``. The
    preflight therefore reserves only the first item page, not the caller's
    per-page bound interpreted as a total-item cap.
    """
    item_requests = 1
    schema_requests = len(REQUIRED_NUMBER_FIELDS) + 1 if options.ensure_fields else 0
    return 2 + item_requests + schema_requests


def project_quota_decision(options: SyncOptions) -> dict[str, Any]:
    """Return the fail-closed quota decision for one score-sync invocation."""
    if options.limit <= 0:
        raise ValueError("limit must be positive")
    if options.min_graphql_remaining < 0:
        raise ValueError("min_graphql_remaining must be non-negative")
    snapshot = read_rate_limit()
    return graphql_budget_decision(
        snapshot,
        expected_graphql_requests=estimated_project_graphql_requests(options),
        min_graphql_remaining=options.min_graphql_remaining,
        expected_core_requests=1,
        min_core_remaining=DEFAULT_CORE_SAFETY_THRESHOLD,
    )


def ensure_project_graphql_budget(
    *,
    expected_graphql_requests: int,
    min_graphql_remaining: int,
    expected_core_requests: int = 1,
) -> None:
    """Fail closed when a future Project API step would cross quota margins."""
    decision = graphql_budget_decision(
        read_rate_limit(),
        expected_graphql_requests=expected_graphql_requests,
        min_graphql_remaining=min_graphql_remaining,
        expected_core_requests=expected_core_requests,
        min_core_remaining=DEFAULT_CORE_SAFETY_THRESHOLD,
    )
    if decision["status"] != "ok":
        raise ProjectQuotaBlockedError(decision)


def _read_project_cache(path: Path) -> dict[str, Any] | None:
    """Read one local cache object, treating filesystem and JSON errors as a miss."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_fields(raw_fields: Any) -> dict[str, dict[str, Any]]:
    """Normalize cached field entries and drop entries without usable IDs."""
    if not isinstance(raw_fields, dict):
        return {}
    fields: dict[str, dict[str, Any]] = {}
    for name, raw_field in raw_fields.items():
        if not isinstance(name, str) or not isinstance(raw_field, dict):
            continue
        field_id = raw_field.get("id")
        if isinstance(field_id, str) and field_id:
            fields[name] = {**raw_field, "name": name, "id": field_id}
    return fields


def load_project_cache(
    path: Path | None, *, owner: str, project_number: int
) -> CachedProjectMetadata | None:
    """Load a matching local Project #5 cache, treating stale or malformed data as a miss."""
    if path is None or not path.is_file():
        return None
    payload = _read_project_cache(path)
    if payload is None:
        return None
    if payload.get("owner") != owner or payload.get("project_number") != project_number:
        return None
    project_id = payload.get("project_id")
    raw_fields = payload.get("fields")
    if not isinstance(project_id, str) or not project_id:
        return None
    fields = _cache_fields(raw_fields)
    if not fields:
        return None
    return CachedProjectMetadata(project_id=project_id, fields=fields)


def clamp(value: float, *, lower: float, upper: float | None = None) -> float:
    """Clamp a numeric value into the allowed range."""

    bounded = max(lower, value)
    if upper is not None:
        bounded = min(upper, bounded)
    return bounded


def _coerce_float(raw: object) -> float | None:
    """Parse numbers emitted by the GitHub CLI item list output."""

    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        return float(stripped)
    raise TypeError(f"unsupported numeric field value: {raw!r}")


def field_keys(name: str) -> tuple[str, ...]:
    """Return the known item-list key variants for a project field name."""

    lower_first = name[:1].lower() + name[1:]
    fully_lower = name.lower()
    if lower_first == fully_lower:
        return (lower_first,)
    return (lower_first, fully_lower)


def field_value(item: dict[str, Any], name: str) -> object:
    """Return a project field value from the gh item-list payload."""

    for key in field_keys(name):
        if key in item:
            return item[key]
    return None


def _normalize_success_probability(raw: float | None) -> float:
    """Return a 0-1 probability, accepting whole-percent project-field inputs."""

    if raw is None:
        return DEFAULT_SUCCESS_PROBABILITY
    if raw > MAX_SUCCESS_PROBABILITY:
        return raw / SUCCESS_PROBABILITY_PERCENT_SCALE
    return raw


def normalize_inputs(item: dict[str, Any]) -> ScoreInputs:
    """Extract and clamp score inputs from a project item payload."""

    improvement = _coerce_float(field_value(item, "Improvement"))
    success_probability = _coerce_float(field_value(item, "Success Probability"))
    effort_hours = _coerce_float(field_value(item, EFFORT_FIELD))
    time_criticality = _coerce_float(field_value(item, "Time Criticality"))
    unlock_factor = _coerce_float(field_value(item, "Unlock Factor"))

    return ScoreInputs(
        improvement=clamp(
            improvement if improvement is not None else DEFAULT_IMPROVEMENT,
            lower=MIN_IMPROVEMENT,
            upper=MAX_IMPROVEMENT,
        ),
        success_probability=clamp(
            _normalize_success_probability(success_probability),
            lower=MIN_SUCCESS_PROBABILITY,
            upper=MAX_SUCCESS_PROBABILITY,
        ),
        effort_hours=clamp(
            effort_hours if effort_hours is not None else DEFAULT_EFFORT_HOURS,
            lower=MIN_EFFORT_HOURS,
        ),
        time_criticality=clamp(
            time_criticality if time_criticality is not None else DEFAULT_TIME_CRITICALITY,
            lower=MIN_TIME_CRITICALITY,
            upper=MAX_TIME_CRITICALITY,
        ),
        unlock_factor=clamp(
            unlock_factor if unlock_factor is not None else DEFAULT_UNLOCK_FACTOR,
            lower=MIN_UNLOCK_FACTOR,
            upper=MAX_UNLOCK_FACTOR,
        ),
    )


def compute_priority_score(inputs: ScoreInputs, *, alpha: float = DEFAULT_ALPHA) -> float:
    """Compute the derived priority score for one issue."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    numerator = (
        inputs.improvement
        * inputs.success_probability
        * inputs.time_criticality
        * inputs.unlock_factor
    )
    return numerator / math.pow(inputs.effort_hours, alpha)


class GhProjectClient:
    """Small wrapper around the gh CLI for project field automation."""

    def __init__(self) -> None:
        """Initialize read telemetry without changing the existing CLI surface."""
        self.last_item_fetch_stats: ProjectItemFetchStats | None = None
        self.last_eligibility_plan: dict[str, Any] | None = None

    def _run_completed(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a gh command and raise a high-signal error on failure."""

        try:
            return subprocess.run(
                ["gh", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            stdout = exc.stdout.strip()
            details = stderr or stdout or "no stderr/stdout captured"
            scope_match = MISSING_PROJECT_SCOPE_RE.search(details)
            if scope_match:
                required_scopes = tuple(
                    scope.strip()
                    for scope in scope_match.group("scopes").split(",")
                    if scope.strip()
                )
                raise MissingProjectScopeError(
                    command=("gh", *args),
                    details=details,
                    required_scopes=required_scopes,
                ) from exc
            raise RuntimeError(
                "gh command failed: "
                + " ".join(["gh", *args])
                + f"\n{details}\n"
                + "For interactive issue/PR/project work, prefer the GitHub MCP/app tools. "
                + "For this scripted fallback, verify `gh auth status` and ensure the token "
                + "has `project` scope."
            ) from exc

    def run_json(self, *args: str) -> dict[str, Any]:
        """Run a gh command and parse the JSON output."""

        completed = self._run_completed(*args)
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "gh command returned invalid JSON: "
                + " ".join(["gh", *args])
                + f"\n{completed.stdout.strip() or '<empty stdout>'}"
            ) from exc

    def run(self, *args: str) -> None:
        """Run a gh command for side effects."""

        self._run_completed(*args)

    def issue_snapshot(self, *, repo: str, issue_number: int) -> dict[str, Any]:
        """Return the exact REST issue fields used by score eligibility checks."""

        payload = self.run_json("api", f"repos/{repo}/issues/{issue_number}")
        if "pull_request" in payload:
            raise RuntimeError(f"#{issue_number} resolved to a pull request, not an issue")
        number = payload.get("number")
        title = payload.get("title")
        state = payload.get("state")
        updated_at = payload.get("updated_at")
        labels_raw = payload.get("labels")
        if (
            number != issue_number
            or not isinstance(title, str)
            or not isinstance(state, str)
            or not isinstance(updated_at, str)
            or not isinstance(labels_raw, list)
        ):
            raise RuntimeError(f"issue #{issue_number} REST snapshot is malformed")
        labels: list[str] = []
        for label in labels_raw:
            name = label.get("name") if isinstance(label, dict) else label
            if not isinstance(name, str) or not name:
                raise RuntimeError(f"issue #{issue_number} REST labels are malformed")
            labels.append(name)
        return {
            "number": number,
            "title": title,
            "state": state.upper(),
            "updated_at": updated_at,
            "labels": sorted(set(labels)),
        }

    def _should_retry_with_at_me(self, *, owner: str, error: RuntimeError) -> bool:
        """Limit `@me` fallback to the known ll7 user-owner gh quirk."""

        return owner == "ll7" and "unknown owner type" in str(error).lower()

    def _run_project_command(
        self,
        subcommand: str,
        *,
        owner: str,
        project_number: int,
        extra_args: Sequence[str] = (),
        as_json: bool = False,
    ) -> Any:
        """Run a `gh project` command with the known user-owner fallback."""

        args: tuple[str, ...] = (
            "project",
            subcommand,
            str(project_number),
            "--owner",
            owner,
            *extra_args,
        )
        if as_json:
            args = (*args, "--format", "json")
        try:
            if as_json:
                return self.run_json(*args)
            self.run(*args)
            return None
        except RuntimeError as exc:
            if not self._should_retry_with_at_me(owner=owner, error=exc):
                raise
            retry_args = (
                "project",
                subcommand,
                str(project_number),
                "--owner",
                "@me",
                *extra_args,
            )
            if as_json:
                return self.run_json(*retry_args, "--format", "json")
            self.run(*retry_args)
            return None

    def project_id(self, *, owner: str, project_number: int) -> str:
        """Return the GraphQL project ID."""

        payload = self._run_project_command(
            "view",
            owner=owner,
            project_number=project_number,
            as_json=True,
        )
        return str(payload["id"])

    def field_list(self, *, owner: str, project_number: int) -> list[dict[str, Any]]:
        """Return the current project fields."""

        payload = self._run_project_command(
            "field-list",
            owner=owner,
            project_number=project_number,
            as_json=True,
        )
        return list(payload["fields"])

    def ensure_number_field(self, *, owner: str, project_number: int, name: str) -> None:
        """Create a number field when it is missing."""

        self._run_project_command(
            "field-create",
            owner=owner,
            project_number=project_number,
            extra_args=(
                "--name",
                name,
                "--data-type",
                "NUMBER",
            ),
        )

    def item_list(self, *, owner: str, project_number: int, limit: int) -> list[dict[str, Any]]:
        """Return project items with their visible field values.

        Fails closed when the result reaches the ``limit`` cap: because this list
        drives score write-backs, a page at the cap (indistinguishable from a full
        page) could silently skip items beyond the limit. Callers that cannot
        bound the project size should use :meth:`item_list_until_issue` instead
        (issue #5870 / #5048 / #4991).
        """

        payload = self._run_project_command(
            "item-list",
            owner=owner,
            project_number=project_number,
            extra_args=(
                "--limit",
                str(limit),
            ),
            as_json=True,
        )
        items = list(payload["items"])
        # Fail closed: this list drives score write-backs, so a result at the cap
        # (indistinguishable from a full page) could silently skip items beyond the
        # limit. Raise instead of writing a partial sync (issue #5048 / #4991).
        assert_not_truncated(items, limit=limit, context="gh project item-list")
        return items

    def item_list_paginated(
        self,
        *,
        owner: str,
        project_number: int,
        project_id: str | None = None,
        limit: int,
        min_graphql_remaining: int = DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """Return every project item after explicit cursor-pagination completion.

        ``gh project item-list`` exposes a numeric cap but not the continuation
        cursor. This owner uses the same Projects API through ``gh api graphql``
        so a full logical page is accepted only when ``pageInfo`` proves that
        pagination is complete. The GraphQL API caps one request at 100 items;
        ``limit`` remains the caller's requested per-page bound and is reduced
        to that server maximum without changing the complete-scan contract.
        """

        if limit <= 0:
            raise ValueError("limit must be positive")
        resolved_project_id = project_id or self.project_id(
            owner=owner,
            project_number=project_number,
        )
        if not resolved_project_id:
            raise RuntimeError("Project #5 item pagination requires a non-empty project ID")

        cursor: str | None = None
        seen_cursors: set[str] = set()
        seen_item_ids: set[str] = set()
        items: list[dict[str, Any]] = []
        page_count = 0
        page_size = min(limit, PROJECT_ITEM_GRAPHQL_PAGE_SIZE)

        while True:
            ensure_project_graphql_budget(
                expected_graphql_requests=1,
                min_graphql_remaining=min_graphql_remaining,
            )
            payload = self._run_project_item_page(
                project_id=resolved_project_id,
                page_size=page_size,
                cursor=cursor,
            )
            connection = self._project_item_connection(payload)
            page_items = [self._normalize_project_item(item) for item in connection["nodes"]]
            for item in page_items:
                item_id = str(item["id"])
                if item_id in seen_item_ids:
                    raise RuntimeError(f"project item pagination repeated item ID: {item_id}")
                seen_item_ids.add(item_id)
            items.extend(page_items)
            page_count += 1

            page_info = connection["page_info"]
            has_next_page = page_info["has_next_page"]
            end_cursor = page_info["end_cursor"]
            if not has_next_page:
                self.last_item_fetch_stats = ProjectItemFetchStats(
                    pages=page_count,
                    accumulated_items=len(items),
                )
                return items
            if not isinstance(end_cursor, str) or not end_cursor:
                raise RuntimeError(
                    "project item pagination reported hasNextPage without a non-empty endCursor"
                )
            if end_cursor == cursor or end_cursor in seen_cursors:
                raise RuntimeError(f"project item pagination repeated cursor: {end_cursor}")
            seen_cursors.add(end_cursor)
            cursor = end_cursor

    def _run_project_item_page(
        self,
        *,
        project_id: str,
        page_size: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        """Fetch one validated GraphQL project-item page."""

        args: tuple[str, ...] = (
            "api",
            "graphql",
            "-f",
            f"query={PROJECT_ITEM_GRAPHQL_QUERY}",
            "-f",
            f"projectId={project_id}",
            "-F",
            f"first={page_size}",
        )
        if cursor is not None:
            args += ("-f", f"after={cursor}")
        payload = self.run_json(*args)
        errors = payload.get("errors")
        if errors:
            if not isinstance(errors, list):
                raise RuntimeError("GitHub GraphQL project-item response has malformed errors")
            messages = [
                str(error.get("message") or error) if isinstance(error, dict) else str(error)
                for error in errors
            ]
            raise RuntimeError("GitHub GraphQL project-item query failed: " + "; ".join(messages))
        return payload

    @staticmethod
    def _project_item_connection(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate one GraphQL project-item connection and normalize page metadata."""

        data = payload.get("data")
        node = data.get("node") if isinstance(data, dict) else None
        if not isinstance(node, dict) or node.get("__typename") != "ProjectV2":
            raise RuntimeError("GitHub GraphQL project-item response is missing a ProjectV2 node")
        connection = node.get("items")
        if not isinstance(connection, dict):
            raise RuntimeError("GitHub GraphQL project-item connection is missing")
        nodes = connection.get("nodes")
        page_info = connection.get("pageInfo")
        if not isinstance(nodes, list) or any(not isinstance(item, dict) for item in nodes):
            raise RuntimeError("GitHub GraphQL project-item nodes are malformed")
        if not isinstance(page_info, dict):
            raise RuntimeError("GitHub GraphQL project-item pageInfo is missing")
        if not {"hasNextPage", "endCursor"}.issubset(page_info):
            raise RuntimeError("GitHub GraphQL project-item pageInfo is missing required fields")
        has_next_page = page_info.get("hasNextPage")
        end_cursor = page_info.get("endCursor")
        if type(has_next_page) is not bool:
            raise RuntimeError("GitHub GraphQL project-item hasNextPage is malformed")
        if end_cursor is not None and not isinstance(end_cursor, str):
            raise RuntimeError("GitHub GraphQL project-item endCursor is malformed")
        return {
            "nodes": nodes,
            "page_info": {
                "has_next_page": has_next_page,
                "end_cursor": end_cursor,
            },
        }

    @staticmethod
    def _normalize_project_item(item: dict[str, Any]) -> dict[str, Any]:
        """Project the GraphQL item shape onto the existing score-sync item contract."""

        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise RuntimeError("GitHub GraphQL project item has no stable ID")
        normalized = {
            "id": item_id,
            "content": GhProjectClient._normalize_project_content(
                item_id,
                item.get("content"),
            ),
        }
        normalized.update(
            GhProjectClient._normalize_project_field_values(item_id, item.get("fieldValues"))
        )
        return normalized

    @staticmethod
    def _normalize_project_content(item_id: str, content_raw: Any) -> dict[str, Any] | None:
        """Normalize issue/PR content while preserving draft or empty project items."""

        if content_raw is None:
            return None
        if not isinstance(content_raw, dict):
            raise RuntimeError(f"project item {item_id} has malformed content")
        content_type = content_raw.get("__typename")
        if not isinstance(content_type, str) or not content_type:
            raise RuntimeError(f"project item {item_id} has malformed content type")
        content = {"type": content_type}
        for key in ("number", "title"):
            if key in content_raw:
                content[key] = content_raw[key]
        repository = content_raw.get("repository")
        if isinstance(repository, dict):
            name_with_owner = repository.get("nameWithOwner")
            if isinstance(name_with_owner, str) and name_with_owner:
                content["repository"] = name_with_owner
        return content

    @staticmethod
    def _normalize_project_field_values(item_id: str, field_values: Any) -> dict[str, Any]:
        """Validate and normalize the field-value connection for one project item."""

        if not isinstance(field_values, dict):
            raise RuntimeError(f"project item {item_id} has malformed fieldValues")
        value_nodes = field_values.get("nodes")
        field_page_info = field_values.get("pageInfo")
        if not isinstance(value_nodes, list) or any(
            not isinstance(value, dict) for value in value_nodes
        ):
            raise RuntimeError(f"project item {item_id} has malformed field values")
        if not isinstance(field_page_info, dict) or not {
            "hasNextPage",
            "endCursor",
        }.issubset(field_page_info):
            raise RuntimeError(f"project item {item_id} field values have malformed pageInfo")
        if field_page_info.get("hasNextPage") is not False or (
            field_page_info.get("endCursor") is not None
            and not isinstance(field_page_info.get("endCursor"), str)
        ):
            raise RuntimeError(
                f"project item {item_id} field values are incomplete; refusing a partial read"
            )
        normalized: dict[str, Any] = {}
        for value in value_nodes:
            field = GhProjectClient._normalize_project_field_value(value)
            if field is not None:
                name, field_value = field
                normalized[name[:1].lower() + name[1:]] = field_value
        return normalized

    @staticmethod
    def _normalize_project_field_value(value: dict[str, Any]) -> tuple[str, Any] | None:
        """Map one supported GraphQL project field-value union to its visible name and value."""

        field_name = GhProjectClient._project_item_field_name(value)
        if field_name is None:
            return None
        value_type = value.get("__typename")
        if not isinstance(value_type, str):
            return None
        value_key = {
            "ProjectV2ItemFieldNumberValue": "number",
            "ProjectV2ItemFieldSingleSelectValue": "name",
            "ProjectV2ItemFieldTextValue": "text",
        }.get(value_type)
        if value_key is None:
            return None
        return field_name, value.get(value_key)

    @staticmethod
    def _project_item_field_name(value: dict[str, Any]) -> str | None:
        """Return a recognized project field name from one field-value node."""

        field = value.get("field")
        if not isinstance(field, dict):
            return None
        name = field.get("name")
        return name if isinstance(name, str) and name else None

    def item_list_until_issue(
        self,
        *,
        owner: str,
        project_number: int,
        issue_number: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return one issue-backed item through the project query surface.

        Newer GitHub CLI/API combinations support a server-side Projects
        ``--query`` filter. Older CLI versions reject that flag, so the helper
        falls back to the script-owned cursor paginator with a per-page quota
        guard. Both paths exact-match the issue number locally, and neither
        treats a capped partial list as proof that the target is absent.
        """

        if limit <= 0:
            raise ValueError("limit must be positive")

        try:
            payload = self._run_project_command(
                "item-list",
                owner=owner,
                project_number=project_number,
                extra_args=(
                    "--limit",
                    str(limit),
                    "--query",
                    f"is:issue {issue_number}",
                ),
                as_json=True,
            )
        except RuntimeError as exc:
            if not self._is_unsupported_item_query_error(exc):
                raise
        else:
            return self._select_exact_issue_item(
                list(payload["items"]),
                issue_number=issue_number,
                limit=limit,
                context=f"gh project item-list query for issue #{issue_number}",
            )

        complete_items = self.item_list_paginated(
            owner=owner,
            project_number=project_number,
            limit=limit,
        )
        return self._select_exact_issue_item(
            complete_items,
            issue_number=issue_number,
            limit=len(complete_items) + 1,
            context=f"complete cursor lookup for issue #{issue_number}",
        )

    @staticmethod
    def _is_unsupported_item_query_error(error: RuntimeError) -> bool:
        """Recognize CLI/API query incompatibility without hiding real failures."""

        details = str(error).lower()
        return any(
            marker in details
            for marker in (
                "unknown flag: --query",
                "unknown option: --query",
                "flag provided but not defined: --query",
                "unknown argument: --query",
                "unknown argument query",
                "unknown field 'query'",
                'unknown field "query"',
                "does not support --query",
                "does not support query",
                "unsupported query",
            )
        )

    @staticmethod
    def _select_exact_issue_item(
        items: list[dict[str, Any]],
        *,
        issue_number: int,
        limit: int,
        context: str,
    ) -> list[dict[str, Any]]:
        """Return an exact issue match or fail closed on a possibly capped list."""

        for item in items:
            content = item.get("content")
            if isinstance(content, dict) and content.get("type") == "Issue":
                if content.get("number") == issue_number:
                    return [item]

        assert_not_truncated(items, limit=limit, context=context)
        return []

    def update_number_field(
        self,
        *,
        item_id: str,
        field_id: str,
        project_id: str,
        number: float,
    ) -> None:
        """Write a numeric field value back to the project item."""

        number_literal = _numeric_field_literal(number)
        # ``gh project item-edit --number`` parses its argument as float32 on
        # the supported CLI version. Values such as ``0.7`` therefore reach
        # GraphQL as a binary float with more than eight decimal places and
        # are rejected by GitHub Projects. Keep the validated decimal literal
        # in the GraphQL document so the server sees the intended value.
        payload = self.run_json(
            "api",
            "graphql",
            "-f",
            "query="
            + f"""
mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!) {{
  updateProjectV2ItemFieldValue(input: {{
    projectId: $projectId
    itemId: $itemId
    fieldId: $fieldId
    value: {{ number: {number_literal} }}
  }}) {{
    projectV2Item {{ id }}
  }}
}}
""".strip(),
            "-F",
            f"projectId={project_id}",
            "-F",
            f"itemId={item_id}",
            "-F",
            f"fieldId={field_id}",
        )
        data = payload.get("data")
        mutation = data.get("updateProjectV2ItemFieldValue") if isinstance(data, dict) else None
        updated_item = mutation.get("projectV2Item") if isinstance(mutation, dict) else None
        if (
            not isinstance(updated_item, dict)
            or not isinstance(updated_item.get("id"), str)
            or not updated_item["id"]
        ):
            raise RuntimeError(
                "GitHub Projects numeric update returned no updated project item: "
                + json.dumps(payload, sort_keys=True)
            )


#: GitHub Projects numeric fields reject values with more than 8 decimal
#: places, and ``Format.General`` style floats can produce scientific
#: notation. Every written literal is quantized to at most 8 decimal places
#: and validated against this shape before the GraphQL mutation.
_NUMERIC_FIELD_LITERAL_RE = re.compile(r"-?\d+(\.\d{1,8})?")


def _numeric_field_literal(number: float) -> str:
    """Return a plain decimal literal with at most 8 decimal places.

    Raises:
        ValueError: When the quantized value would not match GitHub's
            documented numeric shape (never scientific notation).
    """
    literal = f"{number:.8f}".rstrip("0").rstrip(".")
    if not literal or not _NUMERIC_FIELD_LITERAL_RE.fullmatch(literal):
        raise ValueError(
            "numeric project value exceeds the documented 8-decimal limit: "
            f"{literal!r} (from {number!r})"
        )
    return literal


def field_map(fields: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index fields by their visible project name."""

    return {str(field["name"]): field for field in fields}


def ensure_required_fields(
    client: GhProjectClient,
    *,
    owner: str,
    project_number: int,
) -> dict[str, dict[str, Any]]:
    """Create missing numeric fields required by the score model."""

    fields = field_map(client.field_list(owner=owner, project_number=project_number))
    created_missing_field = False
    for name in REQUIRED_NUMBER_FIELDS:
        if name not in fields:
            client.ensure_number_field(owner=owner, project_number=project_number, name=name)
            created_missing_field = True
    if created_missing_field:
        return field_map(client.field_list(owner=owner, project_number=project_number))
    return fields


def build_previews(
    items: Iterable[dict[str, Any]],
    *,
    alpha: float,
    round_digits: int,
    issue_number: int | None,
    skip_statuses: set[str],
    only_empty: bool = False,
) -> list[SyncPreview]:
    """Compute score updates for the eligible project items.

    When ``only_empty`` is set, items that already have a ``Priority Score`` are skipped, so the
    auto-fill loop only assesses never-scored issues and never churns existing (often human-set)
    priorities.
    """

    previews: list[SyncPreview] = []
    for item in items:
        status = str(item.get("status", ""))
        if status in skip_statuses:
            continue

        content = item.get("content") or {}
        if content.get("type") != "Issue":
            continue

        raw_number = content.get("number")
        if not isinstance(raw_number, int) or raw_number <= 0:
            continue
        number = raw_number
        if issue_number is not None and number != issue_number:
            continue

        old_score = _coerce_float(field_value(item, PRIORITY_SCORE_FIELD))
        if only_empty and old_score is not None:
            continue

        inputs = normalize_inputs(item)
        score = round(compute_priority_score(inputs, alpha=alpha), round_digits)
        previews.append(
            SyncPreview(
                issue_number=number,
                title=str(content["title"]),
                status=status,
                old_score=old_score,
                new_score=score,
                inputs=inputs,
            )
        )
    return previews


def _pending_score_updates(
    previews: Sequence[SyncPreview],
    items_by_issue: dict[int, dict[str, Any]],
    *,
    dry_run: bool,
    round_digits: int,
) -> list[tuple[SyncPreview, dict[str, Any]]]:
    """Return score mutations that remain after no-op and dry-run filtering."""
    if dry_run:
        return []
    return [
        (preview, items_by_issue[preview.issue_number])
        for preview in previews
        if preview.old_score is None
        or not math.isclose(
            preview.old_score,
            preview.new_score,
            rel_tol=1e-9,
            abs_tol=10 ** (-round_digits),
        )
    ]


def write_summary(
    path: Path,
    previews: Sequence[SyncPreview],
    eligibility_plan: dict[str, Any] | None = None,
) -> None:
    """Persist a machine-readable sync summary."""

    payload = {
        "items": [
            {
                "issue_number": preview.issue_number,
                "title": preview.title,
                "status": preview.status,
                "old_score": preview.old_score,
                "new_score": preview.new_score,
                "inputs": asdict(preview.inputs),
            }
            for preview in previews
        ]
    }
    if eligibility_plan is not None:
        payload["eligibility_plan"] = eligibility_plan
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _project_metadata(
    client: GhProjectClient, options: SyncOptions
) -> tuple[dict[str, dict[str, Any]], str]:
    """Resolve project metadata, using a local cache only for no-write reads."""
    cached = load_project_cache(
        options.cache_file,
        owner=options.owner,
        project_number=options.project_number,
    )
    required = {*REQUIRED_NUMBER_FIELDS, EFFORT_FIELD}
    if options.dry_run and cached is not None and required.issubset(cached.fields):
        return cached.fields, cached.project_id

    fields = (
        ensure_required_fields(
            client,
            owner=options.owner,
            project_number=options.project_number,
        )
        if options.ensure_fields
        else field_map(
            client.field_list(owner=options.owner, project_number=options.project_number)
        )
    )
    project_id = client.project_id(owner=options.owner, project_number=options.project_number)
    return fields, project_id


def _normalized_issue_snapshot(snapshot: dict[str, Any], *, issue_number: int) -> dict[str, Any]:
    """Validate a client-provided issue snapshot into a deterministic compare shape."""

    if snapshot.get("number") != issue_number:
        raise RuntimeError(f"issue #{issue_number} REST snapshot has a mismatched number")
    title = snapshot.get("title")
    state = snapshot.get("state")
    updated_at = snapshot.get("updated_at")
    labels_raw = snapshot.get("labels")
    if (
        not isinstance(title, str)
        or not isinstance(state, str)
        or not isinstance(updated_at, str)
        or not isinstance(labels_raw, list)
        or any(not isinstance(label, str) or not label for label in labels_raw)
    ):
        raise RuntimeError(f"issue #{issue_number} REST snapshot is malformed")
    return {
        "number": issue_number,
        "title": title,
        "state": state.upper(),
        "updated_at": updated_at,
        "labels": sorted(set(labels_raw)),
    }


def _eligibility_entry(
    preview: SyncPreview,
    item: dict[str, Any],
    *,
    decision: str,
    reason_code: str,
    issue_updated_at: str | None,
) -> dict[str, Any]:
    """Build one stable eligibility-plan row."""

    return {
        "issue_number": preview.issue_number,
        "project_item_id": str(item.get("id", "")),
        "project_status": str(item.get("status", "")),
        "decision": decision,
        "reason_code": reason_code,
        "issue_updated_at": issue_updated_at,
    }


def _plan_counts(entries: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count stable eligibility decisions."""

    return {
        decision: sum(entry.get("decision") == decision for entry in entries)
        for decision in ("eligible", "skipped", "blocked")
    }


def _classify_issue_for_score(
    snapshot: dict[str, Any],
    *,
    project_title: str,
) -> tuple[str, str]:
    """Classify one exact REST issue snapshot for default score admission."""

    if snapshot["state"] != "OPEN":
        return "skipped", "issue_terminal"
    labels = set(snapshot["labels"])
    conflicting_labels = labels & AMBIGUOUS_OR_BLOCKING_LABELS
    if READY_LABEL in labels and conflicting_labels:
        return "blocked", "issue_ambiguous"
    if conflicting_labels:
        if labels & {"decision-required", "needs-triage"}:
            return "blocked", "issue_ambiguous"
        return "skipped", "issue_not_ready"
    if READY_LABEL not in labels:
        return "skipped", "issue_not_ready"
    if snapshot["title"] != project_title:
        return "blocked", "project_issue_title_stale"
    return "eligible", "open_ready_exact_state"


def _malformed_issue_entries(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report unscored issue-backed rows that lack a valid issue number."""

    entries: list[dict[str, Any]] = []
    for item in items:
        content = item.get("content")
        if not isinstance(content, dict) or content.get("type") != "Issue":
            continue
        if _coerce_float(field_value(item, PRIORITY_SCORE_FIELD)) is not None:
            continue
        raw_number = content.get("number")
        if isinstance(raw_number, int) and raw_number > 0:
            continue
        entries.append(
            {
                "issue_number": None,
                "project_item_id": str(item.get("id", "")),
                "project_status": str(item.get("status", "")),
                "decision": "blocked",
                "reason_code": "malformed_issue_number",
                "issue_updated_at": None,
            }
        )
    return entries


def _build_eligibility_plan(
    client: GhProjectClient,
    options: SyncOptions,
    previews: Sequence[SyncPreview],
    items_by_issue: dict[int, dict[str, Any]],
    items: Sequence[dict[str, Any]],
) -> tuple[list[SyncPreview], dict[int, dict[str, Any]]]:
    """Build a complete live-state plan before guarded default-score writes."""

    ensure_project_graphql_budget(
        expected_graphql_requests=0,
        min_graphql_remaining=options.min_graphql_remaining,
        expected_core_requests=len(previews) + 1,
    )
    entries = _malformed_issue_entries(items)
    snapshots: dict[int, dict[str, Any]] = {}
    eligible: list[SyncPreview] = []
    for preview in previews:
        item = items_by_issue[preview.issue_number]
        content = item.get("content")
        project_repo = content.get("repository") if isinstance(content, dict) else None
        if project_repo != options.repo:
            entries.append(
                _eligibility_entry(
                    preview,
                    item,
                    decision="blocked",
                    reason_code="project_repo_mismatch",
                    issue_updated_at=None,
                )
            )
            continue
        project_status = str(item.get("status", "")).strip()
        if not project_status:
            entries.append(
                _eligibility_entry(
                    preview,
                    item,
                    decision="blocked",
                    reason_code="project_status_unavailable",
                    issue_updated_at=None,
                )
            )
            continue
        if (
            project_status in options.skip_statuses
            or project_status.casefold() in TERMINAL_PROJECT_STATUSES
        ):
            entries.append(
                _eligibility_entry(
                    preview,
                    item,
                    decision="skipped",
                    reason_code="project_status_terminal",
                    issue_updated_at=None,
                )
            )
            continue
        try:
            snapshot = _normalized_issue_snapshot(
                client.issue_snapshot(repo=options.repo, issue_number=preview.issue_number),
                issue_number=preview.issue_number,
            )
        except (RuntimeError, ValueError, TypeError):
            entries.append(
                _eligibility_entry(
                    preview,
                    item,
                    decision="blocked",
                    reason_code="issue_state_unavailable",
                    issue_updated_at=None,
                )
            )
            continue
        decision, reason_code = _classify_issue_for_score(
            snapshot,
            project_title=preview.title,
        )
        entries.append(
            _eligibility_entry(
                preview,
                item,
                decision=decision,
                reason_code=reason_code,
                issue_updated_at=snapshot["updated_at"],
            )
        )
        if decision == "eligible":
            eligible.append(preview)
            snapshots[preview.issue_number] = snapshot
    client.last_eligibility_plan = {
        "schema": "project_priority_eligibility_plan.v1",
        "status": "planned",
        "counts": _plan_counts(entries),
        "writes_performed": False,
        "items": entries,
    }
    return eligible, snapshots


def _project_item_compare_state(item: dict[str, Any]) -> dict[str, Any]:
    """Return the exact Project item fields guarded before mutation."""

    content = item.get("content")
    if not isinstance(content, dict):
        return {"malformed": True}
    return {
        "id": item.get("id"),
        "status": item.get("status"),
        "content_type": content.get("type"),
        "repository": content.get("repository"),
        "issue_number": content.get("number"),
        "title": content.get("title"),
        "priority_score": field_value(item, PRIORITY_SCORE_FIELD),
    }


def _mark_drift(
    plan: dict[str, Any],
    *,
    issue_number: int,
    reason_code: str,
) -> None:
    """Convert one planned eligible row into a fail-closed drift row."""

    for entry in plan["items"]:
        if entry["issue_number"] == issue_number:
            entry["decision"] = "blocked"
            entry["reason_code"] = reason_code
            break
    plan["status"] = "blocked_drift"
    plan["counts"] = _plan_counts(plan["items"])
    plan["writes_performed"] = False


def _revalidate_guarded_updates(
    client: GhProjectClient,
    options: SyncOptions,
    updates: Sequence[tuple[SyncPreview, dict[str, Any]]],
    issue_snapshots: dict[int, dict[str, Any]],
) -> bool:
    """Re-read every issue and Project item before allowing the first write."""

    plan = client.last_eligibility_plan
    if plan is None:
        raise RuntimeError("guarded score updates require an eligibility plan")
    ensure_project_graphql_budget(
        expected_graphql_requests=2 * len(updates),
        min_graphql_remaining=options.min_graphql_remaining,
        expected_core_requests=len(updates) + 1,
    )
    for preview, original_item in updates:
        try:
            current_issue = _normalized_issue_snapshot(
                client.issue_snapshot(repo=options.repo, issue_number=preview.issue_number),
                issue_number=preview.issue_number,
            )
        except (RuntimeError, ValueError, TypeError):
            _mark_drift(plan, issue_number=preview.issue_number, reason_code="issue_state_drift")
            return False
        if current_issue != issue_snapshots[preview.issue_number]:
            _mark_drift(plan, issue_number=preview.issue_number, reason_code="issue_state_drift")
            return False
        try:
            current_items = client.item_list_until_issue(
                owner=options.owner,
                project_number=options.project_number,
                issue_number=preview.issue_number,
                limit=options.limit,
            )
        except (RuntimeError, ValueError, TypeError):
            _mark_drift(
                plan,
                issue_number=preview.issue_number,
                reason_code="project_item_unavailable",
            )
            return False
        if len(current_items) != 1 or _project_item_compare_state(
            current_items[0]
        ) != _project_item_compare_state(original_item):
            _mark_drift(
                plan,
                issue_number=preview.issue_number,
                reason_code="project_item_drift",
            )
            return False
    return True


def _fetch_score_items(
    client: GhProjectClient,
    options: SyncOptions,
    *,
    project_id: str,
) -> list[dict[str, Any]]:
    """Fetch the targeted item or a proven-complete unscoped item inventory."""

    if options.issue_number is not None:
        return client.item_list_until_issue(
            owner=options.owner,
            project_number=options.project_number,
            issue_number=options.issue_number,
            limit=options.limit,
        )
    return client.item_list_paginated(
        owner=options.owner,
        project_number=options.project_number,
        project_id=project_id,
        limit=options.limit,
        min_graphql_remaining=options.min_graphql_remaining,
    )


def _index_issue_items(items: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index valid issue-backed items without inventing malformed identifiers."""

    indexed: dict[int, dict[str, Any]] = {}
    for item in items:
        content = item.get("content")
        if not isinstance(content, dict) or content.get("type") != "Issue":
            continue
        issue_number = content.get("number")
        if isinstance(issue_number, int) and issue_number > 0:
            indexed[issue_number] = item
    return indexed


def _apply_score_updates(
    client: GhProjectClient,
    options: SyncOptions,
    updates: Sequence[tuple[SyncPreview, dict[str, Any]]],
    *,
    issue_snapshots: dict[int, dict[str, Any]],
    score_field_id: str,
    project_id: str,
) -> bool:
    """Revalidate guarded updates and apply the complete admitted mutation set."""

    if not updates:
        return True
    if options.only_empty:
        if not _revalidate_guarded_updates(client, options, updates, issue_snapshots):
            return False
    else:
        ensure_project_graphql_budget(
            expected_graphql_requests=len(updates),
            min_graphql_remaining=options.min_graphql_remaining,
        )
    plan = client.last_eligibility_plan
    attempted_rows: list[dict[str, Any]] = []
    writes_performed = 0
    try:
        for preview, item in updates:
            row = {
                "issue_number": preview.issue_number,
                "item_id": str(item.get("id")),
                "written": False,
            }
            attempted_rows.append(row)
            client.update_number_field(
                item_id=str(item["id"]),
                field_id=score_field_id,
                project_id=project_id,
                number=preview.new_score,
            )
            row["written"] = True
            writes_performed += 1
    except (RuntimeError, ValueError, TypeError) as exc:
        # A rejected write (for example GitHub's numeric-shape enforcement) must
        # not crash the run before the structured summary is written: record the
        # attempted rows, the writes that landed, and the retryability verdict.
        if plan is not None:
            plan["status"] = "apply_failed"
            plan["failure"] = {
                "error": str(exc),
                "attempted_rows": attempted_rows,
                "writes_performed": writes_performed,
                "retryable": True,
            }
        return False
    if plan is not None:
        plan["attempted_rows"] = attempted_rows
        plan["writes_performed_count"] = writes_performed
    return True


def _finalize_eligibility_plan(
    client: GhProjectClient,
    options: SyncOptions,
    *,
    updates: Sequence[tuple[SyncPreview, dict[str, Any]]],
) -> None:
    """Record the terminal no-write, dry-run, or applied plan status."""

    if not options.only_empty:
        return
    plan = client.last_eligibility_plan
    if plan is None:
        raise RuntimeError("only-empty score sync completed without an eligibility plan")
    if options.dry_run:
        plan["status"] = "dry_run"
    elif updates:
        plan["status"] = "applied"
        plan["writes_performed"] = True
    else:
        plan["status"] = "no_eligible_items"


def sync_scores(
    client: GhProjectClient,
    options: SyncOptions,
) -> list[SyncPreview]:
    """Compute and optionally write derived scores for project items."""

    fields, project_id = _project_metadata(client, options)
    missing = [name for name in (*REQUIRED_NUMBER_FIELDS, EFFORT_FIELD) if name not in fields]
    if missing:
        raise ValueError(
            "project is missing required fields: "
            + ", ".join(sorted(missing))
            + ". Re-run with --ensure-fields or create them manually."
        )

    items = _fetch_score_items(client, options, project_id=project_id)
    previews = build_previews(
        items,
        alpha=options.alpha,
        round_digits=options.round_digits,
        issue_number=options.issue_number,
        skip_statuses=set() if options.only_empty else options.skip_statuses,
        only_empty=options.only_empty,
    )
    items_by_issue = _index_issue_items(items)

    score_field_id = str(fields[PRIORITY_SCORE_FIELD]["id"])
    # The guarded write set joins to the eligibility plan's eligible rows only:
    # a row the plan skipped (for example a terminal Project status) must never
    # produce an item-edit, even when its preview carries a computed score.
    score_previews = previews
    issue_snapshots: dict[int, dict[str, Any]] = {}
    if options.only_empty:
        eligible_previews, issue_snapshots = _build_eligibility_plan(
            client,
            options,
            previews,
            items_by_issue,
            items,
        )
        eligible_numbers = {preview.issue_number for preview in eligible_previews}
        score_previews = [
            preview for preview in previews if preview.issue_number in eligible_numbers
        ]

    updates = _pending_score_updates(
        score_previews,
        items_by_issue,
        dry_run=options.dry_run,
        round_digits=options.round_digits,
    )

    if not _apply_score_updates(
        client,
        options,
        updates,
        issue_snapshots=issue_snapshots,
        score_field_id=score_field_id,
        project_id=project_id,
    ):
        return []
    _finalize_eligibility_plan(client, options, updates=updates)

    return score_previews


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for priority-score project commands.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="Compute and sync the derived Priority Score field.")
    sync.add_argument("--owner", default="ll7", help="GitHub owner of the target project.")
    sync.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Issue repository used for live REST eligibility verification.",
    )
    sync.add_argument("--project-number", type=int, default=5, help="Projects v2 number.")
    sync.add_argument(
        "--limit",
        type=int,
        default=400,
        help="Per-page item bound for the complete cursor-paginated project scan.",
    )
    sync.add_argument(
        "--min-graphql-remaining",
        type=int,
        default=DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
        help="GraphQL quota safety margin retained after the estimated sync budget.",
    )
    sync.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="Effort dampening exponent."
    )
    sync.add_argument(
        "--round-digits",
        type=int,
        default=6,
        help="Decimal digits kept in the written Priority Score.",
    )
    sync.add_argument(
        "--ensure-fields",
        action="store_true",
        help="Create the missing numeric score fields before syncing.",
    )
    sync.add_argument(
        "--issue-number",
        type=int,
        help="Restrict sync to one issue-backed project item.",
    )
    sync.add_argument(
        "--cache-file",
        type=Path,
        default=Path(".github/cache/project5.json"),
        help=(
            "Optional local Project #5 metadata cache; used for validated no-write reads and "
            "ignored for score writes until live IDs are refreshed."
        ),
    )
    sync.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the score updates without writing them back.",
    )
    sync.add_argument(
        "--summary-file",
        type=Path,
        help="Optional JSON file describing the computed sync results.",
    )
    sync.add_argument(
        "--skip-status",
        action="append",
        default=["Done"],
        help="Project status values to skip. Repeatable. Default: Done.",
    )
    sync.add_argument(
        "--only-empty",
        action="store_true",
        help=(
            "Only assess issues whose Priority Score is currently empty; never re-score or "
            "overwrite an existing priority. Used by the autopilot auto-fill loop to stay cheap "
            "and avoid churning human-set priorities. Missing read:project access returns a "
            "non-fatal blocked result so live-label ordering can continue."
        ),
    )
    return parser


def _blocked_project_scope_payload(
    *, owner: str, project_number: int, error: MissingProjectScopeError
) -> dict[str, Any]:
    """Build the stable non-fatal payload for the autopilot's blocked auto-fill path."""
    return {
        "status": "blocked",
        "reason": "missing_project_scope",
        "owner": owner,
        "project_number": project_number,
        "required_scopes": list(error.required_scopes),
        "items": [],
        "non_fatal": True,
        "writes_performed": False,
        "fallback": "live-label ordering",
        "message": (
            "Project #5 priority auto-fill was skipped because the GitHub token lacks "
            + ", ".join(error.required_scopes)
            + ". Continue with live-label ordering; refresh the token scope before retrying."
        ),
    }


def _blocked_project_quota_payload(
    *, owner: str, project_number: int, decision: dict[str, Any], non_fatal: bool
) -> dict[str, Any]:
    """Build a resumable no-write payload for a quota-blocked Project #5 pass."""
    return {
        "status": "quota_blocked",
        "reason": decision.get("reason", "project_quota_blocked"),
        "owner": owner,
        "project_number": project_number,
        "quota": decision,
        "items": [],
        "non_fatal": non_fatal,
        "writes_performed": False,
        "resume_after": decision.get("resume_after"),
        "message": decision.get(
            "message",
            "Project #5 operation is blocked until the configured quota safety margin is available.",
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = _build_parser().parse_args(argv)
    if args.command != "sync":
        raise ValueError(f"unsupported command: {args.command}")

    options = SyncOptions(
        owner=args.owner,
        project_number=args.project_number,
        ensure_fields=args.ensure_fields,
        limit=args.limit,
        alpha=args.alpha,
        round_digits=args.round_digits,
        issue_number=args.issue_number,
        dry_run=args.dry_run,
        skip_statuses=set(args.skip_status),
        only_empty=args.only_empty,
        min_graphql_remaining=args.min_graphql_remaining,
        cache_file=args.cache_file,
        repo=args.repo,
    )
    decision = project_quota_decision(options)
    if decision["status"] != "ok":
        payload = _blocked_project_quota_payload(
            owner=args.owner,
            project_number=args.project_number,
            decision=decision,
            non_fatal=args.only_empty,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.only_empty else 2

    client = GhProjectClient()
    try:
        previews = sync_scores(client, options)
    except MissingProjectScopeError as exc:
        if not args.only_empty:
            raise
        print(
            json.dumps(
                _blocked_project_scope_payload(
                    owner=args.owner,
                    project_number=args.project_number,
                    error=exc,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except ProjectQuotaBlockedError as exc:
        if not args.only_empty:
            raise
        print(
            json.dumps(
                _blocked_project_quota_payload(
                    owner=args.owner,
                    project_number=args.project_number,
                    decision=exc.decision,
                    non_fatal=True,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.summary_file is not None:
        write_summary(args.summary_file, previews, client.last_eligibility_plan)

    print(
        json.dumps(
            {
                "project_number": args.project_number,
                "owner": args.owner,
                "item_fetch": (
                    asdict(client.last_item_fetch_stats)
                    if client.last_item_fetch_stats is not None
                    else None
                ),
                "eligibility_plan": client.last_eligibility_plan,
                "writes_performed": bool(
                    client.last_eligibility_plan
                    and client.last_eligibility_plan.get("writes_performed") is True
                ),
                "items": [
                    {
                        "issue_number": preview.issue_number,
                        "title": preview.title,
                        "status": preview.status,
                        "old_score": preview.old_score,
                        "new_score": preview.new_score,
                        "inputs": asdict(preview.inputs),
                    }
                    for preview in previews
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
