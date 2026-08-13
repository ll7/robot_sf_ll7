"""Sync a derived priority score into GitHub Project #5 items.

The score is intentionally simple and benchmark-oriented:

    score = improvement * success_probability * time_criticality * unlock_factor
            / effort_hours**alpha

This helper is the deterministic `gh` fallback for Project #5 score sync. It is
intentionally kept scriptable for local/manual batch routing even as
interactive issue/PR/project work moves toward GitHub MCP / app tools.

The helper reads issue-backed project items via `gh project item-list`, applies
defaults and clamping for missing or invalid inputs, and writes the derived
numeric score back to a `Priority Score` project field.

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
    """Estimate bounded Project #5 GraphQL requests before any project mutation."""
    item_requests = (
        1 if options.issue_number is not None else max(1, math.ceil(max(options.limit, 1) / 100))
    )
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

        args = (
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

    def item_list_until_issue(
        self,
        *,
        owner: str,
        project_number: int,
        issue_number: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return one issue-backed item through the project query surface.

        The numeric query narrows discovery before ``gh`` applies ``--limit``;
        the exact-number check then rejects textual false positives. If the
        bounded query reaches its cap without finding the issue, fail closed
        because the exact match may have been truncated.
        """

        if limit <= 0:
            raise ValueError("limit must be positive")
        payload = self._run_project_command(
            "item-list",
            owner=owner,
            project_number=project_number,
            extra_args=(
                "--query",
                str(issue_number),
                "--limit",
                str(limit),
            ),
            as_json=True,
        )
        items = list(payload["items"])
        for item in items:
            content = item.get("content")
            if isinstance(content, dict) and content.get("type") == "Issue":
                if content.get("number") == issue_number:
                    return [item]

        assert_not_truncated(
            items,
            limit=limit,
            context=f"gh project item-list query for issue #{issue_number}",
        )
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

        number_literal = format(number, ".8g")
        self.run(
            "project",
            "item-edit",
            "--id",
            item_id,
            "--project-id",
            project_id,
            "--field-id",
            field_id,
            "--number",
            number_literal,
        )


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
        if not isinstance(raw_number, int) or raw_number < 0:
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


def write_summary(path: Path, previews: Sequence[SyncPreview]) -> None:
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

    if options.issue_number is not None:
        # Targeted sync: locate the single issue-backed item without requiring a
        # full untruncated project page. Query before applying the cap, then verify
        # the exact issue number so textual false positives cannot be updated
        # (issue #5870). The original fail-closed full-project guard is preserved
        # for unscoped sync below.
        items = client.item_list_until_issue(
            owner=options.owner,
            project_number=options.project_number,
            issue_number=options.issue_number,
        )
    else:
        # Unscoped sync keeps the explicit truncation protection: a full project
        # page at the cap must fail closed rather than silently skip items.
        items = client.item_list(
            owner=options.owner,
            project_number=options.project_number,
            limit=options.limit,
        )
    previews = build_previews(
        items,
        alpha=options.alpha,
        round_digits=options.round_digits,
        issue_number=options.issue_number,
        skip_statuses=options.skip_statuses,
        only_empty=options.only_empty,
    )
    items_by_issue: dict[int, dict[str, Any]] = {}
    for item in items:
        content = item.get("content")
        if not isinstance(content, dict) or content.get("type") != "Issue":
            continue
        issue_number = content.get("number")
        if not isinstance(issue_number, int) or issue_number < 0:
            continue
        items_by_issue[issue_number] = item

    score_field_id = str(fields[PRIORITY_SCORE_FIELD]["id"])
    for preview in previews:
        item = items_by_issue[preview.issue_number]
        if preview.old_score is not None and math.isclose(
            preview.old_score,
            preview.new_score,
            rel_tol=1e-9,
            abs_tol=10 ** (-options.round_digits),
        ):
            continue
        if options.dry_run:
            continue
        client.update_number_field(
            item_id=str(item["id"]),
            field_id=score_field_id,
            project_id=project_id,
            number=preview.new_score,
        )

    return previews


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for priority-score project commands.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="Compute and sync the derived Priority Score field.")
    sync.add_argument("--owner", default="ll7", help="GitHub owner of the target project.")
    sync.add_argument("--project-number", type=int, default=5, help="Projects v2 number.")
    sync.add_argument("--limit", type=int, default=400, help="Maximum project items to inspect.")
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

    try:
        previews = sync_scores(GhProjectClient(), options)
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

    if args.summary_file is not None:
        write_summary(args.summary_file, previews)

    print(
        json.dumps(
            {
                "project_number": args.project_number,
                "owner": args.owner,
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
