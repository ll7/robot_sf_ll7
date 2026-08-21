#!/usr/bin/env python3
"""Re-surface blocked issues whose explicit issue-graph blocker is resolved.

The watcher is deliberately fail-closed.  It reads the complete issue thread,
parses the ``blocked-triage-v1`` comment block, and evaluates only an explicit
closed/merged issue-or-pull-request condition or a strict v1 adapter mapping.
All references for the run are resolved by one batched GraphQL request.
Unsupported or malformed adapter mappings remain ``unevaluatable``; adapter
and API failures are explicit ``error`` results.

Report mode is the default.  ``--apply`` may add ``needs-triage`` to fired
issues, but this module has no code path that writes ``state:ready``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from scripts.dev._gh_rest import parse_json as _parse_json
from scripts.dev._gh_rest import run_gh_command as _run_gh_command
from scripts.dev.gh_issue_rest import fetch_issue_with_comments
from scripts.dev.gh_pr_label_rest import add_label

DEFAULT_REPO = "ll7/robot_sf_ll7"
TRIAGE_SCHEMA = "blocked-triage-v1"
REPORT_SCHEMA = "blocked_queue_watch_report.v1"
TRIAGE_LABEL = "needs-triage"
FORBIDDEN_WRITE_LABEL = "state:ready"
MAX_ISSUES = 100
MAX_COMMENT_PAGES = 3
ADAPTER_SCHEMA_VERSION = 1
MAX_REPO_PREDICATE_FILES = 256
MAX_REPO_PREDICATE_FILE_BYTES = 1_000_000
MAX_REPO_PREDICATE_TOTAL_BYTES = 8_000_000
REPO_PREDICATE_ROOTS = frozenset({"configs", "docs", "robot_sf", "scripts", "tests"})

TRIAGE_MARKER_RE = re.compile(r"<!--\s*blocked-triage-v1\b[^>]*-->", re.IGNORECASE)
YAML_FENCE_RE = re.compile(r"```(?:yaml|yml)\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
ISSUE_REF_RE = re.compile(r"(?<![\w-])#(?P<number>[1-9][0-9]*)\b")
GRAPH_STATE_RE = re.compile(
    r"(?<![\w-])#(?:[1-9][0-9]*)\b[^\n.]{0,120}\b(?:closed|merged)\b"
    r"|\b(?:closed|merged)\b[^\n.]{0,120}(?<![\w-])#(?:[1-9][0-9]*)\b",
    re.IGNORECASE,
)
NEGATIVE_CONDITION_RE = re.compile(
    r"\b(?:currently\s+false|may\s+never\s+fire|not\s+(?:met|satisfied)|"
    r"no\s+longer\s+(?:true|valid|available))\b",
    re.IGNORECASE,
)

EvaluationStatus = Literal["fired", "not-fired", "unevaluatable", "error"]

AdapterKind = Literal["path_presence", "external_probe", "repo_predicate"]


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """One versioned, allow-listed machine-readable unblock adapter."""

    version: int
    kind: AdapterKind
    name: str
    path: str = ""
    path_type: Literal["any", "file", "directory"] = "any"
    text: str = ""
    minimum_remaining: int | None = None


@dataclass(frozen=True, slots=True)
class AdapterOutcome:
    """Result of one adapter evaluation, including its proof provenance."""

    status: EvaluationStatus
    reason: str
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TriageRecord:
    """Validated fields from one ``blocked-triage-v1`` comment."""

    blocker_class: str
    unblock_condition: str
    watcher: str
    next_check_at: str
    last_meaningful_progress_at: str
    source_url: str
    adapter: AdapterSpec | None = None


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    """Result of parsing the latest triage block in an issue thread."""

    status: Literal["ok", "missing", "malformed"]
    record: TriageRecord | None
    reason: str


@dataclass(frozen=True, slots=True)
class IssueCandidate:
    """Blocked issue inventory row plus its complete thread."""

    number: int
    title: str
    body: str
    labels: tuple[str, ...]
    url: str
    comments: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class DependencyNode:
    """GraphQL state for one referenced issue or pull request."""

    number: int
    kind: Literal["issue", "pull_request"]
    state: str
    merged_at: str
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One fail-closed watcher decision."""

    number: int
    title: str
    status: EvaluationStatus
    tier: str
    reason: str
    references: tuple[int, ...] = ()
    resolved_references: tuple[int, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphQLResolution:
    """One batched dependency lookup result."""

    nodes: Mapping[int, DependencyNode | None]
    error: str = ""


def _run_gh(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Run a GitHub CLI command without masking failures."""
    return _run_gh_command(args, timeout=timeout)


def _split_repo(repo: str) -> tuple[str, str]:
    """Split and validate an ``owner/repository`` identifier."""
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid repository {repo!r}; expected owner/repository")
    return parts[0], parts[1]


def _json_result(result: subprocess.CompletedProcess[str], *, what: str) -> Any:
    """Decode one CLI JSON response or raise a bounded diagnostic."""
    payload, error = _parse_json(result, what=what)
    if error:
        raise RuntimeError(error)
    return payload


def _as_text(value: Any) -> str:
    """Normalize nullable API values without turning ``None`` into text."""
    return "" if value is None else str(value)


def _valid_relative_path(value: Any) -> bool:
    """Return whether a path is a bounded repository-relative path."""
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    if "\\" in value or ":" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and "." not in path.parts and ".." not in path.parts


ADAPTER_ALLOWED_FIELDS: dict[str, set[str]] = {
    "path_presence": {"version", "kind", "name", "path", "path_type"},
    "external_probe": {"version", "kind", "name", "minimum_remaining"},
    "repo_predicate": {"version", "kind", "name", "path", "text"},
}


def _parse_path_presence_adapter(
    value: Mapping[str, Any], *, name: str
) -> tuple[AdapterSpec | None, str | None]:
    """Validate the path-presence adapter fields."""
    if name != "path_exists":
        return None, "path_presence adapter name must be path_exists"
    if not _valid_relative_path(value.get("path")):
        return None, "path_presence path must be repository-relative"
    path_type = value.get("path_type", "any")
    if path_type not in {"any", "file", "directory"}:
        return None, "path_presence path_type must be any, file, or directory"
    return (
        AdapterSpec(
            version=ADAPTER_SCHEMA_VERSION,
            kind="path_presence",
            name=name,
            path=value["path"],
            path_type=path_type,
        ),
        None,
    )


def _parse_external_probe_adapter(
    value: Mapping[str, Any], *, name: str
) -> tuple[AdapterSpec | None, str | None]:
    """Validate the fixed GitHub quota probe fields."""
    if name != "github_graphql_quota":
        return None, "external_probe name is not allow-listed"
    minimum = value.get("minimum_remaining")
    if type(minimum) is not int or not 0 <= minimum <= 5_000:
        return None, "github_graphql_quota minimum_remaining must be an integer in [0, 5000]"
    return (
        AdapterSpec(
            version=ADAPTER_SCHEMA_VERSION,
            kind="external_probe",
            name=name,
            minimum_remaining=minimum,
        ),
        None,
    )


def _parse_repo_predicate_adapter(
    value: Mapping[str, Any], *, name: str
) -> tuple[AdapterSpec | None, str | None]:
    """Validate the bounded literal repository predicate fields."""
    if name != "text_present":
        return None, "repo_predicate name must be text_present"
    if not _valid_relative_path(value.get("path")):
        return None, "repo_predicate path must be repository-relative"
    text = value.get("text")
    if not isinstance(text, str) or not text or len(text) > 256 or "\x00" in text:
        return None, "repo_predicate text must be a non-empty string of at most 256 characters"
    top_level = Path(value["path"]).parts[0]
    if top_level not in REPO_PREDICATE_ROOTS:
        return None, "repo_predicate path is outside the allow-listed source roots"
    return (
        AdapterSpec(
            version=ADAPTER_SCHEMA_VERSION,
            kind="repo_predicate",
            name=name,
            path=value["path"],
            text=text,
        ),
        None,
    )


def _parse_adapter_spec(value: Any) -> tuple[AdapterSpec | None, str | None]:
    """Validate one strict v1 adapter mapping without executing issue prose."""
    if not isinstance(value, Mapping):
        return None, "adapter must be a mapping"
    if type(value.get("version")) is not int or value.get("version") != ADAPTER_SCHEMA_VERSION:
        return None, f"adapter version must be {ADAPTER_SCHEMA_VERSION}"
    kind = value.get("kind")
    name = value.get("name")
    if not isinstance(kind, str) or kind not in ADAPTER_ALLOWED_FIELDS:
        return None, "adapter kind is not allow-listed"
    if not isinstance(name, str) or not name:
        return None, "adapter name is required"
    unknown = sorted(
        str(key)
        for key in value
        if not isinstance(key, str) or key not in ADAPTER_ALLOWED_FIELDS[kind]
    )
    if unknown:
        return None, "adapter has unsupported field(s): " + ", ".join(unknown)
    if kind == "path_presence":
        return _parse_path_presence_adapter(value, name=name)
    if kind == "external_probe":
        return _parse_external_probe_adapter(value, name=name)
    return _parse_repo_predicate_adapter(value, name=name)


def _inventory(
    repo: str, *, runner: Callable[[list[str]], subprocess.CompletedProcess[str]]
) -> list[dict[str, Any]]:
    """Read the bounded open ``state:blocked`` issue inventory."""
    result = runner(
        [
            "api",
            f"repos/{repo}/issues?state=open&labels=state:blocked&per_page={MAX_ISSUES}&page=1",
        ]
    )
    payload = _json_result(result, what="blocked issue inventory")
    if not isinstance(payload, list):
        raise RuntimeError("blocked issue inventory was not a JSON list")
    if len(payload) >= MAX_ISSUES:
        raise RuntimeError(
            f"blocked issue inventory reached the {MAX_ISSUES}-row limit; refusing truncated input"
        )
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or type(row.get("number")) is not int:
            raise RuntimeError(f"blocked issue inventory row {index} is malformed")
        if "pull_request" in row:
            continue
        labels = row.get("labels")
        if not isinstance(labels, list):
            raise RuntimeError(f"blocked issue inventory row {row['number']} has malformed labels")
        rows.append(row)
    return rows


def _thread_candidate(
    row: Mapping[str, Any],
    *,
    repo: str,
    thread_reader: Callable[[int, str, int], Mapping[str, Any]],
) -> tuple[IssueCandidate | None, str | None]:
    """Read one complete issue thread and normalize it into a candidate."""
    number = row["number"]
    try:
        thread = thread_reader(number, repo, MAX_COMMENT_PAGES)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        return None, f"issue #{number} thread read raised {type(exc).__name__}: {exc}"
    if thread.get("status") != "ok":
        return None, f"issue #{number} thread read failed: {_as_text(thread.get('error'))}"
    raw_labels = row.get("labels", [])
    labels = tuple(
        sorted(
            _as_text(label.get("name"))
            for label in raw_labels
            if isinstance(label, Mapping) and _as_text(label.get("name"))
        )
    )
    comments = thread.get("comments", [])
    if not isinstance(comments, list):
        return None, f"issue #{number} thread comments are malformed"
    return (
        IssueCandidate(
            number=number,
            title=_as_text(row.get("title")),
            body=_as_text(row.get("body")),
            labels=labels,
            url=_as_text(row.get("html_url", row.get("url"))),
            comments=tuple(item for item in comments if isinstance(item, Mapping)),
        ),
        None,
    )


def collect_candidates(
    repo: str = DEFAULT_REPO,
    *,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    thread_reader: Callable[[int, str, int], Mapping[str, Any]] | None = None,
) -> tuple[list[IssueCandidate], list[str]]:
    """Collect blocked issues and complete threads, preserving read failures."""
    command_runner = runner or _run_gh
    reader = thread_reader or (
        lambda number, issue_repo, max_pages: fetch_issue_with_comments(
            number,
            repo=issue_repo,
            max_comment_pages=max_pages,
        )
    )
    candidates: list[IssueCandidate] = []
    errors: list[str] = []
    for row in _inventory(repo, runner=command_runner):
        candidate, error = _thread_candidate(row, repo=repo, thread_reader=reader)
        if candidate is not None:
            candidates.append(candidate)
        if error:
            errors.append(error)
    return candidates, errors


def _source_items(candidate: IssueCandidate) -> Iterable[Mapping[str, Any]]:
    """Yield the issue body followed by comments in display order."""
    if candidate.body:
        yield {"body": candidate.body, "url": candidate.url}
    yield from candidate.comments


def _parse_triage_payload(payload: Any, *, source_url: str) -> ParseOutcome:
    """Validate one decoded triage payload and its optional adapter."""
    if not isinstance(payload, Mapping):
        return ParseOutcome("malformed", None, "triage YAML must be a mapping")
    required = (
        "blocker_class",
        "unblock_condition",
        "watcher",
        "next_check_at",
        "last_meaningful_progress_at",
    )
    missing = [key for key in required if not _as_text(payload.get(key)).strip()]
    if missing:
        return ParseOutcome("malformed", None, f"triage YAML is missing: {', '.join(missing)}")
    adapter: AdapterSpec | None = None
    if "adapter" in payload:
        adapter, adapter_error = _parse_adapter_spec(payload["adapter"])
        if adapter_error:
            return ParseOutcome("malformed", None, adapter_error)
    record = TriageRecord(
        blocker_class=_as_text(payload["blocker_class"]).strip().lower(),
        unblock_condition=_as_text(payload["unblock_condition"]).strip(),
        watcher=_as_text(payload["watcher"]).strip(),
        next_check_at=_as_text(payload["next_check_at"]).strip(),
        last_meaningful_progress_at=_as_text(payload["last_meaningful_progress_at"]).strip(),
        source_url=source_url,
        adapter=adapter,
    )
    return ParseOutcome("ok", record, "parsed")


def parse_triage_record(candidate: IssueCandidate) -> ParseOutcome:
    """Parse the latest complete ``blocked-triage-v1`` block."""
    matches: list[tuple[str, str]] = []
    for source in _source_items(candidate):
        body = _as_text(source.get("body"))
        marker = TRIAGE_MARKER_RE.search(body)
        if marker is None:
            continue
        fence = YAML_FENCE_RE.search(body, marker.end())
        if fence is None:
            matches.append(("", _as_text(source.get("url"))))
            continue
        matches.append((fence.group(1), _as_text(source.get("url"))))
    if not matches:
        return ParseOutcome("missing", None, "blocked-triage-v1 marker is absent")

    yaml_text, source_url = matches[-1]
    if not yaml_text:
        return ParseOutcome("malformed", None, "triage marker has no YAML fence")
    try:
        payload = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return ParseOutcome("malformed", None, f"triage YAML is invalid: {exc}")
    return _parse_triage_payload(payload, source_url=source_url)


def extract_references(record: TriageRecord, *, issue_number: int) -> tuple[int, ...]:
    """Return unique referenced issue/PR numbers, excluding the current issue."""
    text = f"{record.unblock_condition}\n{record.watcher}"
    refs = {int(match.group("number")) for match in ISSUE_REF_RE.finditer(text)}
    refs.discard(issue_number)
    return tuple(sorted(refs))


def is_issue_graph_condition(record: TriageRecord, *, issue_number: int) -> bool:
    """Return whether the record explicitly describes a closed/merged graph edge."""
    condition = record.unblock_condition
    return bool(
        extract_references(record, issue_number=issue_number)
        and GRAPH_STATE_RE.search(condition)
        and not NEGATIVE_CONDITION_RE.search(condition)
    )


def _build_graphql_query(references: Iterable[int]) -> tuple[str, dict[int, str]]:
    """Build one bounded GraphQL query for issue-or-PR aliases."""
    refs = tuple(sorted(set(references)))
    fields: list[str] = []
    aliases: dict[int, str] = {}
    for index, number in enumerate(refs, start=1):
        alias = f"item_{index}"
        aliases[number] = alias
        fields.append(
            f"{alias}: issueOrPullRequest(number: {number}) "
            "{ __typename ... on Issue { number state title url } "
            "... on PullRequest { number state mergedAt title url } }"
        )
    query = "query($owner: String!, $repo: String!) { repository(owner: $owner, name: $repo) { "
    query += " ".join(fields)
    query += " } }"
    return query, aliases


def resolve_dependencies(
    references: Iterable[int],
    *,
    repo: str = DEFAULT_REPO,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> GraphQLResolution:
    """Resolve all references in exactly one GraphQL request."""
    refs = tuple(sorted(set(references)))
    if not refs:
        return GraphQLResolution({})
    try:
        owner, repo_name = _split_repo(repo)
        query, aliases = _build_graphql_query(refs)
    except ValueError as exc:
        return GraphQLResolution({}, str(exc))
    command_runner = runner or _run_gh
    result = command_runner(
        [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"repo={repo_name}",
        ]
    )
    try:
        payload = _json_result(result, what="blocked dependency GraphQL query")
    except RuntimeError as exc:
        return GraphQLResolution({}, str(exc))
    if not isinstance(payload, Mapping):
        return GraphQLResolution({}, "blocked dependency GraphQL response was not an object")
    errors = payload.get("errors")
    if errors:
        messages = [
            _as_text(error.get("message")) if isinstance(error, Mapping) else _as_text(error)
            for error in errors
        ]
        return GraphQLResolution({}, "GraphQL returned errors: " + "; ".join(messages))
    data = payload.get("data")
    repository = data.get("repository") if isinstance(data, Mapping) else None
    if not isinstance(repository, Mapping):
        return GraphQLResolution({}, "GraphQL response is missing repository data")
    nodes: dict[int, DependencyNode | None] = {}
    for number, alias in aliases.items():
        raw = repository.get(alias)
        if raw is None:
            nodes[number] = None
            continue
        if not isinstance(raw, Mapping) or type(raw.get("number")) is not int:
            return GraphQLResolution({}, f"GraphQL dependency node #{number} is malformed")
        kind: Literal["issue", "pull_request"] = (
            "pull_request" if raw.get("__typename") == "PullRequest" else "issue"
        )
        nodes[number] = DependencyNode(
            number=number,
            kind=kind,
            state=_as_text(raw.get("state")).upper(),
            merged_at=_as_text(raw.get("mergedAt")),
            title=_as_text(raw.get("title")),
            url=_as_text(raw.get("url")),
        )
    return GraphQLResolution(nodes)


def _node_resolved(node: DependencyNode, condition: str) -> bool:
    """Evaluate a dependency node against its explicit closed/merged wording."""
    if node.kind == "pull_request" and re.search(r"\bmerged\b", condition, re.IGNORECASE):
        return bool(node.merged_at)
    return node.state == "CLOSED"


def _adapter_provenance(spec: AdapterSpec, *, repo_root: Path, source_url: str) -> dict[str, Any]:
    """Build stable provenance shared by all local adapter outcomes."""
    return {
        "schema": "blocked_queue_adapter_provenance.v1",
        "adapter_version": spec.version,
        "adapter_kind": spec.kind,
        "adapter_name": spec.name,
        "triage_source_url": source_url,
        "repository_root": str(repo_root),
    }


def _resolve_repository_path(repo_root: Path, relative_path: str) -> tuple[Path | None, str | None]:
    """Resolve a path while preserving the repository-root boundary."""
    try:
        root = repo_root.resolve()
        target = (root / relative_path).resolve()
        target.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"repository path cannot be resolved safely: {exc}"
    return target, None


def _evaluate_path_presence(
    spec: AdapterSpec, *, repo_root: Path, source_url: str
) -> AdapterOutcome:
    """Evaluate the allow-listed repository path presence adapter."""
    target, error = _resolve_repository_path(repo_root, spec.path)
    provenance = _adapter_provenance(spec, repo_root=repo_root, source_url=source_url)
    provenance.update({"path": spec.path, "path_type": spec.path_type})
    if error or target is None:
        return AdapterOutcome("error", error or "repository path is unavailable", provenance)
    try:
        exists = target.exists()
        matches_type = (
            spec.path_type == "any"
            or (spec.path_type == "file" and target.is_file())
            or (spec.path_type == "directory" and target.is_dir())
        )
    except OSError as exc:
        return AdapterOutcome("error", f"path presence probe failed: {exc}", provenance)
    provenance["observed_exists"] = exists
    provenance["observed_type_match"] = matches_type
    if exists and matches_type:
        return AdapterOutcome("fired", "allow-listed repository path is present", provenance)
    return AdapterOutcome("not-fired", "allow-listed repository path is absent", provenance)


def _evaluate_external_probe(
    spec: AdapterSpec,
    *,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
    repo_root: Path,
    source_url: str,
) -> AdapterOutcome:
    """Evaluate the single fixed external probe supported by v1."""
    args = ["api", "rate_limit", "--jq", ".resources.graphql.remaining"]
    provenance = _adapter_provenance(spec, repo_root=repo_root, source_url=source_url)
    provenance["command"] = ["gh", *args]
    try:
        result = runner(args)
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        return AdapterOutcome("error", f"external probe failed to run: {exc}", provenance)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return AdapterOutcome(
            "error", f"external probe failed: {detail or result.returncode}", provenance
        )
    try:
        remaining = int(result.stdout.strip())
    except (TypeError, ValueError) as exc:
        return AdapterOutcome("error", f"external probe returned invalid quota: {exc}", provenance)
    minimum = spec.minimum_remaining
    if minimum is None:
        return AdapterOutcome("error", "external probe minimum is missing", provenance)
    provenance["observed_remaining"] = remaining
    provenance["minimum_remaining"] = minimum
    if remaining >= minimum:
        return AdapterOutcome(
            "fired", "GitHub GraphQL quota is above the configured margin", provenance
        )
    return AdapterOutcome(
        "not-fired", "GitHub GraphQL quota is below the configured margin", provenance
    )


def _discover_predicate_files(target: Path) -> tuple[list[Path], str | None]:
    """Discover at most the configured number of predicate files."""
    try:
        if not target.exists():
            return [], None
        if target.is_file():
            return [target], None
        if not target.is_dir():
            return [], "repository predicate target is not a file or directory"
        files: list[Path] = []
        for path in target.rglob("*"):
            if path.is_file() and not path.is_symlink():
                files.append(path)
                if len(files) > MAX_REPO_PREDICATE_FILES:
                    return [], "repository predicate file bound exceeded"
        return sorted(files), None
    except OSError as exc:
        return [], f"repository predicate discovery failed: {exc}"


def _read_predicate_file(path: Path, *, total_bytes: int) -> tuple[bytes | None, int, str | None]:
    """Read one predicate file with per-file and total byte bounds."""
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_REPO_PREDICATE_FILE_BYTES + 1)
    except OSError as exc:
        return None, total_bytes, f"repository predicate read failed: {exc}"
    if len(payload) > MAX_REPO_PREDICATE_FILE_BYTES:
        return None, total_bytes, "repository predicate file-size bound exceeded"
    total_bytes += len(payload)
    if total_bytes > MAX_REPO_PREDICATE_TOTAL_BYTES:
        return None, total_bytes, "repository predicate total-byte bound exceeded"
    return payload, total_bytes, None


def _evaluate_repo_predicate(
    spec: AdapterSpec, *, repo_root: Path, source_url: str
) -> AdapterOutcome:
    """Search bounded, allow-listed repository text without executing commands."""
    target, error = _resolve_repository_path(repo_root, spec.path)
    provenance = _adapter_provenance(spec, repo_root=repo_root, source_url=source_url)
    provenance.update(
        {
            "path": spec.path,
            "text": spec.text,
            "max_files": MAX_REPO_PREDICATE_FILES,
            "max_file_bytes": MAX_REPO_PREDICATE_FILE_BYTES,
            "max_total_bytes": MAX_REPO_PREDICATE_TOTAL_BYTES,
        }
    )
    if error or target is None:
        return AdapterOutcome(
            "error", error or "repository predicate path is unavailable", provenance
        )
    files, discovery_error = _discover_predicate_files(target)
    if discovery_error:
        return AdapterOutcome("error", discovery_error, provenance)
    if not files:
        provenance["scanned_files"] = 0
        return AdapterOutcome("not-fired", "repository predicate path is absent", provenance)

    needle = spec.text.encode("utf-8")
    total_bytes = 0
    for scanned_files, path in enumerate(files, start=1):
        payload, total_bytes, read_error = _read_predicate_file(path, total_bytes=total_bytes)
        if read_error:
            return AdapterOutcome("error", read_error, provenance)
        if payload is not None and needle in payload:
            provenance["scanned_files"] = scanned_files
            provenance["matched_path"] = str(path.relative_to(repo_root.resolve()))
            return AdapterOutcome(
                "fired", "allow-listed repository text predicate matched", provenance
            )
    provenance["scanned_files"] = len(files)
    provenance["scanned_bytes"] = total_bytes
    return AdapterOutcome(
        "not-fired", "allow-listed repository text predicate did not match", provenance
    )


def _evaluate_adapter(
    spec: AdapterSpec,
    *,
    repo_root: Path,
    source_url: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
    external_cache: dict[AdapterSpec, AdapterOutcome] | None = None,
) -> AdapterOutcome:
    """Dispatch one validated adapter without allowing arbitrary commands."""
    if spec.kind == "path_presence":
        return _evaluate_path_presence(spec, repo_root=repo_root, source_url=source_url)
    if spec.kind == "repo_predicate":
        return _evaluate_repo_predicate(spec, repo_root=repo_root, source_url=source_url)
    if spec.kind == "external_probe":
        if external_cache is not None and spec in external_cache:
            cached = external_cache[spec]
            provenance = dict(cached.provenance)
            provenance["triage_source_url"] = source_url
            return AdapterOutcome(cached.status, cached.reason, provenance)
        outcome = _evaluate_external_probe(
            spec,
            runner=runner,
            repo_root=repo_root,
            source_url=source_url,
        )
        if external_cache is not None:
            external_cache[spec] = outcome
        return outcome
    return AdapterOutcome("error", f"unsupported adapter kind: {spec.kind}")


def evaluate_candidate(
    candidate: IssueCandidate,
    parse_outcome: ParseOutcome,
    resolution: GraphQLResolution,
    *,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    repo_root: Path | None = None,
    external_cache: dict[AdapterSpec, AdapterOutcome] | None = None,
) -> Evaluation:
    """Classify one candidate without performing writes."""
    if parse_outcome.status != "ok" or parse_outcome.record is None:
        return Evaluation(
            candidate.number,
            candidate.title,
            "unevaluatable",
            "parser",
            parse_outcome.reason,
        )
    record = parse_outcome.record
    references = extract_references(record, issue_number=candidate.number)
    if FORBIDDEN_WRITE_LABEL in candidate.labels:
        return Evaluation(
            candidate.number,
            candidate.title,
            "unevaluatable",
            "safety",
            "contradictory state:ready label is present; refusing automatic triage",
            references,
        )
    if record.adapter is not None:
        if references:
            return Evaluation(
                candidate.number,
                candidate.title,
                "unevaluatable",
                f"adapter-{record.adapter.kind}",
                "adapter conditions must not mix issue/PR references",
                references,
                provenance={"triage_source_url": record.source_url},
            )
        outcome = _evaluate_adapter(
            record.adapter,
            repo_root=repo_root or Path.cwd(),
            source_url=record.source_url,
            runner=runner or _run_gh,
            external_cache=external_cache,
        )
        return Evaluation(
            candidate.number,
            candidate.title,
            outcome.status,
            f"adapter-{record.adapter.kind}",
            outcome.reason,
            provenance=outcome.provenance,
        )
    if not references:
        return Evaluation(
            candidate.number,
            candidate.title,
            "unevaluatable",
            "unsupported",
            "condition has no referenced issue or pull request",
        )
    if not is_issue_graph_condition(record, issue_number=candidate.number):
        return Evaluation(
            candidate.number,
            candidate.title,
            "unevaluatable",
            "unsupported",
            "condition is not an explicit closed/merged issue-graph predicate",
            references,
        )
    if resolution.error:
        return Evaluation(
            candidate.number,
            candidate.title,
            "error",
            "tier-1-issue-graph",
            f"dependency API error: {resolution.error}",
            references,
        )
    missing = tuple(reference for reference in references if reference not in resolution.nodes)
    if missing or any(resolution.nodes[reference] is None for reference in references):
        unknown = missing or tuple(
            reference for reference in references if resolution.nodes[reference] is None
        )
        return Evaluation(
            candidate.number,
            candidate.title,
            "unevaluatable",
            "tier-1-issue-graph",
            "unresolvable issue/PR reference(s): " + ", ".join(f"#{ref}" for ref in unknown),
            references,
        )
    nodes = [resolution.nodes[reference] for reference in references]
    resolved = tuple(
        reference
        for reference, node in zip(references, nodes, strict=True)
        if node is not None and _node_resolved(node, record.unblock_condition)
    )
    any_semantics = bool(re.search(r"\bor\b", record.unblock_condition, re.IGNORECASE))
    is_fired = bool(resolved) if any_semantics else len(resolved) == len(references)
    if is_fired:
        return Evaluation(
            candidate.number,
            candidate.title,
            "fired",
            "tier-1-issue-graph",
            "all required issue/PR dependency states are resolved"
            if not any_semantics
            else "at least one alternative issue/PR dependency state is resolved",
            references,
            resolved,
        )
    return Evaluation(
        candidate.number,
        candidate.title,
        "not-fired",
        "tier-1-issue-graph",
        "referenced issue/PR dependency remains open or unmerged",
        references,
        resolved,
    )


def evaluate_candidates(
    candidates: Iterable[IssueCandidate],
    *,
    repo: str = DEFAULT_REPO,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    repo_root: Path | None = None,
) -> list[Evaluation]:
    """Parse and evaluate candidates, using one dependency batch for the run."""
    rows = list(candidates)
    parsed = [(candidate, parse_triage_record(candidate)) for candidate in rows]
    refs = {
        reference
        for candidate, outcome in parsed
        if outcome.status == "ok"
        and outcome.record is not None
        and is_issue_graph_condition(outcome.record, issue_number=candidate.number)
        for reference in extract_references(outcome.record, issue_number=candidate.number)
    }
    resolution = resolve_dependencies(refs, repo=repo, runner=runner)
    external_cache: dict[AdapterSpec, AdapterOutcome] = {}
    return [
        evaluate_candidate(
            candidate,
            outcome,
            resolution,
            runner=runner,
            repo_root=repo_root,
            external_cache=external_cache,
        )
        for candidate, outcome in parsed
    ]


def apply_fired(
    evaluations: Iterable[Evaluation],
    candidates: Iterable[IssueCandidate],
    *,
    repo: str = DEFAULT_REPO,
    writer: Callable[[int, str], Mapping[str, Any]] | None = None,
) -> tuple[list[int], list[str]]:
    """Add only ``needs-triage`` to fired, non-contradictory candidates."""
    candidate_by_number = {candidate.number: candidate for candidate in candidates}
    label_writer = writer or (lambda number, label: add_label(number, label, repo=repo))
    applied: list[int] = []
    errors: list[str] = []
    for evaluation in evaluations:
        if evaluation.status != "fired":
            continue
        candidate = candidate_by_number.get(evaluation.number)
        if candidate is None or FORBIDDEN_WRITE_LABEL in candidate.labels:
            errors.append(f"issue #{evaluation.number}: refused triage write due to state:ready")
            continue
        result = label_writer(evaluation.number, TRIAGE_LABEL)
        if result.get("status") != "ok":
            errors.append(f"issue #{evaluation.number}: {_as_text(result.get('error'))}")
            continue
        applied.append(evaluation.number)
    return applied, errors


def _evaluation_errors(evaluations: Iterable[Evaluation]) -> list[str]:
    """Return adapter/API failures that must make the report itself fail closed."""
    return [
        f"issue #{evaluation.number}: {evaluation.reason}"
        for evaluation in evaluations
        if evaluation.status == "error"
    ]


def build_report(
    candidates: Iterable[IssueCandidate],
    evaluations: Iterable[Evaluation],
    *,
    repo: str,
    errors: Iterable[str] = (),
    applied: Iterable[int] = (),
    write_errors: Iterable[str] = (),
    apply_requested: bool = False,
) -> dict[str, Any]:
    """Build the stable JSON report consumed by CI and humans."""
    candidate_rows = list(candidates)
    evaluation_rows = list(evaluations)
    applied_rows = tuple(applied)
    all_errors = [*errors, *_evaluation_errors(evaluation_rows), *write_errors]
    counts = {
        status: sum(row.status == status for row in evaluation_rows)
        for status in ("fired", "not-fired", "unevaluatable", "error")
    }
    return {
        "schema": REPORT_SCHEMA,
        "repo": repo,
        "mode": "apply" if apply_requested else "report-only",
        "inventory_count": len(candidate_rows),
        "summary": {
            "fired": counts["fired"],
            "not_fired": counts["not-fired"],
            "unevaluatable": counts["unevaluatable"],
            "error": counts["error"],
            "errors": len(all_errors),
            "applied": len(applied_rows),
        },
        "evaluations": [asdict(row) for row in evaluation_rows],
        "errors": all_errors,
        "applied_issue_numbers": sorted(set(applied_rows)),
        "safety": {
            "write_label": TRIAGE_LABEL,
            "forbidden_write_label": FORBIDDEN_WRITE_LABEL,
            "state_ready_writes": 0,
        },
        "status": "error" if all_errors else "ok",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repository.")
    parser.add_argument("--apply", action="store_true", help="Add needs-triage to fired issues.")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the report-only or explicitly applied watcher."""
    args = _parser().parse_args(argv)
    errors: list[str] = []
    try:
        candidates, collection_errors = collect_candidates(args.repo)
        errors.extend(collection_errors)
        evaluations = evaluate_candidates(candidates, repo=args.repo)
    except (RuntimeError, ValueError) as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "repo": args.repo,
            "mode": "apply" if args.apply else "report-only",
            "inventory_count": 0,
            "summary": {
                "fired": 0,
                "not_fired": 0,
                "unevaluatable": 0,
                "error": 0,
                "errors": 1,
                "applied": 0,
            },
            "evaluations": [],
            "errors": [str(exc)],
            "applied_issue_numbers": [],
            "safety": {
                "write_label": TRIAGE_LABEL,
                "forbidden_write_label": FORBIDDEN_WRITE_LABEL,
                "state_ready_writes": 0,
            },
            "status": "error",
        }
    else:
        applied: list[int] = []
        write_errors: list[str] = []
        if args.apply and not errors and not _evaluation_errors(evaluations):
            applied, write_errors = apply_fired(evaluations, candidates, repo=args.repo)
        report = build_report(
            candidates,
            evaluations,
            repo=args.repo,
            errors=errors,
            applied=applied,
            write_errors=write_errors,
            apply_requested=args.apply,
        )
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        print(output)
    else:
        print(
            "blocked queue watcher: "
            f"status={report['status']} "
            f"fired={report['summary']['fired']} "
            f"not-fired={report['summary']['not_fired']} "
            f"unevaluatable={report['summary']['unevaluatable']} "
            f"errors={report['summary']['errors']}"
        )
        print(output)
    return 2 if report["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
