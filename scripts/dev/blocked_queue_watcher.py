#!/usr/bin/env python3
"""Re-surface blocked issues whose explicit issue-graph blocker is resolved.

The watcher is deliberately fail-closed.  It reads the complete issue thread,
parses the ``blocked-triage-v1`` comment block, and evaluates only an explicit
closed/merged issue-or-pull-request condition.  All references for the run are
resolved by one batched GraphQL request.  Unsupported path, external, and
in-repository predicates remain ``unevaluatable`` until a separately reviewed
adapter exists.

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
from dataclasses import asdict, dataclass
from typing import Any, Literal

import yaml

from scripts.dev.gh_issue_rest import fetch_issue_with_comments
from scripts.dev.gh_pr_label_rest import add_label

DEFAULT_REPO = "ll7/robot_sf_ll7"
TRIAGE_SCHEMA = "blocked-triage-v1"
REPORT_SCHEMA = "blocked_queue_watch_report.v1"
TRIAGE_LABEL = "needs-triage"
FORBIDDEN_WRITE_LABEL = "state:ready"
MAX_ISSUES = 100
MAX_COMMENT_PAGES = 3

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

EvaluationStatus = Literal["fired", "not-fired", "unevaluatable"]


@dataclass(frozen=True, slots=True)
class TriageRecord:
    """Validated fields from one ``blocked-triage-v1`` comment."""

    blocker_class: str
    unblock_condition: str
    watcher: str
    next_check_at: str
    last_meaningful_progress_at: str
    source_url: str


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


@dataclass(frozen=True, slots=True)
class GraphQLResolution:
    """One batched dependency lookup result."""

    nodes: Mapping[int, DependencyNode | None]
    error: str = ""


def _run_gh(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Run a GitHub CLI command without masking failures."""
    try:
        return subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            ["gh", *args],
            127,
            "",
            "gh CLI not found on PATH; install GitHub CLI and authenticate it",
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            ["gh", *args],
            124,
            "",
            f"gh command timed out after {timeout} seconds",
        )


def _split_repo(repo: str) -> tuple[str, str]:
    """Split and validate an ``owner/repository`` identifier."""
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid repository {repo!r}; expected owner/repository")
    return parts[0], parts[1]


def _json_result(result: subprocess.CompletedProcess[str], *, what: str) -> Any:
    """Decode one CLI JSON response or raise a bounded diagnostic."""
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{what} failed: {detail or f'exit code {result.returncode}'}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{what} returned invalid JSON: {exc}") from exc


def _as_text(value: Any) -> str:
    """Normalize nullable API values without turning ``None`` into text."""
    return "" if value is None else str(value)


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
    record = TriageRecord(
        blocker_class=_as_text(payload["blocker_class"]).strip().lower(),
        unblock_condition=_as_text(payload["unblock_condition"]).strip(),
        watcher=_as_text(payload["watcher"]).strip(),
        next_check_at=_as_text(payload["next_check_at"]).strip(),
        last_meaningful_progress_at=_as_text(payload["last_meaningful_progress_at"]).strip(),
        source_url=source_url,
    )
    return ParseOutcome("ok", record, "parsed")


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


def evaluate_candidate(
    candidate: IssueCandidate,
    parse_outcome: ParseOutcome,
    resolution: GraphQLResolution,
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
            "unevaluatable",
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
    return [evaluate_candidate(candidate, outcome, resolution) for candidate, outcome in parsed]


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
    """Return API failures that must make the report itself fail closed."""
    return [
        f"issue #{evaluation.number}: {evaluation.reason}"
        for evaluation in evaluations
        if evaluation.status == "unevaluatable"
        and evaluation.reason.startswith("dependency API error:")
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
        for status in ("fired", "not-fired", "unevaluatable")
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
            "summary": {"fired": 0, "not_fired": 0, "unevaluatable": 0, "errors": 1, "applied": 0},
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
