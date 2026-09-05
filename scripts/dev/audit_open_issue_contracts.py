#!/usr/bin/env python3
"""Audit every open issue against the canonical implementation-admission contract.

The audit is report-only. It pages through the GitHub REST issues endpoint, excludes pull
requests, re-reads each issue through the canonical exact-item owner, and delegates classification
to :mod:`scripts.dev.issue_implementability`. It never edits labels, bodies, assignments, issue
state, projects, comments, or parent/sub-issue relations.

Offline fixtures provide deterministic regression coverage without GitHub or Git access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.dev import issue_implementability
from scripts.dev._gh_rest import parse_json, run_gh_api

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SCHEMA = "open_issue_contract_audit.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"
# Keep the default developer invocation bounded. A full exact-read evaluation of every open issue
# can exceed the local command timeout as the repository grows; callers that need complete coverage
# must opt into a fresh run with larger explicit limits and preserve the applicable=false result when
# pagination is truncated.
DEFAULT_PAGE_SIZE = 20
DEFAULT_MAX_PAGES = 1
DEFAULT_ITEM_LIMIT = 20

NEXT_ACTIONS: dict[str, str] = {
    "ready": "dispatch_via_goal_issue_admission",
    "needs_spec": "complete_missing_contract_fields",
    "needs_ready_label": "maintainer_preparation_required",
    "parent": "split_or_select_bounded_leaf",
    "human_decision": "prepare_decision_envelope",
    "needs_dependency": "resolve_typed_dependency",
    "needs_compute": "route_to_compute_authority",
    "blocked": "resolve_named_blocker",
    "wrong_owner_repo": "move_or_split_cross_repository_issue",
    "state_conflict": "reconcile_state_labels",
    "stale_running": "reconcile_stale_running_state",
    "assigned": "do_not_duplicate_active_work",
    "already_claimed": "do_not_duplicate_active_work",
    "working": "do_not_duplicate_active_work",
    "review": "do_not_duplicate_active_work",
    "closed": "exclude_non_open_item",
    "error": "repair_or_repeat_exact_read",
}

AUTHORITIES: dict[str, str] = {
    "ready": "goal_issue_admission",
    "needs_spec": "specification_owner",
    "needs_ready_label": "maintainer_preparation",
    "parent": "parent_owner",
    "human_decision": "maintainer_or_author",
    "needs_dependency": "dependency_owner",
    "needs_compute": "compute_authority",
    "blocked": "blocker_owner",
    "wrong_owner_repo": "repository_owner",
    "state_conflict": "lifecycle_owner",
    "stale_running": "lifecycle_owner",
    "assigned": "current_owner",
    "already_claimed": "current_owner",
    "working": "current_owner",
    "review": "current_owner",
    "closed": "terminal_reconciliation",
    "error": "audit_operator",
}


def _stable_json(payload: object) -> str:
    """Return canonical compact JSON for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(payload: object) -> str:
    """Return the SHA-256 digest of canonical JSON."""
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _normalize_named_values(raw: Any, *, field: str, key: str) -> list[str]:
    """Normalize REST strings or named objects to sorted unique values."""
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list")
    values: list[str] = []
    for item in raw:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict) and isinstance(item.get(key), str):
            value = item[key]
        else:
            raise ValueError(f"each {field} entry must be a string or an object with {key!r}")
        value = value.strip()
        if value:
            values.append(value)
    return sorted(set(values))


def _normalize_listing_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one row from the REST issues endpoint."""
    number = raw.get("number")
    if type(number) is not int or number < 1:
        raise ValueError("listing row number must be a positive integer")
    title = raw.get("title")
    state = raw.get("state")
    url = raw.get("html_url", raw.get("url", ""))
    if not isinstance(title, str):
        raise ValueError(f"listing row {number} title must be a string")
    if not isinstance(state, str) or not state.strip():
        raise ValueError(f"listing row {number} state must be a non-empty string")
    if not isinstance(url, str):
        raise ValueError(f"listing row {number} URL must be a string")
    return {
        "number": number,
        "kind": "pull_request" if raw.get("pull_request") is not None else "issue",
        "title": title.strip(),
        "state": state.strip().upper(),
        "url": url,
        "labels": _normalize_named_values(raw.get("labels", []), field="labels", key="name"),
        "assignees": _normalize_named_values(
            raw.get("assignees", []), field="assignees", key="login"
        ),
    }


def _fetch_live_pages(
    *, repo: str, page_size: int, max_pages: int
) -> tuple[list[list[dict[str, Any]]], bool, list[str]]:
    """Fetch bounded REST pages and fail closed when the page budget may truncate results."""
    pages: list[list[dict[str, Any]]] = []
    for page_number in range(1, max_pages + 1):
        path = f"repos/{repo}/issues?state=open&per_page={page_size}&page={page_number}"
        payload, error = parse_json(
            run_gh_api(path),
            what=f"open issues page {page_number}",
        )
        if error:
            return pages, False, [error]
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            return pages, False, [f"open issues page {page_number} must be a JSON array of objects"]
        page = list(payload)
        pages.append(page)
        if len(page) < page_size:
            return pages, True, []
    return (
        pages,
        False,
        [
            f"open issue inventory reached max_pages={max_pages} with a full final page; "
            "pagination may be truncated"
        ],
    )


def _validate_fixture(payload: Any) -> dict[str, Any]:
    """Validate the offline fixture envelope."""
    if not isinstance(payload, dict):
        raise ValueError("fixture must be a JSON object")
    pages = payload.get("pages")
    exact_issues = payload.get("exact_issues")
    claims = payload.get("claims")
    dependencies = payload.get("dependencies", {})
    if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
        raise ValueError("fixture.pages must be a list of page arrays")
    if any(any(not isinstance(row, dict) for row in page) for page in pages):
        raise ValueError("every fixture page row must be an object")
    for field, value in (
        ("exact_issues", exact_issues),
        ("claims", claims),
        ("dependencies", dependencies),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"fixture.{field} must be an object keyed by issue number")
    return {
        "pages": pages,
        "exact_issues": exact_issues,
        "claims": claims,
        "dependencies": dependencies,
    }


def _load_fixture(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate one offline fixture.

    Returns:
        The normalized fixture and the input-file SHA-256 digest.
    """
    raw = path.read_bytes()
    return _validate_fixture(json.loads(raw.decode("utf-8"))), hashlib.sha256(raw).hexdigest()


def _fixture_pagination(
    fixture: Mapping[str, Any], *, page_size: int, max_pages: int
) -> tuple[list[list[dict[str, Any]]], bool, list[str]]:
    """Apply the canonical bounded fixture-pagination decision."""
    raw_pages = fixture.get("pages")
    if not isinstance(raw_pages, list) or any(not isinstance(page, list) for page in raw_pages):
        raise ValueError("fixture.pages must be a list of page arrays")
    if any(any(not isinstance(row, dict) for row in page) for page in raw_pages):
        raise ValueError("every fixture page row must be an object")
    pages = raw_pages[:max_pages]
    complete = bool(pages) and len(raw_pages) <= max_pages and len(pages[-1]) < page_size
    errors = []
    if not complete:
        errors.append("fixture pagination is incomplete under the configured page-size/page-limit")
    return pages, complete, errors


def _fixture_lookup(mapping: Mapping[str, Any], number: int, *, field: str) -> Any:
    """Read one fixture value by its JSON string issue key."""
    key = str(number)
    if key not in mapping:
        raise ValueError(f"fixture.{field} has no entry for issue {number}")
    return mapping[key]


def _fixture_evaluator(
    fixture: Mapping[str, Any],
    *,
    repository: str = DEFAULT_REPO,
    route_preflight: Mapping[str, Any] | None = None,
) -> Callable[[int], dict[str, Any]]:
    """Build an exact issue evaluator backed only by fixture data."""

    def evaluate(number: int) -> dict[str, Any]:
        exact = _fixture_lookup(fixture["exact_issues"], number, field="exact_issues")
        claim = _fixture_lookup(fixture["claims"], number, field="claims")
        if not isinstance(exact, dict):
            raise ValueError(f"fixture exact issue {number} must be an object")
        if not isinstance(claim, dict):
            raise ValueError(f"fixture claim {number} must be an object")
        if exact.get("number") != number:
            raise ValueError(f"fixture exact issue {number} has a mismatched number")
        dependency = fixture["dependencies"].get(str(number))
        if dependency is not None and not isinstance(dependency, dict):
            raise ValueError(f"fixture dependency evaluation {number} must be an object")
        return issue_implementability.evaluate_issue(
            exact,
            claim,
            dependency_evaluation=dependency,
            repository=repository,
            route_preflight=route_preflight,
        )

    return evaluate


def _live_evaluator(
    *, repo: str, remote: str, route_preflight: Mapping[str, Any] | None = None
) -> Callable[[int], dict[str, Any]]:
    """Build a live evaluator that reuses the canonical exact-read and dependency path."""

    def evaluate(number: int) -> dict[str, Any]:
        return issue_implementability.live_issue_report(
            number,
            repo=repo,
            remote=remote,
            repo_root=Path.cwd(),
            route_preflight=route_preflight,
        )

    return evaluate


def _listing_drift(listed: Mapping[str, Any], exact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return field-level differences between the listing row and exact read."""
    drift: list[dict[str, Any]] = []
    for field in ("title", "state", "url", "labels", "assignees"):
        if listed.get(field) != exact.get(field):
            drift.append({"field": field, "listed": listed.get(field), "exact": exact.get(field)})
    return drift


def _error_item(listed: Mapping[str, Any], message: str) -> dict[str, Any]:
    """Build one deterministic fail-closed item packet."""
    return {
        "number": listed.get("number"),
        "title": listed.get("title", ""),
        "url": listed.get("url", ""),
        "labels": listed.get("labels", []),
        "assignees": listed.get("assignees", []),
        "claim": None,
        "observed_classification": "error",
        "classification": "error",
        "admission_reason": "error",
        "reasons": [message],
        "body_sha256": None,
        "contract_fields": {},
        "execution_contract": None,
        "missing_fields": [],
        "dependency_gate": None,
        "listing_drift": [],
        "applicable": False,
        "dispatch_eligible": False,
        "next_action": NEXT_ACTIONS["error"],
        "authority": AUTHORITIES["error"],
    }


def _item_from_report(listed: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one canonical implementability report to a bounded preparation packet."""
    issue = report.get("issue")
    claim = report.get("claim")
    observed = report.get("classification")
    if not isinstance(issue, dict):
        return _error_item(listed, "implementability report has no normalized issue object")
    if not isinstance(claim, dict):
        return _error_item(listed, "implementability report has no normalized claim object")
    if not isinstance(observed, str) or observed not in NEXT_ACTIONS:
        return _error_item(listed, "implementability report has an unknown classification")

    drift = _listing_drift(listed, issue)
    effective = "error" if drift else observed
    contract = report.get("contract") if isinstance(report.get("contract"), dict) else {}
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    missing = contract.get("missing_fields")
    reasons = report.get("reasons")
    normalized_reasons = list(reasons) if isinstance(reasons, list) else []
    if drift:
        normalized_reasons.append("listing state changed before exact issue evaluation")
    applicable = effective != "error"
    return {
        "number": issue.get("number", listed.get("number")),
        "title": issue.get("title", listed.get("title", "")),
        "url": issue.get("url", listed.get("url", "")),
        "labels": issue.get("labels", listed.get("labels", [])),
        "assignees": issue.get("assignees", listed.get("assignees", [])),
        "claim": claim,
        "observed_classification": observed,
        "classification": effective,
        "admission_reason": (
            "listing_drift" if drift else str(report.get("admission_reason") or observed)
        ),
        "reasons": normalized_reasons,
        "body_sha256": contract.get("body_sha256"),
        "contract_fields": fields,
        "execution_contract": report.get("execution_contract"),
        "missing_fields": list(missing) if isinstance(missing, list) else [],
        "dependency_gate": report.get("dependency_gate"),
        "listing_drift": drift,
        "applicable": applicable,
        "dispatch_eligible": (
            applicable
            and effective == "ready"
            and report.get("ready") is True
            and report.get("write_allowed") is True
        ),
        "next_action": NEXT_ACTIONS[effective],
        "authority": AUTHORITIES[effective],
    }


def _evaluate_item(
    listed: Mapping[str, Any], evaluator: Callable[[int], dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate and normalize one issue inside the per-item isolation boundary."""
    result = evaluator(listed["number"])
    if not isinstance(result, dict):
        raise ValueError("implementability evaluator returned a non-object payload")
    return _item_from_report(listed, result)


def _prepare_listing(
    pages: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    """Normalize issue rows, count pull requests, and reject duplicate issue identities."""
    issues: list[dict[str, Any]] = []
    seen: set[int] = set()
    errors: list[str] = []
    counts = {"raw_rows": 0, "issue_rows": 0, "excluded_pull_requests": 0}
    for page_index, page in enumerate(pages, start=1):
        for row_index, raw in enumerate(page, start=1):
            counts["raw_rows"] += 1
            try:
                row = _normalize_listing_row(raw)
            except ValueError as exc:
                errors.append(f"page {page_index} row {row_index}: {exc}")
                continue
            if row["kind"] == "pull_request":
                counts["excluded_pull_requests"] += 1
                continue
            number = row["number"]
            if number in seen:
                errors.append(f"duplicate issue number {number} in paginated listing")
                continue
            seen.add(number)
            issues.append(row)
    issues.sort(key=lambda item: item["number"])
    counts["issue_rows"] = len(issues)
    return issues, counts, errors


def _claim_state(claim: object) -> str:
    """Return one compact claim-state aggregate value."""
    if not isinstance(claim, Mapping) or claim.get("ok") is not True:
        return "unavailable"
    return "claimed" if claim.get("claimed") is True else "unclaimed"


def _summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build aggregate classification, claim, contract, and admission-reason counts."""
    classifications: Counter[str] = Counter()
    next_actions: Counter[str] = Counter()
    claim_states: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    admission_reasons: Counter[str] = Counter()
    executable: list[int] = []
    for item in items:
        classifications[str(item.get("classification", "error"))] += 1
        admission_reasons[str(item.get("admission_reason", "unknown"))] += 1
        next_actions[str(item.get("next_action", NEXT_ACTIONS["error"]))] += 1
        claim_states[_claim_state(item.get("claim"))] += 1
        for field in item.get("missing_fields", []):
            missing_fields[str(field)] += 1
        for label in item.get("labels", []):
            labels[str(label)] += 1
        number = item.get("number")
        if item.get("dispatch_eligible") is True and isinstance(number, int):
            executable.append(number)
    return {
        "classifications": dict(sorted(classifications.items())),
        "next_actions": dict(sorted(next_actions.items())),
        "claim_states": dict(sorted(claim_states.items())),
        "missing_fields": dict(sorted(missing_fields.items())),
        "labels": dict(sorted(labels.items())),
        "admission_reason_histogram": dict(sorted(admission_reasons.items())),
        "not_admitted": dict(
            sorted(
                (reason, count)
                for reason, count in admission_reasons.items()
                if reason != "claimable"
            )
        ),
        "executable_leaf_numbers": sorted(executable),
    }


def _report_digest(report: Mapping[str, Any]) -> str:
    """Return a content digest that excludes only the digest carrier itself."""
    payload = dict(report)
    payload.pop("content_sha256", None)
    return _sha256_json(payload)


def _build_report(
    *,
    repo: str,
    source: str,
    pages: Sequence[Sequence[Mapping[str, Any]]],
    pagination: Mapping[str, Any],
    evaluator: Callable[[int], dict[str, Any]],
    input_sha256: str | None,
) -> dict[str, Any]:
    """Build the deterministic repository-wide audit report."""
    listed, counts, listing_errors = _prepare_listing(pages)
    pagination_complete = pagination.get("complete") is True
    pagination_errors = pagination.get("errors", [])
    page_size = pagination.get("page_size")
    max_pages = pagination.get("max_pages")
    if not isinstance(pagination_errors, list):
        raise ValueError("pagination.errors must be a list")
    if type(page_size) is not int or page_size < 1:
        raise ValueError("pagination.page_size must be a positive integer")
    if type(max_pages) is not int or max_pages < 1:
        raise ValueError("pagination.max_pages must be a positive integer")

    errors = [*(str(error) for error in pagination_errors), *listing_errors]
    items: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="open-issue-audit") as executor:
        for row in listed:
            future = executor.submit(_evaluate_item, row, evaluator)
            if future.cancelled():
                item = _error_item(row, "issue evaluation was cancelled")
            else:
                exception = future.exception()
                if isinstance(exception, (KeyboardInterrupt, SystemExit)):
                    raise exception
                if exception is not None:
                    item = _error_item(row, f"{type(exception).__name__}: {exception}")
                else:
                    item = future.result()
            if item["classification"] == "error":
                errors.append(f"issue {row['number']}: " + "; ".join(item["reasons"]))
            items.append(item)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": repo,
        "source": source,
        "input_sha256": input_sha256,
        "mutation_authorized": False,
        "complete": pagination_complete,
        "applicable": pagination_complete and not errors,
        "pagination": {
            "page_size": page_size,
            "max_pages": max_pages,
            "pages_read": len(pages),
            "resume_hint": (
                "Pagination is intentionally bounded. Rerun a fresh audit with explicit "
                "--page-size/--max-pages values for broader coverage; suffix-only continuation "
                "is unsupported."
                if not pagination_complete
                else None
            ),
            **counts,
        },
        "errors": errors,
        "summary": _summary(items),
        "items": items,
    }
    report["content_sha256"] = _report_digest(report)
    return report


def _markdown_cell(value: object) -> str:
    """Escape a value for one compact Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_markdown(report: Mapping[str, Any], *, item_limit: int, json_report: str | None) -> str:
    """Render a bounded Markdown summary without reproducing issue bodies."""
    summary = report.get("summary", {})
    pagination = report.get("pagination", {})
    lines = [
        "# Open issue contract audit",
        "",
        f"- Schema: `{report.get('schema', '')}`",
        f"- Repository: `{report.get('repository', '')}`",
        f"- Complete: `{str(bool(report.get('complete'))).lower()}`",
        f"- Applicable: `{str(bool(report.get('applicable'))).lower()}`",
        f"- Mutation authorized: `{str(bool(report.get('mutation_authorized'))).lower()}`",
        f"- Content SHA-256: `{report.get('content_sha256', '')}`",
        f"- Pages / issues / excluded PRs: `{pagination.get('pages_read', 0)}` / "
        f"`{pagination.get('issue_rows', 0)}` / "
        f"`{pagination.get('excluded_pull_requests', 0)}`",
    ]
    if json_report:
        lines.append(f"- Full JSON report: `{json_report}`")
    lines.extend(
        ["", "## Classification counts", "", "| Classification | Count |", "| --- | ---: |"]
    )
    for classification, count in sorted(summary.get("classifications", {}).items()):
        lines.append(f"| `{_markdown_cell(classification)}` | {count} |")

    lines.extend(
        [
            "",
            "## Admission reason counts",
            "",
            "| Admission reason | Count |",
            "| --- | ---: |",
        ]
    )
    for reason, count in sorted(summary.get("admission_reason_histogram", {}).items()):
        lines.append(f"| `{_markdown_cell(reason)}` | {count} |")

    lines.extend(
        [
            "",
            "## Bounded preparation queue",
            "",
            "| Issue | Classification | Next action | Missing fields |",
            "| ---: | --- | --- | --- |",
        ]
    )
    items = list(report.get("items", []))
    non_ready_count = sum(item.get("classification") != "ready" for item in items)
    selected = [
        item for item in items if item.get("classification") != "ready" or item.get("listing_drift")
    ][:item_limit]
    for item in selected:
        missing = ", ".join(str(value) for value in item.get("missing_fields", [])) or "—"
        lines.append(
            f"| #{item.get('number', '')} | `{_markdown_cell(item.get('classification', ''))}` | "
            f"`{_markdown_cell(item.get('next_action', ''))}` | {_markdown_cell(missing)} |"
        )
    if len(selected) < non_ready_count:
        lines.extend(
            ["", f"Output is capped at {item_limit} rows. Use the full JSON report for all items."]
        )
    if report.get("errors"):
        lines.extend(["", "## Operational errors", ""])
        lines.extend(f"- {_markdown_cell(error)}" for error in report["errors"][:item_limit])
    return "\n".join(lines).rstrip() + "\n"


def _write_text(path: Path, content: str) -> None:
    """Write one UTF-8 report, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_route_preflight(path: Path | None) -> Mapping[str, Any] | None:
    """Load one route-plan object for explicit multi-repository issue contracts."""
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("route preflight JSON must be an object")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository as OWNER/REPO.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Git remote for claim reads.")
    parser.add_argument("--fixture", type=Path, help="Offline fixture JSON path.")
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"REST issues page size (default: {DEFAULT_PAGE_SIZE}).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=(
            "Maximum REST pages (default: "
            f"{DEFAULT_MAX_PAGES}; rerun with explicit larger limits for full coverage)."
        ),
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, help="Write selected output format to this path.")
    parser.add_argument("--json-report", type=Path, help="Also write the full JSON report here.")
    parser.add_argument(
        "--route-preflight-json",
        type=Path,
        help="Optional fresh route-plan JSON for explicitly multi-repository issue contracts.",
    )
    parser.add_argument(
        "--item-limit",
        type=int,
        default=DEFAULT_ITEM_LIMIT,
        help=f"Maximum rows in Markdown output (default: {DEFAULT_ITEM_LIMIT}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return nonzero when pagination or exact-item evaluation is non-applicable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the open-issue contract audit.

    Returns:
        ``0`` for a completed command, ``2`` with ``--check`` for a fail-closed non-applicable
        report, and ``1`` for malformed input or an unexpected operational error.
    """
    args = _build_parser().parse_args(argv)
    try:
        if not 1 <= args.page_size <= 100:
            raise ValueError("--page-size must be between 1 and 100")
        if args.max_pages < 1:
            raise ValueError("--max-pages must be positive")
        if args.item_limit < 1:
            raise ValueError("--item-limit must be positive")

        if args.fixture:
            fixture, input_sha256 = _load_fixture(args.fixture)
            pages, complete, page_errors = _fixture_pagination(
                fixture,
                page_size=args.page_size,
                max_pages=args.max_pages,
            )
            route_preflight = _load_route_preflight(args.route_preflight_json)
            evaluator = _fixture_evaluator(
                fixture,
                repository=args.repo,
                route_preflight=route_preflight,
            )
            source = "fixture"
        else:
            pages, complete, page_errors = _fetch_live_pages(
                repo=args.repo,
                page_size=args.page_size,
                max_pages=args.max_pages,
            )
            input_sha256 = None
            route_preflight = _load_route_preflight(args.route_preflight_json)
            evaluator = _live_evaluator(
                repo=args.repo,
                remote=args.remote,
                route_preflight=route_preflight,
            )
            source = "github_rest"

        report = _build_report(
            repo=args.repo,
            source=source,
            pages=pages,
            pagination={
                "complete": complete,
                "errors": page_errors,
                "page_size": args.page_size,
                "max_pages": args.max_pages,
            },
            evaluator=evaluator,
            input_sha256=input_sha256,
        )
        json_content = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.json_report:
            _write_text(args.json_report, json_content)
        output = (
            json_content
            if args.format == "json"
            else _render_markdown(
                report,
                item_limit=args.item_limit,
                json_report=str(args.json_report) if args.json_report else None,
            )
        )
        if args.output:
            _write_text(args.output, output)
        else:
            print(output, end="")
        return 2 if args.check and not report["applicable"] else 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"open issue contract audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
