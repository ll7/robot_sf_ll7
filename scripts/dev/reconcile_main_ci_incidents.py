#!/usr/bin/env python3
# ruff: noqa: C901, PLR0913
"""Reconcile stale ``main``-red incident issues through guarded REST writes.

The existing :mod:`main_ci_incident_reconcile` module is the source of truth
for the active/stale/pending signal.  This wrapper owns the missing hosted
state path: it inventories open issues carrying the canonical incident label,
extracts their deciding run, requires two newer consecutive decisive green
runs, and then posts evidence before closing the issue.  It is report-only by
default; ``--apply`` enables the guarded comment and close operations.

Every mutation is preceded by a fresh issue read and followed by readback.  A
changed body, label set, or state causes the item to be skipped or reported as
failed.  Malformed incidents, active incidents, incomplete green evidence, and
API failures never close an issue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from scripts.dev._gh_rest import parse_json, run_gh_api
from scripts.dev.main_ci_incident_reconcile import (
    build_incident_signal,
    fetch_runs,
    incident_reconcile_status,
)
from scripts.dev.main_ci_is_green import classify

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_WORKFLOW = "CI"
DEFAULT_RUN_LIMIT = 10
DEFAULT_MAX_PAGES = 10
DEFAULT_MAX_COMMENT_PAGES = 10
DEFAULT_MAX_ISSUES = 1000
DEFAULT_MAX_MUTATIONS = 100
PER_PAGE = 100
SCHEMA = "main_ci_incident_reconciliation.v1"
INCIDENT_LABEL = "ll7-main-red-incident:v1"
INCIDENT_MARKER = "ll7-main-red-incident:v1"
RECONCILIATION_MARKER = "main-ci-incident-reconciled:v1"
INCIDENT_MARKER_RE = re.compile(rf"(?im)^\s*<!--\s*{re.escape(INCIDENT_MARKER)}\s*-->\s*$")
RECONCILIATION_MARKER_RE = re.compile(
    rf"<!--\s*{re.escape(RECONCILIATION_MARKER)}\s+"
    r"issue=(?P<issue>[1-9][0-9]*)\s+"
    r"deciding-run=(?P<run>[1-9][0-9]*)\s*-->"
)

Runner = Callable[..., Any]
RunFetcher = Callable[[str, str, int], list[dict[str, Any]]]


class ReconciliationError(RuntimeError):
    """Raised when a read, validation, or guarded write cannot be verified."""


def _default_runner(
    path: str,
    payload: object | None = None,
    *,
    method: str | None = None,
    extra_args: list[str] | None = None,
) -> Any:
    """Run one REST request through the shared JSON-stdin transport."""
    return run_gh_api(
        path,
        payload,
        method=method,
        extra_args=extra_args,
        timeout=90,
        timeout_context="main-CI incident reconciliation was not verified",
    )


def _api_json(
    path: str,
    *,
    runner: Runner,
    operation: str,
    method: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> Any:
    """Call a REST endpoint and fail closed on transport or JSON errors."""
    result = runner(path, payload, method=method, extra_args=None)
    data, error = parse_json(result, what=operation)
    if error:
        raise ReconciliationError(error)
    return data


def _positive_int(value: Any, *, field: str) -> int:
    """Return a positive integer, rejecting booleans and malformed IDs."""
    if isinstance(value, bool):
        raise ReconciliationError(f"{field} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ReconciliationError(f"{field} is not an integer: {value!r}") from exc
    if number < 1:
        raise ReconciliationError(f"{field} must be positive")
    return number


def _paginate_collection(
    path: str,
    *,
    runner: Runner,
    operation: str,
    max_pages: int,
) -> list[dict[str, Any]]:
    """Read a bounded REST collection without silently truncating pages."""
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        separator = "&" if "?" in path else "?"
        endpoint = f"{path}{separator}per_page={PER_PAGE}&page={page}"
        payload = _api_json(endpoint, runner=runner, operation=operation)
        if not isinstance(payload, list):
            raise ReconciliationError(f"{operation} returned a non-list payload")
        if any(not isinstance(row, dict) for row in payload):
            raise ReconciliationError(f"{operation} returned a malformed row")
        rows.extend(row for row in payload if isinstance(row, dict))
        if len(payload) < PER_PAGE:
            return rows
    raise ReconciliationError(
        f"{operation} exceeded the {max_pages}-page budget; refusing a partial inventory"
    )


def _label_names(row: Mapping[str, Any], *, issue: int) -> set[str]:
    """Validate and normalize an issue's REST label objects."""
    raw_labels = row.get("labels")
    if not isinstance(raw_labels, list):
        raise ReconciliationError(f"issue #{issue} has a malformed labels field")
    names: set[str] = set()
    for label in raw_labels:
        if not isinstance(label, Mapping) or not isinstance(label.get("name"), str):
            raise ReconciliationError(f"issue #{issue} has a malformed label entry")
        names.add(label["name"])
    return names


def _is_pull_request(row: Mapping[str, Any]) -> bool:
    """Return whether a REST issue-shaped row is actually a pull request."""
    return "pull_request" in row or "/pull/" in str(row.get("html_url") or "")


def _issue_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one candidate issue row and keep its precondition fields."""
    number = _positive_int(row.get("number"), field="issue number")
    labels = _label_names(row, issue=number)
    state = str(row.get("state") or "").lower()
    if state != "open":
        raise ReconciliationError(f"issue #{number} inventory row is not open")
    body = row.get("body")
    if body is not None and not isinstance(body, str):
        raise ReconciliationError(f"issue #{number} has a malformed body")
    updated_at = row.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        raise ReconciliationError(f"issue #{number} has no usable updated_at precondition")
    return {
        "number": number,
        "title": str(row.get("title") or ""),
        "body": body if isinstance(body, str) else "",
        "updated_at": updated_at,
        "url": str(row.get("html_url") or row.get("url") or ""),
        "labels": sorted(labels),
    }


def list_open_incidents(
    *,
    repo: str,
    max_pages: int,
    runner: Runner,
) -> list[dict[str, Any]]:
    """Return all open, non-PR issues carrying the label or body marker.

    The incident creator's body marker is the durable identity contract.  The
    label is retained as a compatibility signal, but cannot be the inventory
    filter: the live issue stream contains canonical incidents created without
    that label.  Enumerating the bounded open-issue collection keeps those
    incidents visible and lets malformed marker-only cases fail closed.
    """
    query = urlencode({"state": "open"})
    rows = _paginate_collection(
        f"repos/{quote(repo, safe='/')}/issues?{query}",
        runner=runner,
        operation="open main-CI incident inventory",
        max_pages=max_pages,
    )
    incidents: list[dict[str, Any]] = []
    for row in rows:
        if _is_pull_request(row):
            continue
        candidate = _issue_row(row)
        has_label = INCIDENT_LABEL in candidate["labels"]
        has_marker = INCIDENT_MARKER_RE.search(candidate["body"]) is not None
        if has_label or has_marker:
            incidents.append(candidate)
    return sorted(incidents, key=lambda issue: issue["number"])


def _get_issue(*, repo: str, issue: int, runner: Runner) -> dict[str, Any]:
    """Read one current issue record through REST."""
    payload = _api_json(
        f"repos/{quote(repo, safe='/')}/issues/{issue}",
        runner=runner,
        operation=f"read issue #{issue}",
    )
    if not isinstance(payload, dict):
        raise ReconciliationError(f"issue #{issue} read returned a non-object payload")
    if _is_pull_request(payload):
        raise ReconciliationError(f"issue #{issue} resolved to a pull request")
    number = _positive_int(payload.get("number"), field=f"issue #{issue} number")
    if number != issue:
        raise ReconciliationError(f"issue read-back returned #{number}, expected #{issue}")
    _label_names(payload, issue=issue)
    if not isinstance(payload.get("updated_at"), str) or not payload["updated_at"]:
        raise ReconciliationError(f"issue #{issue} read has no usable updated_at")
    if payload.get("body") is not None and not isinstance(payload.get("body"), str):
        raise ReconciliationError(f"issue #{issue} read has a malformed body")
    return payload


def _get_comments(
    *,
    repo: str,
    issue: int,
    max_pages: int,
    runner: Runner,
) -> list[dict[str, Any]]:
    """Read the complete bounded comment thread for one issue."""
    rows = _paginate_collection(
        f"repos/{quote(repo, safe='/')}/issues/{issue}/comments",
        runner=runner,
        operation=f"comments for issue #{issue}",
        max_pages=max_pages,
    )
    for row in rows:
        if row.get("body") is not None and not isinstance(row.get("body"), str):
            raise ReconciliationError(f"comments for issue #{issue} contain a malformed body")
    return rows


def parse_deciding_run_id(body: str, *, repo: str) -> tuple[int | None, str | None]:
    """Parse exactly one canonical deciding-run field from an incident body."""
    if not isinstance(body, str) or not body.strip():
        return None, "incident body is missing"
    if len(INCIDENT_MARKER_RE.findall(body)) != 1:
        return None, "incident body does not contain exactly one canonical marker"
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        return None, f"repository is not owner/name: {repo!r}"
    owner, name = parts
    pattern = re.compile(
        rf"(?im)^\s*Deciding failing run:\s*"
        rf"https://github\.com/{re.escape(owner)}/{re.escape(name)}"
        rf"/actions/runs/(?P<run>[1-9][0-9]*)\s*$"
    )
    matches = pattern.findall(body)
    if len(matches) != 1:
        return None, "incident body does not contain exactly one canonical deciding run"
    return int(matches[0]), None


def _validate_run_window(runs: Sequence[Any]) -> None:
    """Validate the identity fields needed to render any fetched run safely."""
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ReconciliationError(f"run window row {index} is malformed")
        _positive_int(run.get("databaseId"), field=f"run window row {index} databaseId")
        if not isinstance(run.get("createdAt"), str) or not run["createdAt"]:
            raise ReconciliationError(f"run window row {index} has no usable createdAt")


def _ordered_decisive_runs(runs: Sequence[Any]) -> list[dict[str, Any]]:
    """Validate and order completed green/red runs for consecutive evidence."""
    decisive: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ReconciliationError(f"run window row {index} is malformed")
        if str(run.get("status") or "") != "completed":
            continue
        verdict = classify(run.get("conclusion"))
        if verdict not in {"green", "red"}:
            continue
        run_copy = dict(run)
        run_copy["_run_id"] = _positive_int(run.get("databaseId"), field="run databaseId")
        created_at = run.get("createdAt")
        if not isinstance(created_at, str) or not created_at:
            raise ReconciliationError("decisive run has no usable createdAt")
        decisive.append(run_copy)
    decisive.sort(key=lambda run: (str(run["createdAt"]), int(run["_run_id"])), reverse=True)
    return decisive


def _public_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """Project a run onto stable evidence fields used in reports/comments."""
    run_id = int(run.get("_run_id", run.get("databaseId")))
    return {
        "id": run_id,
        "url": f"https://github.com/{run.get('_repo', '')}/actions/runs/{run_id}",
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "created_at": run.get("createdAt"),
        "head_sha": run.get("headSha"),
    }


def _green_evidence(deciding_run_id: int, runs: Sequence[Any]) -> list[dict[str, Any]]:
    """Return two newer consecutive decisive green runs, or an empty list."""
    newer = [run for run in _ordered_decisive_runs(runs) if int(run["_run_id"]) > deciding_run_id]
    if len(newer) < 2:
        return []
    first_two = newer[:2]
    if any(classify(run.get("conclusion")) != "green" for run in first_two):
        return []
    return first_two


def _comment_body(
    *,
    issue: int,
    repo: str,
    deciding_run_id: int,
    green_runs: Sequence[Mapping[str, Any]],
) -> str:
    """Build the deterministic evidence comment posted before closure."""
    marker = f"<!-- {RECONCILIATION_MARKER} issue={issue} deciding-run={deciding_run_id} -->"
    lines = [
        marker,
        "Automated main-CI incident reconciliation.",
        "",
        f"- Deciding failing run: https://github.com/{repo}/actions/runs/{deciding_run_id}",
        "- Two newer consecutive decisive green runs:",
    ]
    for run in green_runs:
        run_id = int(run.get("_run_id", run.get("id")))
        lines.append(
            f"  - https://github.com/{repo}/actions/runs/{run_id} "
            f"(completed success; created {run.get('createdAt', run.get('created_at', ''))})"
        )
    lines.extend(
        [
            "",
            "Classifier: `main_ci_incident_reconcile.v1` reported `stale`.",
            "Disposition: closed as completed because the deciding failure was superseded.",
        ]
    )
    return "\n".join(lines) + "\n"


def _validate_issue_content(
    current: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    issue: int,
) -> None:
    """Verify fields that must not drift during a guarded mutation."""
    if _is_pull_request(current):
        raise ReconciliationError(f"issue #{issue} became a pull request")
    body = current.get("body") if isinstance(current.get("body"), str) else ""
    if body != expected["body"]:
        raise ReconciliationError(f"issue #{issue} body changed before mutation")
    labels = sorted(_label_names(current, issue=issue))
    if labels != expected["labels"]:
        raise ReconciliationError(f"issue #{issue} labels changed before mutation")


def _precondition_action(current: Mapping[str, Any], expected: Mapping[str, Any]) -> str | None:
    """Return a safe skip action for a changed state or timestamp."""
    state = str(current.get("state") or "").lower()
    if state != "open":
        return "already_closed" if state == "closed" else "state_changed"
    if current.get("updated_at") != expected["updated_at"]:
        return "precondition_changed"
    return None


def _find_existing_comment(
    comments: Sequence[Mapping[str, Any]],
    *,
    issue: int,
    deciding_run_id: int,
    expected_body: str,
) -> bool:
    """Recognize exactly one matching evidence comment for idempotent retries."""
    matches: list[Mapping[str, Any]] = []
    for comment in comments:
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        marker = RECONCILIATION_MARKER_RE.search(body)
        if marker is None or int(marker.group("issue")) != issue:
            continue
        if int(marker.group("run")) != deciding_run_id:
            raise ReconciliationError(f"issue #{issue} has evidence for a different deciding run")
        if body != expected_body:
            raise ReconciliationError(f"issue #{issue} has a conflicting evidence comment")
        matches.append(comment)
    if len(matches) > 1:
        raise ReconciliationError(f"issue #{issue} has duplicate reconciliation comments")
    return bool(matches)


def _post_comment(
    *,
    repo: str,
    issue: int,
    body: str,
    runner: Runner,
) -> None:
    """Post one evidence comment and require exact response readback."""
    payload = _api_json(
        f"repos/{quote(repo, safe='/')}/issues/{issue}/comments",
        runner=runner,
        operation=f"post reconciliation comment for issue #{issue}",
        method="POST",
        payload={"body": body},
    )
    if not isinstance(payload, dict) or payload.get("body") != body:
        raise ReconciliationError(f"reconciliation comment readback failed for issue #{issue}")


def _close_issue(*, repo: str, issue: int, runner: Runner) -> None:
    """Close an issue as completed and verify the resulting state."""
    payload = _api_json(
        f"repos/{quote(repo, safe='/')}/issues/{issue}",
        runner=runner,
        operation=f"close issue #{issue}",
        method="PATCH",
        payload={"state": "closed", "state_reason": "completed"},
    )
    if (
        not isinstance(payload, dict)
        or str(payload.get("state") or "").lower() != "closed"
        or str(payload.get("state_reason") or "").lower() != "completed"
    ):
        raise ReconciliationError(f"issue #{issue} close readback was not closed as completed")


def _apply_candidate(
    issue: Mapping[str, Any],
    *,
    repo: str,
    deciding_run_id: int,
    green_runs: Sequence[Mapping[str, Any]],
    max_comment_pages: int,
    runner: Runner,
) -> dict[str, Any]:
    """Apply one stale candidate using read-before-write and readback guards."""
    number = int(issue["number"])
    expected_body = _comment_body(
        issue=number,
        repo=repo,
        deciding_run_id=deciding_run_id,
        green_runs=green_runs,
    )
    current = _get_issue(repo=repo, issue=number, runner=runner)
    _validate_issue_content(current, issue, issue=number)
    action = _precondition_action(current, issue)
    if action is not None:
        return {"action": action, "reason": "issue changed after inventory"}

    comments = _get_comments(
        repo=repo,
        issue=number,
        max_pages=max_comment_pages,
        runner=runner,
    )
    existing = _find_existing_comment(
        comments,
        issue=number,
        deciding_run_id=deciding_run_id,
        expected_body=expected_body,
    )

    # Re-read immediately before the first mutation.  This closes the race
    # between the comment inventory and the write while keeping retries safe.
    current = _get_issue(repo=repo, issue=number, runner=runner)
    _validate_issue_content(current, issue, issue=number)
    action = _precondition_action(current, issue)
    if action is not None:
        return {"action": action, "reason": "issue changed before mutation"}

    if not existing:
        _post_comment(repo=repo, issue=number, body=expected_body, runner=runner)
        comment_action = "comment_created"
    else:
        comment_action = "comment_existing"

    current = _get_issue(repo=repo, issue=number, runner=runner)
    _validate_issue_content(current, issue, issue=number)
    if str(current.get("state") or "").lower() != "open":
        return {
            "action": "already_closed_after_comment",
            "comment_action": comment_action,
        }
    _close_issue(repo=repo, issue=number, runner=runner)
    return {"action": "closed", "comment_action": comment_action}


def _evaluate_issue(
    issue: Mapping[str, Any],
    *,
    repo: str,
    workflow: str,
    runs: Sequence[Any],
) -> dict[str, Any]:
    """Classify one issue and distinguish classifier staleness from close eligibility."""
    number = int(issue["number"])
    result: dict[str, Any] = {
        "issue": number,
        "title": issue["title"],
        "url": issue["url"],
        "action": "none",
        "classifier": None,
        "green_runs": [],
    }
    deciding_run_id, parse_error = parse_deciding_run_id(issue["body"], repo=repo)
    result["deciding_failure_run_id"] = deciding_run_id
    if parse_error is not None:
        result.update({"status": "pending", "reason": parse_error})
        return result

    try:
        classifier_status = incident_reconcile_status(deciding_run_id, list(runs))
        signal = build_incident_signal(
            classifier_status,
            deciding_run_id,
            list(runs),
            repo=repo,
            workflow=workflow,
        )
        green_runs = _green_evidence(deciding_run_id, runs)
    except (TypeError, ValueError, ReconciliationError) as exc:
        result.update({"status": "pending", "reason": f"run window invalid: {exc}"})
        return result

    result["classifier"] = signal
    result["classifier_status"] = classifier_status
    result["green_runs"] = [_public_run({**run, "_repo": repo}) for run in green_runs]
    if classifier_status != "stale":
        result.update(
            {"status": classifier_status, "reason": f"classifier status is {classifier_status}"}
        )
        return result
    if len(green_runs) < 2:
        result.update(
            {
                "status": "pending",
                "reason": "fewer than two newer consecutive decisive green runs",
            }
        )
        return result
    result.update({"status": "stale", "action": "would_close"})
    return result


def _report_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Build stable disposition counts for the machine-readable report."""
    counts = Counter(str(result.get("action") or "none") for result in results)
    for key in (
        "none",
        "would_close",
        "closed",
        "already_closed",
        "already_closed_after_comment",
        "state_changed",
        "precondition_changed",
        "failed",
    ):
        counts.setdefault(key, 0)
    return dict(sorted(counts.items()))


def reconcile_batch(
    *,
    repo: str = DEFAULT_REPO,
    workflow: str = DEFAULT_WORKFLOW,
    apply: bool = False,
    run_limit: int = DEFAULT_RUN_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_comment_pages: int = DEFAULT_MAX_COMMENT_PAGES,
    max_issues: int = DEFAULT_MAX_ISSUES,
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
    runner: Runner | None = None,
    run_fetcher: RunFetcher | None = None,
) -> dict[str, Any]:
    """Inventory, classify, and optionally reconcile all open incidents."""
    if run_limit <= 0 or max_issues <= 0 or max_mutations <= 0:
        raise ValueError("run_limit, max_issues, and max_mutations must be positive")
    rest_runner = runner or _default_runner
    incidents = list_open_incidents(repo=repo, max_pages=max_pages, runner=rest_runner)
    if len(incidents) > max_issues:
        raise ReconciliationError(
            f"incident inventory contains {len(incidents)} rows, above limit {max_issues}"
        )
    if not incidents:
        return {
            "schema_version": SCHEMA,
            "repo": repo,
            "workflow": workflow,
            "mode": "apply" if apply else "report-only",
            "fail_closed": True,
            "source": {
                "label": INCIDENT_LABEL,
                "open_incident_count": 0,
                "pagination_complete": True,
                "run_limit": run_limit,
                "run_count": 0,
            },
            "run_window": [],
            "counts": _report_counts([]),
            "results": [],
            "errors": [],
            "status": "ok",
        }
    try:
        runs = (run_fetcher or fetch_runs)(repo, workflow, run_limit)
    except (RuntimeError, json.JSONDecodeError) as exc:
        raise ReconciliationError(f"main-CI run fetch failed: {exc}") from exc
    if not isinstance(runs, list):
        raise ReconciliationError("main-CI run fetch returned a non-list payload")
    # Validate once before any issue can become eligible.  The classifier and
    # the two-green evidence helper then consume the same run window.
    _validate_run_window(runs)
    _ordered_decisive_runs(runs)

    results = [
        _evaluate_issue(issue, repo=repo, workflow=workflow, runs=runs) for issue in incidents
    ]
    eligible = [result for result in results if result.get("action") == "would_close"]
    if apply and len(eligible) > max_mutations:
        raise ReconciliationError(
            f"eligible incident count {len(eligible)} exceeds mutation budget {max_mutations}"
        )

    errors: list[str] = []
    if apply:
        for result, issue in zip(results, incidents, strict=True):
            if result.get("action") != "would_close":
                continue
            try:
                applied = _apply_candidate(
                    issue,
                    repo=repo,
                    deciding_run_id=int(result["deciding_failure_run_id"]),
                    green_runs=[
                        {
                            "_run_id": int(run["id"]),
                            "createdAt": run.get("created_at"),
                        }
                        for run in result["green_runs"]
                    ],
                    max_comment_pages=max_comment_pages,
                    runner=rest_runner,
                )
            except ReconciliationError as exc:
                result.update({"action": "failed", "reason": str(exc)})
                errors.append(f"issue #{issue['number']}: {exc}")
            else:
                result.update(applied)

    report = {
        "schema_version": SCHEMA,
        "repo": repo,
        "workflow": workflow,
        "mode": "apply" if apply else "report-only",
        "fail_closed": True,
        "source": {
            "label": INCIDENT_LABEL,
            "open_incident_count": len(incidents),
            "pagination_complete": True,
            "run_limit": run_limit,
            "run_count": len(runs),
        },
        "run_window": [
            _public_run({**run, "_repo": repo}) for run in runs if isinstance(run, dict)
        ],
        "counts": _report_counts(results),
        "results": results,
        "errors": errors,
        "status": "error" if errors else "ok",
    }
    return report


def _base_report(*, repo: str, workflow: str, apply: bool) -> dict[str, Any]:
    """Create a stable report envelope for failures before inventory completes."""
    return {
        "schema_version": SCHEMA,
        "repo": repo,
        "workflow": workflow,
        "mode": "apply" if apply else "report-only",
        "fail_closed": True,
        "source": {"label": INCIDENT_LABEL, "pagination_complete": False},
        "results": [],
        "errors": [],
        "status": "error",
    }


def _write_report(path: Path | None, report: Mapping[str, Any]) -> str:
    """Serialize a report and optionally persist it as a local artifact."""
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    return encoded


def build_parser() -> argparse.ArgumentParser:
    """Build the scheduled reconciler CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--run-limit", type=int, default=DEFAULT_RUN_LIMIT)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--max-comment-pages", type=int, default=DEFAULT_MAX_COMMENT_PAGES)
    parser.add_argument("--max-issues", type=int, default=DEFAULT_MAX_ISSUES)
    parser.add_argument("--max-mutations", type=int, default=DEFAULT_MAX_MUTATIONS)
    parser.add_argument(
        "--apply", action="store_true", help="post evidence and close eligible issues"
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true", help="emit the machine-readable report"
    )
    parser.add_argument("--output", type=Path, help="also write the report to this local path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reconciler and return nonzero only for incomplete/error execution."""
    args = build_parser().parse_args(argv)
    base = _base_report(repo=args.repo, workflow=args.workflow, apply=args.apply)
    try:
        report = reconcile_batch(
            repo=args.repo,
            workflow=args.workflow,
            apply=args.apply,
            run_limit=args.run_limit,
            max_pages=args.max_pages,
            max_comment_pages=args.max_comment_pages,
            max_issues=args.max_issues,
            max_mutations=args.max_mutations,
        )
    except (ReconciliationError, ValueError) as exc:
        report = {**base, "errors": [str(exc)]}
    encoded = _write_report(args.output, report)
    if args.as_json:
        print(encoded, end="")
    else:
        print(
            f"main-CI incident reconciliation: {report['status']} "
            f"({report.get('mode', 'unknown')}); "
            f"issues={len(report.get('results', []))}"
        )
        for error in report.get("errors", []):
            print(f"error: {error}", file=sys.stderr)
    return 1 if report.get("status") == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
