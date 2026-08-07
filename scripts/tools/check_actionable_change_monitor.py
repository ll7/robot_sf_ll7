#!/usr/bin/env python3
"""Scan RobotSF research surfaces and update one alert issue on fingerprint deltas.

The monitor is read-only with respect to repository research state. Its sole
write is a body update on the explicitly configured dedicated alert issue, and
only when the canonical finding fingerprint changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime, timedelta
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_ISSUE = 6819
DEFAULT_LIMIT = 500
PAGE_SIZE = 100
STALE_DRAFT_HOURS = 72
SCHEMA_VERSION = "robot_sf.actionable_change_monitor.v1"
FINGERPRINT_RE = re.compile(r"<!--\s*actionable-change-monitor:fingerprint:([0-9a-f]{64})\s*-->")
RESEARCH_LABELS = frozenset(
    {
        "benchmark",
        "blocked",
        "evidence:analysis-only",
        "evidence:launch-packet",
        "evidence:nominal",
        "paper-critical",
        "research",
        "resource:slurm",
        "slurm",
        "training",
        "validation",
        "state:blocked",
    }
)
WATCH_LABEL_PREFIXES = ("blocked:", "evidence:", "resource:", "state:")
WATCH_TERMS = (
    "prediction",
    "navigation",
    "planner",
    "slurm",
    "paper-critical",
    "launch packet",
    "benchmark",
    "evidence",
)
SUCCESS_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})
JOB_ID_RE = re.compile(r"\b(?:job(?:[_ -]?id)?|slurm(?:[_ -]?job)?)[\s:#=]+\d+\b", re.I)
HANDOFF_TERMS = ("dissertation", "diss#", "evidence handoff", "parent handoff")
PROPAGATION_TERMS = ("parent", "propagat", "follow-up", "follow up", "refs #")
TERMINAL_TERMS = ("complete", "completed", "terminal", "admitted", "evidence freeze")
ADMISSION_TERMS = ("admit", "admission", "promot", "evidence freeze")


class MonitorError(RuntimeError):
    """Raised when the monitor cannot complete a fail-closed read or write."""


def _compact(value: Any) -> str:
    """Normalize a display value into one safe Markdown line."""
    return " ".join(str(value or "").split()).replace("|", "\\|")


def _labels(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return sorted label names from a GitHub issue or pull-request payload."""
    values = payload.get("labels") or []
    return tuple(sorted(str(item.get("name", "")) for item in values if item.get("name")))


def _text(payload: dict[str, Any]) -> str:
    """Return lower-case searchable title/body text."""
    return f"{payload.get('title') or ''}\n{payload.get('body') or ''}".casefold()


def _has_watch_label(labels: tuple[str, ...]) -> bool:
    """Return whether labels identify a monitored prediction/navigation/blocked surface."""
    return bool(
        set(labels) & {"paper-critical", "slurm", "resource:slurm", "state:blocked", "blocked"}
        or any(label.startswith(WATCH_LABEL_PREFIXES) for label in labels)
    )


def _is_research_surface(payload: dict[str, Any]) -> bool:
    """Return whether a PR or issue belongs to the research/benchmark watch lane."""
    labels = _labels(payload)
    if set(labels) & RESEARCH_LABELS or _has_watch_label(labels):
        return True
    text = _text(payload)
    return any(term in text for term in WATCH_TERMS)


def _finding(
    *,
    finding_id: str,
    kind: str,
    target: dict[str, Any],
    detail: str,
) -> dict[str, Any]:
    """Build one stable, JSON-serializable finding."""
    return {
        "id": finding_id,
        "kind": kind,
        "number": int(target["number"]),
        "title": str(target.get("title") or ""),
        "url": str(target.get("html_url") or target.get("url") or ""),
        "detail": detail,
    }


def _canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value with stable key and separator ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _check_run_identity(check: dict[str, Any]) -> str:
    """Return a stable identity for a check run, including duplicate-named runs."""
    check_id = check.get("id")
    if check_id is not None:
        return f"id:{check_id}"
    payload = _canonical_json(check).encode("utf-8")
    return f"payload:{hashlib.sha256(payload).hexdigest()}"


def _check_findings(
    pull_request: dict[str, Any],
    check_runs: list[dict[str, Any]],
    statuses: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return pending or failed check-run and legacy-status findings for one pull request."""
    findings: list[dict[str, Any]] = []
    for check in sorted(
        check_runs,
        key=lambda row: (
            str(row.get("name") or ""),
            _check_run_identity(row),
            _canonical_json(row),
        ),
    ):
        name = str(check.get("name") or "unnamed-check")
        check_identity = _check_run_identity(check)
        status = str(check.get("status") or "unknown").casefold()
        conclusion = str(check.get("conclusion") or "").casefold()
        if status != "completed":
            findings.append(
                _finding(
                    finding_id=(
                        f"pr:{pull_request['number']}:check:{check_identity}:{name}:pending"
                    ),
                    kind="check_pending",
                    target=pull_request,
                    detail=f"{name}: status={status}",
                )
            )
        elif conclusion not in SUCCESS_CONCLUSIONS:
            findings.append(
                _finding(
                    finding_id=(
                        f"pr:{pull_request['number']}:check:{check_identity}:{name}:failed"
                    ),
                    kind="check_failed",
                    target=pull_request,
                    detail=f"{name}: conclusion={conclusion or 'missing'}",
                )
            )
    for commit_status in sorted(statuses or [], key=lambda row: str(row.get("context") or "")):
        context = str(commit_status.get("context") or "unnamed-status")
        state = str(commit_status.get("state") or "unknown").casefold()
        if state == "pending":
            findings.append(
                _finding(
                    finding_id=f"pr:{pull_request['number']}:status:{context}:pending",
                    kind="check_pending",
                    target=pull_request,
                    detail=f"{context}: state={state}",
                )
            )
        elif state != "success":
            findings.append(
                _finding(
                    finding_id=f"pr:{pull_request['number']}:status:{context}:failed",
                    kind="check_failed",
                    target=pull_request,
                    detail=f"{context}: state={state}",
                )
            )
    return findings


def _surface_findings(issue: dict[str, Any]) -> list[dict[str, Any]]:
    """Return issue-level readiness, launch, evidence, and state findings."""
    labels = _labels(issue)
    body = str(issue.get("body") or "").casefold()
    findings: list[dict[str, Any]] = []
    label_set = set(labels)
    blocked = bool(
        {"blocked", "state:blocked"} & label_set
        or any(label.startswith("blocked:") for label in labels)
    )
    ready = "state:ready" in label_set
    if blocked and ready:
        findings.append(
            _finding(
                finding_id=f"issue:{issue['number']}:contradictory-readiness",
                kind="contradictory_readiness",
                target=issue,
                detail="blocked and ready labels are present together",
            )
        )

    stale_ready_text = any(
        phrase in body
        for phrase in (
            "ready to proceed",
            "ready for training",
            "all dependencies are closed",
            "ready to launch",
        )
    )
    if blocked and stale_ready_text:
        findings.append(
            _finding(
                finding_id=f"issue:{issue['number']}:blocked-with-ready-text",
                kind="blocked_surface_with_ready_text",
                target=issue,
                detail="blocked surface still contains readiness language",
            )
        )

    if (
        "evidence:launch-packet" in label_set
        and ("resource:slurm" in label_set or "slurm" in label_set or "slurm" in body)
        and not JOB_ID_RE.search(body)
    ):
        findings.append(
            _finding(
                finding_id=f"issue:{issue['number']}:launch-without-job-id",
                kind="launch_packet_without_job_id",
                target=issue,
                detail="SLURM launch packet has no recorded job ID",
            )
        )

    terminal_evidence = bool(
        any(term in body for term in TERMINAL_TERMS)
        and (
            {"evidence:analysis-only", "evidence:nominal", "state:review"} & label_set
            or JOB_ID_RE.search(body)
        )
    )
    if terminal_evidence and not any(term in body for term in PROPAGATION_TERMS):
        findings.append(
            _finding(
                finding_id=f"issue:{issue['number']}:terminal-evidence-without-parent",
                kind="terminal_evidence_without_parent_propagation",
                target=issue,
                detail="terminal evidence language has no parent propagation or follow-up reference",
            )
        )

    evidence_admission = bool(
        "evidence:nominal" in label_set and any(term in body for term in ADMISSION_TERMS)
    )
    if evidence_admission and not any(term in body for term in HANDOFF_TERMS):
        findings.append(
            _finding(
                finding_id=f"issue:{issue['number']}:evidence-without-dissertation-handoff",
                kind="evidence_admission_without_dissertation_handoff",
                target=issue,
                detail="evidence admission language has no dissertation handoff reference",
            )
        )

    if _has_watch_label(labels) or any(term in body for term in WATCH_TERMS):
        watched_labels = ", ".join(labels) or "keyword-only"
        findings.append(
            _finding(
                finding_id=f"issue:{issue['number']}:watched-surface",
                kind="watched_research_surface",
                target=issue,
                detail=f"labels/terms={watched_labels}; updated={issue.get('updated_at', 'unknown')}",
            )
        )
    return findings


def build_findings(
    pull_requests: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    check_runs_by_pr: dict[int, list[dict[str, Any]]],
    *,
    now: datetime,
    target_issue: int = DEFAULT_ISSUE,
    statuses_by_pr: dict[int, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic findings from a read-only GitHub snapshot."""
    findings: list[dict[str, Any]] = []
    stale_before = now - timedelta(hours=STALE_DRAFT_HOURS)
    for pr in pull_requests:
        if not _is_research_surface(pr):
            continue
        findings.append(
            _finding(
                finding_id=f"pr:{pr['number']}:research-activity",
                kind="research_pr_activity",
                target=pr,
                detail=(
                    f"updated={pr.get('updated_at', 'unknown')}; head={pr.get('head', {}).get('sha', 'unknown')}; "
                    f"draft={bool(pr.get('draft'))}; labels={', '.join(_labels(pr)) or 'none'}"
                ),
            )
        )
        if pr.get("draft") and _parse_timestamp(pr.get("updated_at")) < stale_before:
            findings.append(
                _finding(
                    finding_id=f"pr:{pr['number']}:stale-draft",
                    kind="stale_research_draft",
                    target=pr,
                    detail=f"draft has not been updated since {pr.get('updated_at', 'unknown')}",
                )
            )
        pr_number = int(pr["number"])
        findings.extend(
            _check_findings(
                pr,
                check_runs_by_pr.get(pr_number, []),
                (statuses_by_pr or {}).get(pr_number, []),
            )
        )

    for issue in issues:
        number = int(issue["number"])
        if number != target_issue and _is_research_surface(issue):
            findings.extend(_surface_findings(issue))
    return sorted(findings, key=lambda row: (row["kind"], row["number"], row["id"]))


def _parse_timestamp(value: Any) -> datetime:
    """Parse GitHub's ISO timestamp, returning the epoch for missing values."""
    if not value:
        return datetime.fromtimestamp(0, tz=UTC)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0, tz=UTC)


def compute_fingerprint(findings: list[dict[str, Any]]) -> str:
    """Return a stable SHA-256 fingerprint for the complete finding set."""
    canonical_findings = sorted(
        findings,
        key=lambda row: (
            str(row.get("kind", "")),
            int(row.get("number", 0)),
            str(row.get("id", "")),
            _canonical_json(row),
        ),
    )
    canonical = _canonical_json(canonical_findings).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def extract_previous_fingerprint(body: str) -> str | None:
    """Read the monitor fingerprint marker from the dedicated issue body."""
    match = FINGERPRINT_RE.search(body)
    return match.group(1) if match else None


def render_issue_body(
    findings: list[dict[str, Any]],
    *,
    fingerprint: str,
    scanned_at: datetime,
) -> str:
    """Render the latest alert report and preserve the monitor's safety contract."""
    lines = [
        "# RobotSF actionable-change monitor",
        "",
        "This advisory report is updated only when the monitored repository snapshot changes.",
        "It never merges pull requests, changes research state, submits compute, or writes any",
        "issue other than this dedicated target.",
        "",
        f"Scanned at: `{scanned_at.astimezone(UTC).isoformat()}`",
        f"Finding count: `{len(findings)}`",
        "",
        f"<!-- actionable-change-monitor:fingerprint:{fingerprint} -->",
        "",
    ]
    if not findings:
        lines.extend(
            ["## No actionable changes", "", "The monitored snapshot is unchanged and clear.", ""]
        )
    else:
        lines.extend(["## Findings", "", "| Kind | Target | Detail |", "| --- | --- | --- |"])
        for finding in findings:
            target = (
                f"[#{finding['number']}]({finding['url']})"
                if finding["url"]
                else f"#{finding['number']}"
            )
            lines.append(
                f"| `{_compact(finding['kind'])}` | {target} {_compact(finding['title'])} | {_compact(finding['detail'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Safety contract",
            "",
            "- Permissions: repository contents read, pull requests read, checks read, commit statuses read, and issues write.",
            "- The only allowed write is this issue body, after a fingerprint change.",
            "- API errors fail closed with no issue write.",
            "- Empty scans and unchanged fingerprints are no-ops.",
            "",
        ]
    )
    return "\n".join(lines)


def _gh_json(
    path: str,
    *,
    paginate: bool = False,
    max_pages: int = 1,
    repo: str = DEFAULT_REPO,
) -> Any:
    """Run a bounded GitHub CLI JSON read."""
    endpoint = path if path.startswith("repos/") else f"repos/{repo}/{path.lstrip('/')}"
    page_count = max(1, max_pages) if paginate else 1
    rows: list[Any] = []
    for page in range(1, page_count + 1):
        page_endpoint = endpoint
        if paginate:
            page_endpoint = f"{endpoint}{'&' if '?' in endpoint else '?'}page={page}"
        command = ["gh", "api", page_endpoint]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=90, check=False
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise MonitorError(f"GitHub read failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise MonitorError(f"GitHub read failed for {page_endpoint}: {detail}")
        try:
            payload = json.loads(result.stdout or "null")
        except json.JSONDecodeError as exc:
            raise MonitorError(f"GitHub returned invalid JSON for {page_endpoint}") from exc
        if not paginate:
            return payload
        if not isinstance(payload, list):
            raise MonitorError(f"GitHub paginated response for {page_endpoint} is not a list")
        rows.extend(payload)
        if len(payload) < 100:
            break
    return rows


def _require_object_rows(payload: Any, *, resource: str) -> list[dict[str, Any]]:
    """Reject a non-list or any malformed top-level row from a paginated API response."""
    if not isinstance(payload, list):
        raise MonitorError(f"{resource} response is not a list")
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise MonitorError(f"{resource} response contains malformed row at index {index}")
    return payload


def _fetch_commit_collection(
    repo: str,
    sha: str,
    *,
    resource: str,
    collection_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Read one bounded, paginated commit collection and reject malformed pages."""
    if limit < 1:
        raise MonitorError("monitor limit must be positive")
    page_size = min(limit, PAGE_SIZE)
    max_pages = max(1, (limit + page_size - 1) // page_size)
    rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = _gh_json(
            f"repos/{repo}/commits/{sha}/{resource}?per_page={page_size}&page={page}"
        )
        if not isinstance(payload, dict):
            raise MonitorError(f"{resource} response for commit {sha} is not an object")
        collection = payload.get(collection_key)
        if not isinstance(collection, list) or any(not isinstance(row, dict) for row in collection):
            raise MonitorError(
                f"{resource} response for commit {sha} has no valid {collection_key} list"
            )
        total_count = payload.get("total_count")
        if total_count is not None and (
            isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 0
        ):
            raise MonitorError(f"{resource} response for commit {sha} has invalid total_count")
        rows.extend(collection)
        if total_count is not None and total_count < len(rows):
            raise MonitorError(
                f"{resource} response for commit {sha} reports total_count={total_count} "
                f"below {len(rows)} returned rows"
            )
        if len(rows) >= limit:
            return rows[:limit]
        if len(collection) < page_size:
            if total_count is not None and total_count > len(rows):
                raise MonitorError(
                    f"{resource} response for commit {sha} returned a short page with "
                    f"total_count={total_count} but only {len(rows)} rows were returned"
                )
            return rows[:limit]
    return rows[:limit]


def _fetch_snapshot(
    repo: str, limit: int
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
]:
    """Read open PRs, open issues, check-runs, and legacy statuses for research PR heads."""
    if limit < 1:
        raise MonitorError("monitor limit must be positive")
    pull_requests = _require_object_rows(
        _gh_json(
            f"repos/{repo}/pulls?state=open&sort=updated&direction=desc&per_page={min(limit, 100)}",
            paginate=True,
            max_pages=max(1, (limit + 99) // 100),
        ),
        resource="pull requests",
    )
    issues_payload = _require_object_rows(
        _gh_json(
            f"repos/{repo}/issues?state=open&sort=updated&direction=desc&per_page={min(limit, 100)}",
            paginate=True,
            max_pages=max(1, (limit + 99) // 100),
        ),
        resource="issues",
    )
    issues = [row for row in issues_payload if "pull_request" not in row]
    checks: dict[int, list[dict[str, Any]]] = {}
    statuses: dict[int, list[dict[str, Any]]] = {}
    for pr in pull_requests:
        if not isinstance(pr, dict) or not _is_research_surface(pr):
            continue
        sha = str(pr.get("head", {}).get("sha") or "")
        if not sha:
            raise MonitorError(f"research PR #{pr.get('number')} has no head SHA")
        pr_number = int(pr["number"])
        checks[pr_number] = _fetch_commit_collection(
            repo,
            sha,
            resource="check-runs",
            collection_key="check_runs",
            limit=limit,
        )
        statuses[pr_number] = _fetch_commit_collection(
            repo,
            sha,
            resource="status",
            collection_key="statuses",
            limit=limit,
        )
    return pull_requests, issues, checks, statuses


def _assert_canonical_target(repo: str, issue_number: int) -> None:
    """Reject any target other than the repository's dedicated alert issue."""
    if repo != DEFAULT_REPO or issue_number != DEFAULT_ISSUE:
        raise MonitorError(
            f"monitor writes are restricted to {DEFAULT_REPO} issue #{DEFAULT_ISSUE}"
        )


def _read_target_issue(repo: str, issue_number: int) -> dict[str, Any]:
    """Read and validate the one permitted write target."""
    _assert_canonical_target(repo, issue_number)
    payload = _gh_json(f"repos/{repo}/issues/{issue_number}")
    if not isinstance(payload, dict) or int(payload.get("number", -1)) != issue_number:
        raise MonitorError("dedicated monitor issue identity could not be verified")
    if payload.get("state") != "open" or "pull_request" in payload:
        raise MonitorError("dedicated monitor target is not an open issue")
    return payload


def _update_target_issue(
    repo: str,
    issue_number: int,
    body: str,
    *,
    expected_previous: str | None,
) -> None:
    """Revalidate the marker, then perform the monitor's sole allowed write."""
    _assert_canonical_target(repo, issue_number)
    latest_target = _read_target_issue(repo, issue_number)
    observed_previous = extract_previous_fingerprint(str(latest_target.get("body") or ""))
    if observed_previous != expected_previous:
        raise MonitorError("dedicated monitor target changed during scan; refusing stale write")
    endpoint = f"repos/{repo}/issues/{issue_number}"
    command = ["gh", "api", "--method", "PATCH", endpoint, "--field", f"body={body}"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise MonitorError(f"dedicated issue update failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise MonitorError(f"dedicated issue update failed: {detail}")


def run_monitor(
    *,
    repo: str,
    issue_number: int,
    limit: int = DEFAULT_LIMIT,
    now: datetime | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one delta-only scan and update only the dedicated issue when needed."""
    scanned_at = now or datetime.now(UTC)
    _assert_canonical_target(repo, issue_number)
    pull_requests, issues, checks, statuses = _fetch_snapshot(repo, limit)
    findings = build_findings(
        pull_requests,
        issues,
        checks,
        now=scanned_at,
        target_issue=issue_number,
        statuses_by_pr=statuses,
    )
    fingerprint = compute_fingerprint(findings)
    target = _read_target_issue(repo, issue_number)
    previous = extract_previous_fingerprint(str(target.get("body") or ""))
    if not findings:
        return {
            "schema_version": SCHEMA_VERSION,
            "repo": repo,
            "issue": issue_number,
            "fingerprint": fingerprint,
            "previous_fingerprint": previous,
            "changed": False,
            "write_performed": False,
            "finding_count": 0,
        }
    changed = previous != fingerprint
    if changed and not dry_run:
        _update_target_issue(
            repo,
            issue_number,
            render_issue_body(findings, fingerprint=fingerprint, scanned_at=scanned_at),
            expected_previous=previous,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "issue": issue_number,
        "fingerprint": fingerprint,
        "previous_fingerprint": previous,
        "changed": changed,
        "write_performed": changed and not dry_run,
        "finding_count": len(findings),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the actionable-change monitor CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--issue", type=int, default=DEFAULT_ISSUE)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_monitor(
            repo=args.repo,
            issue_number=args.issue,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    except MonitorError as exc:
        raise SystemExit(f"actionable-change monitor failed closed: {exc}") from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
