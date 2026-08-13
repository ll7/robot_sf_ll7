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
SCHEMA_VERSION = "robot_sf.actionable_change_monitor.v2"
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
SLURM_STATE_BLOCK_RE = re.compile(
    r"(?im)^(?P<indent>[ \t]*)(?P<block>"
    r"slurm_experiment_state|slurm_issue_pointer|slurm_issue_status"
    r")[ \t]*:[ \t]*(?:#.*)?$"
)
SLURM_STATE_FIELD_RE = re.compile(
    r"^(?P<indent>[ \t]+)(?P<key>state|slurm_state|slurm_job_id)"
    r"[ \t]*:[ \t]*(?P<value>.*?)[ \t]*$"
)
SLURM_STATES = frozenset(
    {
        "not_submitted",
        "submitted_running",
        "completed_pending_artifact_promotion",
        "artifact_rescue",
        "rerun_required",
        "failed_closed",
        "inconclusive_close",
        "completed_with_durable_evidence",
        "parent_blocked",
        "insufficient_data",
    }
)
SUBMITTED_SLURM_STATE = "submitted_running"
SLURM_STATE_BLOCK_SPECS = {
    "slurm_experiment_state": {"state_key": "state", "job_id_required": True},
    "slurm_issue_pointer": {"state_key": "slurm_state", "job_id_required": False},
    "slurm_issue_status": {"state_key": "state", "job_id_required": True},
}
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


def _parse_scalar(raw_value: str, field: str) -> tuple[str | None, str | None]:
    """Parse one conservative YAML scalar without treating prose as state."""
    value = raw_value.strip()
    if value.startswith(("'", '"')):
        quote = value[0]
        closing_quote = value.find(quote, 1)
        if closing_quote < 0:
            return None, f"unterminated quoted {field}"
        trailing = value[closing_quote + 1 :].strip()
        if trailing and not trailing.startswith("#"):
            return None, f"unexpected text after quoted {field}"
        value = value[1:closing_quote].strip()
    else:
        value = value.split("#", 1)[0].strip()
    if not value:
        return None, f"missing {field}"
    return value, None


def _parse_state_block_fields(
    lines: list[str],
    start: int,
    header_indent: int,
    block: str,
    state_key: str,
) -> tuple[dict[str, str], list[str]]:
    """Extract the supported direct fields from one state block."""
    fields: dict[str, str] = {}
    errors: list[str] = []
    for line in lines[start:]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent <= header_indent:
            break
        field_match = SLURM_STATE_FIELD_RE.match(line)
        if not field_match:
            continue
        key = field_match.group("key")
        if key not in {state_key, "slurm_job_id"}:
            errors.append(f"unexpected {key} in {block}")
            continue
        if key in fields:
            errors.append(f"duplicate {key} in {block}")
            continue
        parsed, error = _parse_scalar(field_match.group("value"), key)
        if error:
            errors.append(f"{block}: {error}")
        elif parsed is not None:
            fields[key] = parsed
    return fields, errors


def _state_block_record(
    block: str,
    fields: dict[str, str],
    *,
    state_key: str,
    job_id_required: bool,
) -> tuple[tuple[str, str, str | None, bool] | None, list[str]]:
    """Validate one block and return its normalized state/job-id record."""
    errors: list[str] = []
    state = fields.get(state_key)
    if state is None:
        return None, [f"{block}: missing {state_key}"]
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", state):
        errors.append(f"malformed {state_key}={state!r}")
    elif state not in SLURM_STATES:
        errors.append(f"unsupported {state_key}={state!r}")

    raw_job_id = fields.get("slurm_job_id")
    job_id: str | None = None
    if raw_job_id == "not_submitted":
        job_id = raw_job_id
    elif raw_job_id is not None and re.fullmatch(r"\d+", raw_job_id):
        job_id = raw_job_id
    elif raw_job_id is not None:
        errors.append(f"malformed slurm_job_id={raw_job_id!r}")
    elif job_id_required:
        errors.append(f"{block}: missing slurm_job_id")
    return (block, state, job_id, job_id_required), errors


def _parse_slurm_execution_state(
    body: str,
) -> tuple[str | None, str | None, str | None]:
    """Read the canonical ledger block and its documented compatibility aliases.

    The monitor intentionally does not accept a free-standing ``slurm_state``
    line: only ``slurm_experiment_state``, ``slurm_issue_pointer``, and
    ``slurm_issue_status`` are structured state surfaces.  Multiple surfaces
    are allowed when their values agree, which lets the short pointer coexist
    with the full ledger block without creating a second execution vocabulary.
    """
    lines = body.splitlines()
    records: list[tuple[str, str, str | None, bool]] = []
    errors: list[str] = []
    for header in SLURM_STATE_BLOCK_RE.finditer(body):
        block = header.group("block")
        header_indent = len(header.group("indent").expandtabs(8))
        spec = SLURM_STATE_BLOCK_SPECS[block]
        start = body[: header.start()].count("\n") + 1
        fields, field_errors = _parse_state_block_fields(
            lines, start, header_indent, block, spec["state_key"]
        )
        errors.extend(field_errors)
        record, record_errors = _state_block_record(
            block,
            fields,
            state_key=spec["state_key"],
            job_id_required=spec["job_id_required"],
        )
        errors.extend(record_errors)
        if record is not None:
            records.append(record)

    if not records:
        return None, None, "missing structured SLURM execution state"
    if errors:
        return None, None, "; ".join(errors)

    states = {record[1] for record in records}
    if len(states) != 1:
        return None, None, "contradictory structured SLURM states"
    state = states.pop()
    job_ids = {record[2] for record in records if record[2] is not None}
    if len(job_ids) > 1:
        return None, None, "contradictory structured SLURM job IDs"
    job_id = next(iter(job_ids), None)

    if state == "not_submitted" and job_id not in (None, "not_submitted"):
        return None, None, "not_submitted state has a numeric SLURM job ID"
    if state == SUBMITTED_SLURM_STATE and job_id == "not_submitted":
        return None, None, "submitted_running state has slurm_job_id=not_submitted"
    return state, job_id, None


def _parse_slurm_state(body: str) -> tuple[str | None, str | None]:
    """Return the normalized state and an unavailable reason for compatibility."""
    state, _job_id, error = _parse_slurm_execution_state(body)
    return state, error


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

    if "evidence:launch-packet" in label_set and (
        "resource:slurm" in label_set or "slurm" in label_set or "slurm" in body
    ):
        slurm_state, slurm_job_id, state_error = _parse_slurm_execution_state(
            str(issue.get("body") or "")
        )
        if state_error:
            findings.append(
                _finding(
                    finding_id=f"issue:{issue['number']}:launch-state-unavailable",
                    kind="launch_packet_state_unavailable",
                    target=issue,
                    detail=(
                        f"SLURM launch packet has no usable canonical state ({state_error}); "
                        "no execution claim inferred"
                    ),
                )
            )
        elif slurm_state == SUBMITTED_SLURM_STATE and slurm_job_id is None:
            findings.append(
                _finding(
                    finding_id=f"issue:{issue['number']}:launch-without-job-id",
                    kind="launch_packet_without_job_id",
                    target=issue,
                    detail=(
                        "SLURM launch packet explicitly reports submitted_running but has "
                        "no recorded job ID"
                    ),
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
            findings.extend(
                finding
                for finding in _surface_findings(issue)
                if finding["kind"] != "watched_research_surface"
            )
    return sorted(findings, key=lambda row: (row["kind"], row["number"], row["id"]))


def build_inventory(
    pull_requests: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    target_issue: int = DEFAULT_ISSUE,
) -> list[dict[str, Any]]:
    """Build stable watch inventory without presenting it as an actionable finding."""
    inventory: list[dict[str, Any]] = []
    for pr in pull_requests:
        if _is_research_surface(pr):
            inventory.append(
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

    for issue in issues:
        number = int(issue["number"])
        if number == target_issue or not _is_research_surface(issue):
            continue
        labels = _labels(issue)
        body = str(issue.get("body") or "").casefold()
        if _has_watch_label(labels) or any(term in body for term in WATCH_TERMS):
            watched_labels = ", ".join(labels) or "keyword-only"
            inventory.append(
                _finding(
                    finding_id=f"issue:{number}:watched-surface",
                    kind="watched_research_surface",
                    target=issue,
                    detail=f"labels/terms={watched_labels}; updated={issue.get('updated_at', 'unknown')}",
                )
            )
    return sorted(inventory, key=lambda row: (row["kind"], row["number"], row["id"]))


def summarize_inventory(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Return deterministic counts for stable inventory rows."""
    by_kind: dict[str, int] = {}
    for row in inventory:
        kind = str(row.get("kind", "unknown"))
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {"total": len(inventory), "by_kind": dict(sorted(by_kind.items()))}


def _parse_timestamp(value: Any) -> datetime:
    """Parse a GitHub ISO timestamp, failing closed on unusable values.

    A silent epoch fallback is unsafe here: every classification that compares
    a timestamp against ``now`` would then treat malformed API data as
    arbitrarily old and publish a fabricated ``stale_research_draft`` finding.
    Surface rows are validated up front by ``_validate_surface_identity``, so
    reaching this function with an unparseable value means the snapshot cannot
    be classified and the monitor must stop before it writes anything.
    """
    return _require_timestamp(value, context="classified surface timestamp")


def _require_timestamp(value: Any, *, context: str) -> datetime:
    """Return a timezone-aware UTC timestamp or raise ``MonitorError``."""
    if not isinstance(value, str) or not value.strip():
        raise MonitorError(f"{context} is missing or not a string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise MonitorError(f"{context} is not an ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise MonitorError(f"{context} has no timezone: {value!r}")
    return parsed.astimezone(UTC)


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


def compute_report_fingerprint(findings: list[dict[str, Any]], inventory: dict[str, Any]) -> str:
    """Return a stable fingerprint for actionable findings plus compact inventory counts."""
    payload = {
        "findings": sorted(
            findings,
            key=lambda row: (
                str(row.get("kind", "")),
                int(row.get("number", 0)),
                str(row.get("id", "")),
                _canonical_json(row),
            ),
        ),
        "inventory": inventory,
    }
    canonical = _canonical_json(payload).encode("utf-8")
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
    inventory: dict[str, Any] | None = None,
) -> str:
    """Render the latest alert report and preserve the monitor's safety contract."""
    inventory = inventory or {"total": 0, "by_kind": {}}
    lines = [
        "# RobotSF actionable-change monitor",
        "",
        "This advisory report is updated only when the monitored repository snapshot changes.",
        "It never merges pull requests, changes research state, submits compute, or writes any",
        "issue other than this dedicated target.",
        "",
        f"Scanned at: `{scanned_at.astimezone(UTC).isoformat()}`",
        f"Finding count: `{len(findings)}`",
        f"Stable inventory count: `{inventory.get('total', 0)}`",
        "",
        f"<!-- actionable-change-monitor:fingerprint:{fingerprint} -->",
        "",
    ]
    if not findings:
        lines.extend(
            [
                "## No actionable changes",
                "",
                "No actionable contradictions or failures were found; stable watch inventory is summarized below.",
                "",
            ]
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
    inventory_by_kind = inventory.get("by_kind", {})
    if inventory_by_kind:
        lines.extend(
            [
                "## Stable watch inventory",
                "",
                "Stable watched surfaces are summarized here and are not emitted as one actionable row per issue.",
                "",
                "| Kind | Count |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| `{kind}` | {count} |" for kind, count in inventory_by_kind.items())
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


def _validate_surface_row(
    row: dict[str, Any],
    *,
    resource: str,
    index: int,
    pull_request: bool,
) -> dict[str, Any]:
    """Validate the fields used to classify one GitHub issue or pull request."""
    _validate_surface_identity(row, resource=resource, index=index)
    _validate_surface_labels(row, resource=resource, index=index)
    if pull_request:
        _validate_pull_request_fields(row, resource=resource, index=index)
    elif "pull_request" in row and not isinstance(row["pull_request"], dict):
        raise MonitorError(f"{resource} response has malformed pull_request at index {index}")
    return row


def _validate_surface_identity(row: dict[str, Any], *, resource: str, index: int) -> None:
    """Validate common issue and pull-request identity fields."""
    number = row.get("number")
    if type(number) is not int or number < 1:
        raise MonitorError(f"{resource} response has invalid number at index {index}")
    if not isinstance(row.get("title"), str):
        raise MonitorError(f"{resource} response has invalid title at index {index}")
    body = row.get("body")
    if body is not None and not isinstance(body, str):
        raise MonitorError(f"{resource} response has invalid body at index {index}")
    html_url = row.get("html_url")
    if not isinstance(html_url, str) or not html_url:
        raise MonitorError(f"{resource} response has invalid html_url at index {index}")
    # Timestamps decide staleness classification, so they are validated as real
    # timezone-aware instants here rather than as non-empty strings.  A row with
    # ``updated_at='not-an-ISO-timestamp'`` must stop the scan instead of being
    # silently classified as epoch-old and published as a false finding.
    _require_timestamp(
        row.get("updated_at"),
        context=f"{resource} response updated_at at index {index}",
    )


def _validate_surface_labels(row: dict[str, Any], *, resource: str, index: int) -> None:
    """Validate the label objects used to select monitored surfaces."""
    labels = row.get("labels")
    if not isinstance(labels, list):
        raise MonitorError(f"{resource} response has invalid labels at index {index}")
    for label_index, label in enumerate(labels):
        if not isinstance(label, dict) or not isinstance(label.get("name"), str):
            raise MonitorError(
                f"{resource} response has malformed label at index {index}:{label_index}"
            )


def _validate_pull_request_fields(row: dict[str, Any], *, resource: str, index: int) -> None:
    """Validate pull-request-only fields used by the monitor."""
    if type(row.get("draft")) is not bool:
        raise MonitorError(f"{resource} response has invalid draft at index {index}")
    head = row.get("head")
    if not isinstance(head, dict) or not isinstance(head.get("sha"), str) or not head["sha"]:
        raise MonitorError(f"{resource} response has invalid head at index {index}")


def _require_surface_rows(
    payload: Any,
    *,
    resource: str,
    pull_request: bool,
) -> list[dict[str, Any]]:
    """Validate all fields needed from a paginated issue or pull-request response."""
    rows = _require_object_rows(payload, resource=resource)
    return [
        _validate_surface_row(
            row,
            resource=resource,
            index=index,
            pull_request=pull_request,
        )
        for index, row in enumerate(rows)
    ]


def _validate_commit_collection_rows(
    collection: list[Any],
    *,
    resource: str,
    collection_key: str,
    sha: str,
) -> list[dict[str, Any]]:
    """Validate the fields used to classify check runs or legacy commit statuses."""
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(collection):
        if not isinstance(row, dict):
            raise MonitorError(
                f"{resource} response for commit {sha} has malformed row at index {index}"
            )
        if collection_key == "check_runs":
            if type(row.get("id")) is not int or row["id"] < 1:
                raise MonitorError(
                    f"{resource} response for commit {sha} has invalid id at index {index}"
                )
            if not isinstance(row.get("name"), str) or not row["name"]:
                raise MonitorError(
                    f"{resource} response for commit {sha} has invalid name at index {index}"
                )
            if not isinstance(row.get("status"), str) or not row["status"]:
                raise MonitorError(
                    f"{resource} response for commit {sha} has invalid status at index {index}"
                )
            if row.get("conclusion") is not None and not isinstance(row["conclusion"], str):
                raise MonitorError(
                    f"{resource} response for commit {sha} has invalid conclusion at index {index}"
                )
        else:
            if not isinstance(row.get("context"), str) or not row["context"]:
                raise MonitorError(
                    f"{resource} response for commit {sha} has invalid context at index {index}"
                )
            if not isinstance(row.get("state"), str) or not row["state"]:
                raise MonitorError(
                    f"{resource} response for commit {sha} has invalid state at index {index}"
                )
        rows.append(row)
    return rows


def _read_commit_collection_page(
    repo: str,
    sha: str,
    *,
    resource: str,
    collection_key: str,
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """Read and validate one bounded commit collection page."""
    payload = _gh_json(f"repos/{repo}/commits/{sha}/{resource}?per_page={page_size}&page={page}")
    if not isinstance(payload, dict):
        raise MonitorError(f"{resource} response for commit {sha} is not an object")
    collection = payload.get(collection_key)
    if not isinstance(collection, list):
        raise MonitorError(
            f"{resource} response for commit {sha} has no valid {collection_key} list"
        )
    collection = _validate_commit_collection_rows(
        collection,
        resource=resource,
        collection_key=collection_key,
        sha=sha,
    )
    total_count = payload.get("total_count")
    if type(total_count) is not int or total_count < 0:
        raise MonitorError(f"{resource} response for commit {sha} has invalid total_count")
    if len(collection) > page_size:
        raise MonitorError(
            f"{resource} response for commit {sha} returned more than {page_size} rows"
        )
    return collection, total_count


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
        collection, total_count = _read_commit_collection_page(
            repo,
            sha,
            resource=resource,
            collection_key=collection_key,
            page=page,
            page_size=page_size,
        )
        rows.extend(collection)
        if total_count < len(rows):
            raise MonitorError(
                f"{resource} response for commit {sha} reports total_count={total_count} "
                f"below {len(rows)} returned rows"
            )
        if len(rows) >= limit:
            return rows[:limit]
        if len(collection) < page_size:
            if total_count > len(rows):
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
    pull_requests = _require_surface_rows(
        _gh_json(
            f"repos/{repo}/pulls?state=open&sort=updated&direction=desc&per_page={min(limit, 100)}",
            paginate=True,
            max_pages=max(1, (limit + 99) // 100),
        ),
        resource="pull requests",
        pull_request=True,
    )
    issues_payload = _require_surface_rows(
        _gh_json(
            f"repos/{repo}/issues?state=open&sort=updated&direction=desc&per_page={min(limit, 100)}",
            paginate=True,
            max_pages=max(1, (limit + 99) // 100),
        ),
        resource="issues",
        pull_request=False,
    )
    issues = [row for row in issues_payload if "pull_request" not in row]
    checks: dict[int, list[dict[str, Any]]] = {}
    statuses: dict[int, list[dict[str, Any]]] = {}
    for pr in pull_requests:
        if not _is_research_surface(pr):
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
    if not isinstance(payload, dict):
        raise MonitorError("dedicated monitor issue identity could not be verified")
    _validate_surface_row(
        payload,
        resource="dedicated monitor issue",
        index=0,
        pull_request=False,
    )
    if payload["number"] != issue_number:
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
    expected_body: str | None,
) -> None:
    """Revalidate the marker, write once, then prove the write survived.

    GitHub's issue API has no ``If-Match`` compare-and-swap, so the monitor
    composes the two mechanisms it does have into a single-writer contract:

    1. The workflow serializes every run under one target-wide concurrency
       group, so scheduled and ``workflow_dispatch`` runs from any ref queue
       behind each other instead of writing issue #6819 concurrently.
    2. This function re-reads the body immediately before the PATCH (rejecting
       any change since the scan) and re-reads it immediately after (rejecting
       any change since the PATCH).  The post-write read-back is what closes
       the residual time-of-check/time-of-use window: if a second writer
       overwrites this report, the losing writer observes a body it did not
       author and fails closed instead of reporting a successful publish.
    """
    _assert_canonical_target(repo, issue_number)
    latest_target = _read_target_issue(repo, issue_number)
    observed_previous = extract_previous_fingerprint(str(latest_target.get("body") or ""))
    if observed_previous != expected_previous or latest_target.get("body") != expected_body:
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
    _assert_write_survived(repo, issue_number, body)


def _assert_write_survived(repo: str, issue_number: int, body: str) -> None:
    """Fail closed when a concurrent writer overwrote the report just published."""
    committed = _read_target_issue(repo, issue_number)
    committed_body = str(committed.get("body") or "")
    if committed_body != body:
        raise MonitorError(
            "dedicated monitor target was overwritten by a concurrent writer; "
            "published report was not durable"
        )


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
    inventory = summarize_inventory(
        build_inventory(pull_requests, issues, target_issue=issue_number)
    )
    fingerprint = compute_report_fingerprint(findings, inventory)
    target = _read_target_issue(repo, issue_number)
    target_body = target.get("body")
    previous = extract_previous_fingerprint(str(target_body or ""))
    if not findings and not inventory["total"]:
        return {
            "schema_version": SCHEMA_VERSION,
            "repo": repo,
            "issue": issue_number,
            "fingerprint": fingerprint,
            "previous_fingerprint": previous,
            "changed": False,
            "write_performed": False,
            "finding_count": 0,
            "inventory_count": 0,
            "inventory_by_kind": {},
        }
    changed = previous != fingerprint
    if changed and not dry_run:
        _update_target_issue(
            repo,
            issue_number,
            render_issue_body(
                findings,
                fingerprint=fingerprint,
                scanned_at=scanned_at,
                inventory=inventory,
            ),
            expected_previous=previous,
            expected_body=target_body,
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
        "inventory_count": inventory["total"],
        "inventory_by_kind": inventory["by_kind"],
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
