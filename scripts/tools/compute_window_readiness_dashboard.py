#!/usr/bin/env python3
"""Sanitized compute-window readiness dashboard generator (issue #8911).

Renders one deterministic, public-safe JSON plus concise Markdown dashboard (schema
``robot_sf.compute_window_dashboard.v1``) from versioned canonical input reports (schema
``robot_sf.compute_window_dashboard_input.v1``). Reuses the operational-reporting
conventions of ``locator_snapshot.py`` and ``scheduler_allocation_receipt.py``: stable
reason codes, no private values in output, fail-closed validation, byte-stable rendering.

Implementation, compute, scheduler, artifact, evidence, review, and claim states stay
separate; no scientific score or admission decision is computed. Stale, missing,
contradictory, duplicate, wrong-schema, or unsanitized input masks affected rows as
explicit ``unavailable``. CLI: ``--inputs PATH [PATH ...] [--check] [--format
json|markdown]``; exit codes: 0 available/check-pass, 2 fail-closed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "robot_sf.compute_window_dashboard_input.v1"
DASHBOARD_SCHEMA = "robot_sf.compute_window_dashboard.v1"
CLAIM_BOUNDARY = (
    "Operational dashboard only: separated implementation/compute/scheduler/artifact/"
    "evidence/review/claim states are projections of sanitized inputs. No scientific score, "
    "campaign admission, or release decision is computed or implied."
)
AXES = ("implementation", "compute", "scheduler", "artifact", "evidence", "review", "claim")
AXIS_STATES = {
    "implementation": "not_started in_progress complete blocked unavailable".split(),
    "compute": "not_evaluated admission_ready blocked exhausted unavailable".split(),
    "scheduler": (
        "not_submitted pending running succeeded failed cancelled timeout unavailable".split()
    ),
    "artifact": (
        "none partial harvest_pending harvested preserved restore_verified cluster_only "
        "unavailable".split()
    ),
    "evidence": "pending available blocked not_applicable unavailable".split(),
    "review": "not_started in_review approved changes_requested unavailable".split(),
    "claim": "none diagnostic_only bounded not_applicable unavailable".split(),
}
ENUMS = {
    "priority": "p0 p1 p2 p3".split(),
    "tier": "tier_0 tier_1 tier_2 tier_3".split(),
    "resource_class": "local slurm_cpu slurm_gpu carla_host multi_host".split(),
    "access_urgency": "expiring deadline_bound routine".split(),
    "prerequisite_state": "met not_required unmet blocked".split(),
    "readiness": "ready stale missing not_required".split(),
    "harvest_state": "not_applicable pending partial harvested failed unavailable".split(),
    "preservation_state": "none cluster_only transfer_pending preserved failed unavailable".split(),
    "environment_state": "not_captured captured verified unavailable".split(),
    "restore_state": "not_applicable pending verified failed unavailable".split(),
    "deadline_fit": "fits tight misses unknown".split(),
}
EXPLICIT = frozenset(("not_observed", "unavailable", "not_applicable"))
NOT_SUBMITTED = frozenset(("not_submitted", "pending", "running"))
UNAVAILABLE = "unavailable"
DEFAULT_JSON_OUTPUT = Path("output/compute_window_dashboard/compute_window_dashboard.json")
URGENCY_RANK = {"expiring": 0, "deadline_bound": 1, "routine": 2}
PRIORITY_RANK = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
READINESS_RANK = {"met": 0, "not_required": 1, "unmet": 2, "blocked": 3}
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SECRET_RE = re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|token|credential|bearer)")
_SIGNED_QUERY_RE = re.compile(r"(?i)[?&](?:sig|signature|token|expires|x-amz-|x-goog-)")
_HOSTNAME_RE = re.compile(r"(?i)(?:[a-z0-9][a-z0-9-]{0,61}\.){2,}[a-z]{2,24}")
_FORBIDDEN_FIELDS = frozenset(
    "account command command_line cmdline env environment host hostname node node_list nodelist "
    "nodes partition password private_path qos scheduler_account secret secrets slurm_account "
    "token url user user_name username".split()
)
_ROW_KEYS = (
    "campaign_id public_owner priority tier resource_class access_urgency prerequisite_state "
    "source_status config_status checkpoint_status expected_rows observed_rows harvest_state "
    "preservation_state independent_copy_count environment_state restore_state deadline_fit "
    "next_owner".split()
)
_READINESS_KEYS = ("source_status", "config_status", "checkpoint_status")
_REPORT_KEYS = frozenset(("schema", "report_id", "source_owner", "generated_at", "rows"))


@dataclass(frozen=True, slots=True)
class DashboardIssue:
    """One sanitized fail-closed issue, free of private values."""

    code: str
    location: str
    message: str


@dataclass(frozen=True, slots=True)
class InputReport:
    """One parsed report: public identity, timestamp, shape-valid rows, and issues."""

    index: int
    report_id: str
    source_owner: str
    generated_at: datetime | None
    rows: tuple[Mapping[str, Any], ...]
    issues: tuple[DashboardIssue, ...]


def _issue(code: str, location: str, message: str) -> DashboardIssue:
    return DashboardIssue(code=code, location=location, message=message)


def _matches(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or value in EXPLICIT:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _render_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _scan_private(value: Any, location: str, issues: list[DashboardIssue]) -> None:  # noqa: C901
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{location}/{key}"
            if str(key).lower() in _FORBIDDEN_FIELDS:
                issues.append(_issue("forbidden_field", child, "private topology field"))
            _scan_private(item, child, issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_private(item, f"{location}[{index}]", issues)
    elif isinstance(value, str):
        if "://" in value or _SIGNED_QUERY_RE.search(value) or value.startswith(("/", "~", "\\")):
            issues.append(_issue("forbidden_value", location, "URL or private path"))
        if "@" in value or ".." in value.split("/") or any(ord(c) < 32 for c in value):
            issues.append(_issue("forbidden_value", location, "private identity or path"))
        if _HOSTNAME_RE.search(value):
            issues.append(_issue("forbidden_value", location, "hostname-like content"))
        if _SECRET_RE.search(value):
            issues.append(_issue("forbidden_value", location, "credential-like content"))


def _in(options: Sequence[str]):
    def check(value: Any) -> bool:
        return value in options

    return check


def _count(value: Any) -> bool:
    return _int_or_none(value) is not None and value >= 0


_LEAF_CHECKS: dict[str, Any] = {
    "slug": lambda value: _matches(value, _SLUG_RE),
    "row_count": lambda value: value in EXPLICIT or _count(value),
    "count": _count,
}
_ROW_SPEC: dict[str, Any] = {
    "campaign_id": "slug",
    "public_owner": "slug",
    "priority": _in(ENUMS["priority"]),
    "tier": _in(ENUMS["tier"]),
    "resource_class": _in(ENUMS["resource_class"]),
    "access_urgency": _in(ENUMS["access_urgency"]),
    "prerequisite_state": _in(ENUMS["prerequisite_state"]),
    "source_status": _in(ENUMS["readiness"]),
    "config_status": _in(ENUMS["readiness"]),
    "checkpoint_status": _in(ENUMS["readiness"]),
    "expected_rows": "row_count",
    "observed_rows": "row_count",
    "harvest_state": _in(ENUMS["harvest_state"]),
    "preservation_state": _in(ENUMS["preservation_state"]),
    "independent_copy_count": "count",
    "environment_state": _in(ENUMS["environment_state"]),
    "restore_state": _in(ENUMS["restore_state"]),
    "deadline_fit": _in(ENUMS["deadline_fit"]),
    "next_owner": "slug",
    "states": {axis: _in(states) for axis, states in AXIS_STATES.items()},
}


def _check_spec(
    node: Any, spec: Mapping[str, Any], location: str, issues: list[DashboardIssue]
) -> None:
    if not isinstance(node, Mapping):
        issues.append(_issue("invalid_field", location, "mapping required"))
        return
    for key, check in spec.items():
        if key not in node:
            issues.append(_issue("missing_field", f"{location}/{key}", "required"))
        elif isinstance(check, Mapping):
            _check_spec(node[key], check, f"{location}/{key}", issues)
        elif not (check if callable(check) else _LEAF_CHECKS[check])(node[key]):
            issues.append(_issue("invalid_field", f"{location}/{key}", "invalid value"))
    issues.extend(
        _issue("unknown_field", f"{location}/{key}", "not in the schema")
        for key in node
        if key not in spec
    )


def _row_conflicts(row: Mapping[str, Any]) -> list[str]:  # noqa: C901 - one consistency pass
    states = row["states"]
    scheduler, artifact = states["scheduler"], states["artifact"]
    preserve, restore = row["preservation_state"], row["restore_state"]
    copies = row["independent_copy_count"]
    expected, observed = _int_or_none(row["expected_rows"]), _int_or_none(row["observed_rows"])
    unready = any(row[key] in {"stale", "missing"} for key in _READINESS_KEYS)
    out: list[str] = []
    if scheduler == "running" and row["harvest_state"] == "harvested":
        out.append("harvest before terminal state")
    if scheduler in NOT_SUBMITTED and observed not in (None, 0):
        out.append("rows before terminal state")
    if artifact == "preserved" and (preserve != "preserved" or copies < 2):
        out.append("preservation does not match copies")
    if artifact == "restore_verified" and (
        restore != "verified" or preserve != "preserved" or copies < 2
    ):
        out.append("restore does not match custody")
    if artifact == "cluster_only" and (preserve != "cluster_only" or copies > 1):
        out.append("cluster-only custody conflict")
    if artifact == "harvested" and row["harvest_state"] != "harvested":
        out.append("artifact and harvest states differ")
    if artifact in {"none", "partial", "harvest_pending"} and preserve == "preserved":
        out.append("preserved before harvest")
    if restore == "verified" and preserve != "preserved":
        out.append("restore before preservation")
    if expected is not None and observed is not None and observed > expected:
        out.append("observed rows exceed expected")
    if states["compute"] == "admission_ready" and (
        row["prerequisite_state"] in {"unmet", "blocked"} or unready
    ):
        out.append("admission without prerequisites or readiness")
    if states["implementation"] == "complete" and unready:
        out.append("implementation status conflict")
    if states["evidence"] == "available" and row["harvest_state"] != "harvested":
        out.append("evidence before harvest")
    if states["claim"] == "bounded" and states["review"] != "approved":
        out.append("claim before review")
    return out


def _load_report(path: Path, index: int) -> InputReport:  # noqa: C901 - one report pass
    location = f"/inputs[{index}]"
    issues: list[DashboardIssue] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        issue = _issue("unreadable_input", location, f"cannot load JSON ({type(exc).__name__})")
        return InputReport(index, UNAVAILABLE, UNAVAILABLE, None, (), (issue,))
    _scan_private(payload, location, issues)
    if not isinstance(payload, Mapping):
        issues.append(_issue("invalid_report", location, "expected a JSON mapping"))
        return InputReport(index, UNAVAILABLE, UNAVAILABLE, None, (), tuple(issues))
    if payload.get("schema") != INPUT_SCHEMA:
        issues.append(_issue("invalid_schema", location, f"must equal {INPUT_SCHEMA}"))
    for key in payload:
        if key not in _REPORT_KEYS:
            issues.append(_issue("unknown_field", f"{location}/{key}", "not in the schema"))
    for key in ("report_id", "source_owner"):
        if not _matches(payload.get(key), _SLUG_RE):
            issues.append(_issue("invalid_field", f"{location}/{key}", "invalid slug"))
    generated_at = _parse_timestamp(payload.get("generated_at"))
    if generated_at is None:
        issues.append(_issue("invalid_field", f"{location}/generated_at", "invalid timestamp"))
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        issues.append(_issue("invalid_field", f"{location}/rows", "list required"))
        raw_rows = []
    rows: list[Mapping[str, Any]] = []
    for row_index, row in enumerate(raw_rows):
        row_location = f"{location}/rows[{row_index}]"
        before = len(issues)
        _scan_private(row, row_location, issues)
        _check_spec(row, _ROW_SPEC, row_location, issues)
        if len(issues) > before or not isinstance(row, Mapping):
            continue
        issues.extend(
            _issue("conflicting_state", row_location, message) for message in _row_conflicts(row)
        )
        rows.append(row)
    report_id, source_owner = payload.get("report_id"), payload.get("source_owner")
    return InputReport(
        index,
        report_id if _matches(report_id, _SLUG_RE) else UNAVAILABLE,
        source_owner if _matches(source_owner, _SLUG_RE) else UNAVAILABLE,
        generated_at,
        tuple(rows),
        tuple(issues),
    )


def _duplicate_issues(reports: Sequence[InputReport]) -> dict[int, list[DashboardIssue]]:
    found: dict[int, list[DashboardIssue]] = {}
    id_counts = Counter(report.report_id for report in reports)
    campaign_counts = Counter(str(row["campaign_id"]) for report in reports for row in report.rows)
    for report in reports:
        location = f"/inputs[{report.index}]"
        codes = []
        if report.report_id != UNAVAILABLE and id_counts[report.report_id] > 1:
            codes.append(_issue("duplicate_report", location, "report id is not unique"))
        if any(campaign_counts[str(row["campaign_id"])] > 1 for row in report.rows):
            codes.append(_issue("duplicate_campaign", location, "campaign id repeated"))
        if codes:
            found[report.index] = codes
    return found


def _entry(report: InputReport, row: Mapping[str, Any], codes: Sequence[str]) -> dict[str, Any]:
    masked = bool(codes)
    states = {axis: (UNAVAILABLE if masked else row["states"][axis]) for axis in AXES}
    values = {key: (UNAVAILABLE if masked else row[key]) for key in _ROW_KEYS}
    return {
        **values,
        "campaign_id": row["campaign_id"],
        "public_owner": row["public_owner"],
        "source_report": report.report_id,
        "source_owner": report.source_owner,
        "display": UNAVAILABLE if masked else "available",
        "reason_codes": sorted(set(codes)),
        "states": states,
        "job_state": states["scheduler"],
    }


def _input_record(
    report: InputReport, freshness: Sequence[DashboardIssue], max_age: int, reference: datetime
) -> dict[str, Any]:
    codes = sorted({issue.code for issue in report.issues} | {issue.code for issue in freshness})
    generated = report.generated_at
    state = UNAVAILABLE if generated is None else "stale" if "stale_input" in codes else "fresh"
    return {
        "report_id": report.report_id,
        "source_owner": report.source_owner,
        "generated_at": _render_timestamp(generated) if generated else UNAVAILABLE,
        "age_seconds": int((reference - generated).total_seconds()) if generated else None,
        "freshness": state,
        "max_age_seconds": max_age,
        "row_count": len(report.rows),
        "status": "rejected" if codes else "accepted",
        "issue_codes": codes,
    }


def build_dashboard(
    inputs: Sequence[Path], max_age_seconds: int = 86400, as_of: str | None = None
) -> dict[str, Any]:
    """Load canonical input reports and return one sanitized dashboard payload."""

    paths: list[Path] = []
    for raw in inputs:
        paths.extend(sorted(raw.glob("*.json")) if raw.is_dir() else [raw])
    reports = [_load_report(path, index) for index, path in enumerate(paths)]
    issues: list[DashboardIssue] = []
    if not paths:
        issues.append(_issue("missing_inputs", "/inputs", "no input reports found"))
    timestamps = [report.generated_at for report in reports if report.generated_at is not None]
    reference = max(timestamps) if timestamps else datetime(1970, 1, 1, tzinfo=UTC)
    if as_of is not None:
        parsed = _parse_timestamp(as_of)
        if parsed is None:
            raise ValueError("--as-of must be an ISO 8601 timestamp with a timezone")
        reference = parsed
    freshness: dict[int, list[DashboardIssue]] = {}
    for report in reports:
        generated = report.generated_at
        stale = (
            generated is not None and abs((reference - generated).total_seconds()) > max_age_seconds
        )
        freshness[report.index] = (
            [_issue("stale_input", f"/inputs[{report.index}]", "report is stale")] if stale else []
        )
    duplicates = _duplicate_issues(reports)
    entries: list[dict[str, Any]] = []
    for report in reports:
        combined = (*report.issues, *freshness[report.index], *duplicates.get(report.index, ()))
        issues.extend(combined)
        entries.extend(_entry(report, row, [i.code for i in combined]) for row in report.rows)
    entries.sort(
        key=lambda entry: (
            URGENCY_RANK.get(str(entry["access_urgency"]), 9),
            PRIORITY_RANK.get(str(entry["priority"]), 9),
            READINESS_RANK.get(str(entry["prerequisite_state"]), 9),
            str(entry["campaign_id"]),
            str(entry["source_report"]),
        )
    )
    available = sum(1 for entry in entries if entry["display"] == "available")
    return {
        "schema": DASHBOARD_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "status": UNAVAILABLE if issues else "available",
        "ok": not issues,
        "as_of": _render_timestamp(reference) if timestamps or as_of else UNAVAILABLE,
        "max_age_seconds": max_age_seconds,
        "summary": {
            "campaign_count": len(entries),
            "available_count": available,
            "unavailable_count": len(entries) - available,
            "input_count": len(reports),
            "issue_count": len(issues),
        },
        "inputs": [
            _input_record(report, freshness[report.index], max_age_seconds, reference)
            for report in reports
        ],
        "issues": [asdict(issue) for issue in issues],
        "campaigns": entries,
    }


def render_dashboard_json(payload: Mapping[str, Any]) -> str:
    """Return byte-stable dashboard JSON with sorted keys and a trailing newline."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_dashboard_markdown(payload: Mapping[str, Any]) -> str:
    """Return the concise deterministic Markdown dashboard projection."""

    summary = payload["summary"]
    lines = [
        "# Compute-Window Readiness Dashboard",
        "",
        f"- Schema: `{payload['schema']}` | Status: `{payload['status']}` | As of: `{payload['as_of']}`",
        f"- Inputs: {summary['input_count']} | Campaigns: {summary['campaign_count']}"
        f" | Available: {summary['available_count']}"
        f" | Unavailable: {summary['unavailable_count']} | Issues: {summary['issue_count']}",
        "",
        payload["claim_boundary"],
        "",
        "## Inputs",
        "",
    ]
    lines.extend(
        f"- `{item['report_id']}` ({item['source_owner']}): `{item['generated_at']}`,"
        f" age {item['age_seconds']}s, {item['freshness']}, {item['status']},"
        f" rows {item['row_count']}"
        for item in payload["inputs"]
    )
    for entry in payload["campaigns"]:
        states = entry["states"]
        fields = [f"{axis}: `{states[axis]}`" for axis in AXES]
        fields += [f"{key}: `{entry[key]}`" for key in _ROW_KEYS[2:]]
        fields.append(f"job_state: `{entry['job_state']}`")
        lines.append(
            f"\n### `{entry['campaign_id']}` ({entry['public_owner']}) — {entry['display']}"
            f"\n\n- " + " | ".join(fields) + "\n"
        )
    lines.extend(["", "## Fail-Closed Issues", ""])
    lines.extend(
        f"- `{issue['code']}` at `{issue['location']}`: {issue['message']}"
        for issue in payload["issues"]
    )
    if not payload["issues"]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the dashboard argument parser."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--check", action="store_true", help="validate and print without writing")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--max-age-seconds", type=int, default=86400)
    parser.add_argument("--as-of", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dashboard generator and return a shell-friendly exit code."""

    args = build_arg_parser().parse_args(argv)
    try:
        payload = build_dashboard(args.inputs, args.max_age_seconds, args.as_of)
    except ValueError as exc:
        sys.stderr.write(f"FAIL invalid_argument: {exc}\n")
        return 2
    if args.check:
        render = render_dashboard_json if args.format == "json" else render_dashboard_markdown
        sys.stdout.write(render(payload))
        return 0 if payload["ok"] else 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_dashboard_json(payload), encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_dashboard_markdown(payload), encoding="utf-8")
    sys.stdout.write(f"wrote {args.output} and {markdown}\n")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
