#!/usr/bin/env python3
"""Generate post-access-loss compute and artifact handoff (#8830).

Produces a deterministic, sanitized post-access handoff in JSON and Markdown
summarizing workloads, scheduler receipts, artifact custody, and environment
recreation states while eliminating private paths, credentials, and hosts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INPUT_SCHEMAS = frozenset(
    {"robot_sf.post_access_handoff_input.v1", "robot_sf.expiring_compute_inventory.v1"}
)
OUTPUT_SCHEMA = "robot_sf.post_access_handoff.v1"

WORKLOAD_STATES = (
    "completed_validated",
    "completed_unvalidated",
    "running",
    "queued",
    "failed",
    "cancelled",
    "harvest_blocked",
    "transfer_blocked",
    "not_submitted",
    "safe_to_defer",
)
INCOMPLETE_STATES = frozenset(
    {
        "completed_unvalidated",
        "running",
        "queued",
        "failed",
        "cancelled",
        "harvest_blocked",
        "transfer_blocked",
        "not_submitted",
    }
)

PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=])(/(?:home|tmp|var|opt|private|root)/|[a-zA-Z]:[/\\]Users[/\\])",
    re.IGNORECASE,
)
CREDENTIAL_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|AWS_SECRET_ACCESS_KEY|bearer\s+[A-Za-z0-9_\-\.]+"
    r"|['\"]?(?:password|passwd|api_key|secret_key)['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]",
    re.IGNORECASE,
)
SIGNED_URL_RE = re.compile(
    r"[?&](?:X-Amz-Signature|Signature|sig|token|api_key)=[a-zA-Z0-9_\-\.]+",
    re.IGNORECASE,
)
PRIVATE_HOST_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[\w-]+\.(?:cluster|local|internal|corp)\b",
    re.IGNORECASE,
)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (data, None) if isinstance(data, dict) else (None, "not a JSON object")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, str(exc)


def sanitize_text(text: str) -> str:
    """Sanitize private paths, credentials, signed URLs, and internal hosts.

    Returns:
        Redacted text safe for public documentation.
    """
    s = PRIVATE_PATH_RE.sub("[REDACTED_PATH]", text)
    s = CREDENTIAL_RE.sub("[REDACTED_CREDENTIAL]", s)
    s = SIGNED_URL_RE.sub("?[REDACTED_QUERY]", s)
    return PRIVATE_HOST_RE.sub("[REDACTED_HOST]", s)


def _check_private_str(text: str, path: str, disc: list[str], reasons: list[str]) -> None:
    checks = (
        (CREDENTIAL_RE, "credential_leak"),
        (SIGNED_URL_RE, "signed_url_leak"),
        (PRIVATE_PATH_RE, "private_path_leak"),
        (PRIVATE_HOST_RE, "private_host_leak"),
    )
    for pat, tag in checks:
        if pat.search(text) and tag not in reasons:
            disc.append(f"{tag}: {path}")
            reasons.append(tag)


def _scan_for_private_values(data: Any, path: str, disc: list[str], reasons: list[str]) -> None:
    if isinstance(data, str):
        _check_private_str(data, path, disc, reasons)
    elif isinstance(data, dict):
        for k, v in data.items():
            _scan_for_private_values(v, f"{path}.{k}", disc, reasons)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _scan_for_private_values(item, f"{path}[{idx}]", disc, reasons)


def _validate_contradictions(
    w: dict[str, Any], loc: str, disc: list[str], reasons: list[str]
) -> None:
    state = w.get("current_state")
    if state == "completed_validated":
        if w.get("restore_status") == "blocked":
            disc.append(f"contradictory_status: {loc} completed_validated with blocked restore")
            reasons.append("contradictory_status")
        exp_r, obs_r = w.get("expected_rows"), w.get("observed_rows")
        if isinstance(exp_r, int) and isinstance(obs_r, int) and obs_r < exp_r:
            disc.append(f"contradictory_status: {loc} observed rows {obs_r} < expected {exp_r}")
            reasons.append("contradictory_status")
        if w.get("remaining_blocker"):
            disc.append(f"contradictory_status: {loc} completed_validated has remaining blocker")
            reasons.append("contradictory_status")
    elif state == "not_submitted" and w.get("scheduler_job"):
        disc.append(f"contradictory_status: {loc} not_submitted has scheduler job")
        reasons.append("contradictory_status")


def _validate_workload(w: dict[str, Any], idx: int, disc: list[str], reasons: list[str]) -> None:
    loc = f"workload[{idx}]"
    state = w.get("current_state")
    issue_num = w.get("public_issue")
    if not issue_num:
        disc.append(f"missing_public_issue: {loc}")
        reasons.append("missing_public_issue")
    if state not in WORKLOAD_STATES:
        disc.append(f"invalid_state: {loc} ({state})")
        reasons.append("invalid_state")
        return
    if state in INCOMPLETE_STATES:
        cmd = w.get("next_command")
        if not cmd or not str(cmd).strip():
            disc.append(f"missing_next_action: {loc} (issue {issue_num})")
            if "missing_next_action" not in reasons:
                reasons.append("missing_next_action")
    _validate_contradictions(w, loc, disc, reasons)


def _check_workload_identities(
    workloads: list[dict[str, Any]], disc: list[str], reasons: list[str]
) -> tuple[set[Any], set[str], set[str]]:
    seen_identities: set[tuple[Any, Any, Any]] = set()
    ref_jobs: set[Any] = set()
    ref_arts: set[str] = set()
    issue_nums: set[str] = set()
    for idx, w in enumerate(workloads):
        p_iss = str(w.get("public_issue", ""))
        issue_nums.add(p_iss)
        ident = w.get("identities", {})
        key = (p_iss, ident.get("config_path"), ident.get("seed"))
        if key in seen_identities and "duplicate_identity" not in reasons:
            disc.append(f"duplicate_identity: workload[{idx}] {key}")
            reasons.append("duplicate_identity")
        seen_identities.add(key)
        job = w.get("scheduler_job")
        if isinstance(job, dict) and "job_id" in job:
            ref_jobs.add(job["job_id"])
        elif isinstance(job, (int, str)):
            ref_jobs.add(job)
        if m := w.get("artifact_manifest"):
            ref_arts.add(str(m))
    return ref_jobs, ref_arts, issue_nums


def _validate_orphans(
    receipts: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    ref_jobs: set[Any],
    ref_arts: set[str],
    issue_nums: set[str],
    disc: list[str],
    reasons: list[str],
) -> None:
    for idx, r in enumerate(receipts):
        j_id = r.get("job_id")
        if j_id and j_id not in ref_jobs:
            disc.append(f"orphan_job: receipt[{idx}] job_id {j_id}")
            if "orphan_job" not in reasons:
                reasons.append("orphan_job")
    for idx, a in enumerate(artifacts):
        a_id = str(a.get("artifact_id", ""))
        consumer = str(a.get("consumer", ""))
        if a_id not in ref_arts and consumer not in issue_nums:
            disc.append(f"orphan_artifact: artifact[{idx}] {a_id}")
            if "orphan_artifact" not in reasons:
                reasons.append("orphan_artifact")


def generate_handoff(inventory_path: Path) -> dict[str, Any]:
    """Generate post-access-loss compute and artifact handoff report.

    Returns:
        A dictionary report adhering to robot_sf.post_access_handoff.v1 schema.
    """
    disc: list[str] = []
    reasons: list[str] = []
    if not inventory_path.is_file():
        return {
            "schema": OUTPUT_SCHEMA,
            "verdict": "blocked",
            "as_of": "unknown",
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": {},
            "workloads": [],
            "artifacts": [],
            "environments": [],
            "discrepancies": [f"missing inventory file: {inventory_path}"],
            "reasons": ["inventory_not_found"],
        }
    raw_data, err = _read_json(inventory_path)
    if err or not isinstance(raw_data, dict):
        return {
            "schema": OUTPUT_SCHEMA,
            "verdict": "fail",
            "as_of": "unknown",
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": {},
            "workloads": [],
            "artifacts": [],
            "environments": [],
            "discrepancies": [f"inventory JSON error: {err}"],
            "reasons": ["unsupported_schema"],
        }
    if raw_data.get("schema") not in INPUT_SCHEMAS:
        disc.append(f"unsupported_input_schema: {raw_data.get('schema')}")
        reasons.append("unsupported_schema")
    as_of = raw_data.get("as_of", "2026-09-12T12:00:00Z")
    workloads = raw_data.get("workloads", [])
    artifacts = raw_data.get("artifacts", [])
    environments = raw_data.get("environments", [])
    receipts = raw_data.get("scheduler_receipts", [])
    _scan_for_private_values(raw_data, "inventory", disc, reasons)
    for idx, w in enumerate(workloads):
        _validate_workload(w, idx, disc, reasons)
    ref_jobs, ref_arts, issue_nums = _check_workload_identities(workloads, disc, reasons)
    _validate_orphans(receipts, artifacts, ref_jobs, ref_arts, issue_nums, disc, reasons)
    blocked_reasons = {
        "private_path_leak",
        "credential_leak",
        "signed_url_leak",
        "private_host_leak",
    }
    is_blocked = any(r in reasons for r in blocked_reasons)
    verdict = "blocked" if is_blocked else ("fail" if disc else "pass")
    state_counts: dict[str, int] = {}
    for w in workloads:
        st = w.get("current_state", "unknown")
        state_counts[st] = state_counts.get(st, 0) + 1
    summary = {
        "total_workloads": len(workloads),
        "total_artifacts": len(artifacts),
        "total_environments": len(environments),
        "states": state_counts,
    }
    return {
        "schema": OUTPUT_SCHEMA,
        "verdict": verdict,
        "as_of": as_of,
        "generated_at": as_of,
        "summary": summary,
        "workloads": workloads,
        "artifacts": artifacts,
        "environments": environments,
        "discrepancies": disc,
        "reasons": reasons,
    }


def render_markdown_summary(report: dict[str, Any]) -> str:
    """Render a concise markdown handoff report.

    Returns:
        Formatted markdown summary string.
    """
    lines = [
        "# Compute Window Post-Access Handoff",
        "",
        f"**As Of**: `{report.get('as_of', 'unknown')}` | **Verdict**: `{report.get('verdict', 'unknown')}`",
        "",
        "## 1. Workload Portfolio",
        "",
        "| Issue | Purpose | Resource | State | Restore | Next Action |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for w in report.get("workloads", []):
        iss = f"#{w.get('public_issue', '')}"
        purp = sanitize_text(str(w.get("purpose", "")))
        res = w.get("resource_class", "")
        st = w.get("current_state", "")
        rest = w.get("restore_status", "")
        nxt = sanitize_text(str(w.get("next_command") or "none"))
        lines.append(f"| {iss} | {purp} | {res} | {st} | {rest} | `{nxt}` |")
    lines.extend(
        [
            "",
            "## 2. Artifact Custody",
            "",
            "| Artifact ID | Retention | Size (bytes) | Custody | Consumer |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for a in report.get("artifacts", []):
        a_id = sanitize_text(str(a.get("artifact_id", "")))
        ret = a.get("retention_class", "")
        sz = a.get("size_bytes", 0)
        cust = a.get("custody_state", "")
        cons = a.get("consumer", "")
        lines.append(f"| {a_id} | {ret} | {sz} | {cust} | {cons} |")
    lines.extend(
        [
            "",
            "## 3. Environments",
            "",
            "| Environment ID | Recreation Capability | Unavailable Fields |",
            "| :--- | :--- | :--- |",
        ]
    )
    for e in report.get("environments", []):
        e_id = sanitize_text(str(e.get("environment_id", "")))
        cap = e.get("recreation_capability", "")
        unavail = ", ".join(e.get("unavailable_fields", [])) or "none"
        lines.append(f"| {e_id} | {cap} | {unavail} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for generate_post_access_handoff.py.

    Returns:
        Exit code: 0 for pass, 1 for fail, 2 for blocked.
    """
    parser = argparse.ArgumentParser(description="Generate post-access compute handoff.")
    parser.add_argument(
        "--inventory", required=True, type=Path, help="Path to compute inventory JSON."
    )
    parser.add_argument("--check", action="store_true", help="Perform check-only verification.")
    parser.add_argument(
        "--format", choices=["json", "text", "markdown"], default="json", help="Output format."
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output path.")
    args = parser.parse_args(argv)

    report = generate_handoff(args.inventory)
    if args.format == "json":
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    elif args.format == "markdown":
        rendered = render_markdown_summary(report)
    else:
        rendered = f"Verdict: {report['verdict'].upper()}\nSummary: {report['summary']}\n"
        if report["discrepancies"]:
            rendered += f"Discrepancies: {'; '.join(report['discrepancies'])}\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)

    if report["verdict"] == "pass":
        return 0
    if report["verdict"] == "fail":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
