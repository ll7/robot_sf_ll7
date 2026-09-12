#!/usr/bin/env python3
"""Reconcile sanitized scheduler-job projections with public owners and artifacts (#8838).

Joins one sanitized scheduler inventory projection (per-job public scheduler allocation
receipts plus explicit array/lineage context) with a public metadata snapshot (issue/PR
states, immutable launch-manifest packet digests, expected row counts, artifact roots,
owners, harvest/transfer states), and emits one deterministic custody row per job.

Check-only: never submits, cancels, relabels, claims, comments, or deletes scheduler,
GitHub, or artifact state, and never contacts live infrastructure. Ownership binds only
through exact sanitized identities (job alias, task alias, packet digest, output root);
mutable job names are never inferred into ownership, and unprovable ownership stays
explicit. Scheduler terminal state is custody metadata only: it never implies result
validity, artifact completeness, or scientific success.

Classifications: ``owned_active``, ``owned_terminal_unharvested``, ``owned_harvested``,
``duplicate_candidate``, ``orphan_unknown``, ``stale_input``, ``missing_output_owner``,
``projection_unavailable``. CLI: ``--check --projection <json> [--public <json>]
[--format json|text]``. Exit codes: 0 every row active or harvested, 1 actionable rows,
2 malformed input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from scripts.tools.classify_scheduler_failure import EXPLICIT_TOKENS, sanitize_text
from scripts.tools.generate_slurm_launch_manifest import SCHEMA_VERSION as LAUNCH_MANIFEST_SCHEMA
from scripts.tools.scheduler_allocation_receipt import validate_receipt
from scripts.tools.slurm_job_finalize import classify_durable_status, normalize_job_state
from scripts.tools.validate_receipt_projection import load_contracts, private_content_reason

PROJECTION_SCHEMA = "robot_sf.scheduler_job_reconciliation.projection.v1"
REPORT_SCHEMA = "robot_sf.scheduler_job_reconciliation.v1"
PUBLIC_URL_PREFIX = "https://github.com/ll7/robot_sf_ll7"
CLAIM_BOUNDARY = (
    "Scheduler-job custody reconciliation only. A bound row does not establish scientific "
    "validity, artifact completeness, benchmark eligibility, or durable custody, and scheduler "
    "completion is never result validity."
)
TERMINAL_STATES = frozenset({"completed", "failed", "timeout", "cancelled"})
CLASSIFICATIONS = (
    "owned_active",
    "owned_terminal_unharvested",
    "owned_harvested",
    "duplicate_candidate",
    "orphan_unknown",
    "stale_input",
    "missing_output_owner",
    "projection_unavailable",
)
ACTION_REQUIRED = frozenset(CLASSIFICATIONS) - {"owned_active", "owned_harvested"}
NEXT_ACTIONS = {
    "owned_active": "monitor_active_job",
    "owned_terminal_unharvested": "harvest_terminal_output",
    "owned_harvested": "no_action",
    "duplicate_candidate": "resolve_duplicate_or_conflicting_binding",
    "orphan_unknown": "attach_exact_public_owner",
    "stale_input": "refresh_immutable_input_packet",
    "missing_output_owner": "assign_artifact_owner",
    "projection_unavailable": "repair_sanitized_projection",
}
HARVEST_STATES = frozenset(
    "pending harvested harvest_blocked not_applicable unavailable not_observed".split()
)
TRANSFER_STATES = frozenset("pending transferred not_applicable unavailable not_observed".split())
VOLATILE_FIELDS = frozenset(
    "generated_at snapshot_at observed_at checked_at reported_at collected_at exported_at".split()
)
_UNTRUSTED_CODES = frozenset({"under_redacted_value", "forbidden_field_name"})


class ProjectionError(ValueError):
    """Malformed projection input that cannot be reconciled."""


class PublicIndex(NamedTuple):
    """Indexed public snapshot: task rows, owner states, and ambiguous task ids."""

    tasks: Mapping[str, Mapping[str, Any]]
    owner_states: Mapping[tuple[str, int], str]
    duplicates: frozenset[str]


def _issue(code: str, location: str, message: str) -> dict[str, str]:
    return {"code": code, "location": location, "message": message}


def _sorted_issues(issues: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    unique = {(item["code"], item["location"], item["message"]): dict(item) for item in issues}
    return [unique[key] for key in sorted(unique)]


def _normalize_volatile(value: Any, found: set[str]) -> Any:
    """Replace documented volatile fields with ``normalized`` before classification."""
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in VOLATILE_FIELDS:
                found.add(name)
                normalized[name] = "normalized"
            else:
                normalized[name] = _normalize_volatile(item, found)
        return normalized
    if isinstance(value, list):
        return [_normalize_volatile(item, found) for item in value]
    return value


def _scan_private(
    value: Any, location: str, issues: list[dict[str, str]], rejected: frozenset[str]
) -> None:
    """Fail closed on forbidden field names and values the shared scanner flags as private."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{location}/{key}"
            if str(key).lower() in rejected:
                issues.append(_issue("forbidden_field_name", child, "private topology field name"))
            _scan_private(item, child, issues, rejected)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_private(item, f"{location}[{index}]", issues, rejected)
    elif isinstance(value, str) and (reason := private_content_reason(value)):
        issues.append(_issue("under_redacted_value", location, reason))


def _rejected_names() -> frozenset[str]:
    contracts = load_contracts()
    return next(iter(contracts.values())).rejected_names if contracts else frozenset()


def _explicit(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in EXPLICIT_TOKENS


def _usable_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() and not _explicit(value) else None


def _usable_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _enum(value: Any, options: frozenset[str]) -> str:
    text = _usable_text(value)
    return text if text in options else "not_observed"


def _owner_ref(value: Any) -> tuple[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    kind, number = value.get("kind"), _usable_int(value.get("number"))
    if kind not in ("issue", "pull_request"):
        return None
    return (kind, number) if number is not None and number > 0 else None


def _owner_url(kind: str, number: int) -> str:
    return f"{PUBLIC_URL_PREFIX}/{'issues' if kind == 'issue' else 'pull'}/{number}"


def _owner_key(owner: Any) -> tuple[str, int] | None:
    return (owner["kind"], owner["number"]) if isinstance(owner, Mapping) else None


def _index_public(  # noqa: C901 - one fail-closed public snapshot pass
    public: Any, issues: list[dict[str, str]], rejected: frozenset[str]
) -> PublicIndex:
    """Index the public snapshot by exact task alias and public owner reference."""
    if not isinstance(public, Mapping):
        issues.append(
            _issue("missing_public_metadata", "/public_metadata", "public snapshot required")
        )
        return PublicIndex({}, {}, frozenset())
    owner_states: dict[tuple[str, int], str] = {}
    for kind, key in (("issue", "issues"), ("pull_request", "pull_requests")):
        entries = public.get(key)
        if entries is not None and not isinstance(entries, list):
            issues.append(_issue("invalid_public_owner_list", f"/{key}", "list required"))
        for index, entry in enumerate(entries if isinstance(entries, list) else []):
            number = _usable_int(entry.get("number")) if isinstance(entry, Mapping) else None
            if number is None or number <= 0:
                issues.append(
                    _issue("invalid_public_owner", f"/{key}[{index}]", "kind/number required")
                )
                continue
            state = _usable_text(entry.get("state")) if isinstance(entry, Mapping) else None
            owner_states[(kind, number)] = state or "not_observed"
    tasks: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    entries = public.get("tasks")
    if entries is not None and not isinstance(entries, list):
        issues.append(_issue("invalid_public_tasks", "/tasks", "list required"))
    for index, entry in enumerate(entries if isinstance(entries, list) else []):
        if not isinstance(entry, Mapping):
            issues.append(_issue("invalid_public_task", f"/tasks[{index}]", "mapping required"))
            continue
        task_id = _usable_text(entry.get("task_id"))
        if task_id is None:
            issues.append(_issue("invalid_task_identity", f"/tasks[{index}]/task_id", "required"))
            continue
        if task_id in tasks:
            duplicates.add(task_id)
        tasks[task_id] = entry
    _scan_private(public, "/public_metadata", issues, rejected)
    return PublicIndex(tasks, owner_states, frozenset(duplicates))


def _next_action(classification: str, row: Mapping[str, Any]) -> str:
    if classification == "owned_active":
        owner = row.get("public_owner")
        state = owner.get("state") if isinstance(owner, Mapping) else None
        if state == "closed":
            return "reconcile_closed_public_owner_with_active_job"
    return NEXT_ACTIONS[classification]


def _row(
    base: Mapping[str, Any], classification: str, findings: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    row = dict(base)
    row["classification"] = classification
    row["next_action"] = _next_action(classification, row)
    row["scientific_status"] = "not_evaluated"
    row["findings"] = _sorted_issues(findings)
    return row


def _classify_job(  # noqa: C901, PLR0912, PLR0915 - one deterministic fail-closed pass
    raw: Any,
    index: PublicIndex,
    *,
    position: int,
    projection_ok: bool,
    rejected: frozenset[str],
) -> dict[str, Any]:
    """Classify one job row without inferring ownership from mutable names."""
    base: dict[str, Any] = {
        "job_alias": None,
        "array_index": None,
        "parent_job_alias": None,
        "scheduler_state": "unavailable",
        "state_history": [],
        "task_id": None,
        "task_class": "unavailable",
        "public_owner": None,
        "packet_digest": None,
        "manifest_sha256": None,
        "expected_row_count": None,
        "output_root_identity": None,
        "artifact_owner": None,
        "harvest_state": "not_observed",
        "transfer_state": "not_observed",
        "durability": "not_applicable",
    }
    findings: list[Mapping[str, str]] = []
    job: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    receipt = job.get("receipt")
    alias = _usable_text(receipt.get("job_alias")) if isinstance(receipt, Mapping) else None
    location = f"jobs/{alias}" if alias else f"jobs/row-{position}"
    _scan_private(job, location, findings, rejected)
    if any(item["code"] in _UNTRUSTED_CODES for item in findings):
        return _row(base, "projection_unavailable", findings)
    if not isinstance(receipt, Mapping):
        findings.append(_issue("missing_receipt", f"{location}/receipt", "public receipt required"))
        return _row(base, "projection_unavailable", findings)
    receipt_errors = validate_receipt(receipt)
    if receipt_errors:
        findings.append(_issue("invalid_receipt", f"{location}/receipt", receipt_errors[0].code))
        return _row(base, "projection_unavailable", findings)
    base["job_alias"] = receipt["job_alias"]
    state = normalize_job_state(receipt["scheduler_state"])
    base["scheduler_state"] = state.lower()
    base["state_history"] = [str(item) for item in receipt.get("state_history", [])]
    base["array_index"] = _usable_int(job.get("array_index"))
    base["parent_job_alias"] = _usable_text(job.get("parent_job_alias"))
    task_id = receipt["campaign_id"]
    base["task_id"] = task_id
    packet = _usable_text(receipt["submission"].get("source_config_command_digest"))
    root = _usable_text(receipt["artifacts"].get("result_root_identity"))
    base["packet_digest"], base["output_root_identity"] = packet, root
    if packet is None:
        findings.append(
            _issue("packet_identity_unavailable", f"{location}/packet_digest", "exact digest")
        )
    if root is None:
        findings.append(
            _issue("output_root_unavailable", f"{location}/output_root_identity", "exact root")
        )
    if not projection_ok:
        findings.append(
            _issue("projection_status_unavailable", location, "projection not complete")
        )
    if packet is None or root is None or not projection_ok:
        return _row(base, "projection_unavailable", findings)
    task = index.tasks.get(task_id)
    if task is None or task_id in index.duplicates:
        findings.append(_issue("public_task_unmatched", location, "no exact public task binding"))
        return _row(base, "orphan_unknown", findings)
    job_ref, task_ref = _owner_ref(job.get("issue_ref")), _owner_ref(task.get("public_ref"))
    if job_ref and task_ref and job_ref != task_ref:
        findings.append(
            _issue("conflicting_public_owner", location, "job and task owner bindings differ")
        )
    ref = job_ref or task_ref
    if ref is None:
        findings.append(_issue("public_owner_unbound", location, "no exact public owner mapping"))
        return _row(base, "orphan_unknown", findings)
    owner_state = index.owner_states.get(ref, "not_observed")
    kind, number = ref
    base["public_owner"] = {
        "kind": kind,
        "number": number,
        "state": owner_state,
        "url": _owner_url(kind, number),
    }
    if owner_state == "not_observed":
        findings.append(
            _issue("public_owner_state_unavailable", location, "owner state not observed")
        )
    manifest = task.get("launch_manifest")
    manifest_packet = manifest_sha = None
    expected = None
    if isinstance(manifest, Mapping):
        manifest_sha = _usable_text(manifest.get("manifest_sha256"))
        manifest_packet = _usable_text(manifest.get("packet_digest"))
        expected = _usable_int(manifest.get("expected_episode_cells"))
        if manifest.get("schema_version") != LAUNCH_MANIFEST_SCHEMA:
            findings.append(
                _issue(
                    "launch_manifest_schema_mismatch",
                    f"{location}/launch_manifest",
                    "unsupported schema",
                )
            )
            manifest_packet = None
    else:
        findings.append(
            _issue(
                "launch_manifest_unavailable",
                f"{location}/launch_manifest",
                "immutable packet binding required",
            )
        )
    base["manifest_sha256"], base["expected_row_count"] = manifest_sha, expected
    if manifest_packet is None:
        findings.append(
            _issue("packet_identity_unavailable", f"{location}/launch_manifest", "exact digest")
        )
        return _row(base, "projection_unavailable", findings)
    if manifest_packet != packet:
        findings.append(
            _issue(
                "stale_packet_digest",
                f"{location}/packet_digest",
                "packet digest differs from public manifest",
            )
        )
        return _row(base, "stale_input", findings)
    owner = _usable_text(task.get("owner"))
    task_root = _usable_text(task.get("artifact_root_identity"))
    base["artifact_owner"] = owner
    base["task_class"] = _usable_text(task.get("task_class")) or "unavailable"
    base["harvest_state"] = _enum(task.get("harvest_state"), HARVEST_STATES)
    base["transfer_state"] = _enum(task.get("transfer_state"), TRANSFER_STATES)
    if owner is None or task_root is None or task_root != root:
        findings.append(
            _issue("missing_artifact_owner", location, "artifact owner/root binding unavailable")
        )
        return _row(base, "missing_output_owner", findings)
    terminal = state in {item.upper() for item in TERMINAL_STATES}
    harvested = terminal and base["harvest_state"] == "harvested"
    durable_pointer = "recorded" if task.get("durable_pointer") is True else None
    base["durability"] = classify_durable_status(
        "success" if harvested else "not_success", durable_pointer
    )
    if not terminal:
        return _row(base, "owned_active", findings)
    complete = harvested and base["transfer_state"] == "transferred"
    return _row(base, "owned_harvested" if complete else "owned_terminal_unharvested", findings)


def _apply_duplicate_findings(rows: list[dict[str, Any]]) -> None:  # noqa: C901 - one grouping pass
    """Mark equivalent active duplicates and conflicting or repeated job identities."""
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        if (
            row["classification"] == "owned_active"
            and row["task_id"]
            and row["packet_digest"]
            and row["parent_job_alias"] is None
        ):
            key = (row["task_id"], row["packet_digest"], row["array_index"])
            groups.setdefault(key, []).append(row)
    for group in groups.values():
        if len(group) < 2:
            continue
        for row in group:
            row["classification"] = "duplicate_candidate"
            row["next_action"] = NEXT_ACTIONS["duplicate_candidate"]
            row["findings"] = _sorted_issues(
                [
                    *row["findings"],
                    _issue(
                        "duplicate_active_jobs",
                        f"jobs/{row['job_alias']}",
                        "equivalent active jobs share task, packet, and array index",
                    ),
                ]
            )
    aliases: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["job_alias"]:
            aliases.setdefault(row["job_alias"], []).append(row)
    for alias, group in aliases.items():
        if len(group) < 2:
            continue
        refs = {_owner_key(row["public_owner"]) for row in group}
        for row in group:
            extra = [
                _issue("duplicate_job_identity", f"jobs/{alias}", "alias appears more than once")
            ]
            if len(refs) > 1:
                extra.append(
                    _issue("conflicting_public_owner", f"jobs/{alias}", "owner bindings differ")
                )
            if row["classification"].startswith("owned_"):
                row["classification"] = "duplicate_candidate"
                row["next_action"] = NEXT_ACTIONS["duplicate_candidate"]
            row["findings"] = _sorted_issues([*row["findings"], *extra])


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, int, str, str, str]:
    array_index = row["array_index"] if isinstance(row["array_index"], int) else -1
    return (
        row["job_alias"] or "",
        array_index,
        row["task_id"] or "",
        row["packet_digest"] or "",
        row["classification"],
    )


def reconcile_projection(
    projection: Mapping[str, Any], *, public_override: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Reconcile one sanitized projection and return the deterministic custody report."""
    normalized_fields: set[str] = set()
    normalized = _normalize_volatile(projection, normalized_fields)
    if normalized.get("schema") != PROJECTION_SCHEMA:
        raise ProjectionError(f"projection schema must be {PROJECTION_SCHEMA}")
    if not isinstance(normalized.get("jobs"), list):
        raise ProjectionError("projection must declare a jobs list")
    findings: list[dict[str, str]] = []
    rejected = _rejected_names()
    envelope = {
        key: value for key, value in normalized.items() if key not in ("jobs", "public_metadata")
    }
    _scan_private(envelope, "/", findings, rejected)
    projection_status = normalized.get("projection_status")
    projection_ok = projection_status == "complete"
    if not projection_ok:
        findings.append(
            _issue(
                "projection_status_unavailable",
                "/projection_status",
                f"status={projection_status if isinstance(projection_status, str) else 'missing'}",
            )
        )
    public_source = (
        public_override if public_override is not None else normalized.get("public_metadata")
    )
    public = _normalize_volatile(public_source, normalized_fields)
    index = _index_public(public, findings, rejected)
    if findings:
        projection_ok = False
    rows = [
        _classify_job(
            job,
            index,
            position=position,
            projection_ok=projection_ok,
            rejected=rejected,
        )
        for position, job in enumerate(normalized["jobs"])
    ]
    _apply_duplicate_findings(rows)
    rows.sort(key=_row_sort_key)
    counts = dict.fromkeys(CLASSIFICATIONS, 0)
    for row in rows:
        counts[row["classification"]] += 1
    return {
        "schema": REPORT_SCHEMA,
        "check_only": True,
        "generated_at": "normalized",
        "projection_schema": PROJECTION_SCHEMA,
        "projection_status": (
            projection_status if isinstance(projection_status, str) else "unavailable"
        ),
        "public_metadata_source": (
            "override"
            if public_override is not None
            else ("embedded" if isinstance(public_source, Mapping) else "unavailable")
        ),
        "volatile_fields_normalized": sorted(normalized_fields),
        "row_count": len(rows),
        "classification_counts": counts,
        "action_required_count": sum(counts[name] for name in ACTION_REQUIRED),
        "rows": rows,
        "findings": _sorted_issues(findings),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def render_report_json(report: Mapping[str, Any]) -> str:
    """Return byte-stable sorted-key reconciliation JSON with a trailing newline."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Return the concise deterministic Markdown reconciliation summary, free of private values."""
    lines = [
        "# Scheduler Job Reconciliation",
        "",
        f"- rows: {report['row_count']}",
        f"- action required: {report['action_required_count']}",
        f"- projection status: `{report['projection_status']}`",
        "",
        "| job | scheduler | public owner | packet digest | classification | next action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        owner = row["public_owner"]
        owner_text = (
            f"{owner['kind']} #{owner['number']} ({owner['state']})" if owner else "unbound"
        )
        packet = row["packet_digest"] or "unavailable"
        lines.append(
            f"| {row['job_alias'] or 'unidentified'} | {row['scheduler_state']} | {owner_text} | "
            f"`{packet}` | {row['classification']} | {row['next_action']} |"
        )
    if report["findings"]:
        lines += ["", "## Findings", ""]
        lines += [
            f"- `{item['code']}` at `{item['location']}`: {item['message']}"
            for item in report["findings"]
        ]
    lines += ["", report["claim_boundary"], ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the reconciliation argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="validate and print only; never mutates state"
    )
    parser.add_argument(
        "--projection", type=Path, required=True, help="sanitized scheduler inventory projection"
    )
    parser.add_argument(
        "--public", type=Path, help="optional public issue/PR and artifact metadata snapshot"
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProjectionError(f"cannot read {path} ({type(exc).__name__})") from exc
    if not isinstance(payload, Mapping):
        raise ProjectionError(f"{path} must contain a JSON object")
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check-only reconciliation and return a shell-friendly exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("only --check is supported; this tool never mutates state")
    try:
        projection = _load_json(args.projection)
        public_override = _load_json(args.public) if args.public else None
        report = reconcile_projection(projection, public_override=public_override)
    except ProjectionError as exc:
        sys.stderr.write(f"FAIL malformed_input: {sanitize_text(str(exc))}\n")
        return 2
    render = render_report_json if args.format == "json" else render_report_markdown
    sys.stdout.write(render(report))
    return 0 if report["action_required_count"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
