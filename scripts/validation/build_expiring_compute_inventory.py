#!/usr/bin/env python3
"""Deterministic read-only inventory of public work tied to expiring compute access (#8822).

Rows carry the public owner, dependence class (``resource_dependent``, ``locally_runnable_later``,
or ``unknown``), capability, prerequisite/approval state, exact source/config/checkpoint/seed
identity status, job/harvest/transfer state, access-loss consequence, and one recommended action
(``submit_now``, ``prestage_now``, ``harvest_now``, ``transfer_now``, ``blocked_do_not_submit``,
``safe_to_defer``, or ``unknown``). Dependence is never inferred from titles, labels, or keyword
mentions: rows need cited structured evidence, and missing, contradictory, unsanitized, or
owner-disagreeing facts stay explicit ``unknown``.

The tool only reads sanitized JSON documents; it never contacts GitHub, the scheduler, or
credentials and never submits, cancels, labels, or mutates scientific state. A private companion
inventory can be bound by SHA-256 without publishing its bytes. Campaign deadline feasibility
stays owned by ``scripts/validation/check_expiring_resource_feasibility.py``.

CLI: ``--inputs PATH [PATH ...] [--case NAME] --check [--format json|text] [--as-of ISO]
[--private-inventory PATH --private-owner NAME]``. Exit codes: 0 report produced, 2 malformed or
unreadable input. Operational note: ``docs/dev/slurm_submission.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.tools.validate_receipt_projection import ALLOWED_PUBLIC_URL_RE, private_content_reason

# Dense layout honors issue #8822's hard 800 net-line cap, as the sibling compute-window capacity
# planner (#8829) did; ruff lint stays active, only the formatter is scoped out.
# fmt: off
INPUT_SCHEMA = "robot_sf.expiring_compute_inventory_input.v1"
CASES_SCHEMA = "robot_sf.expiring_compute_inventory_cases.v1"
REPORT_SCHEMA = "robot_sf.expiring_compute_inventory.v1"
CLAIM_BOUNDARY = ("Operational inventory only: dependence classes and recommended actions are "
                  "read-only projections of sanitized structured evidence. They are not scheduler "
                  "admission, compute or domain approval, scientific priority, benchmark evidence, "
                  "or a promise that any action is possible; scheduler success is never "
                  "scientific success.")
RECOMMENDED_ACTIONS = ("harvest_now transfer_now submit_now prestage_now blocked_do_not_submit "
                       "safe_to_defer unknown".split())
DEPENDENCE_CLASSES = ("resource_dependent", "locally_runnable_later", "unknown")
ACTION_RANK = {name: rank for rank, name in enumerate(RECOMMENDED_ACTIONS)}
RESOURCE_CLASSES = "local slurm_cpu slurm_gpu carla_host multi_host unknown".split()
GATE_STATES = "met not_required unmet blocked unknown".split()
ARTIFACT_LOCALITIES = "local cluster_only unknown".split()
JOB_STATES = "not_submitted pending running completed failed cancelled unknown".split()
HARVEST_STATES = "not_applicable pending partial harvested unknown".split()
TRANSFER_STATES = "not_applicable not_started pending transferred preserved unknown".split()
CONSEQUENCES = "none recompute_required manual_recapture_required unrecoverable unknown".split()
URGENCIES = "expiring deadline_bound routine unknown".split()
PRIORITIES = "p0 p1 p2 p3".split()
IDENTITY_KEYS = "source config checkpoint seed_set".split()
BLOCKED_STATES = frozenset(("unmet", "blocked"))
HARVEST_NEEDED = frozenset(("pending", "partial"))
SECURED_TRANSFER = frozenset(("transferred", "preserved", "not_applicable"))
TERMINAL_JOB_STATES = frozenset(("completed", "failed", "cancelled"))
SUBMISSION_ACTIONS = frozenset(("harvest_now", "transfer_now", "submit_now", "prestage_now"))
URGENCY_RANK = {"expiring": 0, "deadline_bound": 1, "routine": 2, "unknown": 3}
PRIORITY_RANK = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
IDENTITY_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_ENVELOPE_KEYS = frozenset(("schema", "report_id", "source_owner", "generated_at", "as_of", "items"))
_ITEM_KEYS = frozenset(
    "work_id public_owner evidence_refs public_metadata dependence prerequisite_state "
    "approval_state expected_compute_units expected_storage_bytes command_owner artifact_output "
    "artifact_locality job access_loss_consequence urgency access_deadline_utc identities "
    "identity_required priority status_disclosure_risk exclusion".split())
_DEPENDENCE_KEYS = frozenset(
    "basis resource_class required_capabilities local_hardware_feasible projections".split())
_FATAL_CODES = frozenset(
    "invalid_item invalid_field unknown_field invalid_work_id missing_public_owner "
    "unsanitized_input missing_as_of missing_items".split())


def _issue(code: str, location: str, message: str) -> dict[str, str]:
    return {"code": code, "location": location, "message": message}


def _slug(value: Any) -> str | None:
    return value if isinstance(value, str) and SLUG_RE.fullmatch(value) else None


def _enum(value: Any, allowed: Sequence[str]) -> str | None:
    return value if isinstance(value, str) and value in allowed else None


def _public_ref(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if ALLOWED_PUBLIC_URL_RE.fullmatch(value) or SLUG_RE.fullmatch(value) else None


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def _render(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z") if moment else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scan_private(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_scan_private(item) for item in value.values())
    if isinstance(value, list):
        return any(_scan_private(item) for item in value)
    return isinstance(value, str) and private_content_reason(value) != ""


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value if value >= 0 else None


@dataclass(frozen=True, slots=True)
class InputDocument:
    """One loaded input document: provenance, declared as_of, raw items, and issues."""
    index: int
    report_id: str
    source_owner: str
    generated_at: datetime | None
    as_of: datetime | None
    items: tuple[Any, ...]
    issues: tuple[dict[str, str], ...]


def _load_document(payload: Mapping[str, Any], index: int) -> InputDocument:
    if payload.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"input schema must be {INPUT_SCHEMA}")
    location = f"/inputs[{index}]"
    issues = [_issue("unknown_field", f"{location}/{key}", "not in the schema")
              for key in payload if key not in _ENVELOPE_KEYS]
    report_id, source_owner = _slug(payload.get("report_id")), _slug(payload.get("source_owner"))
    for name, value in (("report_id", report_id), ("source_owner", source_owner)):
        if value is None:
            issues.append(_issue("invalid_field", f"{location}/{name}", "invalid slug"))
    generated_at = _timestamp(payload.get("generated_at"))
    if payload.get("generated_at") is not None and generated_at is None:
        issues.append(_issue("invalid_field", f"{location}/generated_at", "invalid timestamp"))
    as_of = _timestamp(payload.get("as_of"))
    if as_of is None:
        issues.append(_issue("missing_as_of", f"{location}/as_of", "timestamp required"))
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        issues.append(_issue("missing_items", f"{location}/items", "non-empty item list required"))
        raw_items = []
    return InputDocument(index, report_id or "unavailable", source_owner or "unavailable",
                         generated_at, as_of, tuple(raw_items), tuple(issues))


def _keyword_mentions(item: Mapping[str, Any]) -> list[str]:
    metadata = item.get("public_metadata")
    labels = metadata.get("labels") if isinstance(metadata, Mapping) else None
    return sorted(str(label) for label in labels) if isinstance(labels, list) else []


def _enum_field(node: Mapping[str, Any], key: str, allowed: Sequence[str],
                problems: list[str]) -> str:
    if key not in node or node.get(key) is None:
        return "unknown"
    value = _enum(node[key], allowed)
    if value is None:
        problems.append("invalid_field")
    return value or "unknown"


def _projections(dep: Mapping[str, Any]) -> tuple[set[str], bool]:
    raw = dep.get("projections")
    if raw is None:
        return set(), False
    if not isinstance(raw, list):
        return set(), True
    values: set[str] = set()
    malformed = False
    for entry in raw:
        owner = _slug(entry.get("owner")) if isinstance(entry, Mapping) else None
        parsed = _enum(entry.get("dependence"), DEPENDENCE_CLASSES) if isinstance(entry, Mapping) else None
        if owner is None or parsed is None:
            malformed = True
        else:
            values.add(parsed)
    return values, malformed


def _dependence(  # noqa: C901, PLR0912 - one explicit fail-closed evidence decision
        item: Mapping[str, Any]) -> tuple[str, dict[str, Any], list[str], list[str]]:
    problems: list[str] = []
    evidence: dict[str, Any] = {"resource_class": "unknown", "capabilities": []}
    dep = item.get("dependence")
    if not isinstance(dep, Mapping):
        if _keyword_mentions(item):
            problems.append("keyword_mentions_ignored")
        return "unknown", evidence, [], [*problems, "insufficient_structured_evidence"]
    problems.extend("invalid_field" for key in dep if key not in _DEPENDENCE_KEYS)
    basis = _slug(dep.get("basis"))
    if basis is None:
        problems.append("missing_dependence_basis")
    resource_class = _enum(dep.get("resource_class"), RESOURCE_CLASSES)
    if dep.get("resource_class") is not None and resource_class is None:
        problems.append("invalid_resource_class")
    raw_caps = dep.get("required_capabilities")
    if raw_caps is not None and not isinstance(raw_caps, list):
        problems.append("invalid_field")
        raw_caps = []
    valid_caps = [cap for cap in raw_caps or [] if _slug(cap)]
    if raw_caps is not None and len(valid_caps) != len(raw_caps):
        problems.append("invalid_field")
    capabilities = sorted({str(cap) for cap in valid_caps})
    local = dep.get("local_hardware_feasible")
    if not isinstance(local, bool):
        problems.append("missing_local_feasibility_evidence")
    values, malformed = _projections(dep)
    if malformed:
        problems.append("invalid_field")
    established = sorted(values - {"unknown"})
    if len(established) > 1:
        return "unknown", evidence, [], [*problems, "owner_disagreement"]
    citations = ["dependence.basis"] if basis is not None else []
    if resource_class is not None:
        citations.append("dependence.resource_class")
    if capabilities:
        citations.append("dependence.required_capabilities")
    if isinstance(local, bool):
        citations.append("dependence.local_hardware_feasible")
    evidence = {"resource_class": resource_class or "unknown", "capabilities": capabilities}
    if basis is None or not isinstance(local, bool):
        return "unknown", evidence, citations, [*problems, "insufficient_structured_evidence"]
    if local:
        dependence = "locally_runnable_later"
    elif resource_class == "local":
        return "unknown", evidence, citations, [*problems, "dependence_evidence_conflict"]
    elif resource_class in (None, "unknown"):
        return "unknown", evidence, citations, [*problems, "missing_resource_class"]
    elif not capabilities:
        return "unknown", evidence, citations, [*problems, "missing_required_capabilities"]
    else:
        dependence = "resource_dependent"
    if established and established[0] != dependence:
        return "unknown", evidence, citations, [*problems, "dependence_projection_conflict"]
    return dependence, evidence, citations, problems


def _recommend(dependence: str, states: Mapping[str, str], *,
               elapsed: bool) -> tuple[str, list[str]]:
    reasons = ["access_window_elapsed"] if elapsed else []
    if dependence == "locally_runnable_later":
        return "safe_to_defer", reasons
    blocked = states["prerequisite_state"] in BLOCKED_STATES or states["approval_state"] in BLOCKED_STATES
    if blocked:
        return "blocked_do_not_submit", [*reasons, "blocked_prerequisite"]
    if states["harvest_state"] in HARVEST_NEEDED and (
            states["job_state"] == "running" or states["job_state"] in TERMINAL_JOB_STATES):
        return "harvest_now", [*reasons, "harvest_required"]
    if (states["artifact_locality"] == "cluster_only" and states["harvest_state"] == "harvested"
            and states["transfer_state"] in {"pending", "not_started"}):
        return "transfer_now", [*reasons, "transfer_required"]
    if (states["job_state"] == "completed" and states["harvest_state"] in {"harvested", "not_applicable"}
            and states["transfer_state"] in SECURED_TRANSFER):
        return "safe_to_defer", [*reasons, "outputs_secured"]
    if states["job_state"] in {"not_submitted", "pending"}:
        if elapsed:
            return "blocked_do_not_submit", reasons
        if states["urgency"] in {"expiring", "deadline_bound"}:
            return "submit_now", [*reasons, "window_open"]
        return "prestage_now", [*reasons, "routine_urgency"]
    return "prestage_now", [*reasons, "prestage_remaining_work"]


def _item_states(item: Mapping[str, Any], problems: list[str]) -> dict[str, str]:
    states = {"prerequisite_state": _enum_field(item, "prerequisite_state", GATE_STATES, problems),
              "approval_state": _enum_field(item, "approval_state", GATE_STATES, problems),
              "artifact_locality": _enum_field(item, "artifact_locality", ARTIFACT_LOCALITIES, problems),
              "access_loss_consequence": _enum_field(item, "access_loss_consequence", CONSEQUENCES, problems),
              "urgency": _enum_field(item, "urgency", URGENCIES, problems)}
    raw_job = item.get("job")
    if raw_job is not None and not isinstance(raw_job, Mapping):
        problems.append("invalid_field")
    job = raw_job if isinstance(raw_job, Mapping) else {}
    states["job_state"] = _enum_field(job, "state", JOB_STATES, problems)
    states["harvest_state"] = _enum_field(job, "harvest_state", HARVEST_STATES, problems)
    states["transfer_state"] = _enum_field(job, "transfer_state", TRANSFER_STATES, problems)
    return states


def _identity_problems(item: Mapping[str, Any], problems: list[str]) -> bool:
    identities = item.get("identities") if isinstance(item.get("identities"), Mapping) else {}
    raw_required = item.get("identity_required", [])
    if raw_required is not None and not isinstance(raw_required, list):
        problems.append("invalid_field")
        raw_required = []
    missing = False
    for key in raw_required or []:
        if key not in IDENTITY_KEYS:
            problems.append("invalid_field")
            continue
        value = identities.get(key)
        if not (isinstance(value, str) and IDENTITY_RE.fullmatch(value)):
            missing = True
            problems.append(f"missing_identity_{key}")
    if missing:
        problems.append("missing_exact_identity")
    return missing


def _classify_item(  # noqa: C901, PLR0912, PLR0915 - one fail-closed decision per item
        item: Any, document: InputDocument, as_of: datetime, *,
        duplicates: frozenset[str]) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        item = {}
        document_issues = [*document.issues, _issue("invalid_item", "/items", "item must be a map")]
    else:
        document_issues = list(document.issues)
    problems = [issue["code"] for issue in document_issues]
    if _scan_private(item):
        problems.append("unsanitized_input")
    problems.extend("unknown_field" for key in item if key not in _ITEM_KEYS)
    work_id = _slug(item.get("work_id"))
    if work_id is None:
        problems.append("invalid_work_id")
    owner = _public_ref(item.get("public_owner"))
    if owner is None:
        problems.append("missing_public_owner")
    raw_refs = item.get("evidence_refs", [])
    if not isinstance(raw_refs, list):
        problems.append("invalid_field")
        raw_refs = []
    evidence_refs = [ref for raw in raw_refs if (ref := _public_ref(raw)) is not None]
    if len(evidence_refs) != len(raw_refs):
        problems.append("invalid_field")
    states = _item_states(item, problems)
    expected_compute = _number(item.get("expected_compute_units"))
    if item.get("expected_compute_units") is not None and expected_compute is None:
        problems.append("invalid_field")
    expected_storage = _number(item.get("expected_storage_bytes"))
    if item.get("expected_storage_bytes") is not None and expected_storage is None:
        problems.append("invalid_field")
    missing_identity = _identity_problems(item, problems)
    deadline = _timestamp(item.get("access_deadline_utc"))
    if item.get("access_deadline_utc") is not None and deadline is None:
        problems.append("invalid_field")
    elapsed = deadline is not None and deadline <= as_of
    if elapsed:
        problems.append("access_window_elapsed")
    raw_exclusion = item.get("exclusion")
    exclusion_valid = False
    exclusion_reason = exclusion_owner = None
    if raw_exclusion is not None:
        if isinstance(raw_exclusion, Mapping):
            exclusion_reason = _slug(raw_exclusion.get("reason"))
            exclusion_owner = _public_ref(raw_exclusion.get("owner"))
            exclusion_valid = exclusion_reason is not None and exclusion_owner is not None
        if not exclusion_valid:
            problems.append("invalid_field")
    disclosure = item.get("status_disclosure_risk")
    if disclosure is not None and not isinstance(disclosure, bool):
        problems.append("invalid_field")
        disclosure = None
    priority = item.get("priority")
    if priority is not None and _enum(priority, PRIORITIES) is None:
        problems.append("invalid_field")
        priority = None
    dependence, evidence, recommended = "unknown", {"resource_class": "unknown"}, "unknown"
    evidence_fields: list[str] = []
    if exclusion_valid:
        problems.append("explicitly_excluded")
    elif disclosure is True:
        problems.append("secret_disclosure_risk")
    elif work_id is not None and work_id in duplicates:
        problems.append("duplicate_work_item")
    elif missing_identity or (_FATAL_CODES & set(problems)):
        pass
    else:
        dependence, evidence, evidence_fields, dep_problems = _dependence(item)
        problems.extend(dep_problems)
        if dependence != "unknown":
            recommended, action_problems = _recommend(dependence, states, elapsed=elapsed)
            problems.extend(action_problems)
    return {"work_id": work_id or "unidentified", "public_owner": owner,
            "dependence": dependence, "recommended_action": recommended,
            "excluded": exclusion_valid, "exclusion_reason": exclusion_reason,
            "exclusion_owner": exclusion_owner, "resource_class": evidence.get("resource_class", "unknown"),
            "required_capabilities": evidence.get("capabilities", []),
            "prerequisite_state": states["prerequisite_state"], "approval_state": states["approval_state"],
            "expected_compute_units": expected_compute, "expected_storage_bytes": expected_storage,
            "command_owner": _slug(item.get("command_owner")), "artifact_output": _slug(item.get("artifact_output")),
            "artifact_locality": states["artifact_locality"], "job_state": states["job_state"],
            "harvest_state": states["harvest_state"], "transfer_state": states["transfer_state"],
            "access_loss_consequence": states["access_loss_consequence"], "urgency": states["urgency"],
            "priority": priority, "access_deadline_utc": _render(deadline), "evidence_refs": evidence_refs,
            "evidence_fields": sorted(evidence_fields), "keyword_mentions": _keyword_mentions(item),
            "reason_codes": sorted(set(problems))}


def build_inventory(documents: Sequence[Mapping[str, Any]], *, as_of: datetime | None = None,
                    private_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build one deterministic sanitized inventory report from input documents."""
    if not documents:
        raise ValueError("at least one inventory input document is required")
    loaded = [_load_document(document, index) for index, document in enumerate(documents)]
    issues = [issue for document in loaded for issue in document.issues]
    stamps = [document.as_of for document in loaded if document.as_of is not None]
    reference = as_of or (max(stamps) if stamps else None)
    if reference is None:
        raise ValueError("an explicit --as-of or a document as_of is required")
    ids = Counter(work_id for document in loaded for item in document.items
                  if isinstance(item, Mapping) and (work_id := _slug(item.get("work_id"))))
    duplicates = frozenset(name for name, count in ids.items() if count > 1)
    rows = sorted((_classify_item(item, document, reference, duplicates=duplicates)
                   for document in loaded for item in document.items),
                  key=lambda row: (ACTION_RANK.get(str(row["recommended_action"]), 9),
                                   URGENCY_RANK.get(str(row["urgency"]), 9),
                                   PRIORITY_RANK.get(str(row["priority"]), 9),
                                   str(row["work_id"]), str(row["public_owner"])))
    counts = Counter(row["dependence"] for row in rows)
    actions = Counter(row["recommended_action"] for row in rows)
    unknown_rows = sum(1 for row in rows if row["dependence"] == "unknown" and not row["excluded"])
    records = [{"index": document.index, "report_id": document.report_id,
                "source_owner": document.source_owner,
                "generated_at": _render(document.generated_at), "as_of": _render(document.as_of),
                "item_count": len(document.items),
                "status": "rejected" if document.issues else "accepted",
                "issue_codes": sorted({issue["code"] for issue in document.issues})}
               for document in loaded]
    return {"schema": REPORT_SCHEMA, "check_only": True, "claim_boundary": CLAIM_BOUNDARY,
            "status": "incomplete" if unknown_rows or issues else "complete",
            "ok": not unknown_rows and not issues, "as_of_utc": _render(reference),
            "inputs": records,
            "private_binding": dict(private_binding) if private_binding else None,
            "summary": {"row_count": len(rows), "excluded_count": sum(1 for row in rows if row["excluded"]),
                        "unknown_count": counts.get("unknown", 0),
                        "resource_dependent_count": counts.get("resource_dependent", 0),
                        "locally_runnable_later_count": counts.get("locally_runnable_later", 0),
                        "action_counts": {name: actions.get(name, 0) for name in RECOMMENDED_ACTIONS},
                        "keyword_only_count": sum(1 for row in rows if "keyword_mentions_ignored" in row["reason_codes"])},
            "submission_order": [row["work_id"] for row in rows if row["recommended_action"] in SUBMISSION_ACTIONS],
            "follow_ups": [{"work_id": row["work_id"], "public_owner": row["public_owner"],
                            "recommended_action": row["recommended_action"]}
                           for row in rows if row["dependence"] != "unknown" and row["recommended_action"] != "unknown"],
            "issues": sorted(issues, key=lambda issue: (issue["location"], issue["code"], issue["message"])),
            "rows": rows}


def render_inventory_json(report: Mapping[str, Any]) -> str:
    """Return compact deterministic sorted-key JSON with a trailing newline."""
    return json.dumps(report, sort_keys=True) + "\n"


def render_inventory_text(report: Mapping[str, Any]) -> str:
    """Return the concise deterministic human inventory summary."""
    summary = report["summary"]
    lines = [f"inventory status={report['status']} rows={summary['row_count']} "
             f"resource_dependent={summary['resource_dependent_count']} "
             f"local_later={summary['locally_runnable_later_count']} "
             f"unknown={summary['unknown_count']}",
             "actions " + " ".join(f"{name}={count}"
                                   for name, count in summary["action_counts"].items())]
    lines.extend(f"- {row['work_id']}: dependence={row['dependence']} "
                 f"action={row['recommended_action']} "
                 f"reasons={','.join(row['reason_codes']) or '-'}" for row in report["rows"])
    lines.append(report["claim_boundary"])
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name}: inventory input must be a JSON object")
    return payload


def load_case_inventories(path: Path) -> dict[str, Mapping[str, Any]]:
    """Load a named inventory case pack, sorted by case name."""
    payload = _read_json(path)
    cases = payload.get("cases")
    if payload.get("schema") != CASES_SCHEMA or not isinstance(cases, Mapping):
        raise ValueError(f"{path.name}: expected schema {CASES_SCHEMA}")
    if not all(isinstance(case, Mapping) for case in cases.values()):
        raise ValueError(f"{path.name}: every case must be a mapping")
    return {str(name): case for name, case in sorted(cases.items())}


def _documents(paths: Sequence[Path], case: str | None) -> list[Mapping[str, Any]]:
    documents: list[Mapping[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        if payload.get("schema") == CASES_SCHEMA:
            if len(paths) != 1:
                raise ValueError("a case pack must be the only --inputs entry")
            cases = load_case_inventories(path)
            if case is None:
                raise ValueError("a case pack requires --case")
            if case not in cases:
                raise ValueError(f"unknown case {case!r}")
            documents.append(cases[case])
        else:
            if case is not None:
                raise ValueError("--case requires a case pack")
            documents.append(payload)
    return documents


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the inventory argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--case", default=None, help="Case name in a case pack.")
    parser.add_argument("--as-of", default=None, help="ISO 8601 override for the document as_of.")
    parser.add_argument("--private-inventory", type=Path, default=None)
    parser.add_argument("--private-owner", default=None)
    parser.add_argument("--check", action="store_true", help="required; never writes or mutates")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only inventory CLI and return a shell-friendly exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("--check is required; this inventory never writes or mutates state")
    try:
        as_of = _timestamp(args.as_of) if args.as_of is not None else None
        if args.as_of is not None and as_of is None:
            raise ValueError("--as-of must be a timezone-aware ISO 8601 timestamp")
        binding = None
        if args.private_inventory is not None:
            owner = _slug(args.private_owner)
            if owner is None:
                raise ValueError("--private-inventory requires --private-owner")
            binding = {"owner": owner, "class": "external_private_inventory",
                       "sha256": _sha256_file(args.private_inventory), "content_published": False}
        elif args.private_owner is not None:
            raise ValueError("--private-owner requires --private-inventory")
        report = build_inventory(_documents(args.inputs, args.case), as_of=as_of,
                                 private_binding=binding)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"FAIL invalid_input: {exc}\n")
        return 2
    rendered = (render_inventory_json(report) if args.format == "json"
                else render_inventory_text(report))
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
