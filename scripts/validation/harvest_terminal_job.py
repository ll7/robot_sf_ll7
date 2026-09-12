#!/usr/bin/env python3
"""Fail-closed terminal-job harvest and durable transfer-manifest builder (issue #8824).

Reads one explicit ``terminal_job_harvest_request.v1`` document plus one explicit local
artifact root, verifies scheduler state, source/config/environment identity, the expected
row contract, inventory roles, and destination capacity, and emits a deterministic
``terminal_job_harvest.v1`` receipt, a private detailed receipt, and ``SHA256SUMS``.
Terminal states (completed/failed/cancelled/timeout) are never scientific success, and row
execution modes (native/adapter/fallback/degraded) never convert a failed or degraded row
into a passing one. Every expected row receives one explicit disposition and reason. Output
holds normalized relative paths only; the helper never copies, uploads, or deletes, and it
reads only explicit local inputs. Cleanup eligibility is emitted only after a destination
copy verifies against the exact manifest member set. Exit codes: 0 ready, 2 harvest_blocked,
3 malformed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Ensure repository-root imports resolve when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tools.chunk_manifest import (  # noqa: E402
    ChunkManifestError,
    normalize_relative_path,
    scan_root,
)

REQUEST_SCHEMA = "terminal_job_harvest_request.v1"
RECEIPT_SCHEMA = "terminal_job_harvest.v1"
PRIVATE_RECEIPT_SCHEMA = "terminal_job_harvest_private.v1"
RECEIPT_FILENAME = "terminal_job_harvest.v1.json"
PRIVATE_RECEIPT_FILENAME = "terminal_job_harvest_private.v1.json"
SUMS_FILENAME = "SHA256SUMS"
CLAIM_BOUNDARY = (
    "Operational transfer manifest only. Terminal scheduler state, artifact completeness, and "
    "row execution modes are recorded, never interpreted as scientific success. No private "
    "host, account, or artifact data is read or emitted."
)
INVENTORY_ROLES = (
    "stdout",
    "stderr",
    "manifest",
    "rows",
    "report",
    "checkpoint",
    "environment",
    "auxiliary_trace",
)
REQUIRED_ROLES = ("manifest", "rows", "environment")
ROLE_RETENTION = {
    "stdout": "diagnostic",
    "stderr": "diagnostic",
    "manifest": "durable_required",
    "rows": "durable_required",
    "report": "durable_required",
    "checkpoint": "release_facing",
    "environment": "durable_required",
    "auxiliary_trace": "diagnostic",
}
EXECUTION_MODES = ("native", "adapter", "fallback", "degraded")
ROW_DISPOSITIONS = ("present", "duplicate", "corrupt", "failed", "unavailable", "missing")
ROW_STATUS = {
    "completed": ("present", None),
    "failed": ("failed", "row_failed"),
    "unavailable": ("unavailable", "row_unavailable"),
}
TERMINAL_STATES = frozenset(("completed", "failed", "cancelled", "timeout"))
NONTERMINAL_STATES = frozenset(("pending", "running"))
SCHEDULER_STATES = frozenset(TERMINAL_STATES | NONTERMINAL_STATES)
EXIT_READY, EXIT_BLOCKED, EXIT_MALFORMED = 0, 2, 3
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ISSUE_URL_RE = re.compile(r"^https://github\.com/ll7/robot_sf_ll7/issues/(\d+)$")
_PRIVATE_LOCATOR_RE = re.compile(r"(^/|^[A-Za-z]:[\\/]|~|\$|://|@github\.com)")
Problems = list[tuple[str, str, str]]


class HarvestContractError(ValueError):
    """Raised when the request cannot be read as a harvest contract at all."""


def _add(problems: Problems, code: str, location: str, message: str) -> None:
    problems.append((code, location, message))


def _get(node: Any, *keys: str) -> Any:
    for key in keys:
        node = node.get(key) if isinstance(node, Mapping) else None
    return node


def _match(value: Any, pattern: re.Pattern[str]) -> str | None:
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def _int_at_least(value: Any, minimum: int) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
        return value
    return None


def _stable_digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_request(path: Path) -> Mapping[str, Any]:
    """Load one harvest request JSON object, failing closed on malformed input."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HarvestContractError(f"cannot read request file {path.name!r}") from exc
    except ValueError as exc:
        raise HarvestContractError(f"request file {path.name!r} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise HarvestContractError(f"request file {path.name!r} must contain a JSON object")
    return payload


def _identity_contract(
    request: Mapping[str, Any], problems: Problems, expected_issue: int | None
) -> dict[str, Any]:
    issue = _int_at_least(request.get("issue"), 1)
    url = request.get("issue_url")
    match = _ISSUE_URL_RE.fullmatch(url) if isinstance(url, str) else None
    if issue is None:
        _add(problems, "missing_issue_identity", "issue", "positive issue number required")
    if match is None:
        _add(problems, "missing_issue_identity", "issue_url", "public issue url required")
    elif issue is not None and int(match.group(1)) != issue:
        _add(problems, "issue_mismatch", "issue", "issue number and url disagree")
    if expected_issue is not None and issue != expected_issue:
        _add(problems, "issue_mismatch", "issue", "request issue does not match --issue")

    identity: dict[str, Any] = {
        "issue": {"number": issue, "url": url if match is not None else None},
        "job": {},
        "ownership": {},
        "source": {},
    }
    rules = (
        ("job", "job_id", _SAFE_ID_RE, "missing_job_identity"),
        ("ownership", "owner", _SAFE_ID_RE, "missing_artifact_ownership"),
        ("ownership", "campaign_id", _SAFE_ID_RE, "missing_artifact_ownership"),
        ("source", "commit", _COMMIT_RE, "missing_source_identity"),
        ("source", "config_sha256", _SHA256_RE, "missing_source_identity"),
    )
    for section, key, pattern, code in rules:
        value = _match(_get(request, section, key), pattern)
        identity[section][key] = value
        if value is None:
            _add(problems, code, f"{section}.{key}", f"{key} is required")
    state = _get(request, "job", "state")
    if not isinstance(state, str) or state not in SCHEDULER_STATES:
        _add(
            problems,
            "missing_scheduler_state" if state is None else "unsupported_scheduler_state",
            "job.state",
            "unknown scheduler state",
        )
        state = "unknown"
    elif state in NONTERMINAL_STATES:
        _add(problems, "job_not_terminal", "job.state", "job is not terminal")
    identity["scheduler"] = {
        "state": state,
        "terminal": state in TERMINAL_STATES,
        "scientific_status": "not_evaluated",
    }
    return identity


def _output_contract(  # noqa: C901, PLR0912 - bounded row/inventory/capacity validation pass
    request: Mapping[str, Any], problems: Problems
) -> tuple[list[dict[str, str]], list[tuple[str, str]], int, bool]:
    rows_raw = request.get("rows")
    contract: list[dict[str, str]] = []
    seen: set[str] = set()
    if not isinstance(rows_raw, list) or not rows_raw:
        _add(problems, "missing_row_contract", "rows", "non-empty rows required")
    else:
        for index, item in enumerate(rows_raw):
            location = f"rows[{index}]"
            row_id = _match(_get(item, "row_id"), _SAFE_ID_RE)
            digest = _match(_get(item, "sha256"), _SHA256_RE)
            if row_id is None or digest is None:
                _add(problems, "missing_row_contract", location, "row_id and sha256 required")
            elif row_id in seen:
                _add(problems, "invalid_row_contract", location, "duplicate expected row")
            else:
                seen.add(row_id)
                contract.append({"row_id": row_id, "sha256": digest})
    inventory_raw = request.get("inventory")
    surfaces: list[tuple[str, str]] = []
    required_role_path_counts = dict.fromkeys(REQUIRED_ROLES, 0)
    if not isinstance(inventory_raw, Mapping):
        _add(problems, "missing_field", "inventory", "inventory mapping required")
    else:
        for role, paths in inventory_raw.items():
            if role not in INVENTORY_ROLES:
                _add(problems, "unsupported_role", f"inventory.{role}", "unknown inventory role")
            elif not isinstance(paths, list):
                _add(problems, "invalid_field", f"inventory.{role}", "paths must be a list")
            else:
                for index, raw_path in enumerate(paths):
                    location = f"inventory.{role}[{index}]"
                    relative = None
                    if not isinstance(raw_path, str) or not raw_path:
                        _add(problems, "missing_field", location, "non-empty path required")
                    elif _PRIVATE_LOCATOR_RE.search(raw_path):
                        _add(problems, "private_locator_rejected", location, "private locator")
                    else:
                        try:
                            relative = normalize_relative_path(raw_path)
                        except ChunkManifestError as exc:
                            _add(problems, exc.code, location, str(exc))
                    if relative is not None:
                        surfaces.append((role, relative))
                        if role in required_role_path_counts:
                            required_role_path_counts[role] += 1
        for role in REQUIRED_ROLES:
            if role not in inventory_raw:
                _add(problems, "missing_required_role", f"inventory.{role}", "required role")
            elif not isinstance(inventory_raw[role], list):
                required_role_path_counts[role] = 0
            elif not inventory_raw[role]:
                _add(
                    problems,
                    "empty_required_role",
                    f"inventory.{role}",
                    "required role must contain at least one path",
                )
                required_role_path_counts[role] = 0
    capacity = _int_at_least(_get(request, "destination", "capacity_bytes"), 1)
    if capacity is None:
        capacity = 0
        _add(
            problems, "missing_capacity_receipt", "destination.capacity_bytes", "positive capacity"
        )
    required_roles_valid = isinstance(inventory_raw, Mapping) and all(
        required_role_path_counts[role] > 0 for role in REQUIRED_ROLES
    )
    return contract, surfaces, capacity, required_roles_valid


def _scan_artifact_root(root: Path, problems: Problems) -> list[dict[str, Any]]:
    try:
        members, _excluded = scan_root(root)
    except ChunkManifestError as exc:
        code = "artifact_root_unavailable" if exc.code == "root_not_directory" else exc.code
        _add(problems, code, "artifact_root", str(exc))
        return []
    return [
        {"relative_path": relative, "path": path, "byte_size": identity[0]}
        for relative, path, identity in members
    ]


def _covers(surface_path: str, member_path: str) -> bool:
    return member_path == surface_path or member_path.startswith(f"{surface_path}/")


def _member_row(
    relative: str,
    roles: list[str],
    retention: str,
    digest: str | None,
    size: int,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "relative_path": relative,
        "roles": roles,
        "retention_class": retention,
        "sha256": digest,
        "byte_size": size,
        "status": "present" if digest else "unavailable",
        "reason": reason,
    }


def _cover_members(  # noqa: C901 - one fail-closed coverage/inventory pass over members
    surfaces: list[tuple[str, str]], members: list[dict[str, Any]], problems: Problems, root: Path
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    covered: dict[str, dict[str, Any]] = {}
    for role, surface_path in surfaces:
        for member in members:
            if not _covers(surface_path, member["relative_path"]):
                continue
            entry = covered.setdefault(
                member["relative_path"], dict(member, roles=set(), retention=set())
            )
            entry["roles"].add(role)
            entry["retention"].add(ROLE_RETENTION[role])
    inventory: list[dict[str, Any]] = []
    for member in members:
        if member["relative_path"] not in covered:
            _add(problems, "unclassified_member", member["relative_path"], "no declared role")
    for relative, entry in sorted(covered.items()):
        if len(entry["retention"]) != 1:
            _add(problems, "conflicting_retention_class", relative, "conflicting retention")
            entry["retention_class"] = "unspecified"
        else:
            entry["retention_class"] = entry.pop("retention").pop()
        entry["roles"] = sorted(entry["roles"])
        try:
            entry["sha256"] = _sha256_file(entry["path"])
        except OSError:
            _add(problems, "member_unreadable", relative, "member could not be read")
            entry["sha256"] = None
        reason = None if entry["sha256"] else "member_unreadable"
        inventory.append(
            _member_row(
                relative,
                entry["roles"],
                entry["retention_class"],
                entry["sha256"],
                entry["byte_size"],
                reason,
            )
        )
    for role, surface_path in surfaces:
        if any(_covers(surface_path, member["relative_path"]) for member in members):
            continue
        reason = "empty_surface" if (root / surface_path).exists() else "not_present"
        inventory.append(_member_row(surface_path, [role], ROLE_RETENTION[role], None, 0, reason))
        if role in REQUIRED_ROLES:
            _add(problems, "missing_required_surface", surface_path, f"required {reason}")
    inventory.sort(key=lambda item: item["relative_path"].encode("utf-8"))
    return covered, inventory


def _bind_environment(
    covered: dict[str, dict[str, Any]], source: Mapping[str, Any], problems: Problems
) -> dict[str, Any]:
    record: dict[str, Any] = {"record": None, "source_commit": None, "config_sha256": None}
    candidates = sorted(
        (entry for entry in covered.values() if "environment" in entry["roles"]),
        key=lambda item: item["relative_path"].encode("utf-8"),
    )
    if not candidates:
        _add(problems, "missing_environment_record", "environment", "environment record required")
        return record

    parsed: list[tuple[dict[str, Any], str | None, str | None]] = []
    for entry in candidates:
        try:
            payload = json.loads(entry["path"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if not isinstance(payload, Mapping):
            _add(problems, "corrupt_environment_record", entry["relative_path"], "unreadable JSON")
            continue
        commit, config = payload.get("source_commit"), payload.get("config_sha256")
        parsed.append(
            (
                entry,
                commit if isinstance(commit, str) else None,
                config if isinstance(config, str) else None,
            )
        )
    if not parsed:
        return record
    if len({(commit, config) for _entry, commit, config in parsed}) > 1:
        for entry, _commit, _config in parsed:
            _add(
                problems,
                "conflicting_environment_records",
                entry["relative_path"],
                "environment identity differs from another record",
            )
        return record

    entry, commit, config = parsed[0]
    record["record"] = entry["relative_path"]
    record["source_commit"] = commit if isinstance(commit, str) else None
    record["config_sha256"] = config if isinstance(config, str) else None
    if source["commit"] is not None and record["source_commit"] != source["commit"]:
        _add(problems, "stale_source", entry["relative_path"], "source_commit differs")
    if source["config_sha256"] is not None and record["config_sha256"] != source["config_sha256"]:
        _add(problems, "config_digest_mismatch", entry["relative_path"], "config digest differs")
    return record


def _parse_row(entry: Mapping[str, Any], problems: Problems) -> dict[str, Any]:
    relative = entry["relative_path"]
    parsed: dict[str, Any] = {
        "relative_path": relative,
        "row_id": None,
        "mode": "unknown",
        "status": "corrupt",
        "sha256": entry["sha256"],
    }
    payload: Any = None
    if relative.endswith(".json"):
        try:
            payload = json.loads(entry["path"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
    row_id = _match(_get(payload, "row_id"), _SAFE_ID_RE)
    if row_id is None or row_id != relative.rsplit("/", 1)[-1][: -len(".json")]:
        _add(problems, "corrupt_row_record", relative, "row_id must match the filename stem")
        return parsed
    mode = payload.get("mode", "unknown") if isinstance(payload, Mapping) else "unknown"
    status = payload.get("status") if isinstance(payload, Mapping) else None
    if mode not in EXECUTION_MODES:
        _add(
            problems,
            "unsupported_row_mode",
            relative,
            "mode must be native/adapter/fallback/degraded",
        )
        mode = "unknown"
    if status not in ROW_STATUS:
        _add(
            problems,
            "unsupported_row_status",
            relative,
            "status must be completed/failed/unavailable",
        )
        status = "corrupt"
    parsed.update({"row_id": row_id, "mode": mode, "status": status})
    return parsed


def _row_entry(
    row_id: str | None, disposition: str, mode: str, digest: str | None, reason: str | None
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "disposition": disposition,
        "mode": mode,
        "sha256": digest,
        "reason": reason,
    }


def _reconcile_rows(
    contract: list[dict[str, str]], covered: dict[str, dict[str, Any]], problems: Problems
) -> dict[str, Any]:
    row_entries = sorted(
        (entry for entry in covered.values() if "rows" in entry["roles"]),
        key=lambda entry: entry["relative_path"].encode("utf-8"),
    )
    parsed = [_parse_row(entry, problems) for entry in row_entries]
    expected = {row["row_id"]: row for row in contract}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in parsed:
        if record["row_id"] is not None:
            grouped.setdefault(record["row_id"], []).append(record)
    rows: list[dict[str, Any]] = []
    for row_id in sorted(expected):
        files = grouped.pop(row_id, [])
        if not files:
            rows.append(_row_entry(row_id, "missing", "unknown", None, "expected_row_absent"))
        elif len(files) > 1:
            _add(problems, "duplicate_row", row_id, "multiple records for one expected row")
            rows.append(_row_entry(row_id, "duplicate", files[0]["mode"], None, "multiple_records"))
        else:
            record = files[0]
            if record["sha256"] != expected[row_id]["sha256"]:
                _add(problems, "row_checksum_mismatch", record["relative_path"], "digest differs")
                disposition, reason = "corrupt", "checksum_mismatch"
            else:
                disposition, reason = ROW_STATUS.get(
                    record["status"], ("corrupt", "record_invalid")
                )
            rows.append(_row_entry(row_id, disposition, record["mode"], record["sha256"], reason))
    for row_id in sorted(grouped):
        _add(problems, "unexpected_row", row_id, "not in the expected row contract")
    for record in parsed:
        if record["row_id"] is None:
            _add(problems, "unexpected_row", record["relative_path"], "not identifiable")
    disposition_counts = Counter(row["disposition"] for row in rows)
    mode_counts = Counter(row["mode"] for row in rows)
    return {
        "expected_count": len(expected),
        "accounted_count": sum(row["disposition"] != "missing" for row in rows),
        "disposition_counts": {name: disposition_counts.get(name, 0) for name in ROW_DISPOSITIONS},
        "mode_counts": {name: mode_counts.get(name, 0) for name in (*EXECUTION_MODES, "unknown")},
        "rows": rows,
    }


def _byte_counts(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    present = [item for item in inventory if item["status"] == "present"]
    return {"member_count": len(present), "total_bytes": sum(item["byte_size"] for item in present)}


def verify_destination(  # noqa: C901 - one exact destination membership verification pass
    destination_root: Path, inventory: list[dict[str, Any]], problems: Problems
) -> dict[str, Any]:
    """Verify an exact destination copy against the inventory; never writes.

    The destination member set must match the present source inventory exactly, following the
    canonical chunk-manifest verifier. Missing, mismatched, or unexpected members therefore keep
    the copy unverified and block cleanup eligibility.
    """
    expected = {
        item["relative_path"]: item["sha256"] for item in inventory if item["status"] == "present"
    }
    unavailable = {
        "verification": "unavailable",
        "missing_members": 0,
        "mismatched_members": 0,
        "unexpected_members": 0,
    }
    root = Path(destination_root)
    if root.is_symlink() or not root.is_dir():
        _add(problems, "destination_unavailable", "destination", "root must be a directory")
        return unavailable
    try:
        members, _excluded = scan_root(root)
    except ChunkManifestError as exc:
        _add(problems, exc.code, "destination", str(exc))
        return unavailable
    found = {relative: path for relative, path, _identity in members}
    expected_paths = set(expected)
    found_paths = set(found)
    missing = sorted(expected_paths - found_paths)
    unexpected = sorted(found_paths - expected_paths)
    mismatched = []
    for relative in sorted(expected_paths & found_paths):
        try:
            observed = _sha256_file(found[relative])
        except OSError:
            observed = None
        if observed != expected[relative]:
            mismatched.append(relative)
    for code, locations, message in (
        ("destination_incomplete", missing, "member is missing"),
        ("destination_digest_mismatch", mismatched, "digest differs"),
        ("destination_unexpected_member", unexpected, "member is not in the source inventory"),
    ):
        for relative in locations:
            _add(problems, code, relative, message)
    if missing:
        verification = "incomplete"
    elif mismatched:
        verification = "digest_mismatch"
    elif unexpected:
        verification = "unexpected_members"
    else:
        verification = "verified"
    return {
        "verification": verification,
        "missing_members": len(missing),
        "mismatched_members": len(mismatched),
        "unexpected_members": len(unexpected),
    }


def _cleanup_eligibility(
    status: str, artifact_status: str, destination: Mapping[str, Any]
) -> dict[str, Any]:
    reasons = []
    if destination["verification"] != "verified":
        reasons.append("destination_not_verified")
    if status != "ready":
        reasons.append("harvest_problems_present")
    if artifact_status != "complete":
        reasons.append("artifact_incomplete")
    return {"eligible": not reasons, "reason_codes": reasons}


def _finalize(report: dict[str, Any], problems: Problems) -> dict[str, Any]:
    report["problems"] = [
        {"code": code, "location": location, "message": message}
        for code, location, message in sorted(
            problems, key=lambda item: (item[1], item[0], item[2])
        )
    ]
    report["problem_count"] = len(report["problems"])
    report["reason_codes"] = sorted({item["code"] for item in report["problems"]})
    report["status"] = "ready" if not problems else "blocked"
    report["harvest_state"] = "harvested" if not problems else "harvest_blocked"
    report.pop("receipt_id", None)
    report["receipt_id"] = _stable_digest(report)
    return report


def build_report(
    request: Mapping[str, Any],
    artifact_root: Path,
    expected_issue: int | None = None,
    destination_root: Path | None = None,
) -> dict[str, Any]:
    """Build one deterministic terminal-job harvest receipt, ready or harvest_blocked."""
    if not isinstance(request, Mapping):
        raise HarvestContractError("harvest request must be a JSON object")
    problems: Problems = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        _add(problems, "schema_version_mismatch", "schema_version", f"must be {REQUEST_SCHEMA}")
    identity = _identity_contract(request, problems, expected_issue)
    contract, surfaces, capacity, required_roles_valid = _output_contract(request, problems)
    root = Path(artifact_root)
    root_ok = root.is_dir() and not root.is_symlink()
    members = _scan_artifact_root(root, problems) if root_ok else []
    if not root_ok:
        _add(problems, "artifact_root_unavailable", "artifact_root", "root must be a directory")
    covered, inventory = _cover_members(surfaces, members, problems, root)
    environment = _bind_environment(covered, identity["source"], problems)
    rows = _reconcile_rows(contract, covered, problems)
    if not root_ok or not inventory:
        artifact_status = "unavailable"
    elif (
        required_roles_valid
        and all(item["status"] == "present" for item in inventory)
        and all(row["disposition"] == "present" for row in rows["rows"])
    ):
        artifact_status = "complete"
    else:
        artifact_status = "partial"
    required = sum(item["byte_size"] for item in inventory if item["status"] == "present")
    fits = required <= capacity
    if not fits:
        _add(problems, "destination_capacity_exceeded", "destination.capacity_bytes", "too small")
    destination = {
        "capacity_bytes": capacity,
        "required_bytes": required,
        "fits": fits,
        "verification": "not_attempted",
        "missing_members": 0,
        "mismatched_members": 0,
        "unexpected_members": 0,
    }
    if destination_root is not None:
        destination.update(verify_destination(destination_root, inventory, problems))
    status = "ready" if not problems else "blocked"
    report: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "issue": identity["issue"],
        "job": identity["job"],
        "scheduler": identity["scheduler"],
        "ownership": identity["ownership"],
        "source": identity["source"],
        "environment": environment,
        "artifact_status": artifact_status,
        "row_reconciliation": rows,
        "inventory": inventory,
        "byte_counts": _byte_counts(inventory),
        "destination": destination,
        "cleanup_eligibility": _cleanup_eligibility(status, artifact_status, destination),
        "scientific_status": "not_evaluated",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return _finalize(report, problems)


def write_receipts(report: dict[str, Any], output_root: Path) -> tuple[dict[str, Any], list[str]]:
    """Write the receipt artifacts, refusing to overwrite different immutable bytes."""
    if report["status"] != "ready":
        raise HarvestContractError("refusing to write a blocked terminal job harvest")
    public = dict(report)
    public["problems"] = [
        {"code": p["code"], "location": p["location"]} for p in report["problems"]
    ]
    private = dict(report)
    private["schema_version"] = PRIVATE_RECEIPT_SCHEMA
    private["public_receipt"] = RECEIPT_FILENAME
    texts = {
        RECEIPT_FILENAME: json.dumps(public, indent=2, sort_keys=True) + "\n",
        PRIVATE_RECEIPT_FILENAME: json.dumps(private, indent=2, sort_keys=True) + "\n",
        SUMS_FILENAME: "".join(
            f"{item['sha256']}  {item['relative_path']}\n"
            for item in report["inventory"]
            if item["status"] == "present"
        ),
    }
    output_root = Path(output_root)
    for name, text in texts.items():
        target = output_root / name
        if target.exists() and target.read_text(encoding="utf-8") != text:
            _add(report["problems"], "immutable_output_conflict", name, "different bytes exist")
            report["cleanup_eligibility"] = _cleanup_eligibility(
                "blocked", report["artifact_status"], report["destination"]
            )
            return _finalize(report, report["problems"]), []
    output_root.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (output_root / name).write_text(text, encoding="utf-8")
    return report, sorted(texts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harvest_terminal_job", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--request", type=Path, default=None, help="Harvest request JSON.")
    parser.add_argument("--artifact-root", type=Path, default=None, help="Explicit artifact root.")
    parser.add_argument("--fixture", type=Path, default=None, help="request.json + artifact_root/.")
    parser.add_argument(
        "--destination-root", type=Path, default=None, help="Destination to verify."
    )
    parser.add_argument("--output-root", type=Path, default=None, help="Receipt output root.")
    parser.add_argument("--issue", type=int, default=None, help="Expected public issue number.")
    parser.add_argument("--check", action="store_true", help="Report only; write nothing.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    return parser


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.fixture is not None:
        if args.request is not None or args.artifact_root is not None:
            raise HarvestContractError("--fixture conflicts with --request/--artifact-root")
        return args.fixture / "request.json", args.fixture / "artifact_root"
    if args.request is None or args.artifact_root is None:
        raise HarvestContractError("pass --fixture or both --request and --artifact-root")
    return args.request, args.artifact_root


def main(argv: Sequence[str] | None = None) -> int:
    """Run the terminal-job harvest CLI and return the process exit code."""
    args = _parser().parse_args(argv)
    if args.check == (args.output_root is not None):
        print("error: pass exactly one of --check or --output-root", file=sys.stderr)
        return EXIT_MALFORMED
    written: list[str] = []
    try:
        request_path, artifact_root = _resolve_inputs(args)
        request = load_request(request_path)
        report = build_report(request, artifact_root, args.issue, args.destination_root)
        if report["status"] == "ready" and args.output_root is not None:
            report, written = write_receipts(report, args.output_root)
    except HarvestContractError as exc:
        print(f"terminal job harvest: malformed input: {exc}", file=sys.stderr)
        return EXIT_MALFORMED
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["status"] == "ready":
        rows = report["row_reconciliation"]
        print(
            "terminal job harvest: harvested "
            f"scheduler={report['scheduler']['state']} artifact={report['artifact_status']} "
            f"rows={rows['accounted_count']}/{rows['expected_count']} written={len(written)}"
        )
    else:
        print(
            f"terminal job harvest: {report['harvest_state']} ({', '.join(report['reason_codes'])})"
        )
        for problem in report["problems"]:
            print(f"  [{problem['code']}] {problem['location']}: {problem['message']}")
    return EXIT_READY if report["status"] == "ready" else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
