#!/usr/bin/env python3
"""Bounded read-only batch preflight runner for registered urgent packets (#8847).

Runs one canonical, side-effect-free local preflight per explicitly registered urgent
campaign packet and emits one deterministic readiness matrix. Commands come only from the
registry (never inferred from prose), are invoked as argument vectors without ``shell=True``
under a minimal sanitized environment, and are bounded by a per-command timeout and a
captured-output cap. Cheap shared checks with identical identities run once, while each
packet row keeps its own evidence (command, source/config digests, exit status, duration,
normalized output digest, first blocker, expiry time).

Classifications: ``ready_for_private_submission_check``, ``local_preflight_failed``,
``blocked_prerequisite``, ``stale_input``, ``duplicate_active``,
``resource_projection_unavailable``, ``unsupported``. A local pass is never compute
authority or scheduler admission: every row stays ``compute_authority=not_granted`` and
``scientific_status=not_evaluated``. Timeouts, malformed output, path escapes, source
drift, command mismatch, missing validators, and unsupported entries fail closed.

Check-only CLI: ``--check --registry <registry.json> [--repo-root PATH]
[--format json|text]``. Exit codes: 0 every packet ready, 1 actionable rows, 2 malformed
registry. The runner never writes durable state, submits, cancels, relabels, or contacts
external services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dev.run_compact_validation import run_compact_validation  # noqa: E402
from scripts.tools.classify_scheduler_failure import sanitize_text  # noqa: E402

REGISTRY_SCHEMA = "robot_sf.urgent_packet_preflight_registry.v1"
REPORT_SCHEMA = "robot_sf.urgent_packet_preflight_report.v1"
CLAIM_BOUNDARY = (
    "Read-only batch preflight only: local pass states are not compute authority, scheduler "
    "admission, private-submission approval, artifact custody, or scientific evidence, and "
    "this check-only runner never submits, cancels, or mutates GitHub or scientific state."
)
READY = "ready_for_private_submission_check"
FAILED = "local_preflight_failed"
BLOCKED = "blocked_prerequisite"
STALE = "stale_input"
DUPLICATE = "duplicate_active"
RESOURCE = "resource_projection_unavailable"
UNSUPPORTED = "unsupported"
CLASSIFICATIONS = (READY, FAILED, BLOCKED, STALE, DUPLICATE, RESOURCE, UNSUPPORTED)
NEXT_ACTIONS = {
    READY: "request_private_submission_admission",
    FAILED: "repair_local_preflight",
    BLOCKED: "resolve_prerequisite",
    STALE: "refresh_registered_inputs",
    DUPLICATE: "reconcile_duplicate_registration",
    RESOURCE: "refresh_resource_projection",
    UNSUPPORTED: "register_side_effect_free_preflight",
}
OUTPUT_CONTRACTS = ("json_object", "json_tail_object", "text")
UNSUPPORTED_REASONS = frozenset(
    (
        "no_canonical_side_effect_free_preflight",
        "requires_external_services",
        "requires_substantive_workload_execution",
        "requires_scheduler_or_host_access",
    )
)
BLOCKED_TOKENS = frozenset(
    "blocked not_ready prerequisite_unmet admission_blocked not_certified".split()
)
RESOURCE_TOKENS = frozenset(
    "capacity_unknown capacity_exceeded resource_unavailable resource_projection_unavailable "
    "capacity_unavailable".split()
)
STALE_TOKENS = frozenset("stale stale_input source_drift drift expired".split())
MAX_CAPTURE_BYTES = 65536
MAX_TIMEOUT_SECONDS = 900.0
_SHELL_CHARS_RE = re.compile(r"[\x00-\x1f|&;$`<>]")
_PYTHON_NAME_RE = re.compile(r"^python3(?:\.\d+)?$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_ENV_RE = re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|token|credential|bearer)")
_ENV_ALLOWLIST = (
    "PATH HOME LANG LC_ALL TERM TMPDIR PYTHONPATH VIRTUAL_ENV UV_PROJECT_ENVIRONMENT "
    "UV_NO_SYNC UV_CACHE_DIR UV_OFFLINE MPLCONFIGDIR XDG_CACHE_HOME COVERAGE_FILE".split()
)
_REGISTRY_KEYS = frozenset(("schema", "registry_id", "generated_at", "entries"))
_ENTRY_KEYS = frozenset(
    (
        "packet_id issue preflight_argv timeout_seconds output_contract preflight_sha256 "
        "packet_sha256 expires_at resource_projection shared_check unsupported_reason"
    ).split()
)
_SHARED_CHECK_KEYS = frozenset("check_id argv timeout_seconds output_contract".split())


class RegistryError(ValueError):
    """Malformed registry input that prevents any packet row from being produced."""


@dataclass(frozen=True, slots=True)
class SharedCheck:
    """One explicitly registered cheap shared check; identical identities run once."""

    check_id: str
    argv: tuple[str, ...]
    timeout_seconds: float
    output_contract: str


@dataclass(frozen=True, slots=True)
class Preflight:
    """One canonical side-effect-free preflight command and its pinned identity."""

    argv: tuple[str, ...]
    timeout_seconds: float
    output_contract: str
    script_relative: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Entry:
    """One registered urgent packet; parse or policy problems make it unsupported."""

    packet_id: str
    issue: int
    preflight: Preflight | None
    packet_sha256: str | None
    expires_at: datetime | None
    resource_projection: tuple[str, str | None] | None
    shared_check: SharedCheck | None
    unsupported_reason: str | None
    problem: str | None


def _finding(code: str, location: str, message: str) -> dict[str, str]:
    return {"code": code, "location": location, "message": message}


def _sorted_findings(findings: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    unique = {(item["code"], item["location"], item["message"]): dict(item) for item in findings}
    return [unique[key] for key in sorted(unique)]


def _stable_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _slug(value: Any) -> str | None:
    return value if isinstance(value, str) and _SLUG_RE.fullmatch(value) else None


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _positive_float(value: Any, *, maximum: float = MAX_TIMEOUT_SECONDS) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and 0 < parsed <= maximum else None


def _sha256_hex(value: Any) -> str | None:
    text = value.lower() if isinstance(value, str) else ""
    return text if _SHA256_RE.fullmatch(text) else None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _render_timestamp(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_text(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()


def _sanitized_environment() -> dict[str, str]:
    """Return a minimal environment free of credentials and private topology."""
    env = {
        name: os.environ[name]
        for name in _ENV_ALLOWLIST
        if name in os.environ and not _SECRET_ENV_RE.search(name)
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def _is_python_name(name: str) -> bool:
    return name in {"python", "python3"} or _PYTHON_NAME_RE.fullmatch(name) is not None


def _validate_argv(  # noqa: C901 - one fail-closed argv policy pass
    argv: Sequence[str], repo_root: Path
) -> tuple[str | None, list[str]]:
    """Validate one argument vector and return (script_relative, problems)."""
    problems: list[str] = []
    tokens = list(argv)
    if not tokens:
        return None, ["missing_preflight_argv"]
    index = 1
    executable = Path(tokens[0]).name
    if executable == "uv":
        if len(tokens) < 4 or tokens[1] != "run" or not _is_python_name(Path(tokens[2]).name):
            return None, ["unsupported_argv_shape"]
        index = 3
    elif _is_python_name(executable):
        index = 1
    else:
        return None, ["unsupported_executable"]
    for token in tokens:
        if _SHELL_CHARS_RE.search(token):
            problems.append("shell_metacharacter_in_argv")
    if index >= len(tokens):
        return None, [*problems, "missing_preflight_script"]
    script_token = tokens[index]
    if Path(script_token).is_absolute():
        problems.append("absolute_validator_path")
    resolved = (repo_root / script_token).resolve(strict=False)
    if not resolved.is_relative_to(repo_root.resolve()):
        problems.append("path_escape_in_argv")
    for token in tokens[index + 1 :]:
        if token.startswith("-"):
            continue
        candidate = repo_root / token
        if ("/" in token or token.endswith((".py", ".json", ".yaml", ".yml"))) and not (
            candidate.resolve(strict=False).is_relative_to(repo_root.resolve())
        ):
            problems.append("path_escape_in_argv")
    if not resolved.is_file():
        problems.append("missing_validator")
    return script_token, problems


def _parse_output(contract: str, text: str) -> Mapping[str, Any] | str | None:
    """Return the parsed contract payload, or ``None`` for malformed output."""
    stripped = _normalize_text(text)
    if not stripped:
        return None
    if contract == "text":
        return stripped
    try:
        whole = json.loads(stripped)
    except ValueError:
        whole = None
    if isinstance(whole, Mapping):
        return whole
    if contract != "json_tail_object":
        return None
    lines = stripped.splitlines()
    for start in range(len(lines) - 1, -1, -1):
        if not lines[start].strip().startswith("{"):
            continue
        try:
            candidate = json.loads("\n".join(lines[start:]))
        except ValueError:
            continue
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _classify_result(
    *, exit_status: int, timed_out: bool, truncated: bool, payload: Mapping[str, Any] | str | None
) -> tuple[str, str | None, str | None]:
    """Classify one bounded command result and return (state, first_blocker, status_token)."""
    if timed_out:
        return FAILED, "timeout", None
    if truncated:
        return FAILED, "output_limit_exceeded", None
    if payload is None:
        blocker = f"exit_status_{exit_status}" if exit_status != 0 else "malformed_output"
        return FAILED, blocker, None
    token = None
    if isinstance(payload, Mapping) and isinstance(payload.get("status"), str):
        token = payload["status"].strip().lower()
    if token in STALE_TOKENS:
        return STALE, f"status_token_{token}", token
    if token in RESOURCE_TOKENS:
        return RESOURCE, f"status_token_{token}", token
    if token in BLOCKED_TOKENS:
        return BLOCKED, f"status_token_{token}", token
    if exit_status != 0:
        return FAILED, f"exit_status_{exit_status}", token
    return READY, None, token


def _execute(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    output_contract: str,
    env: Mapping[str, str],
    run_dir: Path,
) -> dict[str, Any]:
    """Run one argv under the bounded validation owner and return compact evidence."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_compact_validation(
        list(argv),
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        artifact_dir=run_dir,
        env=dict(env),
    )
    raw = Path(summary["log_path"]).read_bytes()
    truncated = len(raw) > MAX_CAPTURE_BYTES
    text = raw[:MAX_CAPTURE_BYTES].decode("utf-8", errors="replace")
    payload = None if truncated else _parse_output(output_contract, text)
    digest_source: Any = payload if isinstance(payload, Mapping) else _normalize_text(text)
    return {
        "exit_status": int(summary["exit_code"]),
        "timed_out": bool(summary["timed_out"]),
        "duration_seconds": round(float(summary["elapsed_seconds"]), 3),
        "truncated": truncated,
        "payload": payload,
        "output_digest": _stable_digest(digest_source),
    }


def _parse_resource_projection(
    raw: Any, location: str, repo_root: Path, findings: list[dict[str, str]]
) -> tuple[str, str | None] | None:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
        findings.append(_finding("invalid_resource_projection", location, "path required"))
        return None
    path_text = raw["path"]
    resolved = (repo_root / path_text).resolve(strict=False)
    if Path(path_text).is_absolute() or not resolved.is_relative_to(repo_root.resolve()):
        findings.append(_finding("path_escape_in_resource_projection", location, "path escapes"))
        return None
    digest = _sha256_hex(raw.get("sha256"))
    if raw.get("sha256") is not None and digest is None:
        findings.append(_finding("invalid_resource_projection", location, "sha256 malformed"))
        return None
    return path_text, digest


def _parse_shared_check(
    raw: Any, location: str, repo_root: Path, findings: list[dict[str, str]]
) -> SharedCheck | None:
    if not isinstance(raw, Mapping) or set(raw) - _SHARED_CHECK_KEYS:
        findings.append(_finding("invalid_shared_check", location, "unknown or missing fields"))
        return None
    check_id = _slug(raw.get("check_id"))
    timeout = _positive_float(raw.get("timeout_seconds"))
    contract = raw.get("output_contract")
    argv = raw.get("argv")
    _, argv_problems = _validate_argv(argv, repo_root) if isinstance(argv, list) else (None, [])
    if check_id is None or timeout is None or contract not in OUTPUT_CONTRACTS or argv_problems:
        findings.append(
            _finding("invalid_shared_check", location, argv_problems[0] if argv_problems else "bad")
        )
        return None
    return SharedCheck(check_id, tuple(argv), timeout, contract)


def _parse_entry(  # noqa: C901 - one fail-closed validation pass
    raw: Any, index: int, repo_root: Path, findings: list[dict[str, str]]
) -> Entry | None:
    location = f"/entries[{index}]"
    if not isinstance(raw, Mapping):
        findings.append(_finding("invalid_entry", location, "mapping required"))
        return None
    problems: list[str] = []
    for key in sorted(set(raw) - _ENTRY_KEYS):
        findings.append(_finding("unknown_entry_field", f"{location}/{key}", "not in schema"))
        problems.append("unknown_entry_field")
    packet_id = _slug(raw.get("packet_id"))
    issue = _positive_int(raw.get("issue"))
    if packet_id is None or issue is None:
        findings.append(_finding("invalid_entry_identity", location, "packet_id/issue required"))
        return None
    expires_text = raw.get("expires_at")
    expires = _parse_timestamp(expires_text)
    if expires_text is not None and expires is None:
        findings.append(_finding("invalid_expiry", f"{location}/expires_at", "timestamp malformed"))
        problems.append("invalid_expiry")
    resource_raw = raw.get("resource_projection")
    resource_projection = (
        _parse_resource_projection(
            resource_raw, f"{location}/resource_projection", repo_root, findings
        )
        if resource_raw is not None
        else None
    )
    if resource_raw is not None and resource_projection is None:
        problems.append("invalid_resource_projection")
    shared_raw = raw.get("shared_check")
    shared_check = (
        _parse_shared_check(shared_raw, f"{location}/shared_check", repo_root, findings)
        if shared_raw is not None
        else None
    )
    if shared_raw is not None and shared_check is None:
        problems.append("invalid_shared_check")
    identity = {
        "packet_id": packet_id,
        "issue": issue,
        "packet_sha256": _sha256_hex(raw.get("packet_sha256")),
        "resource_projection": resource_projection,
        "shared_check": shared_check,
    }
    expires_kwargs = {"expires_at": expires}
    reason = raw.get("unsupported_reason")
    if reason is not None:
        if reason not in UNSUPPORTED_REASONS:
            findings.append(_finding("unsupported_reason_invalid", location, "unknown reason"))
            problems.append("unsupported_reason_invalid")
        if raw.get("preflight_argv") is not None:
            findings.append(
                _finding("unsupported_entry_has_argv", location, "unsupported entries take no argv")
            )
            problems.append("unsupported_entry_has_argv")
        return Entry(
            **identity,
            **expires_kwargs,
            preflight=None,
            unsupported_reason=reason if reason in UNSUPPORTED_REASONS else None,
            problem=problems[0] if problems else None,
        )
    argv = raw.get("preflight_argv")
    script_relative, argv_problems = (
        _validate_argv(argv, repo_root)
        if isinstance(argv, list)
        else (None, ["missing_preflight_argv"])
    )
    timeout = _positive_float(raw.get("timeout_seconds"))
    contract = raw.get("output_contract")
    sha256 = _sha256_hex(raw.get("preflight_sha256"))
    for code in argv_problems:
        findings.append(_finding(code, f"{location}/preflight_argv", "argv rejected"))
    problems.extend(argv_problems)
    if timeout is None:
        problems.append("missing_timeout")
        findings.append(_finding("invalid_timeout", f"{location}/timeout_seconds", "positive"))
    if contract not in OUTPUT_CONTRACTS:
        problems.append("invalid_output_contract")
        findings.append(_finding("invalid_output_contract", location, "unknown contract"))
    if sha256 is None:
        problems.append("missing_preflight_sha256")
        findings.append(_finding("missing_preflight_sha256", location, "digest required"))
    preflight = None
    if not problems:
        assert script_relative is not None and timeout is not None and sha256 is not None
        preflight = Preflight(tuple(argv), timeout, contract, script_relative, sha256)
    return Entry(
        **identity,
        **expires_kwargs,
        preflight=preflight,
        unsupported_reason=None,
        problem=problems[0] if problems else None,
    )


def parse_registry(payload: Any, repo_root: Path) -> tuple[str, list[Entry], list[dict[str, str]]]:
    """Validate one registry document and return (registry_id, entries, findings)."""
    if not isinstance(payload, Mapping):
        raise RegistryError("registry must be a JSON object")
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise RegistryError(f"registry schema must be {REGISTRY_SCHEMA}")
    unknown = sorted(set(payload) - _REGISTRY_KEYS)
    if unknown:
        raise RegistryError(f"registry contains unknown fields: {', '.join(unknown)}")
    registry_id = _slug(payload.get("registry_id"))
    if registry_id is None:
        raise RegistryError("registry_id slug required")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RegistryError("registry must list at least one explicit packet entry")
    findings: list[dict[str, str]] = []
    entries: list[Entry] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries):
        entry = _parse_entry(raw, index, repo_root, findings)
        if entry is None:
            continue
        if entry.packet_id in seen:
            raise RegistryError(f"duplicate packet_id in registry: {entry.packet_id}")
        seen.add(entry.packet_id)
        entries.append(entry)
    return registry_id, entries, findings


def _early_blocker(entry: Entry, repo_root: Path, now: datetime) -> str | None:
    """Return the first fail-closed blocker resolved without running the preflight."""
    if entry.problem is not None:
        return entry.problem
    if entry.unsupported_reason is not None:
        return f"unsupported:{entry.unsupported_reason}"
    if entry.preflight is None:
        return "missing_preflight"
    if entry.expires_at is not None and entry.expires_at <= now:
        return "registration_expired"
    if _file_sha256(repo_root / entry.preflight.script_relative) != entry.preflight.sha256:
        return "preflight_source_drift"
    projection = entry.resource_projection
    if projection is not None:
        path_text, expected = projection
        target = repo_root / path_text
        if not target.is_file():
            return "resource_projection_missing"
        if expected is not None and _file_sha256(target) != expected:
            return "resource_projection_digest_mismatch"
    return None


def _resolve_shared_checks(
    entries: Sequence[Entry], blockers: Mapping[str, str], findings: list[dict[str, str]]
) -> tuple[dict[str, SharedCheck], dict[str, str]]:
    """Group identical shared checks and flag conflicting registrations."""
    groups: dict[str, SharedCheck] = {}
    conflicts: dict[str, str] = {}
    for entry in sorted(entries, key=lambda item: (item.issue, item.packet_id)):
        check = entry.shared_check
        if check is None or entry.packet_id in blockers:
            continue
        existing = groups.get(check.check_id)
        if existing is None:
            groups[check.check_id] = check
        elif existing != check:
            conflicts[entry.packet_id] = "conflicting_shared_check"
            findings.append(
                _finding(
                    "conflicting_shared_check",
                    f"/entries/{entry.packet_id}/shared_check",
                    "same check_id with different argv or bounds",
                )
            )
    return groups, conflicts


def _duplicate_owners(entries: Sequence[Entry], blockers: Mapping[str, str]) -> dict[str, str]:
    """Map later identical registrations to the primary packet that owns the work."""
    owners: dict[str, str] = {}
    seen: dict[tuple[Any, ...], str] = {}
    for entry in sorted(entries, key=lambda item: (item.issue, item.packet_id)):
        if entry.preflight is None or entry.packet_id in blockers:
            continue
        key = (
            entry.issue,
            entry.preflight.argv,
            entry.preflight.sha256,
            entry.packet_sha256,
        )
        if key in seen:
            owners[entry.packet_id] = seen[key]
        else:
            seen[key] = entry.packet_id
    return owners


def _shared_record(check_id: str, check: SharedCheck, result: Mapping[str, Any]) -> dict[str, Any]:
    if result["timed_out"]:
        status = "timeout"
    elif result["truncated"]:
        status = "malformed"
    elif result["exit_status"] != 0:
        status = "failed"
    elif result["payload"] is None:
        status = "malformed"
    else:
        status = "passed"
    return {
        "check_id": check_id,
        "command": list(check.argv),
        "command_sha256": _stable_digest(list(check.argv)),
        "exit_status": result["exit_status"],
        "timed_out": result["timed_out"],
        "duration_seconds": result["duration_seconds"],
        "output_digest": result["output_digest"],
        "status": status,
    }


def _row(
    entry: Entry,
    classification: str,
    *,
    first_blocker: str | None,
    executed: bool,
    evidence: Mapping[str, Any],
    shared_check_id: str | None,
    duplicate_of: str | None,
) -> dict[str, Any]:
    return {
        "packet_id": entry.packet_id,
        "issue": entry.issue,
        "classification": classification,
        "next_action": NEXT_ACTIONS[classification],
        "first_blocker": first_blocker,
        "executed": executed,
        "duplicate_of": duplicate_of,
        "expires_at": _render_timestamp(entry.expires_at),
        "shared_check_id": shared_check_id,
        "compute_authority": "not_granted",
        "scientific_status": "not_evaluated",
        "evidence": {
            "command": list(entry.preflight.argv) if entry.preflight else None,
            "command_sha256": (
                _stable_digest(list(entry.preflight.argv)) if entry.preflight else None
            ),
            "preflight_source_sha256": entry.preflight.sha256 if entry.preflight else None,
            "packet_sha256": entry.packet_sha256,
            "unsupported_reason": entry.unsupported_reason,
            **evidence,
        },
    }


def run_preflight(payload: Any, *, repo_root: Path, now: datetime | None = None) -> dict[str, Any]:
    """Run the bounded batch preflight and return the deterministic readiness matrix."""
    repo_root = repo_root.resolve()
    now = now or datetime.now(UTC)
    registry_id, entries, findings = parse_registry(payload, repo_root)
    blockers = {
        entry.packet_id: blocker
        for entry in entries
        if (blocker := _early_blocker(entry, repo_root, now)) is not None
    }
    groups, conflicts = _resolve_shared_checks(entries, blockers, findings)
    blockers.update(conflicts)
    owners = _duplicate_owners(entries, blockers)
    rows: list[dict[str, Any]] = []
    shared_records: dict[str, dict[str, Any]] = {}
    shared_status: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="urgent-packet-preflight-") as temp_dir:
        artifact_root = Path(temp_dir)
        env = _sanitized_environment()
        run_index = 0
        for check_id in sorted(groups):
            check = groups[check_id]
            result = _execute(
                check.argv,
                cwd=repo_root,
                timeout_seconds=check.timeout_seconds,
                output_contract=check.output_contract,
                env=env,
                run_dir=artifact_root / f"shared-{check_id}",
            )
            record = _shared_record(check_id, check, result)
            shared_records[check_id] = record
            shared_status[check_id] = record["status"]
        for entry in sorted(entries, key=lambda item: (item.issue, item.packet_id)):
            evidence: dict[str, Any] = {
                "exit_status": None,
                "timed_out": False,
                "duration_seconds": None,
                "output_digest": None,
                "status_token": None,
                "shared_check_status": None,
            }
            duplicate_of = owners.get(entry.packet_id)
            check_id = entry.shared_check.check_id if entry.shared_check else None
            if duplicate_of is not None:
                rows.append(
                    _row(
                        entry,
                        DUPLICATE,
                        first_blocker=f"duplicate_of:{duplicate_of}",
                        executed=False,
                        evidence=evidence,
                        shared_check_id=check_id,
                        duplicate_of=duplicate_of,
                    )
                )
                continue
            if entry.packet_id in conflicts:
                rows.append(
                    _row(
                        entry,
                        UNSUPPORTED,
                        first_blocker=conflicts[entry.packet_id],
                        executed=False,
                        evidence=evidence,
                        shared_check_id=check_id,
                        duplicate_of=None,
                    )
                )
                continue
            if entry.packet_id in blockers:
                classification = (
                    UNSUPPORTED
                    if entry.problem or entry.unsupported_reason
                    else (
                        STALE
                        if blockers[entry.packet_id]
                        in {"registration_expired", "preflight_source_drift"}
                        else RESOURCE
                        if blockers[entry.packet_id].startswith("resource_projection")
                        else UNSUPPORTED
                    )
                )
                rows.append(
                    _row(
                        entry,
                        classification,
                        first_blocker=blockers[entry.packet_id],
                        executed=False,
                        evidence=evidence,
                        shared_check_id=check_id,
                        duplicate_of=None,
                    )
                )
                continue
            if check_id is not None:
                evidence["shared_check_status"] = shared_status.get(check_id)
                if shared_status.get(check_id) != "passed":
                    rows.append(
                        _row(
                            entry,
                            BLOCKED,
                            first_blocker=f"shared_check_not_passed:{check_id}",
                            executed=False,
                            evidence=evidence,
                            shared_check_id=check_id,
                            duplicate_of=None,
                        )
                    )
                    continue
            assert entry.preflight is not None
            run_index += 1
            result = _execute(
                entry.preflight.argv,
                cwd=repo_root,
                timeout_seconds=entry.preflight.timeout_seconds,
                output_contract=entry.preflight.output_contract,
                env=env,
                run_dir=artifact_root / f"packet-{run_index:03d}",
            )
            classification, first_blocker, token = _classify_result(
                exit_status=result["exit_status"],
                timed_out=result["timed_out"],
                truncated=result["truncated"],
                payload=result["payload"],
            )
            evidence.update(
                exit_status=result["exit_status"],
                timed_out=result["timed_out"],
                duration_seconds=result["duration_seconds"],
                output_digest=result["output_digest"],
                status_token=token,
            )
            rows.append(
                _row(
                    entry,
                    classification,
                    first_blocker=first_blocker,
                    executed=True,
                    evidence=evidence,
                    shared_check_id=check_id,
                    duplicate_of=None,
                )
            )
    counts = dict.fromkeys(CLASSIFICATIONS, 0)
    for row in rows:
        counts[row["classification"]] += 1
    ready_count = counts[READY]
    return {
        "schema": REPORT_SCHEMA,
        "check_only": True,
        "registry_id": registry_id,
        "registry_sha256": _stable_digest(payload),
        "summary": {
            "packet_count": len(rows),
            "ready_count": ready_count,
            "action_required_count": len(rows) - ready_count,
            "classification_counts": counts,
            "shared_check_count": len(shared_records),
            "finding_count": len(findings),
        },
        "shared_checks": [shared_records[key] for key in sorted(shared_records)],
        "rows": rows,
        "findings": _sorted_findings(findings),
        "compute_authority": "not_granted",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def render_report_json(report: Mapping[str, Any]) -> str:
    """Return byte-stable sorted-key readiness JSON with a trailing newline."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_report_text(report: Mapping[str, Any]) -> str:
    """Return the concise deterministic human readiness summary."""
    summary = report["summary"]
    lines = [
        f"urgent-packet-preflight: {summary['ready_count']} ready,"
        f" {summary['action_required_count']} action required"
        f" (packets={summary['packet_count']}, shared_checks={summary['shared_check_count']})",
    ]
    lines.extend(
        f"- issue {row['issue']} [{row['packet_id']}] {row['classification']}"
        f" blocker={row['first_blocker'] or 'none'}"
        for row in report["rows"]
    )
    if report["findings"]:
        lines.append("findings: " + ", ".join(item["code"] for item in report["findings"]))
    lines.append(report["claim_boundary"])
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the check-only runner argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate only; never writes")
    parser.add_argument("--registry", type=Path, required=True, help="urgent packet registry JSON")
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help="repository root for argv resolution"
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check-only batch preflight and return a shell-friendly exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("only --check is supported; the runner never submits or mutates state")
    try:
        payload = json.loads(args.registry.read_text(encoding="utf-8"))
        report = run_preflight(payload, repo_root=args.repo_root)
    except (OSError, ValueError) as exc:
        sys.stderr.write(
            f"FAIL malformed_registry: {sanitize_text(f'{type(exc).__name__}: {exc}')}\n"
        )
        return 2
    rendered = render_report_json(report) if args.format == "json" else render_report_text(report)
    sys.stdout.write(rendered)
    return 0 if report["summary"]["action_required_count"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
