#!/usr/bin/env python3
"""Build a bounded, public-safe, check-only log retention report (#8856).

The manifest records provenance; this helper never deletes or changes logging. Failed or unknown
jobs retain full logs until diagnosis and durable transfer are proven.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.tools.check_prune_eligibility import APPROVED_DURABLE
from scripts.tools.chunk_manifest import ChunkManifestError, normalize_relative_path

INPUT_SCHEMA = "log_retention_manifest.v1"
REPORT_SCHEMA = "log_retention_report.v1"
POLICY_VERSION = "log_excerpt.v1"
CLAIM_BOUNDARY = (
    "Operational log retention diagnostics only: this records provenance, bounded sanitized "
    "excerpts, custody gates, and storage estimates. It never deletes, compresses, changes "
    "logging, job retries, or establishes scientific evidence."
)
LOG_ROLES = tuple("submission environment_preflight scheduler task_stdout task_stderr application_structured failure_excerpt summary harvest_transfer".split())  # fmt: skip
TASK_ROLES = frozenset("task_stdout task_stderr application_structured failure_excerpt".split())
COMPLETIONS = frozenset("complete active truncated unknown".split())
JOB_STATES = frozenset("completed failed unknown active cancelled timeout".split())
RETENTION_CLASSES = frozenset("durable_required release_facing historical diagnostic superseded disposable".split())  # fmt: skip
VERIFIED_MEMBER_STATES = frozenset("already_verified copied verified".split())
EXIT_OK, EXIT_BLOCKED, EXIT_MALFORMED = 0, 2, 3
MAX_LOGS = 256
MAX_INSPECT_BYTES = 4 * 1024 * 1024
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRIVATE_LOCATOR_RE = re.compile(r"(^/|^[A-Za-z]:[\\/]|~|\$|://|@)")
SECRET_RE = re.compile(
    r"(?ix)(?:\b(?:password|passwd|token|secret|api[_-]?key|authorization|bearer|private[_-]?key)\b\s*[:=]\s*\S+|gh[pousr]_[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{16,}|-----begin [^-]*private key-----)"
)
ERROR_RE = re.compile(r"(?i)\b(?:error|exception|traceback|failed|failure|fatal|oom|timeout)\b")
URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b(password|passwd|token|secret|api[_-]?key|authorization|bearer|private[_-]?key)\b\s*[:=]\s*\S+"
)
PRIVATE_PATH_RE = re.compile(
    r"(?i)(?<![\w])(?:/[^\s,;]+|[a-z]:[\\/][^\s,;]+|\\\\[^\s,;]+|~[/\\][^\s,;]+)"
)
STRUCTURED_SECRET_RE = re.compile(r"""(?ix)(["'](?:password|passwd|token|secret|api[_-]?key|authorization|bearer|private[_-]?key)["']\s*:\s*)(?:"[^"\n]*"|'[^'\n]*'|[^,}\s]+)""")  # fmt: skip
STRUCTURED_SECRET_KEY_RE = re.compile(
    r"""(?ix)(?:\\+)?["'](?:password|passwd|token|secret|api[_-]?key|authorization|bearer|private[_-]?key)(?:\\+)?["']"""
)
STRUCTURED_IDENTITY_KEY_RE = re.compile(r"""(?ix)(["'])(?:username|account|user)\1""")
STRUCTURED_IDENTITY_KEY_CANDIDATE_RE = re.compile(
    r"""(?ix)(?:[{,]\s*)(?P<quote>["'])(?:username|account|user)(?![A-Za-z0-9_])"""
)
STRUCTURED_IDENTITY_ESCAPED_KEY_RE = re.compile(
    r"""(?ix)(?:\\+["'](?:user|username|account)["']|["'](?:user|username|account)\\+["'])"""
)
IDENTITY_RE = re.compile(r"(?i)\b(?:user|username|account)\s*[:=]\s*\S+")
TOPOLOGY_RE = re.compile(
    r"(?i)\b(?:host|hostname|node|partition|topology|gpu|cpu|rank|world_size)\s*[:=]\s*\S+"
)
HOST_RE = re.compile(
    r"(?i)\b(?:[a-z0-9-]+\.)+(?:internal|local|cluster|example\.com)\b|\b(?:node|login|gpu|cn|host)[-_]?\d+\b"
)


def _problem(code: str, location: str, message: str = "") -> dict[str, str]:
    return {"code": code, "location": location, "message": message or code}


def _safe_id(value: Any) -> str | None:
    return value if isinstance(value, str) and SAFE_ID_RE.fullmatch(value) else None


def _member(value: Any, allowed: frozenset[str] | tuple[str, ...]) -> str | None:
    return value if isinstance(value, str) and value in allowed else None


def _digest(value: Any) -> str | None:
    return value if isinstance(value, str) and SHA256_RE.fullmatch(value) else None


def _size(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("manifest is unreadable or invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be a JSON object")
    return payload


def _relative(value: Any, location: str, problems: list[dict[str, str]]) -> str | None:
    if not isinstance(value, str) or not value or PRIVATE_LOCATOR_RE.search(value):
        problems.append(
            _problem("invalid_source_location", location, "relative public path required")
        )
        return None
    try:
        return normalize_relative_path(value)
    except ChunkManifestError as exc:
        problems.append(_problem(exc.code, location, str(exc)))
        return None


def _symlinked(path: Path) -> bool:
    """Return True when a path or an existing parent is a symlink."""
    return any(candidate.is_symlink() for candidate in (path, *path.parents))


def _safe_source(
    root: Path, resolved_root: Path, relative: str, location: str, problems: list[dict[str, str]]
) -> Path | None:
    """Resolve one source without reading links or paths outside the declared root."""
    source = root / relative
    linked = _symlinked(source)
    try:
        resolved = source.resolve(strict=False)
        escaped = resolved != resolved_root and resolved_root not in resolved.parents
    except (OSError, RuntimeError):
        problems.append(_problem("source_path_unavailable", location))
        return None
    if linked:
        problems.append(_problem("source_path_symlink", location))
    if escaped:
        problems.append(_problem("source_outside_root", location))
    return None if linked or escaped else source


def _policy(raw: Any, problems: list[dict[str, str]]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        problems.append(_problem("excerpt_policy_missing", "excerpt_policy"))
        return {}
    if raw.get("version") != POLICY_VERSION:
        problems.append(_problem("excerpt_policy_mismatch", "excerpt_policy.version"))
    result: dict[str, Any] = {"version": raw.get("version")}
    for name, minimum, maximum in (
        ("max_bytes", 32, 65536),
        ("first_lines", 1, 20),
        ("last_lines", 1, 20),
        ("context_lines", 0, 20),
        ("max_log_bytes", 1, MAX_INSPECT_BYTES),
    ):
        value = raw.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            problems.append(_problem("excerpt_policy_invalid", f"excerpt_policy.{name}"))
        else:
            result[name] = value
    if raw.get("loss_declared") is not True:
        problems.append(_problem("loss_not_declared", "excerpt_policy.loss_declared"))
    result["loss_declared"] = raw.get("loss_declared")
    return result


def _completion(
    raw: Mapping[str, Any],
    location: str,
    problems: list[dict[str, str]],
    reasons: list[str],
) -> str:
    """Reconcile completion aliases and retain only an established state."""
    fields = [(name, raw[name]) for name in ("completion", "completion_status") if name in raw]
    values: list[str] = []
    for name, value in fields:
        if not isinstance(value, str):
            code, reason = "completion_malformed", "completion_malformed"
        elif value not in COMPLETIONS:
            code, reason = "completion_unknown", "completion_unknown"
        else:
            values.append(value)
            if value == "unknown":
                code, reason = "completion_unknown", "unknown_completion"
            else:
                continue
        problems.append(_problem(code, f"{location}.{name}"))
        reasons.append(reason)
    if not fields:
        problems.append(_problem("completion_missing", f"{location}.completion"))
        reasons.append("completion_missing")
    if len(set(values)) > 1:
        problems.append(_problem("completion_contradictory", location))
        reasons.append("completion_contradictory")
    completion = values[0] if values else "unknown"
    if completion == "unknown" and "unknown_completion" not in reasons:
        reasons.append("unknown_completion")
    return completion


def _scan_file(path: Path, encoding: str | None) -> dict[str, Any]:
    """Hash and count a regular file, retaining text only within the bounded scan size."""
    digest = hashlib.sha256()
    size = lines = 0
    last_byte: int | None = None
    binary = False
    raw = bytearray()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                lines += chunk.count(b"\n")
                last_byte = chunk[-1] if chunk else last_byte
                binary |= b"\x00" in chunk
                if len(raw) <= MAX_INSPECT_BYTES:
                    raw.extend(chunk[: max(0, MAX_INSPECT_BYTES - len(raw))])
    except OSError as exc:
        return {"error": "source_unreadable", "message": str(exc)}
    if size and last_byte != 10:
        lines += 1
    text = None
    encoding_error = False
    if encoding is not None and not binary and len(raw) == size:
        try:
            text = bytes(raw).decode(encoding)
        except UnicodeDecodeError:
            encoding_error = True
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "line_count": lines,
        "text": text,
        "binary": binary,
        "encoding_error": encoding_error,
        "inspection_bounded": len(raw) != size,
    }


def _sanitize_structured_identity(  # noqa: C901, PLR0912 - fail-closed parser branches are explicit
    value: str,
) -> str:
    """Redact quoted identity fields, rejecting escaped or malformed values."""
    if STRUCTURED_IDENTITY_ESCAPED_KEY_RE.search(value):
        raise ValueError("structured identity escape is unsupported")
    for candidate in STRUCTURED_IDENTITY_KEY_CANDIDATE_RE.finditer(value):
        if candidate.end() >= len(value) or value[candidate.end()] != candidate.group("quote"):
            raise ValueError("structured identity is malformed")
    rendered: list[str] = []
    cursor = 0
    for match in STRUCTURED_IDENTITY_KEY_RE.finditer(value):
        if match.start() < cursor or (match.start() and value[match.start() - 1] == "\\"):
            raise ValueError("structured identity is malformed")
        colon = match.end()
        while colon < len(value) and value[colon].isspace():
            colon += 1
        if colon >= len(value) or value[colon] != ":":
            raise ValueError("structured identity is malformed")
        start = colon + 1
        while start < len(value) and value[start].isspace():
            start += 1
        if start >= len(value):
            raise ValueError("structured identity is malformed")
        if value[start] in "\"'":
            quote, end = value[start], start + 1
            while end < len(value) and value[end] != quote:
                if value[end] == "\\":
                    raise ValueError("structured identity escape is unsupported")
                end += 1
            if end >= len(value):
                raise ValueError("structured identity is malformed")
            end += 1
        else:
            end = start
            while end < len(value) and not value[end].isspace() and value[end] not in ",}":
                end += 1
            if end == start:
                raise ValueError("structured identity is malformed")
            if "\\" in value[start:end]:
                raise ValueError("structured identity escape is unsupported")
        separator = end
        while separator < len(value) and value[separator].isspace():
            separator += 1
        if separator < len(value) and value[separator] not in ",}":
            raise ValueError("structured identity is malformed")
        rendered.extend((value[cursor:start], "<redacted-identity>"))
        cursor = end
    rendered.append(value[cursor:])
    return "".join(rendered)


def _sanitize(value: str) -> str:
    value = URL_RE.sub("<redacted-url>", value)
    value = SECRET_ASSIGN_RE.sub(lambda match: f"{match.group(1)}=<redacted-secret>", value)
    value = STRUCTURED_SECRET_RE.sub(r"\1<redacted-secret>", value)
    value = _sanitize_structured_identity(value)
    value = PRIVATE_PATH_RE.sub("<redacted-path>", value)
    value = IDENTITY_RE.sub("<redacted-identity>", value)
    value = TOPOLOGY_RE.sub("<redacted-topology>", value)
    value = HOST_RE.sub("<redacted-host>", value)
    clean = "".join(char if char >= " " else " " for char in value)
    return " ".join(clean.split())


def _bounded(value: str, maximum: int) -> str:
    encoded = value.encode("utf-8")
    return encoded[:maximum].decode("utf-8", errors="ignore")


def _excerpt(text: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines:
        return {"ranges": [], "text": ""}
    raw_ranges: list[tuple[int, int, str]] = [
        (1, min(len(lines), policy["first_lines"]), "first"),
        (max(1, len(lines) - policy["last_lines"] + 1), len(lines), "last"),
    ]
    context = policy["context_lines"]
    for index, line in enumerate(lines):
        if ERROR_RE.search(line):
            raw_ranges.append(
                (max(1, index + 1 - context), min(len(lines), index + 1 + context), "error_context")
            )
    merged: list[list[Any]] = []
    for start, end, kind in sorted(raw_ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].add(kind)
        else:
            merged.append([start, end, {kind}])
    rendered: list[str] = []
    previous = 0
    ranges = []
    for start, end, kinds in merged:
        if previous and start > previous + 1:
            rendered.append("... <omitted> ...")
        for number in range(start, end + 1):
            rendered.append(f"{number}: {_sanitize(lines[number - 1])}")
        ranges.append({"kind": "+".join(sorted(kinds)), "start_line": start, "end_line": end})
        previous = end
    return {"ranges": ranges, "text": _bounded("\n".join(rendered), policy["max_bytes"])}


def _custody(  # noqa: C901, PLR0912
    raw: Any, entries: list[dict[str, Any]], problems: list[dict[str, str]]
) -> bool:
    if not isinstance(raw, Mapping):
        problems.append(_problem("durable_custody_missing", "custody"))
        return False
    verified = raw.get("status") == "verified"
    if not verified:
        problems.append(_problem("durable_custody_unverified", "custody.status"))
    if (
        not isinstance(raw.get("durability_class"), str)
        or raw.get("durability_class") not in APPROVED_DURABLE
    ):
        problems.append(_problem("durable_custody_unapproved", "custody.durability_class"))
        verified = False
    for name, aliases in (
        ("independent_verification", ("independent_verification",)),
        ("transfer_verification", ("transfer_verification", "transfer_verified")),
        ("consumer_review", ("consumer_review", "consumer_reviewed")),
    ):
        values = [raw[alias] for alias in aliases if alias in raw]
        if not values:
            problems.append(_problem("custody_proof_missing", f"custody.{name}"))
            verified = False
        elif any(value is not True for value in values):
            problems.append(_problem("custody_proof_invalid", f"custody.{name}"))
            verified = False
    files = raw.get("files")
    by_path: dict[str, Mapping[str, Any]] = {}
    if not isinstance(files, list):
        problems.append(_problem("durable_custody_incomplete", "custody.files"))
        return False
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            problems.append(_problem("durable_custody_incomplete", f"custody.files[{index}]"))
            continue
        path = item.get("relative_path", item.get("path"))
        if not isinstance(path, str):
            problems.append(_problem("durable_custody_incomplete", f"custody.files[{index}]"))
            continue
        path = _relative(path, f"custody.files[{index}].path", problems)
        if path is None:
            verified = False
            continue
        if _digest(item.get("sha256")) is None or not _size(item.get("byte_size")):
            problems.append(_problem("durable_custody_invalid_member", f"custody.files[{index}]"))
            verified = False
        if path in by_path:
            problems.append(_problem("duplicate_custody_member", f"custody.files[{index}]"))
        by_path[path] = item
    for index, entry in enumerate(entries):
        path = entry["path"]
        item = by_path.get(path) if path else None
        location = f"logs[{index}]"
        if _digest(entry.get("sha256")) is None or not _size(entry.get("size_bytes")):
            problems.append(_problem("custody_source_metadata_missing", location))
            verified = False
        if item is None:
            problems.append(_problem("durable_custody_missing_log", location))
            verified = False
        elif item.get("state") not in VERIFIED_MEMBER_STATES:
            problems.append(_problem("durable_custody_member_unverified", location))
            verified = False
        elif item.get("sha256") != entry.get("sha256") or item.get("byte_size") != entry.get(
            "size_bytes"
        ):
            problems.append(_problem("custody_mismatch", location))
            verified = False
    return verified


def build_report(  # noqa: C901, PLR0912, PLR0915
    manifest: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Return a deterministic report for an explicit log manifest and source root."""
    problems: list[dict[str, str]] = []
    if manifest.get("schema") != INPUT_SCHEMA:
        problems.append(_problem("schema_mismatch", "schema"))
    job = manifest.get("job")
    if not isinstance(job, Mapping):
        problems.append(_problem("job_identity_missing", "job"))
        job = {}
    job_id = _safe_id(job.get("job_id"))
    state = _member(job.get("state"), JOB_STATES) or "unknown"
    if job_id is None:
        problems.append(_problem("job_identity_missing", "job.job_id"))
    if "state" not in job or (state == "unknown" and job.get("state") != "unknown"):
        problems.append(_problem("job_state_unknown", "job.state"))
    if state == "active":
        problems.append(_problem("active_job", "job.state"))
    diagnosis = _member(
        job.get("diagnosis", manifest.get("diagnosis")),
        ("complete", "pending", "unknown", "not_required"),
    )
    if diagnosis is None:
        diagnosis = "unknown"
        problems.append(_problem("diagnosis_unknown", "job.diagnosis"))
    policy = _policy(manifest.get("excerpt_policy"), problems)
    policy_valid = (
        policy.get("version") == POLICY_VERSION
        and policy.get("loss_declared") is True
        and all(
            name in policy
            for name in ("max_bytes", "first_lines", "last_lines", "context_lines", "max_log_bytes")
        )
    )
    root = Path(root)
    root_linked = _symlinked(root)
    try:
        resolved_root = root.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved_root = root.absolute()
        root_linked = True
    if not root.is_dir():
        problems.append(_problem("source_root_unavailable", "source_root"))
    elif root_linked:
        problems.append(_problem("source_root_symlink", "source_root"))

    raw_logs = manifest.get("logs")
    if not isinstance(raw_logs, list) or not raw_logs:
        problems.append(_problem("logs_missing", "logs"))
        raw_logs = []
    if len(raw_logs) > MAX_LOGS:
        problems.append(_problem("too_many_logs", "logs"))
        raw_logs = raw_logs[:MAX_LOGS]
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_identities: set[tuple[str | None, str | None, str | None]] = set()
    for index, raw in enumerate(raw_logs):
        location = f"logs[{index}]"
        if not isinstance(raw, Mapping):
            problems.append(_problem("log_entry_invalid", location))
            continue
        path = _relative(raw.get("relative_path", raw.get("path")), f"{location}.path", problems)
        role = _member(raw.get("role"), LOG_ROLES)
        entry_reasons: list[str] = []
        if role is None:
            problems.append(_problem("log_role_missing", f"{location}.role"))
            entry_reasons.append("log_role_missing")
        item_job_id = _safe_id(raw.get("job_id"))
        if item_job_id is None:
            problems.append(_problem("job_identity_missing", f"{location}.job_id"))
            entry_reasons.append("job_identity_missing")
        elif job_id is not None and item_job_id != job_id:
            problems.append(_problem("identity_mismatch", f"{location}.job_id"))
            entry_reasons.append("identity_mismatch")
        task_id = _safe_id(raw.get("task_id"))
        if role in TASK_ROLES and task_id is None:
            problems.append(_problem("task_identity_missing", f"{location}.task_id"))
            entry_reasons.append("task_identity_missing")
        elif role not in TASK_ROLES and raw.get("task_id") not in (None, "job"):
            problems.append(_problem("task_identity_invalid", f"{location}.task_id"))
            entry_reasons.append("task_identity_invalid")
        identity = (role, item_job_id, task_id if role in TASK_ROLES else None)
        if path in seen_paths:
            problems.append(_problem("duplicate_log", location))
            entry_reasons.append("duplicate_log")
        elif path is not None:
            seen_paths.add(path)
        if identity in seen_identities:
            problems.append(_problem("duplicate_identity", location))
            entry_reasons.append("duplicate_identity")
        else:
            seen_identities.add(identity)
        completion = _completion(raw, location, problems, entry_reasons)
        for flag in ("active_writer", "truncated"):
            if flag in raw and not isinstance(raw[flag], bool):
                problems.append(_problem("flag_invalid", f"{location}.{flag}"))
                entry_reasons.append(f"{flag}_invalid")
        encoding = _member(raw.get("encoding"), ("utf-8", "utf-8-sig", "ascii"))
        if encoding is None:
            problems.append(_problem("encoding_missing", f"{location}.encoding"))
            entry_reasons.append("encoding_missing")
            encoding = None
        retention = _member(raw.get("retention_class"), RETENTION_CLASSES)
        if retention is None:
            problems.append(_problem("retention_class_missing", f"{location}.retention_class"))
            entry_reasons.append("retention_class_missing")
            retention = "unknown"
        declared_size = raw.get("size_bytes", raw.get("byte_size"))
        declared_lines = raw.get("line_count")
        declared_digest = _digest(raw.get("sha256"))
        for value, name in ((declared_size, "size_bytes"), (declared_lines, "line_count")):
            if not _size(value):
                problems.append(_problem("metadata_missing", f"{location}.{name}"))
                entry_reasons.append("metadata_missing")
        if declared_digest is None:
            problems.append(_problem("metadata_missing", f"{location}.sha256"))
            entry_reasons.append("metadata_missing")
        entry: dict[str, Any] = {
            "path": path,
            "role": role,
            "identity": {"job_id": item_job_id, "task_id": task_id},
            "size_bytes": None,
            "line_count": None,
            "encoding": encoding,
            "completion": completion,
            "sha256": None,
            "retention_class": retention,
            "source_location": f"root-relative:{path}" if path else None,
            "excerpt": None,
            "reasons": entry_reasons,
            "prune_eligible": False,
        }
        source = _safe_source(root, resolved_root, path, location, problems) if path else None
        if source is None or not source.is_file():
            problems.append(_problem("source_unreadable", location))
        elif encoding is not None:
            scan = _scan_file(source, encoding)
            if "error" in scan:
                problems.append(_problem(scan["error"], location))
            else:
                entry.update(
                    {
                        "size_bytes": scan["size_bytes"],
                        "line_count": scan["line_count"],
                        "sha256": scan["sha256"],
                    }
                )
                if scan["size_bytes"] != declared_size:
                    problems.append(_problem("size_mismatch", f"{location}.size_bytes"))
                    entry["reasons"].append("size_mismatch")
                if scan["line_count"] != declared_lines:
                    problems.append(_problem("line_count_mismatch", f"{location}.line_count"))
                    entry["reasons"].append("line_count_mismatch")
                if scan["sha256"] != declared_digest:
                    problems.append(_problem("digest_mismatch", f"{location}.sha256"))
                    entry["reasons"].append("digest_mismatch")
                if scan["binary"]:
                    problems.append(_problem("binary_log", location))
                    entry["reasons"].append("binary_log")
                if scan["encoding_error"]:
                    problems.append(_problem("encoding_error", location))
                    entry["reasons"].append("encoding_error")
                if scan["size_bytes"] > policy.get("max_log_bytes", 0):
                    problems.append(_problem("unbounded_verbosity", location))
                    entry["reasons"].append("unbounded_verbosity")
                if completion == "active" or raw.get("active_writer") is True:
                    problems.append(_problem("active_log", location))
                    entry["reasons"].append("active_log")
                if completion == "truncated" or raw.get("truncated") is True:
                    problems.append(_problem("truncated_log", location))
                    entry["reasons"].append("truncated_log")
                if scan["text"] is not None:
                    if (
                        SECRET_RE.search(scan["text"])
                        or STRUCTURED_SECRET_RE.search(scan["text"])
                        or STRUCTURED_SECRET_KEY_RE.search(scan["text"])
                    ):
                        problems.append(_problem("secret_like_log", location))
                        entry["reasons"].append("secret_like_log")
                    elif not entry["reasons"] and policy_valid:
                        try:
                            excerpt = _excerpt(scan["text"], policy)
                        except ValueError:
                            problems.append(_problem("redaction_failed", location))
                            entry["reasons"].append("redaction_failed")
                        else:
                            excerpt.update(
                                {
                                    "source_sha256": scan["sha256"],
                                    "source_location": entry["source_location"],
                                    "policy_version": policy.get("version"),
                                }
                            )
                            entry["excerpt"] = excerpt
                elif scan["inspection_bounded"]:
                    problems.append(_problem("excerpt_unavailable", location))
                    entry["reasons"].append("excerpt_unavailable")
        entries.append(entry)

    custody_ok = _custody(manifest.get("custody"), entries, problems)
    full_hold = state in {"failed", "unknown"} and (not custody_ok or diagnosis != "complete")
    global_blocked = bool(problems)
    for entry in entries:
        if state == "failed" and full_hold:
            entry["reasons"].append("failed_job_full_retention")
        elif state == "unknown" and full_hold:
            entry["reasons"].append("unknown_job_full_retention")
        entry["reasons"] = sorted(set(entry["reasons"]))
        entry["retention_action"] = (
            "retain_full"
            if full_hold
            else "retain_until_custody"
            if not custody_ok
            else "retain_full"
            if entry["reasons"] or global_blocked
            else "retain"
            if entry["retention_class"] != "disposable"
            else "prune_candidate"
        )
        entry["prune_eligible"] = bool(
            custody_ok and not global_blocked and not entry["reasons"] and full_hold is False
        )
    full_bytes = sum(entry["size_bytes"] or 0 for entry in entries)
    full_lines = sum(entry["line_count"] or 0 for entry in entries)
    excerpt_bound = len(entries) * policy.get("max_bytes", 0)
    prunable = sum(entry["size_bytes"] or 0 for entry in entries if entry["prune_eligible"])
    if not custody_ok:
        prune_state = "blocked_durable_custody"
    elif global_blocked:
        prune_state = "blocked"
    elif prunable:
        prune_state = "eligible"
    else:
        prune_state = "retain_required"
    problems.sort(key=lambda item: (item["location"], item["code"], item["message"]))
    return {
        "schema": REPORT_SCHEMA,
        "status": "blocked" if problems else "eligible",
        "check_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "job": {"job_id": job_id, "state": state, "diagnosis": diagnosis},
        "policy": policy,
        "custody": {
            "status": manifest.get("custody", {}).get("status")
            if isinstance(manifest.get("custody"), Mapping)
            else None,
            "verified": custody_ok,
        },
        "logs": sorted(entries, key=lambda item: (item["path"] or "", item["role"] or "")),
        "storage_estimate": {
            "full_log_bytes": full_bytes,
            "full_log_lines": full_lines,
            "excerpt_bytes_upper_bound": excerpt_bound,
            "retained_full_bytes": full_bytes - prunable,
            "prune_eligible_bytes": prunable if custody_ok else 0,
            "prune_eligibility": prune_state,
        },
        "deletion_plan": [],
        "problems": problems,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the check-only command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="inspect only; never deletes")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="log_retention_manifest.v1 JSON"
    )
    parser.add_argument("--root", type=Path, default=None, help="explicit source log root")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check-only log retention report."""
    args = build_arg_parser().parse_args(argv)
    if not args.check:
        print("error: --check is required; this helper never deletes", file=sys.stderr)
        return EXIT_MALFORMED
    try:
        report = build_report(_load(args.manifest), args.root or args.manifest.parent)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MALFORMED
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        estimate = report["storage_estimate"]
        print(
            f"log retention: {report['status']} logs={len(report['logs'])} "
            f"bytes={estimate['full_log_bytes']} prune={estimate['prune_eligibility']}"
        )
    return EXIT_OK if report["status"] == "eligible" else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
