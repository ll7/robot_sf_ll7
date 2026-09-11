#!/usr/bin/env python3
"""Inventory compute-window output schemas and their exact readers.

This is an operational preservation check, not a converter or an analysis tool.  A packet
contains one record per output role and binds its schema, reader source, dependencies, units,
and compact source references.  The checker reads representative bytes when possible and emits
stable JSON or a concise table.  It deliberately keeps unavailable readers visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA = "compute-window-schema-reader-inventory.v1"
STATUSES = (
    "readable_verified",
    "readable_with_declared_adapter",
    "schema_only",
    "reader_unavailable",
    "unversioned",
    "conflict",
)
FORMATS = {"json", "jsonl", "parquet-like", "trace", "snapshot", "migrated"}
SHA256_LENGTH = 64
REQUIRED_ROLE_FIELDS = {
    "role",
    "path",
    "format",
    "schema",
    "reader",
    "dependencies",
    "compatibility",
    "units_display",
    "source_material",
    "check_command",
}


def sha256_file(path: Path) -> str:
    """Return the content digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_matches(path: Path, expected: str) -> bool:
    """Return whether a regular file has the expected digest, failing closed on I/O errors."""
    try:
        return path.is_file() and sha256_file(path) == expected
    except OSError:
        return False


def _error(code: str, location: str, message: str) -> dict[str, str]:
    return {"code": code, "location": location, "message": message}


def _relative_path(value: Any, location: str, errors: list[dict[str, str]]) -> str | None:
    if not isinstance(value, str) or not value:
        errors.append(_error("invalid_path", location, "path must be non-empty"))
        return None
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in path.parts
        or path.as_posix() != value
        or "\\" in value
        or value.startswith("~")
        or any(ord(character) < 32 for character in value)
    ):
        errors.append(
            _error("hidden_absolute_path", location, "path must be normalized and relative")
        )
        return None
    return value


def _digest(value: Any, location: str, errors: list[dict[str, str]]) -> str | None:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        errors.append(_error("invalid_digest", location, "sha256 must be 64 hex characters"))
        return None
    try:
        int(value, 16)
    except ValueError:
        errors.append(_error("invalid_digest", location, "sha256 must be hexadecimal"))
        return None
    return value


def _representative_schema_version(payload: Mapping[str, Any]) -> Any:
    """Extract a conventional schema version from a representative payload."""
    version = payload.get("schema_version", payload.get("schema"))
    if isinstance(version, Mapping):
        version = version.get("version")
    if version is None and isinstance(payload.get("metadata"), Mapping):
        version = payload["metadata"].get("schema_version")
    return version


def _load_jsonl_payloads(text: str) -> tuple[list[Mapping[str, Any]] | None, str | None]:
    """Parse non-empty JSONL objects without loading an optional data engine."""
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None, f"JSONL line {line_number} is invalid"
        if not isinstance(row, Mapping):
            return None, f"JSONL line {line_number} must be an object"
        rows.append(row)
    if not rows:
        return None, "JSONL must contain at least one object row"
    return rows, None


def _load_representative_payloads(
    path: Path, fmt: str
) -> tuple[list[Mapping[str, Any]] | None, str | None]:
    """Read lightweight representative objects without optional data engines."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, "bytes are not readable as the declared lightweight fixture format"
    if fmt == "jsonl":
        return _load_jsonl_payloads(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, "representative bytes are not valid JSON"
    if not isinstance(payload, Mapping):
        return None, "representative JSON must be an object"
    return [payload], None


def _load_and_validate_bytes(
    path: Path, fmt: str, expected_version: str
) -> tuple[bool, str | None]:
    """Read lightweight representative bytes and require the exact schema version."""
    payloads, reason = _load_representative_payloads(path, fmt)
    if reason:
        return False, reason
    assert payloads is not None
    for payload in payloads:
        if _representative_schema_version(payload) != expected_version:
            return False, "representative bytes have a schema-version mismatch"
    return True, None


def _rooted_path(
    root: Path, relative: str | None, location: str, errors: list[dict[str, str]]
) -> Path | None:
    """Resolve a relative packet path and reject symlinks that escape the source root."""
    if relative is None:
        return None
    try:
        root_resolved = root.resolve()
        candidate = (root / relative).resolve()
        candidate.relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError):
        errors.append(_error("hidden_absolute_path", location, "path resolves outside source root"))
        return None
    return candidate


def _role_result(  # noqa: C901, PLR0912, PLR0915
    role: Mapping[str, Any], root: Path, index: int
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    location = f"roles[{index}]"
    missing = sorted(REQUIRED_ROLE_FIELDS - set(role))
    for field in missing:
        errors.append(_error("missing_field", f"{location}.{field}", "required field is missing"))
    if missing:
        return {
            "role": role.get("role", f"#{index}"),
            "status": "conflict",
            "errors": errors,
        }, errors
    name = role.get("role")
    path_value = _relative_path(role.get("path"), f"{location}.path", errors)
    fmt = role.get("format")
    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(fmt, str)
        or fmt not in FORMATS
    ):
        errors.append(_error("ambiguous_role", location, "role and supported format are required"))
    schema = role.get("schema")
    version = schema.get("version") if isinstance(schema, Mapping) else None
    if (
        not isinstance(schema, Mapping)
        or not isinstance(schema.get("name"), str)
        or not schema.get("name", "").strip()
    ):
        errors.append(_error("missing_schema", f"{location}.schema", "schema name is required"))
    if not isinstance(version, str) or not version.strip():
        errors.append(
            _error("unversioned", f"{location}.schema.version", "schema version is required")
        )
    reader = role.get("reader")
    reader_available = isinstance(reader, Mapping) and reader.get("available") is True
    reader_path = (
        _relative_path(reader.get("source_path"), f"{location}.reader.source_path", errors)
        if isinstance(reader, Mapping)
        else None
    )
    reader_digest = (
        _digest(reader.get("source_sha256"), f"{location}.reader.source_sha256", errors)
        if isinstance(reader, Mapping)
        else None
    )
    if (
        not isinstance(reader, Mapping)
        or not isinstance(reader.get("symbol"), str)
        or not reader.get("symbol", "").strip()
    ):
        errors.append(
            _error("reader_unavailable", f"{location}.reader", "reader symbol is required")
        )
    elif not reader_available:
        errors.append(
            _error("reader_unavailable", f"{location}.reader", "declared reader is unavailable")
        )
    elif reader.get("schema_version") is not None and reader.get("schema_version") != version:
        errors.append(
            _error(
                "reader_schema_mismatch",
                f"{location}.reader.schema_version",
                "reader version differs from schema",
            )
        )
    if reader_available and reader_path and reader_digest:
        source = _rooted_path(root, reader_path, f"{location}.reader.source_path", errors)
        if source is None or not _digest_matches(source, reader_digest):
            errors.append(
                _error(
                    "reader_schema_mismatch",
                    f"{location}.reader",
                    "reader source is missing or digest differs",
                )
            )
        else:
            try:
                source_text = source.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                errors.append(
                    _error(
                        "reader_schema_mismatch",
                        f"{location}.reader.source_path",
                        "reader source is not readable text",
                    )
                )
            else:
                if reader["symbol"] not in source_text:
                    errors.append(
                        _error(
                            "reader_schema_mismatch",
                            f"{location}.reader.symbol",
                            "reader symbol is absent from the bound source",
                        )
                    )
    dependencies = role.get("dependencies")
    if not isinstance(dependencies, list):
        errors.append(
            _error("missing_field", f"{location}.dependencies", "dependencies must be a list")
        )
        dependencies = []
    for dep_index, dep in enumerate(dependencies):
        if (
            not isinstance(dep, Mapping)
            or not isinstance(dep.get("name"), str)
            or not dep.get("name", "").strip()
            or not isinstance(dep.get("required"), bool)
            or not isinstance(dep.get("available"), bool)
        ):
            errors.append(
                _error(
                    "optional_dependency_gap",
                    f"{location}.dependencies[{dep_index}]",
                    "dependency name is required",
                )
            )
        elif dep["required"] is True and dep["available"] is not True:
            errors.append(
                _error(
                    "optional_dependency_gap",
                    f"{location}.dependencies[{dep_index}]",
                    "required dependency is unavailable",
                )
            )
    compatibility = role.get("compatibility")
    if not isinstance(compatibility, Mapping) or compatibility.get("mode") not in {
        "native",
        "declared_adapter",
        "none",
    }:
        errors.append(
            _error("stale_adapter", f"{location}.compatibility", "compatibility mode is required")
        )
    elif compatibility.get("mode") == "declared_adapter" and not compatibility.get(
        "adapter_symbol"
    ):
        errors.append(
            _error("stale_adapter", f"{location}.compatibility", "adapter symbol is required")
        )
    elif (
        compatibility.get("target_schema") is not None
        and compatibility.get("target_schema") != version
    ):
        errors.append(
            _error(
                "stale_adapter",
                f"{location}.compatibility.target_schema",
                "adapter target differs from schema",
            )
        )
    units_display = role.get("units_display")
    if not isinstance(units_display, Mapping):
        errors.append(
            _error(
                "unit_display_drift",
                f"{location}.units_display",
                "units/display metadata is required",
            )
        )
    elif not isinstance(schema, Mapping) or not isinstance(schema.get("units_display"), Mapping):
        errors.append(
            _error(
                "unit_display_drift",
                f"{location}.schema.units_display",
                "schema-bound units/display metadata is required",
            )
        )
    elif dict(schema["units_display"]) != dict(units_display):
        errors.append(
            _error(
                "unit_display_drift",
                f"{location}.units_display",
                "metadata differs from schema-bound units/display",
            )
        )
    check_command = role.get("check_command")
    if not isinstance(check_command, str) or not check_command.strip():
        errors.append(
            _error(
                "missing_check_command",
                f"{location}.check_command",
                "validation command is required",
            )
        )
    source_material = role.get("source_material")
    if not isinstance(source_material, list) or not source_material:
        errors.append(
            _error(
                "missing_source_material",
                f"{location}.source_material",
                "exact source reference is required",
            )
        )
    else:
        for ref_index, ref in enumerate(source_material):
            if not isinstance(ref, Mapping):
                errors.append(
                    _error(
                        "missing_source_material",
                        f"{location}.source_material[{ref_index}]",
                        "source reference must be an object",
                    )
                )
                continue
            ref_path = _relative_path(
                ref.get("path"), f"{location}.source_material[{ref_index}].path", errors
            )
            ref_digest = _digest(
                ref.get("sha256"), f"{location}.source_material[{ref_index}].sha256", errors
            )
            reference = _rooted_path(
                root,
                ref_path,
                f"{location}.source_material[{ref_index}].path",
                errors,
            )
            if reference and ref_digest and not _digest_matches(reference, ref_digest):
                errors.append(
                    _error(
                        "missing_source_material",
                        f"{location}.source_material[{ref_index}]",
                        "source reference is not content-addressed to this tree",
                    )
                )
    output_path = _rooted_path(root, path_value, f"{location}.path", errors)
    if output_path is None or not output_path.is_file():
        errors.append(
            _error("missing_output", f"{location}.path", "representative output is missing")
        )
    elif fmt in FORMATS and isinstance(version, str) and version.strip():
        readable, reason = _load_and_validate_bytes(output_path, fmt, version)
        if not readable:
            errors.append(
                _error(
                    "reader_schema_mismatch",
                    f"{location}.path",
                    reason or "representative bytes are unreadable",
                )
            )
    status = "readable_verified"
    if errors:
        codes = {item["code"] for item in errors}
        if "unversioned" in codes:
            status = "unversioned"
        elif codes & {
            "hidden_absolute_path",
            "reader_schema_mismatch",
            "optional_dependency_gap",
            "stale_adapter",
            "unit_display_drift",
            "ambiguous_role",
            "missing_check_command",
        }:
            status = "conflict"
        elif "reader_unavailable" in codes or not reader_available:
            status = (
                "reader_unavailable" if not errors or "reader_unavailable" in codes else "conflict"
            )
        elif "missing_output" in codes or "missing_source_material" in codes:
            status = "schema_only"
        else:
            status = "conflict"
    elif isinstance(compatibility, Mapping) and compatibility.get("mode") == "declared_adapter":
        status = "readable_with_declared_adapter"
    result = {
        "role": name,
        "path": path_value,
        "format": fmt,
        "schema": dict(schema) if isinstance(schema, Mapping) else {},
        "reader": dict(reader) if isinstance(reader, Mapping) else {},
        "check_command": check_command if isinstance(check_command, str) else None,
        "dependencies": sorted(
            (dict(dep) for dep in dependencies if isinstance(dep, Mapping)),
            key=lambda item: str(item.get("name", "")),
        ),
        "compatibility": dict(compatibility) if isinstance(compatibility, Mapping) else {},
        "units_display": dict(units_display) if isinstance(units_display, Mapping) else {},
        "source_material": sorted(
            (dict(ref) for ref in source_material if isinstance(ref, Mapping)),
            key=lambda item: str(item.get("path", "")),
        )
        if isinstance(source_material, list)
        else [],
        "status": status,
        "errors": errors,
    }
    return result, errors


def build_inventory(packet: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Validate and normalize one inventory packet."""
    errors: list[dict[str, str]] = []
    if packet.get("schema_version") != SCHEMA:
        errors.append(_error("wrong_schema", "schema_version", f"must equal {SCHEMA}"))
    if packet.get("issue") != 8861:
        errors.append(_error("wrong_issue", "issue", "packet must be bound to issue 8861"))
    roles = packet.get("roles")
    if not isinstance(roles, list) or not roles:
        errors.append(_error("missing_role", "roles", "at least one output role is required"))
        roles = []
    results = []
    seen: set[str] = set()
    for index, role in enumerate(roles):
        if not isinstance(role, Mapping):
            results.append(
                {
                    "role": f"#{index}",
                    "status": "conflict",
                    "errors": [
                        _error("ambiguous_role", f"roles[{index}]", "role must be an object")
                    ],
                }
            )
            continue
        normalized, role_errors = _role_result(role, root, index)
        role_name = str(normalized.get("role"))
        if role_name in seen:
            role_errors.append(_error("ambiguous_role", f"roles[{index}].role", "duplicate role"))
            normalized["status"] = "conflict"
            normalized["errors"] = role_errors
        seen.add(role_name)
        results.append(normalized)
    results.sort(key=lambda item: (str(item.get("role")), str(item.get("path"))))
    errors.extend(error for item in results for error in item.get("errors", []))
    errors.sort(key=lambda item: (item["location"], item["code"], item["message"]))
    return {
        "schema_version": SCHEMA,
        "issue": packet.get("issue"),
        "status": "ok" if not errors else "blocked",
        "ok": not errors,
        "roles": results,
        "errors": errors,
    }


def load_packet(path: Path) -> Mapping[str, Any]:
    """Load one JSON inventory packet."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("packet must be a JSON object")
    return payload


def render_table(report: Mapping[str, Any]) -> str:
    """Render the stable human-readable role summary."""

    def cell(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")

    lines = ["role | format | schema | status", "--- | --- | --- | ---"]
    for role in report.get("roles", []):
        schema = role.get("schema") if isinstance(role.get("schema"), Mapping) else {}
        lines.append(
            f"{cell(role.get('role', '?'))} | {cell(role.get('format', '?'))} | "
            f"{cell(schema.get('name', '?'))}@{cell(schema.get('version', '?'))} | "
            f"{cell(role.get('status', 'conflict'))}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the inventory checker CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=None, help="packet source root; defaults to packet parent"
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--format", choices=("json", "table"), default="json")
    args = parser.parse_args(argv)
    try:
        report = build_inventory(load_packet(args.packet), args.root or args.packet.parent)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "blocked",
                    "ok": False,
                    "errors": [_error("malformed_packet", "packet", str(exc))],
                },
                sort_keys=True,
            ),
            file=sys.stdout,
        )
        return 3
    output = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_table(report)
    )
    print(output, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
