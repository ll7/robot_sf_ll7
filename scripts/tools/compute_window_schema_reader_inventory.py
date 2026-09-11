"""Inventory compute-window schemas/readers without converting outputs."""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA = "compute-window-schema-reader-inventory.v1"
FORMATS = {"json", "jsonl", "parquet-like", "trace", "snapshot", "migrated"}
SHA256_LENGTH = 64
REQUIRED_ROLE_FIELDS = {
    *"role path format schema reader dependencies compatibility units_display source_material check_command".split()
}
CONFLICT_CODES = set(
    "hidden_absolute_path reader_schema_mismatch optional_dependency_gap stale_adapter "
    "unit_display_drift ambiguous_role missing_check_command source_material_mismatch "
    "schema_source_mismatch".split()
)
_READER_CHILD = """import importlib.util,sys;from pathlib import Path
s=importlib.util.spec_from_file_location('r',sys.argv[1]);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m
s.loader.exec_module(m);getattr(m,sys.argv[2])(Path(sys.argv[3]))"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_matches(path: Path, expected: str) -> bool:
    try:
        return path.is_file() and _sha256_file(path) == expected
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
    invalid = (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in path.parts
        or path.as_posix() != value
        or "\\" in value
        or value.startswith("~")
        or any(ord(character) < 32 for character in value)
    )
    if invalid:
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
    version = payload.get("schema_version", payload.get("schema", payload.get("version")))
    if isinstance(version, Mapping):
        version = version.get("version")
    if version is None and isinstance(payload.get("metadata"), Mapping):
        version = payload["metadata"].get("schema_version")
    return version


def _load_jsonl_payloads(text: str) -> tuple[list[Mapping[str, Any]] | None, str | None]:
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
    payloads, reason = _load_representative_payloads(path, fmt)
    if reason:
        return False, reason
    valid = all(
        _representative_schema_version(payload) == expected_version for payload in payloads or []
    )
    return valid, None if valid else "representative bytes have a schema-version mismatch"


def _rooted_path(
    root: Path, relative: str | None, location: str, errors: list[dict[str, str]]
) -> Path | None:
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


def _execute_reader(
    root: Path, output: Path, reader: Mapping[str, Any], location: str
) -> dict[str, str] | None:
    source = _rooted_path(root, str(reader["source_path"]), location, [])
    if source is None or not _digest_matches(source, str(reader["source_sha256"])):
        return _error("reader_schema_mismatch", location, "reader source binding is invalid")
    if reader.get("execution") != "python_path":
        return _error("reader_unavailable", f"{location}.execution", "reader hook is missing")
    try:
        process = subprocess.run(
            [sys.executable, "-c", _READER_CHILD, str(source), str(reader["symbol"]), str(output)],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        return _error("reader_schema_mismatch", location, "reader timed out")
    except OSError:
        return _error("reader_schema_mismatch", location, "reader execution could not start")
    if process.returncode != 0:
        return _error("reader_schema_mismatch", location, "reader rejected: child process failed")
    return None


def _source_declares_schema(text: str, suffix: str, name: str, version: str) -> bool:
    if suffix == ".json":
        content = json.loads(text)
        properties = content.get("properties", {}) if isinstance(content, Mapping) else {}
        field = (
            properties.get("schema_version", properties.get("version", {}))
            if isinstance(properties, Mapping)
            else {}
        )
        return (
            str(content.get("$id", "") if isinstance(content, Mapping) else "")
            .rsplit("/", 1)[-1]
            .removesuffix(".json")
            == name
            and isinstance(field, Mapping)
            and field.get("const") == version
        )
    if suffix != ".py" or not (version in {name, f"{name}.v1"} or version.endswith(f".{name}.v1")):
        return False
    return any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and not node.col_offset
        and isinstance(node.value, ast.Constant)
        and node.value.value == version
        and any(
            isinstance(target, ast.Name) and target.id.removesuffix("_VERSION").endswith("SCHEMA")
            for target in (getattr(node, "targets", None) or [node.target])
        )
        for node in ast.walk(ast.parse(text))
    )


def _schema_source_error(
    root: Path,
    path: str | None,
    name: Any,
    version: Any,
    location: str,
) -> dict[str, str] | None:
    if not path or not isinstance(name, str) or not isinstance(version, str):
        return None
    source = _rooted_path(root, path, location, [])
    if source is None:
        return _error("schema_source_mismatch", location, "source is outside root")
    try:
        valid = _source_declares_schema(
            source.read_text(encoding="utf-8"), source.suffix, name, version
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SyntaxError):
        return _error("schema_source_mismatch", location, "source is unreadable")
    return (
        None
        if valid
        else _error("schema_source_mismatch", location, "source does not declare name/version")
    )


def _source_material_errors(
    source_material: Any,
    expected: Mapping[str, tuple[str | None, str | None]],
    root: Path,
    location: str,
) -> list[dict[str, str]]:
    if not isinstance(source_material, list) or not source_material:
        return [_error("missing_source_material", location, "exact source reference is required")]
    errors: list[dict[str, str]] = []
    add = errors.append
    present: set[str] = set()
    for ref_index, ref in enumerate(source_material):
        ref_location = f"{location}[{ref_index}]"
        if not isinstance(ref, Mapping):
            add(_error("missing_source_material", ref_location, "source ref must be an object"))
            continue
        ref_path = _relative_path(ref.get("path"), f"{ref_location}.path", errors)
        ref_digest = _digest(ref.get("sha256"), f"{ref_location}.sha256", errors)
        kind = ref.get("kind")
        if not isinstance(kind, str) or kind not in expected:
            add(_error("missing_source_material", f"{ref_location}.kind", "unknown source kind"))
        elif (ref_path, ref_digest) != expected[kind]:
            add(_error("source_material_mismatch", ref_location, "source binding differs"))
        else:
            present.add(kind)
        reference = _rooted_path(root, ref_path, f"{ref_location}.path", errors)
        if reference and ref_digest and not _digest_matches(reference, ref_digest):
            add(_error("missing_source_material", ref_location, "source digest mismatch"))
    for kind in sorted(set(expected) - present):
        add(_error("missing_source_material", location, f"{kind} source reference is required"))
    return errors


def _role_result(  # noqa: C901, PLR0912, PLR0915
    role: Mapping[str, Any], root: Path, index: int
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    location = f"roles[{index}]"

    def add(code: str, suffix: str, message: str) -> None:
        errors.append(_error(code, f"{location}.{suffix}" if suffix else location, message))

    missing = sorted(REQUIRED_ROLE_FIELDS - set(role))
    for field in missing:
        add("missing_field", field, "required field is missing")
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
        add("ambiguous_role", "", "role and supported format are required")
    schema = role.get("schema")
    version = schema.get("version") if isinstance(schema, Mapping) else None
    schema_path = (
        _relative_path(schema.get("source_path"), f"{location}.schema.source_path", errors)
        if isinstance(schema, Mapping)
        else None
    )
    schema_digest = (
        _digest(schema.get("source_sha256"), f"{location}.schema.source_sha256", errors)
        if isinstance(schema, Mapping)
        else None
    )
    if (
        not isinstance(schema, Mapping)
        or not isinstance(schema.get("name"), str)
        or not schema.get("name", "").strip()
    ):
        add("missing_schema", "schema", "schema name is required")
    if not isinstance(version, str) or not version.strip():
        add("unversioned", "schema.version", "schema version is required")
    schema_error = _schema_source_error(
        root,
        schema_path,
        schema.get("name") if isinstance(schema, Mapping) else None,
        version,
        f"{location}.schema",
    )
    if schema_error:
        errors.append(schema_error)
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
        add("reader_unavailable", "reader", "reader symbol is required")
    elif not reader_available:
        add("reader_unavailable", "reader", "declared reader is unavailable")
    elif reader.get("schema_version") is not None and reader.get("schema_version") != version:
        add("reader_schema_mismatch", "reader.schema_version", "reader version differs from schema")
    dependencies = role.get("dependencies")
    if not isinstance(dependencies, list):
        add("missing_field", "dependencies", "dependencies must be a list")
        dependencies = []
    for dep_index, dep in enumerate(dependencies):
        invalid = (
            not isinstance(dep, Mapping)
            or not isinstance(dep.get("name"), str)
            or not dep.get("name", "").strip()
            or not isinstance(dep.get("required"), bool)
            or not isinstance(dep.get("available"), bool)
        )
        if invalid or (dep["required"] is True and dep["available"] is not True):
            message = (
                "dependency name is required" if invalid else "required dependency is unavailable"
            )
            add("optional_dependency_gap", f"dependencies[{dep_index}]", message)
    compatibility = role.get("compatibility")
    adapter_path = adapter_digest = None
    mode = compatibility.get("mode") if isinstance(compatibility, Mapping) else None
    if mode not in {
        "native",
        "declared_adapter",
        "none",
    }:
        add("stale_adapter", "compatibility", "compatibility mode is required")
    elif mode == "declared_adapter" and not compatibility.get("adapter_symbol"):
        add("stale_adapter", "compatibility", "adapter symbol is required")
    elif mode == "declared_adapter":
        adapter_path = _relative_path(
            compatibility.get("adapter_source_path"),
            f"{location}.compatibility.adapter_source_path",
            errors,
        )
        adapter_digest = _digest(
            compatibility.get("adapter_source_sha256"),
            f"{location}.compatibility.adapter_source_sha256",
            errors,
        )
        if not adapter_path or not adapter_digest:
            add("stale_adapter", "compatibility", "adapter source path and digest are required")
    if isinstance(compatibility, Mapping) and compatibility.get("target_schema") not in {
        None,
        version,
    }:
        add("stale_adapter", "compatibility.target_schema", "adapter target differs from schema")
    units_display = role.get("units_display")
    if not isinstance(units_display, Mapping):
        add("unit_display_drift", "units_display", "units/display metadata is required")
    elif not isinstance(schema, Mapping) or not isinstance(schema.get("units_display"), Mapping):
        add(
            "unit_display_drift",
            "schema.units_display",
            "schema-bound units/display metadata is required",
        )
    elif dict(schema["units_display"]) != dict(units_display):
        add(
            "unit_display_drift",
            "units_display",
            "metadata differs from schema-bound units/display",
        )
    check_command = role.get("check_command")
    if not isinstance(check_command, str) or not check_command.strip():
        add("missing_check_command", "check_command", "validation command is required")
    expected_material: dict[str, tuple[str | None, str | None]] = {
        "schema": (schema_path, schema_digest)
    }
    if reader_path and reader_digest:
        expected_material["reader"] = (reader_path, reader_digest)
    if adapter_path and adapter_digest:
        expected_material["adapter"] = (adapter_path, adapter_digest)
    source_material = role.get("source_material")
    errors.extend(
        _source_material_errors(
            source_material, expected_material, root, f"{location}.source_material"
        )
    )
    output_path = _rooted_path(root, path_value, f"{location}.path", errors)
    if output_path is None or not output_path.is_file():
        add("missing_output", "path", "representative output is missing")
    elif fmt in FORMATS and isinstance(version, str) and version.strip():
        readable, reason = _load_and_validate_bytes(output_path, fmt, version)
        if not readable:
            add("reader_schema_mismatch", "path", reason or "representative bytes are unreadable")
        elif reader_available and reader_path and reader_digest:
            reader_error = _execute_reader(root, output_path, reader, f"{location}.reader")
            if reader_error:
                errors.append(reader_error)
    status = "readable_verified"
    if errors:
        codes = {item["code"] for item in errors}
        if "unversioned" in codes:
            status = "unversioned"
        elif codes & CONFLICT_CODES:
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
            key=lambda item: (str(item.get("kind", "")), str(item.get("path", ""))),
        )
        if isinstance(source_material, list)
        else [],
        "status": status,
        "errors": errors,
    }
    return result, errors


def build_inventory(packet: Mapping[str, Any], root: Path) -> dict[str, Any]:  # noqa: D103
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
            error = _error("ambiguous_role", f"roles[{index}]", "role must be an object")
            results.append({"role": f"#{index}", "status": "conflict", "errors": [error]})
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


def load_packet(path: Path) -> Mapping[str, Any]:  # noqa: D103
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("packet must be a JSON object")
    return payload


def render_table(report: Mapping[str, Any]) -> str:  # noqa: D103
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


def main(argv: list[str] | None = None) -> int:  # noqa: D103
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
