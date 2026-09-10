#!/usr/bin/env python3
"""Compare captured execution environments across hosts and flag inequivalent factors.

A host manifest is a pseudonymous JSON object (``schema_version``, ``host_label``)
with nested sections such as ``os``, ``python``, ``distributions``, ``editable``,
``native_extensions``, ``threading``, ``accelerators``, ``locale``, ``filesystem``,
``environment``, ``scheduler``, and ``container``.  Additional sections compare too.

Every leaf field is classified as exactly one of ``exact_match``,
``compatible_declared``, ``different_material``, ``different_unclassified``,
``unavailable``, ``redacted_not_comparable``, or ``not_applicable``.  Materiality
comes from the workload requirements manifest, never from version-prefix
heuristics: equal package versions with different native builds stay material
unless the workload declares the variation with an explicit rationale.  Missing,
redacted, unknown, or undeclared differences fail closed and block a cross-host
reproducibility claim.  Environment equivalence never implies output equivalence;
optional empirical repeat evidence stays separate.  Reports never emit home
paths, IPs, or user@host strings.

    uv run python scripts/validation/compare_execution_environments.py \
        --host-a host_a.json --host-b host_b.json --requirements requirements.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HOST_SCHEMA_VERSION = "cross_host_environment.v1"
REQUIREMENTS_SCHEMA_VERSION = "cross_host_workload_requirements.v1"
REPORT_SCHEMA_VERSION = "cross_host_environment_comparison.v1"
CLASSIFICATIONS = (
    "exact_match compatible_declared different_material different_unclassified "
    "unavailable redacted_not_comparable not_applicable".split()
)
BLOCKING_CLASSIFICATIONS = frozenset(CLASSIFICATIONS[2:6])
CLAIM_BOUNDARY = (
    "Environment equivalence is scoped to the declared workload requirements and does not "
    "establish output equivalence; empirical repeat evidence stays separate."
)
REDACTED_MARKERS = {"<redacted>", "redacted", "***", "[redacted]", "<private>", "<secret>"}
UNSET_MARKERS = {"", "unset", "default", "auto", "not_set"}
TRUEISH_MARKERS = {"true", "yes", "1"}
THREAD_LIMIT_FIELDS = frozenset(
    ["threading/omp_num_threads", "threading/blas_num_threads", "threading/mkl_num_threads"]
    + [f"environment/{name}_NUM_THREADS" for name in ("OMP", "BLAS", "MKL", "NUMEXPR", "OPENBLAS")]
)
PLATFORM_MARKER_FIELDS = frozenset(
    "os/system os/architecture_class python/implementation python/abi_tag "
    "filesystem/path_separator filesystem/case_sensitive".split()
)
PRIVATE_KEY_NAMES = frozenset(
    "hostname fqdn username user home homedir homepath ip ipaddress macaddress "
    "serialnumber sshkey email".split()
)
METADATA_KEYS = frozenset({"schema_version", "host_label"})
REQUIREMENT_KEYS = frozenset(
    "schema_version workload_id description not_applicable_fields material_fields "
    "compatible_variations".split()
)
DECLARATION_KEYS = {"field", "host_a_values", "host_b_values", "rationale"}
HOST_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f]")
_HOME_RE = re.compile(r"(?P<prefix>/(?:home|Users)/)(?P<user>[^/\s\"']+)")
_WIN_HOME_RE = re.compile(r"(?i)(?P<prefix>[a-z]:\\users\\)(?P<user>[^\\\s\"']+)")
_USER_AT_HOST_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){3,}[0-9a-fA-F]{0,4}\b|\b[0-9a-fA-F:]*::[0-9a-fA-F:]*\b"
)


class ContractError(ValueError):
    """Raised when a manifest, requirement set, or comparison cannot be trusted."""


@dataclass(frozen=True)
class Declaration:
    """One declared compatible difference with its explicit rationale."""

    field: str
    host_a_values: tuple[str, ...]
    host_b_values: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class Requirements:
    """Workload capability requirements that scope a cross-host comparison."""

    workload_id: str
    not_applicable_fields: frozenset[str]
    material_fields: frozenset[str]
    declarations: Mapping[str, Declaration]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read {path.name}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path.name}: {exc}") from exc


def sanitize_text(text: str) -> str:
    """Replace private-identity patterns with stable placeholders."""
    text = _HOME_RE.sub(lambda match: match.group("prefix") + "<user>", text)
    text = _WIN_HOME_RE.sub(lambda match: match.group("prefix") + "<user>", text)
    text = _USER_AT_HOST_RE.sub("<user>@<host>", text)
    return _IPV6_RE.sub("<ip>", _IPV4_RE.sub("<ip>", text))


def _display(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    return [_display(item) for item in value] if isinstance(value, list) else value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_node(value: Any, location: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(isinstance(key, str) and bool(key), f"non-string key at {location}")
            _require("/" not in key, f"invalid manifest key: {location}{key}")
            _require(CONTROL_CHARACTER_RE.search(key) is None, f"control char in key: {key}")
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            _require(normalized not in PRIVATE_KEY_NAMES, f"private identity key: {location}{key}")
            _validate_node(item, f"{location}{key}.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_node(item, f"{location}{index}.")
        return
    if isinstance(value, str):
        _require(CONTROL_CHARACTER_RE.search(value) is None, f"control char in value: {location}")


def parse_host_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return one pseudonymous host environment manifest."""
    _require(isinstance(payload, Mapping), "host manifest must be a JSON object")
    _require(payload.get("schema_version") == HOST_SCHEMA_VERSION, "host schema mismatch")
    label = payload.get("host_label")
    _require(
        isinstance(label, str) and HOST_LABEL_RE.fullmatch(label) is not None,
        "host_label must be a stable lowercase pseudonym",
    )
    _validate_node(payload)
    return dict(payload)


def _field_list(raw: Any, name: str) -> frozenset[str]:
    _require(isinstance(raw, list), f"{name} must be a list")
    for entry in raw:
        _require(
            isinstance(entry, str) and bool(entry) and entry == entry.strip("/"),
            f"{name} entries must be non-empty field paths",
        )
        _require("//" not in entry, f"{name} entry has an empty segment: {entry}")
        _require(CONTROL_CHARACTER_RE.search(entry) is None, f"{name} entry has a control char")
    return frozenset(raw)


def _matches_field_set(path: str, prefixes: frozenset[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


def _value_list(raw: Any, field: str, name: str) -> tuple[str, ...]:
    _require(isinstance(raw, list) and bool(raw), f"declaration {field} needs {name}")
    return tuple(_canonical(value) for value in raw)


def _parse_declaration(entry: Any, index: int) -> Declaration:
    _require(isinstance(entry, Mapping), f"compatible_variations[{index}] must be a mapping")
    unknown = sorted(set(entry) - DECLARATION_KEYS)
    _require(not unknown, f"unknown declaration keys: {unknown}")
    field = entry.get("field")
    _require(isinstance(field, str) and bool(field), f"declaration {index} needs a field")
    rationale = entry.get("rationale")
    _require(
        isinstance(rationale, str) and bool(rationale.strip()),
        f"declaration for {field} needs a rationale",
    )
    a_values = _value_list(entry.get("host_a_values"), field, "host_a_values")
    b_values = _value_list(entry.get("host_b_values"), field, "host_b_values")
    return Declaration(field, a_values, b_values, rationale)


def parse_requirements(payload: Mapping[str, Any]) -> Requirements:
    """Validate and return one workload capability requirement set."""
    _require(isinstance(payload, Mapping), "requirements must be a JSON object")
    _require(
        payload.get("schema_version") == REQUIREMENTS_SCHEMA_VERSION,
        "requirements schema mismatch",
    )
    unknown = sorted(set(payload) - REQUIREMENT_KEYS)
    _require(not unknown, f"unknown requirements keys: {unknown}")
    workload_id = payload.get("workload_id")
    _require(isinstance(workload_id, str) and bool(workload_id), "workload_id is required")
    not_applicable = _field_list(payload.get("not_applicable_fields", []), "not_applicable_fields")
    material = _field_list(payload.get("material_fields", []), "material_fields")
    for field in material:
        _require(not _matches_field_set(field, not_applicable), f"ambiguous material: {field}")
    for field in not_applicable:
        _require(not _matches_field_set(field, material), f"ambiguous NA: {field}")
    declarations: dict[str, Declaration] = {}
    for index, entry in enumerate(payload.get("compatible_variations") or []):
        declaration = _parse_declaration(entry, index)
        _require(
            not _matches_field_set(declaration.field, not_applicable),
            f"declared field is not_applicable: {declaration.field}",
        )
        _require(declaration.field not in declarations, "duplicate declaration")
        declarations[declaration.field] = declaration
    return Requirements(workload_id, not_applicable, material, declarations)


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in sorted(payload):
        if key in METADATA_KEYS:
            continue
        value = payload[key]
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            fields.update(_flatten(value, f"{path}/"))
        else:
            fields[path] = value
    return fields


def _is_redacted(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in REDACTED_MARKERS


def _is_unset(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in UNSET_MARKERS)


def _truthy(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in TRUEISH_MARKERS)


def _is_build_provenance_field(path: str) -> bool:
    if path.startswith("native_extensions/"):
        return True
    if path.startswith(("editable/", "container/")):
        return path.endswith(("/commit", "/dirty", "/sha256", "/image_digest", "/image_sha256"))
    return path.startswith("distributions/") and path.endswith(("/build", "/wheel_sha256"))


def _detection(kind: str, field: str, detail: str) -> dict[str, str]:
    return {"kind": kind, "field": field, "detail": detail}


def _make_row(
    path: str,
    classification: str,
    value_a: Any,
    value_b: Any,
    rationale: str | None,
    declaration_rationale: str | None = None,
) -> dict[str, Any]:
    return {
        "field": path,
        "classification": classification,
        "host_a_value": _display(value_a),
        "host_b_value": _display(value_b),
        "rationale": rationale,
        "declaration_rationale": declaration_rationale,
    }


def _difference_detections(
    path: str, value_a: Any, value_b: Any, declaration: Declaration | None
) -> list[dict[str, str]]:
    detections: list[dict[str, str]] = []
    if path.endswith("/dirty") and (_truthy(value_a) or _truthy(value_b)):
        detections.append(_detection("dirty_editable_package", path, "editable source differs"))
    if path in PLATFORM_MARKER_FIELDS:
        detections.append(_detection("platform_marker_differs", path, "platform marker differs"))
    if declaration is not None:
        detections.append(_detection("declaration_not_satisfied", path, "observed pair undeclared"))
    if path.startswith("native_extensions/") and path.endswith("/sha256"):
        detail = "native build differs"
        detections.append(_detection("source_identical_build_different", path, detail))
    return detections


def _classify_difference(
    path: str, value_a: Any, value_b: Any, requirements: Requirements
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    declaration = requirements.declarations.get(path)
    detections = _difference_detections(path, value_a, value_b, declaration)
    if declaration is not None and _declaration_covers(declaration, value_a, value_b):
        row = _make_row(
            path,
            "compatible_declared",
            value_a,
            value_b,
            "declared_compatible",
            declaration.rationale,
        )
        return row, detections
    if _is_build_provenance_field(path):
        classification, rationale = "different_material", "build_provenance"
    elif _matches_field_set(path, requirements.material_fields):
        classification, rationale = "different_material", "workload_material"
    else:
        classification, rationale = "different_unclassified", "undeclared_difference"
    return _make_row(path, classification, value_a, value_b, rationale), detections


def _declaration_covers(declaration: Declaration, value_a: Any, value_b: Any) -> bool:
    return (
        _canonical(value_a) in declaration.host_a_values
        and _canonical(value_b) in declaration.host_b_values
    )


def _classify_field(
    path: str,
    fields_a: Mapping[str, Any],
    fields_b: Mapping[str, Any],
    requirements: Requirements,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if _matches_field_set(path, requirements.not_applicable_fields):
        row = _make_row(path, "not_applicable", None, None, "declared_not_applicable")
        return row, []
    if path not in fields_a or path not in fields_b:
        if path in fields_a:
            detail = "missing_on_host_b"
        elif path in fields_b:
            detail = "missing_on_host_a"
        else:
            detail = "missing_on_both"
        row = _make_row(path, "unavailable", fields_a.get(path), fields_b.get(path), detail)
        return row, [_detection("missing_field", path, detail)]
    value_a = fields_a[path]
    value_b = fields_b[path]
    if _is_redacted(value_a) or _is_redacted(value_b):
        detail = "value redacted on at least one host"
        row = _make_row(path, "redacted_not_comparable", value_a, value_b, "redacted_value")
        return row, [_detection("redacted_field", path, detail)]
    if path in THREAD_LIMIT_FIELDS and (_is_unset(value_a) or _is_unset(value_b)):
        row = _make_row(path, "unavailable", value_a, value_b, "hidden_thread_default")
        return row, []
    if _canonical(value_a) == _canonical(value_b):
        if path.endswith("/dirty") and _truthy(value_a):
            row = _make_row(path, "unavailable", value_a, value_b, "dirty_editable_source")
            return row, [_detection("dirty_editable_package", path, "editable source is dirty")]
        return _make_row(path, "exact_match", value_a, value_b, None), []
    return _classify_difference(path, value_a, value_b, requirements)


def _named_leaf(fields: Mapping[str, Any], section: str, leaf: str) -> dict[str, Any]:
    named: dict[str, Any] = {}
    for path, value in fields.items():
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == section and parts[2] == leaf:
            named[parts[1]] = value
    return named


def _cross_field_detections(
    fields_a: Mapping[str, Any], fields_b: Mapping[str, Any]
) -> list[dict[str, str]]:
    detections: list[dict[str, str]] = []
    versions_a = _named_leaf(fields_a, "distributions", "version")
    versions_b = _named_leaf(fields_b, "distributions", "version")
    builds_a = _named_leaf(fields_a, "distributions", "build")
    builds_b = _named_leaf(fields_b, "distributions", "build")
    for name in sorted(set(versions_a) & set(versions_b)):
        if name not in builds_a or name not in builds_b:
            continue
        same_version = _canonical(versions_a[name]) == _canonical(versions_b[name])
        if same_version and _canonical(builds_a[name]) != _canonical(builds_b[name]):
            detail = f"{name} shares version {_canonical(versions_a[name])} but builds differ"
            kind = "source_identical_build_different"
            detections.append(_detection(kind, f"distributions/{name}/build", detail))
    digest_path = "container/image_digest"
    digest_a = fields_a.get(digest_path)
    digest_b = fields_b.get(digest_path)
    tag = fields_a.get("container/image_tag") or fields_b.get("container/image_tag")
    if tag and (not digest_a or not digest_b or _is_unset(digest_a) or _is_unset(digest_b)):
        detail = "container tag present without a comparable image digest"
        detections.append(_detection("mutable_container_tag", digest_path, detail))
    return detections


def compare_environments(
    host_a: Mapping[str, Any],
    host_b: Mapping[str, Any],
    requirements: Requirements,
    repeat_evidence: Any = None,
) -> dict[str, Any]:
    """Compare two validated host manifests under one workload requirement set."""
    parsed_a = parse_host_manifest(host_a)
    parsed_b = parse_host_manifest(host_b)
    fields_a = _flatten(parsed_a)
    fields_b = _flatten(parsed_b)
    rows: list[dict[str, Any]] = []
    detections: list[dict[str, str]] = []
    for path in sorted(set(fields_a) | set(fields_b)):
        row, row_detections = _classify_field(path, fields_a, fields_b, requirements)
        rows.append(row)
        detections.extend(row_detections)
    detections.extend(_cross_field_detections(fields_a, fields_b))
    detections.sort(key=lambda item: (item["kind"], item["field"], item["detail"]))
    counts = dict.fromkeys(CLASSIFICATIONS, 0)
    for row in rows:
        counts[row["classification"]] += 1
    if counts["unavailable"] or counts["redacted_not_comparable"]:
        status = "not_comparable"
    elif any(counts[name] for name in BLOCKING_CLASSIFICATIONS):
        status = "not_equivalent"
    else:
        status = "equivalent_for_workload"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "workload_id": requirements.workload_id,
        "host_a_label": parsed_a["host_label"],
        "host_b_label": parsed_b["host_label"],
        "comparison_status": status,
        "can_claim_cross_host_reproducibility": status == "equivalent_for_workload",
        "output_equivalence_inferred": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "classification_counts": counts,
        "fields": rows,
        "detections": detections,
        "declared_compatible_differences": [
            {"field": row["field"], "rationale": row["declaration_rationale"]}
            for row in rows
            if row["classification"] == "compatible_declared"
        ],
        "empirical_repeat_evidence": repeat_evidence,
    }


def _cell(value: Any) -> str:
    return json.dumps(_display(value), ensure_ascii=True).replace("|", "\\|")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render one comparison report as concise deterministic Markdown."""
    counts = report["classification_counts"]
    lines = [
        "# Cross-host execution environment comparison",
        "",
        f"- workload: `{report['workload_id']}`",
        f"- host A: `{report['host_a_label']}`",
        f"- host B: `{report['host_b_label']}`",
        f"- status: `{report['comparison_status']}`",
        f"- cross-host reproducibility claim: `{report['can_claim_cross_host_reproducibility']}`",
        f"- claim boundary: {report['claim_boundary']}",
        "",
        "## Classification counts",
        "",
        "| classification | count |",
        "| --- | --- |",
    ]
    lines += [f"| `{name}` | {counts[name]} |" for name in CLASSIFICATIONS]
    non_exact = [row for row in report["fields"] if row["classification"] != "exact_match"]
    lines += ["", "## Non-matching fields", ""]
    if non_exact:
        lines += [
            "| field | host A | host B | classification | rationale |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in non_exact:
            rationale = row["declaration_rationale"] or row["rationale"]
            lines.append(
                f"| `{row['field']}` | `{_cell(row['host_a_value'])}` "
                f"| `{_cell(row['host_b_value'])}` | `{row['classification']}` "
                f"| {_cell(rationale)} |"
            )
    else:
        lines.append("All compared fields are exact matches.")
    lines += ["", "## Detections", ""]
    lines += [
        f"- `{item['kind']}` `{item['field']}`: {sanitize_text(item['detail'])}"
        for item in report["detections"]
    ] or ["No non-equivalence factors detected."]
    included = report["empirical_repeat_evidence"] is not None
    lines += ["", f"- empirical repeat evidence included: `{str(included).lower()}` (separate)"]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-a", type=Path, required=True, help="Host A manifest JSON.")
    parser.add_argument("--host-b", type=Path, required=True, help="Host B manifest JSON.")
    parser.add_argument(
        "--requirements", type=Path, required=True, help="Workload requirements JSON."
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparator and return 0 only for an equivalent-for-workload result."""
    args = _parser().parse_args(argv)
    try:
        report = compare_environments(
            parse_host_manifest(_load_json(args.host_a)),
            parse_host_manifest(_load_json(args.host_b)),
            parse_requirements(_load_json(args.requirements)),
        )
    except ContractError as exc:
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "comparison_status": "malformed",
            "error": sanitize_text(str(exc)),
        }
    if args.format == "markdown" and report["comparison_status"] != "malformed":
        rendered = render_markdown(report)
    else:
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)
    return 0 if report["comparison_status"] == "equivalent_for_workload" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
