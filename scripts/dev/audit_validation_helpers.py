#!/usr/bin/env python3
"""AST-backed inventory of validation/coercion helpers (issue #7900).

Parses Python with ``ast`` without importing application modules, finds
private/public helper definitions implementing finite-number, string,
mapping, path, enum, sequence, or required/optional validation/coercion
contracts, and records structural features plus call sites.  Genuinely
behavior-identical definitions (identical signatures/defaults and normalized
AST bodies) are partitioned into candidate clusters; similar purpose is not
equivalence.  No production callers are migrated.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "validation_helper_inventory.v1"
DEFAULT_ROOT = "robot_sf"
MIN_CLUSTER_SIZE = 3

#: Feature trigger words for classifying a helper as validation/coercion.
_VALIDATION_NAME_HINTS = (
    "validate",
    "coerce",
    "parse",
    "check",
    "require",
    "normalize",
    "as_float",
    "as_int",
    "as_str",
    "to_float",
    "to_int",
    "finite",
    "mapping",
    "nonempty",
    "non_empty",
)
_VALIDATION_BODY_HINTS = (
    "isinstance",
    "ValueError",
    "TypeError",
    "isfinite",
    "strip",
    "float(",
    "int(",
    "raise ",
    "return None",
)


@dataclass(frozen=True)
class HelperRecord:
    """One validation/coercion helper record."""

    module: str
    qualified_name: str
    signature: str
    normalized_body_hash: str
    source_digest: str
    return_paths: int
    raises: tuple[str, ...]
    call_sites: int
    layer: str
    features: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


def _digest(text: str) -> str:
    """Return the SHA-256 digest of a text payload."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_validation_helper(name: str, body: str) -> bool:
    """Heuristic: name hints or body hints indicate a validation/coercion helper."""
    lowered_name = name.lower()
    if any(hint in lowered_name for hint in _VALIDATION_NAME_HINTS):
        return True
    lowered_body = body.lower()
    return any(hint.lower() in lowered_body for hint in _VALIDATION_BODY_HINTS)


def _signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render the function signature deterministically."""
    args = node.args
    parts: list[str] = []
    for arg in [*args.posonlyargs, *args.args]:
        parts.append(arg.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for arg in args.kwonlyargs:
        parts.append(arg.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    defaults = len(args.defaults)
    if defaults:
        parts.append(f"defaults={defaults}")
    return f"({', '.join(parts)})"


def _normalized_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a normalized AST-body dump (names/constants intact, formatting-free)."""
    return ast.dump(ast.Module(body=node.body, type_ignores=[]), indent=None)


def _features_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Derive statically provable semantic features; unknown otherwise."""
    text = ast.unparse(node)
    return {
        "none_policy": "unknown",
        "bool_policy": "unknown",
        "coercion": "unknown",
        "strips_whitespace": "strip(" in text,
        "empty_string_policy": "unknown",
        "non_finite_handling": "isfinite" in text,
        "object_float_behavior": "unknown",
        "mapping_subclass_policy": "unknown",
        "exception_type": _exception_types(text),
        "error_aggregation": "unknown",
    }


def _exception_types(text: str) -> list[str]:
    """List exception types referenced in the body."""
    found: list[str] = []
    for name in ("ValueError", "TypeError", "KeyError", "RuntimeError", "AssertionError"):
        if name in text:
            found.append(name)
    return found


def _layer(module: str) -> str:
    """Classify the dependency layer from the module path."""
    if module.startswith("robot_sf/common"):
        return "common"
    if module.startswith("robot_sf/benchmark"):
        return "benchmark"
    if module.startswith("robot_sf/planner"):
        return "planner"
    if module.startswith("robot_sf/sim"):
        return "sim"
    if module.startswith("robot_sf/nav"):
        return "nav"
    return "other"


def _scan_file_for_helpers_and_calls(
    path: Path, root: Path
) -> tuple[list[HelperRecord], dict[str, int]]:
    """Parse one file once, returning (helper records, call-site counts by short name)."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module = str(path.relative_to(root)).replace("\\", "/").removesuffix(".py")
    digest = _digest(source)
    records: list[HelperRecord] = []
    calls: dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls[func.id] = calls.get(func.id, 0) + 1
            elif isinstance(func, ast.Attribute):
                calls[func.attr] = calls.get(func.attr, 0) + 1
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body_text = ast.unparse(node)
        if not _is_validation_helper(node.name, body_text):
            continue
        return_paths = sum(
            1 for child in ast.walk(node) if isinstance(child, (ast.Return, ast.Raise))
        )
        records.append(
            HelperRecord(
                module=module,
                qualified_name=f"{module}.{node.name}",
                signature=_signature_of(node),
                normalized_body_hash=_digest(_normalized_body(node)),
                source_digest=digest,
                return_paths=return_paths,
                raises=tuple(_exception_types(body_text)),
                call_sites=0,
                layer=_layer(module),
                features=_features_of(node),
            )
        )
    return records, calls


def _cluster(records: list[HelperRecord]) -> dict[str, list[HelperRecord]]:
    """Partition records by (signature, normalized body hash)."""
    groups: dict[str, list[HelperRecord]] = {}
    for record in records:
        key = f"{record.signature}|{record.normalized_body_hash}"
        groups.setdefault(key, []).append(record)
    return groups


def run_inventory(root: Path, *, include_all: bool = False) -> dict[str, Any]:
    """Scan ``root`` and return the versioned inventory report."""
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    records: list[HelperRecord] = []
    call_counts: dict[str, int] = {}
    for path in files:
        try:
            file_records, file_calls = _scan_file_for_helpers_and_calls(path, root)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"scan failed for {path}: {exc}") from exc
        records.extend(file_records)
        for name, count in file_calls.items():
            call_counts[name] = call_counts.get(name, 0) + count

    for index, record in enumerate(records):
        records[index] = HelperRecord(
            module=record.module,
            qualified_name=record.qualified_name,
            signature=record.signature,
            normalized_body_hash=record.normalized_body_hash,
            source_digest=record.source_digest,
            return_paths=record.return_paths,
            raises=record.raises,
            call_sites=call_counts.get(record.qualified_name.rsplit(".", maxsplit=1)[-1], 0),
            layer=record.layer,
            features=record.features,
        )

    clusters = {
        key: [record.as_dict() for record in group]
        for key, group in _cluster(records).items()
        if len(group) >= MIN_CLUSTER_SIZE
    }
    return {
        "schema": SCHEMA,
        "root": str(root),
        "scan": {
            "file_count": len(files),
            "helper_count": len(records),
        },
        "candidate_clusters": clusters,
        "cluster_count": len(clusters),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the inventory CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--markdown", action="store_true", help="emit a Markdown report")
    parser.add_argument(
        "--report", type=Path, default=None, help="write the JSON report to this path"
    )
    parser.add_argument(
        "--include-all", action="store_true", help="scan excluded/third-party paths too"
    )
    args = parser.parse_args(argv)

    try:
        report = run_inventory(args.root, include_all=args.include_all)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown:
        print(_markdown(report))
    elif args.report is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _markdown(report: dict[str, Any]) -> str:
    """Render a concise Markdown report."""
    lines = [
        f"# Validation-helper inventory (`{report['schema']}`)",
        "",
        f"- Root: `{report['root']}` | helpers: {report['scan']['helper_count']} | "
        f"candidate clusters: {report['cluster_count']}",
        "",
    ]
    if not report["candidate_clusters"]:
        lines.append("No safe cluster of at least three behavior-identical definitions found.")
        return "\n".join(lines) + "\n"
    for key, members in sorted(report["candidate_clusters"].items()):
        lines.append(f"## Cluster `{key[:60]}...` ({len(members)} definitions)")
        for member in members:
            lines.append(
                f"- `{member['qualified_name']}` ({member['signature']}, "
                f"calls={member['call_sites']}, layer={member['layer']})"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
