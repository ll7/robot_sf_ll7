#!/usr/bin/env python3
"""Build the declared public reference inventory for the locality audit (issue #8929).

The durable-artifact locality audit (#8907) validates a caller-supplied
``references`` inventory against a sanitized locator projection, but it cannot
detect drift between that inventory and the canonical public surfaces it is
meant to represent. This command owns the public-side inventory: it reads the
declared surfaces in ``configs/validation/public_reference_inventory.yaml`` and
resolves an exact artifact identity (ID, version, digest) from each surface
without URL-based inference. Output is a deterministic
``public_reference_inventory.v1`` payload whose ``references`` rows feed the
locality audit directly.

Usage::

    uv run python scripts/validation/build_public_reference_inventory.py --check
    uv run python scripts/validation/build_public_reference_inventory.py --output /tmp/inventory.json

Exit codes:
    0 - inventory built; no findings
    1 - findings present (``--check`` fails closed)
    2 - declaration or usage error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "public_reference_inventory.v1"
SOURCES_SCHEMA = "public_reference_inventory_sources.v1"
DEFAULT_DECLARATION = Path("configs/validation/public_reference_inventory.yaml")
REFERENCE_FIELDS = (
    "reference_id",
    "artifact_id",
    "version",
    "digest",
    "consumer_path",
    "consumer_status",
    "retention_class",
)
CONSUMER_STATUSES = ("active", "inactive")
RETENTION_CLASSES = ("durable_required", "release_facing", "historical")
KINDS = ("release_checksum_manifest",)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE = re.compile(r"^(?:/|[A-Za-z]:[\\/])")


@dataclass(frozen=True, slots=True)
class Finding:
    """One public-safe inventory finding with a stable reason code."""

    code: str
    source_id: str | None
    message: str


def _finding(code: str, source_id: str | None = None, message: str = "") -> Finding:
    return Finding(code, source_id, message)


def _is_private_locator(value: str) -> bool:
    """Return True when *value* looks like a URL or absolute filesystem path."""
    return "://" in value or _ABSOLUTE.match(value) is not None


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _require_repo_relative(
    value: str, *, findings: list[Finding], source_id: str, name: str
) -> bool:
    """Validate that *value* is a repository-relative, public-safe path."""
    parts = Path(value).parts
    if _is_private_locator(value) or ".." in parts:
        findings.append(
            _finding(
                "private_locator_value",
                source_id,
                f"{name} must be repository-relative and public-safe",
            )
        )
        return False
    return True


def load_declaration(path: Path) -> tuple[list[dict[str, Any]], list[Finding]]:
    """Load and validate the declared public-surface list."""
    findings: list[Finding] = []
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        findings.append(_finding("declaration_unreadable", None, str(path)))
        return [], findings
    except yaml.YAMLError:
        findings.append(_finding("declaration_invalid", None, "declaration is not valid YAML"))
        return [], findings
    if not isinstance(payload, dict) or payload.get("schema_version") != SOURCES_SCHEMA:
        findings.append(
            _finding("declaration_invalid", None, f"schema_version must be {SOURCES_SCHEMA}")
        )
        return [], findings
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        findings.append(_finding("declaration_invalid", None, "sources must be a non-empty list"))
        return [], findings

    sources: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_sources):
        if not isinstance(item, dict):
            findings.append(
                _finding("declaration_invalid", None, f"sources[{index}] not a mapping")
            )
            continue
        source_id = _text(item.get("id"))
        kind = _text(item.get("kind"))
        surface = _text(item.get("path"))
        consumer_path = _text(item.get("consumer_path"))
        consumer_status = _text(item.get("consumer_status"))
        retention_class = _text(item.get("retention_class"))
        if (
            source_id is None
            or kind not in KINDS
            or surface is None
            or consumer_path is None
            or consumer_status not in CONSUMER_STATUSES
            or retention_class not in RETENTION_CLASSES
        ):
            findings.append(
                _finding("declaration_invalid", source_id, f"sources[{index}] has invalid fields")
            )
            continue
        if source_id in seen_ids:
            findings.append(_finding("duplicate_source_id", source_id, "duplicate source id"))
            continue
        seen_ids.add(source_id)
        _require_repo_relative(surface, findings=findings, source_id=source_id, name="path")
        _require_repo_relative(
            consumer_path, findings=findings, source_id=source_id, name="consumer_path"
        )
        sources.append(
            {
                "id": source_id,
                "kind": kind,
                "path": surface,
                "consumer_path": consumer_path,
                "consumer_status": consumer_status,
                "retention_class": retention_class,
            }
        )
    return sources, findings


def _file_rows(
    component: str, files: list[object], *, findings: list[Finding], source_id: str
) -> list[tuple[str, str]]:
    """Resolve per-file identity rows from one manifest file list."""
    rows: list[tuple[str, str]] = []
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            findings.append(
                _finding(
                    "unresolved_identity", source_id, f"{component}.files[{index}] is not a mapping"
                )
            )
            continue
        file_path = _text(item.get("path"))
        file_digest = _text(item.get("sha256"))
        identity = file_path or f"{component}[{index}]"
        if file_path is not None:
            _require_repo_relative(
                file_path,
                findings=findings,
                source_id=source_id,
                name=f"{component}.files[{index}].path",
            )
        if file_digest is None:
            findings.append(
                _finding(
                    "unresolved_identity", source_id, f"{component}.files[{index}] has no sha256"
                )
            )
            continue
        rows.append((identity, file_digest))
    return rows


def _entry_rows(
    component: str, entry: object, *, findings: list[Finding], source_id: str
) -> list[tuple[str, str]]:
    """Resolve identity rows from one artifact-set or embedded-artifact entry."""
    if not isinstance(entry, dict):
        findings.append(_finding("unresolved_identity", source_id, f"{component} is not a mapping"))
        return []
    files = entry.get("files")
    if isinstance(files, list):
        return _file_rows(component, files, findings=findings, source_id=source_id)
    digest = _text(entry.get("sha256"))
    if digest is not None:
        return [(component, digest)]
    findings.append(_finding("unresolved_identity", source_id, f"{component} has no sha256"))
    return []


def _release_rows(
    manifest: dict[str, Any], *, findings: list[Finding], source_id: str
) -> list[tuple[str, str]]:
    """Return ``(artifact_component, digest)`` rows from one release manifest."""
    rows: list[tuple[str, str]] = []
    for field, prefix in (
        ("artifact_set", "artifact_set."),
        ("embedded_artifacts", "embedded_artifacts."),
    ):
        mapping = manifest.get(field)
        if not isinstance(mapping, dict):
            continue
        for key, entry in mapping.items():
            rows.extend(
                _entry_rows(f"{prefix}{key}", entry, findings=findings, source_id=source_id)
            )
    return rows


def _references_for_source(
    source: dict[str, Any], manifest: dict[str, Any], findings: list[Finding]
) -> list[dict[str, str]]:
    """Build reference rows for one declared release-checksum-manifest surface."""
    source_id = source["id"]
    release_id = _text(manifest.get("release_id"))
    release_tag = _text(manifest.get("release_tag"))
    if manifest.get("schema_version") != "release-checksum-manifest.v1" or not (
        release_id and release_tag
    ):
        findings.append(
            _finding(
                "source_schema_mismatch",
                source_id,
                "expected release-checksum-manifest.v1 with release_id and release_tag",
            )
        )
        return []
    references: list[dict[str, str]] = []
    for component, digest in _release_rows(manifest, findings=findings, source_id=source_id):
        if _SHA256.fullmatch(digest) is None:
            findings.append(
                _finding("invalid_digest", source_id, f"{component} sha256 is not 64-hex")
            )
            continue
        references.append(
            {
                "reference_id": f"release:{release_id}:{component}",
                "artifact_id": f"{release_id}:{component}",
                "version": release_tag,
                "digest": digest,
                "consumer_path": source["consumer_path"],
                "consumer_status": source["consumer_status"],
                "retention_class": source["retention_class"],
            }
        )
    return references


def build_inventory(declaration: Path, *, root: Path) -> tuple[dict[str, Any], tuple[Finding, ...]]:
    """Build the declared inventory and return ``(payload, findings)``."""
    findings: list[Finding] = []
    sources, declaration_findings = load_declaration(declaration)
    findings.extend(declaration_findings)

    references: list[dict[str, str]] = []
    seen_reference_ids: set[str] = set()
    generated_from: list[dict[str, str]] = []
    for source in sources:
        surface = root / source["path"]
        generated_from.append({"id": source["id"], "kind": source["kind"], "path": source["path"]})
        try:
            manifest = yaml.safe_load(surface.read_text(encoding="utf-8"))
        except OSError:
            findings.append(_finding("source_unreadable", source["id"], source["path"]))
            continue
        except yaml.YAMLError:
            findings.append(_finding("source_invalid_yaml", source["id"], source["path"]))
            continue
        if not isinstance(manifest, dict):
            findings.append(
                _finding("source_schema_mismatch", source["id"], "manifest not mapping")
            )
            continue
        for reference in _references_for_source(source, manifest, findings):
            reference_id = reference["reference_id"]
            if reference_id in seen_reference_ids:
                findings.append(_finding("duplicate_reference_id", source["id"], reference_id))
                continue
            seen_reference_ids.add(reference_id)
            references.append(reference)

    if not references:
        findings.append(_finding("empty_inventory", None, "no references resolved"))
    payload = {
        "schema": SCHEMA,
        "status": "fail" if findings else "ok",
        "generated_from": generated_from,
        "references": sorted(references, key=lambda reference: reference["reference_id"]),
        "findings": [
            {"code": finding.code, "source_id": finding.source_id, "message": finding.message}
            for finding in findings
        ],
    }
    return payload, tuple(findings)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--declaration",
        type=Path,
        default=DEFAULT_DECLARATION,
        help=f"Declared public-surface list (default: {DEFAULT_DECLARATION}).",
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON to this path.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when any finding is present (fail closed).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    payload, findings = build_inventory(args.declaration, root=args.root)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    for finding in findings:
        sys.stderr.write(f"{finding.code}: {finding.source_id or '-'} {finding.message}\n")
    if findings and args.check:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
