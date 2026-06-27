#!/usr/bin/env python3
"""Presence-only preflight for the v0.1 benchmark publication prerequisites (epic #2910).

This checker reads a declarative prerequisites checklist (default:
``configs/benchmarks/releases/benchmark_v0_1_publication_prerequisites.yaml``) and
verifies that every referenced canonical-owner path exists on disk.

Scope and limits (deliberate):

- This is a *presence* preflight. It only confirms that the prerequisite owners a
  v0.1 benchmark package must be built on are present.
- It does NOT release, tag, upload artifacts, run a benchmark/falsification
  campaign, evaluate scenario certification, or declare the benchmark "ready".
- A passing run means the prerequisite artifacts exist, NOT that any
  benchmark/falsification claim is established.

Exit code is ``0`` when no *required* prerequisite is missing, otherwise ``1`` so
the check can fail closed in CI or local preflight without asserting readiness.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION = "benchmark-v0_1-publication-prerequisites.v1"
DEFAULT_CHECKLIST_PATH = Path(
    "configs/benchmarks/releases/benchmark_v0_1_publication_prerequisites.yaml"
)
# Explicit boundary string surfaced in every report so the output cannot be
# mistaken for a benchmark-readiness or claim declaration.
CLAIM_BOUNDARY = (
    "presence-only preflight; existing prerequisites do NOT imply benchmark "
    "readiness or any established claim"
)


def get_repository_root() -> Path:
    """Return the repository root for resolving relative checklist paths."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PrerequisiteResult:
    """Resolved presence status for a single checklist prerequisite."""

    item_id: str
    category: str
    required: bool
    description: str
    present: bool
    missing_paths: tuple[str, ...]

    @property
    def status(self) -> str:
        """Return ``present`` when every referenced path exists, else ``missing``."""
        return "present" if self.present else "missing"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this result."""
        return {
            "id": self.item_id,
            "category": self.category,
            "required": self.required,
            "description": self.description,
            "status": self.status,
            "missing_paths": list(self.missing_paths),
        }


def _load_checklist(path: Path) -> dict[str, Any]:
    """Load and minimally validate the checklist mapping.

    Raises:
        ValueError: When the payload is not a mapping, the schema version is
            unexpected, or the ``prerequisites`` list is missing/empty.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Checklist {path} must be a mapping")
    schema = payload.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Checklist {path} has schema_version {schema!r}, expected {SCHEMA_VERSION!r}"
        )
    prerequisites = payload.get("prerequisites")
    if not isinstance(prerequisites, list) or not prerequisites:
        raise ValueError(f"Checklist {path} must define a non-empty 'prerequisites' list")
    return payload


def evaluate_prerequisite(item: dict[str, Any], repo_root: Path) -> PrerequisiteResult:
    """Evaluate one checklist item against the filesystem.

    Args:
        item: One ``prerequisites`` entry; must carry ``id`` and a non-empty
            ``paths`` list of repository-relative paths.
        repo_root: Root used to resolve relative paths.

    Returns:
        The resolved :class:`PrerequisiteResult`.

    Raises:
        ValueError: When the item lacks an ``id`` or any usable ``paths``.
    """
    item_id = item.get("id")
    if not item_id:
        raise ValueError(f"Prerequisite entry missing 'id': {item!r}")
    raw_paths = item.get("paths") or []
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"Prerequisite {item_id!r} must list at least one path")
    missing = tuple(str(p) for p in raw_paths if not (repo_root / str(p)).exists())
    return PrerequisiteResult(
        item_id=str(item_id),
        category=str(item.get("category", "uncategorized")),
        required=bool(item.get("required", True)),
        description=str(item.get("description", "")),
        present=not missing,
        missing_paths=missing,
    )


def build_report(checklist_path: Path, repo_root: Path) -> dict[str, Any]:
    """Evaluate every prerequisite and assemble a deterministic report payload."""
    payload = _load_checklist(checklist_path)
    results = [evaluate_prerequisite(item, repo_root) for item in payload["prerequisites"]]
    required_missing = [r.item_id for r in results if r.required and not r.present]
    optional_missing = [r.item_id for r in results if not r.required and not r.present]
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "release_target": payload.get("release_target"),
        "epic": payload.get("epic"),
        "checklist_path": str(checklist_path),
        "total": len(results),
        "present": sum(1 for r in results if r.present),
        "required_missing": required_missing,
        "optional_missing": optional_missing,
        # Prerequisites *present* is necessary but not sufficient for readiness.
        "prerequisites_satisfied": not required_missing,
        "prerequisites": [r.to_dict() for r in results],
    }


def _render_text(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary of the report."""
    lines = [
        f"Benchmark v0.1 publication prerequisites ({report['checklist_path']})",
        f"Boundary: {report['claim_boundary']}",
        f"Present: {report['present']}/{report['total']}",
        "",
    ]
    for item in report["prerequisites"]:
        marker = "ok  " if item["status"] == "present" else "MISS"
        req = "required" if item["required"] else "optional"
        lines.append(f"  [{marker}] {item['id']} ({item['category']}, {req})")
        for missing in item["missing_paths"]:
            lines.append(f"         missing: {missing}")
    lines.append("")
    if report["required_missing"]:
        lines.append(f"FAIL: required prerequisites missing: {report['required_missing']}")
    else:
        lines.append("PASS: all required prerequisite owners present (presence-only).")
    if report["optional_missing"]:
        lines.append(f"Note: optional prerequisites missing: {report['optional_missing']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns ``0`` when no required prerequisite is missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="Path to the prerequisites checklist YAML (default: repo v0.1 checklist).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root used to resolve relative paths (default: inferred).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve() if args.repo_root else get_repository_root()
    checklist_path = args.checklist if args.checklist else repo_root / DEFAULT_CHECKLIST_PATH

    try:
        report = build_report(checklist_path, repo_root)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    return 0 if report["prerequisites_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
