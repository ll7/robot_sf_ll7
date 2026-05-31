#!/usr/bin/env python3
"""Check configs for silent local-only model artifact dependencies.

The checker targets benchmark/config portability for Issue #1638. It flags
``model_path`` and ``resume_from`` values under ``output/`` unless the exact
reference is listed in the blocklist with an artifact-promotion follow-up.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCAN_ROOTS = (Path("configs/baselines"),)
DEFAULT_BLOCKLIST = Path("configs/baselines/local_model_artifact_blocklist.yaml")
LOCAL_MODEL_KEYS = {"model_path", "resume_from"}


@dataclass(frozen=True)
class LocalModelReference:
    """One local model path found in a YAML config."""

    path: str
    field: str
    value: str
    status: str
    reason: str


def _load_yaml(path: Path) -> Any:
    """Load a YAML file and return the parsed payload."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_output_model_path(value: Any) -> bool:
    """Return whether a scalar points at a local output model artifact."""
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    return normalized.startswith(("output/model_cache/", "output/models/", "output/slurm/"))


def _iter_yaml_files(paths: list[Path]) -> list[Path]:
    """Expand scan roots into sorted YAML files.

    Returns:
        list[Path]: YAML files to inspect.
    """
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(path.rglob("*.yaml"))
            files.update(path.rglob("*.yml"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            files.add(path)
    return sorted(files)


def _find_local_references(
    payload: Any, *, path_parts: tuple[str, ...] = ()
) -> list[tuple[str, str]]:
    """Return dotted-field/value pairs for local model references."""
    references: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            next_parts = (*path_parts, key_str)
            if key_str in LOCAL_MODEL_KEYS and _is_output_model_path(value):
                references.append((".".join(next_parts), str(value).strip()))
            references.extend(_find_local_references(value, path_parts=next_parts))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            references.extend(_find_local_references(value, path_parts=(*path_parts, str(index))))
    return references


def _load_blocklist(path: Path) -> dict[tuple[str, str, str], str]:
    """Load the explicit local-artifact blocklist.

    Returns:
        dict[tuple[str, str, str], str]: Mapping from ``(path, field, value)`` to reason.
    """
    if not path.exists():
        return {}
    payload = _load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    if payload.get("version") != 1:
        raise ValueError(f"{path}: version must be 1")
    entries = payload.get("blocked_references")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: blocked_references must be a list")

    blocklist: dict[tuple[str, str, str], str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: blocked_references[{index}] must be a mapping")
        config_path = str(entry.get("path") or "").strip()
        field = str(entry.get("field") or "").strip()
        value = str(entry.get("value") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if not config_path or not field or not value or not reason:
            raise ValueError(
                f"{path}: blocked_references[{index}] requires path, field, value, and reason"
            )
        blocklist[(config_path, field, value)] = reason
    return blocklist


def check_local_model_artifacts(
    scan_paths: list[Path],
    *,
    blocklist_path: Path = DEFAULT_BLOCKLIST,
) -> list[LocalModelReference]:
    """Inspect YAML configs and classify local model references.

    Returns:
        list[LocalModelReference]: Classified references. ``status=unblocked`` rows should fail
        preflight; ``status=blocked`` rows are intentionally explicit follow-up work.
    """
    blocklist = _load_blocklist(blocklist_path)
    rows: list[LocalModelReference] = []
    for yaml_path in _iter_yaml_files(scan_paths):
        if yaml_path == blocklist_path:
            continue
        rel_path = yaml_path.as_posix()
        payload = _load_yaml(yaml_path)
        for field, value in _find_local_references(payload):
            reason = blocklist.get((rel_path, field, value))
            if reason:
                rows.append(LocalModelReference(rel_path, field, value, "blocked", reason))
            else:
                rows.append(
                    LocalModelReference(
                        rel_path,
                        field,
                        value,
                        "unblocked",
                        "replace with model_id or add an explicit artifact-promotion blocker",
                    )
                )
    return rows


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_SCAN_ROOTS),
        help="YAML files or directories to scan; defaults to configs/baselines",
    )
    parser.add_argument(
        "--blocklist",
        type=Path,
        default=DEFAULT_BLOCKLIST,
        help="Explicit blocklist for known local-only artifact references.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON rows.")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Also fail when all local references are explicitly blocked.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the local model artifact preflight."""
    args = _parse_args()
    rows = check_local_model_artifacts(args.paths, blocklist_path=args.blocklist)
    if args.json:
        print(json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True))
    else:
        if not rows:
            print("OK: no local output model_path/resume_from references found.")
        for row in rows:
            print(f"{row.status.upper()}: {row.path}:{row.field}: {row.value}")
            print(f"  reason: {row.reason}")

    has_unblocked = any(row.status == "unblocked" for row in rows)
    has_blocked = any(row.status == "blocked" for row in rows)
    return 1 if has_unblocked or (args.fail_on_blocked and has_blocked) else 0


if __name__ == "__main__":
    raise SystemExit(main())
