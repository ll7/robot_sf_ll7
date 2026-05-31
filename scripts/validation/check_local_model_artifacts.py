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
DEFAULT_PROMOTED_SURFACES = Path("configs/benchmarks/promoted_config_surfaces.yaml")
LOCAL_MODEL_KEYS = {"model_path", "resume_from"}
PROMOTED_BLOCKED_STATUS = "promoted_blocked"


@dataclass(frozen=True)
class LocalModelReference:
    """One local model path found in a YAML config."""

    path: str
    field: str
    value: str
    status: str
    reason: str
    surface: str = "local_experimental"


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


def _path_lookup_candidates(path: Path) -> list[str]:
    """Return stable path spellings for config-surface matching."""
    candidates = [path.as_posix()]
    if path.is_absolute():
        try:
            candidates.append(path.relative_to(Path.cwd()).as_posix())
        except ValueError:
            pass
    candidates.append(path.name)
    return list(dict.fromkeys(candidates))


def _display_path(path: Path) -> str:
    """Return the repository-relative path when available."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_promoted_surfaces(path: Path) -> dict[str, str]:
    """Load benchmark-promoted config paths and their reasons."""
    if not path.exists():
        return {}
    payload = _load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    if payload.get("version") != 1:
        raise ValueError(f"{path}: version must be 1")
    entries = payload.get("promoted_configs")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: promoted_configs must be a list")

    surfaces: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            config_path = entry.strip()
            reason = "Benchmark-promoted config surface."
        elif isinstance(entry, dict):
            config_path = str(entry.get("path") or "").strip()
            reason = str(entry.get("reason") or "").strip()
        else:
            raise ValueError(f"{path}: promoted_configs[{index}] must be a mapping or string")
        if not config_path or not reason:
            raise ValueError(f"{path}: promoted_configs[{index}] requires path and reason")
        surfaces[config_path] = reason
    return surfaces


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
    promoted_surfaces_path: Path = DEFAULT_PROMOTED_SURFACES,
) -> list[LocalModelReference]:
    """Inspect YAML configs and classify local model references.

    Returns:
        list[LocalModelReference]: Classified references. ``status=unblocked`` rows should fail
        preflight; ``status=blocked`` rows are intentionally explicit follow-up work.
    """
    blocklist = _load_blocklist(blocklist_path)
    promoted_surfaces = _load_promoted_surfaces(promoted_surfaces_path)
    expanded_scan_paths = list(scan_paths)
    expanded_scan_paths.extend(Path(path) for path in promoted_surfaces)
    rows: list[LocalModelReference] = []
    for yaml_path in _iter_yaml_files(expanded_scan_paths):
        if yaml_path == blocklist_path:
            continue
        rel_path = _display_path(yaml_path)
        lookup_paths = _path_lookup_candidates(yaml_path)
        promoted_reason = next(
            (
                promoted_surfaces[candidate]
                for candidate in lookup_paths
                if candidate in promoted_surfaces
            ),
            "",
        )
        payload = _load_yaml(yaml_path)
        for field, value in _find_local_references(payload):
            if promoted_reason:
                rows.append(
                    LocalModelReference(
                        rel_path,
                        field,
                        value,
                        PROMOTED_BLOCKED_STATUS,
                        (
                            f"{promoted_reason} Replace the local output/ reference with a "
                            "durable model_id or artifact pointer before using it as a promoted "
                            "benchmark config. Follow-up: "
                            "https://github.com/ll7/robot_sf_ll7/issues/1764"
                        ),
                        "benchmark_promoted",
                    )
                )
                continue
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
                        "local_experimental",
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
    parser.add_argument(
        "--promoted-surfaces",
        type=Path,
        default=DEFAULT_PROMOTED_SURFACES,
        help="YAML file listing benchmark-promoted config surfaces that must never use output/ model paths.",
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
    rows = check_local_model_artifacts(
        args.paths,
        blocklist_path=args.blocklist,
        promoted_surfaces_path=args.promoted_surfaces,
    )
    if args.json:
        print(json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True))
    else:
        if not rows:
            print("OK: no local output model_path/resume_from references found.")
        for row in rows:
            print(f"{row.status.upper()}: {row.path}:{row.field}: {row.value}")
            print(f"  reason: {row.reason}")

    has_unblocked = any(row.status in {"unblocked", PROMOTED_BLOCKED_STATUS} for row in rows)
    has_blocked = any(row.status == "blocked" for row in rows)
    return 1 if has_unblocked or (args.fail_on_blocked and has_blocked) else 0


if __name__ == "__main__":
    raise SystemExit(main())
