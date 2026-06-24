#!/usr/bin/env python3
"""Validate or repair repository AI assistant configuration links."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ".agents/mirror_manifest.yaml"


@dataclass(frozen=True)
class LinkSpec:
    """Expected compatibility symlink from a tool-specific path into `.agents`."""

    path: str
    target: str


@dataclass(frozen=True)
class PointerSpec:
    """Expected thin instruction pointer file contract."""

    path: str
    must_reference: str


@dataclass(frozen=True)
class MirrorManifest:
    """Supported AI assistant compatibility surfaces."""

    symlink_mirrors: tuple[LinkSpec, ...]
    pointer_files: tuple[PointerSpec, ...]


def _repo_path(path: str) -> Path:
    """Resolve a repo-relative path and reject paths that escape the repository root."""
    candidate = REPO_ROOT / path
    if not candidate.parent.resolve(strict=False).is_relative_to(REPO_ROOT):
        raise ValueError(f"Path escapes repository root: {path}")
    return candidate


def _manifest_entries(manifest: object, key: str) -> list[dict[str, object]]:
    """Return a manifest list section after light shape validation."""
    if not isinstance(manifest, dict):
        raise ValueError(f"{MANIFEST_PATH}: expected mapping at document root")
    entries = manifest.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"{MANIFEST_PATH}: {key} must be a list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{MANIFEST_PATH}: {key}[{index}] must be a mapping")
    return entries


def _manifest_str(entry: dict[str, object], section: str, index: int, field: str) -> str:
    """Read a required string field from a manifest entry."""
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{MANIFEST_PATH}: {section}[{index}].{field} must be a non-empty string")
    return value


def load_manifest() -> MirrorManifest:
    """Load the supported AI configuration mirror manifest."""
    manifest_path = _repo_path(MANIFEST_PATH)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)

    symlink_mirrors = tuple(
        LinkSpec(
            path=_manifest_str(entry, "symlink_mirrors", index, "path"),
            target=_manifest_str(entry, "symlink_mirrors", index, "target"),
        )
        for index, entry in enumerate(_manifest_entries(manifest, "symlink_mirrors"))
    )
    pointer_files = tuple(
        PointerSpec(
            path=_manifest_str(entry, "pointer_files", index, "path"),
            must_reference=_manifest_str(entry, "pointer_files", index, "must_reference"),
        )
        for index, entry in enumerate(_manifest_entries(manifest, "pointer_files"))
    )
    return MirrorManifest(symlink_mirrors=symlink_mirrors, pointer_files=pointer_files)


LINK_SPECS = load_manifest().symlink_mirrors


def _check_link(spec: LinkSpec, *, fix: bool) -> list[str]:
    """Check one symlink spec, optionally repairing missing or stale symlinks."""
    path = _repo_path(spec.path)
    target = Path(spec.target)
    expected_abs = (path.parent / target).resolve()
    if not expected_abs.is_relative_to(REPO_ROOT):
        raise ValueError(f"Symlink target escapes repository root: {spec.path} -> {spec.target}")
    errors: list[str] = []

    if path.is_symlink():
        if not expected_abs.exists():
            errors.append(
                f"{spec.path}: target does not exist: {expected_abs.relative_to(REPO_ROOT)}"
            )
            return errors
        actual = Path(path.readlink())
        if actual == target:
            return []
        if not fix:
            return [f"{spec.path}: expected symlink to {spec.target}, found {actual}"]
        path.unlink()
    elif path.exists():
        return [f"{spec.path}: exists but is not a symlink"]
    elif not expected_abs.exists():
        errors.append(f"{spec.path}: target does not exist: {expected_abs.relative_to(REPO_ROOT)}")
        return errors
    elif not fix:
        return [f"{spec.path}: missing symlink to {spec.target}"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)
    return errors


def _check_pointer_file(path: str, expected_text: str) -> list[str]:
    """Return drift errors when a tool-specific instruction pointer stops pointing canonical."""
    pointer_path = _repo_path(path)
    if not pointer_path.exists():
        return [f"{path}: missing pointer file referencing {expected_text!r}"]
    if not pointer_path.is_file():
        return [f"{path}: pointer path exists but is not a file"]
    try:
        content = pointer_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: cannot read pointer file: {exc}"]
    if expected_text not in content:
        return [f"{path}: expected to reference {expected_text!r}"]
    return []


def main() -> int:
    """Return non-zero when supported AI configuration surfaces drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="validate without changing files")
    group.add_argument("--fix", action="store_true", help="repair supported symlinks")
    args = parser.parse_args()

    fix = args.fix
    manifest = load_manifest()
    errors: list[str] = []
    for spec in manifest.symlink_mirrors:
        errors.extend(_check_link(spec, fix=fix))

    for spec in manifest.pointer_files:
        errors.extend(_check_pointer_file(spec.path, spec.must_reference))

    if errors:
        raise SystemExit("AI config drift detected:\n- " + "\n- ".join(errors))

    mode = "repaired" if fix else "validated"
    print(f"AI config links {mode}: {len(manifest.symlink_mirrors)} symlinks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
