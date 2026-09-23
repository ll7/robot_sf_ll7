#!/usr/bin/env python3
"""Check or refresh root-lockfile digests declared by bootstrap recipes.

Dependency updates can change ``uv.lock`` without changing the frozen recipe
steps. Each recipe still binds to the exact lockfile bytes, so those declared
digests must be refreshed after a lockfile update.

``--check`` (default) reports drift without writing. ``--write`` updates only
the ``lockfile_sha256`` value in recipe files that reference the root
``uv.lock``; all files are validated before any write occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPOSITORY_ROOT / "configs/bootstrap_recipes"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HASH_FIELD_RE = re.compile(r'("lockfile_sha256"\s*:\s*")([0-9a-fA-F]{64})(")')


def _declared_lockfile_hash(path: Path) -> tuple[str, str]:
    """Return the declared root lockfile path and digest from one recipe."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read recipe JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"recipe root must be an object: {path}")
    source = payload.get("source_identity")
    if not isinstance(source, dict):
        raise ValueError(f"recipe source_identity must be an object: {path}")
    lockfile = source.get("lockfile")
    declared = source.get("lockfile_sha256")
    if not isinstance(lockfile, str) or not lockfile:
        raise ValueError(f"recipe lockfile declaration is missing: {path}")
    if not isinstance(declared, str) or not SHA256_RE.fullmatch(declared):
        raise ValueError(f"recipe lockfile_sha256 is malformed: {path}")
    return lockfile, declared


def _recipe_updates(repo_root: Path, recipe_root: Path) -> list[tuple[Path, str, str, str]]:
    """Validate all relevant recipes and return (path, old, new, text) updates."""

    lockfile = repo_root / "uv.lock"
    if not lockfile.is_file():
        raise ValueError(f"root uv.lock is missing or not a regular file: {lockfile}")
    if not recipe_root.is_dir():
        raise ValueError(f"bootstrap recipe directory is missing: {recipe_root}")
    actual = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    updates: list[tuple[Path, str, str, str]] = []
    for path in sorted(recipe_root.glob("*.v1.json")):
        lockfile_ref, declared = _declared_lockfile_hash(path)
        if lockfile_ref != "uv.lock":
            continue
        text = path.read_text(encoding="utf-8")
        matches = list(HASH_FIELD_RE.finditer(text))
        if len(matches) != 1 or matches[0].group(2) != declared:
            raise ValueError(f"recipe must contain one matching lockfile_sha256 field: {path}")
        if declared != actual:
            updates.append((path, declared, actual, text))
    if not updates and not any(
        _declared_lockfile_hash(path)[0] == "uv.lock"
        for path in sorted(recipe_root.glob("*.v1.json"))
    ):
        raise ValueError(f"no recipes reference the root uv.lock under {recipe_root}")
    return updates


def refresh(repo_root: Path, recipe_root: Path, *, write: bool) -> list[tuple[Path, str, str]]:
    """Check or refresh recipe hashes, returning the observed changes."""

    updates = _recipe_updates(repo_root, recipe_root)
    changes = [(path, old, new) for path, old, new, _text in updates]
    if write:
        for path, old, new, text in updates:
            if path.read_text(encoding="utf-8") != text:
                raise ValueError(f"recipe changed during refresh; refusing to write: {path}")
            match = HASH_FIELD_RE.search(text)
            if match is None or match.group(2) != old:
                raise ValueError(f"recipe changed during refresh; refusing to write: {path}")
            updated = text[: match.start(2)] + new + text[match.end(2) :]
            json.loads(updated)
            path.write_text(updated, encoding="utf-8")
        remaining = _recipe_updates(repo_root, recipe_root)
        if remaining:
            raise ValueError("recipe lockfile hashes remain stale after refresh")
    return changes


def main(argv: list[str] | None = None) -> int:
    """Run the non-mutating check or explicit write mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Report stale digests without writing.")
    mode.add_argument("--write", action="store_true", help="Refresh stale digests.")
    args = parser.parse_args(argv)
    try:
        changes = refresh(REPOSITORY_ROOT, RECIPE_ROOT, write=args.write)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"bootstrap recipe lock hashes: {error}", file=sys.stderr)
        return 2
    if not changes:
        print("All bootstrap recipe root lockfile hashes match uv.lock.")
        return 0
    for path, old, new in changes:
        print(f"{path.relative_to(REPOSITORY_ROOT)}: {old} -> {new}")
    if args.write:
        print(f"Refreshed {len(changes)} bootstrap recipe lockfile hash(es).")
        return 0
    print(f"Found {len(changes)} stale bootstrap recipe lockfile hash(es); rerun with --write.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
