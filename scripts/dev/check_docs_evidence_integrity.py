#!/usr/bin/env python3
"""Lightweight, changed-path-scoped integrity check for docs/evidence surfaces.

This guard treats documentation, compact evidence bundles, schemas, catalogues,
issue templates, and governance files as first-class research-facing state. It is
intentionally **changed-path scoped**: it only inspects files a change actually
touches, so it can be mandatory in CI without failing on pre-existing legacy drift
in untouched files (issue #3476).

Checks performed on each changed file:

- ``.json`` files must parse as JSON.
- ``.yaml`` / ``.yml`` files must parse as YAML (multi-document allowed).
- Markdown files: every *explicit* repository-local relative link (a target that
  starts with ``./`` or ``../``) must resolve to an existing path. This is the
  conservative subset of link checking that cannot produce false positives on
  external URLs, anchors, or ambiguous bare references.

The check is independent of the full Python test suite.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# Markdown inline link target: capture the URL portion, ignoring an optional title.
_MD_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
# Only validate unambiguously repo-local relative links.
_RELATIVE_PREFIXES = ("./", "../")


def _repo_root() -> Path:
    """Return the Git repository root."""
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def changed_files(base_ref: str, *, root: Path) -> list[str]:
    """Return added/copied/modified/renamed paths relative to ``base_ref``."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def _check_json(path: Path) -> list[str]:
    """Return parse errors for a JSON file."""
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{path}: invalid JSON: {exc}"]
    return []


def _check_yaml(path: Path) -> list[str]:
    """Return parse errors for a YAML file (multi-document allowed)."""
    try:
        list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as exc:
        return [f"{path}: invalid YAML: {exc}"]
    return []


def _check_markdown_links(path: Path, *, root: Path) -> list[str]:
    """Return broken explicit repo-local relative links in a Markdown file."""
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: unreadable Markdown: {exc}"]
    for raw_target in _MD_LINK.findall(text):
        target = raw_target.strip()
        if not target.startswith(_RELATIVE_PREFIXES):
            continue
        # Drop any anchor fragment; the file existence is what we verify.
        file_part = target.split("#", 1)[0]
        if not file_part:
            continue
        resolved = (path.parent / file_part).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            problems.append(f"{path}: relative link escapes the repository: {target}")
            continue
        if not resolved.exists():
            problems.append(f"{path}: broken repo-local link: {target}")
    return problems


def check_files(files: Iterable[str], *, root: Path) -> list[str]:
    """Run the integrity checks over the given repo-relative files."""
    problems: list[str] = []
    for rel in files:
        path = (root / rel).resolve()
        if not path.is_file():
            # Deleted/renamed-away paths are not our concern (filtered upstream).
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            problems.extend(_check_json(path))
        elif suffix in {".yaml", ".yml"}:
            problems.extend(_check_yaml(path))
        elif suffix in {".md", ".markdown"}:
            problems.extend(_check_markdown_links(path, root=root))
    return problems


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git ref to diff against for changed-file scoping (default: origin/main).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit repo-relative files to check, bypassing git diff (mainly for tests).",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help=(
            "Advisory mode: emit findings as non-blocking GitHub warning annotations "
            "and always exit 0. This is the first-rollout default for CI so the check "
            "surfaces problems without blocking unrelated docs fixes."
        ),
    )
    return parser


def _emit_warnings(problems: list[str]) -> None:
    """Emit findings as GitHub Actions warning annotations (non-blocking)."""
    for problem in problems:
        # `::warning::` annotations surface in the PR/run UI but do not fail the job.
        sys.stdout.write(f"::warning::docs/evidence integrity: {problem}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the docs/evidence integrity check and return a shell exit code."""
    args = _build_parser().parse_args(argv)
    root = _repo_root()
    files = list(args.files) if args.files is not None else changed_files(args.base_ref, root=root)
    if not files:
        print("docs/evidence integrity: no changed docs/evidence files to check.")
        return 0
    problems = check_files(files, root=root)
    if not problems:
        print(f"docs/evidence integrity: {len(files)} changed file(s) passed.")
        return 0
    if args.warn_only:
        _emit_warnings(problems)
        print(
            f"docs/evidence integrity: {len(problems)} advisory finding(s) "
            "(warning-only, not blocking)."
        )
        return 0
    sys.stderr.write("docs/evidence integrity check failed:\n")
    sys.stderr.write("\n".join(f"  {problem}" for problem in problems) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
