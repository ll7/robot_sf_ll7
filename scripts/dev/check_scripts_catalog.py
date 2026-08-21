#!/usr/bin/env python3
"""Fail-closed validator for the scripts command catalog.

Usage:
    uv run python scripts/dev/check_scripts_catalog.py [--catalog scripts/catalog.yaml]

Exits 0 when the catalog satisfies every schema and completeness rule; exits 1
with one error per line otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts_catalog import CatalogError, load_catalog, validate_catalog


def build_parser() -> argparse.ArgumentParser:
    """Return the checker argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=None, help="Catalog path override")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run catalog validation and print one line per failure."""
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        catalog = load_catalog(repo_root, args.catalog)
    except CatalogError as exc:
        print(f"catalog invalid: {exc}", file=sys.stderr)
        return 1
    errors = validate_catalog(catalog, repo_root)
    for error in errors:
        print(f"catalog error: {error}")
    if errors:
        return 1
    print(
        f"scripts catalog OK: {len(catalog.commands)} commands, "
        f"{len(catalog.directories)} directory rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
