#!/usr/bin/env python3
"""Render the generated scripts/README.md sections from the command catalog.

Usage:
    uv run python scripts/dev/render_scripts_readme.py [--check]

Rewrites only the content between the generated-section markers in
``scripts/README.md``. With ``--check``, exits 1 when the committed README
differs from the deterministic render instead of writing it.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from scripts_catalog import (
    CATALOG_PATH,
    README_PATH,
    CatalogError,
    load_catalog,
    render_readme,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the renderer argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=None, help="Catalog path override")
    parser.add_argument("--readme", type=Path, default=None, help="README path override")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed README differs from the deterministic render",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render or check ``scripts/README.md`` against the catalog."""
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    readme_path = repo_root / (args.readme or README_PATH)
    try:
        catalog = load_catalog(repo_root, args.catalog or CATALOG_PATH)
        current = readme_path.read_text(encoding="utf-8")
        rendered = render_readme(current, catalog)
    except (CatalogError, OSError) as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if rendered == current:
            print("scripts README up to date")
            return 0
        diff = difflib.unified_diff(
            current.splitlines(),
            rendered.splitlines(),
            fromfile=str(readme_path),
            tofile=f"{readme_path} (rendered)",
            lineterm="",
        )
        snippet = "\n".join(diff[:40])
        print(f"scripts README drift detected; first diff lines:\n{snippet}")
        return 1

    readme_path.write_text(rendered, encoding="utf-8")
    print(f"rendered {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
