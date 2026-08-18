#!/usr/bin/env python3
"""Build one diagnostic-only, cell-bound trace dossier package from existing artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="JSON list of cell rows.")
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--campaign-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-search-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--command",
        default="scripts/tools/build_trace_dossier_package.py",
        help="Stable command recorded in the renderer manifest.",
    )
    return parser


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"candidate JSON is unreadable: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("candidates")
    if not isinstance(payload, list):
        raise ValueError("candidate JSON must be a list or an object with candidates")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the trace dossier package CLI."""

    args = _parser().parse_args(argv)
    from robot_sf.benchmark.trace_dossier_package import (
        TraceDossierPackageError,
        build_trace_dossier_package,
    )

    try:
        result = build_trace_dossier_package(
            candidates=_load_candidates(args.candidates),
            release_manifest_path=args.release_manifest,
            campaign_store_dir=args.campaign_store,
            output_dir=args.output_dir,
            trace_search_roots=tuple(args.trace_search_root),
            command=args.command,
        )
    except (OSError, TraceDossierPackageError, ValueError, json.JSONDecodeError) as exc:
        print(f"trace dossier package failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote trace dossier package {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
