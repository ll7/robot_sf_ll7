#!/usr/bin/env python3
"""Run a fail-closed release preflight checklist.

This is a thin CLI wrapper around ``robot_sf.benchmark.release_preflight``.
It does not publish, tag, upload, regenerate evidence, or declare release
readiness. It only writes the evaluator's JSON and Markdown reports and
returns a shell-friendly status code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.release_preflight import (
    ReleasePreflightError,
    evaluate_release_preflight,
    load_release_preflight_checklist,
    render_markdown,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse release-preflight CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checklist",
        required=True,
        type=Path,
        help="Path to a release_preflight_checklist.v1 YAML/JSON file.",
    )
    parser.add_argument(
        "--repo-root",
        default=Path("."),
        type=Path,
        help="Repository root used to resolve checklist artifact paths.",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        type=Path,
        help="Where to write the structured preflight report JSON.",
    )
    parser.add_argument(
        "--out-md",
        required=True,
        type=Path,
        help="Where to write the rendered Markdown preflight report.",
    )
    return parser.parse_args(argv)


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run release preflight and return POSIX-style exit code.

    Exit codes:
    - 0: checklist evaluated with status ``passed``.
    - 2: checklist evaluated with status ``blocked``.
    - 1: malformed checklist, unexpected evaluator status, or report write error.
    """
    args = _parse_args(argv)

    try:
        checklist = load_release_preflight_checklist(args.checklist)
        result = evaluate_release_preflight(checklist, args.repo_root)
        report_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
        report_markdown = render_markdown(result) + "\n"
        _write_text(args.out_json, report_json)
        _write_text(args.out_md, report_markdown)
    except (OSError, ReleasePreflightError, ValueError, json.JSONDecodeError) as exc:
        print(f"release preflight failed: {exc}", file=sys.stderr)
        return 1

    if result["status"] == "passed":
        return 0
    if result["status"] == "blocked":
        return 2

    print(f"release preflight returned unexpected status {result['status']!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
