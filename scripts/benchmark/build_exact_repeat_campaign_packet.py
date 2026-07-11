#!/usr/bin/env python3
"""Create or verify the exact-repeat evidence packet for issue #5263."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.exact_repeat_campaign import (
    build_manifest,
    compare_verified_hosts,
    verify_host_report,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    manifest = subcommands.add_parser("manifest", help="build the immutable seven-cell request")
    manifest.add_argument("--baseline-report", type=Path, required=True)
    manifest.add_argument("--source-episodes", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    verify = subcommands.add_parser("verify-host", help="verify a completed host report")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--host-report", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    compare = subcommands.add_parser("compare-hosts", help="compare two verified host reports")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Parse a packet action and write its schema-checked JSON result."""
    args = _parser().parse_args()
    if args.command == "manifest":
        payload = build_manifest(
            _read_json(args.baseline_report), _read_jsonl(args.source_episodes)
        )
    elif args.command == "verify-host":
        payload = verify_host_report(_read_json(args.manifest), _read_json(args.host_report))
    else:
        payload = compare_verified_hosts(
            _read_json(args.manifest), _read_json(args.first), _read_json(args.second)
        )
    _write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
