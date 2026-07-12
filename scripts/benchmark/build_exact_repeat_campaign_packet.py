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
    execute_campaign,
    resolve_runnable_definitions,
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
    resolve = subcommands.add_parser(
        "resolve-definitions", help="recover and hash-check all runnable target definitions"
    )
    resolve.add_argument("--manifest", type=Path, required=True)
    resolve.add_argument("--campaign-config", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    verify = subcommands.add_parser("verify-host", help="verify a completed host report")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--host-report", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    compare = subcommands.add_parser("compare-hosts", help="compare two verified host reports")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    execute = subcommands.add_parser(
        "execute", help="execute repeat cells and emit host_result.json"
    )
    execute.add_argument("--resolved-bundle", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="subset of scenario_id--seed to execute",
    )
    return parser


def main() -> int:
    """Parse a packet action and write its schema-checked JSON result."""
    args = _parser().parse_args()
    if args.command == "manifest":
        payload = build_manifest(
            _read_json(args.baseline_report), _read_jsonl(args.source_episodes)
        )
        _write_json(args.output, payload)
    elif args.command == "resolve-definitions":
        payload = resolve_runnable_definitions(_read_json(args.manifest), args.campaign_config)
        _write_json(args.output, payload)
    elif args.command == "verify-host":
        payload = verify_host_report(_read_json(args.manifest), _read_json(args.host_report))
        _write_json(args.output, payload)
    elif args.command == "execute":
        payload = execute_campaign(
            _read_json(args.resolved_bundle),
            output_dir=args.output_dir,
            target_filter=args.targets,
        )
    else:
        payload = compare_verified_hosts(
            _read_json(args.manifest), _read_json(args.first), _read_json(args.second)
        )
        _write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
