"""Analyze the issue #5303 diagnostic search-stage outcome accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.issue_5303_search_promotion_analysis import (
    DEFAULT_CONTRACT_PATH,
    analyze_issue_5303_search_promotion,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the diagnostic-accounting CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Frozen issue #5303 contract YAML.",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        required=True,
        help="Complete per-attempt diagnostic JSONL emitted by the frozen runner.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve repository-relative paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON result path. The command still prints the payload.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate diagnostic accounting and write a fixed inconclusive analysis result."""
    args = parse_args(argv)
    result = analyze_issue_5303_search_promotion(
        args.outcomes,
        contract_path=args.contract,
        repo_root=args.repo_root,
    )
    payload = result.to_payload()
    if args.output is not None:
        output = (
            args.output if args.output.is_absolute() else args.repo_root.resolve() / args.output
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
