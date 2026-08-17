#!/usr/bin/env python3
"""Replay frozen agent-figure interpretation evaluation fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.agent_figure_interpretation_eval import (
    AgentFigureEvalError,
    canonical_json,
    evaluate_manifest,
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "agent_figure_interpretation_eval"
    / "v1"
    / "manifest.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Digest-pinned fixture manifest to replay.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print indented JSON instead of canonical compact JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run fixture-only replay and print the evaluation JSON."""

    args = _parser().parse_args(argv)
    try:
        result = evaluate_manifest(args.manifest)
    except AgentFigureEvalError as exc:
        print(f"agent figure interpretation eval failed closed: {exc}", file=sys.stderr)
        return 2
    if args.pretty:
        import json

        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
