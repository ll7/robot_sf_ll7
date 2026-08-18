#!/usr/bin/env python3
"""Run the repository WIP admission check immediately before PR creation.

This is a thin canonical PR-opener entry point. It deliberately delegates the decision to
``wip_capacity.py`` so issue claims, worktree starts, and PR publication cannot grow separate
capacity rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support the documented ``python3 scripts/dev/pr_open_preflight.py`` invocation.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dev import wip_capacity  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the PR-opening preflight parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", required=True, type=int)
    parser.add_argument("--repo", default=wip_capacity.DEFAULT_REPO)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--policy", default=str(wip_capacity.DEFAULT_POLICY_PATH))
    parser.add_argument("--mode", choices=sorted(wip_capacity.VALID_MODES), default="policy")
    parser.add_argument("--snapshot-file")
    parser.add_argument("--claims-file")
    parser.add_argument("--proposed-lane", choices=sorted(wip_capacity.VALID_LANES))
    parser.add_argument("--proposed-label", action="append", default=[])
    parser.add_argument("--exemption-file")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Delegate to the shared capacity decision."""
    args = build_parser().parse_args(argv)
    delegated = [
        "--repo",
        args.repo,
        "--remote",
        args.remote,
        "--policy",
        args.policy,
        "--mode",
        args.mode,
        "--proposed-issue",
        str(args.issue),
    ]
    for option, value in (
        ("--snapshot-file", args.snapshot_file),
        ("--claims-file", args.claims_file),
        ("--exemption-file", args.exemption_file),
    ):
        if value:
            delegated.extend((option, value))
    if args.proposed_lane:
        delegated.extend(("--proposed-lane", args.proposed_lane))
    for label in args.proposed_label:
        delegated.extend(("--proposed-label", label))
    if args.json:
        delegated.append("--json")
    return wip_capacity.main(delegated)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
