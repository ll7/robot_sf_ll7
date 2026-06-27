"""CLI for the learned probabilistic graph predictor v1 capability inventory (issue #2844).

Read-only preflight. Enumerates the *code-level* prerequisites a v1 learned predictor
would extend and reports whether each hook is present in the current checkout. It does
not implement, train, or run a predictor, and it never claims the training lane is
unblocked — that decision is owned by the readiness evidence gate
(scripts/validation/validate_learned_prediction_readiness.py).

Exit codes:
  0  every capability hook is present (wiring is in place; NOT a training-unblock claim).
  2  one or more capability hooks are missing (a wiring blocker to fix first).

A clean exit (0) means the surfaces to extend exist, matching the 2026-06-23 audit that
found the lane blocked on evidence rather than missing wiring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.learned_predictor_capability_inventory import (
    LEARNED_PREDICTOR_V1_HOOKS,
    build_inventory,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inventory code-level prerequisites for the learned probabilistic graph "
            "predictor v1 lane (read-only preflight; does not unblock training)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving file-based hooks (default: cwd).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON inventory report instead of human-readable text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the capability inventory and return a shell-friendly exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = build_inventory(LEARNED_PREDICTOR_V1_HOOKS, repo_root=args.repo_root.resolve())

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = report["summary"]
        verdict = "COMPLETE" if report["capability_status"] == "complete" else "INCOMPLETE"
        print(f"learned-predictor v1 capability wiring: {verdict}")
        print(f" present {summary['present']}/{summary['total']} hooks")
        for hook in report["hooks"]:
            mark = "ok " if hook["present"] else "MISS"
            print(f" [{mark}] {hook['category']:13s} {hook['name']}: {hook['detail']}")
        print(
            f" unblocks_training: {report['unblocks_training']} "
            f"(unblock owned by {report['unblock_owner']})"
        )

    return 0 if report["capability_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
