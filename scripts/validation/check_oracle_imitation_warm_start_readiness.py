"""Check oracle-imitation warm-start training prerequisites before any launch (issue #1496).

Read-only preflight: it validates the durable dataset launch packet (via the canonical
validator) and the training-side config/contract files named in a readiness manifest, then
prints a compact readiness report plus an explicit blocker list. It trains nothing, collects
no data, and submits no compute.

Example:
    uv run python scripts/validation/check_oracle_imitation_warm_start_readiness.py \\
        --manifest configs/training/imitation/oracle_warm_start_readiness.yaml --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.oracle_imitation_warm_start_readiness import (
    WarmStartReadinessError,
    check_warm_start_readiness,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Preflight oracle-imitation warm-start training prerequisites."
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Warm-start readiness manifest YAML path."
    )
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used to resolve relative paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON readiness report.")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero if any prerequisite blocker remains (fail-closed launch gate).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the readiness check and return a shell-friendly exit code.

    Returns ``0`` when ready, ``1`` when blocked (or when ``--require-ready`` rejects a
    blocked manifest), and ``2`` when the manifest itself is malformed.
    """
    args = build_arg_parser().parse_args(argv)
    try:
        report = check_warm_start_readiness(
            args.manifest,
            repo_root=args.repo_root,
            require_ready=args.require_ready,
        )
    except WarmStartReadinessError as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        # A blocked manifest under --require-ready is a gate rejection (exit 1); a structural
        # manifest error is exit 2. Distinguish via the error text prefix.
        return 1 if str(exc).startswith("oracle-imitation warm-start prerequisites") else 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"oracle-imitation warm-start readiness: {report['status']} "
            f"({report['experiment_id']}, {len(report['blockers'])} blockers)"
        )
        for blocker in report["blockers"]:
            print(f"  - {blocker}")
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
