#!/usr/bin/env python3
"""Run and check the force-coupled potential-field paired diagnostic comparator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.force_coupled_comparator import (
    CANONICAL_CONFIG_PATH,
    FAILURE_CLASS_SIMULATOR,
    run_force_coupled_comparator,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CANONICAL_CONFIG_PATH),
        help="Path to force-coupled potential field configuration YAML.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write receipt JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print receipt JSON to stdout.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke check and assert status is ok.",
    )
    return parser.parse_args(argv)


def _simulator_error_row_labels(receipt: dict[str, Any]) -> list[str]:
    """Return labels for rows that carry an actual simulator-error signal.

    Generated receipts identify the taxonomy class explicitly. Legacy receipts may omit that
    additive field, so simulator reason text is also recognized. An explicit non-simulator class or
    a planner-owned ``plan_exception:``/``planner_diagnostic:`` prefix cannot mask an independent
    simulator signal; this mirrors the simulator-first taxonomy precedence.
    """
    results = receipt.get("results")
    if not isinstance(results, list):
        return []

    labels: list[str] = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        failure_class = row.get("failure_class")
        reasons = row.get("degradation_reasons")
        has_simulator_reason = (
            isinstance(reasons, list)
            and all(isinstance(reason, str) for reason in reasons)
            and any(
                "simulator" in reason.strip().lower() or "sim_error" in reason.strip().lower()
                for reason in reasons
            )
        )
        if failure_class == FAILURE_CLASS_SIMULATOR or has_simulator_reason:
            labels.append(f"{row.get('planner_id', '<unknown>')}/{row.get('scenario_id', index)}")
    return labels


def main(argv: list[str] | None = None) -> int:
    """Run comparator and report results.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    args = parse_args(argv)
    receipt = run_force_coupled_comparator(
        config_path=args.config if args.config.exists() else None,
        repo_root=args.repo_root,
    )

    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    if args.json or not args.smoke:
        print(text)

    if args.smoke:
        status = receipt.get("status")
        digest = receipt.get("receipt_digest")
        results = receipt.get("results")
        runs = len(results) if isinstance(results, list) else 0
        if status != "ok" or not digest or runs == 0:
            print(
                f"FAIL: force-coupled comparator check failed (status={status}, runs={runs})",
                file=sys.stderr,
            )
            return 1
        simulator_rows = _simulator_error_row_labels(receipt)
        if simulator_rows:
            print(
                "FAIL: force-coupled comparator check found simulator-error rows "
                f"({', '.join(simulator_rows)})",
                file=sys.stderr,
            )
            return 1
        print(f"PASS: force-coupled comparator check passed (runs={runs}, digest={digest[:16]}...)")

    return 0 if receipt.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
