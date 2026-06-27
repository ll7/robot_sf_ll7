"""CLI for the real micromobility trace validation-contract checker (issue #3278).

Checks a metadata-only candidate-dataset descriptor against the Robot SF
trace-failure predicate input contract and prints a structured report. It does
NOT ingest, copy, or read any external/private trace data and makes NO
real-world validation claim.

Example:
    uv run python scripts/tools/check_real_trace_validation_contract.py \
        --descriptor configs/benchmarks/issue_3278_real_trace_validation_contract_example.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.analysis_workbench.real_trace_validation_contract import (
    CONTRACT_STATUS_READY,
    RealTraceValidationContractError,
    check_real_trace_validation_contract,
    load_real_trace_validation_contract,
)
from robot_sf.analysis_workbench.trace_failure_predicates import load_trace_predicate_matrix

DEFAULT_DESCRIPTOR = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "issue_3278_real_trace_validation_contract_example.yaml"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor",
        type=Path,
        default=DEFAULT_DESCRIPTOR,
        help="Path to a real_trace_validation_contract.v1 descriptor (JSON or YAML).",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Optional trace_predicate_matrix.v1 YAML to union extra required fields.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless the contract status is 'ready'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the contract check and print a JSON report.

    Returns:
        Process exit code (0 on success, 2 on descriptor error, 1 when
        ``--require-ready`` is set and the contract is blocked).
    """
    args = _parse_args(argv)
    try:
        descriptor = load_real_trace_validation_contract(args.descriptor)
        matrix = load_trace_predicate_matrix(args.matrix) if args.matrix else None
        report = check_real_trace_validation_contract(
            descriptor, matrix=matrix, source=args.descriptor
        )
    except (RealTraceValidationContractError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.require_ready and report.contract_status != CONTRACT_STATUS_READY:
        print(
            f"contract status is {report.contract_status!r}, expected 'ready'",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
