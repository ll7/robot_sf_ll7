"""Build the strict issue #6814 provenance trace packet."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from robot_sf.benchmark.issue_6814_trace_reexport import (
    Issue6814SourceIntegrityError,
    build_issue_6814_trace_packet,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument(
        "--arm-root",
        action="append",
        required=True,
        metavar="ARM=PATH",
        help="Approved external arm root; repeat once for each arm.",
    )
    parser.add_argument("--external-output-root", type=Path, required=True)
    parser.add_argument("--compact-output", type=Path, default=None)
    parser.add_argument("--execution-repository", type=Path, default=None)
    parser.add_argument("--check-determinism", action="store_true")
    return parser


def _arm_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--arm-root must use ARM=PATH: {value!r}")
        arm, path = value.split("=", 1)
        if not arm or not path or arm in roots:
            raise ValueError(f"invalid or duplicate --arm-root: {value!r}")
        roots[arm] = Path(path)
    return roots


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build the packet, and return the contract exit code."""

    args = _parser().parse_args(argv)
    try:
        arm_roots = _arm_roots(args.arm_root)
    except ValueError as exc:
        logger.error("{}", exc)
        return 2
    try:
        manifest = build_issue_6814_trace_packet(
            package_root=args.source_package,
            arm_roots=arm_roots,
            external_output_root=args.external_output_root,
            compact_output=args.compact_output,
            execution_repository=args.execution_repository,
            check_determinism=args.check_determinism,
        )
    except (ValueError, OSError) as exc:
        logger.error("{}", exc)
        return 2 if isinstance(exc, Issue6814SourceIntegrityError) else 1
    print(f"issue #6814 packet disposition: {manifest['disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
