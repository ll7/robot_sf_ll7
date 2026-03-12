#!/usr/bin/env python3
"""Resolve a pytest-xdist worker count that is safe for the current host."""

from __future__ import annotations

import argparse
import os
import platform
import sys

MACOS_MAX_WORKERS = 8
MACOS_MIN_WORKERS = 2


def _resolve_worker_spec(
    *,
    requested: str | None,
    cpu_count: int | None,
    system: str,
) -> tuple[str, str]:
    """Return the xdist worker spec and a short explanation."""
    requested_value = requested.strip() if requested else ""
    if requested_value:
        if requested_value == "auto":
            return "auto", "explicit override via PYTEST_NUM_WORKERS=auto"
        workers = int(requested_value)
        if workers <= 0:
            raise ValueError("PYTEST_NUM_WORKERS must be a positive integer or 'auto'")
        return str(workers), "explicit override via PYTEST_NUM_WORKERS"

    logical_cpus = max(1, int(cpu_count or 1))
    normalized_system = system.lower()
    if normalized_system == "darwin":
        workers = max(MACOS_MIN_WORKERS, min(MACOS_MAX_WORKERS, logical_cpus // 2 or 1))
        return (
            str(workers),
            f"macOS-safe default derived from {logical_cpus} logical CPUs "
            f"(cap={MACOS_MAX_WORKERS}, floor={MACOS_MIN_WORKERS})",
        )

    return "auto", f"default xdist auto worker count on {system}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requested",
        help="Optional requested worker override; otherwise PYTEST_NUM_WORKERS is used.",
    )
    parser.add_argument(
        "--show-reason",
        action="store_true",
        help="Print a human-readable explanation to stderr.",
    )
    return parser


def main() -> int:
    """Resolve and print the worker spec used by scripts/dev pytest wrappers."""
    args = _build_parser().parse_args()
    requested = (
        args.requested if args.requested is not None else os.environ.get("PYTEST_NUM_WORKERS")
    )
    try:
        workers, reason = _resolve_worker_spec(
            requested=requested,
            cpu_count=os.cpu_count(),
            system=platform.system(),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(workers)
    if args.show_reason:
        print(reason, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
