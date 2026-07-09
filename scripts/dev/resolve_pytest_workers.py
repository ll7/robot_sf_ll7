#!/usr/bin/env python3
"""Resolve a pytest-xdist worker count that is safe for the current host."""

from __future__ import annotations

import argparse
import os
import platform
import sys

MACOS_MAX_WORKERS = 8
MACOS_MIN_WORKERS = 2

# GitHub Actions ubuntu-latest runners (and other compact CI hosts) have 2 vCPUs
# and limited memory. Spawning more workers than this cap will OOM the runner.
# Only applies to low-CPU hosts (fewer than 8 logical CPUs); high-CPU hosts keep
# the explicit value so local CI or larger runners are unaffected.
LOW_CPU_WORKER_CAP = 16
LOW_CPU_THRESHOLD = 8


def _cap_workers_for_host(
    *,
    requested: int,
    cpu_count: int,
    system: str,
) -> tuple[int, str]:
    """Apply host-resource caps to an explicit worker request.

    Returns the possibly-capped worker count and an explanation of any change.
    """
    normalized_system = system.lower()
    if normalized_system == "darwin":
        capped = max(MACOS_MIN_WORKERS, min(MACOS_MAX_WORKERS, requested))
        if capped != requested:
            return capped, (
                f"capped explicit from {requested} to {capped} "
                f"(macOS max={MACOS_MAX_WORKERS}, floor={MACOS_MIN_WORKERS})"
            )

    if cpu_count < LOW_CPU_THRESHOLD:
        capped = min(requested, LOW_CPU_WORKER_CAP)
        if capped != requested:
            return capped, (
                f"capped explicit from {requested} to {capped} "
                f"(low-CPU host with {cpu_count} CPUs, max={LOW_CPU_WORKER_CAP})"
            )

    return requested, ""


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
        try:
            workers = int(requested_value)
        except ValueError:
            raise ValueError("PYTEST_NUM_WORKERS must be a positive integer or 'auto'") from None
        if workers <= 0:
            raise ValueError("PYTEST_NUM_WORKERS must be a positive integer or 'auto'") from None

        logical_cpus = max(1, int(cpu_count or 1))
        capped, cap_reason = _cap_workers_for_host(
            requested=workers,
            cpu_count=logical_cpus,
            system=system,
        )
        if cap_reason:
            return str(capped), f"explicit override ({cap_reason})"
        return str(capped), "explicit override via PYTEST_NUM_WORKERS"

    logical_cpus = max(1, int(cpu_count or 1))
    normalized_system = system.lower()
    if normalized_system == "darwin":
        workers = max(MACOS_MIN_WORKERS, min(MACOS_MAX_WORKERS, max(1, logical_cpus // 2)))
        return (
            str(workers),
            f"macOS-safe default derived from {logical_cpus} logical CPUs "
            f"(cap={MACOS_MAX_WORKERS}, floor={MACOS_MIN_WORKERS})",
        )

    return "auto", f"default xdist auto worker count on {system}"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for pytest worker resolution.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
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
