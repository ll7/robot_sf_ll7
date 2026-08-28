#!/usr/bin/env python3
"""Resolve a pytest-xdist worker count that is safe for the current host and lane."""

from __future__ import annotations

import argparse
import os
import platform
import sys

MACOS_MAX_WORKERS = 8
MACOS_MIN_WORKERS = 2

# GitHub Actions ubuntu-latest runners are compact shared hosts (currently
# 4 vCPUs, ~16 GB RAM per the Nightly Performance run logs). Spawning far more
# xdist workers than cores — e.g. the old hardcoded 32 on a 4-core runner —
# saturates runner memory and triggers GitHub's runner eviction: the job
# receives a "shutdown signal" and exits 143 (SIGTERM), which is what repeatedly
# killed the Nightly Performance xdist-race job. (Note: this is the runner
# watchdog reclaiming the VM under pressure, not the kernel OOM-killer, which
# would deliver SIGKILL / exit 137.) Cap explicit integer overrides on low-CPU
# hosts; high-CPU hosts keep the explicit value so local CI and larger runners
# are unaffected. "auto" always bypasses these caps.
LOW_CPU_WORKER_CAP = 16
LOW_CPU_THRESHOLD = 8
CUDA_SAFE_DEFAULT_WORKERS = 1
CUDA_LANES = frozenset(("optional", "all"))
CUDA_RUNTIME_STATUSES = frozenset(("usable", "unavailable", "unusable_nvml", "unknown"))


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
        # macOS uses its own explicit caps; return early so the low-CPU cap
        # below can never shadow the platform-specific strategy if the two
        # constants are ever reordered.
        return requested, ""

    if cpu_count < LOW_CPU_THRESHOLD:
        capped = min(requested, LOW_CPU_WORKER_CAP)
        if capped != requested:
            return capped, (
                f"capped explicit from {requested} to {capped} "
                f"(low-CPU host with {cpu_count} CPUs, max={LOW_CPU_WORKER_CAP})"
            )

    return requested, ""


def _probe_cuda_runtime() -> tuple[str, str]:
    """Return the shared CUDA classification for worker-policy decisions."""
    try:
        from robot_sf.telemetry.gpu import classify_cuda_runtime

        classification = classify_cuda_runtime()
    except (ImportError, OSError) as exc:
        return "unknown", f"CUDA runtime probe failed: {exc}"
    return classification.status, classification.reason


def _resolve_cuda_runtime(
    *,
    lane: str,
    cuda_runtime: tuple[str, str] | None,
) -> tuple[str, str] | None:
    """Resolve the CUDA status relevant to a test lane, if any."""
    if lane not in CUDA_LANES:
        return None
    if cuda_runtime is not None:
        return cuda_runtime
    return _probe_cuda_runtime()


def _resolve_default_worker_spec(
    *,
    lane: str,
    logical_cpus: int,
    system: str,
    cuda_runtime: tuple[str, str] | None,
) -> tuple[str, str]:
    """Resolve a non-explicit worker request for the selected host and readiness lane."""
    normalized_system = system.lower()
    cuda_resolution = _resolve_cuda_runtime(lane=lane, cuda_runtime=cuda_runtime)
    if cuda_resolution is not None:
        cuda_status, cuda_reason = cuda_resolution
        runtime_summary = f"CUDA runtime={cuda_status} ({cuda_reason})"
        if cuda_status == "usable":
            return (
                str(CUDA_SAFE_DEFAULT_WORKERS),
                "CUDA-safe default for GPU-spawning readiness lane "
                f"(in-process serial); {runtime_summary}",
            )
        if cuda_status == "unknown":
            return (
                str(CUDA_SAFE_DEFAULT_WORKERS),
                f"safe serial default because CUDA capability is uncertain; {runtime_summary}",
            )
        if normalized_system != "darwin":
            return "auto", f"default xdist auto worker count on {system}; {runtime_summary}"

    if normalized_system == "darwin":
        workers = max(MACOS_MIN_WORKERS, min(MACOS_MAX_WORKERS, logical_cpus // 2))
        reason = (
            f"macOS-safe default derived from {logical_cpus} logical CPUs "
            f"(cap={MACOS_MAX_WORKERS}, floor={MACOS_MIN_WORKERS})"
        )
        if cuda_resolution is not None:
            reason = f"{reason}; {runtime_summary}"
        return str(workers), reason

    return "auto", f"default xdist auto worker count on {system}"


def _resolve_worker_spec(
    *,
    requested: str | None,
    cpu_count: int | None,
    system: str,
    lane: str = "core",
    cuda_runtime: tuple[str, str] | None = None,
) -> tuple[str, str]:
    """Return the xdist worker spec and a short explanation."""
    normalized_lane = lane.strip().lower()
    if normalized_lane not in {"core", "optional", "all"}:
        raise ValueError("lane must be one of: core, optional, all")

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

    return _resolve_default_worker_spec(
        lane=normalized_lane,
        logical_cpus=max(1, int(cpu_count or 1)),
        system=system,
        cuda_runtime=cuda_runtime,
    )


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
    parser.add_argument(
        "--lane",
        choices=("core", "optional", "all"),
        default="core",
        help="Readiness lane whose worker policy is being resolved (default: core).",
    )
    parser.add_argument(
        "--cuda-runtime",
        choices=tuple(sorted(CUDA_RUNTIME_STATUSES | {"auto"})),
        default="auto",
        help="Use a supplied CUDA classification, or probe it when set to auto.",
    )
    return parser


def main() -> int:
    """Resolve and print the worker spec used by scripts/dev pytest wrappers."""
    args = _build_parser().parse_args()
    requested = (
        args.requested if args.requested is not None else os.environ.get("PYTEST_NUM_WORKERS")
    )
    try:
        cuda_runtime = None
        if args.cuda_runtime != "auto":
            cuda_runtime = (
                args.cuda_runtime,
                f"provided via --cuda-runtime={args.cuda_runtime}",
            )
        workers, reason = _resolve_worker_spec(
            requested=requested,
            cpu_count=os.cpu_count(),
            system=platform.system(),
            lane=args.lane,
            cuda_runtime=cuda_runtime,
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
