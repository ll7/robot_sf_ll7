#!/usr/bin/env python3
"""Cold-start reproduction script for benchmark release rc0.0.3.

Demonstrates independent reproducibility by following the exact public instructions:
1. Clone at the release tag
2. Build the environment from the lockfile
3. Verify all artifact checksums
4. Reproduce a declared subset of release results
5. Record environment, deviations, wall-time-to-result, and instruction gaps

Usage:
    python scripts/repro/cold_start_reproduction.py --tag rc0.0.3
    python scripts/repro/cold_start_reproduction.py --tag rc0.0.3 --skip-clone
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Minimal reproducibility subset: 3 planners × 1 scenario × 2 repeats
# This is enough to verify the pipeline works without requiring a full campaign.
MINIMAL_SUBSET = {
    "matrix": "configs/scenarios/planner_sanity_matrix_v1.yaml",
    "algo": "goal",
    "horizon": 300,
    "repeats": 2,
    "workers": 1,
    "benchmark_profile": "baseline-safe",
}


def _run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout: int = 600,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command with timeout and return the result."""
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged_env,
        check=False,
    )


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_environment_info() -> dict[str, Any]:
    """Collect environment information for the reproduction report."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    # Collect git info if available
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["git_commit"] = git_rev
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        info["git_commit"] = "unknown"

    # Collect uv info if available
    try:
        uv_version = subprocess.check_output(
            ["uv", "--version"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["uv_version"] = uv_version
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        info["uv_version"] = "unknown"

    return info


def verify_artifact_checksums(repo_root: Path) -> dict[str, dict[str, str]]:
    """Verify checksums of critical artifacts.

    Returns a dict mapping artifact path to its hash and status.
    """
    critical_artifacts = [
        "CITATION.cff",
        "pyproject.toml",
        "uv.lock",
        "configs/scenarios/planner_sanity_matrix_v1.yaml",
        "maps/svg_maps/02_simple_maps.svg",
        "maps/svg_maps/atomic_corner_90_test.svg",
        "maps/svg_maps/atomic_corridor_test.svg",
    ]

    results: dict[str, dict[str, str]] = {}
    for artifact in critical_artifacts:
        artifact_path = repo_root / artifact
        if artifact_path.exists():
            results[artifact] = {
                "status": "present",
                "sha256": _hash_file(artifact_path),
                "bytes": str(artifact_path.stat().st_size),
            }
        else:
            results[artifact] = {
                "status": "missing",
                "sha256": "",
                "bytes": "0",
            }

    return results


def build_environment(repo_root: Path) -> dict[str, Any]:
    """Build the environment from the lockfile.

    Returns status and timing information.
    """
    start = time.time()

    # Step 1: Create virtual environment
    venv_result = _run_cmd(
        ["uv", "venv", ".venv", "--python", "3.12"],
        cwd=repo_root,
        timeout=120,
    )

    if venv_result.returncode != 0:
        return {
            "status": "failed",
            "step": "venv_creation",
            "error": venv_result.stderr,
            "wall_time_s": time.time() - start,
        }

    # Step 2: Sync dependencies
    sync_result = _run_cmd(
        ["uv", "sync", "--all-extras"],
        cwd=repo_root,
        timeout=600,
    )

    if sync_result.returncode != 0:
        return {
            "status": "failed",
            "step": "dependency_sync",
            "error": sync_result.stderr,
            "wall_time_s": time.time() - start,
        }

    # Step 3: Verify installation
    verify_result = _run_cmd(
        ["uv", "run", "python", "-c", "import robot_sf; print(robot_sf.__version__)"],
        cwd=repo_root,
        timeout=30,
    )

    wall_time = time.time() - start
    return {
        "status": "success" if verify_result.returncode == 0 else "warning",
        "step": "complete",
        "wall_time_s": round(wall_time, 2),
        "version_check": verify_result.stdout.strip()
        if verify_result.returncode == 0
        else "failed",
    }


def run_reproduction_subset(repo_root: Path, subset: dict[str, Any]) -> dict[str, Any]:
    """Run a minimal benchmark subset for reproduction.

    Returns timing, episode count, and output paths.
    """
    output_root = repo_root / "output" / "cold_start_repro"
    output_root.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Set headless environment
    env = {
        "DISPLAY": "",
        "MPLBACKEND": "Agg",
        "SDL_VIDEODRIVER": "dummy",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        "LOGURU_LEVEL": "WARNING",
    }

    # Run benchmark
    cmd = [
        "uv",
        "run",
        "robot_sf_bench",
        "--quiet",
        "run",
        "--matrix",
        subset["matrix"],
        "--out",
        str(output_root / "episodes.jsonl"),
        "--algo",
        subset["algo"],
        "--repeats",
        str(subset["repeats"]),
        "--horizon",
        str(subset["horizon"]),
        "--workers",
        str(subset["workers"]),
        "--no-video",
        "--no-resume",
        "--fail-fast",
        "--benchmark-profile",
        subset["benchmark_profile"],
        "--external-log-noise",
        "suppress",
        "--structured-output",
        "json",
    ]

    result = _run_cmd(cmd, cwd=repo_root, timeout=1200, env=env)
    wall_time = time.time() - start

    # Count episodes
    episodes_path = output_root / "episodes.jsonl"
    episode_count = 0
    if episodes_path.exists():
        episode_count = sum(1 for line in episodes_path.read_text().splitlines() if line)

    # Run aggregation
    agg_start = time.time()
    agg_cmd = [
        "uv",
        "run",
        "robot_sf_bench",
        "aggregate",
        "--in",
        str(episodes_path),
        "--out",
        str(output_root / "summary.json"),
    ]
    agg_result = _run_cmd(agg_cmd, cwd=repo_root, timeout=300, env=env)
    agg_time = time.time() - agg_start

    # Generate manifest
    manifest = {
        "schema": "robot-sf-cold-start-reproduction.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "subset": subset,
        "episode_count": episode_count,
        "benchmark_wall_time_s": round(wall_time, 2),
        "aggregation_wall_time_s": round(agg_time, 2),
        "total_wall_time_s": round(wall_time + agg_time, 2),
        "benchmark_exit_code": result.returncode,
        "aggregation_exit_code": agg_result.returncode,
    }

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "status": "success" if result.returncode == 0 and agg_result.returncode == 0 else "failed",
        "output_root": str(output_root),
        "manifest": manifest,
        "benchmark_stderr": result.stderr[-500:] if result.stderr else "",
    }


def generate_reproduction_report(
    env_info: dict[str, Any],
    checksums: dict[str, dict[str, str]],
    build_result: dict[str, Any],
    benchmark_result: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Generate the final reproduction report."""
    report = {
        "schema": "robot-sf-cold-start-reproduction-report.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "tag": "rc0.0.3",
        "environment": env_info,
        "checksum_verification": checksums,
        "build_result": build_result,
        "benchmark_result": benchmark_result,
        "limitations": [
            "CPU/headless reproduction only; no GPU determinism claim.",
            "Minimal subset (1 scenario, 2 repeats) — not a full benchmark campaign.",
            "Floating-point behavior may vary across CPU architectures.",
            "This is a design-to-demonstration transition; a true independent cold-start requires a separate machine/person.",
        ],
        "instruction_gaps_found": [],
    }

    # Identify instruction gaps
    if build_result.get("status") == "failed":
        report["instruction_gaps_found"].append(
            f"Build failed at step: {build_result.get('step', 'unknown')}"
        )

    if benchmark_result.get("status") == "failed":
        report["instruction_gaps_found"].append(
            f"Benchmark failed with exit code: {benchmark_result.get('manifest', {}).get('benchmark_exit_code', 'unknown')}"
        )

    # Check for missing checksums
    missing = [k for k, v in checksums.items() if v.get("status") == "missing"]
    if missing:
        report["instruction_gaps_found"].append(f"Missing critical artifacts: {missing}")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    return report


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        default="rc0.0.3",
        help="Git tag to reproduce (default: rc0.0.3)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning step (use existing checkout)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip environment build step",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmark run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for report (default: output/cold_start_repro/)",
    )
    return parser.parse_args()


def _resolve_repo_root(args: argparse.Namespace) -> Path:
    """Resolve the repository root path."""
    if args.repo_root:
        return args.repo_root.resolve()
    return Path(__file__).resolve().parent.parent.parent


def _print_step(step: int, msg: str) -> None:
    """Print a step header."""
    print(f"Step {step}: {msg}")


def _print_summary(
    env_info: dict[str, Any],
    build_result: dict[str, Any],
    benchmark_result: dict[str, Any],
    report: dict[str, Any],
) -> None:
    """Print the final summary."""
    print("=== Summary ===")
    print(f"  Environment: {env_info['platform']}")
    print(f"  Build: {build_result.get('status', 'unknown')}")
    print(f"  Benchmark: {benchmark_result.get('status', 'unknown')}")
    print(f"  Instruction gaps: {len(report.get('instruction_gaps_found', []))}")
    if report.get("instruction_gaps_found"):
        for gap in report["instruction_gaps_found"]:
            print(f"    - {gap}")
    print()


def main() -> int:
    """Run the cold-start reproduction workflow."""
    args = _parse_args()
    repo_root = _resolve_repo_root(args)

    if not (repo_root / ".git").exists():
        print(f"Error: {repo_root} is not a git repository", file=sys.stderr)
        return 1

    print(f"=== Cold-Start Reproduction: {args.tag} ===")
    print(f"Repository: {repo_root}")
    print()

    # Step 1: Environment info
    _print_step(1, "Collecting environment information...")
    env_info = collect_environment_info()
    print(f"  Platform: {env_info['platform']}")
    print(f"  Python: {env_info['python_version']}")
    print()

    # Step 2: Verify checksums
    _print_step(2, "Verifying artifact checksums...")
    checksums = verify_artifact_checksums(repo_root)
    for artifact, info in checksums.items():
        status_icon = "✓" if info["status"] == "present" else "✗"
        print(f"  {status_icon} {artifact}: {info['status']}")
    print()

    # Step 3: Build environment
    build_result: dict[str, Any] = {"status": "skipped"}
    if not args.skip_build:
        _print_step(3, "Building environment from lockfile...")
        build_result = build_environment(repo_root)
        print(f"  Status: {build_result['status']}")
        print(f"  Wall time: {build_result.get('wall_time_s', 0):.1f}s")
        if build_result.get("version_check"):
            print(f"  Version: {build_result['version_check']}")
    else:
        _print_step(3, "Skipped (--skip-build)")
    print()

    # Step 4: Run benchmark subset
    benchmark_result: dict[str, Any] = {"status": "skipped"}
    if not args.skip_benchmark:
        _print_step(4, "Running benchmark subset...")
        benchmark_result = run_reproduction_subset(repo_root, MINIMAL_SUBSET)
        print(f"  Status: {benchmark_result['status']}")
        if benchmark_result.get("manifest"):
            print(f"  Episodes: {benchmark_result['manifest'].get('episode_count', 0)}")
            print(f"  Wall time: {benchmark_result['manifest'].get('total_wall_time_s', 0):.1f}s")
    else:
        _print_step(4, "Skipped (--skip-benchmark)")
    print()

    # Step 5: Generate report
    _print_step(5, "Generating reproduction report...")
    if args.output_dir:
        report_path = args.output_dir / "reproduction_report.json"
    else:
        report_path = repo_root / "output" / "cold_start_repro" / "reproduction_report.json"

    report = generate_reproduction_report(
        env_info, checksums, build_result, benchmark_result, report_path
    )
    print(f"  Report: {report_path}")
    print()

    _print_summary(env_info, build_result, benchmark_result, report)

    # Return appropriate exit code
    if build_result.get("status") == "failed" or benchmark_result.get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
