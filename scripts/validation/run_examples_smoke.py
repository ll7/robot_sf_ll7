"""Execute the manifest-driven example smoke test via pytest.

The script coordinates ``tests/examples/test_examples_run.py`` and exposes a
``--dry-run`` flag to list the targeted examples without executing them. It also
contains a lightweight tracker-progress check that exercises the imitation
pipeline in tracker smoke mode so regressions in the progress output surface in
CI before the full suite runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import pytest

from robot_sf.examples.manifest_loader import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_EXAMPLE = Path("examples/advanced/16_imitation_learning_pipeline.py")
DEFAULT_PERF_SCENARIO = "configs/validation/minimal.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse args.

    Args:
        argv: Auto-generated placeholder description.

    Returns:
        argparse.Namespace: Auto-generated placeholder description.
    """
    parser = argparse.ArgumentParser(
        description="Run the manifest-driven examples smoke test (pytest harness).",
        epilog=(
            "Additional pytest arguments can be supplied after --pytest-args, e.g.\n"
            "  uv run python scripts/validation/run_examples_smoke.py --pytest-args -k quickstart\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List CI-enabled examples without invoking pytest.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=(),
        metavar="PYTEST_ARGS",
        help="Additional arguments forwarded directly to pytest.",
    )
    parser.add_argument(
        "--skip-tracker-check",
        action="store_true",
        help="Skip the imitation pipeline tracker smoke check.",
    )
    parser.add_argument(
        "--tracker-check-only",
        action="store_true",
        help="Run only the tracker progress check and skip pytest execution.",
    )
    parser.add_argument(
        "--skip-perf-tests",
        action="store_true",
        help="Skip the telemetry performance wrapper.",
    )
    parser.add_argument(
        "--perf-tests-only",
        action="store_true",
        help="Run the telemetry performance wrapper (plus tracker check) and skip pytest execution.",
    )
    parser.add_argument(
        "--perf-scenario",
        default=DEFAULT_PERF_SCENARIO,
        help=(
            "Scenario config passed to the telemetry performance wrapper"
            f" (default: {DEFAULT_PERF_SCENARIO})."
        ),
    )
    parser.add_argument(
        "--perf-num-resets",
        type=int,
        default=3,
        help="Number of environment resets benchmarked by the telemetry performance wrapper (default: 3).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main.

    Args:
        argv: Auto-generated placeholder description.

    Returns:
        int: Auto-generated placeholder description.
    """
    args = parse_args(argv)
    manifest = load_manifest(validate_paths=True)
    ci_examples = tuple(manifest.iter_ci_enabled_examples())

    tracker_ok = _maybe_run_tracker_progress_check(args)
    if not tracker_ok:
        return 1
    if args.tracker_check_only:
        return 0

    perf_ok = _maybe_run_perf_tests(args)
    if not perf_ok:
        return 1
    if args.perf_tests_only:
        return 0

    if args.dry_run:
        if not ci_examples:
            print("No CI-enabled examples declared in manifest.")
        else:
            print("CI-enabled examples (dry run):")
            for example in ci_examples:
                tag_suffix = f" [{', '.join(example.tags)}]" if example.tags else ""
                print(f" - {example.path.as_posix()}{tag_suffix}")
            print(f"Total: {len(ci_examples)} example(s)")
        return 0

    pytest_args = ["tests/examples/test_examples_run.py"]
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    return int(pytest.main(pytest_args))


def _maybe_run_tracker_progress_check(args: argparse.Namespace) -> bool:
    """Maybe run tracker progress check.

    Args:
        args: Auto-generated placeholder description.

    Returns:
        bool: Auto-generated placeholder description.
    """
    if args.dry_run or args.skip_tracker_check:
        return True
    script_path = _resolve_pipeline_example()
    if script_path is None:
        print("Tracker progress check skipped: pipeline example not found.")
        return True
    env = _build_example_env()
    env["ROBOT_SF_ENABLE_PROGRESS_TRACKER"] = "1"
    env["ROBOT_SF_TRACKER_SMOKE"] = "1"
    command = [sys.executable, str(script_path), "--enable-tracker", "--tracker-smoke"]
    print("Running tracker progress check:", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive path
        tail = _tail_output(exc.stdout, exc.stderr)
        print("Tracker progress check timed out. Output tail:\n" + tail)
        return False
    if completed.returncode != 0:
        tail = _tail_output(completed.stdout, completed.stderr)
        print(
            "Tracker progress check failed with exit code"
            f" {completed.returncode}. Output tail:\n{tail}"
        )
        return False
    try:
        _assert_progress_output(completed.stdout, completed.stderr)
    except AssertionError as exc:
        tail = _tail_output(completed.stdout, completed.stderr)
        print(f"{exc}\nOutput tail:\n{tail}")
        return False
    print("Tracker progress check passed.")
    return True


def _resolve_pipeline_example() -> Path | None:
    """Resolve pipeline example.

    Returns:
        Path | None: Auto-generated placeholder description.
    """
    candidate = REPO_ROOT / PIPELINE_EXAMPLE
    if candidate.is_file():
        return candidate
    return None


def _build_example_env() -> dict[str, str]:
    """Build example env.

    Returns:
        dict[str, str]: Auto-generated placeholder description.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("DISPLAY", "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env["PYTHONPATH"] = _merge_pythonpath(REPO_ROOT, env.get("PYTHONPATH"))
    env.setdefault("ROBOT_SF_FAST_DEMO", "1")
    env.setdefault("ROBOT_SF_EXAMPLES_MAX_STEPS", "64")
    return env


def _maybe_run_perf_tests(args: argparse.Namespace) -> bool:
    """Maybe run perf tests.

    Args:
        args: Auto-generated placeholder description.

    Returns:
        bool: Auto-generated placeholder description.
    """
    if args.dry_run or args.skip_perf_tests:
        return True
    try:
        from scripts.telemetry.run_perf_tests import run_perf_tests
    except ModuleNotFoundError as exc:  # pragma: no cover - guard rail for partial installs
        print(f"Telemetry perf wrapper unavailable: {exc}")
        return False

    scenario = args.perf_scenario or DEFAULT_PERF_SCENARIO
    num_resets = max(1, args.perf_num_resets)
    print(
        f"Running telemetry perf wrapper (scenario={scenario}, num_resets={num_resets})",
    )
    try:
        exit_code, run_dir = run_perf_tests(
            scenario=scenario,
            output_hint=None,
            num_resets=num_resets,
        )
    except Exception as exc:
        print(f"Telemetry perf wrapper failed: {exc}")
        return False
    if run_dir is not None:
        print(f"Telemetry perf wrapper artifacts: {run_dir}")
    if exit_code != 0:
        print(f"Telemetry perf wrapper exited with {exit_code}")
        return False
    print("Telemetry perf wrapper passed.")
    return True


def _merge_pythonpath(root: Path, existing: str | None) -> str:
    """Merge pythonpath.

    Args:
        root: Auto-generated placeholder description.
        existing: Auto-generated placeholder description.

    Returns:
        str: Auto-generated placeholder description.
    """
    parts: list[str] = [str(root)]
    if existing:
        parts.extend(element for element in existing.split(os.pathsep) if element)
    ordered: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in seen:
            seen.add(part)
            ordered.append(part)
    return os.pathsep.join(ordered)


def _assert_progress_output(stdout: str | None, stderr: str | None) -> None:
    """Assert progress output.

    Args:
        stdout: Auto-generated placeholder description.
        stderr: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    combined = "\n".join(part for part in (stdout, stderr) if part)
    lowered = combined.lower()
    if "step 1/" not in lowered or "eta=" not in lowered:
        raise AssertionError(
            "Progress tracker output missing expected 'Step 1/' or 'eta=' markers."
        )


def _tail_output(stdout: str | None, stderr: str | None, limit: int = 20) -> str:
    """Tail output.

    Args:
        stdout: Auto-generated placeholder description.
        stderr: Auto-generated placeholder description.
        limit: Auto-generated placeholder description.

    Returns:
        str: Auto-generated placeholder description.
    """
    combined = "\n".join(part for part in (stdout, stderr) if part)
    lines = [line.rstrip() for line in combined.splitlines()]
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[-limit:])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
