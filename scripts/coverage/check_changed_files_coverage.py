#!/usr/bin/env python3
"""Check coverage for files changed relative to a base ref.

This tool is intended for PR readiness checks. It:
- Detects files changed vs a base ref (default: origin/main).
- Filters to included patterns (default: robot_sf/**/*.py).
- Reads coverage.json and enforces per-file coverage thresholds.

Goal: 100% coverage on changed files. Minimum requirement: 80%.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _repo_root() -> Path:
    return Path(_run(["git", "rev-parse", "--show-toplevel"]))


def _changed_files(base: str, repo_root: Path) -> list[Path]:
    output = _run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{base}...HEAD"],
        cwd=repo_root,
    )
    files = [Path(line.strip()) for line in output.splitlines() if line.strip()]
    return files


def _normalize_path(path: Path, repo_root: Path) -> str:
    if path.is_absolute():
        try:
            path = path.relative_to(repo_root)
        except ValueError:
            pass
    return path.as_posix()


def _matches_any(path_str: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(path_str, pattern) for pattern in patterns)


def _resolve_coverage(
    path_str: str,
    coverage_index: dict[str, float],
) -> tuple[float | None, str | None]:
    if path_str in coverage_index:
        return coverage_index[path_str], path_str
    matches = [(k, v) for k, v in coverage_index.items() if k.endswith(path_str)]
    if len(matches) == 1:
        key, value = matches[0]
        return value, key
    return None, None


def _load_coverage_index(coverage_path: Path, repo_root: Path) -> dict[str, float]:
    data = json.loads(coverage_path.read_text(encoding="utf-8"))
    index: dict[str, float] = {}
    for file_path, file_data in data.get("files", {}).items():
        summary = file_data.get("summary", {})
        percent = summary.get("percent_covered", 0.0)
        try:
            percent_value = float(percent)
        except (TypeError, ValueError):
            percent_value = 0.0
        normalized = _normalize_path(Path(file_path), repo_root)
        index[normalized] = percent_value
    return index


def _print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check per-file coverage for changed files relative to a base ref.",
    )
    parser.add_argument("--base", default="origin/main", help="Base ref to diff against")
    parser.add_argument(
        "--coverage",
        default="output/coverage/coverage.json",
        help="Path to coverage.json",
    )
    parser.add_argument("--min", type=float, default=80.0, help="Minimum required coverage")
    parser.add_argument("--goal", type=float, default=100.0, help="Target coverage goal")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob pattern(s) to include (can be repeated)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern(s) to exclude (can be repeated)",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print files skipped by include/exclude filters",
    )
    return parser.parse_args()


def _resolve_coverage_path(coverage_arg: str, repo_root: Path) -> Path:
    coverage_path = Path(coverage_arg)
    if not coverage_path.is_absolute():
        coverage_path = repo_root / coverage_path
    return coverage_path


def _select_changed_files(
    changed_files: list[Path],
    repo_root: Path,
    include_patterns: Iterable[str],
    exclude_patterns: Iterable[str],
) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    skipped: list[str] = []
    for path in changed_files:
        if not (repo_root / path).exists():
            continue
        path_str = _normalize_path(path, repo_root)
        if _matches_any(path_str, include_patterns) and not _matches_any(
            path_str,
            exclude_patterns,
        ):
            selected.append(path_str)
        else:
            skipped.append(path_str)
    return selected, skipped


def _build_results(
    selected: list[str],
    coverage_index: dict[str, float],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for path_str in selected:
        coverage, resolved = _resolve_coverage(path_str, coverage_index)
        results.append(
            {
                "file": path_str,
                "coverage": coverage,
                "resolved": resolved,
            }
        )
    return results


def _summarize_results(
    results: list[dict[str, object]],
    min_required: float,
    goal: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    missing = [r for r in results if r["coverage"] is None]
    below_min = [r for r in results if r["coverage"] is not None and r["coverage"] < min_required]
    below_goal = [r for r in results if r["coverage"] is not None and r["coverage"] < goal]
    return missing, below_min, below_goal


def _print_results(
    results: list[dict[str, object]],
    min_required: float,
    goal: float,
) -> None:
    for r in results:
        cov = r["coverage"]
        file_path = r["file"]
        if cov is None:
            print(f"- {file_path}: coverage missing")
            continue
        status = "OK"
        if cov < min_required:
            status = "FAIL"
        elif cov < goal:
            status = "WARN"
        print(f"- {file_path}: {cov:.1f}% [{status}]")


def _report_failures(
    missing: list[dict[str, object]],
    below_min: list[dict[str, object]],
) -> None:
    print("Coverage requirement not met:")
    if missing:
        _print_lines(["- Missing coverage data for:"] + [f"  - {r['file']}" for r in missing])
    if below_min:
        _print_lines(
            ["- Below minimum coverage:"]
            + [f"  - {r['file']} ({r['coverage']:.1f}%)" for r in below_min]
        )


def _report_warnings(below_goal: list[dict[str, object]]) -> None:
    _print_lines(
        ["Coverage goal not met (warning only):"]
        + [f"- {r['file']} ({r['coverage']:.1f}%)" for r in below_goal]
    )


def _ensure_coverage_path(coverage_path: Path) -> bool:
    if not coverage_path.exists():
        print(f"coverage.json not found at {coverage_path}", file=sys.stderr)
        return False
    return True


def _log_header(args: argparse.Namespace) -> None:
    print(
        "Changed files test coverage check "
        f"(base={args.base}, min={args.min:.1f}%, goal={args.goal:.1f}%)"
    )


def _report_skipped(skipped: list[str], show_skipped: bool) -> None:
    if show_skipped and skipped:
        _print_lines(["Skipped:"] + [f"- {p}" for p in skipped])


def _handle_missing_or_below_min(
    missing: list[dict[str, object]],
    below_min: list[dict[str, object]],
) -> int:
    if missing or below_min:
        _report_failures(missing, below_min)
        return 1
    return 0


def _run_check(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    coverage_path = _resolve_coverage_path(args.coverage, repo_root)
    if not _ensure_coverage_path(coverage_path):
        return 1

    include_patterns = args.include or ["robot_sf/*.py", "robot_sf/**/*.py"]
    exclude_patterns = args.exclude or []

    changed_files = _changed_files(args.base, repo_root)
    if not changed_files:
        print(f"No changed files vs {args.base}.")
        return 0

    selected, skipped = _select_changed_files(
        changed_files,
        repo_root,
        include_patterns,
        exclude_patterns,
    )
    if not selected:
        print("No changed files matched include patterns.")
        _report_skipped(skipped, args.show_skipped)
        return 0

    coverage_index = _load_coverage_index(coverage_path, repo_root)
    results = _build_results(selected, coverage_index)

    _log_header(args)
    _print_results(results, args.min, args.goal)
    _report_skipped(skipped, args.show_skipped)

    missing, below_min, below_goal = _summarize_results(results, args.min, args.goal)
    failure_code = _handle_missing_or_below_min(missing, below_min)
    if failure_code:
        return failure_code

    if below_goal:
        _report_warnings(below_goal)

    return 0


def main() -> int:
    """Run the changed-files coverage check and return exit code."""
    args = _parse_args()
    return _run_check(args)


if __name__ == "__main__":
    sys.exit(main())
