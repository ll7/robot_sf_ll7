#!/usr/bin/env python3
"""Coverage tools programmatic usage example.

Purpose: Demonstrates how to use robot_sf.coverage_tools modules programmatically
for custom coverage analysis workflows.

Usage:
    # Generate coverage data first
    uv run pytest tests

    # Run this example
    uv run python examples/plotting/coverage_example.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.coverage_tools.baseline_comparator import (
    CoverageSnapshot,
    compare,
    generate_warning,
    load_baseline,
)


def _sample_coverage_payload() -> dict[str, Any]:
    """Return a deterministic sample coverage payload for demonstrations."""

    timestamp = datetime.now(tz=UTC).isoformat()
    return {
        "meta": {
            "version": "6.5.0",
            "timestamp": timestamp,
            "branch_coverage": False,
        },
        "totals": {
            "covered_lines": 92,
            "num_statements": 108,
            "percent_covered": 85.19,
        },
        "files": {
            "robot_sf/examples/demo_module.py": {
                "executed_lines": [1, 2, 3, 4, 5, 6, 7],
                "missing_lines": [8, 9],
                "summary": {
                    "covered_lines": 7,
                    "num_statements": 9,
                    "percent_covered": 77.78,
                },
            },
            "robot_sf/examples/utilities.py": {
                "executed_lines": [1, 2, 3, 4, 5],
                "missing_lines": [6],
                "summary": {
                    "covered_lines": 5,
                    "num_statements": 6,
                    "percent_covered": 83.33,
                },
            },
            "robot_sf/examples/cli.py": {
                "executed_lines": list(range(1, 21)),
                "missing_lines": [],
                "summary": {
                    "covered_lines": 20,
                    "num_statements": 20,
                    "percent_covered": 100.0,
                },
            },
        },
    }


def _load_or_create_coverage(path: Path) -> dict[str, Any]:
    """Load coverage data, falling back to a sample payload when needed."""

    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning(
                "Invalid coverage JSON at %s (%s). Falling back to sample payload.",
                path,
                exc,
            )

    logger.info("Generating sample coverage dataset for demonstration purposes.")
    sample = _sample_coverage_payload()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
        logger.debug("Wrote sample coverage dataset to %s", path)
    except OSError as exc:
        logger.debug("Unable to persist sample coverage dataset: %s", exc)
    return sample


def example_load_coverage() -> CoverageSnapshot:
    """Load current coverage data from JSON file."""
    coverage_path = resolve_artifact_path(Path("coverage.json"))
    data = _load_or_create_coverage(coverage_path)

    snapshot = CoverageSnapshot.from_coverage_json(data)
    logger.info(f"Loaded coverage: {snapshot.total_coverage:.2f}%")
    logger.info(f"Files analyzed: {len(snapshot.file_coverage)}")

    return snapshot


def example_snapshot_inspection(snapshot: CoverageSnapshot) -> None:
    """Demonstrate inspecting coverage snapshot data."""
    logger.info("=== Coverage Snapshot Inspection ===")
    logger.info(f"Overall coverage: {snapshot.total_coverage:.2f}%")
    logger.info(f"Timestamp: {snapshot.timestamp}")
    logger.info(f"Total files: {len(snapshot.file_coverage)}")

    # Show top 5 files by coverage
    sorted_files = sorted(snapshot.file_coverage.items(), key=lambda x: x[1], reverse=True)
    logger.info("\nTop 5 files by coverage:")
    for path, cov in sorted_files[:5]:
        logger.info(f"  {path}: {cov:.2f}%")

    # Show files with lowest coverage
    logger.info("\nBottom 5 files by coverage:")
    for path, cov in sorted_files[-5:]:
        logger.info(f"  {path}: {cov:.2f}%")


def example_baseline_comparison() -> None:
    """Demonstrate baseline comparison workflow."""
    baseline_path = resolve_artifact_path(Path("coverage/.coverage-baseline.json"))
    current_path = resolve_artifact_path(Path("coverage.json"))
    current_data = _load_or_create_coverage(current_path)

    # Check if baseline exists
    if not baseline_path.exists():
        logger.warning(f"No baseline found at {baseline_path}")
        logger.info("Creating baseline from current coverage...")
        try:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(current_data, indent=2) + "\n", encoding="utf-8")
            logger.info("Baseline created. Run 'uv run pytest tests' again to see comparison")
        except OSError as exc:
            logger.error(f"Failed to create baseline: {exc}")
        return

    # Load baseline
    baseline = load_baseline(baseline_path)
    if baseline is None:
        logger.error("Failed to load baseline")
        return

    logger.info(f"Baseline: {baseline.snapshot.total_coverage:.2f}%")

    # Compare current vs baseline
    try:
        delta = compare(current_path=current_path, baseline=baseline, threshold=1.0)

        if delta.has_decrease:
            logger.warning("Coverage decreased!")
            warning = generate_warning(delta, format_type="terminal")
            print(warning)
        elif delta.has_increase:
            logger.success("Coverage improved!")
            warning = generate_warning(delta, format_type="terminal")
            print(warning)
        else:
            logger.info("Coverage unchanged (within threshold)")

        # Show detailed stats
        logger.info(f"Overall change: {delta.delta:+.2f}%")
        logger.info(f"Files changed: {len(delta.changed_files)}")
        logger.info(f"Warnings: {len(delta.warnings)}")

    except FileNotFoundError as e:
        logger.error(f"Comparison failed: {e}")


def example_generate_warnings() -> None:
    """Demonstrate different warning formats."""
    baseline_path = resolve_artifact_path(Path("coverage/.coverage-baseline.json"))
    current_path = resolve_artifact_path(Path("coverage.json"))

    if not baseline_path.exists() or not current_path.exists():
        logger.warning("Baseline or current coverage missing, skipping warning demo")
        return

    baseline = load_baseline(baseline_path)
    if baseline is None:
        logger.error("Failed to load baseline")
        return

    delta = compare(current_path=current_path, baseline=baseline, threshold=0.1)

    if not delta.has_decrease and not delta.has_increase:
        logger.info("No changes to demonstrate warnings")
        return

    logger.info("=== Warning Formats ===")

    # Terminal format
    logger.info("\n--- Terminal Format ---")
    print(generate_warning(delta, format_type="terminal"))

    # GitHub Actions format
    logger.info("\n--- GitHub Actions Format ---")
    print(generate_warning(delta, format_type="github"))

    # JSON format
    logger.info("\n--- JSON Format ---")
    print(generate_warning(delta, format_type="json"))


def main():
    """Run all examples."""
    logger.info("=== Coverage Tools Examples ===")

    try:
        # Example 1: Load coverage data
        logger.info("\n--- Example 1: Load Coverage ---")
        snapshot = example_load_coverage()

        # Example 2: Inspect snapshot
        logger.info("\n--- Example 2: Inspect Snapshot ---")
        example_snapshot_inspection(snapshot)

        # Example 3: Baseline comparison
        logger.info("\n--- Example 3: Baseline Comparison ---")
        example_baseline_comparison()

        # Example 4: Warning formats
        logger.info("\n--- Example 4: Warning Formats ---")
        example_generate_warnings()

        logger.success("All examples completed!")

    except FileNotFoundError as e:
        logger.error(f"Missing file: {e}")
        logger.info("Make sure to run 'uv run pytest tests' first")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
