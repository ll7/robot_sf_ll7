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
from pathlib import Path

from loguru import logger

from robot_sf.coverage_tools.baseline_comparator import (
    CoverageSnapshot,
    compare,
    generate_warning,
    load_baseline,
)


def example_load_coverage() -> CoverageSnapshot:
    """Load current coverage data from JSON file."""
    coverage_path = Path("coverage.json")

    if not coverage_path.exists():
        logger.error(f"Coverage file not found: {coverage_path}")
        logger.info("Run 'uv run pytest tests' first to generate coverage data")
        raise FileNotFoundError(coverage_path)

    with coverage_path.open(encoding="utf-8") as f:
        data = json.load(f)

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
    baseline_path = Path(".coverage-baseline.json")
    current_path = Path("coverage.json")

    # Check if baseline exists
    if not baseline_path.exists():
        logger.warning(f"No baseline found at {baseline_path}")
        logger.info("Creating baseline from current coverage...")
        if current_path.exists():
            import shutil

            shutil.copy(current_path, baseline_path)
            logger.info("Baseline created. Run 'uv run pytest tests' again to see comparison")
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
    baseline_path = Path(".coverage-baseline.json")
    current_path = Path("coverage.json")

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
