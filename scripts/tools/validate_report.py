#!/usr/bin/env python3
"""
Validation script for research reports.
Checks schema compliance, file completeness, and artifact integrity.
Usage: scripts/tools/validate_report.py --report-dir <report_dir>
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from robot_sf.research.exceptions import ValidationError
from robot_sf.research.schema_loader import load_schema, validate_data


def validate_report_structure(report_dir: Path) -> bool:
    """Check that all required files and directories exist."""
    required_files = [
        "report.md",
        "metadata.json",
        "data/metrics.json",
        "data/metrics.csv",
        "data/hypothesis.json",
    ]

    required_dirs = [
        "figures",
        "data",
        "configs",
    ]

    all_valid = True

    for file_path in required_files:
        full_path = report_dir / file_path
        if not full_path.exists():
            logger.error(f"Missing required file: {file_path}")
            all_valid = False
        else:
            logger.debug(f"Found: {file_path}")

    for dir_path in required_dirs:
        full_path = report_dir / dir_path
        if not full_path.is_dir():
            logger.error(f"Missing required directory: {dir_path}")
            all_valid = False
        else:
            logger.debug(f"Found directory: {dir_path}")

    return all_valid


def validate_metadata_schema(report_dir: Path) -> bool:
    """Validate metadata.json against schema."""
    metadata_path = report_dir / "metadata.json"

    if not metadata_path.exists():
        logger.error("metadata.json not found")
        return False

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    try:
        schema = load_schema("report_metadata.schema.v1.json")
        validate_data(metadata, schema)
    except ValidationError as exc:
        logger.error(f"Schema validation failed: {exc}")
        return False

    logger.info("metadata.json validates against schema")
    return True


def validate_figures(report_dir: Path) -> bool:
    """Check that figures exist in both PDF and PNG formats."""
    figures_dir = report_dir / "figures"

    if not figures_dir.is_dir():
        logger.error("figures/ directory not found")
        return False

    pdf_files = list(figures_dir.glob("*.pdf"))
    png_files = list(figures_dir.glob("*.png"))

    if not pdf_files:
        logger.warning("No PDF figures found")
        return False

    if not png_files:
        logger.warning("No PNG figures found")
        return False

    logger.info(f"Found {len(pdf_files)} PDF and {len(png_files)} PNG figures")

    # Check for matching pairs
    pdf_stems = {f.stem for f in pdf_files}
    png_stems = {f.stem for f in png_files}

    missing_png = pdf_stems - png_stems
    missing_pdf = png_stems - pdf_stems

    if missing_png:
        logger.warning(f"PDF figures missing PNG counterparts: {missing_png}")

    if missing_pdf:
        logger.warning(f"PNG figures missing PDF counterparts: {missing_pdf}")

    return True


def validate_report(report_dir: Path) -> bool:
    """Run all validation checks on a report."""
    logger.info(f"Validating report at: {report_dir}")

    all_checks = [
        ("Structure", validate_report_structure(report_dir)),
        ("Metadata Schema", validate_metadata_schema(report_dir)),
        ("Figures", validate_figures(report_dir)),
    ]

    logger.info("\n=== Validation Summary ===")
    for check_name, passed in all_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{check_name}: {status}")

    all_passed = all(passed for _, passed in all_checks)

    if all_passed:
        logger.info("\n✓ Report validation PASSED")
    else:
        logger.error("\n✗ Report validation FAILED")

    return all_passed


def main() -> None:
    """CLI entry point for report validation."""
    parser = argparse.ArgumentParser(
        description="Validate research report structure and schema compliance."
    )
    parser.add_argument("--report-dir", type=str, required=True, help="Path to report directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    report_dir = Path(args.report_dir)

    if not report_dir.is_dir():
        logger.error(f"Report directory not found: {report_dir}")
        sys.exit(1)

    success = validate_report(report_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
