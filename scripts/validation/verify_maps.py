#!/usr/bin/env python3
"""Map verification CLI entry point.

This script validates all SVG maps in the repository for structural integrity,
metadata completeness, and runtime compatibility. It can be run locally for
development or in CI for automated quality gates.

Usage:
    # Verify all maps
    uv run python scripts/validation/verify_maps.py --scope all --mode local
    
    # CI mode with structured output
    uv run python scripts/validation/verify_maps.py --scope ci --mode ci --output output/validation/map_verification.json
    
    # Verify changed files only
    uv run python scripts/validation/verify_maps.py --scope changed --mode local
    
    # Fix auto-remediable issues
    uv run python scripts/validation/verify_maps.py --scope all --mode local --fix
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from robot_sf.maps.verification.context import VerificationContext
from robot_sf.maps.verification.scope_resolver import resolve_scope
from robot_sf.maps.verification.runner import verify_maps


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify SVG maps for structural and runtime integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--scope",
        choices=["all", "ci", "changed"],
        default="all",
        help="Which maps to verify (default: all)",
    )
    
    parser.add_argument(
        "--mode",
        choices=["local", "ci"],
        default="local",
        help="Execution mode: local for development, ci for automated checks (default: local)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for JSON manifest output (default: output/validation/map_verification.json)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic environment instantiation",
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix remediable issues",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for map verification.
    
    Returns:
        Exit code: 0 for success, 1 for validation failures
    """
    args = parse_args()
    
    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
    )
    
    logger.info("Map verification starting")
    logger.info(f"Scope: {args.scope}, Mode: {args.mode}")
    
    # Create verification context
    output_path = args.output or Path("output/validation/map_verification.json")
    context = VerificationContext(
        mode=args.mode,
        output_path=output_path,
        fix_mode=args.fix,
        seed=args.seed,
    )
    
    logger.info(f"Run ID: {context.run_id}, Git SHA: {context.git_sha[:8]}")
    
    # Resolve scope to get maps to verify
    try:
        maps = resolve_scope(args.scope)
    except Exception as e:
        logger.error(f"Failed to resolve scope: {e}")
        return 1
    
    if not maps:
        logger.warning("No maps found to verify")
        return 0
    
    # Run verification
    try:
        results = verify_maps(maps, context)
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return 1
    
    # Summarize results
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    warned = sum(1 for r in results if r.status == "warn")
    
    logger.info("=" * 60)
    logger.info(f"VERIFICATION SUMMARY")
    logger.info(f"  Total maps: {len(results)}")
    logger.info(f"  Passed: {passed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Warned: {warned}")
    logger.info("=" * 60)
    
    # In CI mode, fail if any maps failed
    if context.is_ci_mode and failed > 0:
        logger.error(f"CI mode: {failed} maps failed verification")
        return 1
    
    # TODO: Write JSON manifest (Phase 5)
    logger.info(f"Manifest output will be written to: {output_path}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
