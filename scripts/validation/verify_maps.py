#!/usr/bin/env python3
"""Map verification CLI entry point.

Validates SVG maps in the repository for structural integrity, metadata
completeness, and runtime compatibility.

Usage Examples
--------------
Validate all maps locally:
    python scripts/validation/verify_maps.py --scope all --mode local

Validate CI-enabled maps only:
    python scripts/validation/verify_maps.py --scope ci --mode ci

Validate specific map:
    python scripts/validation/verify_maps.py --scope classic_doorway.svg --mode local

Output structured JSON manifest:
    python scripts/validation/verify_maps.py --scope all --output output/validation/map_verification.json

See Also
--------
- specs/001-map-verification/quickstart.md : Usage guide
- robot_sf.maps.verification : Implementation module
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from robot_sf.common.logging import configure_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify SVG maps for structural and metadata issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--scope",
        type=str,
        default="all",
        help="Scope of maps to verify: 'all', 'ci', 'changed', or specific filename (default: all)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "ci"],
        default="local",
        help="Verification mode: 'local' (informative) or 'ci' (strict, enforces timeouts)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write JSON verification manifest (default: none)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic environment instantiation (default: none)",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt automatic remediation for auto-fixable issues",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Configure logging
    configure_logging(verbose=args.verbose)

    logger.info("Map verification starting")
    logger.info(f"Scope: {args.scope}, Mode: {args.mode}")

    # Import and call verification runner
    from robot_sf.maps.verification.runner import verify_maps

    try:
        results = verify_maps(
            scope=args.scope,
            mode=args.mode,
            output_path=args.output,
            seed=args.seed,
            fix=args.fix,
        )

        # Determine exit code based on mode and results
        if args.mode == "ci":
            # CI mode: fail if any maps failed
            if results.failed > 0:
                logger.error(f"CI mode: {results.failed} map(s) failed validation")
                sys.exit(1)
            else:
                logger.info("CI mode: All maps passed validation")
                sys.exit(0)
        else:
            # Local mode: always exit 0 (informational)
            sys.exit(0)

    except Exception as e:
        logger.exception(f"Verification failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
