"""Export interval-inclusive paper Results handoff rows from a frozen benchmark bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_CONFIDENCE = 0.95
_DEFAULT_BOOTSTRAP_SAMPLES = 400
_DEFAULT_BOOTSTRAP_SEED = 123


def _build_parser() -> argparse.ArgumentParser:
    """Create the paper Results handoff CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Campaign root or publication bundle directory to export from.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory. Defaults to "
            "output/benchmarks/publication/<campaign_id>_paper_results_handoff."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=_DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap resamples over per-seed means.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=_DEFAULT_CONFIDENCE,
        help="Confidence level for interval bounds.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=_DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for bootstrap resampling.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the paper Results handoff exporter."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        from loguru import logger

        logger.disable("robot_sf")
        try:
            from robot_sf.benchmark.paper_results_handoff import export_paper_results_handoff

            result = export_paper_results_handoff(
                args.source,
                output_dir=args.out_dir,
                confidence_settings={
                    "method": "bootstrap_mean_over_seed_means",
                    "confidence": float(args.confidence),
                    "bootstrap_samples": int(args.bootstrap_samples),
                    "bootstrap_seed": int(args.bootstrap_seed),
                },
            )
        finally:
            logger.enable("robot_sf")
    except Exception as exc:
        print(f"paper Results handoff export failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "json_path": str(result.json_path),
                "csv_path": str(result.csv_path),
                "row_count": result.row_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
