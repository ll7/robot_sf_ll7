"""Automated results collection and analysis for imitation learning runs.

Loads baseline and pretrained training manifests, computes sample-efficiency
statistics, generates comparison figures, and writes a summary compliant with
``training_summary.schema.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from loguru import logger

from robot_sf.training import analyze_imitation_results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        required=True,
        help="Identifier for this comparison group (becomes the summary run_id).",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline training run ID.",
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="Pre-trained training run ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory (defaults under imitation report root).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = analyze_imitation_results(
        group_id=args.group,
        baseline_run_id=args.baseline,
        pretrained_run_id=args.pretrained,
        output_dir=args.output,
    )
    logger.success("Analysis complete", summary=str(artifacts["summary_json"]))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
