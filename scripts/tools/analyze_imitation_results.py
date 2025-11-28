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

from robot_sf.research.cli_args import add_imitation_report_common_args
from robot_sf.training import analyze_imitation_results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for imitation results analysis."""
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
    add_imitation_report_common_args(
        parser,
        alpha_flag="--significance-level",
        alpha_dest="significance_level",
        include_threshold=False,
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate Markdown/LaTeX report after analysis.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: run analysis and emit summary artifacts."""
    args = parse_args(argv)
    artifacts = analyze_imitation_results(
        group_id=args.group,
        baseline_run_id=args.baseline,
        pretrained_run_id=args.pretrained,
        output_dir=args.output,
    )
    logger.success("Analysis complete", summary=str(artifacts["summary_json"]))

    if args.generate_report:
        from robot_sf.research.imitation_report import (
            ImitationReportConfig,
            generate_imitation_report,
        )

        cfg = ImitationReportConfig(
            experiment_name=args.experiment_name,
            hypothesis=args.hypothesis,
            alpha=args.significance_level,
            export_latex=args.export_latex,
            baseline_run_id=args.baseline,
            pretrained_run_id=args.pretrained,
            num_seeds=args.num_seeds,
        )
        report_paths = generate_imitation_report(
            summary_path=artifacts["summary_json"],
            output_root=Path("output/research_reports"),
            config=cfg,
        )
        logger.success("Report generated", report=str(report_paths["report"]))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
