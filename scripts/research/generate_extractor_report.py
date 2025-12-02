"""Generate research-ready report from multi-extractor summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from robot_sf.research.extractor_report import ReportConfig, generate_extractor_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.json emitted by multi_extractor_training.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/research_reports"),
        help="Base directory for generated reports.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="imitation",
        help="Name used for the report folder prefix.",
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        default=None,
        help="Optional hypothesis statement to record in the report.",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.05,
        help="Alpha level for statistical tests.",
    )
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Also emit a LaTeX version of the report.",
    )
    parser.add_argument(
        "--baseline-extractor",
        type=str,
        default=None,
        help="Name of the baseline extractor to compare against (defaults to first).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config file to copy into the report for reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = ReportConfig(
        experiment_name=args.experiment_name,
        hypothesis=args.hypothesis,
        significance_level=float(args.significance_level),
        export_latex=bool(args.export_latex),
        baseline_extractor=args.baseline_extractor,
    )

    paths = generate_extractor_report(
        summary_path=args.summary,
        output_root=args.output_root,
        config=cfg,
        config_path=args.config,
    )
    logger.success(
        "Report generated", report=str(paths["report"]), output_root=str(args.output_root)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
