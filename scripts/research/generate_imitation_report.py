"""Generate imitation learning report from training summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from robot_sf.research.imitation_report import ImitationReportConfig, generate_imitation_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for imitation report generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.json emitted by imitation analysis.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/imitation_reports"),
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
        default="BC pre-training reduces timesteps by ≥30%",
        help="Hypothesis statement to record in the report.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for tests.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Improvement percentage threshold for hypothesis evaluation.",
    )
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Also emit a LaTeX version of the report.",
    )
    parser.add_argument(
        "--baseline-run-id",
        type=str,
        default=None,
        help="Name of the baseline run to prioritize in the report.",
    )
    parser.add_argument(
        "--pretrained-run-id",
        type=str,
        default=None,
        help="Name of the pretrained run to include (required if summary has >2 records).",
    )
    parser.add_argument(
        "--ablation-label",
        type=str,
        default=None,
        help="Optional ablation label (e.g., dataset size or BC epochs).",
    )
    parser.add_argument(
        "--hparam",
        action="append",
        default=None,
        help="Hyperparameter key=value pairs to record in the report (repeatable).",
    )
    return parser.parse_args(argv)


def _parse_hparams(pairs: list[str]) -> dict[str, str]:
    """Parse key=value CLI hyperparameters into a dict."""
    result: dict[str, str] = {}
    for pair in pairs:
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def main(argv: list[str] | None = None) -> int:
    """Entrypoint: generate imitation report artifacts from a summary JSON."""
    args = parse_args(argv)
    cfg = ImitationReportConfig(
        experiment_name=args.experiment_name,
        hypothesis=args.hypothesis,
        alpha=args.alpha,
        improvement_threshold_pct=args.threshold,
        export_latex=args.export_latex,
        baseline_run_id=args.baseline_run_id,
        pretrained_run_id=args.pretrained_run_id,
        ablation_label=args.ablation_label,
        hyperparameters=_parse_hparams(args.hparam or []),
    )

    paths = generate_imitation_report(
        summary_path=args.summary,
        output_root=args.output_root,
        config=cfg,
    )
    logger.success("Imitation report generated", report=str(paths["report"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
