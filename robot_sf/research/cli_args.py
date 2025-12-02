"""Shared CLI argument helpers for research reporting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def add_imitation_report_common_args(
    parser: argparse.ArgumentParser,
    *,
    alpha_flag: str = "--alpha",
    alpha_dest: str = "alpha",
    include_threshold: bool = False,
    include_export_latex: bool = True,
) -> None:
    """Add common imitation report generation arguments to an argparse parser."""

    parser.add_argument(
        "--experiment-name",
        default="imitation",
        help="Experiment name used for report folder naming.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of random seeds represented in the comparison (metadata only).",
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="BC pre-training reduces timesteps by ≥30%",
        help="Hypothesis statement for report generation.",
    )
    parser.add_argument(
        alpha_flag,
        type=float,
        dest=alpha_dest,
        default=0.05,
        help="Significance threshold for statistical tests.",
    )
    if include_threshold:
        parser.add_argument(
            "--threshold",
            type=float,
            default=30.0,
            help="Improvement percentage threshold for hypothesis evaluation.",
        )
    if include_export_latex:
        parser.add_argument(
            "--export-latex",
            action="store_true",
            help="Emit LaTeX/PDF alongside Markdown.",
        )
