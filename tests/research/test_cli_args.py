"""Tests for shared CLI argument helpers in robot_sf/research/cli_args.py."""

from __future__ import annotations

import argparse

from robot_sf.research.cli_args import add_imitation_report_common_args


def _make_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def test_defaults_empty_argv():
    parser = _make_parser()
    add_imitation_report_common_args(parser)
    args = parser.parse_args([])
    assert args.experiment_name == "imitation"
    assert args.num_seeds is None
    assert args.hypothesis == "BC pre-training reduces timesteps by \u226530%"
    assert args.alpha == 0.05
    assert args.export_latex is False
    assert not hasattr(args, "threshold")


def test_custom_alpha_flag_and_dest():
    parser = _make_parser()
    add_imitation_report_common_args(parser, alpha_flag="--p-val", alpha_dest="p_value")
    args = parser.parse_args(["--p-val", "0.01"])
    assert args.p_value == 0.01
    assert not hasattr(args, "alpha")


def test_threshold_present_when_included():
    parser = _make_parser()
    add_imitation_report_common_args(parser, include_threshold=True)
    help_text = parser.format_help()
    assert "--threshold" in help_text
    args = parser.parse_args(["--threshold", "15.0"])
    assert args.threshold == 15.0


def test_threshold_absent_when_excluded():
    parser = _make_parser()
    add_imitation_report_common_args(parser, include_threshold=False)
    help_text = parser.format_help()
    assert "--threshold" not in help_text


def test_export_latex_present_when_included():
    parser = _make_parser()
    add_imitation_report_common_args(parser, include_export_latex=True)
    help_text = parser.format_help()
    assert "--export-latex" in help_text


def test_export_latex_absent_when_excluded():
    parser = _make_parser()
    add_imitation_report_common_args(parser, include_export_latex=False)
    help_text = parser.format_help()
    assert "--export-latex" not in help_text
