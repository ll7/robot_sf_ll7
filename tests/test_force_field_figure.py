"""Tests for the force field figure generation benchmark utility."""

from pathlib import Path

from robot_sf.benchmark.figures.force_field import generate_force_field_figure


def test_force_field_figure_runs(tmp_path: Path):
    """Verify that generate_force_field_figure creates PNG and PDF output artifacts.

    Args:
        tmp_path: Pytest temporary directory fixture for isolated figure output files.
    """
    png = tmp_path / "ff.png"
    pdf = tmp_path / "ff.pdf"

    generate_force_field_figure(out_png=str(png), out_pdf=str(pdf))

    assert png.exists()
    assert pdf.exists()
