from pathlib import Path

from robot_sf.benchmark.figures.force_field import generate_force_field_figure


def test_force_field_figure_runs(tmp_path: Path):
    png = tmp_path / "ff.png"
    pdf = tmp_path / "ff.pdf"

    generate_force_field_figure(out_png=str(png), out_pdf=str(pdf))

    assert png.exists()
    assert pdf.exists()
