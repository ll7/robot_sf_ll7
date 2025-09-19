from pathlib import Path


def test_force_field_figure_runs(tmp_path: Path):
    from results.figures.fig_force_field import generate_force_field_figure

    png = tmp_path / "ff.png"
    pdf = tmp_path / "ff.pdf"

    generate_force_field_figure(out_png=str(png), out_pdf=str(pdf))

    assert png.exists()
    assert pdf.exists()
