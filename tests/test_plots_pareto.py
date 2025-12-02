"""Module test_plots_pareto auto-generated docstring."""

from __future__ import annotations

from robot_sf.benchmark.plots import (
    compute_pareto_points,
    pareto_front_indices,
    save_pareto_png,
)


def _rec(g: str, x: float, y: float) -> dict:
    """Rec.

    Args:
        g: Auto-generated placeholder description.
        x: Auto-generated placeholder description.
        y: Auto-generated placeholder description.

    Returns:
        dict: Auto-generated placeholder description.
    """
    return {
        "scenario_id": f"scn-{g}",
        "scenario_params": {"algo": g},
        "algo": g,
        "metrics": {"collisions": x, "comfort_exposure": y},
    }


def test_compute_points_and_front():
    """Test compute points and front.

    Returns:
        Any: Auto-generated placeholder description.
    """
    records = [
        _rec("A", 1.0, 0.5),
        _rec("A", 1.0, 0.6),
        _rec("B", 0.8, 0.9),
        _rec("C", 1.5, 0.4),
    ]
    pts, labels = compute_pareto_points(records, "collisions", "comfort_exposure")
    assert len(pts) == len(labels) == 3
    front = pareto_front_indices(pts, x_higher_better=False, y_higher_better=False)
    assert set(front).issubset(set(range(len(pts))))
    assert len(front) >= 1


def test_save_png_creates_file(tmp_path):
    """Test save png creates file.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    records = [
        _rec("A", 1.0, 0.5),
        _rec("B", 0.8, 0.9),
        _rec("C", 1.5, 0.4),
    ]
    out = tmp_path / "pareto.png"
    meta = save_pareto_png(records, str(out), "collisions", "comfort_exposure", title="Test")
    assert out.exists()
    assert out.stat().st_size > 0
    assert "front_size" in meta


def test_save_pdf_option(tmp_path):
    """Test save pdf option.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    records = [
        _rec("A", 1.0, 0.5),
        _rec("B", 0.8, 0.9),
        _rec("C", 1.5, 0.4),
    ]
    out_png = tmp_path / "pareto.png"
    out_pdf = tmp_path / "pareto.pdf"
    meta = save_pareto_png(
        records,
        str(out_png),
        "collisions",
        "comfort_exposure",
        title="PDF Test",
        out_pdf=str(out_pdf),
    )
    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert meta.get("pdf") == str(out_pdf)
