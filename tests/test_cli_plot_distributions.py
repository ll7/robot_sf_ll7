from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.cli import cli_main

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _write_matrix(path: Path, repeats: int = 3) -> None:
    scenarios = [
        {
            "id": "dist-smoke",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": repeats,
        }
    ]
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)


def test_cli_plot_distributions_png_and_pdf(tmp_path: Path, capsys):
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, repeats=4)
    episodes = tmp_path / "episodes.jsonl"

    rc_run = cli_main(
        [
            "run",
            "--matrix",
            str(matrix_path),
            "--out",
            str(episodes),
            "--schema",
            SCHEMA_PATH,
            "--base-seed",
            "0",
            "--horizon",
            "8",
            "--dt",
            "0.1",
        ]
    )
    capsys.readouterr()
    assert rc_run == 0

    out_dir = tmp_path / "plots"
    rc_plot = cli_main(
        [
            "plot-distributions",
            "--in",
            str(episodes),
            "--out-dir",
            str(out_dir),
            "--metrics",
            "collisions,comfort_exposure",
            "--bins",
            "10",
            "--kde",
            "--out-pdf",
        ]
    )
    cap = capsys.readouterr()
    assert rc_plot == 0, f"plot-distributions failed: {cap.err}"
    # Expect at least one PNG and one PDF for the two metrics
    pngs = list(out_dir.glob("dist_*.png"))
    pdfs = list(out_dir.glob("dist_*.pdf"))
    assert pngs, "No PNGs generated"
    assert pdfs, "No PDFs generated"
