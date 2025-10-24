from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _write_matrix(path: Path, repeats: int = 4) -> None:
    scenarios = [
        {
            "id": "dist-ci",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": repeats,
        },
    ]
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)


essential_markers = [
    # PNGs only; we just check existence of image files after CLI completes
]


def test_cli_plot_distributions_ci(tmp_path: Path, capsys):
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
        ],
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
            "--ci",
            "--ci-samples",
            "200",
            "--ci-confidence",
            "0.90",
            "--ci-seed",
            "123",
            "--out-pdf",
        ],
    )
    cap = capsys.readouterr()
    assert rc_plot == 0, f"plot-distributions --ci failed: {cap.err}"
    pngs = list(out_dir.glob("dist_*.png"))
    pdfs = list(out_dir.glob("dist_*.pdf"))
    assert pngs, "No PNGs generated with CI overlay"
    assert pdfs, "No PDFs generated with CI overlay"
