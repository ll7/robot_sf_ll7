from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.cli import cli_main

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _write_matrix(path: Path, repeats: int = 2) -> None:
    scenarios = [
        {
            "id": "table-tex",
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


def test_cli_table_tex(tmp_path: Path, capsys):
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, repeats=2)
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
            "6",
            "--dt",
            "0.1",
        ]
    )
    capsys.readouterr()
    assert rc_run == 0

    out_tex = tmp_path / "table.tex"
    rc = cli_main(
        [
            "table",
            "--in",
            str(episodes),
            "--out",
            str(out_tex),
            "--metrics",
            "collisions,comfort_exposure",
            "--format",
            "tex",
        ]
    )
    cap = capsys.readouterr()
    assert rc == 0, f"table tex failed: {cap.err}"
    content = out_tex.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in content
    assert "\\toprule" in content
    assert "\\midrule" in content
    assert "\\bottomrule" in content
    assert content.endswith("\n")
