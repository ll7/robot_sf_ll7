"""Tests for the table CLI subcommand generating LaTeX tabular exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

from tests._cli_fixtures import write_scenario_matrix

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_cli_table_tex(tmp_path: Path, capsys):
    """Verify table CLI with format tex outputs valid LaTeX tabular markup."""
    matrix_path = tmp_path / "matrix.yaml"
    write_scenario_matrix(matrix_path, "table-tex", repeats=2)
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
        ],
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
        ],
    )
    cap = capsys.readouterr()
    assert rc == 0, f"table tex failed: {cap.err}"
    content = out_tex.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in content
    assert "\\toprule" in content
    assert "\\midrule" in content
    assert "\\bottomrule" in content
    assert content.endswith("\n")
