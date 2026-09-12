"""Tests for the ranking CLI subcommand outputting formatted algorithm rankings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

from tests._cli_fixtures import write_scenario_matrix

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_cli_rank_md(tmp_path: Path, capsys):
    """Verify rank CLI generates a valid Markdown table ranking algorithms by metric."""
    matrix_path = tmp_path / "matrix.yaml"
    write_scenario_matrix(matrix_path, "rank-smoke", repeats=3)
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

    out_md = tmp_path / "ranking.md"
    rc = cli_main(
        [
            "rank",
            "--in",
            str(episodes),
            "--out",
            str(out_md),
            "--metric",
            "collisions",
            "--format",
            "md",
        ],
    )
    cap = capsys.readouterr()
    assert rc == 0, f"rank failed: {cap.err}"
    content = out_md.read_text(encoding="utf-8")
    assert "| Rank |" in content
    assert content.endswith("\n")
