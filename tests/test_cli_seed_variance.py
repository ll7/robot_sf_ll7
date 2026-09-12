"""Tests for the seed-variance CLI subcommand computing cross-seed variability statistics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

from tests._cli_fixtures import write_scenario_matrix

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_cli_seed_variance(tmp_path: Path, capsys):
    """Verify seed-variance CLI calculates mean, standard deviation, and coefficient of variation."""
    matrix_path = tmp_path / "matrix.yaml"
    write_scenario_matrix(matrix_path, "sv-smoke", repeats=4)
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
    cap = capsys.readouterr()
    assert rc_run == 0, f"run failed: {cap.err}"

    out_json = tmp_path / "seed_var.json"
    rc = cli_main(
        [
            "seed-variance",
            "--in",
            str(episodes),
            "--out",
            str(out_json),
            "--group-by",
            "scenario_id",
        ],
    )
    cap2 = capsys.readouterr()
    assert rc == 0, f"seed-variance failed: {cap2.err}"

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "sv-smoke" in data
    # Pick any metric (e.g., success) and ensure cv key present
    any_metric = next(iter(data["sv-smoke"].keys()))
    stats = data["sv-smoke"][any_metric]
    assert all(k in stats for k in ("mean", "std", "cv", "count"))
