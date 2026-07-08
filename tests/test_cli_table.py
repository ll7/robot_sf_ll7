"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _write_matrix(path: Path, repeats: int = 3) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
        repeats: TODO docstring.
    """
    scenarios = [
        {
            "id": "table-smoke",
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


def test_cli_table_md(tmp_path: Path, capsys):
    """Table CLI writes a Markdown report without track columns for legacy records.

    Args:
        tmp_path: Temporary directory for generated inputs and outputs.
        capsys: Pytest capture fixture for CLI output.
    """
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, repeats=3)
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

    out_md = tmp_path / "table.md"
    rc = cli_main(
        [
            "table",
            "--in",
            str(episodes),
            "--out",
            str(out_md),
            "--metrics",
            "collisions,comfort_exposure",
            "--format",
            "md",
        ],
    )
    cap = capsys.readouterr()
    assert rc == 0, f"table failed: {cap.err}"
    content = out_md.read_text(encoding="utf-8")
    assert "| Group |" in content
    assert "Benchmark Track" not in content
    assert "Collision rate" in content
    assert content.endswith("\n")


def test_cli_table_diagnostic_cross_track_labels_benchmark_track(
    tmp_path: Path,
    capsys,
) -> None:
    """Explicit cross-track tables should show observation contracts as caveated groups."""
    episodes = tmp_path / "mixed_tracks.jsonl"
    rows = [
        {
            "episode_id": "grid-1",
            "scenario_id": "mixed-track",
            "seed": 1,
            "benchmark_track": "grid_socnav_v1",
            "scenario_params": {"algo": "planner-a", "benchmark_track": "grid_socnav_v1"},
            "metrics": {"score": 1.0},
        },
        {
            "episode_id": "lidar-1",
            "scenario_id": "mixed-track",
            "seed": 1,
            "benchmark_track": "lidar_2d_v1",
            "scenario_params": {"algo": "planner-a", "benchmark_track": "lidar_2d_v1"},
            "metrics": {"score": 3.0},
        },
    ]
    episodes.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    strict_out = tmp_path / "strict.md"
    rc_strict = cli_main(
        [
            "table",
            "--in",
            str(episodes),
            "--out",
            str(strict_out),
            "--metrics",
            "score",
        ],
    )
    capsys.readouterr()
    assert rc_strict == 2
    assert not strict_out.exists()

    diagnostic_out = tmp_path / "diagnostic.md"
    rc = cli_main(
        [
            "table",
            "--in",
            str(episodes),
            "--out",
            str(diagnostic_out),
            "--metrics",
            "score",
            "--observation-track-mode",
            "diagnostic-cross-track",
        ],
    )
    cap = capsys.readouterr()
    assert rc == 0, f"table failed: {cap.err}"

    content = diagnostic_out.read_text(encoding="utf-8")
    assert "| Benchmark Track | Group | Score |" in content
    assert "| grid_socnav_v1 | grid_socnav_v1 :: planner-a | 1.0000 |" in content
    assert "| lidar_2d_v1 | lidar_2d_v1 :: planner-a | 3.0000 |" in content
