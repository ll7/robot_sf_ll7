from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _write_matrix(path: Path, repeats: int = 3) -> None:
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
    assert "collisions" in content
    assert content.endswith("\n")
