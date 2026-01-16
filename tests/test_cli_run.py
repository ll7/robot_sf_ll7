"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_cli_run_subcommand(tmp_path: Path, capsys):
    # Minimal scenario matrix YAML
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        capsys: TODO docstring.
    """
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {
            "id": "cli-run-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 2,
        },
    ]
    import yaml  # type: ignore

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    out_jsonl = tmp_path / "episodes.jsonl"

    rc = cli_main(
        [
            "run",
            "--matrix",
            str(matrix_path),
            "--out",
            str(out_jsonl),
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
    assert rc == 0, f"CLI run failed: {cap.err}"

    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    # Expect 2 episodes due to repeats
    assert len(lines) == 2

    # Spot-check JSON structure including algorithm metadata
    rec = json.loads(lines[0])
    assert "episode_id" in rec and "metrics" in rec
    assert "algorithm_metadata" in rec and isinstance(rec["algorithm_metadata"], dict)
