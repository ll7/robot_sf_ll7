"""CLI smoke tests for benchmark episode generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_cli_run_subcommand(tmp_path: Path, capsys):
    """Run a minimal benchmark matrix through the CLI and verify row metadata."""
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
            "--observation-mode",
            "socnav_state",
            "--observation-level",
            "tracked_agents_no_noise",
            "--benchmark-track",
            "grid_socnav_v1",
            "--track-schema-version",
            "observation-track.v1",
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
    assert rec["observation_mode"] == "socnav_state"
    assert rec["observation_level"] == "tracked_agents_no_noise"
    assert rec["benchmark_track"] == "grid_socnav_v1"
    assert rec["track_schema_version"] == "observation-track.v1"
    assert rec["scenario_params"]["observation_mode"] == "socnav_state"
    assert rec["scenario_params"]["observation_level"] == "tracked_agents_no_noise"
    assert rec["scenario_params"]["benchmark_track"] == "grid_socnav_v1"
    assert rec["algorithm_metadata"]["benchmark_track"] == {
        "benchmark_track": "grid_socnav_v1",
        "track_schema_version": "observation-track.v1",
        "observation_level": "tracked_agents_no_noise",
        "observation_mode": "socnav_state",
    }
