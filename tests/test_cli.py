from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.cli import cli_main

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def test_cli_baseline_subcommand(tmp_path: Path, capsys):
    # Build a minimal scenario matrix YAML
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {
            "id": "cli-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        }
    ]
    # Write YAML list
    import yaml  # type: ignore

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    out_json = tmp_path / "baseline.json"
    out_jsonl = tmp_path / "episodes.jsonl"

    # Run CLI main with arguments
    rc = cli_main(
        [
            "baseline",
            "--matrix",
            str(matrix_path),
            "--out",
            str(out_json),
            "--jsonl",
            str(out_jsonl),
            "--schema",
            SCHEMA_PATH,
            "--base-seed",
            "0",
            "--repeats",
            "1",
            "--horizon",
            "8",
            "--dt",
            "0.1",
        ]
    )
    # Capture output (for sanity, not strictly required)
    captured = capsys.readouterr()
    assert rc == 0, f"CLI returned non-zero: {captured.err}"

    # Check outputs
    assert out_json.exists()
    assert out_jsonl.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    # Should contain at least some baseline keys
    assert "time_to_goal_norm" in data
    assert "collisions" in data


def test_cli_list_scenarios(tmp_path: Path, capsys):
    # Minimal scenario matrix YAML
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {
            "id": "s1",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "repeats": 1,
        },
        {
            "id": "s2",
            "density": "med",
            "flow": "bi",
            "obstacle": "open",
            "repeats": 2,
        },
    ]
    import yaml  # type: ignore

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    rc = cli_main(["list-scenarios", "--matrix", str(matrix_path)])
    captured = capsys.readouterr()
    assert rc == 0
    # Check output includes both IDs
    assert "s1" in captured.out
    assert "s2" in captured.out
