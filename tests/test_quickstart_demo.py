"""Smoke tests for the one-command ``robot-sf demo`` visual demo (issue #5792)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.cli import main as cli_main
from scripts.demo.quickstart_demo import run_demo

if TYPE_CHECKING:
    from pathlib import Path


def _read_jsonl_lines(path: Path) -> list[dict[str, object]]:
    """Return the parsed JSONL records from ``path``."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_demo_produces_expected_artifacts(tmp_path: Path) -> None:
    """Running the demo writes episode.jsonl, summary.json, metrics.json, the viewer, and a thumbnail."""
    output_root = tmp_path / "demo" / "latest"
    result = run_demo(output_root=output_root, seed=270)

    assert result.episode_jsonl.exists()
    assert result.summary_json.exists()
    assert result.metrics_json.exists()
    assert result.viewer_html.exists()
    assert result.thumbnail_png.exists()
    assert result.viewer_html.parent.joinpath("scene.json").exists()

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["scenario"] == "quickstart_demo_crossing_basic"
    assert summary["planner"] == "random"
    assert summary["steps"] > 0

    metrics = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    assert metrics["seed"] == 270
    assert metrics["steps"] == summary["steps"]


def test_demo_episode_is_deterministic_and_records_steps(tmp_path: Path) -> None:
    """The same seed reproduces an identical recorded episode (stable, reviewable demo output)."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    run_a = run_demo(output_root=root_a, seed=270)
    run_b = run_demo(output_root=root_b, seed=270)

    lines_a = _read_jsonl_lines(run_a.episode_jsonl)
    lines_b = _read_jsonl_lines(run_b.episode_jsonl)

    assert len(lines_a) == len(lines_b) > 0
    # The recorded robot trajectory is reproduced across runs.
    traj_a = [line.get("state", {}).get("robot_pose") for line in lines_a]
    traj_b = [line.get("state", {}).get("robot_pose") for line in lines_b]
    assert traj_a == traj_b


def test_demo_cli_dispatches(tmp_path: Path, capsys) -> None:
    """The umbrella ``robot-sf demo`` subcommand runs and prints the plain-English summary."""
    rc = cli_main(["demo", "--output-root", str(tmp_path / "cli"), "--seed", "270"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Robot SF demo complete." in out
    assert (tmp_path / "cli" / "episode.jsonl").exists()
