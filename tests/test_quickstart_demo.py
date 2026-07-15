"""Smoke tests for the one-command ``robot-sf demo`` visual demo (issue #5792)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.cli import main as cli_main
from scripts.demo import quickstart_demo as demo

if TYPE_CHECKING:
    from pathlib import Path


def _read_jsonl_lines(path: Path) -> list[dict[str, object]]:
    """Return the parsed JSONL records from ``path``."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_demo_produces_expected_artifacts(tmp_path: Path) -> None:
    """Running the demo writes episode.jsonl, summary.json, metrics.json, the viewer, and a thumbnail."""
    output_root = tmp_path / "demo" / "latest"
    result = demo.run_demo(output_root=output_root, seed=270)

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
    assert summary["artifacts"]["viewer_html"] == "viewer/index.html"
    assert metrics["episode_jsonl"] == "episode.jsonl"


def test_demo_episode_is_deterministic_and_records_steps(tmp_path: Path) -> None:
    """The same seed reproduces an identical recorded episode (stable, reviewable demo output)."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    run_a = demo.run_demo(output_root=root_a, seed=270)
    run_b = demo.run_demo(output_root=root_b, seed=270)

    lines_a = _read_jsonl_lines(run_a.episode_jsonl)
    lines_b = _read_jsonl_lines(run_b.episode_jsonl)

    assert len(lines_a) == len(lines_b) > 0
    assert lines_a == lines_b
    assert (run_a.viewer_html.parent / "scene.json").read_bytes() == (
        run_b.viewer_html.parent / "scene.json"
    ).read_bytes()


def test_demo_cli_dispatches(tmp_path: Path, capsys, monkeypatch) -> None:
    """The umbrella ``robot-sf demo`` subcommand runs and prints the plain-English summary."""
    monkeypatch.chdir(tmp_path)
    rc = cli_main(["demo", "--output-root", str(tmp_path / "cli"), "--seed", "270"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Robot SF demo complete." in out
    assert (tmp_path / "cli" / "episode.jsonl").exists()


def test_demo_rejects_empty_scenario_manifest(tmp_path: Path) -> None:
    """A missing demo scenario fails closed instead of running a different default."""
    scenario_path = tmp_path / "empty.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing scenarios"):
        demo.run_demo(
            output_root=tmp_path / "output",
            scenario_path=scenario_path,
            check_deps=False,
        )


def test_demo_fails_when_runtime_check_fails(tmp_path: Path, monkeypatch) -> None:
    """A failed required-runtime check stops the demo before simulation output is written."""
    monkeypatch.setattr(demo, "_check_runtime_requirements", lambda: False)

    with pytest.raises(RuntimeError, match="Required runtime checks failed"):
        demo.run_demo(output_root=tmp_path / "output")
