"""CLI contract tests for the classic-interactions runner."""

from __future__ import annotations

from pathlib import Path

import scripts.run_classic_interactions as classic_runner


def test_run_classic_interactions_accepts_custom_scenario_matrix(monkeypatch, tmp_path) -> None:
    """A custom matrix path should reach the benchmark runner without changing defaults."""
    matrix = Path("configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml")
    output = tmp_path / "episodes.jsonl"
    captured = {}

    def fake_run_batch(scenario_matrix: Path, **kwargs):
        captured["scenario_matrix"] = scenario_matrix
        captured.update(kwargs)
        return {"episodes": 0, "status": "mocked"}

    monkeypatch.setattr(classic_runner, "run_batch", fake_run_batch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_classic_interactions.py",
            "--scenario-matrix",
            str(matrix),
            "--output",
            str(output),
            "--workers",
            "1",
            "--horizon",
            "120",
            "--no-resume",
        ],
    )

    assert classic_runner.main() == 0
    assert captured["scenario_matrix"] == matrix
    assert captured["out_path"] == output
    assert captured["workers"] == 1
    assert captured["horizon"] == 120
    assert captured["resume"] is False
    assert output.with_suffix(".summary.json").exists()
