"""Tests for PPO eval timeline analyzer tool."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.imitation_manifest import get_training_run_manifest_path
from scripts.tools.analyze_ppo_eval_timeline import main

if TYPE_CHECKING:
    from pathlib import Path


def test_analyze_ppo_eval_timeline_reads_manifest_artifact(tmp_path: Path, monkeypatch) -> None:
    """Analyzer should use eval_timeline_path from training manifest and emit reports."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))

    timeline_dir = tmp_path / "benchmarks" / "ppo_imitation" / "eval_timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = timeline_dir / "demo_run.json"
    timeline_rows = [
        {
            "eval_step": 100,
            "success_rate": 0.8,
            "collision_rate": 0.2,
            "path_efficiency": 0.7,
            "comfort_exposure": 0.1,
            "snqi": 0.2,
            "eval_episode_return": 1.0,
            "eval_avg_step_reward": 0.01,
        },
        {
            "eval_step": 200,
            "success_rate": 0.7,
            "collision_rate": 0.3,
            "path_efficiency": 0.6,
            "comfort_exposure": 0.1,
            "snqi": 0.1,
            "eval_episode_return": 0.8,
            "eval_avg_step_reward": 0.008,
        },
    ]
    timeline_path.write_text(json.dumps(timeline_rows), encoding="utf-8")

    manifest_path = get_training_run_manifest_path("demo_run")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "demo_run",
                "run_type": "expert_training",
                "input_artefacts": [],
                "seeds": [1],
                "metrics": {},
                "episode_log_path": "",
                "eval_timeline_path": "benchmarks/ppo_imitation/eval_timeline/demo_run.json",
                "wall_clock_hours": 0.0,
                "status": "completed",
                "notes": [],
            }
        ),
        encoding="utf-8",
    )

    output_json = tmp_path / "analysis" / "out.json"
    output_md = tmp_path / "analysis" / "out.md"
    exit_code = main(
        [
            "--run-id",
            "demo_run",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )
    assert exit_code == 0
    assert output_json.exists()
    assert output_md.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_id"] == "demo_run"
    assert payload["analysis"]["rows"] == 2
    assert payload["analysis"]["monotonic_eval_step"] is True
