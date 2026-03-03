"""Tests for canonical PPO eval timeline emission helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.training.train_expert_ppo import _timeline_from_episode_records, _write_eval_timeline

if TYPE_CHECKING:
    from pathlib import Path


def test_timeline_from_episode_records_groups_eval_steps() -> None:
    """Timeline builder should emit one summary row per eval checkpoint."""
    records = [
        {
            "eval_step": 100,
            "metrics": {"success_rate": 1.0, "collision_rate": 0.0, "snqi": 0.5},
        },
        {
            "eval_step": 100,
            "metrics": {"success_rate": 0.0, "collision_rate": 1.0, "snqi": -0.5},
        },
        {
            "eval_step": 200,
            "metrics": {"success_rate": 1.0, "collision_rate": 0.0, "snqi": 0.8},
        },
    ]
    timeline = _timeline_from_episode_records(eval_steps=[100, 200], episode_records=records)
    assert [int(row["eval_step"]) for row in timeline] == [100, 200]
    first = timeline[0]
    assert float(first["success_rate"]) == 0.5
    assert float(first["collision_rate"]) == 0.5


def test_write_eval_timeline_writes_json_and_csv(tmp_path: Path, monkeypatch) -> None:
    """Timeline writer should persist machine-readable JSON and CSV artifacts."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    timeline = [
        {"eval_step": 200, "success_rate": 0.4, "collision_rate": 0.6},
        {"eval_step": 100, "success_rate": 0.8, "collision_rate": 0.2},
    ]
    json_path = _write_eval_timeline(run_id="demo_run", timeline=timeline)
    assert json_path.exists()
    csv_path = json_path.with_suffix(".csv")
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [int(row["eval_step"]) for row in payload] == [100, 200]
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "eval_step" in csv_text
    assert "success_rate" in csv_text
