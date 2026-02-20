"""Tests for camera-ready campaign comparison helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.tools.compare_camera_ready_campaigns import compare_campaigns

if TYPE_CHECKING:
    from pathlib import Path


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_compare_campaigns_reports_prediction_success_gain(tmp_path: Path) -> None:
    """Comparison should capture metric deltas between base and candidate planner rows."""
    base_root = tmp_path / "base_campaign"
    candidate_root = tmp_path / "candidate_campaign"
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "base"},
            "planner_rows": [
                {
                    "planner_key": "prediction_planner",
                    "status": "partial-failure",
                    "episodes": 0,
                    "success_mean": "0.0",
                    "collisions_mean": "0.0",
                }
            ],
        },
    )
    _write_summary(
        candidate_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "candidate"},
            "planner_rows": [
                {
                    "planner_key": "prediction_planner",
                    "status": "ok",
                    "episodes": 135,
                    "success_mean": "0.9778",
                    "collisions_mean": "0.0000",
                }
            ],
        },
    )

    payload = compare_campaigns(base_root, candidate_root)
    planner = payload["planner_deltas"][0]
    assert planner["planner_key"] == "prediction_planner"
    assert planner["base_status"] == "partial-failure"
    assert planner["candidate_status"] == "ok"
    assert planner["base_episodes"] == 0
    assert planner["candidate_episodes"] == 135
    assert planner["metrics"]["success_mean"]["delta"] > 0.9
