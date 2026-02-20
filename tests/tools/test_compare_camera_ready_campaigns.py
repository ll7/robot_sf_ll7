"""Tests for camera-ready campaign comparison helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools.compare_camera_ready_campaigns import (
    _build_markdown,
    _resolve_safe_output_path,
    compare_campaigns,
)

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


def test_compare_campaigns_reports_missing_planners(tmp_path: Path) -> None:
    """Comparison should explicitly report planners present in only one campaign."""
    base_root = tmp_path / "base_campaign"
    candidate_root = tmp_path / "candidate_campaign"
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "base"},
            "planner_rows": [
                {"planner_key": "goal", "status": "ok", "episodes": 10, "success_mean": "1.0"},
                {"planner_key": "orca", "status": "ok", "episodes": 10, "success_mean": "1.0"},
            ],
        },
    )
    _write_summary(
        candidate_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "candidate"},
            "planner_rows": [
                {"planner_key": "goal", "status": "ok", "episodes": 10, "success_mean": "1.0"},
                {
                    "planner_key": "prediction_planner",
                    "status": "ok",
                    "episodes": 10,
                    "success_mean": "1.0",
                },
            ],
        },
    )

    payload = compare_campaigns(base_root, candidate_root)
    assert payload["missing_in_base"] == ["prediction_planner"]
    assert payload["missing_in_candidate"] == ["orca"]


def test_resolve_safe_output_path_rejects_escape(tmp_path: Path) -> None:
    """Output path validation should reject writes outside safe root."""
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    inside = _resolve_safe_output_path(safe_root / "reports" / "ok.json", safe_root)
    assert inside.is_relative_to(safe_root)
    with pytest.raises(ValueError, match="Unsafe output path"):
        _resolve_safe_output_path(tmp_path / ".." / "outside.json", safe_root)


def test_compare_campaigns_filters_non_json_float_values(tmp_path: Path) -> None:
    """Infinite and NaN metrics should be excluded from planner metric deltas."""
    base_root = tmp_path / "base_campaign"
    candidate_root = tmp_path / "candidate_campaign"
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "base"},
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "status": "ok",
                    "episodes": 10,
                    "success_mean": "inf",
                    "collisions_mean": "nan",
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
                    "planner_key": "goal",
                    "status": "ok",
                    "episodes": 10,
                    "success_mean": "0.9",
                    "collisions_mean": "0.1",
                }
            ],
        },
    )

    payload = compare_campaigns(base_root, candidate_root)
    assert payload["planner_deltas"][0]["metrics"] == {}


def test_build_markdown_keeps_planners_without_numeric_metrics() -> None:
    """Markdown should still render planners with empty metrics as explicit N/A rows."""
    payload = {
        "base_campaign_id": "base",
        "candidate_campaign_id": "candidate",
        "planner_deltas": [
            {
                "planner_key": "prediction_planner",
                "base_status": "partial-failure",
                "candidate_status": "ok",
                "base_episodes": 0,
                "candidate_episodes": 135,
                "metrics": {},
            }
        ],
        "missing_in_base": [],
        "missing_in_candidate": [],
    }
    markdown = _build_markdown(payload)
    assert "base_status" in markdown
    assert "candidate_status" in markdown
    assert (
        "| prediction_planner | partial-failure | ok | 0 | 135 | N/A | N/A | N/A | N/A |"
        in markdown
    )
