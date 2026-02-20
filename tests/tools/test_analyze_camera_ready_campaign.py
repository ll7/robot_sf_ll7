"""Tests for camera-ready campaign analysis tooling."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools.analyze_camera_ready_campaign import analyze_campaign

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_analyze_campaign_detects_adapter_status_mismatch(tmp_path: Path) -> None:
    """Analyzer flags adapter-impact status drift between summary and episodes."""
    campaign_root = tmp_path / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    episodes_path = campaign_root / "runs" / "ppo" / "episodes.jsonl"

    _write_jsonl(
        episodes_path,
        [
            {
                "status": "success",
                "metrics": {"success": True, "collisions": 0, "snqi": -0.2},
                "algorithm_metadata": {"adapter_impact": {"status": "complete"}},
            },
            {
                "status": "failure",
                "metrics": {"success": False, "collisions": 0, "snqi": -0.1},
                "algorithm_metadata": {"adapter_impact": {"status": "complete"}},
            },
        ],
    )
    _write_json(
        summary_path,
        {
            "campaign": {
                "campaign_id": "test_campaign",
                "runtime_sec": 1.0,
                "episodes_per_second": 2.0,
            },
            "planner_rows": [
                {
                    "planner_key": "ppo",
                    "success_mean": "0.5000",
                    "collision_mean": "0.0000",
                    "snqi_mean": "-0.1500",
                }
            ],
            "runs": [
                {
                    "planner": {"key": "ppo", "algo": "ppo"},
                    "runtime_sec": 1.0,
                    "episodes_path": "runs/ppo/episodes.jsonl",
                    "summary": {
                        "written": 2,
                        "episodes_per_second": 2.0,
                        "preflight": {"status": "ok"},
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "pending"}},
                    },
                }
            ],
        },
    )

    analysis = analyze_campaign(campaign_root)
    findings = analysis["findings"]
    assert any("adapter impact status mismatch" in finding for finding in findings)
    assert any(item["planner_key"] == "ppo" for item in analysis["planners"])
    assert "runtime_hotspots" in analysis


def test_analyze_campaign_no_findings_on_consistent_payload(tmp_path: Path) -> None:
    """Analyzer emits no findings when summary and episodes are consistent."""
    campaign_root = tmp_path / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    episodes_path = campaign_root / "runs" / "goal" / "episodes.jsonl"

    rows = [
        {
            "status": "success",
            "metrics": {"success": True, "collisions": 0, "snqi": -0.3},
            "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
        },
        {
            "status": "failure",
            "metrics": {"success": False, "collisions": 1, "snqi": -0.1},
            "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
        },
    ]
    _write_jsonl(episodes_path, rows)
    _write_json(
        summary_path,
        {
            "campaign": {
                "campaign_id": "test_campaign",
                "runtime_sec": 1.0,
                "episodes_per_second": 2.0,
            },
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "success_mean": "0.5000",
                    "collision_mean": "0.5000",
                    "snqi_mean": "-0.2000",
                }
            ],
            "runs": [
                {
                    "planner": {"key": "goal", "algo": "goal"},
                    "runtime_sec": 1.0,
                    "episodes_path": "runs/goal/episodes.jsonl",
                    "summary": {
                        "written": 2,
                        "episodes_per_second": 2.0,
                        "preflight": {"status": "ok"},
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "disabled"}},
                    },
                }
            ],
        },
    )

    analysis = analyze_campaign(campaign_root)
    assert analysis["findings"] == []
    assert analysis["runtime_hotspots"]["slowest_planners"][0]["planner_key"] == "goal"


def test_analyze_campaign_flags_absolute_map_paths(tmp_path: Path) -> None:
    """Analyzer flags absolute map paths because they hurt artifact portability."""
    campaign_root = tmp_path / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    episodes_path = campaign_root / "runs" / "goal" / "episodes.jsonl"

    rows = [
        {
            "status": "success",
            "metrics": {"success": True, "collisions": 0, "snqi": -0.1},
            "scenario_params": {"map_file": "/tmp/absolute_map.svg"},
            "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
        },
        {
            "status": "success",
            "metrics": {"success": True, "collisions": 0, "snqi": -0.1},
            "scenario_params": {"map_file": "maps/relative_map.svg"},
            "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
        },
    ]
    _write_jsonl(episodes_path, rows)
    _write_json(
        summary_path,
        {
            "campaign": {
                "campaign_id": "test_campaign",
                "runtime_sec": 1.0,
                "episodes_per_second": 1.0,
            },
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "success_mean": "1.0000",
                    "collision_mean": "0.0000",
                    "snqi_mean": "-0.1000",
                }
            ],
            "runs": [
                {
                    "planner": {"key": "goal", "algo": "goal"},
                    "runtime_sec": 1.0,
                    "episodes_path": "runs/goal/episodes.jsonl",
                    "summary": {
                        "written": 2,
                        "episodes_per_second": 1.0,
                        "preflight": {"status": "ok"},
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "disabled"}},
                    },
                }
            ],
        },
    )

    analysis = analyze_campaign(campaign_root)
    assert any("non-portable provenance" in finding for finding in analysis["findings"])
    planner = next(item for item in analysis["planners"] if item["planner_key"] == "goal")
    assert planner["absolute_map_path_count"] == 1


def test_analyze_campaign_rejects_unsafe_episode_paths(tmp_path: Path) -> None:
    """Analyzer should reject traversal-style episode paths in summary payloads."""
    campaign_root = tmp_path / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    _write_json(
        summary_path,
        {
            "campaign": {
                "campaign_id": "test_campaign",
                "runtime_sec": 1.0,
                "episodes_per_second": 2.0,
            },
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "success_mean": "0.0",
                    "collision_mean": "0.0",
                    "snqi_mean": "0.0",
                }
            ],
            "runs": [
                {
                    "planner": {"key": "goal", "algo": "goal"},
                    "runtime_sec": 1.0,
                    "episodes_path": "../secrets.jsonl",
                    "summary": {
                        "written": 0,
                        "episodes_per_second": 0.0,
                        "preflight": {"status": "ok"},
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "disabled"}},
                    },
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="Unsafe relative episodes_path"):
        analyze_campaign(campaign_root)
