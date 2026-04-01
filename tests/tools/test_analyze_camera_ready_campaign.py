"""Tests for camera-ready campaign analysis tooling."""

from __future__ import annotations

import csv
import json
import sys
from typing import TYPE_CHECKING

import pytest

from scripts.tools.analyze_camera_ready_campaign import analyze_campaign, main

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


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_scenario_difficulty_campaign(campaign_root: Path) -> None:
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    episodes_goal = campaign_root / "runs" / "goal" / "episodes.jsonl"
    episodes_orca = campaign_root / "runs" / "orca" / "episodes.jsonl"
    scenario_breakdown_path = campaign_root / "reports" / "scenario_breakdown.csv"
    seed_variability_path = campaign_root / "reports" / "seed_variability_by_scenario.json"
    preview_path = campaign_root / "preflight" / "preview_scenarios.json"

    _write_jsonl(
        episodes_goal,
        [
            {
                "status": "success",
                "termination_reason": "success",
                "metrics": {"success": True, "collisions": 0, "snqi": 0.2},
                "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
            },
            {
                "status": "failure",
                "termination_reason": "collision",
                "metrics": {"success": False, "collisions": 1, "snqi": -0.3},
                "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
            },
        ],
    )
    _write_jsonl(
        episodes_orca,
        [
            {
                "status": "success",
                "termination_reason": "success",
                "metrics": {"success": True, "collisions": 0, "snqi": 0.1},
                "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
            },
            {
                "status": "failure",
                "termination_reason": "collision",
                "metrics": {"success": False, "collisions": 1, "snqi": -0.2},
                "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
            },
        ],
    )
    _write_csv(
        scenario_breakdown_path,
        [
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.9500",
                "collisions_mean": "0.0000",
                "near_misses_mean": "0.0500",
                "time_to_goal_norm_mean": "0.3000",
                "snqi_mean": "0.3000",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.9000",
                "collisions_mean": "0.0000",
                "near_misses_mean": "0.1000",
                "time_to_goal_norm_mean": "0.3500",
                "snqi_mean": "0.2000",
            },
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.3500",
                "collisions_mean": "0.4500",
                "near_misses_mean": "0.7000",
                "time_to_goal_norm_mean": "0.8500",
                "snqi_mean": "-0.4000",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.4000",
                "collisions_mean": "0.3500",
                "near_misses_mean": "0.6500",
                "time_to_goal_norm_mean": "0.8000",
                "snqi_mean": "-0.3000",
            },
        ],
    )
    _write_json(
        seed_variability_path,
        {
            "rows": [
                {
                    "scenario_id": "easy_case",
                    "planner_key": "goal",
                    "seed_count": 3,
                    "summary": {
                        "success": {"ci_half_width": 0.02, "cv": 0.01},
                        "time_to_goal_norm": {"ci_half_width": 0.03, "cv": 0.02},
                        "snqi": {"ci_half_width": 0.05, "cv": 0.04},
                    },
                },
                {
                    "scenario_id": "easy_case",
                    "planner_key": "orca",
                    "seed_count": 3,
                    "summary": {
                        "success": {"ci_half_width": 0.03, "cv": 0.02},
                        "time_to_goal_norm": {"ci_half_width": 0.04, "cv": 0.03},
                        "snqi": {"ci_half_width": 0.05, "cv": 0.04},
                    },
                },
                {
                    "scenario_id": "hard_case",
                    "planner_key": "goal",
                    "seed_count": 3,
                    "summary": {
                        "success": {"ci_half_width": 0.10, "cv": 0.20},
                        "time_to_goal_norm": {"ci_half_width": 0.09, "cv": 0.10},
                        "snqi": {"ci_half_width": 0.08, "cv": 0.12},
                    },
                },
                {
                    "scenario_id": "hard_case",
                    "planner_key": "orca",
                    "seed_count": 3,
                    "summary": {
                        "success": {"ci_half_width": 0.09, "cv": 0.18},
                        "time_to_goal_norm": {"ci_half_width": 0.08, "cv": 0.11},
                        "snqi": {"ci_half_width": 0.07, "cv": 0.10},
                    },
                },
            ]
        },
    )
    _write_json(
        preview_path,
        {
            "truncated": False,
            "route_clearance_warnings": [
                {
                    "scenario": "hard_case",
                    "warning_scope": "route",
                    "min_clearance_margin_m": 0.2,
                }
            ],
            "scenarios": [
                {
                    "name": "easy_case",
                    "simulation_config": {"ped_density": 0.0},
                    "metadata": {
                        "archetype": "easy_family",
                        "flow": "none",
                        "behavior": "none",
                        "primary_capability": "frame_consistency",
                        "target_failure_mode": "coordinate_transform",
                        "determinism": "deterministic",
                    },
                },
                {
                    "name": "hard_case",
                    "simulation_config": {"ped_density": 0.5},
                    "metadata": {
                        "archetype": "hard_family",
                        "flow": "perpendicular",
                        "behavior": "crowd",
                        "primary_capability": "dynamic_interaction",
                        "target_failure_mode": "social_collision",
                        "determinism": "stochastic",
                    },
                },
            ],
        },
    )
    _write_json(
        summary_path,
        {
            "campaign": {
                "campaign_id": "test_campaign",
                "runtime_sec": 2.0,
                "episodes_per_second": 2.0,
            },
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "algo": "goal",
                    "planner_group": "core",
                    "status": "ok",
                    "preflight_status": "ok",
                    "benchmark_success": "true",
                    "success_mean": "0.5000",
                    "collision_mean": "0.5000",
                    "snqi_mean": "-0.0500",
                },
                {
                    "planner_key": "orca",
                    "algo": "orca",
                    "planner_group": "core",
                    "status": "ok",
                    "preflight_status": "ok",
                    "benchmark_success": "true",
                    "success_mean": "0.5000",
                    "collision_mean": "0.5000",
                    "snqi_mean": "-0.0500",
                },
            ],
            "artifacts": {
                "scenario_breakdown_csv": "reports/scenario_breakdown.csv",
                "seed_variability_json": "reports/seed_variability_by_scenario.json",
                "preflight_preview_scenarios": "preflight/preview_scenarios.json",
            },
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
                },
                {
                    "planner": {"key": "orca", "algo": "orca"},
                    "runtime_sec": 1.0,
                    "episodes_path": "runs/orca/episodes.jsonl",
                    "summary": {
                        "written": 2,
                        "episodes_per_second": 2.0,
                        "preflight": {"status": "ok"},
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "disabled"}},
                    },
                },
            ],
        },
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
            "termination_reason": "success",
            "outcome": {
                "route_complete": True,
                "collision_event": False,
                "timeout_event": False,
            },
            "integrity": {"contradictions": []},
            "metrics": {"success": True, "collisions": 0, "snqi": -0.3},
            "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
        },
        {
            "status": "failure",
            "termination_reason": "collision",
            "outcome": {
                "route_complete": False,
                "collision_event": True,
                "timeout_event": False,
            },
            "integrity": {"contradictions": []},
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


def test_analyze_campaign_accepts_legacy_success_rate_alias(tmp_path: Path) -> None:
    """Analyzer should treat metrics.success_rate as a legacy success alias."""
    campaign_root = tmp_path / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    episodes_path = campaign_root / "runs" / "goal" / "episodes.jsonl"

    rows = [
        {
            "status": "success",
            "outcome": {
                "route_complete": True,
                "collision_event": False,
                "timeout_event": False,
            },
            "metrics": {"success_rate": 1.0, "collisions": 0, "snqi": -0.3},
            "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
        },
        {
            "status": "failure",
            "outcome": {
                "route_complete": False,
                "collision_event": True,
                "timeout_event": False,
            },
            "metrics": {"success_rate": 0.0, "collisions": 1, "snqi": -0.1},
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
                    "collisions_mean": "0.5000",
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


def test_analyze_campaign_flags_success_collision_integrity_violations(tmp_path: Path) -> None:
    """Analyzer flags contradictory success/collision records."""
    campaign_root = tmp_path / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    episodes_path = campaign_root / "runs" / "orca" / "episodes.jsonl"

    _write_jsonl(
        episodes_path,
        [
            {
                "status": "collision",
                "termination_reason": "collision",
                "outcome": {
                    "route_complete": False,
                    "collision_event": True,
                    "timeout_event": False,
                },
                "integrity": {"contradictions": []},
                "metrics": {"success": 1.0, "collisions": 0.0, "snqi": -0.2},
                "algorithm_metadata": {"adapter_impact": {"status": "complete"}},
            }
        ],
    )
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
                    "planner_key": "orca",
                    "success_mean": "1.0000",
                    "collision_mean": "0.0000",
                    "snqi_mean": "-0.2000",
                }
            ],
            "runs": [
                {
                    "planner": {"key": "orca", "algo": "orca"},
                    "runtime_sec": 1.0,
                    "episodes_path": "runs/orca/episodes.jsonl",
                    "summary": {
                        "written": 1,
                        "episodes_per_second": 1.0,
                        "preflight": {"status": "ok"},
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "complete"}},
                    },
                }
            ],
        },
    )

    analysis = analyze_campaign(campaign_root)
    assert any("episode integrity violations" in finding for finding in analysis["findings"])


def test_analyze_campaign_derives_collision_mean_from_termination_reason(tmp_path: Path) -> None:
    """Analyzer should derive collisions from termination_reason when metrics are sparse."""
    campaign_root = tmp_path / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    episodes_path = campaign_root / "runs" / "ppo" / "episodes.jsonl"

    _write_jsonl(
        episodes_path,
        [
            {
                "status": "collision",
                "termination_reason": "collision",
                "metrics": {"snqi": -0.2},
                "algorithm_metadata": {"adapter_impact": {"status": "complete"}},
            },
            {
                "status": "success",
                "termination_reason": "success",
                "metrics": {"snqi": -0.1},
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
                    "collisions_mean": "0.0000",
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
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "complete"}},
                    },
                }
            ],
        },
    )

    analysis = analyze_campaign(campaign_root)
    planner = next(item for item in analysis["planners"] if item["planner_key"] == "ppo")
    assert planner["collision_mean_episodes"] == pytest.approx(0.5)
    assert any("collision_mean mismatch" in finding for finding in analysis["findings"])


def test_analyze_campaign_includes_scenario_difficulty_outputs(tmp_path: Path) -> None:
    """Analyzer should merge scenario-difficulty diagnostics from campaign report artifacts because weak planner results need scenario-level context."""
    campaign_root = tmp_path / "campaign"
    _write_scenario_difficulty_campaign(campaign_root)

    analysis = analyze_campaign(campaign_root)

    difficulty = analysis["scenario_difficulty"]
    assert difficulty["scenario_rows"][0]["scenario_id"] == "hard_case"
    assert difficulty["family_rows"][0]["scenario_count"] == 1
    assert difficulty["verified_simple_assessment"]["status"] == "rerun_required"


def test_analyze_campaign_cli_writes_scenario_difficulty_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI execution should write both campaign and scenario-difficulty reports because issue 692 is user-facing only if the post-run entrypoint emits concrete artifacts."""
    campaign_root = tmp_path / "campaign"
    _write_scenario_difficulty_campaign(campaign_root)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_camera_ready_campaign.py",
            "--campaign-root",
            str(campaign_root),
        ],
    )

    assert main() == 0

    stdout_payload = json.loads(capsys.readouterr().out)
    analysis_json = (campaign_root / "reports" / "campaign_analysis.json").resolve()
    analysis_md = (campaign_root / "reports" / "campaign_analysis.md").resolve()
    difficulty_json = (campaign_root / "reports" / "scenario_difficulty_analysis.json").resolve()
    difficulty_md = (campaign_root / "reports" / "scenario_difficulty_analysis.md").resolve()

    assert stdout_payload == {
        "analysis_json": str(analysis_json),
        "analysis_md": str(analysis_md),
        "scenario_difficulty_json": str(difficulty_json),
        "scenario_difficulty_md": str(difficulty_md),
    }

    analysis_payload = json.loads(analysis_json.read_text(encoding="utf-8"))
    difficulty_payload = json.loads(difficulty_json.read_text(encoding="utf-8"))

    assert analysis_payload["scenario_difficulty"]["scenario_rows"][0]["scenario_id"] == "hard_case"
    assert difficulty_payload["verified_simple_assessment"]["status"] == "rerun_required"
    assert "## Scenario Difficulty" in analysis_md.read_text(encoding="utf-8")
    assert "# Scenario Difficulty" in difficulty_md.read_text(encoding="utf-8")
