"""Tests for camera-ready campaign analysis tooling."""

from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.scenario_difficulty import build_scenario_difficulty_analysis
from scripts.tools.analyze_camera_ready_campaign import (
    _build_markdown_report,
    _build_scenario_difficulty_markdown,
    _recompute_legacy_campaign_integrity,
    analyze_campaign,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    """Write an indented JSON artifact fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write episode rows as a JSONL artifact fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_legacy_integrity_campaign(campaign_root: Path, rows: list[dict]) -> None:
    """Create a frozen pre-integrity campaign bundle with complete provenance inputs."""
    episodes_path = campaign_root / "runs" / "goal" / "episodes.jsonl"
    _write_jsonl(episodes_path, rows)
    _write_json(
        campaign_root / "preflight" / "preview_scenarios.json",
        {
            "truncated": False,
            "scenarios": [
                {"id": "smoke", "seeds": [111]},
                {"id": "corner", "seeds": [222]},
            ],
        },
    )
    _write_json(
        campaign_root / "campaign_manifest.json",
        {
            "seed_policy": {"resolved_seeds": [111, 222]},
            "git": {"commit": "commit-a"},
            "artifacts": {"preflight_preview_scenarios": "preflight/preview_scenarios.json"},
        },
    )
    _write_json(
        campaign_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "legacy"},
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "algo": "goal",
                    "status": "ok",
                    "success_mean": "0.5000",
                    "collision_mean": "0.0000",
                    "snqi_mean": "0.0000",
                }
            ],
            "runs": [
                {
                    "status": "ok",
                    "planner": {"key": "goal", "algo": "goal"},
                    "episodes_path": "runs/goal/episodes.jsonl",
                    "summary": {
                        "written": len(rows),
                        "episodes_total": len(rows),
                        "episodes_per_second": 2.0,
                        "preflight": {"status": "ok"},
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "disabled"}},
                    },
                }
            ],
        },
    )


def _legacy_integrity_row(scenario_id: str, seed: int, config_hash: str, commit: str) -> dict:
    """Build one episode row with the provenance fields required by the shared checker."""
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "status": "success",
        "metrics": {"success": True, "collisions": 0, "snqi": 0.0},
        "config_hash": config_hash,
        "git_hash": commit,
        "result_provenance": {
            "scenario_id": scenario_id,
            "seed": seed,
            "config_hash": config_hash,
            "repo_commit": commit,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write CSV rows using the keys from the first fixture row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_scenario_difficulty_campaign(campaign_root: Path) -> None:
    """Create a compact campaign tree with enough data for difficulty analysis."""
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


def test_analyze_campaign_surfaces_frozen_integrity_blocker(tmp_path: Path) -> None:
    """Frozen analysis carries the final blocker without rerunning any episode."""
    campaign_root = tmp_path / "campaign"
    _write_json(
        campaign_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "frozen"},
            "runs": [],
            "planner_rows": [],
            "campaign_integrity": {
                "status": "invalid",
                "claim_boundary": "A derived clean slice is diagnostic-only.",
                "blockers": [
                    {
                        "arm": "goal (differential_drive)",
                        "invariant": "duplicate_logical_coverage",
                        "details": {"identities": [["smoke", 111]]},
                    }
                ],
            },
        },
    )

    analysis = analyze_campaign(campaign_root)

    assert analysis["campaign_integrity"]["status"] == "invalid"
    assert any("duplicate_logical_coverage" in finding for finding in analysis["findings"])
    report = _build_markdown_report(analysis)
    assert "## Aggregate Integrity" in report
    assert "duplicate_logical_coverage" in report


def test_analyze_campaign_recomputes_legacy_contamination_blockers(tmp_path: Path) -> None:
    """Legacy frozen bundles recompute every shared aggregate-integrity blocker."""
    campaign_root = tmp_path / "campaign"
    _write_legacy_integrity_campaign(
        campaign_root,
        [
            _legacy_integrity_row("smoke", 111, "config-v1", "commit-a"),
            _legacy_integrity_row("smoke", 111, "config-v2", "commit-b"),
            _legacy_integrity_row("corner", 222, "config-corner", "commit-a"),
        ],
    )

    analysis = analyze_campaign(campaign_root)

    assert analysis["campaign_integrity"]["status"] == "invalid"
    assert {blocker["invariant"] for blocker in analysis["campaign_integrity"]["blockers"]} == {
        "count_mismatch",
        "duplicate_logical_coverage",
        "mixed_commit_provenance",
        "mixed_config_provenance",
    }
    aggregate_findings = [
        finding for finding in analysis["findings"] if finding.startswith("aggregate_integrity:")
    ]
    expected_invariants = {
        "count_mismatch",
        "duplicate_logical_coverage",
        "mixed_commit_provenance",
        "mixed_config_provenance",
    }
    assert all(
        any(invariant in finding for finding in aggregate_findings)
        for invariant in expected_invariants
    )


def test_analyze_campaign_recomputes_legacy_clean_bundle(tmp_path: Path) -> None:
    """Legacy clean bundles remain valid without a persisted integrity report."""
    campaign_root = tmp_path / "campaign"
    _write_legacy_integrity_campaign(
        campaign_root,
        [
            _legacy_integrity_row("smoke", 111, "config-smoke", "commit-a"),
            _legacy_integrity_row("corner", 222, "config-corner", "commit-a"),
        ],
    )

    analysis = analyze_campaign(campaign_root)

    assert analysis["campaign_integrity"]["status"] == "valid"
    assert not any(finding.startswith("aggregate_integrity:") for finding in analysis["findings"])


def test_analyze_campaign_fails_closed_when_legacy_provenance_is_missing(tmp_path: Path) -> None:
    """Legacy bundles without a complete scenario inventory are not silently unevaluated."""
    campaign_root = tmp_path / "campaign"
    _write_json(
        campaign_root / "campaign_manifest.json",
        {"seed_policy": {"resolved_seeds": [111]}, "git": {"commit": "commit-a"}},
    )
    _write_json(
        campaign_root / "reports" / "campaign_summary.json",
        {"campaign": {"campaign_id": "legacy"}, "runs": []},
    )

    analysis = analyze_campaign(campaign_root)

    assert analysis["campaign_integrity"]["status"] == "not_evaluable"
    assert analysis["campaign_integrity"]["blockers"][0]["invariant"] == (
        "missing_integrity_provenance"
    )


def test_analyze_campaign_fails_closed_for_non_mapping_integrity_json(tmp_path: Path) -> None:
    """A malformed integrity root remains a structured non-success result."""
    campaign_root = tmp_path / "campaign"
    _write_json(
        campaign_root / "reports" / "campaign_summary.json",
        {"campaign": {"campaign_id": "legacy"}, "runs": []},
    )
    integrity_path = campaign_root / "reports" / "campaign_integrity.json"
    integrity_path.write_text("[\"invalid-root\"]\n", encoding="utf-8")

    analysis = analyze_campaign(campaign_root)

    assert analysis["campaign_integrity"]["status"] == "not_evaluable"
    assert analysis["campaign_integrity"]["blockers"][0]["invariant"] == (
        "unreadable_integrity_provenance"
    )


def test_legacy_recomputation_does_not_mutate_nested_run_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path normalization must isolate nested summary data from validator mutations."""
    campaign_root = tmp_path / "campaign"
    _write_legacy_integrity_campaign(
        campaign_root,
        [_legacy_integrity_row("smoke", 111, "config-smoke", "commit-a")],
    )
    summary_payload = json.loads(
        (campaign_root / "reports" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    original_runs = deepcopy(summary_payload["runs"])

    def mutate_entries(run_entries: list[dict], **_: object) -> dict:
        run_entries[0]["planner"]["key"] = "mutated"
        return {"status": "valid", "blockers": []}

    monkeypatch.setattr(
        "scripts.tools.analyze_camera_ready_campaign.validate_campaign_integrity",
        mutate_entries,
    )

    _recompute_legacy_campaign_integrity(campaign_root, summary_payload)

    assert summary_payload["runs"] == original_runs


def test_analyze_campaign_cli_returns_nonzero_for_invalid_legacy_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI status distinguishes an invalid frozen aggregate from a successful analysis."""
    campaign_root = tmp_path / "campaign"
    _write_legacy_integrity_campaign(
        campaign_root,
        [
            _legacy_integrity_row("smoke", 111, "config-v1", "commit-a"),
            _legacy_integrity_row("smoke", 111, "config-v2", "commit-b"),
            _legacy_integrity_row("corner", 222, "config-corner", "commit-a"),
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_camera_ready_campaign.py", "--campaign-root", str(campaign_root)],
    )

    assert main() == 1
    assert json.loads(capsys.readouterr().out)["analysis_json"].endswith(
        "reports/campaign_analysis.json"
    )


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
                    "planner_group": "core",
                    "status": "ok",
                    "preflight_status": "ok",
                    "benchmark_success": "true",
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


def test_analyze_campaign_matches_rows_by_planner_and_kinematics(tmp_path: Path) -> None:
    """Cross-kinematics rows should compare against their matching run only."""
    campaign_root = tmp_path / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    diff_episodes_path = campaign_root / "runs" / "goal__differential_drive" / "episodes.jsonl"
    holo_episodes_path = campaign_root / "runs" / "goal__holonomic" / "episodes.jsonl"

    base_episode = {
        "status": "failure",
        "termination_reason": "timeout",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
        "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
    }
    _write_jsonl(
        diff_episodes_path,
        [{**base_episode, "metrics": {"success": False, "collisions": 0, "snqi": -0.4}}],
    )
    _write_jsonl(
        holo_episodes_path,
        [{**base_episode, "metrics": {"success": False, "collisions": 0, "snqi": -0.1}}],
    )
    _write_json(
        summary_path,
        {
            "campaign": {
                "campaign_id": "cross_kinematics",
                "runtime_sec": 2.0,
                "episodes_per_second": 1.0,
            },
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "kinematics": "differential_drive",
                    "planner_group": "core",
                    "status": "ok",
                    "preflight_status": "ok",
                    "benchmark_success": "true",
                    "success_mean": "0.0000",
                    "collisions_mean": "0.0000",
                    "snqi_mean": "-0.4000",
                },
                {
                    "planner_key": "goal",
                    "kinematics": "holonomic",
                    "planner_group": "core",
                    "status": "ok",
                    "preflight_status": "ok",
                    "benchmark_success": "true",
                    "success_mean": "0.0000",
                    "collisions_mean": "0.0000",
                    "snqi_mean": "-0.1000",
                },
            ],
            "runs": [
                {
                    "planner": {
                        "key": " goal ",
                        "algo": "goal",
                        "kinematics": "differential_drive",
                    },
                    "runtime_sec": 1.0,
                    "episodes_path": "runs/goal__differential_drive/episodes.jsonl",
                    "summary": {
                        "written": 1,
                        "episodes_per_second": 1.0,
                        "preflight": {"status": "ok"},
                        "kinematics": "differential_drive",
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "disabled"}},
                    },
                },
                {
                    "planner": {"key": "goal", "algo": "goal", "kinematics": "holonomic"},
                    "runtime_sec": 1.0,
                    "episodes_path": "runs/goal__holonomic/episodes.jsonl",
                    "summary": {
                        "written": 1,
                        "episodes_per_second": 1.0,
                        "preflight": {"status": "ok"},
                        "kinematics": "holonomic",
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "disabled"}},
                    },
                },
            ],
        },
    )

    analysis = analyze_campaign(campaign_root)

    assert analysis["findings"] == []
    assert {(item["planner_key"], item["kinematics"]) for item in analysis["planners"]} == {
        ("goal", "differential_drive"),
        ("goal", "holonomic"),
    }


def test_analyze_campaign_accepts_repo_relative_paths_from_campaign_checkout(
    tmp_path: Path,
) -> None:
    """Analyzer should resolve repo-relative output paths using the campaign checkout root."""
    repo_root = tmp_path / "external_checkout"
    campaign_root = repo_root / "output" / "benchmarks" / "camera_ready" / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    episodes_path = campaign_root / "runs" / "goal" / "episodes.jsonl"
    scenario_breakdown_path = campaign_root / "reports" / "scenario_breakdown.csv"

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
    _write_csv(
        scenario_breakdown_path,
        [
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "2",
                "success_mean": "0.5000",
                "collisions_mean": "0.5000",
                "near_misses_mean": "0.0000",
                "time_to_goal_norm_mean": "0.5000",
                "snqi_mean": "-0.2000",
            }
        ],
    )
    _write_json(
        summary_path,
        {
            "campaign": {
                "campaign_id": "external_checkout_campaign",
                "runtime_sec": 1.0,
                "episodes_per_second": 2.0,
            },
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "planner_group": "core",
                    "status": "ok",
                    "preflight_status": "ok",
                    "benchmark_success": "true",
                    "success_mean": "0.5000",
                    "collision_mean": "0.5000",
                    "snqi_mean": "-0.2000",
                }
            ],
            "artifacts": {
                "scenario_breakdown_csv": (
                    "output/benchmarks/camera_ready/campaign/reports/scenario_breakdown.csv"
                ),
            },
            "runs": [
                {
                    "planner": {"key": "goal", "algo": "goal"},
                    "runtime_sec": 1.0,
                    "episodes_path": "output/benchmarks/camera_ready/campaign/runs/goal/episodes.jsonl",
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
    assert analysis["planners"][0]["episodes_file"] == 2
    assert analysis["scenario_difficulty"]["scenario_rows"][0]["scenario_id"] == "easy_case"


def test_analyze_campaign_accepts_repo_relative_paths_after_campaign_relocation(
    tmp_path: Path,
) -> None:
    """A self-contained retrieved campaign should not need its original checkout path."""
    campaign_root = tmp_path / "retrieved" / "13378"
    episodes_path = campaign_root / "runs" / "goal" / "episodes.jsonl"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    _write_jsonl(
        episodes_path,
        [
            {
                "status": "success",
                "termination_reason": "success",
                "outcome": {
                    "route_complete": True,
                    "collision_event": False,
                    "timeout_event": False,
                },
                "integrity": {"contradictions": []},
                "metrics": {"success": True, "collisions": 0, "snqi": -0.1},
                "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
            }
        ],
    )
    _write_json(
        summary_path,
        {
            "campaign": {
                "campaign_id": "original_campaign",
                "runtime_sec": 1.0,
                "episodes_per_second": 1.0,
            },
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "success_mean": "1.0000",
                    "collisions_mean": "0.0000",
                    "snqi_mean": "-0.1000",
                }
            ],
            "runs": [
                {
                    "planner": {"key": "goal", "algo": "goal"},
                    "runtime_sec": 1.0,
                    "episodes_path": (
                        "output/benchmarks/camera_ready/original_campaign/runs/goal/episodes.jsonl"
                    ),
                    "summary": {
                        "written": 1,
                        "episodes_per_second": 1.0,
                        "preflight": {"status": "ok"},
                        "algorithm_metadata_contract": {"adapter_impact": {"status": "disabled"}},
                    },
                }
            ],
        },
    )

    analysis = analyze_campaign(campaign_root)

    assert analysis["planners"][0]["episodes_file"] == 1


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


def test_analyze_campaign_uses_canonical_snqi_episode_metric_value(tmp_path: Path) -> None:
    """Analyzer SNQI rollups should match canonical episode metric extraction."""
    campaign_root = tmp_path / "campaign"
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    episodes_path = campaign_root / "runs" / "goal" / "episodes.jsonl"

    _write_jsonl(
        episodes_path,
        [
            {
                "status": "success",
                "termination_reason": "success",
                "snqi": "nan",
                "metrics": {"success": True, "collisions": 0, "snqi": 0.4},
                "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
            },
            {
                "status": "failure",
                "termination_reason": "collision",
                "snqi": -0.2,
                "metrics": {"success": False, "collisions": 1},
                "algorithm_metadata": {"adapter_impact": {"status": "disabled"}},
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
                    "planner_key": "goal",
                    "success_mean": "0.5000",
                    "collisions_mean": "0.5000",
                    "snqi_mean": "0.1000",
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
    planner = next(item for item in analysis["planners"] if item["planner_key"] == "goal")

    assert planner["snqi_mean_episodes"] == pytest.approx(0.1)
    assert not any("snqi_mean mismatch" in finding for finding in analysis["findings"])


def test_analyze_campaign_includes_scenario_difficulty_outputs(tmp_path: Path) -> None:
    """Analyzer should merge scenario-difficulty diagnostics from campaign report artifacts because weak planner results need scenario-level context."""
    campaign_root = tmp_path / "campaign"
    _write_scenario_difficulty_campaign(campaign_root)

    analysis = analyze_campaign(campaign_root)

    difficulty = analysis["scenario_difficulty"]
    assert difficulty["scenario_rows"][0]["scenario_id"] == "hard_case"
    assert difficulty["family_rows"][0]["scenario_count"] == 1
    assert difficulty["verified_simple_assessment"]["status"] == "rerun_required"


def test_analyze_campaign_markdown_labels_normalized_time_to_goal(tmp_path: Path) -> None:
    """Scenario difficulty markdown should name the normalized time metric explicitly."""
    campaign_root = tmp_path / "campaign"
    _write_scenario_difficulty_campaign(campaign_root)

    analysis = analyze_campaign(campaign_root)
    markdown = _build_scenario_difficulty_markdown(analysis["scenario_difficulty"])
    assert "time_to_goal_norm" in markdown
    assert " | time_to_goal | " not in markdown


def test_analyze_campaign_markdown_surfaces_fallback_description() -> None:
    """Markdown output should mirror the fallback consensus description so non-paper reports do not read like a core-planner consensus."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "planner_group": "experimental",
                "status": "ok",
                "preflight_status": "ok",
                "benchmark_success": "true",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "planner_group": "experimental",
                "status": "ok",
                "preflight_status": "ok",
                "benchmark_success": "true",
            },
        ],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.90",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.10",
                "time_to_goal_norm_mean": "0.35",
                "snqi_mean": "0.20",
            },
        ],
    )

    markdown = _build_scenario_difficulty_markdown(analysis)
    assert "Description:" in markdown
    assert "all planners in the scenario breakdown" in markdown
    assert "no eligible core benchmark-success planners were available" in markdown


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


def test_analyze_campaign_cli_respects_output_overrides_for_difficulty_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sidecar scenario-difficulty outputs should follow explicit output overrides."""
    campaign_root = tmp_path / "campaign"
    _write_scenario_difficulty_campaign(campaign_root)
    output_dir = tmp_path / "redirected_reports"
    output_json = output_dir / "custom_campaign_analysis.json"
    output_md = output_dir / "custom_campaign_analysis.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_camera_ready_campaign.py",
            "--campaign-root",
            str(campaign_root),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    assert main() == 0

    stdout_payload = json.loads(capsys.readouterr().out)
    difficulty_json = output_dir / "scenario_difficulty_analysis.json"
    difficulty_md = output_dir / "scenario_difficulty_analysis.md"

    assert stdout_payload == {
        "analysis_json": str(output_json.resolve()),
        "analysis_md": str(output_md.resolve()),
        "scenario_difficulty_json": str(difficulty_json.resolve()),
        "scenario_difficulty_md": str(difficulty_md.resolve()),
    }
    assert difficulty_json.exists()
    assert difficulty_md.exists()


def test_analyze_campaign_emits_credibility_scorecard(tmp_path: Path) -> None:
    """Campaign analysis emits a diagnostic credibility scorecard per campaign."""
    campaign_root = tmp_path / "campaign"
    _write_scenario_difficulty_campaign(campaign_root)

    analysis = analyze_campaign(campaign_root)
    scorecard = analysis["credibility_scorecard"]

    assert scorecard["schema_version"] == "nasa-7009-style-credibility-scorecard.v1"
    assert scorecard["status"] == "credible_diagnostic"
    assert scorecard["score"] == pytest.approx(0.82)
    assert "not paper-facing evidence" in scorecard["claim_boundary"]
    assert {check["check_id"] for check in scorecard["checks"]} == {
        "traceable_campaign_artifacts",
        "verification_consistency",
        "validation_coverage",
        "uncertainty_characterization",
        "limitations_explicit",
    }
    assert scorecard["fail_closed_blockers"] == []
    checks_by_id = {check["check_id"]: check for check in scorecard["checks"]}
    assert checks_by_id["verification_consistency"]["status"] == "warning"
    assert checks_by_id["limitations_explicit"]["status"] == "warning"

    markdown = _build_markdown_report(analysis)
    assert "## Credibility Scorecard" in markdown
    assert "| Traceable campaign artifacts | pass | 1.0000 |" in markdown
    assert "not a benchmark-success promotion" in markdown
