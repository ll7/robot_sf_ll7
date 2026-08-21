"""Tests for full benchmark-data release acceptance."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from robot_sf.benchmark.release_acceptance import validate_full_benchmark_release_acceptance

_PLANNER_KEYS = tuple(f"planner_{index:02d}" for index in range(14))
_SCENARIO_IDS = tuple(f"scenario_{index:02d}" for index in range(48))
_SEEDS = tuple(range(111, 141))
_SOURCE_SHA = "a" * 40


def _full_manifest() -> SimpleNamespace:
    """Return the fixed S30/H600 acceptance contract."""
    return SimpleNamespace(
        schema_version="benchmark-release-manifest.v0.2",
        expected_episode_cells=20_160,
        expected_horizon_steps=600,
        planner_keys=_PLANNER_KEYS,
        expected_kinematics_matrix=("differential_drive",),
        resolved_scenario_ids=_SCENARIO_IDS,
        resolved_seeds=_SEEDS,
    )


def _write_full_campaign(tmp_path: Path) -> Path:
    """Write a complete 14-arm fixture with 48 scenarios and 30 seeds."""
    campaign_root = tmp_path / "campaign"
    runs: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    for planner_key in _PLANNER_KEYS:
        relative_path = Path("runs") / planner_key / "episodes.jsonl"
        episode_path = campaign_root / relative_path
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for scenario_index, scenario_id in enumerate(_SCENARIO_IDS):
            for seed in _SEEDS:
                lines.append(
                    json.dumps(
                        {
                            "episode_id": f"{planner_key}-{scenario_id}-{seed}",
                            "scenario_id": scenario_id,
                            "seed": seed,
                            "horizon": 600,
                            "status": "success",
                            "git_hash": _SOURCE_SHA,
                            "result_provenance": {
                                "repo_commit": _SOURCE_SHA,
                                "config_hash": f"{scenario_index:016x}",
                                "scenario_id": scenario_id,
                                "seed": seed,
                                "simulator_settings": {"horizon": 600},
                            },
                        }
                    )
                )
        episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        runs.append(
            {
                "planner": {
                    "key": planner_key,
                    "kinematics": "differential_drive",
                    "horizon": 600,
                },
                "status": "ok",
                "episodes_path": relative_path.as_posix(),
                "summary": {"episodes_total": 1440, "written": 1440},
            }
        )
        planner_rows.append(
            {
                "planner_key": planner_key,
                "kinematics": "differential_drive",
                "status": "ok",
                "readiness_status": "available",
                "availability_status": "available",
                "benchmark_success": "true",
                "episodes": 1440,
            }
        )
    (campaign_root / "reports").mkdir(parents=True, exist_ok=True)
    (campaign_root / "reports" / "campaign_summary.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "benchmark_success",
                    "benchmark_success": True,
                    "evidence_status": "valid",
                    "campaign_execution_status": "completed",
                    "git_hash": _SOURCE_SHA,
                    "row_status_summary": {
                        "successful_evidence_rows": 14,
                        "accepted_unavailable_rows": 0,
                        "unexpected_failed_rows": 0,
                        "fallback_or_degraded_rows": 0,
                    },
                },
                "runs": runs,
                "planner_rows": planner_rows,
                "campaign_integrity": {
                    "status": "valid",
                    "benchmark_success_allowed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return campaign_root


def test_full_release_acceptance_requires_all_arms_and_episode_cells(tmp_path: Path) -> None:
    """A complete S30/H600 fixture is accepted as publication-grade evidence."""
    campaign_root = _write_full_campaign(tmp_path)

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
    )

    assert result["status"] == "valid"
    assert result["benchmark_success"] is True
    assert result["successful_planner_arms"] == 14
    assert result["observed_episode_rows"] == 20_160
    assert result["unique_episode_identities"] == 20_160
    assert result["source_commits"] == [_SOURCE_SHA]
    assert result["blockers"] == []


def test_full_release_rejects_fallback_even_when_campaign_reports_success(tmp_path: Path) -> None:
    """A campaign's permissive core-success status cannot authorize publication."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][0]["summary"]["benchmark_availability"] = {"readiness_status": "fallback"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["benchmark_success"] is False
    assert result["forbidden_status_counts"]["fallback"] == 1
    assert any("fallback" in blocker for blocker in result["blockers"])


def test_full_release_rejects_episode_fallback_markers(tmp_path: Path) -> None:
    """Episode-level fallback markers cannot hide behind successful arm summaries."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / _PLANNER_KEYS[0] / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["fallback_triggered"] = True
    rows[1]["algorithm_metadata"] = {"planner_kinematics": {"execution_mode": "fallback"}}
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["forbidden_status_counts"]["true"] == 1
    assert result["forbidden_status_counts"]["fallback"] == 1
    assert any("fallback_triggered" in blocker for blocker in result["blockers"])
    assert any("planner_kinematics.execution_mode" in blocker for blocker in result["blockers"])


def test_full_release_rejects_duplicate_planner_aggregate_roster(tmp_path: Path) -> None:
    """Aggregate rows must cover the exact unique manifest roster."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["planner_rows"][0]["planner_key"] = _PLANNER_KEYS[1]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert any("planner aggregate rows do not match" in blocker for blocker in result["blockers"])


def test_full_release_requires_exact_campaign_source_sha(tmp_path: Path) -> None:
    """The campaign source SHA must be valid and equal to episode provenance."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"]["git_hash"] = "b" * 40
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert any("do not match campaign.git_hash" in blocker for blocker in result["blockers"])


def test_full_release_rejects_arbitrary_same_count_identity_product(tmp_path: Path) -> None:
    """Exact row count cannot replace the manifest-resolved scenario/seed product."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / _PLANNER_KEYS[0] / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["scenario_id"] = "unregistered_scenario"
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["observed_episode_rows"] == 20_160
    assert result["unique_episode_identities"] == 20_160
    assert result["missing_episode_identities"] == 1
    assert result["unexpected_episode_identities"] == 1
    assert any("exact manifest-resolved" in blocker for blocker in result["blockers"])


def test_full_release_rejects_duplicate_or_missing_episode_identity(tmp_path: Path) -> None:
    """A 20,160-row count is insufficient when logical episode coverage is duplicated."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / _PLANNER_KEYS[0] / "episodes.jsonl"
    lines = episode_path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-2]
    episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["unique_episode_identities"] == 20_159
    assert any("duplicate episode identity" in blocker for blocker in result["blockers"])
