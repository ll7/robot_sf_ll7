"""Focused tests for the persisted-result benchmark showcase."""

from __future__ import annotations

import io
import json
import tarfile
from typing import TYPE_CHECKING, Any

import pytest

import scripts.tools.benchmark_showcase as showcase

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_campaign(
    root: Path, *, goal_eligible: bool = False, nonpositive_collision_metrics: bool = False
) -> None:
    """Write a tiny persisted campaign with one missing and one malformed row."""
    planners = [
        {"key": "goal", "enabled": True},
        {"key": "orca", "enabled": True},
    ]
    _write_json(
        root / "campaign_manifest.json",
        {
            "campaign_id": "showcase-fixture",
            "git_hash": "fixture-revision",
            "planners": planners,
            "kinematics_matrix": ["holonomic"],
            "seed_policy": {"resolved_seeds": [11, 22]},
            "scenario_matrix": "fixture-matrix",
            "scenario_matrix_hash": "fixture-matrix-sha",
        },
    )
    _write_json(
        root / "preflight" / "preview_scenarios.json",
        {
            "truncated": False,
            "scenario_count": 2,
            "scenarios": [
                {
                    "scenario_id": "crossing-a",
                    "seeds": [11, 22],
                    "metadata": {"archetype": "crossing"},
                },
                {
                    "scenario_id": "doorway-a",
                    "seeds": [11, 22],
                    "metadata": {"archetype": "doorway"},
                },
            ],
        },
    )
    runs = []
    for planner in ("goal", "orca"):
        run_dir = root / "runs" / f"{planner}__holonomic"
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for scenario_id, seed in (
            ("crossing-a", 11),
            ("crossing-a", 22),
            ("doorway-a", 11),
            ("doorway-a", 22),
        ):
            if planner == "orca" and scenario_id == "doorway-a" and seed == 22:
                continue
            collision = planner == "goal" and scenario_id == "crossing-a" and seed == 11
            episode = {
                "scenario_id": scenario_id,
                "seed": seed,
                "episode_id": f"{planner}-{scenario_id}-{seed}",
                "status": "failure" if collision else "success",
                "termination_reason": "collision" if collision else "success",
                "outcome": {
                    "route_complete": not collision,
                    "collision_event": collision,
                    "timeout_event": False,
                },
                "metrics": {
                    "success": not collision,
                    "collisions": 0 if nonpositive_collision_metrics else int(collision),
                    "total_collision_count": (
                        0 if nonpositive_collision_metrics else int(collision)
                    ),
                    "clearing_distance_min": 0.05 if collision else 0.8,
                    "near_misses": int(collision),
                    "force_exceed_events": 0,
                    "comfort_exposure": 0.1,
                    "time_to_goal_norm": 0.5,
                    "path_efficiency": 0.8,
                    "snqi": 0.2,
                },
            }
            if planner == "goal" and scenario_id == "crossing-a" and seed == 11:
                episode["final_robot_position"] = [1.0, 0.0]
                episode["replay_steps"] = [
                    {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
                    {"t": 1.0, "x": 1.0, "y": 0.0, "heading": 0.0},
                ]
            rows.append(episode)
        episodes_path = run_dir / "episodes.jsonl"
        contents = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        if planner == "goal":
            contents += "{malformed-json-line\n"
        episodes_path.write_text(contents, encoding="utf-8")
        runs.append(
            {
                "status": "failed" if planner == "orca" else "ok",
                "planner": {"key": planner, "kinematics": "holonomic", "algo": planner},
                "episodes_path": episodes_path.relative_to(root).as_posix(),
                "summary": {
                    "kinematics": "holonomic",
                    "written": len(rows),
                    "failed_jobs": 1 if planner == "orca" else 0,
                    "preflight": {
                        "status": "fallback" if planner == "goal" and not goal_eligible else "ok"
                    },
                    "benchmark_availability": {
                        "availability_status": "unavailable" if planner == "orca" else "available"
                    },
                },
            }
        )
    _write_json(
        root / "reports" / "campaign_summary.json",
        {"campaign": {"campaign_id": "showcase-fixture"}, "runs": runs},
    )


def test_showcase_cli_builds_machine_and_human_reports_for_tiny_fixture(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    campaign_root = tmp_path / "campaign"
    _write_campaign(campaign_root)

    def analyzer(_campaign_root: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "campaign_analysis.json").write_text("{}\n", encoding="utf-8")
        return (
            {
                "findings": ["fixture diagnostic is retained"],
                "campaign_integrity": {
                    "status": "not_evaluable",
                    "claim_boundary": "fixture only",
                },
                "credibility_scorecard": {"status": "warning"},
            },
            {"status": "passed", "summary_path": "analysis/campaign_analysis.json"},
        )

    monkeypatch.setattr(showcase, "_run_existing_analyzer", analyzer)
    output_dir = tmp_path / "output"
    exit_code = showcase.main(
        [
            "--campaign-root",
            str(campaign_root),
            "--out-dir",
            str(output_dir),
            "--top-k",
            "2",
            "--render-limit",
            "1",
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    summary = json.loads((output_dir / "showcase_summary.json").read_text())

    assert summary["accounting"]["expected_identity_count"] == 8
    assert summary["accounting"]["present_episode_rows"] == 7
    assert summary["accounting"]["missing_identity_count"] == 1
    assert summary["accounting"]["malformed_line_count"] == 1
    run_status = {run["planner_key"]: run for run in summary["accounting"]["run_statuses"]}
    assert run_status["goal"]["fallback_or_degraded"] is True
    assert run_status["goal"]["benchmark_eligible"] is False
    assert run_status["orca"]["unavailable"] is True
    assert run_status["orca"]["failed"] is True
    assert run_status["orca"]["benchmark_eligible"] is False
    assert all(row["benchmark_eligible_episodes"] == 0 for row in summary["planner_family_matrix"])
    assert summary["camera_ready_analyzer"]["findings"] == ["fixture diagnostic is retained"]
    assert summary["replay_status_counts"]["verified"] == 1
    verified = [case for case in summary["cases"] if case["replay"]["status"] == "verified"]
    assert len(verified) == 1
    assert verified[0]["replay"]["renderer_called"] is True
    assert all(
        case["replay"]["renderer_called"] is False
        for case in summary["cases"]
        if case["replay"]["status"] == "unavailable"
    )
    report = (output_dir / "report.md").read_text()
    assert "fixture diagnostic is retained" in report
    assert "missing identities: 1" in report
    assert "fallback_or_degraded" in report
    assert "run_status=failed" in report
    metric_group = next(
        group for group in summary["critical_case_groups"] if group["name"] == "minimum_clearance"
    )
    selected = {
        summary_case["case_id"]: summary_case["planner_key"] for summary_case in summary["cases"]
    }
    assert {selected[case_id] for case_id in metric_group["selected_case_ids"]} == {"goal", "orca"}


def test_canonical_collision_event_is_reported_when_both_count_metrics_are_zero(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    campaign_root = tmp_path / "campaign"
    _write_campaign(campaign_root, goal_eligible=True, nonpositive_collision_metrics=True)

    def analyzer(_campaign_root: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "campaign_analysis.json").write_text("{}\n", encoding="utf-8")
        return (
            {
                "findings": ["collision metric contradiction fixture finding"],
                "campaign_integrity": {"status": "valid", "claim_boundary": "campaign only"},
            },
            {"status": "passed", "summary_path": "analysis/campaign_analysis.json"},
        )

    monkeypatch.setattr(showcase, "_run_existing_analyzer", analyzer)
    output_dir = tmp_path / "output"
    exit_code = showcase.main(
        [
            "--campaign-root",
            str(campaign_root),
            "--out-dir",
            str(output_dir),
            "--top-k",
            "2",
            "--render-limit",
            "0",
        ]
    )
    assert exit_code == 0
    capsys.readouterr()
    summary = json.loads((output_dir / "showcase_summary.json").read_text())
    consistency = summary["collision_event_metric_consistency"]
    assert consistency["canonical_collision_event_count"] == 1
    assert consistency["collision_event_and_collision_termination_count"] == 1
    assert consistency["both_count_metrics_nonpositive_count"] == 1

    crossing = next(
        row
        for row in summary["planner_family_matrix"]
        if row["planner_key"] == "goal" and row["scenario_family"] == "crossing"
    )
    assert crossing["outcomes"]["collision_event"]["count"] == 1
    assert crossing["metric_summaries"]["collisions_metric"]["mean"] == 0
    assert crossing["metric_summaries"]["total_collision_count"]["mean"] == 0
    collision_group = next(
        group for group in summary["critical_case_groups"] if group["name"] == "collision_event"
    )
    assert collision_group["candidate_count"] == 1
    collision_case = next(case for case in summary["cases"] if case["outcome"]["collision_event"])
    assert collision_case["metrics"]["collisions_metric"] == 0
    assert collision_case["metrics"]["total_collision_count"] == 0

    report = (output_dir / "report.md").read_text()
    assert "Camera-ready analyzer campaign-integrity status: `valid`" in report
    assert "Analyzer campaign-integrity claim boundary: `campaign only`" in report
    assert "Total collision count" in report
    assert "**Data-quality warning:** 1 of the collision-terminated rows" in report
    assert (
        "both `metrics.collisions` and `metrics.total_collision_count` at or below zero" in report
    )
    assert "collision metric contradiction fixture finding" in report


def test_missing_replay_steps_never_invokes_renderer(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    def forbidden_renderer(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("renderer must not run without recorded replay_steps")

    monkeypatch.setattr(showcase.subprocess, "run", forbidden_renderer)
    replay = showcase._render_direct_replay(
        {"case_id": "case-fixture", "source": {"episode_file": "episodes.jsonl"}},
        {"raw": {}, "case_id": "case-fixture"},
        tmp_path,
        tmp_path / "output",
        render_limit_reached=False,
    )

    assert replay["status"] == "unavailable"
    assert replay["renderer_called"] is False
    assert "resimulation" in replay["reason"]


def test_direct_replay_uses_renderer_sidecar_and_verifies_artifact_hashes(
    tmp_path: Path,
) -> None:
    campaign_root = tmp_path / "campaign"
    episode_file = campaign_root / "runs" / "goal" / "episodes.jsonl"
    episode_file.parent.mkdir(parents=True)
    raw = {
        "episode_id": "ep-1",
        "scenario_id": "scenario-1",
        "seed": 11,
        "final_robot_position": [1.0, 0.0],
        "replay_steps": [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
            {"t": 1.0, "x": 1.0, "y": 0.0, "heading": 0.0},
        ],
    }
    episode_file.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    row = {
        "case_id": "case-1",
        "episode_id": "ep-1",
        "scenario_id": "scenario-1",
        "seed": 11,
        "raw": raw,
    }
    case = {
        "case_id": "case-1",
        "episode_id": "ep-1",
        "scenario_id": "scenario-1",
        "seed": 11,
        "source": {
            "episode_file": "runs/goal/episodes.jsonl",
            "episode_file_sha256": showcase.sha256_file(episode_file),
        },
    }
    replay = showcase._render_direct_replay(
        case,
        row,
        campaign_root,
        tmp_path / "output",
        render_limit_reached=False,
    )

    assert replay["status"] == "verified"
    assert replay["artifact_status"] == "generated"
    assert replay["renderer_called"] is True
    assert {artifact["type"] for artifact in replay["artifacts"]} == {
        "still",
        "filmstrip",
        "trajectory",
    }


def test_case_selection_round_robins_families_and_planners() -> None:
    candidates = [
        {
            "planner_key": planner,
            "scenario_family": family,
            "scenario_id": scenario,
            "seed": 1,
            "episode_id": scenario,
            "metrics": {"minimum_clearance_m": clearance},
        }
        for planner in ("goal", "orca")
        for family, scenario, clearance in (
            ("a-family", "a-near", 0.1),
            ("a-family", "a-far", 0.2),
            ("b-family", "b-near", 0.3),
        )
    ]
    selected = showcase._select_diverse(
        candidates,
        {"metric": "clearing_distance_min", "direction": "ascending"},
        top_k=4,
    )

    assert [(row["planner_key"], row["scenario_family"]) for row in selected] == [
        ("goal", "a-family"),
        ("orca", "a-family"),
        ("goal", "b-family"),
        ("orca", "b-family"),
    ]


def test_archive_extractor_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"oops"))

    with pytest.raises(showcase.ShowcaseError, match="unsafe archive member path"):
        showcase._safe_extract_archive(archive_path, tmp_path / "extract")
    assert not (tmp_path / "escape.txt").exists()


def test_fallback_inventory_is_not_described_as_complete_denominator() -> None:
    claim = showcase._denominator_claim_text(
        {"denominator_status": "observed_scenario_breakdown_only"}
    )

    assert "may omit scenarios with no persisted outcomes" in claim
    assert "not a complete campaign denominator" in claim


def test_campaign_input_rejects_output_nested_in_source(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()

    with pytest.raises(showcase.ShowcaseError, match="outside the supplied campaign root"):
        showcase.run_showcase(
            campaign_root=campaign_root,
            out_dir=campaign_root / "showcase",
        )

    assert not (campaign_root / "showcase").exists()
