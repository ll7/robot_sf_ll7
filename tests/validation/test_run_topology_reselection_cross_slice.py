"""Tests for issue-2716 topology reselection cross-slice runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from scripts.validation import run_topology_reselection_cross_slice as runner


def test_checked_in_manifest_names_required_slice_roles() -> None:
    """The issue manifest should name hard slices and a negative control."""

    manifest = runner.load_manifest(
        Path("configs/policy_search/topology_reselection_cross_slice_issue_2716.yaml")
    )

    roles = [row["role"] for row in manifest["slices"]]
    assert roles.count("hard") >= 3
    assert roles.count("negative_control") >= 1
    assert manifest["candidates"]["baseline"] == "topology_guided_hybrid_rule_v0"
    assert manifest["candidates"]["reuse_penalty"] == "topology_guided_hybrid_rule_v0_reuse_penalty"
    assert (
        manifest["candidates"]["progress_gated"]
        == "topology_guided_hybrid_rule_v0_progress_gated_reselection"
    )


def test_build_rows_expands_candidates_and_thresholds(tmp_path: Path) -> None:
    """Rows should include baseline/reuse once and progress-gated per threshold."""

    manifest = runner.load_manifest(
        Path("configs/policy_search/topology_reselection_cross_slice_issue_2716.yaml")
    )

    rows = runner.build_rows(manifest, tmp_path / "runs")
    rows_per_slice = 2 + len(manifest["progress_gate_thresholds_m"])

    assert len(rows) == len(manifest["slices"]) * rows_per_slice
    assert rows[0].candidate_role == "baseline"
    assert rows[1].candidate_role == "reuse_penalty"
    assert rows[2].candidate_role == "progress_gated"
    assert rows[2].threshold_m == manifest["progress_gate_thresholds_m"][0]


def test_materialize_threshold_candidate_registry_overrides_threshold(tmp_path: Path) -> None:
    """Threshold-specific candidates should not mutate the checked-in config."""

    registry_path, candidate_name = runner.materialize_threshold_candidate_registry(
        source_registry=Path("docs/context/policy_search/candidate_registry.yaml"),
        base_candidate="topology_guided_hybrid_rule_v0_progress_gated_reselection",
        threshold_m=0.2,
        work_dir=tmp_path,
    )

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    generated_entry = registry["candidates"][candidate_name]
    generated_config = yaml.safe_load(
        Path(generated_entry["candidate_config_path"]).read_text(encoding="utf-8")
    )
    assert candidate_name.endswith("0p2")
    assert generated_config["params"]["primary_route_progress_gate_threshold_m"] == 0.2


def test_command_for_row_uses_slice_specific_funnel(tmp_path: Path) -> None:
    """Generated commands should select scenarios from each row's source surface."""

    manifest = runner.load_manifest(
        Path("configs/policy_search/topology_reselection_cross_slice_issue_2716.yaml")
    )
    row = runner.build_rows(manifest, tmp_path / "runs")[0]

    command = runner.command_for_row(
        row=row,
        manifest=manifest,
        candidate_registry=Path("docs/context/policy_search/candidate_registry.yaml"),
        temp_root=tmp_path / "tmp",
    )

    funnel_path = Path(command[command.index("--funnel-config") + 1])
    funnel = yaml.safe_load(funnel_path.read_text(encoding="utf-8"))
    assert command[command.index("--stage") + 1] == "issue_2716_slice"
    assert funnel["stages"]["issue_2716_slice"]["scenario_matrix"] == row.source_surface
    assert row.scenario_name in command


def test_dry_run_writes_command_report(tmp_path: Path) -> None:
    """Dry-run mode should write JSON and Markdown reports without executing diagnostics."""

    exit_code = runner.main(["--dry-run", "--max-runs", "3", "--output-dir", str(tmp_path)])

    report = json.loads((tmp_path / "topology_reselection_cross_slice_report.json").read_text())
    markdown = (tmp_path / "topology_reselection_cross_slice_report.md").read_text()
    assert exit_code == 0
    assert report["classification"] == "dry_run"
    assert len(report["commands"]) == 3
    assert "Decision Table" in markdown


def test_main_classifies_patched_successful_rows(tmp_path: Path, monkeypatch) -> None:
    """Patched execution should classify complete progressing rows as promote."""

    def fake_run_row(command: list[str]) -> dict:
        role = "progress_gated" if "progress_gated" in " ".join(command) else "baseline"
        return {
            "diagnostic_status": "diagnostic_complete",
            "summary": {
                "corrective_behavior": {
                    "terminal_outcome": {
                        "outcome": "success",
                        "success": True,
                        "step": 17,
                        "is_pedestrian_collision": False,
                        "is_obstacle_collision": False,
                        "is_robot_collision": False,
                    },
                    "max_route_progress_delta_m": 1.5,
                    "hypothesis_switch_count": 0,
                    "topology_command_steps": 3 if role == "progress_gated" else 2,
                    "non_primary_topology_command_steps": 1,
                },
                "topology_reuse_penalty": {
                    "applied_steps": 1,
                    "progress_gate_satisfied_steps": 1,
                    "progress_suppressed_steps": 1,
                },
            },
        }

    monkeypatch.setattr(runner, "run_row", fake_run_row)

    exit_code = runner.main(["--max-runs", "5", "--output-dir", str(tmp_path)])

    report = json.loads((tmp_path / "topology_reselection_cross_slice_report.json").read_text())
    assert exit_code == 0
    assert report["classification"] == "promote"
    assert report["dry_run"] is False
    assert report["rows"][0]["route_progress_m"] == 1.5


def test_classifier_revises_when_hard_rows_all_exhaust_horizon() -> None:
    """Hard-slice route progress without terminal improvement should not promote."""

    rows = [
        {
            "candidate_role": "progress_gated",
            "slice_id": "classic_bottleneck_medium",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": False,
            "route_progress_m": 2.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "empty_map_8_directions_east",
            "slice_role": "negative_control",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 0,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 1.0,
        },
        {
            "candidate_role": "reuse_penalty",
            "slice_id": "classic_bottleneck_medium",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 2,
            "success": False,
            "route_progress_m": 2.0,
        },
    ]

    classification, rationale = runner.classify_report(rows)

    assert classification == "revise"
    assert "horizon_exhausted" in rationale


def test_classifier_revises_when_only_one_hard_slice_clears() -> None:
    """Promotion should require clearance across the hard slices, not one lucky clearance."""

    rows = [
        {
            "candidate_role": "progress_gated",
            "slice_id": "classic_bottleneck_medium",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 2.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "classic_doorway_medium",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 4,
            "topology_switch_count": 0,
            "success": False,
            "route_progress_m": 3.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "empty_map_8_directions_east",
            "slice_role": "negative_control",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 0,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 1.0,
        },
        {
            "candidate_role": "reuse_penalty",
            "slice_id": "classic_bottleneck_medium",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 2,
            "success": False,
            "route_progress_m": 2.0,
        },
    ]

    classification, rationale = runner.classify_report(rows)

    assert classification == "revise"
    assert "did not clear" in rationale


def test_run_row_handles_child_command_without_json_stdout(monkeypatch) -> None:
    """A failed child command without JSON stdout should become a blocked row, not crash."""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=7, stdout="", stderr="boom")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    result = runner.run_row(["false"])

    assert result["returncode"] == 7
    assert result["diagnostic_status"] == "command_failed"
    assert result["trace"] is None
    assert "boom" in result["stderr_excerpt"]
