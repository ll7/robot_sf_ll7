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
        issue_number=2716,
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


# --- Issue #2742 successor manifest tests ---

_ISSUE_2742_MANIFEST = Path(
    "configs/policy_search/topology_reselection_cross_slice_issue_2742.yaml"
)
_ISSUE_3463_MANIFEST = Path(
    "configs/policy_search/topology_reselection_cross_slice_issue_3463.yaml"
)


def test_issue_2742_manifest_loads_with_decision_rule() -> None:
    """The issue-2742 manifest should load and declare promote/stop/revise decision rules."""

    manifest = runner.load_manifest(_ISSUE_2742_MANIFEST)

    assert manifest["issue"] == 2742
    assert manifest["stage"] == "clearance_targeted"
    assert "promote_if" in manifest["decision_rule"]
    assert "stop_if" in manifest["decision_rule"]
    assert manifest["decision_rule"]["otherwise"] == "revise"
    roles = [row["role"] for row in manifest["slices"]]
    assert roles.count("hard") >= 3
    assert roles.count("negative_control") >= 1


def test_issue_2742_dry_run_uses_correct_issue_and_stage(tmp_path: Path) -> None:
    """Dry-run against the issue-2742 manifest should report issue 2742 and use its stage label."""

    exit_code = runner.main(
        [
            "--manifest",
            str(_ISSUE_2742_MANIFEST),
            "--dry-run",
            "--max-runs",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    report = json.loads((tmp_path / "topology_reselection_cross_slice_report.json").read_text())
    assert exit_code == 0
    assert report["issue"] == 2742
    assert report["classification"] == "dry_run"
    for cmd_entry in report["commands"]:
        assert "--stage" in cmd_entry["command"]
        stage_idx = cmd_entry["command"].index("--stage") + 1
        assert cmd_entry["command"][stage_idx] == "issue_2742_slice"


def test_issue_2742_manifest_derives_default_output_dir() -> None:
    """The manifest issue number should be reflected in the default output dir name."""

    manifest = runner.load_manifest(_ISSUE_2742_MANIFEST)
    dir_name = runner._manifest_output_dir_name(manifest)
    assert "issue_2742" in dir_name


def test_issue_2742_classifier_revises_when_hard_not_all_clear() -> None:
    """Classifier should revise when some hard slices clear but not all."""

    rows = [
        {
            "candidate_role": "progress_gated",
            "slice_id": "bottleneck_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 2.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "doorway_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 4,
            "topology_switch_count": 0,
            "success": False,
            "route_progress_m": 3.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "t_intersection_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 2.5,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "simple_negative_control",
            "slice_role": "negative_control",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 0,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 1.0,
        },
        {
            "candidate_role": "reuse_penalty",
            "slice_id": "bottleneck_transfer",
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


def test_issue_3463_manifest_uses_monotone_progress_gated_candidate() -> None:
    """The issue-3463 packet should route the progress-gated arm through the monotone candidate."""

    manifest = runner.load_manifest(_ISSUE_3463_MANIFEST)

    assert manifest["issue"] == 3463
    assert manifest["stage"] == "corrective_monotone_sensitivity"
    assert (
        manifest["candidates"]["progress_gated"]
        == "topology_guided_hybrid_rule_v0_progress_gated_reselection_monotone"
    )
    roles = [row["role"] for row in manifest["slices"]]
    assert roles.count("hard") >= 3
    assert roles.count("negative_control") >= 1


def test_issue_3463_threshold_materialization_preserves_monotone_accounting(
    tmp_path: Path,
) -> None:
    """Threshold-specific issue-3463 rows should keep the monotone progress-accounting toggle."""

    registry_path, candidate_name = runner.materialize_threshold_candidate_registry(
        source_registry=Path("docs/context/policy_search/candidate_registry.yaml"),
        base_candidate="topology_guided_hybrid_rule_v0_progress_gated_reselection_monotone",
        threshold_m=0.2,
        work_dir=tmp_path,
        issue_number=3463,
    )

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    generated_entry = registry["candidates"][candidate_name]
    generated_config = yaml.safe_load(
        Path(generated_entry["candidate_config_path"]).read_text(encoding="utf-8")
    )

    assert candidate_name.endswith("0p2")
    assert generated_config["params"]["primary_route_progress_gate_threshold_m"] == 0.2
    assert generated_config["params"]["primary_route_progress_gate_use_monotone_accounting"] is True


def test_issue_3463_dry_run_reports_issue_stage_and_monotone_candidate(tmp_path: Path) -> None:
    """Dry-run should emit the issue-3463 stage label and monotone progress-gated candidate."""

    exit_code = runner.main(
        [
            "--manifest",
            str(_ISSUE_3463_MANIFEST),
            "--dry-run",
            "--max-runs",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
    )

    report = json.loads((tmp_path / "topology_reselection_cross_slice_report.json").read_text())

    assert exit_code == 0
    assert report["issue"] == 3463
    assert report["classification"] == "dry_run"
    monotone_rows = [
        entry
        for entry in report["commands"]
        if "topology_guided_hybrid_rule_v0_progress_gated_reselection_monotone_threshold_0p05"
        in entry["command"]
    ]
    assert monotone_rows
    for entry in report["commands"]:
        assert "--stage" in entry["command"]
        stage_idx = entry["command"].index("--stage") + 1
        assert entry["command"][stage_idx] == "issue_3463_slice"


def test_issue_2742_classifier_promotes_only_when_all_hard_clear() -> None:
    """Promotion should require every hard slice to clear with no control switching."""

    rows = [
        {
            "candidate_role": "progress_gated",
            "slice_id": "bottleneck_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 2.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "doorway_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 4,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 3.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "t_intersection_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 2.5,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "simple_negative_control",
            "slice_role": "negative_control",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 0,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 1.0,
        },
        {
            "candidate_role": "reuse_penalty",
            "slice_id": "bottleneck_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 2,
            "success": False,
            "route_progress_m": 2.0,
        },
    ]

    classification, rationale = runner.classify_report(rows)

    assert classification == "promote"
    assert "Every hard slice" in rationale


def test_issue_2742_negative_control_switching_blocks_promote() -> None:
    """Negative-control switching should prevent promotion even when all hard slices clear."""

    rows = [
        {
            "candidate_role": "progress_gated",
            "slice_id": "bottleneck_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 2.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "doorway_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 4,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 3.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "t_intersection_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 2.5,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "simple_negative_control",
            "slice_role": "negative_control",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 1,
            "topology_switch_count": 1,
            "success": True,
            "route_progress_m": 1.0,
        },
        {
            "candidate_role": "reuse_penalty",
            "slice_id": "bottleneck_transfer",
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
    assert "switching" in rationale


def test_issue_2742_classifier_revises_when_hard_all_exhaust() -> None:
    """Hard-slice route progress without terminal improvement should not promote."""

    rows = [
        {
            "candidate_role": "progress_gated",
            "slice_id": "bottleneck_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": False,
            "route_progress_m": 2.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "doorway_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 4,
            "topology_switch_count": 0,
            "success": False,
            "route_progress_m": 3.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "t_intersection_transfer",
            "slice_role": "hard",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 3,
            "topology_switch_count": 0,
            "success": False,
            "route_progress_m": 2.5,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "simple_negative_control",
            "slice_role": "negative_control",
            "diagnostic_status": "diagnostic_complete",
            "topology_command_steps": 0,
            "topology_switch_count": 0,
            "success": True,
            "route_progress_m": 1.0,
        },
        {
            "candidate_role": "reuse_penalty",
            "slice_id": "bottleneck_transfer",
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


def test_blocker_summary_groups_unavailable_progress_gated_rows() -> None:
    """Issue-3463 blocked rows should name the failing slice and outcome."""
    rows = [
        {
            "candidate_role": "progress_gated",
            "slice_id": "doorway_transfer",
            "slice_role": "hard",
            "scenario_name": "classic_doorway_medium",
            "diagnostic_status": "not_available",
            "terminal_outcome": "obstacle_collision",
            "threshold_m": 0.05,
            "collision_rate": 1.0,
            "route_progress_m": -12.0,
        },
        {
            "candidate_role": "progress_gated",
            "slice_id": "doorway_transfer",
            "slice_role": "hard",
            "scenario_name": "classic_doorway_medium",
            "diagnostic_status": "not_available",
            "terminal_outcome": "obstacle_collision",
            "threshold_m": 0.1,
            "collision_rate": 1.0,
            "route_progress_m": -12.0,
        },
        {
            "candidate_role": "reuse_penalty",
            "slice_id": "doorway_transfer",
            "slice_role": "hard",
            "scenario_name": "classic_doorway_medium",
            "diagnostic_status": "not_available",
            "terminal_outcome": "obstacle_collision",
            "threshold_m": None,
            "collision_rate": 1.0,
            "route_progress_m": -12.0,
        },
    ]

    blockers = runner.summarize_blockers(rows)
    classification, rationale = runner.classify_report(rows)

    assert classification == "blocked"
    assert "doorway_transfer:obstacle_collision" in rationale
    assert len(blockers) == 1
    blocker = blockers[0]
    assert blocker["slice_id"] == "doorway_transfer"
    assert blocker["diagnostic_status_counts"] == {"not_available": 3}
    assert blocker["candidate_roles"] == ["progress_gated", "reuse_penalty"]
    assert blocker["thresholds_m"] == [0.05, 0.1]
    assert blocker["classification"] == "fail_closed_not_success_evidence"


def test_report_markdown_renders_fail_closed_blockers() -> None:
    """Markdown report should expose blocker triage before caveats."""
    report = {
        "issue": 3463,
        "claim_boundary": "diagnostic_only_not_benchmark_or_paper_evidence",
        "classification": "blocked",
        "classification_rationale": (
            "Progress-gated rows failed closed before producing diagnostic evidence: "
            "doorway_transfer:obstacle_collision."
        ),
        "rows": [
            {
                "slice_id": "doorway_transfer",
                "slice_role": "hard",
                "candidate_role": "progress_gated",
                "threshold_m": 0.05,
                "diagnostic_status": "not_available",
                "terminal_outcome": "obstacle_collision",
                "route_progress_m": -12.0,
                "topology_switch_count": 0,
                "deadlock_duration_steps": 0,
                "collision_rate": 1.0,
            }
        ],
        "blockers": [
            {
                "slice_id": "doorway_transfer",
                "slice_role": "hard",
                "scenario_name": "classic_doorway_medium",
                "terminal_outcome": "obstacle_collision",
                "diagnostic_status_counts": {"not_available": 1},
                "candidate_roles": ["progress_gated"],
                "thresholds_m": [0.05],
                "row_count": 1,
                "min_route_progress_m": -12.0,
                "max_route_progress_m": -12.0,
                "next_empirical_action": (
                    "repair or replace this slice before benchmark-facing promotion"
                ),
            }
        ],
    }

    markdown = runner.report_markdown(report)

    assert "## Fail-Closed Blockers" in markdown
    assert "doorway_transfer" in markdown
    assert "classic_doorway_medium" in markdown
    assert "repair or replace this slice" in markdown
