"""Tests for hybrid evidence matrix validation."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

from robot_sf.benchmark.hybrid_evidence_matrix import (
    build_hybrid_prerequisite_matrix,
    load_hybrid_evidence_input,
    summarize_issue_1489_integration_status,
    validate_hybrid_evidence_file,
    validate_hybrid_evidence_rows,
)
from scripts.validation.validate_hybrid_evidence_matrix import main as validate_cli_main

FIXTURE_ROOT = Path("tests/fixtures/hybrid_evidence_matrix/v1")


def test_valid_rows_validate_and_preserve_launch_packet_rows() -> None:
    """Valid execution and launch-packet rows should both validate deterministically."""
    report = validate_hybrid_evidence_file(FIXTURE_ROOT / "valid_rows.yaml")

    assert report["status"] == "valid"
    assert report["provenance_validation"] == "format_only"
    assert report["row_count"] == 2
    assert report["valid_row_count"] == 2
    assert report["rows"][0]["component"] == "learned_risk_model_v1"
    assert report["rows"][0]["synthesis_candidate"] is True
    assert report["rows"][0]["synthesis_eligible"] is True
    assert report["rows"][0]["warnings"] == []
    assert report["rows"][1]["component"] == "oracle_imitation_v1"
    assert report["rows"][1]["status"] == "valid"
    assert report["rows"][1]["synthesis_candidate"] is False
    assert report["rows"][1]["synthesis_eligible"] is False


def test_invalid_rows_report_actionable_field_errors() -> None:
    """Invalid fixtures should point at the broken field rather than failing generically."""
    cases = [
        ("invalid_guard_veto_mismatch.yaml", "intervention_fallback_rates.guard_veto_rate"),
        ("invalid_output_provenance.yaml", "commit_artifact"),
        ("invalid_nullability.yaml", "guard_authority.veto_rate"),
        ("invalid_enum.yaml", "evaluation_slice"),
    ]

    for fixture_name, expected_field in cases:
        report = validate_hybrid_evidence_file(FIXTURE_ROOT / fixture_name)

        assert report["status"] == "invalid"
        fields = {error["field"] for error in report["rows"][0]["errors"]}
        assert expected_field in fields


def test_reference_validation_allows_non_root_output_directory_names() -> None:
    """Only the repository-root output directory should be rejected as worktree-local."""
    _input_format, rows = load_hybrid_evidence_input(FIXTURE_ROOT / "valid_rows.yaml")
    mutated_rows = copy.deepcopy(rows[:1])
    mutated_rows[0]["commit_artifact"] = (
        f"deadbee, {FIXTURE_ROOT / 'nested_output_name' / 'summary.json'}"
    )

    report = validate_hybrid_evidence_rows(mutated_rows)

    assert report["status"] == "valid"


def test_validator_warns_when_guard_never_exercises_an_active_component() -> None:
    """Zero guard veto with active learned changes should stay valid but surface a caveat."""
    _input_format, rows = load_hybrid_evidence_input(FIXTURE_ROOT / "valid_rows.yaml")
    mutated_rows = copy.deepcopy(rows[:1])
    mutated_rows[0]["guard_authority"]["veto_rate"] = 0.0
    mutated_rows[0]["intervention_fallback_rates"]["guard_veto_rate"] = 0.0

    report = validate_hybrid_evidence_rows(mutated_rows)

    assert report["status"] == "valid"
    warnings = report["rows"][0]["warnings"]
    assert warnings
    assert warnings[0]["field"] == "guard_authority.veto_rate"


def test_semantic_validation_branches_reject_invalid_rows() -> None:
    """Each semantic branch fixture should produce an error targeting the expected field."""
    cases = [
        (
            "invalid_fallback_rate_in_success_tier.yaml",
            "intervention_fallback_rates.fallback_rate",
        ),
        (
            "invalid_degraded_rate_in_success_tier.yaml",
            "intervention_fallback_rates.degraded_rate",
        ),
        (
            "invalid_fallback_tier_no_fallback_rate.yaml",
            "intervention_fallback_rates.fallback_rate",
        ),
        (
            "invalid_degraded_tier_no_degraded_rate.yaml",
            "intervention_fallback_rates.degraded_rate",
        ),
        (
            "invalid_not_run_wrong_tier.yaml",
            "evidence_tier",
        ),
        (
            "invalid_executed_null_active_rate.yaml",
            "learned_component_contribution.active_rate",
        ),
        (
            "invalid_slice_tier_mismatch.yaml",
            "evaluation_slice",
        ),
    ]

    for fixture_name, expected_field in cases:
        report = validate_hybrid_evidence_file(FIXTURE_ROOT / fixture_name)

        assert report["status"] == "invalid", f"{fixture_name} should be invalid"
        error_fields = {error["field"] for error in report["rows"][0]["errors"]}
        assert expected_field in error_fields, (
            f"{fixture_name}: expected {expected_field!r} in {sorted(error_fields)}"
        )


def test_cli_emits_json_for_invalid_input(capsys) -> None:
    """The CLI should print a structured JSON report and return the invalid-data exit code."""
    exit_code = validate_cli_main(
        ["--input", str(FIXTURE_ROOT / "invalid_guard_veto_mismatch.yaml")]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid"
    assert payload["provenance_validation"] == "format_only"
    assert payload["row_count"] == 1
    assert payload["rows"][0]["errors"][0]["field"]


def test_format_only_validation_keeps_unknown_git_sha_lightweight(tmp_path: Path) -> None:
    """Default validation must remain format-only so unknown SHAs do not fail local schema checks."""
    repo_root, _known_sha = _create_temp_git_repo(tmp_path)
    row = _make_repo_backed_row()
    row["commit_artifact"] = "deadbeef, docs/context/proof.json"

    report = validate_hybrid_evidence_rows([row], repo_root=repo_root)

    assert report["status"] == "valid"
    assert report["provenance_validation"] == "format_only"


def test_git_history_validation_rejects_unknown_git_sha(tmp_path: Path) -> None:
    """Strict provenance mode must fail closed when a commit SHA is not in repository history."""
    repo_root, _known_sha = _create_temp_git_repo(tmp_path)
    row = _make_repo_backed_row()
    row["commit_artifact"] = "deadbeef, docs/context/proof.json"

    report = validate_hybrid_evidence_rows([row], repo_root=repo_root, check_git_history=True)

    assert report["status"] == "invalid"
    assert report["provenance_validation"] == "git_history"
    messages = {
        error["message"]
        for error in report["rows"][0]["errors"]
        if error["field"] == "commit_artifact"
    }
    assert "references unknown git commit SHA in repository history: 'deadbeef'" in messages


def test_git_history_validation_accepts_known_sha_launch_packet_and_remote_pointer(
    tmp_path: Path,
) -> None:
    """Strict provenance must still accept known SHAs for launch packets and durable remote pointers."""
    repo_root, known_sha = _create_temp_git_repo(tmp_path)
    executed_row = _make_repo_backed_row()
    executed_row["commit_artifact"] = f"{known_sha}, wandb://robot-sf/hybrid-evidence/run-1472"

    launch_packet_row = _make_repo_backed_row()
    launch_packet_row["component"] = "oracle_imitation_v1"
    launch_packet_row["source_issue"] = "#1470"
    launch_packet_row["commit_artifact"] = f"{known_sha}, docs/context/launch_packet.md"
    launch_packet_row["evaluation_slice"] = "not_run"
    launch_packet_row["guard_authority"]["active"] = False
    launch_packet_row["guard_authority"]["veto_rate"] = None
    launch_packet_row["learned_component_contribution"]["contribution_type"] = (
        "warm_start_initialisation"
    )
    launch_packet_row["learned_component_contribution"]["bound"] = "oracle dataset only"
    launch_packet_row["learned_component_contribution"]["active_rate"] = None
    launch_packet_row["intervention_fallback_rates"] = {
        "guard_veto_rate": None,
        "fallback_rate": None,
        "degraded_rate": None,
    }
    launch_packet_row["outcomes"] = {
        "success_rate": None,
        "collision_rate": None,
        "near_miss_rate": None,
        "low_progress_rate": None,
        "timeout_rate": None,
    }
    launch_packet_row["evidence_tier"] = "launch_packet"
    launch_packet_row["verdict"] = "pending"

    report = validate_hybrid_evidence_rows(
        [executed_row, launch_packet_row],
        repo_root=repo_root,
        check_git_history=True,
    )

    assert report["status"] == "valid"
    assert report["provenance_validation"] == "git_history"
    assert [row["status"] for row in report["rows"]] == ["valid", "valid"]


def test_cli_check_git_history_emits_json_for_unknown_sha(tmp_path: Path, capsys) -> None:
    """CLI strict mode must surface unknown git SHAs in JSON so proof runs fail closed predictably."""
    repo_root, _known_sha = _create_temp_git_repo(tmp_path)
    row = _make_repo_backed_row()
    row["commit_artifact"] = "deadbeef, docs/context/proof.json"
    input_path = tmp_path / "unknown_sha.json"
    input_path.write_text(json.dumps([row]), encoding="utf-8")

    exit_code = validate_cli_main(
        [
            "--input",
            str(input_path),
            "--repo-root",
            str(repo_root),
            "--check-git-history",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid"
    assert payload["provenance_validation"] == "git_history"
    assert payload["rows"][0]["errors"][0]["field"] == "commit_artifact"


def test_issue_1489_integration_status_keeps_parent_blocked_for_incomplete_lanes() -> None:
    """Prerequisite matrix reports the #1489 blocker and next empirical action."""
    _input_format, rows = load_hybrid_evidence_input(FIXTURE_ROOT / "valid_rows.yaml")

    report = build_hybrid_prerequisite_matrix(
        rows,
        expected_components=[
            "learned_risk_model_v1",
            "oracle_imitation_v1",
            "shielded_ppo_repair_v1",
        ],
    )

    assert report["gate"] == "blocked"
    assert report["state_counts"] == {
        "missing": 1,
        "blocked": 1,
        "ready": 0,
        "complete": 1,
    }
    assert report["integration_status"] == {
        "issue": "#1489",
        "status": "blocked",
        "claim_boundary": "not benchmark evidence; prerequisite/status integration only",
        "complete_count": 1,
        "required_complete_count": 2,
        "remaining_complete_count": 1,
        "blockers": [
            "1 more complete lane(s) required",
            "1 blocked lane(s)",
            "1 missing expected lane(s)",
        ],
        "next_empirical_action": (
            "Keep #1489 blocked; finish component campaign evidence before synthesis."
        ),
    }


def test_issue_1489_cli_prerequisite_matrix_includes_integration_status(capsys) -> None:
    """CLI prerequisite matrix exposes the same compact integration report."""
    exit_code = validate_cli_main(
        [
            "--input",
            str(FIXTURE_ROOT / "valid_rows.yaml"),
            "--prerequisite-matrix",
            "--expected-component",
            "learned_risk_model_v1",
            "--expected-component",
            "oracle_imitation_v1",
            "--expected-component",
            "shielded_ppo_repair_v1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["integration_status"]["status"] == "blocked"
    assert payload["integration_status"]["blockers"] == [
        "1 more complete lane(s) required",
        "1 blocked lane(s)",
        "1 missing expected lane(s)",
    ]


def test_issue_1489_integration_status_distinguishes_ready_non_synthesis_lanes() -> None:
    """Rows with runtime evidence but non-comparable tiers remain integration work."""
    report = summarize_issue_1489_integration_status(
        state_counts={"missing": 0, "blocked": 0, "ready": 2, "complete": 0},
        prerequisite_count=2,
        rows_valid=True,
        invalid_row_count=0,
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == [
        "2 more complete lane(s) required",
        "2 ready but not synthesis-complete lane(s)",
    ]
    assert report["next_empirical_action"] == (
        "Integrate ready lanes into comparable durable component rows before synthesis."
    )


def test_issue_1489_integration_status_opens_only_with_complete_lanes() -> None:
    """The integration summary opens when enough durable complete lanes exist."""
    report = summarize_issue_1489_integration_status(
        state_counts={"missing": 0, "blocked": 0, "ready": 1, "complete": 2},
        prerequisite_count=2,
        rows_valid=True,
        invalid_row_count=0,
    )

    assert report["status"] == "ready_for_synthesis"
    assert report["blockers"] == []
    assert report["remaining_complete_count"] == 0
    assert report["next_empirical_action"] == (
        "Run the conservative #1489 synthesis over the complete durable lanes."
    )


def _create_temp_git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init")
    _git(repo_root, "config", "user.name", "Robot SF Tests")
    _git(repo_root, "config", "user.email", "robot-sf-tests@example.com")
    (repo_root / "docs" / "context").mkdir(parents=True)
    (repo_root / "docs" / "context" / "proof.json").write_text(
        '{"status": "ok"}\n', encoding="utf-8"
    )
    (repo_root / "docs" / "context" / "launch_packet.md").write_text(
        "# Launch packet\n",
        encoding="utf-8",
    )
    _git(repo_root, "add", "docs/context/proof.json", "docs/context/launch_packet.md")
    _git(repo_root, "commit", "-m", "test fixture")
    known_sha = _git(repo_root, "rev-parse", "--short=7", "HEAD")
    return repo_root, known_sha


def _make_repo_backed_row() -> dict[str, object]:
    return {
        "component": "learned_risk_model_v1",
        "source_issue": "#1472",
        "commit_artifact": "deadbeef, docs/context/proof.json",
        "evaluation_slice": "stress_slice",
        "guard_authority": {
            "mechanism": "risk_guarded_ppo",
            "active": True,
            "veto_rate": 0.12,
        },
        "learned_component_contribution": {
            "contribution_type": "auxiliary_cost",
            "bound": "cost_weight in [0,1]",
            "active_rate": 0.85,
        },
        "intervention_fallback_rates": {
            "guard_veto_rate": 0.12,
            "fallback_rate": 0.0,
            "degraded_rate": None,
        },
        "outcomes": {
            "success_rate": 0.72,
            "collision_rate": 0.04,
            "near_miss_rate": 0.08,
            "low_progress_rate": 0.05,
            "timeout_rate": 0.11,
        },
        "evidence_tier": "stress",
        "verdict": "continue",
        "scenario_manifest": "docs/context/proof.json",
    }


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
