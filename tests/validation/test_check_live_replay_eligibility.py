"""Tests for live-replay promotion eligibility classification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.validation import check_live_replay_eligibility as checker

if TYPE_CHECKING:
    from pathlib import Path


def _fixture(tmp_path: Path) -> Path:
    path = tmp_path / "fixtures" / "trace.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"trace": true}\n', encoding="utf-8")
    return path


def _eligible_record(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "fixture_path": str(_fixture(tmp_path)),
        "scenario_id": "issue_2756_occluded_emergence",
        "policy_candidate": "hybrid_rule_v0_minimal",
        "perturbation_config": "configs/perturbations/occluded_emergence.yaml",
        "output_report_path": "output/live_replay/issue_2789/report.json",
        "live_metrics_supported": True,
        "claim_boundary": "diagnostic-only; live replay promotion required; not benchmark evidence",
        "evidence_class": "diagnostic_only",
        "live_replay_candidate": True,
    }
    record.update(overrides)
    return record


def test_eligible_record_reports_eligible(tmp_path: Path) -> None:
    """A complete promotion candidate should be eligible, not proof."""
    report = checker.evaluate_records([_eligible_record(tmp_path)], base_dir=tmp_path)
    payload = report["live_replay_eligibility"]

    assert payload["schema_version"] == "live_replay_eligibility.v1"
    assert payload["counts"] == {"eligible": 1}
    assert payload["classifications"][0]["status"] == "eligible"
    assert "not benchmark evidence" in payload["claim_boundary"]


def test_missing_fixture_reports_missing_fields(tmp_path: Path) -> None:
    """A missing fixture path should fail closed as missing prerequisites."""
    record = _eligible_record(tmp_path, fixture_path="missing/trace.json")

    row = checker.evaluate_records([record], base_dir=tmp_path)["live_replay_eligibility"][
        "classifications"
    ][0]

    assert row["status"] == "missing-fields"
    assert "fixture_path" in row["missing_prerequisites"]


def test_missing_policy_reports_missing_fields(tmp_path: Path) -> None:
    """Policy candidate or planner id is required."""
    record = _eligible_record(tmp_path)
    record.pop("policy_candidate")

    row = checker.evaluate_records([record], base_dir=tmp_path)["live_replay_eligibility"][
        "classifications"
    ][0]

    assert row["status"] == "missing-fields"
    assert "policy_candidate" in row["missing_prerequisites"]


@pytest.mark.parametrize(
    ("field", "expected_missing"),
    [
        ("scenario_id", "scenario"),
        ("perturbation_config", "perturbation_config"),
        ("output_report_path", "output_report_path"),
    ],
)
def test_required_text_prerequisites_report_missing_fields(
    tmp_path: Path,
    field: str,
    expected_missing: str,
) -> None:
    """Each required text prerequisite should have direct missing-field coverage."""
    record = _eligible_record(tmp_path)
    record.pop(field)

    row = checker.evaluate_records([record], base_dir=tmp_path)["live_replay_eligibility"][
        "classifications"
    ][0]

    assert row["status"] == "missing-fields"
    assert expected_missing in row["missing_prerequisites"]


def test_missing_metrics_reports_missing_fields(tmp_path: Path) -> None:
    """Live metric support must be explicit true."""
    record = _eligible_record(tmp_path, live_metrics_supported=False)

    row = checker.evaluate_records([record], base_dir=tmp_path)["live_replay_eligibility"][
        "classifications"
    ][0]

    assert row["status"] == "missing-fields"
    assert "live_metrics_supported" in row["missing_prerequisites"]


def test_blocked_claim_boundary_case(tmp_path: Path) -> None:
    """Missing or unsafe claim boundary should block promotion eligibility."""
    record = _eligible_record(tmp_path, claim_boundary="ready for paper table")

    row = checker.evaluate_records([record], base_dir=tmp_path)["live_replay_eligibility"][
        "classifications"
    ][0]

    assert row["status"] == "blocked"
    assert row["blocked_reasons"] == ["unsafe_claim_boundary"]


def test_claim_boundary_accepts_hyphen_and_underscore_variants(tmp_path: Path) -> None:
    """Boundary matching should normalize common separator variants."""
    record = _eligible_record(
        tmp_path,
        claim_boundary="diagnostic_only; live_replay promotion required; fail_closed",
    )

    row = checker.evaluate_records([record], base_dir=tmp_path)["live_replay_eligibility"][
        "classifications"
    ][0]

    assert row["status"] == "eligible"


def test_missing_claim_boundary_blocks_promotion(tmp_path: Path) -> None:
    """Absent claim boundary should fail closed as blocked."""
    record = _eligible_record(tmp_path)
    record.pop("claim_boundary")

    row = checker.evaluate_records([record], base_dir=tmp_path)["live_replay_eligibility"][
        "classifications"
    ][0]

    assert row["status"] == "blocked"
    assert row["blocked_reasons"] == ["missing_claim_boundary"]


def test_diagnostic_only_without_candidate_stays_diagnostic_only(tmp_path: Path) -> None:
    """Diagnostic records that do not request promotion should stay diagnostic-only."""
    record = _eligible_record(tmp_path, live_replay_candidate=False)
    record.pop("promotion_path", None)

    row = checker.evaluate_records([record], base_dir=tmp_path)["live_replay_eligibility"][
        "classifications"
    ][0]

    assert row["status"] == "diagnostic-only"
    assert row["diagnostic_reasons"] == [
        "record_declares_diagnostic_only_without_live_replay_candidate"
    ]


def test_mixed_batch_exit_code_is_fail_closed(tmp_path: Path, capsys) -> None:
    """Any non-eligible row should make the CLI return nonzero."""
    records = tmp_path / "records.json"
    payload = [
        _eligible_record(tmp_path),
        _eligible_record(tmp_path, live_metrics_supported=False),
    ]
    records.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = checker.main([str(records), "--base-dir", str(tmp_path)])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["live_replay_eligibility"]["counts"] == {
        "eligible": 1,
        "missing-fields": 1,
    }


def test_malformed_record_blocks_fail_closed(tmp_path: Path) -> None:
    """Malformed JSON list entries should fail closed as blocked."""
    row = checker.evaluate_records(["not-a-record"], base_dir=tmp_path)["live_replay_eligibility"][
        "classifications"
    ][0]

    assert row["status"] == "blocked"
    assert row["missing_prerequisites"] == ["record_object"]
    assert row["blocked_reasons"] == ["malformed_record"]
    assert row["fixture_path"] is None
    assert row["policy_candidate"] is None
    assert row["claim_boundary"] is None


def test_cli_writes_json_and_uses_blocked_exit_code(tmp_path: Path, capsys) -> None:
    """CLI should write JSON and return nonzero when blocked rows exist."""
    records = tmp_path / "records.json"
    output = tmp_path / "report.json"
    records.write_text(json.dumps(["bad-record"]), encoding="utf-8")

    exit_code = checker.main(
        [str(records), "--base-dir", str(tmp_path), "--output-json", str(output)]
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert stdout_payload == file_payload
    assert file_payload["live_replay_eligibility"]["counts"] == {"blocked": 1}


def test_cli_returns_success_for_all_eligible_records(tmp_path: Path, capsys) -> None:
    """CLI should return zero only when every record is eligible."""
    records = tmp_path / "records.json"
    records.write_text(json.dumps([_eligible_record(tmp_path)]), encoding="utf-8")

    exit_code = checker.main([str(records), "--base-dir", str(tmp_path)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["live_replay_eligibility"]["counts"] == {
        "eligible": 1
    }
