"""Tests for campaign recovery verifier (scripts/validation/verify_campaign_recovery.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation.verify_campaign_recovery import (
    SCHEMA_VERSION,
    format_summary,
    main,
    validate_schema,
    verify_recovery,
)

FIXTURES_DIR = Path("tests/validation/fixtures/campaign_recovery")


def test_valid_benchmark_resume_passes() -> None:
    """Interrupted campaign cleanly resumed after preemption passes validation."""
    fixture_path = FIXTURES_DIR / "valid_benchmark_resume.json"
    packet = json.loads(fixture_path.read_text(encoding="utf-8"))
    receipt = verify_recovery(packet)

    assert receipt["schema"] == SCHEMA_VERSION
    assert receipt["campaign_id"] == "camp_recovery_01"
    assert receipt["runner_class"] == "benchmark_matrix"
    assert receipt["status"] == "pass"
    assert receipt["verdict"] == "pass"
    assert receipt["retry_admitted"] is True
    assert receipt["interruption_reason"] == "preemption"
    assert receipt["interruption_class"] == "infrastructure"
    assert receipt["completed_rows_preserved"] is True
    assert receipt["authority_inputs_matched"] is True
    assert receipt["reconciled_to_expected_ledger"] is True
    assert receipt["expected_row_count"] == 4
    assert receipt["reconciled_row_count"] == 4
    assert receipt["discrepancies"] == []
    assert len(receipt["attempts"]) == 2

    # Schema validation passes with no errors
    errors = validate_schema(receipt)
    assert not errors


def test_outcome_driven_retry_rejected() -> None:
    """Interruption due to outcome collision failure rejects retry under fail-closed policy."""
    fixture_path = FIXTURES_DIR / "outcome_retry_rejected.json"
    packet = json.loads(fixture_path.read_text(encoding="utf-8"))
    receipt = verify_recovery(packet)

    assert receipt["schema"] == SCHEMA_VERSION
    assert receipt["status"] == "fail"
    assert receipt["verdict"] == "fail"
    assert receipt["retry_admitted"] is False
    assert receipt["interruption_class"] == "outcome"
    assert any("outcome_driven_retry_rejected" in d for d in receipt["discrepancies"])
    assert any("unadmitted_failure_retry" in d for d in receipt["discrepancies"])

    # Schema validation still passes on fail receipt
    errors = validate_schema(receipt)
    assert not errors


def test_unsupported_runner_reports_unsupported() -> None:
    """Unknown runner reports status 'unsupported' and fail verdict."""
    packet = {
        "campaign_id": "c_unsupported",
        "runner_class": "custom_ad_hoc_runner",
        "authority_inputs": {"commit": "abc1234"},
        "initial_run": {
            "attempt_index": 1,
            "job_id": "job-1",
            "status": "completed",
            "rows": [{"row_id": "r1", "status": "completed"}],
        },
        "resume_attempt": {
            "attempt_index": 2,
            "job_id": "job-2",
            "status": "completed",
            "rows": [],
        },
        "expected_rows": ["r1"],
    }
    receipt = verify_recovery(packet)

    assert receipt["status"] == "unsupported"
    assert receipt["verdict"] == "fail"
    assert any("unsupported_runner" in d for d in receipt["discrepancies"])
    assert not validate_schema(receipt)


def test_supported_slurm_array_runner() -> None:
    """The slurm_array runner is recognized as supported."""
    packet = {
        "campaign_id": "c_array",
        "runner_class": "slurm_array",
        "authority_inputs": {"commit": "abc1234"},
        "initial_run": {
            "attempt_index": 1,
            "job_id": "slurm-5001",
            "status": "completed",
            "rows": [{"row_id": "r1", "status": "completed"}],
        },
        "resume_attempt": {
            "attempt_index": 2,
            "job_id": "slurm-5002",
            "status": "completed",
            "rows": [],
        },
        "expected_rows": ["r1"],
    }
    receipt = verify_recovery(packet)
    assert receipt["status"] == "pass"
    assert receipt["verdict"] == "pass"
    assert not receipt["discrepancies"]
    assert not validate_schema(receipt)


def test_completed_identity_overwritten_fails() -> None:
    """Completed valid rows must not be silently rerun or overwritten."""
    packet = {
        "campaign_id": "c_overwrite",
        "runner_class": "benchmark_matrix",
        "authority_inputs": {"commit": "abc1234"},
        "initial_run": {
            "attempt_index": 1,
            "job_id": "job-1",
            "status": "preempted",
            "rows": [{"row_id": "r1", "status": "present"}],
        },
        "resume_attempt": {
            "attempt_index": 2,
            "job_id": "job-2",
            "status": "completed",
            "rows": [{"row_id": "r1", "status": "present"}],
        },
        "expected_rows": ["r1"],
    }
    receipt = verify_recovery(packet)
    assert receipt["verdict"] == "fail"
    assert receipt["completed_rows_preserved"] is False
    assert any("completed_identity_overwritten" in d for d in receipt["discrepancies"])
    assert not validate_schema(receipt)


def test_degraded_became_success_rejected() -> None:
    """Degraded/fallback rows cannot be converted to clean success through resume."""
    packet = {
        "campaign_id": "c_degraded",
        "runner_class": "benchmark_matrix",
        "authority_inputs": {"commit": "abc1234"},
        "initial_run": {
            "attempt_index": 1,
            "job_id": "job-1",
            "status": "preempted",
            "rows": [{"row_id": "r1", "status": "degraded"}],
        },
        "resume_attempt": {
            "attempt_index": 2,
            "job_id": "job-2",
            "status": "completed",
            "rows": [{"row_id": "r1", "status": "present"}],
        },
        "expected_rows": ["r1"],
    }
    receipt = verify_recovery(packet)
    assert receipt["verdict"] == "fail"
    assert any("degraded_became_success" in d for d in receipt["discrepancies"])
    assert not validate_schema(receipt)


def test_authority_input_drift_rejected() -> None:
    """Drift in commit or config hash across resume attempts must fail verification."""
    packet = {
        "campaign_id": "c_drift",
        "runner_class": "benchmark_matrix",
        "initial_run": {
            "attempt_index": 1,
            "job_id": "job-1",
            "status": "completed",
            "authority_inputs": {"commit": "commit_1111", "config_hash": "cfg_aaaa"},
            "rows": [{"row_id": "r1", "status": "present"}],
        },
        "resume_attempt": {
            "attempt_index": 2,
            "job_id": "job-2",
            "status": "completed",
            "authority_inputs": {"commit": "commit_2222", "config_hash": "cfg_aaaa"},
            "rows": [],
        },
        "expected_rows": ["r1"],
    }
    receipt = verify_recovery(packet)
    assert receipt["verdict"] == "fail"
    assert receipt["authority_inputs_matched"] is False
    assert any("authority_input_drift" in d for d in receipt["discrepancies"])
    assert not validate_schema(receipt)


def test_missing_and_unexpected_reconciliation_discrepancies() -> None:
    """Mismatch between reconciled rows and expected rows is flagged."""
    packet = {
        "campaign_id": "c_reconcile",
        "runner_class": "benchmark_matrix",
        "authority_inputs": {"commit": "commit_1111"},
        "initial_run": {
            "attempt_index": 1,
            "job_id": "job-1",
            "status": "completed",
            "rows": [{"row_id": "r_unexpected", "status": "present"}],
        },
        "resume_attempt": {
            "attempt_index": 2,
            "job_id": "job-2",
            "status": "completed",
            "rows": [],
        },
        "expected_rows": ["r_missing"],
    }
    receipt = verify_recovery(packet)
    assert receipt["verdict"] == "fail"
    assert receipt["reconciled_to_expected_ledger"] is False
    assert any("missing_expected_rows" in d for d in receipt["discrepancies"])
    assert any("unexpected_rows" in d for d in receipt["discrepancies"])
    assert not validate_schema(receipt)


def test_format_summary() -> None:
    """Summary output format contains key identifiers and attempt details."""
    fixture_path = FIXTURES_DIR / "valid_benchmark_resume.json"
    packet = json.loads(fixture_path.read_text(encoding="utf-8"))
    receipt = verify_recovery(packet)
    summary = format_summary(receipt)
    assert "camp_recovery_01" in summary
    assert "Attempt 1: job=slurm-1001" in summary
    assert "Attempt 2: job=slurm-1002" in summary


def test_cli_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI exits 0 on valid fixture with --check, and writes output file."""
    fixture_path = FIXTURES_DIR / "valid_benchmark_resume.json"
    out_file = tmp_path / "out_receipt.json"

    ret = main(["--fixture", str(fixture_path), "--output", str(out_file), "--check"])
    assert ret == 0
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["verdict"] == "pass"

    # Testing exit code 2 when check fails
    rejected_path = FIXTURES_DIR / "outcome_retry_rejected.json"
    ret_fail = main(["--fixture", str(rejected_path), "--check"])
    assert ret_fail == 2
