"""Focused contract tests for terminal Slurm closeout receipts."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from scripts.tools.slurm_closeout_receipt import SCHEMA_VERSION, validate_file, validate_payload

if TYPE_CHECKING:
    from pathlib import Path


def _receipt(*, state: str = "COMPLETED", exit_code: str = "0:0") -> dict:
    campaign = "campaign-fixture"
    source = "a" * 40
    job_id = "15180"
    return {
        "schema": SCHEMA_VERSION,
        "status": "terminal",
        "campaign_id": campaign,
        "source_sha": source,
        "job_id": job_id,
        "identity_sha256": hashlib.sha256(f"{campaign}\0{source}\0{job_id}".encode()).hexdigest(),
        "scheduler": {
            "state": state,
            "exit_code": exit_code,
            "derived_exit_code": "0:0",
            "elapsed": "00:01:00",
        },
        "query": {
            "tool": "sacct",
            "tool_version": "robot-sf-sacct-query.v1",
            "queried_at": "2026-09-15T10:00:00Z",
        },
        "allocation": {
            "cluster": "imech192",
            "partition": "l40s",
            "cpus": 36,
            "gpus": 1,
            "mem_gb": 256,
        },
        "reconciliation": {
            "status": "reconciled",
            "prior_scheduler_state": "RUNNING",
        },
        "output": {"digest_sha256": "b" * 64, "kind": "artifact_manifest"},
    }


def test_completed_zero_exit_is_valid() -> None:
    assert validate_payload(_receipt(), expected_campaign_id="campaign-fixture") == []


def test_failed_exit_is_preserved_as_valid_terminal_evidence() -> None:
    payload = _receipt(state="FAILED", exit_code="1:0")
    assert validate_payload(payload, expected_source_sha="a" * 40) == []
    payload = _receipt(state="FAILED", exit_code="2:0")
    assert validate_payload(payload) == []


def test_stale_running_receipt_is_rejected() -> None:
    payload = _receipt(state="RUNNING", exit_code="0:0")
    problems = validate_payload(payload)
    assert any("terminal scheduler state" in problem for problem in problems)
    assert any("active scheduler state" in problem for problem in problems)


def test_unavailable_scheduler_readback_blocks_terminal_admission() -> None:
    payload = _receipt(state="UNAVAILABLE", exit_code="")
    payload["status"] = "unavailable"
    payload["output"]["digest_sha256"] = None
    assert "scheduler readback is unavailable; release admission is blocked" in validate_payload(
        payload
    )


def test_identity_and_source_mismatch_are_blocking(tmp_path: Path) -> None:
    payload = _receipt()
    payload["identity_sha256"] = "c" * 64
    path = tmp_path / "closeout.json"
    path.write_text("not-json\n", encoding="utf-8")
    assert validate_file(path) == ["scheduler closeout receipt cannot be read"]
    assert "closeout receipt identity digest does not match its identity" in validate_payload(
        payload
    )


def test_credential_shaped_fields_are_rejected() -> None:
    payload = _receipt()
    payload["token"] = "must-not-be-recorded"
    assert "closeout receipt contains a credential-shaped field" in validate_payload(payload)


def test_malformed_receipt_reports_required_identity_and_sections() -> None:
    payload = {
        "schema": "wrong-schema",
        "status": "invalid",
        "campaign_id": "",
        "source_sha": "not-a-sha",
        "job_id": "not-a-job",
        "identity_sha256": "not-a-digest",
        "scheduler": {
            "state": "COMPLETED",
            "exit_code": "bad",
            "derived_exit_code": "bad",
            "elapsed": "",
        },
        "query": "missing",
        "allocation": {},
        "reconciliation": {"status": "started"},
        "output": {"digest_sha256": "bad"},
        "items": [{"nested": True}],
    }
    problems = validate_payload(
        payload,
        expected_campaign_id="expected-campaign",
        expected_source_sha="b" * 40,
        expected_job_id="42",
    )
    assert "closeout receipt schema is unsupported" in problems
    assert "closeout receipt status is not terminal or unavailable" in problems
    assert "closeout receipt campaign_id is missing" in problems
    assert "closeout receipt source_sha is not a 40-character commit SHA" in problems
    assert "closeout receipt job_id is not numeric" in problems
    assert "closeout receipt campaign_id does not match the expected campaign" in problems
    assert "closeout receipt source_sha does not match the expected source" in problems
    assert "closeout receipt job_id does not match the expected job" in problems
    assert "closeout receipt identity_sha256 is missing or malformed" in problems
    assert "closeout receipt scheduler exit_code is malformed" in problems
    assert "closeout receipt scheduler derived_exit_code is malformed" in problems
    assert "COMPLETED closeout receipt does not carry exit code 0:0" in problems
    assert "COMPLETED closeout receipt has a non-zero derived exit code" in problems
    assert "closeout receipt query section is missing" in problems
    assert "closeout receipt allocation cluster is missing" in problems
    assert "closeout receipt does not prove terminal admission reconciliation" in problems
    assert "closeout receipt output digest is malformed" in problems


def test_unavailable_and_incomplete_receipt_sections_fail_closed() -> None:
    payload = {
        "schema": SCHEMA_VERSION,
        "status": "unavailable",
        "campaign_id": "campaign-fixture",
        "source_sha": "a" * 40,
        "job_id": "15180",
        "identity_sha256": hashlib.sha256(
            f"campaign-fixture\0{'a' * 40}\0{'15180'}".encode()
        ).hexdigest(),
        "scheduler": {"state": "ACTIVE"},
        "query": {"tool": "sacct"},
        "allocation": "missing",
        "reconciliation": {},
        "output": None,
    }
    problems = validate_payload(payload)
    assert "unavailable closeout receipt does not contain an unavailable scheduler state" in problems
    assert "scheduler readback is unavailable; release admission is blocked" in problems
    assert "closeout receipt scheduler section is missing" not in problems
    assert "closeout receipt query tool version is missing" in problems
    assert "closeout receipt query timestamp is missing" in problems
    assert "closeout receipt allocation contract is missing" in problems
    assert "closeout receipt output section is missing" in problems


def test_non_object_payload_and_missing_file_are_rejected(tmp_path: Path) -> None:
    assert validate_payload(None) == ["closeout receipt is not a JSON object"]
    assert validate_file(tmp_path / "missing.json") == ["scheduler closeout receipt is missing"]


def test_terminal_receipt_requires_scheduler_and_output_details() -> None:
    payload = {
        "schema": SCHEMA_VERSION,
        "status": "terminal",
        "campaign_id": "campaign-fixture",
        "source_sha": "a" * 40,
        "job_id": "15180",
        "identity_sha256": hashlib.sha256(
            f"campaign-fixture\0{'a' * 40}\0{'15180'}".encode()
        ).hexdigest(),
        "output": {"digest_sha256": None},
    }
    problems = validate_payload(payload)
    assert "closeout receipt scheduler section is missing" in problems
    assert "terminal closeout receipt does not contain a terminal scheduler state" in problems
    assert "terminal closeout receipt elapsed time is missing" in problems
    assert "closeout receipt query section is missing" in problems
    assert "closeout receipt allocation contract is missing" in problems
    assert "closeout receipt does not prove terminal admission reconciliation" in problems
    assert "terminal closeout receipt has no output digest" in problems
