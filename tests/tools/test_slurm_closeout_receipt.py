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
