"""Tests for campaign submission binding validator and compare-and-swap guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scripts.validation.validate_campaign_submission_binding import (
    SCHEMA_VERSION,
    main,
    validate_submission_binding,
)


@pytest.fixture
def base_preflight_packet() -> dict[str, Any]:
    """Return a complete, valid preflight packet fixture."""
    return {
        "checkpoint": {
            "alias": "expert_ppo_v2",
            "model_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "path": "checkpoints/expert_ppo_v2.pt",
        },
        "command_tokens": {
            "tokens": ["python3", "-m", "robot_sf.benchmark", "--config", "configs/bench.yaml"],
        },
        "config": {
            "content_sha256": "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
            "path": "configs/benchmarks/paper_matrix_v1.yaml",
            "resolved_config_sha256": "f9e8d7c6b5a40392817a6b5c4d3e2f10f9e8d7c6b5a40392817a6b5c4d3e2f10",
        },
        "duplicate_state": {"active_job_ids": [], "duplicate_detected": False},
        "environment": {
            "lock_sha256": "b0c1d2e3f405162738495a6b7c8d9e0fa1b2c3d4e5f60718293a4b5c6d7e8f90",
            "python_version": "3.13.13",
        },
        "issue_state": {
            "admission_status": "claimed",
            "claim_ref": "agent-claims/issue-8839",
            "issue_number": 8839,
        },
        "observed_at_utc": "2026-09-11T12:00:00Z",
        "output_root": {"path": "output/benchmarks/campaign_5409"},
        "resource_request": {
            "cpus": 16,
            "gpus": 1,
            "nodes": 1,
            "partition": "epyc-gpu",
            "time_limit": "04:00:00",
        },
        "row_ledger": {
            "expected_rows": 2160,
            "ledger_sha256": "c3d4e5f6a7b8091a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a",
        },
        "seed_set": {"seed_policy": "sequential", "seeds": [42, 43, 44, 45, 46]},
        "source": {
            "commit": "8c23af9dba3641a430adc21d0979120e4198eac4",
            "is_dirty": False,
            "untracked_files": [],
        },
        "submission_nonce": "nonce-initial-001",
    }


@pytest.fixture
def schema_definition() -> dict[str, Any]:
    """Load JSON schema definition from docs/contracts."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "contracts"
        / "campaign_submission_binding.v1.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_exact_match_passes(base_preflight_packet, schema_definition):
    """Identical preflight and submission must pass with zero differences."""
    submission = dict(base_preflight_packet)
    receipt = validate_submission_binding(base_preflight_packet, submission)
    assert receipt["verdict"] == "pass"
    assert receipt["status"] == "matched"
    assert receipt["first_differing_field"] is None
    assert receipt["differing_fields"] == []
    assert receipt["preflight_digest"] == receipt["submission_digest"]
    jsonschema.validate(instance=receipt, schema=schema_definition)


@pytest.mark.parametrize(
    ("path", "drift_val", "expected_field"),
    [
        (("source", "commit"), "0" * 40, "source.commit"),
        (("source", "is_dirty"), True, "source.is_dirty"),
        (("source", "untracked_files"), ["scratch/untracked.py"], "source.untracked_files"),
        (("config", "content_sha256"), "b" * 64, "config.content_sha256"),
        (("checkpoint", "model_sha256"), "c" * 64, "checkpoint.model_sha256"),
        (("seed_set", "seeds"), [46, 45, 44, 43, 42], "seed_set.seeds"),
        (("command_tokens", "tokens"), ["python3", "-m", "foo --bar"], "command_tokens.tokens"),
        (("resource_request", "cpus"), 32, "resource_request.cpus"),
        (("output_root", "path"), "output/benchmarks/campaign_other", "output_root.path"),
        (("issue_state", "admission_status"), "stale", "issue_state.admission_status"),
        (("duplicate_state", "duplicate_detected"), True, "duplicate_state.duplicate_detected"),
    ],
)
def test_drift_scenarios_fail_closed(base_preflight_packet, path, drift_val, expected_field):
    """All authority-bearing identity drifts must fail closed with exact first difference."""
    submission = json.loads(json.dumps(base_preflight_packet))
    target = submission
    for p in path[:-1]:
        target = target[p]
    target[path[-1]] = drift_val

    receipt = validate_submission_binding(base_preflight_packet, submission)
    assert receipt["verdict"] == "fail"
    assert receipt["status"] == "drift_detected"
    assert receipt["first_differing_field"] == expected_field
    assert receipt["preflight_digest"] != receipt["submission_digest"]


def test_permitted_volatile_differences_pass(base_preflight_packet, schema_definition):
    """Differences only in permitted volatile fields must pass without drift failure."""
    submission = json.loads(json.dumps(base_preflight_packet))
    submission["observed_at_utc"] = "2026-09-11T12:05:00Z"
    submission["submission_nonce"] = "nonce-live-mutation-999"
    submission["pid"] = 42109

    receipt = validate_submission_binding(base_preflight_packet, submission)
    assert receipt["verdict"] == "pass"
    assert receipt["status"] == "matched"
    assert receipt["first_differing_field"] is None
    assert "observed_at_utc" in receipt["volatile_fields_observed"]
    assert "submission_nonce" in receipt["volatile_fields_observed"]
    assert "pid" in receipt["volatile_fields_observed"]
    jsonschema.validate(instance=receipt, schema=schema_definition)


def test_unpermitted_unknown_difference_fails(base_preflight_packet):
    """Unknown field introduced at submission boundary must fail closed."""
    submission = json.loads(json.dumps(base_preflight_packet))
    submission["unpermitted_ambient_variable"] = "MALICIOUS_OVERRIDE"

    receipt = validate_submission_binding(base_preflight_packet, submission)
    assert receipt["verdict"] == "fail"
    assert receipt["first_differing_field"] == "unpermitted_ambient_variable"


def test_flat_key_mapping_equivalence(base_preflight_packet):
    """Flat preflight declarations map to canonical authority groups cleanly."""
    flat_packet = {
        "commit": base_preflight_packet["source"]["commit"],
        "is_dirty": False,
        "seeds": base_preflight_packet["seed_set"]["seeds"],
        "tokens": base_preflight_packet["command_tokens"]["tokens"],
        "config_sha256": base_preflight_packet["config"]["content_sha256"],
    }
    receipt = validate_submission_binding(flat_packet, flat_packet)
    assert receipt["verdict"] == "pass"
    assert "seed_set" in receipt["authority_fields_checked"]
    assert "source" in receipt["authority_fields_checked"]


def test_cli_execution_and_receipt_output(tmp_path, monkeypatch, base_preflight_packet):
    """CLI should support --format json, write receipts, and return correct exit codes."""
    preflight_file = tmp_path / "preflight.json"
    submission_file = tmp_path / "submission.json"
    receipt_file = tmp_path / "receipt.json"

    preflight_file.write_text(json.dumps(base_preflight_packet), encoding="utf-8")
    submission_file.write_text(json.dumps(base_preflight_packet), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_campaign_submission_binding.py",
            "--check",
            "--preflight",
            str(preflight_file),
            "--submission",
            str(submission_file),
            "--receipt-output",
            str(receipt_file),
            "--format",
            "json",
        ],
    )
    assert main() == 0
    assert receipt_file.is_file()
    saved = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert saved["schema"] == SCHEMA_VERSION
    assert saved["verdict"] == "pass"


def test_cli_drift_exit_code_1(tmp_path, monkeypatch, base_preflight_packet):
    """CLI returns 1 when drift is detected."""
    preflight_file = tmp_path / "preflight.json"
    submission_file = tmp_path / "submission.json"

    drift_submission = json.loads(json.dumps(base_preflight_packet))
    drift_submission["resource_request"]["gpus"] = 8

    preflight_file.write_text(json.dumps(base_preflight_packet), encoding="utf-8")
    submission_file.write_text(json.dumps(drift_submission), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_campaign_submission_binding.py",
            "--check",
            "--preflight",
            str(preflight_file),
            "--submission",
            str(submission_file),
            "--format",
            "text",
        ],
    )
    assert main() == 1


def test_cli_missing_file_exit_code_2(tmp_path, monkeypatch):
    """CLI returns 2 on missing packet file."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_campaign_submission_binding.py",
            "--preflight",
            str(tmp_path / "nonexistent.json"),
            "--submission",
            str(tmp_path / "nonexistent2.json"),
        ],
    )
    assert main() == 2
