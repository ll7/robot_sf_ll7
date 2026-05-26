"""Tests for hybrid evidence matrix validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from robot_sf.benchmark.hybrid_evidence_matrix import (
    load_hybrid_evidence_input,
    validate_hybrid_evidence_file,
    validate_hybrid_evidence_rows,
)
from scripts.validation.validate_hybrid_evidence_matrix import main as validate_cli_main

FIXTURE_ROOT = Path("tests/fixtures/hybrid_evidence_matrix/v1")


def test_valid_rows_validate_and_preserve_launch_packet_rows() -> None:
    """Valid execution and launch-packet rows should both validate deterministically."""
    report = validate_hybrid_evidence_file(FIXTURE_ROOT / "valid_rows.yaml")

    assert report["status"] == "valid"
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
    assert payload["row_count"] == 1
    assert payload["rows"][0]["errors"][0]["field"]
