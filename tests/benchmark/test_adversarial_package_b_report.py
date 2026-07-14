"""Tests for the issue #3079 Package B post-run report gate."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.adversarial_package_b_report import validate_package_b_report
from scripts.tools.validate_adversarial_package_b_report import main

if TYPE_CHECKING:
    from pathlib import Path


def _row(*, sampler: str, budget: int, seed: int, limited: bool = False) -> dict[str, object]:
    """Build one complete synthetic Package B row."""
    certified = 1
    replayable = 1
    return {
        "objective": "worst_case_snqi",
        "sampler": sampler,
        "budget": budget,
        "seed": seed,
        "manifest_path": f"output/{sampler}-{budget}-{seed}/manifest.json",
        "best_bundle_path": f"output/{sampler}-{budget}-{seed}/bundle",
        "best_objective_value": 2.0,
        "best_valid_objective": 2.0,
        "num_candidates": budget,
        "num_valid_candidates": budget,
        "num_invalid_candidates": 0,
        "num_failed_evaluations": 0,
        "invalid_candidate_rate": 0.0,
        "first_failure_iteration": 1,
        "certified_valid_failure_count": certified,
        "replayable_valid_failure_count": replayable,
        "replay_success_rate": 1.0,
        "fallback_candidate_count": 1 if limited else 0,
        "degraded_candidate_count": 0,
        "held_out_family_yield": None,
        "held_out_family_status": "not_evaluated_narrow_archive",
        "caveats": ["diagnostic/local nominal report"],
    }


def _report(*, limited: bool = False) -> dict[str, object]:
    """Build a complete 27-cell synthetic Package B report."""
    return {
        "schema_version": "adversarial-sampler-comparison.v3",
        "report_status": "diagnostic_local_nominal",
        "claim_scope": "not_paper_facing_benchmark_evidence",
        "objectives": ["worst_case_snqi"],
        "budget_grid": [16, 32, 64],
        "seeds": [1101, 2202, 3303],
        "rows": [
            _row(sampler=sampler, budget=budget, seed=seed, limited=limited)
            for sampler in ("random", "coordinate", "optuna")
            for budget in (16, 32, 64)
            for seed in (1101, 2202, 3303)
        ],
    }


def test_report_gate_accepts_complete_matrix_and_preserves_caveats(tmp_path: Path) -> None:
    """A complete matrix is ready for empirical review, not paper-facing claims."""
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")

    gate = validate_package_b_report(report)

    assert gate.ready is True
    assert gate.status == "ready_for_empirical_review"
    payload = gate.to_payload()
    assert payload["matrix"]["observed_row_count"] == 27
    assert payload["blockers"]["intentional"]
    assert "approved compute-capable path" in payload["next_empirical_action"]
    assert "paper-facing" in payload["claim_boundary"]


def test_report_gate_excludes_fallback_rows_from_nominal_readiness(tmp_path: Path) -> None:
    """Fallback/degraded rows remain diagnostic limitations, never success evidence."""
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report(limited=True)), encoding="utf-8")

    gate = validate_package_b_report(report)

    assert gate.ready is False
    assert gate.status == "diagnostic_only_limited_rows"
    assert gate.matrix["fallback_candidate_count"] == 27
    assert any("fallback/degraded" in item for item in gate.blockers["new"])


def test_report_gate_rejects_float_counts_and_numeric_string_rates(tmp_path: Path) -> None:
    """JSON values that resemble numbers do not bypass the native-type contract."""
    payload = _report()
    payload["rows"][0]["fallback_candidate_count"] = 0.0
    payload["rows"][0]["invalid_candidate_rate"] = "0.0"
    report = tmp_path / "numeric-lookalikes.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    gate = validate_package_b_report(report)

    assert gate.ready is False
    assert any("fallback_candidate_count" in error for error in gate.errors)
    assert any("invalid_candidate_rate must be finite" in error for error in gate.errors)


def test_report_gate_fails_closed_for_missing_matrix_cell_and_claim_drift(tmp_path: Path) -> None:
    """Missing cells and claim promotion are hard report-contract failures."""
    payload = _report()
    payload["claim_scope"] = "paper_facing_benchmark_evidence"
    payload["rows"] = payload["rows"][:-1]
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    gate = validate_package_b_report(report)

    assert gate.ready is False
    assert gate.status == "blocked_on_report_contract"
    assert any("claim_scope" in item for item in gate.errors)
    assert any("matrix cells" in item for item in gate.errors)
    assert any("exactly 27" in item for item in gate.errors)


def test_report_gate_cli_writes_payload_and_returns_fail_closed_status(tmp_path: Path) -> None:
    """The CLI writes the same compact gate payload it prints."""
    report = tmp_path / "report.json"
    output = tmp_path / "gate.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")

    assert main([str(report), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["ready"] is True


def test_report_gate_fails_closed_for_unreadable_and_non_mapping_reports(tmp_path: Path) -> None:
    """Malformed JSON, scalar JSON, and a non-list rows field become blockers."""
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert validate_package_b_report(malformed).status == "blocked_on_report_contract"
    assert "could not be read as JSON" in validate_package_b_report(malformed).errors[0]

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert "report payload must be a mapping" in validate_package_b_report(scalar).errors

    rows_not_list = _report()
    rows_not_list["rows"] = "not-a-list"
    rows_path = tmp_path / "rows-not-list.json"
    rows_path.write_text(json.dumps(rows_not_list), encoding="utf-8")
    assert "rows must be a list" in validate_package_b_report(rows_path).errors


def test_report_gate_checks_row_shapes_arithmetic_and_rates(tmp_path: Path) -> None:
    """Invalid row types and denominators are surfaced rather than normalized silently."""
    payload = _report()
    payload["seeds"] = [999]
    rows = payload["rows"]
    rows.extend(
        [
            "not-a-row",
            {"sampler": "random"},
            _row(sampler="unknown", budget=999, seed=999),
            dict(rows[0]),
        ]
    )
    invalid_counts = _row(sampler="random", budget=16, seed=2202)
    for field in (
        "num_candidates",
        "num_valid_candidates",
        "num_invalid_candidates",
        "num_failed_evaluations",
        "certified_valid_failure_count",
        "replayable_valid_failure_count",
        "fallback_candidate_count",
        "degraded_candidate_count",
    ):
        invalid_counts[field] = "not-an-int"
    invalid_counts["invalid_candidate_rate"] = False
    invalid_counts["replay_success_rate"] = None
    rows.append(invalid_counts)

    bad_relationships = _row(sampler="coordinate", budget=16, seed=2202)
    bad_relationships.update(
        {
            "num_candidates": 1,
            "num_valid_candidates": 2,
            "num_invalid_candidates": 2,
            "num_failed_evaluations": 2,
            "certified_valid_failure_count": 3,
            "replayable_valid_failure_count": 4,
            "invalid_candidate_rate": 0.5,
            "replay_success_rate": 0.0,
            "first_failure_iteration": 0,
            "held_out_family_yield": 0.5,
            "held_out_family_status": "evaluated",
            "fallback_candidate_count": 2,
            "degraded_candidate_count": 2,
        }
    )
    rows.append(bad_relationships)

    missing_replay_rate = _row(sampler="coordinate", budget=16, seed=3303)
    missing_replay_rate["replay_success_rate"] = None
    rows.append(missing_replay_rate)
    zero_certified_replay_rate = _row(sampler="optuna", budget=16, seed=3303)
    zero_certified_replay_rate.update(
        {
            "certified_valid_failure_count": 0,
            "replayable_valid_failure_count": 0,
            "replay_success_rate": 0.0,
            "first_failure_iteration": None,
        }
    )
    rows.append(zero_certified_replay_rate)

    report = tmp_path / "invalid-rows.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    gate = validate_package_b_report(report)

    assert gate.ready is False
    assert gate.status == "blocked_on_report_contract"
    assert any("duplicates matrix cell" in error for error in gate.errors)
    assert any("replay_success_rate is missing" in error for error in gate.errors)
    assert any("held_out_family_yield must remain null" in error for error in gate.errors)
