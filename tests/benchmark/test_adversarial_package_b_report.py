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
