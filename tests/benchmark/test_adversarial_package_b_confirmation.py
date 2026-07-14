"""Tests for the issue #3079 Package B confirmation-chain gate."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.adversarial_package_b_confirmation import (
    validate_package_b_confirmation,
)
from scripts.tools.validate_adversarial_package_b_confirmation import main

if TYPE_CHECKING:
    from pathlib import Path


def _base_row(*, manifest_path: Path, sampler: str, budget: int, seed: int) -> dict[str, Any]:
    """Build one complete source-report row backed by a synthetic manifest."""
    return {
        "objective": "worst_case_snqi",
        "sampler": sampler,
        "budget": budget,
        "seed": seed,
        "manifest_path": manifest_path.as_posix(),
        "best_bundle_path": manifest_path.parent.as_posix(),
        "best_objective_value": 2.0,
        "best_valid_objective": 2.0,
        "num_candidates": budget,
        "num_valid_candidates": budget,
        "num_invalid_candidates": 0,
        "num_failed_evaluations": 0,
        "invalid_candidate_rate": 0.0,
        "first_failure_iteration": 1,
        "certified_valid_failure_count": 1,
        "replayable_valid_failure_count": 1,
        "replay_success_rate": 1.0,
        "fallback_candidate_count": 0,
        "degraded_candidate_count": 0,
        "held_out_family_yield": None,
        "held_out_family_status": "not_evaluated_narrow_archive",
        "caveats": ["diagnostic/local nominal report"],
    }


def _source_report(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Write a complete 27-cell report and one synthetic manifest per cell."""
    rows: list[dict[str, Any]] = []
    for sampler in ("random", "coordinate", "optuna"):
        for budget in (16, 32, 64):
            for seed in (1101, 2202, 3303):
                cell = tmp_path / f"{sampler}-{budget}-{seed}"
                cell.mkdir()
                candidates = [
                    {
                        "candidate": {"scenario_seed": 901101},
                        "certification_status": {"status": "passed"},
                        "objective_value": 2.0,
                        "failure_attribution": {"primary_failure": "collision"},
                        "error": None,
                    }
                ]
                candidates.extend(
                    {
                        "candidate": {"scenario_seed": 900_000 + seed + index},
                        "certification_status": {"status": "passed"},
                        "objective_value": 0.0,
                        "failure_attribution": {"primary_failure": "success"},
                        "error": None,
                    }
                    for index in range(1, budget)
                )
                manifest = cell / "manifest.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "schema_version": "adversarial-search-manifest.v1",
                            "candidates": candidates,
                        }
                    ),
                    encoding="utf-8",
                )
                rows.append(
                    _base_row(
                        manifest_path=manifest,
                        sampler=sampler,
                        budget=budget,
                        seed=seed,
                    )
                )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "adversarial-sampler-comparison.v3",
                "report_status": "diagnostic_local_nominal",
                "claim_scope": "not_paper_facing_benchmark_evidence",
                "objectives": ["worst_case_snqi"],
                "budget_grid": [16, 32, 64],
                "seeds": [1101, 2202, 3303],
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return report, rows


def _confirmation_sidecar(
    report: Path,
    rows: list[dict[str, Any]],
    *,
    confirmed: bool = True,
) -> Path:
    """Write a confirmation sidecar with one reviewed candidate per cell."""
    evidence_rows: list[dict[str, Any]] = []
    for row in rows:
        if confirmed:
            evidence = [
                {
                    "candidate_index": 1,
                    "candidate_seed": 901101,
                    "primary_failure": "collision",
                    "status": "confirmed",
                    "simulator_seconds_elapsed": 3.0,
                    "replay": {
                        "status": "passed",
                        "deterministic": True,
                        "artifact_path": str(report.parent / "replay.json"),
                    },
                    "independent_seed_confirmation": {
                        "status": "passed",
                        "seeds": [701, 702],
                        "failure_persistence_rate": 1.0,
                        "artifact_path": str(report.parent / "independent.json"),
                    },
                    "mechanism_attribution": {
                        "status": "stable",
                        "primary_failure": "collision",
                        "artifact_path": str(report.parent / "attribution.json"),
                    },
                }
            ]
            confirmed_count = 1
            time_to_first = 3.0
            censored = False
            per_failure = 10.0
        else:
            evidence = [
                {
                    "candidate_index": 1,
                    "status": "not_confirmed",
                    "reason": "confirmation artifact unavailable",
                }
            ]
            confirmed_count = 0
            time_to_first = None
            censored = True
            per_failure = None
        evidence_rows.append(
            {
                "sampler": row["sampler"],
                "budget": row["budget"],
                "seed": row["seed"],
                "confirmation_status": "complete",
                "certified_failure_count": 1,
                "confirmed_failure_count": confirmed_count,
                "unconfirmed_certified_failure_count": 1 - confirmed_count,
                "time_to_first_confirmed_failure_s": time_to_first,
                "time_to_first_confirmed_failure_censored": censored,
                "simulator_seconds": 10.0,
                "simulator_seconds_per_confirmed_failure": per_failure,
                "evidence": evidence,
            }
        )
    for name in ("replay.json", "independent.json", "attribution.json"):
        path = report.parent / name
        path.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": "adversarial-package-b-confirmation.v1",
        "issue": 3079,
        "source_report_schema_version": "adversarial-sampler-comparison.v3",
        "source_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "claim_scope": "not_paper_facing_benchmark_evidence",
        "rows": evidence_rows,
    }
    sidecar = report.parent / "confirmation.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


def test_confirmation_gate_accepts_complete_chain_and_derived_metrics(tmp_path: Path) -> None:
    """Every certified failure is bound to all required confirmation artifacts."""
    report, rows = _source_report(tmp_path)
    sidecar = _confirmation_sidecar(report, rows)

    gate = validate_package_b_confirmation(report, sidecar)

    assert gate.ready is True
    assert gate.status == "ready_for_confirmed_failure_review"
    assert gate.matrix["confirmed_failure_count"] == 27
    assert gate.matrix["unconfirmed_certified_failure_count"] == 0
    assert gate.matrix["censored_row_count"] == 0
    assert "dissertation" in gate.to_payload()["claim_boundary"]


def test_confirmation_gate_preserves_censored_no_confirmation_rows(tmp_path: Path) -> None:
    """A reviewed but unconfirmed failure is excluded and remains right-censored."""
    report, rows = _source_report(tmp_path)
    sidecar = _confirmation_sidecar(report, rows, confirmed=False)

    gate = validate_package_b_confirmation(report, sidecar)

    assert gate.ready is True
    assert gate.matrix["confirmed_failure_count"] == 0
    assert gate.matrix["unconfirmed_certified_failure_count"] == 27
    assert gate.matrix["censored_row_count"] == 27


def test_confirmation_gate_fails_closed_on_missing_artifact_and_seed_drift(tmp_path: Path) -> None:
    """Artifact absence and reuse of the search seed cannot pass as confirmation."""
    report, rows = _source_report(tmp_path)
    sidecar = _confirmation_sidecar(report, rows)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["rows"][0]["evidence"][0]["candidate_seed"] = 1101
    payload["rows"][0]["evidence"][0]["replay"]["artifact_path"] = str(
        tmp_path / "missing-replay.json"
    )
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    gate = validate_package_b_confirmation(report, sidecar)

    assert gate.ready is False
    assert gate.status == "blocked_on_confirmation_contract"
    assert any("candidate_seed drifted" in error for error in gate.errors)
    assert any("does not resolve to an existing file" in error for error in gate.errors)


def test_confirmation_gate_cli_writes_payload_and_returns_status(tmp_path: Path) -> None:
    """The CLI emits and persists the same compact gate payload."""
    report, rows = _source_report(tmp_path)
    sidecar = _confirmation_sidecar(report, rows)
    output = tmp_path / "gate.json"

    assert main([str(report), str(sidecar), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["ready"] is True
