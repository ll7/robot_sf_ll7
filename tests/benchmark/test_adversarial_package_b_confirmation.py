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


def test_confirmation_gate_fails_closed_for_unreadable_and_malformed_inputs(tmp_path: Path) -> None:
    """Missing files, scalar JSON, and absent matrix rows remain blocked."""
    report = tmp_path / "bad-report.json"
    confirmation = tmp_path / "bad-confirmation.json"
    report.write_text("{", encoding="utf-8")
    confirmation.write_text("[]", encoding="utf-8")

    gate = validate_package_b_confirmation(report, confirmation)

    assert gate.ready is False
    assert any("could not be read as JSON" in error for error in gate.errors)
    assert any("JSON payload must be a mapping" in error for error in gate.errors)
    assert any("base Package B report gate is not ready" in error for error in gate.errors)
    assert any("confirmation rows must be a list" in error for error in gate.errors)


def test_confirmation_gate_rejects_row_shape_drift_and_missing_cells(tmp_path: Path) -> None:
    """Malformed, duplicate, and unknown matrix rows cannot bypass the 27-cell contract."""
    report, rows = _source_report(tmp_path)
    sidecar = _confirmation_sidecar(report, rows)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    valid = dict(payload["rows"][0])
    invalid_key = dict(valid)
    invalid_key["budget"] = "16"
    unknown_key = dict(valid)
    unknown_key["sampler"] = "unknown"
    payload["rows"] = ["not-a-row", {}, invalid_key, valid, dict(valid), unknown_key]
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    gate = validate_package_b_confirmation(report, sidecar)

    assert gate.ready is False
    assert any("must be a mapping" in error for error in gate.errors)
    assert any("missing required fields" in error for error in gate.errors)
    assert any("invalid sampler/budget/seed types" in error for error in gate.errors)
    assert any("duplicates matrix cell" in error for error in gate.errors)
    assert any("does not match a base report matrix cell" in error for error in gate.errors)
    assert any("missing Package B matrix cells" in error for error in gate.errors)


def test_confirmation_gate_rejects_manifest_and_evidence_shape_drift(tmp_path: Path) -> None:
    """Missing source manifests and malformed evidence lists remain hard blockers."""
    report, rows = _source_report(tmp_path)
    sidecar = _confirmation_sidecar(report, rows)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload["rows"][0]["manifest_path"] = str(tmp_path / "missing-manifest.json")
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["source_report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    payload["rows"][0]["evidence"] = "not-a-list"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    gate = validate_package_b_confirmation(report, sidecar)

    assert gate.ready is False
    assert any("base manifest_path does not resolve" in error for error in gate.errors)
    assert any("evidence must be a list" in error for error in gate.errors)


def test_confirmation_gate_rejects_stage_and_metric_drift(tmp_path: Path) -> None:
    """Each confirmation stage and derived time metric is independently fail-closed."""
    report, rows = _source_report(tmp_path)
    sidecar = _confirmation_sidecar(report, rows)
    pristine = sidecar.read_text(encoding="utf-8")
    mutations = (
        (lambda entry: entry.pop("replay"), "replay must be a mapping"),
        (lambda entry: entry["replay"].update(status="failed"), "status must equal 'passed'"),
        (lambda entry: entry["replay"].update(deterministic=False), "deterministic must be true"),
        (
            lambda entry: entry["independent_seed_confirmation"].update(seeds=[701]),
            "needs two distinct integer",
        ),
        (
            lambda entry: entry["independent_seed_confirmation"].update(seeds=[901101, 702]),
            "must not include the search candidate seed",
        ),
        (
            lambda entry: entry["independent_seed_confirmation"].update(
                failure_persistence_rate=0.0
            ),
            "failure_persistence_rate must be finite",
        ),
        (
            lambda entry: entry["mechanism_attribution"].update(primary_failure="timeout"),
            "primary_failure does not match",
        ),
        (lambda entry: entry.update(candidate_seed=1101), "candidate_seed drifted"),
        (
            lambda entry: entry["replay"].update(artifact_path=str(tmp_path / "missing.json")),
            "does not resolve to an existing file",
        ),
        (lambda entry: entry.update(simulator_seconds_elapsed=-1.0), "elapsed"),
    )
    for mutate, expected_error in mutations:
        payload = json.loads(pristine)
        mutate(payload["rows"][0]["evidence"][0])
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        gate = validate_package_b_confirmation(report, sidecar)
        assert gate.ready is False
        assert any(expected_error in error for error in gate.errors), expected_error


def test_confirmation_gate_rejects_censoring_and_rate_drift(tmp_path: Path) -> None:
    """Confirmed and censored rows must agree with their time denominators."""
    report, rows = _source_report(tmp_path)
    sidecar = _confirmation_sidecar(report, rows)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["rows"][0]["time_to_first_confirmed_failure_s"] = 4.0
    payload["rows"][0]["time_to_first_confirmed_failure_censored"] = True
    payload["rows"][0]["simulator_seconds_per_confirmed_failure"] = 11.0
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    gate = validate_package_b_confirmation(report, sidecar)

    assert gate.ready is False
    assert any("disagrees with evidence" in error for error in gate.errors)
    assert any("confirmed rows must not be censored" in error for error in gate.errors)
    assert any("disagrees with its denominator" in error for error in gate.errors)

    censored_sidecar = _confirmation_sidecar(report, rows, confirmed=False)
    censored_payload = json.loads(censored_sidecar.read_text(encoding="utf-8"))
    censored_payload["rows"][0]["time_to_first_confirmed_failure_s"] = 1.0
    censored_payload["rows"][0]["time_to_first_confirmed_failure_censored"] = False
    censored_payload["rows"][0]["simulator_seconds_per_confirmed_failure"] = 0.0
    censored_sidecar.write_text(json.dumps(censored_payload), encoding="utf-8")

    censored_gate = validate_package_b_confirmation(report, censored_sidecar)

    assert censored_gate.ready is False
    assert any("must be null when censored" in error for error in censored_gate.errors)
    assert any("must be censored" in error for error in censored_gate.errors)
    assert any("must be null without confirmed failures" in error for error in censored_gate.errors)
