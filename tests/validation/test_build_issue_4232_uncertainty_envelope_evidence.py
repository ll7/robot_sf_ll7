"""Tests issue #4232 fixture-driven uncertainty-envelope evidence builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/validation/build_issue_4232_uncertainty_envelope_evidence.py"
PACKET = REPO_ROOT / "configs/benchmarks/issue_4232_uncertainty_envelope_claim_packet.yaml"

_SPEC = importlib.util.spec_from_file_location("_issue_4232_evidence_builder", SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _row(  # noqa: PLR0913 - fixture rows need explicit per-field overrides.
    *,
    planner_id: str = "prediction_mpc",
    scenario_id: str = "crossing",
    seed: int = 111,
    alpha_arm_key: str,
    row_status: str = "successful_evidence",
    success_rate: float = 1.0,
    collision_rate: float = 0.0,
    min_clearance_m: float = 1.0,
    runtime_seconds: float = 2.0,
    diagnostics: dict | None = None,
    **extra: object,
) -> dict:
    row = {
        "planner_id": planner_id,
        "scenario_id": scenario_id,
        "scenario_family": "fixture_crossing",
        "seed": seed,
        "alpha_arm_key": alpha_arm_key,
        "row_status": row_status,
        "metrics": {
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "near_miss_rate": 0.0,
            "min_clearance_m": min_clearance_m,
            "path_efficiency": 0.8,
            "runtime_seconds": runtime_seconds,
        },
        "diagnostics": diagnostics or {},
    }
    row.update(extra)
    return row


def _happy_rows() -> list[dict]:
    return [
        _row(alpha_arm_key="envelope_off_alpha_0", min_clearance_m=0.8, runtime_seconds=2.0),
        _row(alpha_arm_key="envelope_on_alpha_0", min_clearance_m=0.8, runtime_seconds=2.1),
        _row(
            alpha_arm_key="envelope_on_alpha_0p10",
            min_clearance_m=1.1,
            runtime_seconds=2.4,
            diagnostics={"envelope_activation_count": 3, "effective_radius_used_by_planner": True},
        ),
        _row(
            seed=112,
            alpha_arm_key="envelope_off_alpha_0",
            min_clearance_m=0.7,
            runtime_seconds=2.0,
        ),
        _row(
            seed=112,
            alpha_arm_key="envelope_on_alpha_0",
            min_clearance_m=0.7,
            runtime_seconds=2.0,
        ),
        _row(
            seed=112,
            alpha_arm_key="envelope_on_alpha_0p10",
            row_status="fallback",
            min_clearance_m=0.7,
            runtime_seconds=4.0,
            diagnostics={"envelope_activation_count": 0, "effective_radius_used_by_planner": False},
        ),
    ]


def _write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def _build(tmp_path: Path, rows: list[dict], *, claim_text: str = "") -> dict:
    return _MODULE.build_evidence(
        packet_path=PACKET,
        results_path=_write_rows(tmp_path, rows),
        output_dir=tmp_path / "evidence",
        claim_text=claim_text,
    )


def test_issue_4232_evidence_builder_writes_compact_tables_and_checksums(tmp_path: Path) -> None:
    """Happy-path fixture writes every review artifact without copying raw outputs."""
    summary = _build(tmp_path, _happy_rows())

    assert summary["ok"] is True
    assert summary["paired_delta_rows"] == 4
    evidence_dir = tmp_path / "evidence"
    expected = {
        "README.md",
        "SHA256SUMS",
        "alpha_arm_metric_table.csv",
        "claim_boundary.md",
        "claim_readiness.md",
        "envelope_activation_diagnostics.json",
        "envelope_activation_diagnostics.md",
        "metadata.json",
        "paired_alpha_delta_table.csv",
        "pre_registration_packet.json",
        "row_status_audit.csv",
        "runtime_cost_report.csv",
    }
    assert {path.name for path in evidence_dir.iterdir()} == expected
    checksums = (evidence_dir / "SHA256SUMS").read_text(encoding="utf-8")
    assert "raw" not in checksums.lower()
    metrics = (evidence_dir / "alpha_arm_metric_table.csv").read_text(encoding="utf-8")
    assert "benchmark_evidence" in metrics
    assert "fallback,excluded" in metrics
    activation = json.loads((evidence_dir / "envelope_activation_diagnostics.json").read_text())
    assert activation["status_counts"]["activated"] == 1
    assert activation["status_counts"]["no_mechanism_activation"] == 1


def test_issue_4232_evidence_builder_fails_without_alpha_zero_baseline(tmp_path: Path) -> None:
    """Evidence cannot be summarized without the envelope-off alpha-zero baseline."""
    rows = [row for row in _happy_rows() if row["alpha_arm_key"] != "envelope_off_alpha_0"]
    with pytest.raises(_MODULE.EvidenceBuildError, match="missing alpha-zero baseline"):
        _build(tmp_path, rows)


def test_issue_4232_evidence_builder_keeps_fallback_rows_excluded(tmp_path: Path) -> None:
    """Fallback/degraded rows remain visible but cannot become benchmark-strength rows."""
    _build(tmp_path, _happy_rows())

    audit = (tmp_path / "evidence" / "row_status_audit.csv").read_text(encoding="utf-8")
    assert "fallback" in audit
    deltas = (tmp_path / "evidence" / "paired_alpha_delta_table.csv").read_text(encoding="utf-8")
    assert "row_status_excluded_from_benchmark_strength" in deltas


def test_issue_4232_evidence_builder_marks_unpaired_seeds_not_comparable(tmp_path: Path) -> None:
    """Unpaired nonzero-alpha fixture rows are retained as not comparable."""
    rows = _happy_rows()
    rows.append(
        _row(
            seed=113,
            alpha_arm_key="envelope_on_alpha_0p10",
            diagnostics={"envelope_activation_count": 2, "effective_radius_used_by_planner": True},
        )
    )
    _build(tmp_path, rows)

    deltas = (tmp_path / "evidence" / "paired_alpha_delta_table.csv").read_text(encoding="utf-8")
    assert "missing_envelope_off_alpha_0_for_seed" in deltas


def test_issue_4232_evidence_builder_fails_without_activation_diagnostics(tmp_path: Path) -> None:
    """Nonzero-alpha rows must say whether the envelope mechanism activated."""
    rows = _happy_rows()
    rows[2]["diagnostics"] = {}
    with pytest.raises(_MODULE.EvidenceBuildError, match="missing envelope activation diagnostics"):
        _build(tmp_path, rows)


def test_issue_4232_evidence_builder_rejects_raw_artifact_references(tmp_path: Path) -> None:
    """Builder refuses raw JSONL, video, log, checkpoint, or cache artifact references."""
    rows = _happy_rows()
    rows[0]["raw_artifact_paths"] = ["output/issue_4232/episodes.jsonl"]
    with pytest.raises(_MODULE.EvidenceBuildError, match="raw artifact references"):
        _build(tmp_path, rows)


def test_issue_4232_evidence_builder_rejects_forbidden_claim_language(tmp_path: Path) -> None:
    """Fixture summaries cannot smuggle conformal or safety claims into readiness output."""
    with pytest.raises(_MODULE.EvidenceBuildError, match="forbidden claim language"):
        _build(tmp_path, _happy_rows(), claim_text="This proves a conformal coverage guarantee.")
