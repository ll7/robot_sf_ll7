"""Cold bundle checks for the versioned Social Navigation Quality Index."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.artifact_publication import _check_snqi_v2_field_consistency

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _v2_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a small, internally consistent v2 bundle with a frozen synthetic spec."""
    from robot_sf.benchmark.snqi.v2_reports import family_vectors
    from robot_sf.benchmark.snqi.v2_spec import SIMULATED_FORCE, WEIGHTS, SnqiV2Spec

    from robot_sf.benchmark.snqi.compute import compute_snqi_v2, normalize_snqi_v2_terms

    payload = tmp_path / "payload"
    paths = {
        role: f"configs/benchmarks/snqi_v2/{role}.v2.0.{'yaml' if role == 'family' else 'json'}"
        for role in ("weights", "anchors", "family")
    }
    spec = SnqiV2Spec(
        weights=WEIGHTS,
        upper_anchors={"T": 3.0, "N": 0.25, "F": 2.0, "J": 1.0, "K": 1.0},
        force_source=SIMULATED_FORCE,
        calibration_split_id="synthetic-dev-101-102",
        calibration_seeds=(101, 102),
        calibration_rho=0.0,
        paths=paths,
        hashes={role: hashlib.sha256(role.encode()).hexdigest() for role in paths},
    )
    monkeypatch.setattr("robot_sf.benchmark.snqi.v2_spec.load_snqi_v2_spec", lambda *args: spec)
    _write_json(
        payload / "release" / "release_manifest.resolved.json",
        {
            "metrics": {
                f"snqi_v2_{role}_{field}": value
                for role in paths
                for field, value in (
                    ("path", paths[role]),
                    ("sha256", spec.hashes[role]),
                )
            }
        },
    )
    provenance = spec.provenance()
    _write_json(
        payload / "reports" / "snqi_v2_family.json",
        {
            "episode_count": 1,
            "family": "V2-F",
            "stratified_count": 2011,
            "vectors": family_vectors(),
            "provenance": provenance,
        },
    )
    _write_json(
        payload / "reports" / "snqi_v2_diagnostics.json",
        {"episode_count": 1, "family_report": "snqi_v2_family.json", "provenance": provenance},
    )
    metrics = {
        "success": 1.0,
        "total_collision_count": 0,
        "time_to_goal_ideal_ratio": 1.2,
        "near_misses": 1,
        "robot_force_impulse_total": 0.5,
        "jerk_mean": 0.1,
        "curvature_mean": 0.1,
    }
    inputs = {**metrics, "executed_steps": 10}
    metrics["snqi_v2_terms"] = normalize_snqi_v2_terms(inputs, spec)
    metrics["snqi_v2"] = compute_snqi_v2(inputs, spec)
    _write_json(
        payload / "runs" / "goal__differential_drive" / "episodes.jsonl",
        {"steps": 10, "metrics": metrics},
    )
    return payload


def test_snqi_v2_cold_check_skips_legacy_bundle_without_output_change(tmp_path: Path) -> None:
    """The optional v2 checker stays inactive for a v1 bundle."""
    result = _check_snqi_v2_field_consistency(tmp_path / "payload")
    assert result == {"checked": False, "violation_count": 0, "violations": []}


def test_snqi_v2_cold_recomputation_accepts_consistent_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every stored term and score is independently recomputed from cold assets."""
    payload = _v2_payload(tmp_path, monkeypatch)
    result = _check_snqi_v2_field_consistency(payload)
    assert result["checked"] is True
    assert result["rows"] == 1
    assert result["mismatch_count"] == 0
    assert result["violations"] == []


@pytest.mark.parametrize("field", ["snqi_v2", "snqi_v2_terms"])
def test_snqi_v2_cold_recomputation_rejects_stored_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """A rechecksummed bundle cannot silently change a stored v2 term or score."""
    payload = _v2_payload(tmp_path, monkeypatch)
    path = payload / "runs" / "goal__differential_drive" / "episodes.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    if field == "snqi_v2":
        row["metrics"][field] += 0.01
    else:
        row["metrics"][field]["F"] += 0.01
    _write_json(path, row)
    result = _check_snqi_v2_field_consistency(payload)
    assert result["mismatch_count"] == 1
    assert result["violation_count"] >= 1


def test_snqi_v2_cold_recomputation_requires_paired_family_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared index cannot ship without its robustness report."""
    payload = _v2_payload(tmp_path, monkeypatch)
    (payload / "reports" / "snqi_v2_family.json").unlink()
    result = _check_snqi_v2_field_consistency(payload)
    assert any("family report is invalid" in value for value in result["violations"])


def test_snqi_v2_cold_recomputation_rejects_tampered_family_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report must contain the declared sensitivity grid, not only its count."""
    payload = _v2_payload(tmp_path, monkeypatch)
    path = payload / "reports" / "snqi_v2_family.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["vectors"][0]["weights"]["F"] += 0.01
    _write_json(path, report)
    result = _check_snqi_v2_field_consistency(payload)
    assert any("weight-family vector 0" in value for value in result["violations"])
