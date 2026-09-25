"""Cold bundle checks for the versioned Social Navigation Quality Index."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.artifact_publication import (
    _check_robot_force_report_consistency,
    _check_snqi_v2_field_consistency,
    _snqi_v2_expected_arm_algorithms,
)

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
            "planners": {
                "keys": ["goal"],
                "config_identities": [{"key": "goal", "algo": "goal"}],
            },
            "kinematics": {"matrix": ["differential_drive"]},
            "metrics": {
                f"snqi_v2_{role}_{field}": value
                for role in paths
                for field, value in (
                    ("path", paths[role]),
                    ("sha256", spec.hashes[role]),
                )
            },
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
        {"algo": "goal", "status": "success", "steps": 10, "metrics": metrics},
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


def test_snqi_v2_arm_mapping_uses_resolved_manifest_identity(tmp_path: Path) -> None:
    """The arm key and its algorithm may differ, and neither comes from a row."""
    payload = tmp_path / "payload"
    _write_json(
        payload / "release" / "release_manifest.resolved.json",
        {
            "planners": {
                "keys": ["hybrid_candidate"],
                "config_identities": [
                    {"key": "hybrid_candidate", "algo": "hybrid_rule_local_planner"}
                ],
            },
            "kinematics": {"matrix": ["differential_drive"]},
        },
    )
    violations: list[str] = []
    assert _snqi_v2_expected_arm_algorithms(payload, violations) == {
        "hybrid_candidate__differential_drive": "hybrid_rule_local_planner"
    }
    assert violations == []


def test_snqi_v2_arm_mapping_rejects_incomplete_manifest(tmp_path: Path) -> None:
    """A missing planner identity cannot be supplied by an episode's self assertion."""
    payload = tmp_path / "payload"
    _write_json(
        payload / "release" / "release_manifest.resolved.json",
        {
            "planners": {"keys": ["goal"], "config_identities": []},
            "kinematics": {"matrix": ["differential_drive"]},
        },
    )
    violations: list[str] = []
    assert _snqi_v2_expected_arm_algorithms(payload, violations) == {}
    assert "do not match" in violations[0]


def test_snqi_v2_cold_recomputation_rejects_row_algorithm_spoof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored score does not authorize a different algorithm for the manifest arm."""
    payload = _v2_payload(tmp_path, monkeypatch)
    path = payload / "runs" / "goal__differential_drive" / "episodes.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["algo"] = "guarded_ppo"
    _write_json(path, row)
    result = _check_snqi_v2_field_consistency(payload)
    assert result["mismatch_count"] == 1
    assert any("algorithm disagrees" in value for value in result["violations"])


def test_snqi_v2_cold_recomputation_binds_guarded_safe_to_manifest_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declared guarded PPO may use its safe shield; a row cannot grant that exception."""
    payload = _v2_payload(tmp_path, monkeypatch)
    manifest_path = payload / "release" / "release_manifest.resolved.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_path = payload / "runs" / "goal__differential_drive" / "episodes.jsonl"
    guarded_path = payload / "runs" / "guarded_ppo__differential_drive" / "episodes.jsonl"
    row = json.loads(run_path.read_text(encoding="utf-8"))
    row["algo"] = "guarded_ppo"
    row["algorithm_metadata"] = {
        "algorithm": "ppo",
        "canonical_algorithm": "guarded_ppo",
        "planner_contract": {"planner_id": "guarded_ppo"},
        "guard_stats": {"fallback_safe": 1},
    }
    guarded_path.parent.mkdir(parents=True)
    _write_json(guarded_path, row)
    run_path.unlink()
    manifest["planners"] = {
        "keys": ["guarded_ppo"],
        "config_identities": [{"key": "guarded_ppo", "algo": "guarded_ppo"}],
    }
    _write_json(manifest_path, manifest)
    assert _check_snqi_v2_field_consistency(payload)["violations"] == []

    _write_json(run_path, {**row, "algo": "goal"})
    guarded_path.unlink()
    manifest["planners"] = {
        "keys": ["goal"],
        "config_identities": [{"key": "goal", "algo": "goal"}],
    }
    _write_json(manifest_path, manifest)
    result = _check_snqi_v2_field_consistency(payload)
    assert result["mismatch_count"] == 1
    assert any("refuses fallback/degraded" in value for value in result["violations"])


def test_snqi_v2_cold_recomputation_rejects_undeclared_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run directory outside the manifest roster cannot enter a signed bundle."""
    payload = _v2_payload(tmp_path, monkeypatch)
    path = payload / "runs" / "goal__differential_drive" / "episodes.jsonl"
    rogue = payload / "runs" / "rogue__differential_drive" / "episodes.jsonl"
    rogue.parent.mkdir(parents=True)
    path.rename(rogue)
    result = _check_snqi_v2_field_consistency(payload)
    assert result["violation_count"] >= 1
    assert any("no declared release arm" in value for value in result["violations"])


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


def test_robot_force_report_cold_check_rejects_resigned_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checking file digests alone must not bless edited correlations."""
    payload = tmp_path / "payload"
    _write_json(
        payload / "release" / "release_manifest.resolved.json",
        {
            "source_sha": "a" * 40,
            "matrix": {"expected_episode_cells": 1},
            "metrics": {"snqi_v2_weights_path": "weights.json"},
        },
    )
    report_path = payload / "reports" / "robot_force_validation.json"
    expected = {"episodes": 1, "correlations": [{"spearman_rho": 0.5}]}
    _write_json(report_path, expected)
    report_path.with_suffix(".md").write_text("# Expected\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.analysis.issue_9668_robot_force_validation.build_report",
        lambda *args, **kwargs: expected,
    )
    monkeypatch.setattr(
        "scripts.analysis.issue_9668_robot_force_validation._render_markdown",
        lambda report: "# Expected\n",
    )
    assert _check_robot_force_report_consistency(payload)["violations"] == []
    _write_json(report_path, {"episodes": 1, "correlations": [{"spearman_rho": 0.9}]})
    assert any(
        "deterministic regeneration" in violation
        for violation in _check_robot_force_report_consistency(payload)["violations"]
    )
    _write_json(report_path, expected)
    _write_json(
        payload / "release" / "release_manifest.resolved.json",
        {
            "source_sha": "a" * 40,
            "matrix": {"expected_episode_cells": 2},
            "metrics": {"snqi_v2_weights_path": "weights.json"},
        },
    )
    assert any(
        "disagrees with resolved release matrix" in violation
        for violation in _check_robot_force_report_consistency(payload)["violations"]
    )
