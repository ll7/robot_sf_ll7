"""Tests for the pre-planner adversarial batch-certification gate (issue #2920)."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_sf.adversarial.batch_certification import (
    ADVERSARIAL_CANDIDATE_QUALITY_SCHEMA,
    BatchCertificationPolicy,
    certify_candidate_batch,
    certify_records,
)
from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.manifest_quality import ManifestQualityRecord
from robot_sf.adversarial.scenario_manifest import compute_control_hash


def _record(status: str, *, path: str, control_hash: str | None = None) -> ManifestQualityRecord:
    return ManifestQualityRecord(
        path=path,
        status=status,
        schema_version="adversarial_scenario_manifest.v1",
        normalized_control_hash=control_hash,
    )


# --- pure gate (certify_records) -----------------------------------------


def test_rejects_invalid_and_degenerate() -> None:
    """Invalid and degenerate candidates are rejected; valid ones pass."""
    records = [
        _record("valid", path="a", control_hash="h_a"),
        _record("invalid", path="b", control_hash="h_b"),
        _record("degenerate", path="c", control_hash="h_c"),
    ]
    result = certify_records(records)
    decisions = {c.path: c.accepted for c in result.candidates}
    assert decisions == {"a": True, "b": False, "c": False}
    assert result.accepted_count == 1
    assert result.rejected_count == 2
    assert result.rejection_counts == {"invalid": 1, "degenerate": 1}
    assert result.validity_rate == 1 / 3
    assert result.accepted is True  # at least one usable candidate


def test_rejects_intra_batch_duplicates() -> None:
    """The first occurrence of a control hash is kept; later repeats are rejected."""
    records = [
        _record("valid", path="first", control_hash="dup"),
        _record("valid", path="second", control_hash="dup"),
        _record("valid", path="unique", control_hash="other"),
    ]
    result = certify_records(records)
    by_path = {c.path: c for c in result.candidates}
    assert by_path["first"].accepted is True
    assert by_path["second"].accepted is False
    assert "duplicate" in by_path["second"].reasons
    assert by_path["unique"].accepted is True
    assert result.rejection_counts == {"duplicate": 1}


def test_combined_reasons_degenerate_and_duplicate() -> None:
    """A candidate can be rejected for multiple reasons at once."""
    records = [
        _record("valid", path="seed", control_hash="dup"),
        _record("degenerate", path="bad", control_hash="dup"),
    ]
    result = certify_records(records)
    bad = next(c for c in result.candidates if c.path == "bad")
    assert set(bad.reasons) == {"degenerate", "duplicate"}
    assert bad.accepted is False


def test_policy_can_keep_duplicates() -> None:
    """Disabling duplicate rejection accepts repeated control hashes."""
    records = [
        _record("valid", path="first", control_hash="dup"),
        _record("valid", path="second", control_hash="dup"),
    ]
    result = certify_records(records, BatchCertificationPolicy(reject_duplicates=False))
    assert result.accepted_count == 2
    assert result.rejection_counts == {}


def test_batch_validity_gate_rejects_low_quality_batch() -> None:
    """A batch below the minimum validity rate is rejected even with some valid rows."""
    records = [
        _record("valid", path="a", control_hash="h_a"),
        _record("invalid", path="b", control_hash="h_b"),
        _record("invalid", path="c", control_hash="h_c"),
    ]
    policy = BatchCertificationPolicy(min_batch_validity_rate=0.5)
    result = certify_records(records, policy)
    assert result.validity_rate == 1 / 3
    assert result.accepted is False  # below the 0.5 gate
    # Individual valid candidate still passes its own certification.
    assert next(c for c in result.candidates if c.path == "a").accepted is True


def test_empty_batch_is_not_accepted() -> None:
    """An empty batch produces a well-formed, non-accepted result."""
    result = certify_records([])
    assert result.total == 0
    assert result.accepted is False
    assert result.validity_rate == 0.0


def test_to_dict_emits_candidate_quality_v1() -> None:
    """The payload uses the adversarial_candidate_quality.v1 schema."""
    result = certify_records([_record("valid", path="a", control_hash="h")])
    payload = result.to_dict()
    assert payload["schema_version"] == ADVERSARIAL_CANDIDATE_QUALITY_SCHEMA
    assert "evidence_boundary" in payload
    assert payload["total"] == 1
    assert payload["candidates"][0]["path"] == "a"


# --- end-to-end path entry -----------------------------------------------


def _controls(start_x: float, scenario_seed: int = 7) -> dict:
    return {
        "start": {"x": float(start_x), "y": 0.0},
        "goal": {"x": 10.0, "y": 0.0},
        "spawn_time_s": 0.0,
        "pedestrian_speed_mps": 1.0,
        "pedestrian_delay_s": 0.0,
        "scenario_seed": scenario_seed,
    }


def _write_manifest(path: Path, controls: dict, status: str) -> None:
    candidate = CandidateSpec(
        start=Pose2D(controls["start"]["x"], controls["start"]["y"]),
        goal=Pose2D(controls["goal"]["x"], controls["goal"]["y"]),
        spawn_time_s=float(controls["spawn_time_s"]),
        pedestrian_speed_mps=float(controls["pedestrian_speed_mps"]),
        pedestrian_delay_s=float(controls["pedestrian_delay_s"]),
        scenario_seed=int(controls["scenario_seed"]),
    )
    payload = {
        "schema_version": "adversarial_scenario_manifest.v1",
        "candidate_controls": controls,
        "validation": {
            "status": status,
            "errors": [],
            "warnings": [],
            "normalized_control_hash": compute_control_hash(candidate),
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_certify_candidate_batch_end_to_end(tmp_path: Path) -> None:
    """Loading real manifests and certifying them rejects bad/duplicate candidates."""
    _write_manifest(tmp_path / "a.yaml", _controls(0.0), "valid")
    _write_manifest(tmp_path / "b.yaml", _controls(1.0), "invalid")
    _write_manifest(tmp_path / "c.yaml", _controls(2.0), "degenerate")
    _write_manifest(tmp_path / "d.yaml", _controls(0.0), "valid")  # duplicate of a.yaml

    result = certify_candidate_batch([tmp_path])

    assert result.total == 4
    # One unique valid candidate is accepted; invalid + degenerate + duplicate rejected.
    assert result.accepted_count == 1
    assert result.rejection_counts.get("invalid") == 1
    assert result.rejection_counts.get("degenerate") == 1
    assert result.rejection_counts.get("duplicate") == 1
    # The quality summary is embedded for provenance.
    assert result.quality_summary is not None
    assert result.to_dict()["quality_summary"]["manifest_count"] == 4


def test_certify_with_reference_manifest_computes_perturbation(tmp_path: Path) -> None:
    """Passing a reference manifest exercises the perturbation-distance provenance path."""
    reference = tmp_path / "reference.yaml"
    _write_manifest(reference, _controls(0.0), "valid")
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_manifest(batch_dir / "near.yaml", _controls(0.1), "valid")
    _write_manifest(batch_dir / "far.yaml", _controls(5.0), "valid")

    result = certify_candidate_batch([batch_dir], reference_manifest=reference)

    assert result.total == 2
    assert result.accepted is True
    assert result.quality_summary is not None
    # Perturbation distances are computed relative to the reference candidate.
    assert result.quality_summary.perturbation_count >= 1
    summary_payload = result.to_dict()["quality_summary"]
    assert "perturbation_from_reference" in summary_payload


def test_certify_records_without_quality_summary(tmp_path: Path) -> None:
    """The path entry can skip the embedded quality summary."""
    _write_manifest(tmp_path / "a.yaml", _controls(0.0), "valid")
    result = certify_candidate_batch([tmp_path], include_quality_summary=False)
    assert result.quality_summary is None
    assert "quality_summary" not in result.to_dict()
