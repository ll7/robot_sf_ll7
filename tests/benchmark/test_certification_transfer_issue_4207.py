"""Tests for issue #4207 certification-transfer probe helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.certification_transfer import (
    CLAIM_BOUNDARY,
    build_certification_transfer_report,
    validate_probe_config,
    write_certification_transfer_evidence,
)
from robot_sf.sim.pedestrian_model_variants import HSFM_TOTAL_FORCE_V1, SOCIAL_FORCE_DEFAULT


def test_transfer_matrix_detects_pass_fail_flips(tmp_path: Path) -> None:
    """A 2x2 fixture emits all transfer statuses deterministically."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    records = [
        _record("stable", SOCIAL_FORCE_DEFAULT, collision_rate=0.0),
        _record("stable", HSFM_TOTAL_FORCE_V1, collision_rate=0.0),
        _record("fragile", SOCIAL_FORCE_DEFAULT, collision_rate=0.0),
        _record("fragile", HSFM_TOTAL_FORCE_V1, collision_rate=1.0),
        _record("conservative", SOCIAL_FORCE_DEFAULT, collision_rate=1.0),
        _record("conservative", HSFM_TOTAL_FORCE_V1, collision_rate=0.0),
        _record("blocked", SOCIAL_FORCE_DEFAULT, collision_rate=0.0, include_required=False),
        _record("blocked", HSFM_TOTAL_FORCE_V1, collision_rate=0.0),
    ]

    report = build_certification_transfer_report(
        records,
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
        generated_at_utc="2026-07-03T00:00:00+00:00",
    )

    rows = report["certification_transfer_matrix"]
    assert len(rows) == 16
    statuses = {row["transfer_status"] for row in rows}
    assert {
        "stable_pass",
        "fragile_pass_to_fail",
        "conservative_fail_to_pass",
        "stable_fail",
        "not_evaluable",
    }.issubset(statuses)
    assert report["flip_cases"]
    assert report["claim_boundary"] == CLAIM_BOUNDARY


def test_missing_gate_metrics_are_not_evaluable_not_pass(tmp_path: Path) -> None:
    """Required missing gate metrics fail closed as not_evaluable."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    report = build_certification_transfer_report(
        [_record("stable", SOCIAL_FORCE_DEFAULT, include_required=False)],
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )

    cell = next(
        row
        for row in report["gate_cells"]
        if row["planner_key"] == "stable" and row["evaluation_model"] == SOCIAL_FORCE_DEFAULT
    )
    assert cell["gate_status"] == "not_evaluable"
    assert "near_miss_rate_limit" in cell["not_evaluable_gate_ids"]


def test_unsupported_pedestrian_model_fails_closed(tmp_path: Path) -> None:
    """Probe config rejects unsupported or undeclared pedestrian-model labels."""

    _config_path, _gate_path, config, _gates = _write_config_pair(tmp_path)
    config["pedestrian_models"] = [SOCIAL_FORCE_DEFAULT, "bogus_model"]
    with pytest.raises(ValueError, match="Unsupported pedestrian_model"):
        validate_probe_config(config, base_dir=tmp_path)


def test_arm_algo_configs_must_resolve(tmp_path: Path) -> None:
    """Planner arm config paths must exist before execution."""

    _config_path, _gate_path, config, _gates = _write_config_pair(tmp_path)
    config["arms"][0]["algo_config"] = "missing.yaml"
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        validate_probe_config(config, base_dir=tmp_path)


def test_provenance_separates_certification_evaluation_and_development(tmp_path: Path) -> None:
    """Certification model is separate from declared policy development provenance."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    config["arms"][0]["development_pedestrian_model"] = "training_manifest_declared_sfm"
    report = build_certification_transfer_report(
        [_record("stable", HSFM_TOTAL_FORCE_V1, collision_rate=0.0)],
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )

    cell = next(
        row
        for row in report["gate_cells"]
        if row["planner_key"] == "stable" and row["evaluation_model"] == HSFM_TOTAL_FORCE_V1
    )
    assert cell["certification_pedestrian_model"] == HSFM_TOTAL_FORCE_V1
    assert cell["evaluation_model"] == HSFM_TOTAL_FORCE_V1
    assert cell["development_pedestrian_model"] == "training_manifest_declared_sfm"


def test_evidence_writer_emits_checksums_without_raw_artifacts(tmp_path: Path) -> None:
    """Evidence writer emits compact report files and excludes raw logs/videos/JSONL."""

    config_path, gate_path, config, gates = _write_config_pair(tmp_path)
    report = build_certification_transfer_report(
        [_record("stable", SOCIAL_FORCE_DEFAULT, collision_rate=0.0)],
        probe_config=config,
        gate_spec=gates,
        config_path=config_path,
        gate_spec_path=gate_path,
    )
    paths = write_certification_transfer_evidence(report, tmp_path / "evidence")

    assert Path(paths["sha256sums"]).exists()
    written_names = {path.name for path in (tmp_path / "evidence").iterdir()}
    assert "summary.json" in written_names
    assert "certification_transfer_matrix.csv" in written_names
    assert not any(
        path.suffix in {".jsonl", ".log", ".mp4"} for path in (tmp_path / "evidence").iterdir()
    )
    loaded = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert loaded["issue"] == 4207
    with Path(paths["certification_transfer_matrix_csv"]).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows


def _write_config_pair(tmp_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    algo_path = tmp_path / "algo.yaml"
    algo_path.write_text("{}\n", encoding="utf-8")
    config: dict[str, object] = {
        "name": "issue_4207_certification_transfer_probe",
        "schema_version": "certification-transfer-probe.v1",
        "issue": 4207,
        "paper_facing": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "pedestrian_models": [SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1],
        "scenario_family": "fixture_family",
        "scenario_matrix": str(scenario_path),
        "seed_policy": {"mode": "fixed-list", "seeds": [111, 112, 113]},
        "arms": [
            {
                "key": "stable",
                "structural_class": "fixture",
                "algo": "goal",
                "algo_config": str(algo_path),
                "development_pedestrian_model": "unknown",
            },
            {"key": "fragile", "structural_class": "fixture", "algo": "goal"},
            {"key": "conservative", "structural_class": "fixture", "algo": "goal"},
            {"key": "blocked", "structural_class": "fixture", "algo": "goal"},
        ],
    }
    gates: dict[str, object] = {
        "schema_version": "benchmark_release_gate_spec.v1",
        "gates": [
            {
                "id": "collision_rate_zero",
                "metric": "collision_rate",
                "threshold": 0.0,
                "direction": "max",
                "category": "safety",
                "provenance": "fixture",
                "required": True,
                "scope": {"scenario_family": "fixture_family"},
            },
            {
                "id": "near_miss_rate_limit",
                "metric": "near_miss_rate",
                "threshold": 0.05,
                "direction": "max",
                "category": "safety",
                "provenance": "fixture",
                "required": True,
                "scope": {"scenario_family": "fixture_family"},
            },
        ],
    }
    config_path = tmp_path / "config.yaml"
    gate_path = tmp_path / "gates.yaml"
    config_path.write_text("fixture\n", encoding="utf-8")
    gate_path.write_text("fixture\n", encoding="utf-8")
    return config_path, gate_path, config, gates


def _record(
    planner_key: str,
    evaluation_model: str,
    *,
    collision_rate: float = 0.0,
    include_required: bool = True,
) -> dict[str, object]:
    metrics = {"collision_rate": collision_rate}
    if include_required:
        metrics["near_miss_rate"] = 0.0
    return {
        "planner_key": planner_key,
        "scenario_family": "fixture_family",
        "evaluation_pedestrian_model": evaluation_model,
        "certification_pedestrian_model": evaluation_model,
        "development_pedestrian_model": "unknown",
        "metrics": metrics,
    }
