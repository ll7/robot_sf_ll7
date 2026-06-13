"""Tests for the learned-prediction readiness validator stub."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.validation.validate_learned_prediction_readiness import (
    main,
    validate_readiness,
)

if TYPE_CHECKING:
    from pathlib import Path


def _readiness_doc_content() -> str:
    """Return a minimal readiness document with all required sections."""
    return """\
# Learned Prediction Readiness Contract

## Trace Dataset Registry

Required dataset registry with source metadata.

## Train / Validation / Test Split Metadata

Split strategy and leakage prevention.

## Target Horizon Definition

- horizon_seconds: 3.0
- horizon_steps: 30
- dt_seconds: 0.1

## Dynamic Actor Types

Supported actor types for prediction.

## Semantic Input Contract

Required and optional semantic fields.

## Calibration Metrics

ADE, FDE, MR, ECE metrics.

## Collision-Relevance Metrics

Collision proxy rate and TTC error.

## Deterministic / Semantic Baselines

Constant velocity and semantic baselines.

## Comparison Protocol

Same test split, seeds, and horizon.

## Training Block Conditions

Conditions that must be satisfied before training.
"""


def _trace_registry_content() -> str:
    """Return a valid trace registry with one viable source."""
    return yaml.safe_dump(
        [
            {
                "source_id": "sim_classic_v1",
                "source_type": "simulation",
                "episode_count": 500,
                "actor_types": ["pedestrian"],
                "horizon_seconds": 5.0,
                "frame_rate_hz": 10,
                "semantic_inputs": ["map_geometry"],
                "provenance": "commit:abc123",
                "license_or_access": "local",
            }
        ]
    )


def _split_manifest_content() -> str:
    """Return a valid split manifest."""
    return yaml.safe_dump(
        {
            "split_strategy": "episode_partition",
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "leakage_prevention": "Episodes are partitioned by seed hash; no seed appears in multiple splits.",
        }
    )


def _baseline_evidence_content() -> str:
    """Return valid baseline evidence with constant_velocity metrics."""
    return yaml.safe_dump(
        {
            "baselines": {
                "constant_velocity": {
                    "ade": 0.45,
                    "fde": 0.82,
                    "mr": 0.12,
                }
            }
        }
    )


def test_blocked_when_readiness_doc_missing(tmp_path: Path) -> None:
    """Validator should fail when the readiness doc does not exist."""
    report = validate_readiness(doc_path=tmp_path / "missing.md")
    assert report["status"] == "blocked"
    assert any("readiness doc not found" in e for e in report["errors"])


def test_blocked_when_sections_missing(tmp_path: Path) -> None:
    """Validator should fail when required sections are absent."""
    doc = tmp_path / "partial.md"
    doc.write_text("# Incomplete\n", encoding="utf-8")
    report = validate_readiness(doc_path=doc)
    assert report["status"] == "blocked"
    assert any("missing required section" in e for e in report["errors"])


def test_blocked_when_trace_registry_missing(tmp_path: Path) -> None:
    """Validator should fail when trace registry is referenced but absent."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    report = validate_readiness(
        doc_path=doc,
        registry_path=tmp_path / "missing_registry.yaml",
    )
    assert report["status"] == "blocked"
    assert any("trace registry not found" in e for e in report["errors"])


def test_blocked_when_trace_registry_has_no_sources(tmp_path: Path) -> None:
    """Validator should report an empty registry instead of failing open."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"sources": []}), encoding="utf-8")
    report = validate_readiness(doc_path=doc, registry_path=registry)
    assert report["status"] == "blocked"
    assert "trace registry has no source entries" in report["errors"]


def test_blocked_when_trace_registry_shape_invalid(tmp_path: Path) -> None:
    """Validator should fail closed on malformed trace registry YAML."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump("not-a-registry"), encoding="utf-8")
    report = validate_readiness(doc_path=doc, registry_path=registry)
    assert report["status"] == "blocked"
    assert "trace registry must be a dictionary or a list" in report["errors"]


def test_blocked_when_prerequisite_paths_not_provided(tmp_path: Path) -> None:
    """Validator should fail closed when prerequisite artifact paths are omitted."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    report = validate_readiness(doc_path=doc)
    assert report["status"] == "blocked"
    assert "trace registry path not provided" in report["errors"]
    assert "split manifest path not provided" in report["errors"]
    assert "baseline evidence path not provided" in report["errors"]


def test_blocked_when_split_manifest_missing(tmp_path: Path) -> None:
    """Validator should fail when split manifest is referenced but absent."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    report = validate_readiness(
        doc_path=doc,
        split_manifest_path=tmp_path / "missing_split.yaml",
    )
    assert report["status"] == "blocked"
    assert any("split manifest not found" in e for e in report["errors"])


def test_blocked_when_baseline_evidence_missing(tmp_path: Path) -> None:
    """Validator should fail when baseline evidence is referenced but absent."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    report = validate_readiness(
        doc_path=doc,
        baseline_path=tmp_path / "missing_baseline.yaml",
    )
    assert report["status"] == "blocked"
    assert any("baseline evidence not found" in e for e in report["errors"])


def test_blocked_when_split_fractions_do_not_sum_to_one(tmp_path: Path) -> None:
    """Validator should fail when split fractions do not sum to 1.0."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {
                "split_strategy": "episode_partition",
                "train_fraction": 0.5,
                "validation_fraction": 0.2,
                "test_fraction": 0.1,
                "leakage_prevention": "none",
            }
        ),
        encoding="utf-8",
    )
    report = validate_readiness(doc_path=doc, split_manifest_path=split)
    assert report["status"] == "blocked"
    assert any("sum to" in e for e in report["errors"])


def test_blocked_when_split_manifest_shape_invalid(tmp_path: Path) -> None:
    """Validator should fail closed on malformed split manifest YAML."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")
    report = validate_readiness(doc_path=doc, split_manifest_path=split)
    assert report["status"] == "blocked"
    assert "split manifest must be a dictionary" in report["errors"]


def test_blocked_when_split_fractions_are_not_numeric(tmp_path: Path) -> None:
    """Validator should fail closed before adding nonnumeric split fractions."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {
                "split_strategy": "episode_partition",
                "train_fraction": "0.7",
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "leakage_prevention": "seed hash split",
            }
        ),
        encoding="utf-8",
    )
    report = validate_readiness(doc_path=doc, split_manifest_path=split)
    assert report["status"] == "blocked"
    assert "split fractions must be numeric" in report["errors"]


def test_blocked_when_baseline_missing_cv_entry(tmp_path: Path) -> None:
    """Validator should fail when constant_velocity baseline is absent."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        yaml.safe_dump({"baselines": {"zero_velocity": {"ade": 1.0, "fde": 1.0}}}),
        encoding="utf-8",
    )
    report = validate_readiness(doc_path=doc, baseline_path=baseline)
    assert report["status"] == "blocked"
    assert any("missing constant_velocity" in e for e in report["errors"])


def test_blocked_when_baseline_shape_invalid(tmp_path: Path) -> None:
    """Validator should fail closed on malformed baseline evidence YAML."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(yaml.safe_dump("not-a-baseline"), encoding="utf-8")
    report = validate_readiness(doc_path=doc, baseline_path=baseline)
    assert report["status"] == "blocked"
    assert "baseline evidence must be a dictionary or a list" in report["errors"]


def test_blocked_when_baseline_missing_ade(tmp_path: Path) -> None:
    """Validator should fail when constant_velocity baseline lacks ADE."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        yaml.safe_dump({"baselines": {"constant_velocity": {"fde": 0.82}}}),
        encoding="utf-8",
    )
    report = validate_readiness(doc_path=doc, baseline_path=baseline)
    assert report["status"] == "blocked"
    assert any("missing ADE" in e for e in report["errors"])


def test_ready_accepts_uppercase_baseline_metric_keys(tmp_path: Path) -> None:
    """Validator should accept ADE/FDE metric keys case-insensitively."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(_trace_registry_content(), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split.write_text(_split_manifest_content(), encoding="utf-8")
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        yaml.safe_dump({"baselines": {"constant_velocity": {"ADE": 0.45, "FDE": 0.82}}}),
        encoding="utf-8",
    )

    report = validate_readiness(
        doc_path=doc,
        registry_path=registry,
        split_manifest_path=split,
        baseline_path=baseline,
    )
    assert report["status"] == "ready"


def test_ready_when_all_checks_pass(tmp_path: Path) -> None:
    """Validator should pass when all prerequisites are satisfied."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(_trace_registry_content(), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split.write_text(_split_manifest_content(), encoding="utf-8")
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(_baseline_evidence_content(), encoding="utf-8")

    report = validate_readiness(
        doc_path=doc,
        registry_path=registry,
        split_manifest_path=split,
        baseline_path=baseline,
    )
    assert report["status"] == "ready"
    assert report["errors"] == []


def test_cli_exits_zero_when_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should exit 0 when all checks pass."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(_trace_registry_content(), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split.write_text(_split_manifest_content(), encoding="utf-8")
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(_baseline_evidence_content(), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_learned_prediction_readiness.py",
            "--readiness-doc",
            str(doc),
            "--trace-registry",
            str(registry),
            "--split-manifest",
            str(split),
            "--baseline-evidence",
            str(baseline),
            "--json",
        ],
    )
    exit_code = main()
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ready"


def test_cli_resolves_relative_paths_against_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should resolve relative input paths against --repo-root."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(_trace_registry_content(), encoding="utf-8")
    split = tmp_path / "split.yaml"
    split.write_text(_split_manifest_content(), encoding="utf-8")
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(_baseline_evidence_content(), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_learned_prediction_readiness.py",
            "--repo-root",
            str(tmp_path),
            "--readiness-doc",
            "readiness.md",
            "--trace-registry",
            "registry.yaml",
            "--split-manifest",
            "split.yaml",
            "--baseline-evidence",
            "baseline.yaml",
            "--json",
        ],
    )
    exit_code = main()
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ready"


def test_cli_exits_two_when_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should exit 2 when prerequisites are missing."""
    doc = tmp_path / "readiness.md"
    doc.write_text("# Incomplete\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_learned_prediction_readiness.py",
            "--readiness-doc",
            str(doc),
            "--json",
        ],
    )
    exit_code = main()
    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked"


def test_cli_text_output_when_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should print BLOCKED with error lines in text mode."""
    doc = tmp_path / "readiness.md"
    doc.write_text("# Incomplete\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_learned_prediction_readiness.py",
            "--readiness-doc",
            str(doc),
        ],
    )
    exit_code = main()
    assert exit_code == 2
    text = capsys.readouterr().out
    assert "BLOCKED" in text
    assert "missing required section" in text
