"""Tests for the learned-prediction readiness validator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.validation.validate_learned_prediction_readiness import (
    DEFAULT_BASELINE_TARGET,
    FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION,
    FORECAST_DATASET_SCHEMA_VERSION,
    FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION,
    main,
    validate_readiness,
)

if TYPE_CHECKING:
    from pathlib import Path


def _readiness_doc_content() -> str:
    """Return a minimal readiness document with required sections and keys."""
    return """\
# Learned Prediction Readiness Contract

## Trace Dataset Registry

## Train / Validation / Test Split Metadata

## Target Horizon Definition

- horizon_seconds: 3.0
- horizon_steps: 30
- dt_seconds: 0.1
- horizon_recommendation: 3s for planner experiments
- timestep_recommendation: 0.1s for prediction rollout

## Dynamic Actor Types

## Semantic Input Contract

## Calibration Metrics

## Collision-Relevance Metrics

## Deterministic / Semantic Baselines

## Comparison Protocol

## Training Block Conditions
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
    """Return a valid legacy-style split manifest."""
    return yaml.safe_dump(
        {
            "split_strategy": "episode_partition",
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "leakage_prevention": "Episode-level partition by seed hash.",
        }
    )


def _dataset_manifest_content() -> str:
    """Return a minimal forecast_dataset.v1-style manifest."""
    return json.dumps(
        {
            "schema_version": FORECAST_DATASET_SCHEMA_VERSION,
            "dataset_id": "unit_test_dataset",
            "example_count": 10,
            "examples_path": "examples.jsonl",
            "splits": {
                "train": {"example_count": 6},
                "validation": {"example_count": 2},
                "test": {"example_count": 2},
            },
            "split_policy": {
                "strategy": "trace_partition",
                "leakage_prevention": ["scenario_seed_keys"],
                "split_names": ["train", "validation", "test"],
            },
            "feature_schema": {"position": {"type": "float"}},
        },
        sort_keys=True,
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


def _calibration_report_content(
    decision: str = "continue", claim_status: str = "analysis-only"
) -> str:
    """Build a minimal calibration report payload."""
    return json.dumps(
        {
            "schema_version": FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION,
            "recommendation": {
                "decision": decision,
                "claim_status": claim_status,
            },
        },
        sort_keys=True,
    )


def _transferability_report_content(
    decision: str = "continue",
    claim_status: str = "benchmark-eligible",
) -> str:
    """Build a minimal transferability matrix with oracle + deployable rows."""
    return json.dumps(
        {
            "schema_version": FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION,
            "matrix_rows": [
                {"observation_tier": "oracle_full_state"},
                {"observation_tier": "deployable_vision"},
            ],
            "recommendation": {
                "decision": decision,
                "claim_status": claim_status,
            },
        },
        sort_keys=True,
    )


def _closed_loop_gate_content(decision: str = "continue") -> str:
    """Build a minimal closed-loop gate artifact."""
    return json.dumps({"recommendation": decision}, sort_keys=True)


def _write_validation_bundle(tmp_path: Path) -> dict[str, Path]:
    """Write full pass-set artifacts for validator success."""
    doc = tmp_path / "readiness.md"
    registry = tmp_path / "registry.yaml"
    split = tmp_path / "split.yaml"
    baseline = tmp_path / "baseline.yaml"
    calibration = tmp_path / "calibration.json"
    transferability = tmp_path / "transferability.json"
    closed_loop = tmp_path / "closed_loop_gate.json"

    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    registry.write_text(_trace_registry_content(), encoding="utf-8")
    split.write_text(_split_manifest_content(), encoding="utf-8")
    baseline.write_text(_baseline_evidence_content(), encoding="utf-8")
    calibration.write_text(_calibration_report_content(), encoding="utf-8")
    transferability.write_text(_transferability_report_content(), encoding="utf-8")
    closed_loop.write_text(_closed_loop_gate_content(), encoding="utf-8")

    return {
        "doc": doc,
        "registry": registry,
        "split": split,
        "baseline": baseline,
        "calibration": calibration,
        "transferability": transferability,
        "closed_loop": closed_loop,
    }


def test_blocked_when_readiness_doc_missing(tmp_path: Path) -> None:
    """Validator should fail when the readiness doc does not exist."""
    report = validate_readiness(doc_path=tmp_path / "missing.md")
    assert report["status"] == "blocked"
    assert report["prerequisites"]["readiness_doc"]["status"] == "blocked"
    assert any("readiness doc not found" in e for e in report["errors"])


def test_blocked_when_sections_missing(tmp_path: Path) -> None:
    """Validator should fail when required sections are absent."""
    doc = tmp_path / "partial.md"
    doc.write_text("# Incomplete\n", encoding="utf-8")
    report = validate_readiness(doc_path=doc)
    assert report["status"] == "blocked"
    assert report["prerequisites"]["readiness_doc"]["status"] == "failed"
    assert any("missing required section" in e for e in report["errors"])


def test_blocked_when_prerequisite_paths_not_provided(tmp_path: Path) -> None:
    """Validator should fail closed when prereq artifact paths are omitted."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    report = validate_readiness(doc_path=doc)
    assert report["status"] == "blocked"
    assert report["prerequisites"]["trace_registry"]["status"] == "blocked"
    assert report["prerequisites"]["split_manifest"]["status"] == "blocked"
    assert report["prerequisites"]["baseline_evidence"]["status"] == "blocked"
    assert report["prerequisites"]["calibration_report"]["status"] == "blocked"
    assert report["prerequisites"]["transferability_split"]["status"] == "blocked"
    assert report["prerequisites"]["closed_loop_gate"]["status"] == "blocked"
    for key in (
        "trace registry path not provided",
        "split manifest path not provided",
        "baseline evidence path not provided",
        "calibration report path not provided",
        "transferability report path not provided",
        "closed-loop gate path not provided",
    ):
        assert key in report["errors"]
    assert [blocker["name"] for blocker in report["blocking_prerequisites"]] == [
        "trace_registry",
        "split_manifest",
        "baseline_evidence",
        "calibration_report",
        "transferability_split",
        "closed_loop_gate",
    ]


def test_ready_report_has_no_blocking_prerequisites(tmp_path: Path) -> None:
    """Ready report exposes no remaining training blockers."""
    bundle = _write_validation_bundle(tmp_path)
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "ready"
    assert report["blocking_prerequisites"] == []


def test_blocked_when_trace_registry_has_no_sources(tmp_path: Path) -> None:
    """Validator should report an empty registry instead of failing open."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["registry"].write_text(yaml.safe_dump({"sources": []}), encoding="utf-8")
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert report["prerequisites"]["trace_registry"]["status"] == "failed"
    assert "trace registry has no source entries" in report["errors"]


def test_blocked_when_trace_registry_shape_invalid(tmp_path: Path) -> None:
    """Validator should fail closed on malformed trace registry YAML."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["registry"].write_text(yaml.safe_dump("not-a-registry"), encoding="utf-8")
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert "trace registry must be a dictionary or a list" in report["errors"]


def test_blocked_when_split_manifest_missing(tmp_path: Path) -> None:
    """Validator should fail when split manifest is referenced but absent."""
    bundle = _write_validation_bundle(tmp_path)
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=tmp_path / "missing_split.yaml",
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert report["prerequisites"]["split_manifest"]["status"] == "blocked"
    assert any("split manifest not found" in e for e in report["errors"])


def test_blocked_when_split_manifest_fails_forecast_schema(tmp_path: Path) -> None:
    """Validator should reject ForecastDataset manifests missing leakage prevention."""
    bundle = _write_validation_bundle(tmp_path)
    manifest = {
        "schema_version": FORECAST_DATASET_SCHEMA_VERSION,
        "dataset_id": "unit",
        "example_count": 1,
        "examples_path": "examples.jsonl",
        "splits": {"train": {}, "validation": {}, "test": {}},
        "split_policy": {
            "strategy": "trace_partition",
            "split_names": ["train", "validation", "test"],
        },
        "feature_schema": {"position": {"type": "float"}},
    }
    bundle["split"].write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert report["prerequisites"]["split_manifest"]["status"] == "failed"
    assert any("leakage_prevention" in e for e in report["errors"])


def test_blocked_when_split_fractions_are_not_numeric(tmp_path: Path) -> None:
    """Validator should fail closed before adding nonnumeric split fractions."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["split"].write_text(
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
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert report["prerequisites"]["split_manifest"]["status"] == "failed"
    assert "split fractions must be numeric" in report["errors"]


def test_blocked_when_baseline_missing_cv_entry(tmp_path: Path) -> None:
    """Validator should fail when constant_velocity baseline is absent."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["baseline"].write_text(
        yaml.safe_dump({"baselines": {"zero_velocity": {"ade": 1.0, "fde": 1.0}}}),
        encoding="utf-8",
    )
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert any("missing constant_velocity" in e for e in report["errors"])


def test_ready_when_named_baseline_target_present(tmp_path: Path) -> None:
    """Validator should accept a requested non-default baseline target."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["baseline"].write_text(
        yaml.safe_dump({"baselines": {"semantic_baseline": {"ade": 0.99, "fde": 1.2}}}),
        encoding="utf-8",
    )
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
        baseline_target="semantic_baseline",
    )
    assert report["status"] == "ready"


def test_ready_accepts_uppercase_baseline_metric_keys(tmp_path: Path) -> None:
    """Validator should accept ADE/FDE metric keys case-insensitively."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["baseline"].write_text(
        yaml.safe_dump({"baselines": {"constant_velocity": {"ADE": 0.45, "FDE": 0.82}}}),
        encoding="utf-8",
    )
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "ready"
    assert report["checked"]["required_baseline_target"] == DEFAULT_BASELINE_TARGET


def test_blocked_when_calibration_recommendation_is_not_continue(tmp_path: Path) -> None:
    """Calibration report must return a continue decision."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["calibration"].write_text(
        _calibration_report_content("revise", "analysis-only"),
        encoding="utf-8",
    )
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert any("recommendation is revise, expected continue" in e for e in report["errors"])


def test_blocked_when_transferability_lacks_oracle_and_deployable_tiers(
    tmp_path: Path,
) -> None:
    """Transferability matrix must include oracle and deployable observation tiers."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["transferability"].write_text(
        json.dumps(
            {
                "schema_version": FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION,
                "matrix_rows": [{"observation_tier": "deployable_vision"}],
                "recommendation": {"decision": "continue", "claim_status": "benchmark-eligible"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert any("missing observation-tier split" in e for e in report["errors"])


def test_blocked_when_transferability_recommendation_is_revise(tmp_path: Path) -> None:
    """Transferability matrix must recommend continue."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["transferability"].write_text(
        _transferability_report_content("revise", "analysis-only"),
        encoding="utf-8",
    )

    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert any("recommendation is revise, expected continue" in e for e in report["errors"])


def test_blocked_when_closed_loop_gate_is_revise(tmp_path: Path) -> None:
    """Closed-loop gate recommendation must be continue."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["closed_loop"].write_text(_closed_loop_gate_content("revise"), encoding="utf-8")
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert report["prerequisites"]["closed_loop_gate"]["status"] == "failed"
    assert any("expected continue" in e for e in report["errors"])


def test_blocked_when_closed_loop_gate_uses_legacy_passed_status(tmp_path: Path) -> None:
    """Closed-loop gate must not pass without an explicit continue recommendation."""
    bundle = _write_validation_bundle(tmp_path)
    bundle["closed_loop"].write_text(
        json.dumps({"closed_loop_gate": {"status": "passed"}}, sort_keys=True),
        encoding="utf-8",
    )
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "blocked"
    assert any("expected explicit recommendation continue" in e for e in report["errors"])


def test_ready_when_all_checks_pass(tmp_path: Path) -> None:
    """Validator should pass when all prerequisites are satisfied."""
    bundle = _write_validation_bundle(tmp_path)
    report = validate_readiness(
        doc_path=bundle["doc"],
        registry_path=bundle["registry"],
        split_manifest_path=bundle["split"],
        baseline_path=bundle["baseline"],
        calibration_report_path=bundle["calibration"],
        transferability_report_path=bundle["transferability"],
        closed_loop_gate_path=bundle["closed_loop"],
    )
    assert report["status"] == "ready"
    assert report["errors"] == []


def test_prerequisite_status_shape_when_blocked(tmp_path: Path) -> None:
    """Each prerequisite should report a status and message list."""
    doc = tmp_path / "readiness.md"
    doc.write_text(_readiness_doc_content(), encoding="utf-8")
    report = validate_readiness(doc_path=doc)
    for payload in report["prerequisites"].values():
        assert sorted(payload.keys()) == ["messages", "status"]
        assert payload["status"] in {"blocked", "failed", "passed"}
        assert isinstance(payload["messages"], list)


def test_cli_exits_zero_when_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should exit 0 when all checks pass."""
    bundle = _write_validation_bundle(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_learned_prediction_readiness.py",
            "--readiness-doc",
            str(bundle["doc"]),
            "--trace-registry",
            str(bundle["registry"]),
            "--split-manifest",
            str(bundle["split"]),
            "--baseline-evidence",
            str(bundle["baseline"]),
            "--calibration-report",
            str(bundle["calibration"]),
            "--transferability-report",
            str(bundle["transferability"]),
            "--closed-loop-gate",
            str(bundle["closed_loop"]),
            "--json",
        ],
    )
    exit_code = main()
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "ready"
    assert output["prerequisites"]["transferability_split"]["status"] == "passed"


def test_cli_resolves_relative_paths_against_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should resolve relative input paths against --repo-root."""
    _write_validation_bundle(tmp_path)
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
            "--calibration-report",
            "calibration.json",
            "--transferability-report",
            "transferability.json",
            "--closed-loop-gate",
            "closed_loop_gate.json",
            "--json",
        ],
    )
    exit_code = main()
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
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
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["status"] == "blocked"
    assert output["prerequisites"]["readiness_doc"]["status"] == "failed"


def test_cli_text_output_when_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should print BLOCKED with per-prereq messages in text mode."""
    doc = tmp_path / "readiness.md"
    doc.write_text("# Incomplete\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["validate_learned_prediction_readiness.py", "--readiness-doc", str(doc)],
    )
    exit_code = main()
    text = capsys.readouterr().out
    assert exit_code == 2
    assert "BLOCKED" in text
    assert "horizon_timestep" in text
