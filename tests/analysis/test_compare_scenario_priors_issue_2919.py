"""Tests for the Issue #2919 scenario-prior gap comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.analysis import compare_scenario_priors_issue_2919 as comparator


def _write_yaml(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _mini_registry(tmp_path: Path) -> Path:
    _write_yaml(
        tmp_path / "authored" / "prior.yaml",
        {
            "variants": [
                {
                    "family": "single_pedestrian_speed_offset",
                    "parameters": {
                        "speed_delta_m_s": 0.25,
                        "max_abs_speed_delta_m_s": 0.5,
                    },
                },
                {
                    "family": "pedestrian_density_offset",
                    "parameters": {
                        "density_delta": 0.02,
                        "max_ped_density": 0.12,
                    },
                },
            ]
        },
    )
    _write_yaml(
        tmp_path / "trace" / "space.yaml",
        {
            "variables": {
                "pedestrian_speed_mps": {"min": 0.8, "max": 1.4},
            }
        },
    )
    _write_json(
        tmp_path / "trace" / "summary.json",
        {"simulation_config": {"ped_density": 0.02}},
    )
    return _write_yaml(
        tmp_path / "configs" / "research" / "scenario_prior_cards_issue_2917.yaml",
        {
            "schema_version": "scenario-prior-card-registry.v1",
            "cards": [
                {
                    "card_id": "authored_card",
                    "classification": "authored",
                    "source_traces": ["authored"],
                },
                {
                    "card_id": "trace_card",
                    "classification": "repository_trace_derived",
                    "source_traces": ["trace"],
                },
                {
                    "card_id": "external_deferred",
                    "classification": "external_dataset_candidate",
                    "source_traces": ["external"],
                },
            ],
        },
    )


def test_extract_parameter_samples_groups_ranges_and_scalars(tmp_path: Path) -> None:
    """Extractor should group supported scalar and range keys under canonical parameters."""

    source_path = _write_yaml(
        tmp_path / "prior.yaml",
        {
            "variables": {"pedestrian_speed_mps": {"min": 0.8, "max": 1.4}},
            "simulation_config": {"ped_density": 0.02},
            "metadata": {"issue": 2919, "scenario_seed": 123},
        },
    )

    samples = comparator.extract_parameter_samples(
        yaml.safe_load(source_path.read_text(encoding="utf-8")),
        card_id="trace_card",
        classification="repository_trace_derived",
        source_path=source_path,
        repo_root=tmp_path,
    )

    by_parameter = {}
    for sample in samples:
        by_parameter.setdefault(sample.parameter, []).append(sample.value)

    assert by_parameter["pedestrian_speed"] == [0.8, 1.4]
    assert by_parameter["pedestrian_density"] == [0.02]
    assert "scenario_seed" not in {sample.source_key for sample in samples}


def test_build_comparison_rows_classifies_gaps() -> None:
    """Comparable authored and trace-derived distributions should receive issue labels."""

    samples = [
        comparator.ParameterSample(
            parameter="pedestrian_speed",
            source_key="speed_delta_m_s",
            value=0.25,
            card_id="authored",
            classification="authored",
            source_path="authored.yaml",
            yaml_path="variants.parameters.speed_delta_m_s",
        ),
        comparator.ParameterSample(
            parameter="pedestrian_speed",
            source_key="max_abs_speed_delta_m_s",
            value=0.5,
            card_id="authored",
            classification="authored",
            source_path="authored.yaml",
            yaml_path="variants.parameters.max_abs_speed_delta_m_s",
        ),
        comparator.ParameterSample(
            parameter="pedestrian_speed",
            source_key="pedestrian_speed_mps",
            value=0.8,
            card_id="trace",
            classification="repository_trace_derived",
            source_path="trace.yaml",
            yaml_path="variables.pedestrian_speed_mps",
        ),
        comparator.ParameterSample(
            parameter="pedestrian_speed",
            source_key="pedestrian_speed_mps",
            value=1.4,
            card_id="trace",
            classification="repository_trace_derived",
            source_path="trace.yaml",
            yaml_path="variables.pedestrian_speed_mps",
        ),
    ]

    rows = comparator.build_comparison_rows(samples)

    assert len(rows) == 1
    assert rows[0]["parameter"] == "pedestrian_speed"
    assert rows[0]["classification"] in {"too_narrow", "too_extreme"}
    assert rows[0]["ks_distance"] is not None
    assert rows[0]["wasserstein_like_distance"] is not None
    assert "family" in rows[0]["proposal"]


def test_run_comparison_writes_report_csv_and_summary(tmp_path: Path) -> None:
    """The end-to-end runner should write durable compact evidence files."""

    registry = _mini_registry(tmp_path)
    output_dir = tmp_path / "docs" / "context" / "evidence" / "issue_2919"

    summary = comparator.run_comparison(
        registry_path=registry.relative_to(tmp_path),
        output_dir=output_dir.relative_to(tmp_path),
        repo_root=tmp_path,
    )

    report_path = output_dir / comparator.REPORT_NAME
    csv_path = output_dir / comparator.CSV_NAME
    summary_path = output_dir / comparator.SUMMARY_NAME

    assert report_path.is_file()
    assert csv_path.is_file()
    assert summary_path.is_file()
    assert summary["evidence_tier"] == "analysis_only"
    assert summary["comparison_count"] >= 2
    assert "deferred to #3161" in report_path.read_text(encoding="utf-8")
    assert "No planner ranking" in report_path.read_text(encoding="utf-8")
    assert "pedestrian_speed" in csv_path.read_text(encoding="utf-8")


def test_run_comparison_fails_when_no_common_parameters(tmp_path: Path) -> None:
    """The runner should fail clearly when registry sources share no comparable parameters."""

    _write_yaml(tmp_path / "authored" / "prior.yaml", {"parameters": {"dx_m": 0.1}})
    _write_yaml(tmp_path / "trace" / "prior.yaml", {"variables": {"pedestrian_speed_mps": 1.2}})
    registry = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "cards": [
                {
                    "card_id": "authored",
                    "classification": "authored",
                    "source_traces": ["authored"],
                },
                {
                    "card_id": "trace",
                    "classification": "repository_trace_derived",
                    "source_traces": ["trace"],
                },
            ]
        },
    )

    with pytest.raises(comparator.ScenarioPriorComparisonError, match="no comparable"):
        comparator.run_comparison(
            registry_path=registry.relative_to(tmp_path),
            output_dir=Path("out"),
            repo_root=tmp_path,
        )
