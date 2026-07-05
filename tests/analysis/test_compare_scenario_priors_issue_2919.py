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


def _dataset_staging_contract(
    tmp_path: Path, staging_status: str = "blocked-external-input"
) -> Path:
    """Write a minimal Issue #3161 dataset-backed staging contract."""

    return _write_yaml(
        tmp_path / "configs" / "research" / "scenario_prior_staging_contract_issue_3161.yaml",
        {
            "schema_version": "scenario_prior_staging_contract.v1",
            "contract_id": "scenario_prior_staging_contract_issue_3161",
            "issue": 3161,
            "claim_boundary": "metadata-only staging readiness; no dataset-backed claim",
            "authored_baseline": "configs/research/scenario_prior_cards_issue_2917.yaml",
            "comparison_harness": "scripts/analysis/compare_scenario_priors_issue_2919.py",
            "datasets": [
                {
                    "dataset_id": "sdd",
                    "asset_id": "sdd",
                    "title": "Stanford Drone Dataset annotations",
                    "staging_status": staging_status,
                    "redistribution": "none",
                    "blocker_issues": [3065],
                    "provenance": {
                        "source_url": "https://cvgl.stanford.edu/projects/uav_data/",
                        "license": "BYO local copy",
                        "license_status": "license-gated",
                        "citation": "Robicquet et al., ECCV 2016.",
                    },
                    "distribution_fields": ["pedestrian_speed", "pedestrian_density"],
                }
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
        dataset_staging_contract=None,
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
            dataset_staging_contract=None,
        )


def test_run_comparison_records_dataset_staging_blocker(tmp_path: Path) -> None:
    """Runner should include Issue #3161 staging readiness without claiming dataset comparison."""

    registry = _mini_registry(tmp_path)
    contract = _dataset_staging_contract(tmp_path)
    output_dir = tmp_path / "out"

    summary = comparator.run_comparison(
        registry_path=registry.relative_to(tmp_path),
        output_dir=output_dir.relative_to(tmp_path),
        repo_root=tmp_path,
        dataset_staging_contract=contract.relative_to(tmp_path),
    )

    staging = summary["dataset_backed_staging"]
    assert staging["contract_status"] == "blocked-external-input"
    assert staging["dataset_backed_comparison_allowed"] is False
    assert staging["comparison_ready_datasets"] == []

    report = (output_dir / comparator.REPORT_NAME).read_text(encoding="utf-8")
    assert "## Dataset-Backed Readiness" in report
    assert "Contract status: `blocked-external-input`" in report
    assert "Dataset-backed comparison allowed: `False`" in report


def test_run_comparison_require_dataset_ready_fails_closed(tmp_path: Path) -> None:
    """Require-ready mode should stop before producing dataset-backed comparison claims."""

    registry = _mini_registry(tmp_path)
    contract = _dataset_staging_contract(tmp_path)

    with pytest.raises(comparator.ScenarioPriorComparisonError, match="staging contract"):
        comparator.run_comparison(
            registry_path=registry.relative_to(tmp_path),
            output_dir=Path("out"),
            repo_root=tmp_path,
            dataset_staging_contract=contract.relative_to(tmp_path),
            require_dataset_backed_ready=True,
        )
