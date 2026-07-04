"""Tests for camera-ready benchmark campaign orchestration."""

from __future__ import annotations

import csv
import io
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from loguru import logger

import robot_sf.benchmark.camera_ready as camera_ready_package
import robot_sf.benchmark.camera_ready._artifacts as camera_ready_artifacts_module
import robot_sf.benchmark.camera_ready._config as camera_ready_config_module
import robot_sf.benchmark.camera_ready._config_types as camera_ready_config_types_module
import robot_sf.benchmark.camera_ready._legacy_campaign_facade as camera_ready_legacy_facade
import robot_sf.benchmark.camera_ready._run_state as camera_ready_run_state_module
import robot_sf.benchmark.camera_ready_campaign as camera_ready_campaign_module
import robot_sf.benchmark.camera_ready_campaign_config as camera_ready_campaign_config_module
from robot_sf.benchmark.artifact_publication import PublicationBundleResult
from robot_sf.benchmark.camera_ready._artifacts import (
    _write_csv,
    _write_json,
    _write_table_artifacts,
)
from robot_sf.benchmark.camera_ready._preflight import (
    _build_preflight_preview_payload,
    _build_preflight_validate_payload,
)
from robot_sf.benchmark.camera_ready._util import (
    _kinematics_matrix_or_default,
    _stable_json_bytes,
)
from robot_sf.benchmark.camera_ready_campaign import (
    DEFAULT_SEED_SETS_PATH,
    CampaignConfig,
    PlannerSpec,
    SeedPolicy,
    _build_actuation_envelope_summary,
    _build_breakdown_rows,
    _build_scenario_amv_lookup,
    _campaign_success_counters,
    _extract_amv_taxonomy,
    _jsonable_repo_relative,
    _load_campaign_scenarios,
    _load_route_clearance_certifications,
    _planner_report_row,
    _resolved_seed_inventory,
    _sanitize_csv_cell,
    _sanitize_git_remote,
    _sanitize_name,
    _scenario_with_kinematics,
    _sha256_file,
    load_campaign_config,
    prepare_campaign_preflight,
    run_campaign,
    write_campaign_report,
)
from robot_sf.benchmark.map_runner_profile_metadata import load_synthetic_actuation_profile
from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError
from robot_sf.benchmark.synthetic_actuation import (
    CALIBRATED_ACTUATION_REQUIRED_PROVENANCE_FIELDS,
    SyntheticActuationProfile,
    actuation_variability_fields,
    sample_synthetic_actuation_profile,
    summarize_synthetic_actuation_samples,
    validate_actuation_profile_claim_boundary,
    validate_synthetic_actuation_profile,
    validate_synthetic_actuation_variability_distribution,
)
from robot_sf.common.artifact_paths import get_repository_root


def test_camera_ready_campaign_reexports_package_artifact_helpers() -> None:
    """Legacy camera_ready_campaign imports expose moved artifact helpers."""
    helper_names = (
        "_markdown_rows_from_mapping_rows",
        "_write_actuation_envelope_artifacts",
        "_write_amv_coverage_artifacts",
        "_write_comparability_artifacts",
        "_write_matrix_summary_artifacts",
        "_write_seed_episode_rows_artifact",
        "_write_seed_variability_artifacts",
        "_write_snqi_diagnostics_artifacts",
        "_write_statistical_sufficiency_artifact",
    )

    for helper_name in helper_names:
        assert getattr(camera_ready_campaign_module, helper_name) is getattr(
            camera_ready_artifacts_module, helper_name
        )


def test_camera_ready_campaign_reexports_package_config_loader() -> None:
    """Legacy camera_ready_campaign imports expose the moved config loader/validator."""
    helper_names = (
        "_validate_campaign_config",
        "load_campaign_config",
    )

    for helper_name in helper_names:
        assert getattr(camera_ready_campaign_module, helper_name) is getattr(
            camera_ready_config_module, helper_name
        )


def test_camera_ready_campaign_legacy_module_is_package_owned_facade() -> None:
    """Legacy campaign module resolves to the package-owned compatibility facade."""
    assert camera_ready_campaign_module is camera_ready_legacy_facade
    assert camera_ready_campaign_module.run_campaign is camera_ready_legacy_facade.run_campaign


def test_camera_ready_config_types_keep_legacy_import_identity() -> None:
    """Config dataclass extraction keeps package and legacy import paths object-identical."""
    public_names = (
        "DEFAULT_SEED_SETS_PATH",
        "AmvProfileConfig",
        "CampaignConfig",
        "PlannerSpec",
        "ScenarioCandidateSelection",
        "SeedPolicy",
        "SnqiContractConfig",
    )

    for name in public_names:
        assert getattr(camera_ready_package, name) is getattr(
            camera_ready_config_types_module, name
        )
        assert getattr(camera_ready_campaign_config_module, name) is getattr(
            camera_ready_config_types_module, name
        )
        assert getattr(camera_ready_campaign_module, name) is getattr(
            camera_ready_config_types_module, name
        )
    assert camera_ready_package._AMV_DIMENSIONS is camera_ready_config_types_module._AMV_DIMENSIONS
    assert (
        camera_ready_campaign_config_module._AMV_DIMENSIONS
        is camera_ready_config_types_module._AMV_DIMENSIONS
    )
    assert (
        camera_ready_package.load_campaign_config is camera_ready_config_module.load_campaign_config
    )


def test_camera_ready_campaign_reexports_package_run_state_helpers() -> None:
    """Legacy camera_ready_campaign imports expose moved run-state helpers."""
    helper_names = (
        "_campaign_id",
        "_campaign_success_counters",
        "_git_context",
        "_resolve_campaign_id",
        "_resolve_execution_mode",
        "_resolve_observation_noise",
        "_resolve_path",
        "_sanitize_git_remote",
    )

    for helper_name in helper_names:
        assert getattr(camera_ready_campaign_module, helper_name) is getattr(
            camera_ready_run_state_module, helper_name
        )


def test_campaign_success_counters_core_success_ignores_experimental_failure() -> None:
    """Campaign success is anchored on complete successful core planners when present."""
    counters = _campaign_success_counters(
        [
            {"status": "ok", "planner": {"planner_group": "core"}},
            {"status": "not_available", "planner": {"planner_group": "experimental"}},
        ],
        expected_core_runs=1,
    )

    assert counters["benchmark_success"] is True
    assert counters["benchmark_success_basis"] == "core"
    assert counters["total_runs"] == 2
    assert counters["successful_runs"] == 1
    assert counters["core_total_runs"] == 1
    assert counters["core_successful_runs"] == 1


def test_scenario_with_kinematics_patches_copy_without_mutating_input() -> None:
    """Scenario kinematics patching preserves unrelated fields and leaves input untouched."""
    scenario = {
        "name": "sample",
        "robot_config": {"radius": 0.3, "type": "differential"},
        "metadata": {"group": "smoke"},
    }

    patched = _scenario_with_kinematics(
        scenario,
        kinematics="unicycle",
        holonomic_command_mode="vx_vy",
    )

    assert patched == {
        "name": "sample",
        "robot_config": {"radius": 0.3, "type": "unicycle"},
        "metadata": {"group": "smoke"},
    }
    assert scenario["robot_config"] == {"radius": 0.3, "type": "differential"}
    assert patched is not scenario
    assert patched["robot_config"] is not scenario["robot_config"]


def test_scenario_with_kinematics_sets_holonomic_command_mode_default() -> None:
    """Holonomic scenarios get the campaign default command mode only when missing."""
    patched = _scenario_with_kinematics(
        {"name": "sample"},
        kinematics="holonomic",
        holonomic_command_mode="vx_vy",
    )
    preserved = _scenario_with_kinematics(
        {"name": "sample", "robot_config": {"command_mode": "speed_heading"}},
        kinematics="holonomic",
        holonomic_command_mode="vx_vy",
    )

    assert patched["robot_config"] == {"type": "holonomic", "command_mode": "vx_vy"}
    assert preserved["robot_config"] == {
        "command_mode": "speed_heading",
        "type": "holonomic",
    }


def test_load_campaign_config_resolves_relative_paths(tmp_path: Path):
    """Config loader should resolve scenario and algo-config paths relative to config file."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True)

    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    algo_cfg_rel = Path("configs/algos/social_force_example.yaml")
    scenario_abs = (config_dir / scenario_rel).resolve()
    algo_cfg_abs = (config_dir / algo_cfg_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    algo_cfg_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    algo_cfg_abs.write_text("v_max: 1.0\n", encoding="utf-8")

    config_path = config_dir / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [101]",
                "planners:",
                "  - key: sf",
                "    algo: social_force",
                f"    algo_config: {algo_cfg_rel.as_posix()}",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)

    assert cfg.name == "test_campaign"
    assert cfg.scenario_matrix_path == scenario_abs
    assert cfg.scenario_matrix_path.exists()
    assert cfg.planners[0].algo_config_path is not None
    assert cfg.planners[0].algo_config_path == algo_cfg_abs


def test_load_campaign_config_resolves_observation_noise_profile(tmp_path: Path) -> None:
    """Campaign configs should accept file-backed observation-noise profiles."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True)
    matrix_path = config_dir / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "s1",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "simulation_config": {"max_episode_steps": 1},
                }
            ]
        ),
        encoding="utf-8",
    )
    noise_path = config_dir / "noise.yaml"
    noise_path.write_text(
        yaml.safe_dump({"profile": "unit_noise", "pedestrian_false_negative_prob": 0.25}),
        encoding="utf-8",
    )
    cfg_path = config_dir / "campaign.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "noise_campaign",
                "scenario_matrix": "matrix.yaml",
                "observation_noise": "noise.yaml",
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_campaign_config(cfg_path)

    assert cfg.observation_noise is not None
    assert cfg.observation_noise["profile"] == "unit_noise"
    assert cfg.observation_noise["enabled"] is True


def test_load_campaign_config_rejects_directory_observation_noise_profile(
    tmp_path: Path,
) -> None:
    """Campaign configs should fail closed when observation_noise points at a directory."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True)
    matrix_path = config_dir / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "s1",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "simulation_config": {"max_episode_steps": 1},
                }
            ]
        ),
        encoding="utf-8",
    )
    noise_dir = config_dir / "noise_dir"
    noise_dir.mkdir()
    cfg_path = config_dir / "campaign.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "noise_campaign",
                "scenario_matrix": "matrix.yaml",
                "observation_noise": "noise_dir",
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Could not resolve observation_noise"):
        load_campaign_config(cfg_path)


def test_load_campaign_config_rejects_malformed_scenario_candidates(
    tmp_path: Path,
) -> None:
    """Malformed scenario candidate selectors should not silently expand the slice."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad_candidates",
                "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
                "scenario_candidates": {"name": "francis2023_blind_corner"},
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="scenario_candidates must be"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_malformed_scenario_amv_overrides(tmp_path: Path) -> None:
    """Scenario AMV overrides must stay a mapping of scenario names to AMV mappings."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad_scenario_amv",
                "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
                "scenario_amv_overrides": {"francis2023_blind_corner": ["delivery_robot"]},
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="scenario_amv_overrides entries must be mappings"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_null_scenario_amv_override_values(
    tmp_path: Path,
) -> None:
    """Scenario AMV override values must not coerce YAML null to the string ``None``."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad_scenario_amv_null",
                "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
                "scenario_amv_overrides": {"francis2023_blind_corner": {"use_case": None}},
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scenario_amv_overrides values must be non-empty"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_malformed_synthetic_actuation_profile(
    tmp_path: Path,
) -> None:
    """Malformed synthetic actuation profile payloads should fail closed."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad_actuation_profile",
                "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
                "synthetic_actuation_profile": [],
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="synthetic_actuation_profile must be a mapping"):
        load_campaign_config(config_path)


def test_load_campaign_config_preserves_synthetic_actuation_variability_metadata(
    tmp_path: Path,
) -> None:
    """Synthetic actuation variability metadata should survive typed config loading."""
    config_path = tmp_path / "campaign.yaml"
    config_payload = {
        "name": "sampled_actuation_profile",
        "paper_facing": False,
        "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
        "kinematics_matrix": ["differential_drive"],
        "synthetic_actuation_profile": {
            "name": "variability_sample_000",
            "profile_version": "v0",
            "claim_scope": "synthetic-only",
            "claim_boundary": "diagnostic-only",
            "max_linear_accel_m_s2": 3.0,
            "max_linear_decel_m_s2": 3.4,
            "max_yaw_rate_rad_s": 1.0,
            "max_angular_accel_rad_s2": 3.0,
            "latency_mode": "one-step-delay",
            "update_mode": "5hz-hold",
            "variability_distribution": {
                "schema_version": "synthetic-actuation-variability-distribution.v1",
                "mode": "synthetic-provisional",
                "claim_boundary": "diagnostic-only",
                "parameters": {
                    "max_linear_accel_m_s2": {
                        "distribution": "uniform",
                        "low": 2.5,
                        "high": 4.5,
                        "provenance": {
                            "units": "m/s^2",
                            "source_status": "synthetic_stress_factor",
                            "caveat": "test-only provisional range",
                        },
                    }
                },
            },
            "variability_sample": {
                "schema_version": "synthetic-actuation-variability-sample.v1",
                "mode": "variability-sweep",
                "sample_index": 0,
                "sample_id": "sample-000",
                "sampling_seed": 17,
                "sampled_parameters": {"max_linear_accel_m_s2": 3.0},
            },
        },
        "planners": [{"key": "goal", "algo": "goal"}],
    }
    config_path.write_text(
        yaml.safe_dump(config_payload),
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)

    assert cfg.synthetic_actuation_profile is not None
    metadata = cfg.synthetic_actuation_profile.to_metadata()
    assert metadata["variability_distribution"]["parameters"]["max_linear_accel_m_s2"]["low"] == 2.5
    assert metadata["variability_sample"]["sample_id"] == "sample-000"

    bad_config_path = tmp_path / "campaign_null_sample.yaml"
    bad_config_payload = dict(config_payload)
    bad_profile_payload = dict(bad_config_payload["synthetic_actuation_profile"])
    bad_profile_payload["variability_sample"] = None
    bad_config_payload["synthetic_actuation_profile"] = bad_profile_payload
    bad_config_path.write_text(yaml.safe_dump(bad_config_payload), encoding="utf-8")
    with pytest.raises(TypeError, match="variability_sample must be a mapping"):
        load_campaign_config(bad_config_path)


def test_synthetic_actuation_variability_distribution_rejects_missing_provenance() -> None:
    """Synthetic variability distributions should fail closed without source caveats."""
    with pytest.raises(ValueError, match="provenance missing fields"):
        validate_synthetic_actuation_variability_distribution(
            {
                "schema_version": "synthetic-actuation-variability-distribution.v1",
                "mode": "synthetic-provisional",
                "claim_boundary": "diagnostic-only",
                "parameters": {
                    "max_linear_accel_m_s2": {
                        "distribution": "uniform",
                        "low": 2.5,
                        "high": 4.5,
                        "provenance": {"units": "m/s^2"},
                    }
                },
            }
        )


def test_synthetic_actuation_variability_sampling_is_seeded_and_summarized() -> None:
    """Seeded distribution sampling should materialize reproducible scalar profiles."""
    base_profile = SyntheticActuationProfile(
        name="amv-actuation-baseline",
        profile_version="v0",
        claim_scope="synthetic-only",
        claim_boundary="diagnostic-only",
        max_linear_accel_m_s2=3.0,
        max_linear_decel_m_s2=3.4,
        max_yaw_rate_rad_s=1.0,
        max_angular_accel_rad_s2=3.0,
        latency_mode="one-step-delay",
        update_mode="5hz-hold",
    )
    distribution = {
        "schema_version": "synthetic-actuation-variability-distribution.v1",
        "mode": "synthetic-provisional",
        "claim_boundary": "diagnostic-only",
        "parameters": {
            "max_linear_accel_m_s2": {
                "distribution": "uniform",
                "low": 2.5,
                "high": 4.5,
                "provenance": {
                    "units": "m/s^2",
                    "source_status": "synthetic_stress_factor",
                    "caveat": "test-only provisional range",
                },
            },
            "update_mode": {
                "distribution": "choice",
                "choices": ["2.5hz-hold", "5hz-hold"],
                "provenance": {
                    "units": "profile-label",
                    "source_status": "synthetic_stress_factor",
                    "caveat": "test-only provisional choices",
                },
            },
        },
    }

    assert "max_linear_accel_m_s2" in actuation_variability_fields()
    first = sample_synthetic_actuation_profile(base_profile, distribution, seed=17, sample_index=0)
    repeated = sample_synthetic_actuation_profile(
        base_profile,
        distribution,
        seed=17,
        sample_index=0,
    )

    assert first.to_metadata() == repeated.to_metadata()
    assert 2.5 <= first.max_linear_accel_m_s2 <= 4.5
    assert first.update_mode in {"2.5hz-hold", "5hz-hold"}
    assert first.variability_sample is not None
    assert first.variability_sample["sample_id"] == "sample-000"

    summary = summarize_synthetic_actuation_samples([base_profile, first])
    assert summary["row_count"] == 1
    assert summary["rows"][0]["sampled_parameters"]["max_linear_accel_m_s2"] == pytest.approx(
        first.max_linear_accel_m_s2
    )

    tuple_metadata = replace(
        base_profile,
        variability_sample={
            "schema_version": "synthetic-actuation-variability-sample.v1",
            "tuple_value": ("a", "b"),
            "list_value": [1, None],
        },
    ).to_metadata()
    assert tuple_metadata["variability_sample"]["tuple_value"] == ["a", "b"]
    assert tuple_metadata["variability_sample"]["list_value"] == [1, None]

    with pytest.raises(ValueError, match="sample_index must be >= 0"):
        sample_synthetic_actuation_profile(base_profile, distribution, seed=17, sample_index=-1)
    with pytest.raises(ValueError, match=r"variability_sample\.schema_version"):
        validate_synthetic_actuation_profile(replace(base_profile, variability_sample={}))
    with pytest.raises(ValueError, match="variability_distribution must be a mapping"):
        validate_synthetic_actuation_profile(
            replace(base_profile, variability_distribution=["not", "a", "mapping"])
        )


def test_map_runner_profile_loader_preserves_variability_metadata() -> None:
    """Map-runner profile normalization should preserve sampled-profile metadata."""
    payload = {
        "name": "variability_sample_000",
        "profile_version": "v0",
        "claim_scope": "synthetic-only",
        "claim_boundary": "diagnostic-only",
        "max_linear_accel_m_s2": 3.0,
        "max_linear_decel_m_s2": 3.4,
        "max_yaw_rate_rad_s": 1.0,
        "max_angular_accel_rad_s2": 3.0,
        "latency_mode": "one-step-delay",
        "update_mode": "5hz-hold",
        "variability_distribution": {
            "schema_version": "synthetic-actuation-variability-distribution.v1",
            "mode": "synthetic-provisional",
            "claim_boundary": "diagnostic-only",
            "parameters": {
                "max_linear_accel_m_s2": {
                    "distribution": "uniform",
                    "low": 2.5,
                    "high": 4.5,
                    "provenance": {
                        "units": "m/s^2",
                        "source_status": "synthetic_stress_factor",
                        "caveat": "test-only provisional range",
                    },
                }
            },
        },
        "variability_sample": {
            "schema_version": "synthetic-actuation-variability-sample.v1",
            "sample_id": "sample-000",
        },
    }

    profile = load_synthetic_actuation_profile(payload)

    assert profile is not None
    assert load_synthetic_actuation_profile(None) is None
    assert load_synthetic_actuation_profile(profile) is profile
    assert profile.to_metadata()["variability_sample"]["sample_id"] == "sample-000"
    assert (
        profile.to_metadata()["variability_distribution"]["parameters"]["max_linear_accel_m_s2"][
            "high"
        ]
        == 4.5
    )

    with pytest.raises(TypeError, match="synthetic_actuation_profile must be a mapping"):
        load_synthetic_actuation_profile([])
    with pytest.raises(ValueError, match="claim_scope must be 'synthetic-only'"):
        load_synthetic_actuation_profile({**payload, "claim_scope": "paper-facing"})
    with pytest.raises(TypeError, match="variability_sample must be a mapping"):
        load_synthetic_actuation_profile({**payload, "variability_sample": []})
    with pytest.raises(TypeError, match="variability_sample must be a mapping"):
        load_synthetic_actuation_profile({**payload, "variability_sample": None})


def test_load_campaign_config_rejects_malformed_latency_stress_profile(
    tmp_path: Path,
) -> None:
    """Malformed latency-stress profile payloads should fail closed."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad_latency_profile",
                "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
                "latency_stress_profile": [],
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="latency_stress_profile must be a mapping"):
        load_campaign_config(config_path)


def test_load_campaign_config_accepts_calibrated_actuation_profile_with_provenance(
    tmp_path: Path,
) -> None:
    """A calibrated-actuation profile with provenance should be accepted by the campaign loader."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "calibrated_actuation_scope",
                "paper_facing": False,
                "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
                "kinematics_matrix": ["differential_drive"],
                "synthetic_actuation_profile": {
                    "name": "amv-actuation-stress-v0",
                    "profile_version": "v0",
                    "claim_scope": "hardware-calibrated",
                    "claim_boundary": "calibrated-amv-actuation",
                    "provenance": {
                        "source_id": "issue-1585-placeholder",
                        "source_uri": "https://github.com/ll7/robot_sf_ll7/issues/1585",
                        "source_type": "maintainer-approved-calibration-source",
                        "profile_version": "v0",
                        "measurement_date": "2026-05-31",
                        "supported_actuation_fields": [
                            "max_linear_accel_m_s2",
                            "max_linear_decel_m_s2",
                            "max_yaw_rate_rad_s",
                            "max_angular_accel_rad_s2",
                            "latency_mode",
                            "update_mode",
                        ],
                        "units": {
                            "max_linear_accel_m_s2": "m/s^2",
                            "max_linear_decel_m_s2": "m/s^2",
                            "max_yaw_rate_rad_s": "rad/s",
                            "max_angular_accel_rad_s2": "rad/s^2",
                            "latency_mode": "profile-label",
                            "update_mode": "profile-label",
                        },
                        "claim_boundary": "calibrated-profile-contract-only",
                    },
                    "max_linear_accel_m_s2": 2.0,
                    "max_linear_decel_m_s2": 2.5,
                    "max_yaw_rate_rad_s": 1.2,
                    "max_angular_accel_rad_s2": 4.0,
                    "latency_mode": "one-step-delay",
                    "update_mode": "5hz-hold",
                },
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    assert cfg.synthetic_actuation_profile is not None
    assert cfg.synthetic_actuation_profile.claim_scope == "hardware-calibrated"


def test_load_campaign_config_rejects_synthetic_profile_without_diagnostic_boundary(
    tmp_path: Path,
) -> None:
    """Synthetic AMV actuation profiles must explicitly stay diagnostic-only."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "missing_actuation_boundary",
                "paper_facing": False,
                "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
                "kinematics_matrix": ["differential_drive"],
                "synthetic_actuation_profile": {
                    "name": "amv-actuation-stress-v0",
                    "profile_version": "v0",
                    "claim_scope": "synthetic-only",
                    "max_linear_accel_m_s2": 2.0,
                    "max_linear_decel_m_s2": 2.5,
                    "max_yaw_rate_rad_s": 1.2,
                    "max_angular_accel_rad_s2": 4.0,
                    "latency_mode": "one-step-delay",
                    "update_mode": "5hz-hold",
                },
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="claim_boundary must be 'diagnostic-only'"):
        load_campaign_config(config_path)


def test_calibrated_labeled_actuation_profile_requires_provenance_fields() -> None:
    """Future calibrated profiles should fail closed until source provenance is explicit."""
    with pytest.raises(ValueError, match="requires provenance fields"):
        validate_actuation_profile_claim_boundary(
            {
                "name": "amv-actuation-calibrated-v0",
                "profile_version": "v0",
                "claim_scope": "hardware-calibrated",
                "claim_boundary": "calibrated-amv-actuation",
                "max_linear_accel_m_s2": 1.5,
                "max_linear_decel_m_s2": 2.0,
                "max_yaw_rate_rad_s": 0.9,
                "max_angular_accel_rad_s2": 2.5,
                "latency_mode": "one-step-delay",
                "update_mode": "5hz-hold",
            },
            label="calibrated_actuation_profile",
        )


def test_uncalibrated_label_does_not_trigger_calibrated_profile_gate() -> None:
    """Exact calibrated marker matching should not reject ordinary uncalibrated labels."""
    validate_actuation_profile_claim_boundary(
        {
            "name": "uncalibrated-synthetic-amv-v0",
            "profile_version": "v0",
            "claim_scope": "synthetic-only",
            "claim_boundary": "diagnostic-only",
            "calibration_status": "uncalibrated",
        }
    )


def test_typed_synthetic_profile_rejects_calibrated_label_without_provenance() -> None:
    """Direct SyntheticActuationProfile callers must not bypass calibrated-claim provenance."""
    profile = SyntheticActuationProfile(
        name="amv-actuation-calibrated-v0",
        profile_version="v0",
        claim_scope="hardware-calibrated",
        claim_boundary="calibrated-amv-actuation",
        max_linear_accel_m_s2=1.5,
        max_linear_decel_m_s2=2.0,
        max_yaw_rate_rad_s=0.9,
        max_angular_accel_rad_s2=2.5,
        latency_mode="one-step-delay",
        update_mode="5hz-hold",
    )

    with pytest.raises(ValueError, match="requires provenance fields"):
        validate_synthetic_actuation_profile(profile)


@pytest.mark.parametrize("bad_bound", [math.nan, math.inf, -math.inf])
def test_typed_synthetic_profile_rejects_non_finite_bounds(bad_bound: float) -> None:
    """Synthetic actuation bounds must be finite before controller math uses them."""
    profile = SyntheticActuationProfile(
        name="amv-actuation-stress-v0",
        profile_version="v0",
        claim_scope="synthetic-only",
        claim_boundary="diagnostic-only",
        max_linear_accel_m_s2=bad_bound,
        max_linear_decel_m_s2=2.0,
        max_yaw_rate_rad_s=0.9,
        max_angular_accel_rad_s2=2.5,
        latency_mode="one-step-delay",
        update_mode="5hz-hold",
    )

    with pytest.raises(ValueError, match="max_linear_accel_m_s2 must be > 0"):
        validate_synthetic_actuation_profile(profile)


def test_calibrated_actuation_profile_provenance_contract_names_required_fields() -> None:
    """The calibrated provenance contract should be inspectable by tests and docs."""
    assert CALIBRATED_ACTUATION_REQUIRED_PROVENANCE_FIELDS == (
        "source_id",
        "source_uri",
        "source_type",
        "profile_version",
        "measurement_date",
        "supported_actuation_fields",
        "units",
        "claim_boundary",
    )


def test_synthetic_profile_with_calibrated_markers_rejected() -> None:
    """A synthetic-only profile that looks calibrated should be rejected as conflation."""
    with pytest.raises(ValueError, match="calibrated-looking markers"):
        validate_actuation_profile_claim_boundary(
            {
                "name": "amv-actuation-calibrated-synthetic-v0",
                "profile_version": "v0",
                "claim_scope": "synthetic-only",
                "claim_boundary": "diagnostic-only",
                "calibration_status": "hardware-calibrated",
                "max_linear_accel_m_s2": 2.0,
                "max_linear_decel_m_s2": 2.5,
                "max_yaw_rate_rad_s": 1.2,
                "max_angular_accel_rad_s2": 4.0,
                "latency_mode": "one-step-delay",
                "update_mode": "5hz-hold",
            }
        )


def test_calibrated_profile_with_provenance_passes_validation() -> None:
    """A calibrated profile with complete provenance should pass validation."""
    validate_actuation_profile_claim_boundary(
        {
            "name": "amv-actuation-calibrated-v0",
            "profile_version": "v0",
            "claim_scope": "hardware-calibrated",
            "claim_boundary": "calibrated-amv-actuation",
            "max_linear_accel_m_s2": 1.5,
            "max_linear_decel_m_s2": 2.0,
            "max_yaw_rate_rad_s": 0.9,
            "max_angular_accel_rad_s2": 2.5,
            "latency_mode": "one-step-delay",
            "update_mode": "5hz-hold",
            "provenance": {
                "source_id": "test-source-001",
                "source_uri": "https://example.com/calibrated-trace",
                "source_type": "hardware-trace-collection",
                "profile_version": "v1-calibrated",
                "measurement_date": "2026-06-01",
                "supported_actuation_fields": [
                    "max_linear_accel_m_s2",
                    "max_linear_decel_m_s2",
                ],
                "units": {
                    "max_linear_accel_m_s2": "m/s^2",
                    "max_linear_decel_m_s2": "m/s^2",
                },
                "claim_boundary": "calibrated-amv-actuation",
            },
        },
        label="calibrated_actuation_profile",
    )


def test_typed_synthetic_profile_with_calibrated_name_rejected() -> None:
    """A SyntheticActuationProfile with synthetic-only scope but calibrated name is rejected."""
    with pytest.raises(ValueError, match="calibrated-looking markers"):
        validate_synthetic_actuation_profile(
            SyntheticActuationProfile(
                name="hardware-calibrated-profile-v0",
                profile_version="v0",
                claim_scope="synthetic-only",
                claim_boundary="diagnostic-only",
                max_linear_accel_m_s2=2.0,
                max_linear_decel_m_s2=2.5,
                max_yaw_rate_rad_s=1.2,
                max_angular_accel_rad_s2=4.0,
                latency_mode="one-step-delay",
                update_mode="5hz-hold",
            )
        )


def test_actuation_envelope_summary_distinguishes_profile_type() -> None:
    """Evidence summary should include an actuation_profile_type field."""
    profile = SyntheticActuationProfile(
        name="test-profile",
        profile_version="v0",
        claim_scope="synthetic-only",
        claim_boundary="diagnostic-only",
        max_linear_accel_m_s2=2.0,
        max_linear_decel_m_s2=2.5,
        max_yaw_rate_rad_s=1.2,
        max_angular_accel_rad_s2=4.0,
        latency_mode="one-step-delay",
        update_mode="5hz-hold",
    )
    from robot_sf.benchmark.camera_ready._summaries import _build_actuation_envelope_summary

    summary = _build_actuation_envelope_summary(
        campaign_id="test-campaign",
        generated_at_utc="2026-06-23T00:00:00Z",
        profile=profile,
        planner_rows=[],
    )
    assert summary["actuation_profile_type"] == "synthetic_diagnostic"
    assert summary["claim_boundary"] == "diagnostic-only"

    calibrated_profile = SyntheticActuationProfile(
        name="test-calibrated-profile",
        profile_version="v0",
        claim_scope="hardware-calibrated",
        claim_boundary="calibrated-amv-actuation",
        max_linear_accel_m_s2=2.0,
        max_linear_decel_m_s2=2.5,
        max_yaw_rate_rad_s=1.2,
        max_angular_accel_rad_s2=4.0,
        latency_mode="one-step-delay",
        update_mode="5hz-hold",
        provenance={
            "source_id": "test-source-001",
            "source_uri": "https://example.com/calibrated-trace",
            "source_type": "hardware-trace-collection",
            "profile_version": "v1-calibrated",
            "measurement_date": "2026-06-01",
            "supported_actuation_fields": ["max_linear_accel_m_s2"],
            "units": {"max_linear_accel_m_s2": "m/s^2"},
            "claim_boundary": "calibrated-amv-actuation",
        },
    )

    calibrated_summary = _build_actuation_envelope_summary(
        campaign_id="test-campaign",
        generated_at_utc="2026-06-23T00:00:00Z",
        profile=calibrated_profile,
        planner_rows=[],
    )
    assert calibrated_summary["actuation_profile_type"] == "calibrated_amv_actuation"
    assert calibrated_summary["claim_boundary"] == "calibrated-amv-actuation"


def test_load_campaign_config_rejects_invalid_latency_stress_scope(
    tmp_path: Path,
) -> None:
    """Latency-stress diagnostics should stay synthetic-only and non-paper-facing."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad_latency_scope",
                "paper_facing": False,
                "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
                "kinematics_matrix": ["differential_drive"],
                "latency_stress_profile": {
                    "name": "learned-policy-latency-stress-v0",
                    "profile_version": "v0",
                    "claim_scope": "hardware-calibrated",
                    "observation_delay_steps": 1,
                    "action_delay_steps": 1,
                    "planner_update_mode": "hold-last",
                    "planner_update_period_steps": 2,
                    "inference_timeout_ms": 200.0,
                },
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="claim_scope must be 'synthetic-only'"):
        load_campaign_config(config_path)


def test_prepare_campaign_preflight_resolves_synthetic_actuation_slice_metadata(
    tmp_path: Path,
) -> None:
    """Preflight should resolve candidate scenarios, eval seeds, and synthetic profile metadata."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "classic_overtaking_medium",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "metadata": {"archetype": "classic_crossing"},
                    "amv": {
                        "use_case": "delivery_robot",
                        "context": "sidewalk",
                        "speed_regime": "walking_speed",
                        "maneuver_type": "overtake",
                    },
                },
                {
                    "name": "classic_bottleneck_high",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "metadata": {"archetype": "classic_crossing"},
                    "amv": {
                        "use_case": "delivery_robot",
                        "context": "sidewalk",
                        "speed_regime": "walking_speed",
                        "maneuver_type": "bottleneck",
                    },
                },
                {
                    "name": "classic_cross_trap_high",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "metadata": {"archetype": "classic_crossing"},
                    "amv": {
                        "use_case": "shared_space_micromobility",
                        "context": "shared_space",
                        "speed_regime": "scooter_speed",
                        "maneuver_type": "crossing",
                    },
                },
                {
                    "name": "francis2023_blind_corner",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "metadata": {"archetype": "classic_crossing"},
                    "amv": {
                        "use_case": "delivery_robot",
                        "context": "sidewalk",
                        "speed_regime": "walking_speed",
                        "maneuver_type": "crossing",
                    },
                },
                {
                    "name": "francis2023_intersection_wait",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "metadata": {"archetype": "classic_crossing"},
                    "amv": {
                        "use_case": "shared_space_micromobility",
                        "context": "shared_space",
                        "speed_regime": "scooter_speed",
                        "maneuver_type": "bottleneck",
                    },
                },
                {
                    "name": "unused_extra_row",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "metadata": {"archetype": "classic_crossing"},
                    "amv": {
                        "use_case": "delivery_robot",
                        "context": "sidewalk",
                        "speed_regime": "walking_speed",
                        "maneuver_type": "crossing",
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "issue_1556_preflight",
                "paper_facing": False,
                "scenario_matrix": str(scenario_path),
                "kinematics_matrix": ["differential_drive"],
                "scenario_candidates": [
                    "classic_overtaking_medium",
                    "classic_bottleneck_high",
                    "classic_cross_trap_high",
                    "francis2023_blind_corner",
                    "francis2023_intersection_wait",
                ],
                "seed_policy": {"mode": "seed-set", "seed_set": "eval"},
                "synthetic_actuation_profile": {
                    "name": "amv-actuation-stress-v0",
                    "profile_version": "v0",
                    "claim_scope": "synthetic-only",
                    "claim_boundary": "diagnostic-only",
                    "max_linear_accel_m_s2": 2.0,
                    "max_linear_decel_m_s2": 2.5,
                    "max_yaw_rate_rad_s": 1.2,
                    "max_angular_accel_rad_s2": 4.0,
                    "latency_mode": "one-step-delay",
                    "update_mode": "5hz-hold",
                },
                "latency_stress_profile": {
                    "name": "learned-policy-latency-stress-v0",
                    "profile_version": "v0",
                    "claim_scope": "synthetic-only",
                    "observation_delay_steps": 1,
                    "action_delay_steps": 1,
                    "planner_update_mode": "hold-last",
                    "planner_update_period_steps": 2,
                    "inference_timeout_ms": 200.0,
                },
                "planners": [{"key": "goal", "algo": "goal", "planner_group": "core"}],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="issue1556")

    preview_payload = json.loads(
        Path(prepared["preview_scenarios_path"]).read_text(encoding="utf-8")
    )
    assert preview_payload["scenario_count"] == 5
    assert [scenario["name"] for scenario in preview_payload["scenarios"]] == [
        "classic_overtaking_medium",
        "classic_bottleneck_high",
        "classic_cross_trap_high",
        "francis2023_blind_corner",
        "francis2023_intersection_wait",
    ]

    manifest = json.loads(
        (Path(prepared["campaign_root"]) / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    validate_payload = json.loads(
        Path(prepared["validate_config_path"]).read_text(encoding="utf-8")
    )
    assert (
        _build_preflight_validate_payload(
            cfg,
            campaign_id=prepared["campaign_id"],
            created_at_utc=prepared["created_at_utc"],
            scenarios=prepared["scenarios"],
            resolved_seeds=prepared["resolved_seeds"],
            scenario_horizons_summary=prepared["manifest_payload"]["scenario_horizons"],
            route_clearance_warnings=validate_payload["route_clearance_warnings"],
            route_clearance_warning_summary=validate_payload["route_clearance_warning_summary"],
            noise_spec=validate_payload["observation_noise"],
            noise_hash=validate_payload["observation_noise_hash"],
        )
        == validate_payload
    )
    assert (
        _build_preflight_preview_payload(
            cfg,
            campaign_id=prepared["campaign_id"],
            created_at_utc=prepared["created_at_utc"],
            scenarios=prepared["scenarios"],
            route_clearance_warnings=preview_payload["route_clearance_warnings"],
            route_clearance_warning_summary=preview_payload["route_clearance_warning_summary"],
        )
        == preview_payload
    )
    assert validate_payload["scenario_candidates"]["requested"] == [
        "classic_overtaking_medium",
        "classic_bottleneck_high",
        "classic_cross_trap_high",
        "francis2023_blind_corner",
        "francis2023_intersection_wait",
    ]
    assert validate_payload["scenario_candidates"]["resolved"] == [
        "classic_overtaking_medium",
        "classic_bottleneck_high",
        "classic_cross_trap_high",
        "francis2023_blind_corner",
        "francis2023_intersection_wait",
    ]
    assert manifest["scenario_candidates"] == [
        "classic_overtaking_medium",
        "classic_bottleneck_high",
        "classic_cross_trap_high",
        "francis2023_blind_corner",
        "francis2023_intersection_wait",
    ]
    assert manifest["synthetic_actuation_profile"]["name"] == "amv-actuation-stress-v0"
    assert manifest["latency_stress_profile"]["name"] == "learned-policy-latency-stress-v0"
    assert manifest["latency_stress_profile"]["observation_delay_ms"] is None
    assert validate_payload["latency_stress_profile"]["action_delay_steps"] == 1
    assert validate_payload["latency_stress_metrics"]["held_action_ratio"] == "not_available"
    assert preview_payload["latency_stress_profile"]["planner_update_mode"] == "hold-last"
    assert manifest["seed_policy"]["seed_set"] == "eval"


def test_prepare_campaign_preflight_applies_scenario_amv_overrides(tmp_path: Path) -> None:
    """Slice-local scenario AMV overrides should populate preview and coverage artifacts."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "classic_overtaking_medium",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "metadata": {"archetype": "classic_crossing"},
                },
                {
                    "name": "francis2023_intersection_wait",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "metadata": {"archetype": "francis2023"},
                },
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "issue_1572_preflight",
                "paper_facing": False,
                "scenario_matrix": str(scenario_path),
                "kinematics_matrix": ["differential_drive"],
                "scenario_candidates": [
                    "classic_overtaking_medium",
                    "francis2023_intersection_wait",
                ],
                "scenario_amv_overrides": {
                    "classic_overtaking_medium": {
                        "use_case": "delivery_robot",
                        "context": "sidewalk",
                        "speed_regime": "walking_speed",
                        "maneuver_type": "overtake",
                    },
                    "francis2023_intersection_wait": {
                        "use_case": "shared_space_micromobility",
                        "context": "shared_space",
                        "speed_regime": "scooter_speed",
                        "maneuver_type": "bottleneck",
                    },
                },
                "amv_profile": {
                    "coverage_enforcement": "warn",
                    "required_dimensions": {
                        "use_case": ["delivery_robot", "shared_space_micromobility"],
                        "context": ["sidewalk", "shared_space"],
                        "speed_regime": ["walking_speed", "scooter_speed"],
                        "maneuver_type": ["overtake", "bottleneck"],
                    },
                },
                "planners": [{"key": "goal", "algo": "goal", "planner_group": "core"}],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="issue1572")

    preview_payload = json.loads(
        Path(prepared["preview_scenarios_path"]).read_text(encoding="utf-8")
    )
    assert preview_payload["scenarios"][0]["amv"]["maneuver_type"] == "overtake"
    assert preview_payload["scenarios"][1]["amv"]["speed_regime"] == "scooter_speed"

    amv_payload = json.loads(Path(prepared["amv_coverage_json_path"]).read_text(encoding="utf-8"))
    assert amv_payload["status"] == "pass"
    assert amv_payload["scenario_rows"][0]["amv"]["use_case"] == "delivery_robot"
    assert amv_payload["scenario_rows"][1]["amv"]["use_case"] == "shared_space_micromobility"

    validate_payload = json.loads(
        Path(prepared["validate_config_path"]).read_text(encoding="utf-8")
    )
    assert (
        validate_payload["scenario_amv_overrides"]["classic_overtaking_medium"]["maneuver_type"]
        == "overtake"
    )
    assert (
        validate_payload["scenario_amv_overrides"]["francis2023_intersection_wait"]["speed_regime"]
        == "scooter_speed"
    )

    manifest = json.loads(
        (Path(prepared["campaign_root"]) / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["amv_coverage_status"] == "pass"
    assert manifest["scenario_amv_overrides"]["classic_overtaking_medium"]["maneuver_type"] == (
        "overtake"
    )


def test_load_campaign_scenarios_rejects_unmatched_scenario_amv_overrides(tmp_path: Path) -> None:
    """AMV overrides should fail closed when they target scenarios outside the loaded slice."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "classic_overtaking_medium",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                },
                {
                    "name": "francis2023_intersection_wait",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                },
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "issue_1572_unmatched_override",
                "scenario_matrix": str(scenario_path),
                "scenario_candidates": ["classic_overtaking_medium"],
                "scenario_amv_overrides": {
                    "francis2023_intersection_wait": {
                        "use_case": "shared_space_micromobility",
                        "context": "shared_space",
                        "speed_regime": "scooter_speed",
                        "maneuver_type": "bottleneck",
                    }
                },
                "planners": [{"key": "goal", "algo": "goal"}],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)

    with pytest.raises(
        ValueError,
        match=r"scenario_amv_overrides did not resolve .*francis2023_intersection_wait",
    ):
        _load_campaign_scenarios(cfg)


def test_load_campaign_config_parses_observation_mode_overrides(tmp_path: Path) -> None:
    """Campaign configs should support global and per-planner observation-mode overrides."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: observation_override_campaign",
                f"scenario_matrix: {scenario_path.name}",
                "observation_mode: socnav_state",
                "planners:",
                "  - key: goal_default_override",
                "    algo: goal",
                "    benchmark_profile: baseline-safe",
                "  - key: goal_explicit_override",
                "    algo: goal",
                "    benchmark_profile: baseline-safe",
                "    observation_mode: goal_state",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)

    assert cfg.observation_mode == "socnav_state"
    assert cfg.planners[0].observation_mode is None
    assert cfg.planners[1].observation_mode == "goal_state"


def test_load_campaign_config_rejects_blank_planner_observation_mode(tmp_path: Path) -> None:
    """Blank planner-level observation-mode overrides should be rejected."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: observation_blank_planner",
                f"scenario_matrix: {scenario_path.name}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    observation_mode: '   '",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Planner entry 'observation_mode' cannot be empty"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_blank_global_observation_mode(tmp_path: Path) -> None:
    """Blank campaign-level observation-mode overrides should be rejected."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: observation_blank_global",
                f"scenario_matrix: {scenario_path.name}",
                "observation_mode: '   '",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Campaign 'observation_mode' cannot be empty"):
        load_campaign_config(config_path)


def test_scenario_horizon_schedule_applies_to_loaded_campaign_scenarios(
    tmp_path: Path,
) -> None:
    """Scenario-specific horizon schedules should patch episode limits with provenance."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "sc_a",
                        "map_file": "maps/svg_maps/classic_crossing.svg",
                        "simulation_config": {"max_episode_steps": 100},
                    },
                    {
                        "name": "sc_blocked",
                        "map_file": "maps/svg_maps/classic_crossing.svg",
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    schedule_path = tmp_path / "scenario_horizons.yaml"
    schedule_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "source": "unit_test",
                "scenarios": {
                    "sc_a": {
                        "recommended_horizon_steps": 176,
                        "status": "recommended",
                        "bucket": "medium",
                    },
                    "sc_blocked": {
                        "recommended_horizon_steps": 600,
                        "status": "planner_blocked",
                        "bucket": "planner_blocked",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: scheduled_campaign",
                f"scenario_matrix: {scenario_path.as_posix()}",
                f"scenario_horizons: {schedule_path.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    scenarios = _load_campaign_scenarios(cfg)

    by_name = {scenario["name"]: scenario for scenario in scenarios}
    assert by_name["sc_a"]["simulation_config"]["max_episode_steps"] == 176
    assert by_name["sc_blocked"]["simulation_config"]["max_episode_steps"] == 600
    assert by_name["sc_a"]["metadata"]["scenario_horizon"]["status"] == "recommended"
    assert by_name["sc_blocked"]["metadata"]["scenario_horizon"]["status"] == "planner_blocked"
    assert by_name["sc_a"]["metadata"]["scenario_horizon"]["source"] == str(schedule_path.resolve())


def test_preflight_reports_scenario_horizon_schedule_summary(tmp_path: Path) -> None:
    """Preflight artifacts should expose scenario-horizon provenance and status counts."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {"name": "sc_a", "map_file": "maps/svg_maps/classic_crossing.svg"},
                    {"name": "sc_blocked", "map_file": "maps/svg_maps/classic_crossing.svg"},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    schedule_path = tmp_path / "scenario_horizons.yaml"
    schedule_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "source": "unit_test",
                "scenarios": {
                    "sc_a": {
                        "recommended_horizon_steps": 176,
                        "status": "recommended",
                        "bucket": "medium",
                    },
                    "sc_blocked": {
                        "recommended_horizon_steps": 600,
                        "status": "planner_blocked",
                        "bucket": "planner_blocked",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: scheduled_campaign",
                f"scenario_matrix: {scenario_path.as_posix()}",
                f"scenario_horizons: {schedule_path.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    prepared = prepare_campaign_preflight(
        cfg,
        output_root=tmp_path / "out",
        campaign_id="scheduled_preflight",
    )
    validate_payload = json.loads(
        Path(prepared["validate_config_path"]).read_text(encoding="utf-8")
    )
    matrix_payload = json.loads(
        Path(prepared["matrix_summary_json_path"]).read_text(encoding="utf-8")
    )

    assert validate_payload["scenario_horizons"] == {
        "path": str(schedule_path.resolve()),
        "scenario_count": 2,
        "min_horizon_steps": 176,
        "max_horizon_steps": 600,
        "status_counts": {"planner_blocked": 1, "recommended": 1},
    }
    assert matrix_payload["rows"][0]["horizon_mode"] == "scenario_horizons"
    assert matrix_payload["rows"][0]["scenario_horizons_path"] == str(schedule_path.resolve())


def test_scenario_horizon_schedule_rejects_fixed_horizon(tmp_path: Path) -> None:
    """Scenario-horizon campaigns should not silently keep a fixed global horizon."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {"scenarios": [{"name": "sc_a", "map_file": "maps/svg_maps/classic_crossing.svg"}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    schedule_path = tmp_path / "scenario_horizons.yaml"
    schedule_path.write_text(
        yaml.safe_dump(
            {
                "scenarios": {
                    "sc_a": {
                        "recommended_horizon_steps": 176,
                        "status": "recommended",
                        "bucket": "medium",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: scheduled_campaign",
                f"scenario_matrix: {scenario_path.as_posix()}",
                f"scenario_horizons: {schedule_path.as_posix()}",
                "horizon: 100",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scenario_horizons cannot be combined with fixed horizon"):
        load_campaign_config(config_path)


def test_scenario_horizon_schedule_rejects_directory_path(tmp_path: Path) -> None:
    """Scenario-horizon campaigns should fail closed when the schedule is not a file."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {"scenarios": [{"name": "sc_a", "map_file": "maps/svg_maps/classic_crossing.svg"}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    schedule_dir = tmp_path / "scenario_horizons"
    schedule_dir.mkdir()
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: scheduled_campaign",
                f"scenario_matrix: {scenario_path.as_posix()}",
                f"scenario_horizons: {schedule_dir.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Scenario horizon schedule not found"):
        load_campaign_config(config_path)


def test_load_campaign_config_parses_snqi_contract_block(tmp_path: Path) -> None:
    """Config loader should parse SNQI contract thresholds and validation fields."""
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: snqi_contract_cfg",
                f"scenario_matrix: {matrix_path.as_posix()}",
                "snqi_contract:",
                "  enabled: true",
                "  enforcement: warn",
                "  rank_alignment_warn_threshold: 0.6",
                "  rank_alignment_fail_threshold: 0.4",
                "  outcome_separation_warn_threshold: 0.1",
                "  outcome_separation_fail_threshold: 0.0",
                "  calibration_seed: 77",
                "  calibration_trials: 1234",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    assert cfg.snqi_contract.enabled is True
    assert cfg.snqi_contract.enforcement == "warn"
    assert cfg.snqi_contract.rank_alignment_warn_threshold == pytest.approx(0.6)
    assert cfg.snqi_contract.rank_alignment_fail_threshold == pytest.approx(0.4)
    assert cfg.snqi_contract.outcome_separation_warn_threshold == pytest.approx(0.1)
    assert cfg.snqi_contract.outcome_separation_fail_threshold == pytest.approx(0.0)
    assert cfg.snqi_contract.max_component_dominance_warn_threshold == pytest.approx(0.24)
    assert cfg.snqi_contract.max_component_dominance_fail_threshold == pytest.approx(0.27)
    assert cfg.snqi_contract.calibration_seed == 77
    assert cfg.snqi_contract.calibration_trials == 1234


def test_load_campaign_config_rejects_invalid_snqi_contract_thresholds(tmp_path: Path) -> None:
    """Config loader should reject inverted SNQI contract fail/warn thresholds."""
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_invalid.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: snqi_contract_invalid",
                f"scenario_matrix: {matrix_path.as_posix()}",
                "snqi_contract:",
                "  rank_alignment_warn_threshold: 0.4",
                "  rank_alignment_fail_threshold: 0.6",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rank_alignment_fail_threshold"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_inverted_dominance_thresholds(tmp_path: Path) -> None:
    """Config loader should reject dominance fail thresholds below warn thresholds."""
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_invalid_dominance.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: snqi_contract_invalid_dominance",
                f"scenario_matrix: {matrix_path.as_posix()}",
                "snqi_contract:",
                "  max_component_dominance_warn_threshold: 0.3",
                "  max_component_dominance_fail_threshold: 0.2",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_component_dominance_fail_threshold"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_non_finite_snqi_thresholds(tmp_path: Path) -> None:
    """Config loader should reject non-finite SNQI threshold values."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_snqi_non_finite.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: snqi_contract_non_finite",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "snqi_contract:",
                "  rank_alignment_warn_threshold: .nan",
                "  rank_alignment_fail_threshold: 0.4",
                "  outcome_separation_warn_threshold: 0.1",
                "  outcome_separation_fail_threshold: 0.0",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a finite float"):
        load_campaign_config(config_path)


def test_load_holonomic_camera_ready_campaign_config() -> None:
    """Holonomic camera-ready profile should stay strict and fail closed."""
    cfg = load_campaign_config(Path("configs/benchmarks/camera_ready_all_planners_holonomic.yaml"))

    assert cfg.name == "camera_ready_all_planners_holonomic"
    assert cfg.kinematics_matrix == ("holonomic",)
    assert cfg.holonomic_command_mode == "vx_vy"
    assert cfg.export_publication_bundle is True
    assert cfg.stop_on_failure is True

    planners = {planner.key: planner for planner in cfg.planners}
    assert planners["orca"].socnav_missing_prereq_policy == "fail-fast"
    assert planners["sacadrl"].socnav_missing_prereq_policy == "fail-fast"
    assert planners["socnav_sampling"].socnav_missing_prereq_policy == "fail-fast"
    assert planners["socnav_bench"].socnav_missing_prereq_policy == "fail-fast"
    assert (
        planners["ppo"].algo_config_path
        == Path("configs/baselines/ppo_15m_grid_socnav_holonomic.yaml").resolve()
    )

    ppo_cfg = yaml.safe_load(planners["ppo"].algo_config_path.read_text(encoding="utf-8"))
    assert ppo_cfg["fallback_to_goal"] is False


def test_load_campaign_config_treats_null_kinematics_matrix_as_default(
    tmp_path: Path,
) -> None:
    """YAML null kinematics matrices should use the same default as an omitted matrix."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "null_kinematics.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: null_kinematics",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "kinematics_matrix:",
                "latency_stress_profile:",
                "  name: learned-policy-latency-stress-v0",
                "  action_delay_steps: 1",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)

    assert cfg.kinematics_matrix == ("differential_drive",)


def test_load_paper_cross_kinematics_v1_campaign_config() -> None:
    """Cross-kinematics profile should be the only paper profile with a 3-mode matrix."""
    cfg = load_campaign_config(
        get_repository_root() / "configs/benchmarks/paper_cross_kinematics_v1.yaml"
    )

    assert cfg.name == "paper_cross_kinematics_v1"
    assert cfg.paper_facing is True
    assert cfg.paper_profile_version == "paper-cross-kinematics-v1"
    assert cfg.kinematics_matrix == ("differential_drive", "bicycle_drive", "holonomic")
    assert cfg.holonomic_command_mode == "vx_vy"
    assert cfg.export_publication_bundle is False
    assert cfg.stop_on_failure is False
    assert cfg.seed_policy.mode == "fixed-list"
    assert list(cfg.seed_policy.seeds) == [111]
    assert [planner.key for planner in cfg.planners] == ["goal", "social_force", "orca"]
    assert all(planner.planner_group == "core" for planner in cfg.planners)
    assert (
        cfg.comparability_mapping_path
        == Path("configs/benchmarks/alyassi_comparability_map_v1.yaml").resolve()
    )

    scenarios = _load_campaign_scenarios(cfg)
    assert [scenario["name"] for scenario in scenarios] == ["classic_cross_trap_low"]


def test_paper_cross_kinematics_v1_compatibility_manifest() -> None:
    """Compatibility manifest should record every configured planner/kinematics pair."""
    config_payload = yaml.safe_load(
        (get_repository_root() / "configs/benchmarks/paper_cross_kinematics_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest_path = get_repository_root() / config_payload["compatibility_manifest"]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["version"] == "paper-cross-kinematics-v1"
    assert manifest["kinematics"] == ["differential_drive", "bicycle_drive", "holonomic"]
    planner_keys = [planner["key"] for planner in config_payload["planners"]]
    assert set(manifest["planners"]) == set(planner_keys)
    for planner_key in planner_keys:
        support = manifest["planners"][planner_key]["support"]
        assert set(support) == set(manifest["kinematics"])
        assert all(entry["status"] == "supported" for entry in support.values())
        assert all(str(entry["reason"]).strip() for entry in support.values())

    excluded = manifest["excluded_planners"]
    assert excluded["ppo"]["status"] == "degraded"
    assert str(excluded["ppo"]["reason"]).strip()
    assert manifest["validation_contract"]["supported_pairs_must_run"] is True


def test_load_cross_kinematics_v1_campaign_config() -> None:
    """General cross-kinematics profile should stay non-paper-facing and explicit."""
    cfg = load_campaign_config(
        get_repository_root() / "configs/benchmarks/cross_kinematics_v1.yaml"
    )

    assert cfg.name == "cross_kinematics_v1"
    assert cfg.paper_facing is False
    assert cfg.paper_profile_version is None
    assert cfg.kinematics_matrix == ("differential_drive", "bicycle_drive", "holonomic")
    assert cfg.holonomic_command_mode == "vx_vy"
    assert cfg.export_publication_bundle is False
    assert cfg.stop_on_failure is False
    assert cfg.seed_policy.mode == "fixed-list"
    assert list(cfg.seed_policy.seeds) == [111]
    assert [planner.key for planner in cfg.planners] == ["goal", "social_force", "orca"]
    assert all(planner.planner_group == "core" for planner in cfg.planners)

    scenarios = _load_campaign_scenarios(cfg)
    assert [scenario["name"] for scenario in scenarios] == ["classic_cross_trap_low"]


def test_cross_kinematics_v1_compatibility_manifest() -> None:
    """General compatibility manifest should expose supported and excluded rows."""
    repo_root = Path(__file__).parents[2]
    config_payload = yaml.safe_load(
        (repo_root / "configs/benchmarks/cross_kinematics_v1.yaml").read_text(encoding="utf-8")
    )
    manifest_path = repo_root / config_payload["compatibility_manifest"]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["version"] == "cross-kinematics-v1"
    assert manifest["profile"] == "cross-kinematics-v1"
    assert manifest["kinematics"] == ["differential_drive", "bicycle_drive", "holonomic"]
    planner_keys = [planner["key"] for planner in config_payload["planners"]]
    assert set(manifest["planners"]) == set(planner_keys)
    for planner_key in planner_keys:
        support = manifest["planners"][planner_key]["support"]
        assert set(support) == set(manifest["kinematics"])
        assert all(entry["status"] == "supported" for entry in support.values())
        assert all(str(entry["reason"]).strip() for entry in support.values())

    excluded = manifest["excluded_planners"]
    assert excluded["ppo"]["status"] == "degraded"
    assert excluded["rvo"]["status"] == "unsupported"
    assert excluded["dwa"]["status"] == "unsupported"
    assert all(str(entry["reason"]).strip() for entry in excluded.values())
    assert manifest["validation_contract"]["supported_pairs_must_run"] is True
    assert manifest["validation_contract"]["unsupported_or_degraded_pairs_must_have_reason"] is True


def test_socnav_bench_reentry_probe_config_is_focused_and_fail_fast() -> None:
    """The SocNavBench re-entry probe should stay narrow and fail closed."""

    cfg = load_campaign_config(Path("configs/benchmarks/socnav_bench_reentry_probe.yaml"))

    assert cfg.name == "socnav_bench_reentry_probe"
    assert cfg.scenario_matrix_path == (
        get_repository_root() / "configs/scenarios/single/francis2023_blind_corner.yaml"
    )
    assert cfg.seed_policy.mode == "fixed-list"
    assert list(cfg.seed_policy.seeds) == [111, 112, 113]
    assert cfg.horizon == 30
    assert cfg.workers == 1

    planners = {planner.key: planner for planner in cfg.planners}
    assert list(planners) == ["goal", "socnav_bench"]
    assert planners["goal"].planner_group == "core"
    assert planners["socnav_bench"].planner_group == "experimental"
    assert planners["socnav_bench"].socnav_missing_prereq_policy == "fail-fast"


def test_issue_791_eval_aligned_ppo_config_is_serial_and_fail_closed() -> None:
    """Issue-791 benchmark candidate should not silently fallback or fork CUDA workers."""
    cfg = load_campaign_config(
        Path("configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml")
    )

    planners = {planner.key: planner for planner in cfg.planners}
    assert (
        planners["ppo"].algo_config_path
        == Path("configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml").resolve()
    )
    assert planners["ppo"].workers_override == 1

    ppo_cfg = yaml.safe_load(planners["ppo"].algo_config_path.read_text(encoding="utf-8"))
    assert (
        ppo_cfg["model_id"]
        == "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"
    )
    assert "model_path" not in ppo_cfg
    assert ppo_cfg["fallback_to_goal"] is False
    assert ppo_cfg["predictive_foresight_enabled"] is True
    assert ppo_cfg["predictive_foresight_model_id"] == "predictive_proxy_selected_v2_full"


def test_issue_857_horizon400_probe_only_changes_horizon_and_bundle_export() -> None:
    """The horizon probe should keep the leader PPO adapter but raise the benchmark horizon."""
    cfg = load_campaign_config(
        Path("configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon400_probe.yaml")
    )

    planners = {planner.key: planner for planner in cfg.planners}
    assert cfg.horizon == 400
    assert cfg.paper_facing is True
    assert cfg.export_publication_bundle is False
    assert (
        planners["ppo"].algo_config_path
        == Path("configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml").resolve()
    )
    assert planners["ppo"].workers_override == 1


def test_issue_1023_scenario_horizon_config_uses_h500_schedule() -> None:
    """Issue 1023 config should expose the h500 schedule as a paper-facing surface."""
    cfg = load_campaign_config(
        Path("configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml")
    )

    assert cfg.paper_facing is True
    assert cfg.paper_profile_version == "paper-matrix-v1"
    assert cfg.horizon is None
    assert (
        cfg.scenario_horizons_path
        == Path("configs/policy_search/scenario_horizons_h500.yaml").resolve()
    )
    assert cfg.export_publication_bundle is False
    assert all(planner.horizon_override is None for planner in cfg.planners)
    planners = {planner.key: planner for planner in cfg.planners}
    assert (
        planners["scenario_adaptive_hybrid_orca_v1"].algo_config_path
        == Path("configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml").resolve()
    )
    assert planners["scenario_adaptive_hybrid_orca_v1"].benchmark_profile == "experimental"
    assert (
        planners["hybrid_rule_v3_fast_progress_static_escape"].algo_config_path
        == Path(
            "configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml"
        ).resolve()
    )
    assert planners["hybrid_rule_v3_fast_progress_static_escape"].benchmark_profile == (
        "experimental"
    )

    scenarios = _load_campaign_scenarios(cfg)
    horizon_meta = [
        scenario["metadata"]["scenario_horizon"]
        for scenario in scenarios
        if "scenario_horizon" in scenario.get("metadata", {})
    ]
    assert len(horizon_meta) == 48
    assert sum(1 for row in horizon_meta if row["status"] == "planner_blocked") == 3
    assert {scenario["simulation_config"]["max_episode_steps"] for scenario in scenarios} <= set(
        range(1, 601)
    )


def test_sanity_v1_smoke_config_is_nominal_calibration_surface() -> None:
    """The sanity_v1 smoke config should stay narrow, non-paper-facing, and baseline-safe."""
    cfg = load_campaign_config(get_repository_root() / "configs/benchmarks/sanity_v1_smoke.yaml")

    assert cfg.paper_facing is False
    assert cfg.paper_interpretation_profile == "sanity-v1-nominal-calibration"
    assert cfg.export_publication_bundle is False
    assert cfg.kinematics_matrix == ("differential_drive",)
    assert cfg.seed_policy.mode == "fixed-list"
    assert list(cfg.seed_policy.seeds) == [111]
    assert cfg.horizon == 250
    assert [planner.key for planner in cfg.planners] == ["goal", "orca"]
    assert {planner.planner_group for planner in cfg.planners} == {"core"}
    assert {planner.benchmark_profile for planner in cfg.planners} == {"baseline-safe"}

    scenarios = _load_campaign_scenarios(cfg)
    assert [scenario["name"] for scenario in scenarios] == [
        "planner_sanity_simple",
        "empty_map_8_directions_east",
        "goal_behind_robot",
        "single_ped_crossing_orthogonal",
    ]
    assert all(scenario["seeds"] == [111] for scenario in scenarios)


def test_paper_extended_seed_configs_preserve_v1_matrix_contract() -> None:
    """Extended seed configs should change only the named seed schedule."""
    base_cfg = load_campaign_config(Path("configs/benchmarks/paper_experiment_matrix_v1.yaml"))
    s5_cfg = load_campaign_config(
        Path("configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s5.yaml")
    )
    s10_cfg = load_campaign_config(
        Path("configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s10.yaml")
    )

    for cfg, seed_set, expected_seeds in (
        (s5_cfg, "paper_eval_s5", [111, 112, 113, 114, 115]),
        (s10_cfg, "paper_eval_s10", [111, 112, 113, 114, 115, 116, 117, 118, 119, 120]),
    ):
        assert cfg.paper_facing is True
        assert cfg.paper_profile_version == base_cfg.paper_profile_version == "paper-matrix-v1"
        assert cfg.scenario_matrix_path == base_cfg.scenario_matrix_path
        assert cfg.comparability_mapping_path == base_cfg.comparability_mapping_path
        assert cfg.kinematics_matrix == base_cfg.kinematics_matrix == ("differential_drive",)
        assert cfg.seed_policy.mode == "seed-set"
        assert cfg.seed_policy.seed_set == seed_set
        assert cfg.planners == base_cfg.planners
        assert _resolved_seed_inventory(_load_campaign_scenarios(cfg)) == expected_seeds


def test_issue_1554_h500_s20_config_preserves_h500_matrix_contract() -> None:
    """Issue #1554 S20 config preserves h500 matrix except PPO CPU safety."""
    base_cfg = load_campaign_config(
        Path("configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml")
    )
    s20_cfg = load_campaign_config(
        Path("configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml")
    )

    assert s20_cfg.paper_facing is True
    assert s20_cfg.paper_profile_version == base_cfg.paper_profile_version == "paper-matrix-v1"
    assert s20_cfg.scenario_matrix_path == base_cfg.scenario_matrix_path
    assert s20_cfg.scenario_horizons_path == base_cfg.scenario_horizons_path
    assert s20_cfg.comparability_mapping_path == base_cfg.comparability_mapping_path
    assert s20_cfg.kinematics_matrix == base_cfg.kinematics_matrix == ("differential_drive",)
    assert s20_cfg.seed_policy.mode == "seed-set"
    assert s20_cfg.seed_policy.seed_set == "paper_eval_s20"

    base_planners = {planner.key: planner for planner in base_cfg.planners}
    s20_planners = {planner.key: planner for planner in s20_cfg.planners}
    assert set(s20_planners) == set(base_planners)
    for key, planner in s20_planners.items():
        if key == "ppo":
            continue
        assert planner == base_planners[key]

    assert s20_planners["ppo"].workers_override == base_planners["ppo"].workers_override == 1
    assert s20_planners["ppo"].algo == base_planners["ppo"].algo == "ppo"
    assert (
        s20_planners["ppo"].algo_config_path
        == Path("configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml").resolve()
    )
    ppo_cfg = yaml.safe_load(s20_planners["ppo"].algo_config_path.read_text(encoding="utf-8"))
    assert (
        ppo_cfg["model_id"]
        == "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"
    )
    assert ppo_cfg["device"] == "cpu"
    assert ppo_cfg["predictive_foresight_device"] == "cpu"
    assert ppo_cfg["fallback_to_goal"] is False
    assert _resolved_seed_inventory(_load_campaign_scenarios(s20_cfg)) == list(range(111, 131))


def test_issue_821_extended_matrix_is_evidence_only_until_release_doi_exists() -> None:
    """Issue 821 matrix should not export bundles while its DOI is a placeholder."""
    cfg = load_campaign_config(
        Path("configs/benchmarks/paper_experiment_matrix_v1_issue_821_extended.yaml")
    )

    assert cfg.paper_facing is True
    assert cfg.export_publication_bundle is False
    assert cfg.release_tag == "{release_tag}"
    assert cfg.doi == "10.5281/zenodo.<record-id>"


def test_sha256_file_raises_clear_error_for_unreadable_path(tmp_path: Path) -> None:
    """Hash helper should raise a path-specific error for missing or unreadable files."""
    missing_path = tmp_path / "missing.json"

    with pytest.raises(RuntimeError, match="Failed to hash file"):
        _sha256_file(missing_path)


def test_artifact_writers_preserve_stable_formatting_and_injection_guards(
    tmp_path: Path,
) -> None:
    """Artifact writers should preserve JSON formatting and escape table/CSV injection cases."""
    json_path = tmp_path / "nested" / "payload.json"
    _write_json(json_path, {"z": 1, "a": {"b": 2}})

    assert json_path.read_text(encoding="utf-8") == '{\n  "z": 1,\n  "a": {\n    "b": 2\n  }\n}\n'

    csv_path = tmp_path / "nested" / "tables" / "rows.csv"
    _write_csv(
        csv_path,
        [
            {"name": "=SUM(1,1)", "notes": "plain"},
            {"name": "+cmd", "notes": "@handle"},
        ],
    )

    csv_rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    assert csv_rows == [
        {"name": "'=SUM(1,1)", "notes": "plain"},
        {"name": "'+cmd", "notes": "'@handle"},
    ]


def test_table_artifact_writer_projects_headers_and_escapes_markdown(
    tmp_path: Path,
) -> None:
    """Table artifact writer should use explicit CSV headers and markdown-safe cells."""
    csv_path, md_path = _write_table_artifacts(
        tmp_path,
        "summary",
        [
            {
                "planner": "goal|baseline",
                "status": "ok\naccepted",
                "notes": None,
                "ignored": "not exported",
            }
        ],
        headers=("planner", "status", "notes", "missing"),
    )

    assert csv_path == tmp_path / "summary.csv"
    assert md_path == tmp_path / "summary.md"
    assert list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8")))) == [
        {"planner": "goal|baseline", "status": "ok\naccepted", "notes": "", "missing": ""}
    ]
    assert md_path.read_text(encoding="utf-8") == (
        "| planner | status | notes | missing |\n"
        "|---|---|---|---|\n"
        "| goal\\|baseline | ok accepted |  |  |\n"
    )

    nested_csv, nested_md = _write_table_artifacts(
        tmp_path / "nested" / "reports",
        "summary",
        [{"planner": "goal"}],
        headers=("planner",),
    )
    assert nested_csv.exists()
    assert nested_md.exists()


def test_run_campaign_writes_core_artifacts(tmp_path: Path, monkeypatch):  # noqa: PLR0915
    """Campaign runner should emit summary artifacts and publication metadata."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign_runner",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "paper_interpretation_profile: baseline-ready-core",
                "preview_scenario_limit: 0",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    benchmark_profile: baseline-safe",
                "    observation_mode: socnav_state",
                "  - key: ppo",
                "    algo: ppo",
                "    benchmark_profile: experimental",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    assert cfg.scenario_matrix_path == scenario_abs
    run_batch_calls: list[dict[str, object]] = []

    def _fake_run_batch(
        scenarios_or_path,
        out_path,
        schema_path,
        *,
        algo,
        benchmark_profile,
        **kwargs,
    ):
        """Write one episode record and return readiness/preflight metadata."""
        scenarios = list(scenarios_or_path) if isinstance(scenarios_or_path, list) else []
        _ = schema_path
        run_batch_calls.append({"algo": algo, **kwargs})
        if scenarios:
            map_file = scenarios[0].get("map_file")
            if isinstance(map_file, str):
                assert not Path(map_file).is_absolute()
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "episode_id": f"e-{algo}-0",
                "scenario_id": "mock",
                "seed": 111,
                "scenario_params": {"algo": algo, "metadata": {"archetype": "crossing"}},
                "metrics": {"success": 1.0, "collisions": 0.0, "near_misses": 0.0},
                "algorithm_metadata": {"algorithm": algo, "status": "ok"},
            },
        ]
        with out_file.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return {
            "total_jobs": len(records),
            "written": len(records),
            "failed_jobs": 0,
            "failures": [],
            "out_path": str(out_file),
            "algorithm_readiness": {
                "name": algo,
                "tier": "experimental" if benchmark_profile == "experimental" else "baseline-ready",
                "profile": benchmark_profile,
            },
            "preflight": {
                "status": "fallback" if algo == "ppo" else "ok",
                "learned_policy_contract": (
                    {
                        "status": "warn",
                        "critical_mismatches": ["obs_mode=image mismatch"],
                        "warnings": [],
                    }
                    if algo == "ppo"
                    else {"status": "not_applicable"}
                ),
            },
        }

    def _fake_compute_aggregates_with_ci(
        records,
        *,
        group_by,
        bootstrap_samples,
        bootstrap_confidence,
        bootstrap_seed,
    ):
        """Return deterministic aggregate metrics for campaign report generation."""
        _ = records
        _ = group_by
        _ = bootstrap_samples
        _ = bootstrap_confidence
        _ = bootstrap_seed
        return {
            "mock_group": {
                "success": {"mean": 1.0, "mean_ci": [1.0, 1.0]},
                "collisions": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
                "near_misses": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
                "time_to_goal_norm": {"mean": 0.5, "mean_ci": [0.4, 0.6]},
                "path_efficiency": {"mean": 0.9, "mean_ci": [0.8, 0.95]},
                "comfort_exposure": {"mean": 0.2, "mean_ci": [0.1, 0.3]},
                "jerk_mean": {"mean": 0.1, "mean_ci": [0.08, 0.12]},
                "snqi": {"mean": 0.7, "mean_ci": [0.65, 0.75]},
            },
            "_meta": {"warnings": [], "missing_algorithms": []},
        }

    def _fake_export_publication_bundle(
        run_dir,
        out_dir,
        *,
        bundle_name,
        include_videos,
        repository_url,
        release_tag,
        doi,
        overwrite,
    ):
        """Return a minimal publication bundle manifest payload."""
        _ = run_dir
        _ = out_dir
        _ = bundle_name
        _ = include_videos
        _ = repository_url
        _ = release_tag
        _ = doi
        _ = overwrite
        bundle_dir = tmp_path / "publication" / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        archive_path = tmp_path / "publication" / "bundle.tar.gz"
        archive_path.write_text("archive", encoding="utf-8")
        manifest_path = bundle_dir / "publication_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        checksums_path = bundle_dir / "checksums.sha256"
        checksums_path.write_text("", encoding="utf-8")
        return PublicationBundleResult(
            bundle_dir=bundle_dir,
            archive_path=archive_path,
            manifest_path=manifest_path,
            checksums_path=checksums_path,
            file_count=3,
            total_bytes=7,
        )

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.export_publication_bundle",
        lambda *args, **kwargs: pytest.fail(
            "Publication export should be skipped when snqi_contract hard-fails."
        ),
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.compute_aggregates_with_ci",
        _fake_compute_aggregates_with_ci,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.export_publication_bundle",
        _fake_export_publication_bundle,
    )

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="test")
    assert run_batch_calls[0]["observation_mode"] == "socnav_state"

    campaign_root = Path(result["campaign_root"])
    assert campaign_root.exists()
    assert (campaign_root / "campaign_manifest.json").exists()
    assert (campaign_root / "reports" / "campaign_summary.json").exists()
    assert (campaign_root / "reports" / "campaign_table.csv").exists()
    assert (campaign_root / "reports" / "campaign_table.md").exists()
    assert (campaign_root / "reports" / "campaign_table_core.csv").exists()
    assert (campaign_root / "reports" / "campaign_table_core.md").exists()
    assert (campaign_root / "reports" / "campaign_table_experimental.csv").exists()
    assert (campaign_root / "reports" / "campaign_table_experimental.md").exists()
    assert (campaign_root / "reports" / "matrix_summary.csv").exists()
    assert (campaign_root / "reports" / "matrix_summary.json").exists()
    assert (campaign_root / "reports" / "amv_coverage_summary.json").exists()
    assert (campaign_root / "reports" / "amv_coverage_summary.md").exists()
    assert (campaign_root / "reports" / "comparability_matrix.json").exists()
    assert (campaign_root / "reports" / "comparability_matrix.md").exists()
    assert (campaign_root / "reports" / "seed_variability_by_scenario.json").exists()
    assert (campaign_root / "reports" / "seed_variability_by_scenario.csv").exists()
    assert (campaign_root / "reports" / "seed_episode_rows.csv").exists()
    assert (campaign_root / "reports" / "statistical_sufficiency.json").exists()
    assert (campaign_root / "reports" / "snqi_diagnostics.json").exists()
    assert (campaign_root / "reports" / "snqi_diagnostics.md").exists()
    assert (campaign_root / "reports" / "snqi_sensitivity.csv").exists()
    assert (campaign_root / "reports" / "scenario_breakdown.csv").exists()
    assert (campaign_root / "reports" / "scenario_breakdown.md").exists()
    assert (campaign_root / "reports" / "scenario_family_breakdown.csv").exists()
    assert (campaign_root / "reports" / "scenario_family_breakdown.md").exists()
    assert (campaign_root / "reports" / "kinematics_parity_table.csv").exists()
    assert (campaign_root / "reports" / "kinematics_parity_table.md").exists()
    assert (campaign_root / "reports" / "kinematics_skipped_combinations.csv").exists()
    assert (campaign_root / "reports" / "kinematics_skipped_combinations.md").exists()
    assert (campaign_root / "reports" / "campaign_report.md").exists()
    assert (campaign_root / "preflight" / "validate_config.json").exists()
    assert (campaign_root / "preflight" / "preview_scenarios.json").exists()
    preview_payload = json.loads(
        (campaign_root / "preflight" / "preview_scenarios.json").read_text(encoding="utf-8")
    )
    assert preview_payload["truncated"] is True
    assert preview_payload["total_scenarios"] == 1
    assert preview_payload["preview_limit"] == 0
    assert preview_payload["scenarios"] == []
    report_text = (campaign_root / "reports" / "campaign_report.md").read_text(encoding="utf-8")
    assert "Campaign status: `accepted_unavailable_only`" in report_text
    assert "Campaign execution status: `completed`" in report_text
    assert "Evidence status: `partial`" in report_text
    assert "Readiness & Degraded/Fallback Status" in report_text
    assert "SocNav Strict-vs-Fallback Disclosure" in report_text
    assert "## Accepted Unavailable/Excluded Planners" in report_text
    assert "## Unexpected Failed/Partial Planners" in report_text
    assert "No unexpected failed/partial planners." in report_text
    assert report_text.index("## Accepted Unavailable/Excluded Planners") < report_text.index(
        "## Campaign Warnings"
    )
    assert report_text.index("## Unexpected Failed/Partial Planners") < report_text.index(
        "## Campaign Warnings"
    )
    assert "fallback" in report_text
    assert "learned contract" in report_text
    table_md = (campaign_root / "reports" / "campaign_table.md").read_text(encoding="utf-8")
    assert "availability_status" in table_md
    assert "benchmark_success" in table_md
    assert "availability_reason" in table_md
    assert "readiness_status" in table_md
    assert "learned_policy_contract_status" in table_md
    assert "socnav_prereq_policy" in table_md
    assert "planner_group" in table_md
    assert "kinematics" in table_md
    run_meta = json.loads((campaign_root / "run_meta.json").read_text(encoding="utf-8"))
    assert "seed_policy" in run_meta
    assert "resolved_seeds" in run_meta["seed_policy"]
    assert run_meta["preflight_artifacts"]["validate_config"].endswith(
        "preflight/validate_config.json"
    )
    summary_payload = json.loads(
        (campaign_root / "reports" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "accepted_unavailable_only"
    assert result["status_reason"] == (
        "campaign contains accepted unavailable/excluded rows and no unexpected failed rows"
    )
    assert result["exit_code"] == 3
    assert result["campaign_execution_status"] == "completed"
    assert result["evidence_status"] == "partial"
    assert result["row_status_summary"] == {
        "successful_evidence_rows": 1,
        "accepted_unavailable_rows": 1,
        "unexpected_failed_rows": 0,
        "fallback_or_degraded_rows": 1,
    }
    assert summary_payload["campaign"]["benchmark_success"] is False
    assert summary_payload["campaign"]["status"] == "accepted_unavailable_only"
    assert summary_payload["campaign"]["campaign_execution_status"] == "completed"
    assert summary_payload["campaign"]["evidence_status"] == "partial"
    assert summary_payload["campaign"]["row_status_summary"] == {
        "successful_evidence_rows": 1,
        "accepted_unavailable_rows": 1,
        "unexpected_failed_rows": 0,
        "fallback_or_degraded_rows": 1,
    }
    assert summary_payload["campaign"]["status_reason"] == (
        "campaign contains accepted unavailable/excluded rows and no unexpected failed rows"
    )
    assert summary_payload["campaign"]["accepted_unavailable_runs"] == 1
    assert summary_payload["campaign"]["unexpected_failed_runs"] == 0
    assert summary_payload["campaign"]["non_success_runs"] == 1
    assert summary_payload["campaign"]["exit_code"] == 3
    ppo_row = next(row for row in summary_payload["planner_rows"] if row["algo"] == "ppo")
    assert ppo_row["status"] == "not_available"
    assert ppo_row["availability_status"] == "not_available"
    assert ppo_row["benchmark_success"] == "false"
    assert summary_payload["campaign"]["paper_interpretation_profile"] == "baseline-ready-core"
    assert summary_payload["artifacts"]["matrix_summary_json"].endswith(
        "reports/matrix_summary.json"
    )
    assert summary_payload["artifacts"]["matrix_summary_csv"].endswith("reports/matrix_summary.csv")
    assert summary_payload["artifacts"]["amv_coverage_json"].endswith(
        "reports/amv_coverage_summary.json"
    )
    assert summary_payload["artifacts"]["comparability_json"].endswith(
        "reports/comparability_matrix.json"
    )
    assert summary_payload["artifacts"]["seed_variability_json"].endswith(
        "reports/seed_variability_by_scenario.json"
    )
    assert summary_payload["artifacts"]["seed_variability_csv"].endswith(
        "reports/seed_variability_by_scenario.csv"
    )
    assert summary_payload["artifacts"]["seed_episode_rows_csv"].endswith(
        "reports/seed_episode_rows.csv"
    )
    assert summary_payload["artifacts"]["statistical_sufficiency_json"].endswith(
        "reports/statistical_sufficiency.json"
    )
    assert summary_payload["artifacts"]["snqi_diagnostics_json"].endswith(
        "reports/snqi_diagnostics.json"
    )
    assert summary_payload["artifacts"]["snqi_diagnostics_md"].endswith(
        "reports/snqi_diagnostics.md"
    )
    assert summary_payload["artifacts"]["snqi_sensitivity_csv"].endswith(
        "reports/snqi_sensitivity.csv"
    )
    assert summary_payload["campaign"]["snqi_contract_status"] in {"pass", "warn", "fail"}
    assert "snqi_weights_version" in summary_payload["campaign"]
    assert "snqi_weights_sha256" in summary_payload["campaign"]
    assert "snqi_baseline_version" in summary_payload["campaign"]
    assert "snqi_baseline_sha256" in summary_payload["campaign"]
    assert "release_url" in summary_payload["campaign"]
    assert "release_asset_url" in summary_payload["campaign"]
    assert "doi_url" in summary_payload["campaign"]
    seed_variability_payload = json.loads(
        (campaign_root / "reports" / "seed_variability_by_scenario.json").read_text(
            encoding="utf-8"
        )
    )
    assert seed_variability_payload["confidence"]["method"] == "bootstrap_mean_over_seed_means"
    assert seed_variability_payload["source"]["campaign_manifest_path"].endswith(
        "campaign_manifest.json"
    )
    assert seed_variability_payload["source"]["run_meta_path"].endswith("run_meta.json")
    assert seed_variability_payload["source"]["episodes_paths"]
    assert all("episodes" in path for path in seed_variability_payload["source"]["episodes_paths"])
    planner_keys = {row["planner_key"] for row in seed_variability_payload["rows"]}
    assert planner_keys == {"goal"}
    success_summary = seed_variability_payload["rows"][0]["summary"]["success"]
    assert "ci_low" in success_summary
    assert "ci_high" in success_summary
    assert "ci_half_width" in success_summary
    seed_episode_rows_csv = (campaign_root / "reports" / "seed_episode_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "scenario_id" in seed_episode_rows_csv
    assert "planner_key" in seed_episode_rows_csv
    assert "kinematics" in seed_episode_rows_csv
    assert "seed" in seed_episode_rows_csv
    assert "repeat_index" in seed_episode_rows_csv
    seed_episode_rows = list(csv.DictReader(io.StringIO(seed_episode_rows_csv)))
    assert seed_episode_rows
    assert {row["planner_key"] for row in seed_episode_rows} == {"goal"}
    assert {row["algo"] for row in seed_episode_rows} == {"goal"}
    statistical_sufficiency_payload = json.loads(
        (campaign_root / "reports" / "statistical_sufficiency.json").read_text(encoding="utf-8")
    )
    assert (
        statistical_sufficiency_payload["confidence"]["method"] == "bootstrap_mean_over_seed_means"
    )
    assert statistical_sufficiency_payload["row_count"] == 1
    assert statistical_sufficiency_payload["rows"][0]["kinematics"] == "differential_drive"
    expected_confidence = seed_variability_payload["confidence"]
    assert run_meta["seed_variability"]["bootstrap_method"] == expected_confidence["method"]
    assert run_meta["seed_variability"]["bootstrap_level"] == pytest.approx(
        expected_confidence["confidence"]
    )
    assert (
        run_meta["seed_variability"]["bootstrap_samples"]
        == expected_confidence["bootstrap_samples"]
    )
    assert run_meta["seed_variability"]["seed"] == expected_confidence["bootstrap_seed"]
    campaign_manifest = json.loads(
        (campaign_root / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        campaign_manifest["seed_variability"]["bootstrap_method"] == expected_confidence["method"]
    )
    assert campaign_manifest["seed_variability"]["bootstrap_level"] == pytest.approx(
        expected_confidence["confidence"]
    )
    assert (
        campaign_manifest["seed_variability"]["bootstrap_samples"]
        == expected_confidence["bootstrap_samples"]
    )
    assert campaign_manifest["seed_variability"]["seed"] == expected_confidence["bootstrap_seed"]
    assert result["publication_bundle"] is None
    assert "publication_bundle" not in summary_payload
    assert any(
        "Publication bundle export skipped because benchmark_success=false." in warning
        for warning in summary_payload["warnings"]
    )


def test_run_campaign_writes_synthetic_actuation_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic actuation slices should emit diagnostic artifacts and propagate profile metadata."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: classic_overtaking_medium\n  map_file: maps/svg_maps/classic_crossing.svg\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "issue_1556_runner",
                "paper_facing": False,
                "scenario_matrix": str(scenario_path),
                "kinematics_matrix": ["differential_drive"],
                "scenario_candidates": ["classic_overtaking_medium"],
                "seed_policy": {"mode": "fixed-list", "seeds": [111]},
                "synthetic_actuation_profile": {
                    "name": "amv-actuation-stress-v0",
                    "profile_version": "v0",
                    "claim_scope": "synthetic-only",
                    "claim_boundary": "diagnostic-only",
                    "max_linear_accel_m_s2": 2.0,
                    "max_linear_decel_m_s2": 2.5,
                    "max_yaw_rate_rad_s": 1.2,
                    "max_angular_accel_rad_s2": 4.0,
                    "latency_mode": "one-step-delay",
                    "update_mode": "5hz-hold",
                },
                "latency_stress_profile": {
                    "name": "learned-policy-latency-stress-v0",
                    "profile_version": "v0",
                    "claim_scope": "synthetic-only",
                    "observation_delay_steps": 1,
                    "action_delay_steps": 1,
                    "planner_update_mode": "hold-last",
                    "planner_update_period_steps": 2,
                    "inference_timeout_ms": 200.0,
                },
                "planners": [{"key": "goal", "algo": "goal", "planner_group": "core"}],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    run_batch_calls: list[dict[str, object]] = []

    def _fake_run_batch(
        scenarios_or_path,
        out_path,
        schema_path,
        *,
        algo,
        benchmark_profile,
        **kwargs,
    ):
        del scenarios_or_path, schema_path, benchmark_profile
        run_batch_calls.append({"algo": algo, **kwargs})
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "episode_id": "e-goal-0",
            "scenario_id": "classic_overtaking_medium",
            "seed": 111,
            "scenario_params": {
                "algo": algo,
                "synthetic_actuation_profile": kwargs["synthetic_actuation_profile"],
                "latency_stress_profile": kwargs["latency_stress_profile"],
            },
            "metrics": {
                "success": 1.0,
                "collisions": 0.0,
                "near_misses": 0.0,
                "command_clip_fraction": 0.25,
                "yaw_rate_saturation_fraction": 0.5,
                "signed_braking_peak_m_s2": -1.5,
                "stalled_time": 0.0,
                "failure_to_progress": 0.0,
                "jerk_mean": 0.1,
                "jerk_max": 0.2,
                "curvature_mean": 0.3,
                "energy": 1.0,
                "min_clearance": 0.8,
                "time_to_collision_min": 1.2,
                "time_to_goal_norm": 0.7,
                "velocity_max": 1.0,
                "acceleration_max": 2.0,
                "total_collision_count": 0.0,
            },
            "algorithm_metadata": {
                "algorithm": algo,
                "status": "ok",
                "synthetic_actuation": {
                    "profile": kwargs["synthetic_actuation_profile"],
                    "summary": {
                        "status": "ok",
                        "command_clip_fraction": 0.25,
                        "yaw_rate_saturation_fraction": 0.5,
                        "signed_braking_peak_m_s2": -1.5,
                    },
                },
            },
        }
        out_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "out_path": str(out_file),
            "algorithm_readiness": {
                "name": algo,
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
            "preflight": {
                "status": "ok",
                "synthetic_actuation_profile": kwargs["synthetic_actuation_profile"],
                "latency_stress_profile": kwargs["latency_stress_profile"],
                "latency_stress_metrics": {"held_action_ratio": "not_available"},
            },
            "synthetic_actuation_profile": kwargs["synthetic_actuation_profile"],
            "latency_stress_profile": kwargs["latency_stress_profile"],
            "latency_stress_metrics": {"held_action_ratio": "not_available"},
        }

    def _fake_compute_aggregates_with_ci(
        records,
        *,
        group_by,
        bootstrap_samples,
        bootstrap_confidence,
        bootstrap_seed,
    ):
        del records, group_by, bootstrap_samples, bootstrap_confidence, bootstrap_seed
        return {
            "mock_group": {
                "success": {"mean": 1.0, "mean_ci": [1.0, 1.0]},
                "time_to_goal_norm": {"mean": 0.7, "mean_ci": [0.6, 0.8]},
                "total_collision_count": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
                "near_misses": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
                "min_clearance": {"mean": 0.8, "mean_ci": [0.8, 0.8]},
                "time_to_collision_min": {"mean": 1.2, "mean_ci": [1.2, 1.2]},
                "velocity_max": {"mean": 1.0, "mean_ci": [1.0, 1.0]},
                "acceleration_max": {"mean": 2.0, "mean_ci": [2.0, 2.0]},
                "jerk_mean": {"mean": 0.1, "mean_ci": [0.1, 0.1]},
                "jerk_max": {"mean": 0.2, "mean_ci": [0.2, 0.2]},
                "curvature_mean": {"mean": 0.3, "mean_ci": [0.3, 0.3]},
                "energy": {"mean": 1.0, "mean_ci": [1.0, 1.0]},
                "stalled_time": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
                "failure_to_progress": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
                "command_clip_fraction": {"mean": 0.25, "mean_ci": [0.25, 0.25]},
                "yaw_rate_saturation_fraction": {"mean": 0.5, "mean_ci": [0.5, 0.5]},
                "signed_braking_peak_m_s2": {"mean": -1.5, "mean_ci": [-1.5, -1.5]},
            },
            "_meta": {"warnings": [], "missing_algorithms": []},
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.compute_aggregates_with_ci",
        _fake_compute_aggregates_with_ci,
    )

    result = run_campaign(cfg, output_root=tmp_path / "out", label="issue1556")
    campaign_root = Path(result["campaign_root"])
    summary_payload = json.loads(
        (campaign_root / "reports" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    actuation_payload = json.loads(
        (campaign_root / "reports" / "actuation_envelope_summary.json").read_text(encoding="utf-8")
    )

    assert run_batch_calls[0]["synthetic_actuation_profile"]["name"] == "amv-actuation-stress-v0"
    assert run_batch_calls[0]["latency_stress_profile"]["name"] == (
        "learned-policy-latency-stress-v0"
    )
    assert summary_payload["campaign"]["synthetic_actuation_profile"]["name"] == (
        "amv-actuation-stress-v0"
    )
    assert summary_payload["campaign"]["latency_stress_profile"]["action_delay_steps"] == 1
    assert summary_payload["campaign"]["latency_stress_metrics"]["held_action_ratio"] == (
        "not_available"
    )
    assert summary_payload["campaign"]["benchmark_success"] is False
    assert summary_payload["campaign"]["evidence_status"] == "blocked"
    assert summary_payload["artifacts"]["actuation_envelope_json"].endswith(
        "reports/actuation_envelope_summary.json"
    )
    assert summary_payload["artifacts"]["actuation_envelope_md"].endswith(
        "reports/actuation_envelope_summary.md"
    )
    assert actuation_payload["claim_boundary"] == "diagnostic-only"
    assert actuation_payload["synthetic_actuation_profile"]["update_mode"] == "5hz-hold"
    assert "metric_means" in actuation_payload["rows"][0]
    assert "metrics" not in actuation_payload["rows"][0]
    assert actuation_payload["rows"][0]["saturation_metrics"]["command_clip_fraction"] == "0.2500"


def test_load_campaign_config_uses_repo_default_seed_sets_path(tmp_path: Path):
    """Seed-set mode without explicit seed_sets_path should use repository default path."""
    config_dir = tmp_path / "cfg" / "nested"
    config_dir.mkdir(parents=True)

    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (config_dir / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = config_dir / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_seed_set_default_path",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: seed-set",
                "  seed_set: canonical",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    assert (
        cfg.seed_policy.seed_sets_path == (get_repository_root() / DEFAULT_SEED_SETS_PATH).resolve()
    )
    assert cfg.kinematics_matrix == ("differential_drive",)


def test_load_seed_variability_pilot_config_and_scenarios() -> None:
    """Seed-variability pilot config should stay narrow and deterministic."""
    cfg = load_campaign_config(Path("configs/benchmarks/paper_seed_variability_pilot_v1.yaml"))

    assert cfg.paper_facing is True
    assert cfg.paper_profile_version == "paper-seed-variability-v1"
    assert cfg.export_publication_bundle is False
    assert cfg.kinematics_matrix == ("differential_drive",)
    assert cfg.seed_policy.mode == "fixed-list"
    assert list(cfg.seed_policy.seeds) == [111, 112, 113, 114, 115, 116, 117, 118]
    assert [planner.key for planner in cfg.planners] == ["orca", "ppo"]

    scenarios = _load_campaign_scenarios(cfg)
    assert [scenario["name"] for scenario in scenarios] == [
        "classic_cross_trap_low",
        "classic_head_on_corridor_low",
        "classic_overtaking_low",
        "classic_t_intersection_low",
    ]
    assert all(
        scenario["seeds"] == [111, 112, 113, 114, 115, 116, 117, 118] for scenario in scenarios
    )


def test_load_campaign_scenarios_converts_absolute_repo_map_path_to_relative(tmp_path: Path):
    """Scenario map paths under repository root should normalize to repo-relative form."""
    repo_root = get_repository_root().resolve()
    abs_map = (repo_root / "maps" / "svg_maps" / "classic_crossing.svg").resolve()
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        f"- name: smoke\n  map_file: {abs_map.as_posix()}\n  seeds: [1]\n",
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        "\n".join(
            [
                "name: map_path_norm",
                f"scenario_matrix: {matrix_path.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(campaign_path)
    scenarios = _load_campaign_scenarios(cfg)
    assert scenarios
    map_file = scenarios[0].get("map_file")
    assert isinstance(map_file, str)
    assert map_file == "maps/svg_maps/classic_crossing.svg"


def test_run_campaign_stops_on_partial_failure_when_configured(tmp_path: Path, monkeypatch) -> None:
    """Campaign should stop after first partial-failure when stop_on_failure is enabled."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "campaign_stop_on_partial.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign_stop_on_partial",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "stop_on_failure: true",
                "planners:",
                "  - key: prediction_planner",
                "    algo: prediction_planner",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    call_order: list[str] = []

    def _fake_run_batch(
        scenarios_or_path,
        out_path,
        schema_path,
        *,
        algo,
        benchmark_profile,
        **kwargs,
    ):
        """Record planner execution order while returning a successful batch."""
        _ = scenarios_or_path
        _ = out_path
        _ = schema_path
        _ = benchmark_profile
        _ = kwargs
        call_order.append(algo)
        if algo == "prediction_planner":
            return {
                "total_jobs": 1,
                "written": 0,
                "failed_jobs": 1,
                "failures": [{"scenario_id": "mock", "seed": 111, "error": "mock"}],
                "preflight": {
                    "status": "ok",
                    "learned_policy_contract": {"status": "not_applicable"},
                },
            }
        raise AssertionError("run_batch must not be called for planners after partial-failure")

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="stop_partial")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    planner_rows = summary_payload["planner_rows"]

    assert call_order == ["prediction_planner"]
    assert len(planner_rows) == 1
    assert planner_rows[0]["planner_key"] == "prediction_planner"
    assert planner_rows[0]["status"] == "partial-failure"


def test_run_campaign_continues_after_not_available_when_stop_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    """Accepted unavailable rows should not halt later planners even with stop_on_failure enabled."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "campaign_stop_on_not_available.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign_stop_on_not_available",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "stop_on_failure: true",
                "planners:",
                "  - key: ppo",
                "    algo: ppo",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    call_order: list[str] = []

    def _fake_run_batch(
        scenarios_or_path,
        out_path,
        schema_path,
        *,
        algo,
        benchmark_profile,
        **kwargs,
    ):
        """Fail the PPO planner to exercise fail-fast campaign behavior."""
        del scenarios_or_path, out_path, schema_path, benchmark_profile, kwargs
        call_order.append(algo)
        if algo == "ppo":
            return {
                "total_jobs": 1,
                "written": 1,
                "failed_jobs": 0,
                "failures": [],
                "preflight": {
                    "status": "fallback",
                    "learned_policy_contract": {
                        "critical_mismatches": ["obs_mode=image mismatch"],
                    },
                },
            }
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {
                "status": "ok",
                "learned_policy_contract": {"status": "not_applicable"},
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="stop_not_available")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    planner_rows = summary_payload["planner_rows"]

    assert call_order == ["ppo", "goal"]
    assert len(planner_rows) == 2
    assert planner_rows[0]["planner_key"] == "goal"
    assert planner_rows[1]["planner_key"] == "ppo"
    ppo_row = next(row for row in planner_rows if row["planner_key"] == "ppo")
    goal_row = next(row for row in planner_rows if row["planner_key"] == "goal")
    assert ppo_row["status"] == "not_available"
    assert goal_row["status"] == "ok"
    assert any("obs_mode=image mismatch" in warning for warning in summary_payload["warnings"])
    assert not any("Campaign halted early" in warning for warning in summary_payload["warnings"])


def test_run_campaign_continues_after_failure_when_stop_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Campaign should continue remaining planners when stop_on_failure is disabled."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "campaign_continue_on_failure.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign_continue_on_failure",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "stop_on_failure: false",
                "planners:",
                "  - key: prediction_planner",
                "    algo: prediction_planner",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    call_order: list[str] = []

    def _fake_run_batch(
        scenarios_or_path,
        out_path,
        schema_path,
        *,
        algo,
        benchmark_profile,
        **kwargs,
    ):
        """Fail the prediction planner to exercise core-planner fail-fast behavior."""
        del scenarios_or_path, out_path, schema_path, benchmark_profile, kwargs
        call_order.append(algo)
        if algo == "prediction_planner":
            return {
                "total_jobs": 1,
                "written": 0,
                "failed_jobs": 1,
                "failures": [{"scenario_id": "mock", "seed": 111, "error": "worker crash"}],
                "preflight": {
                    "status": "ok",
                    "learned_policy_contract": {"status": "not_applicable"},
                },
            }
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {
                "status": "ok",
                "learned_policy_contract": {"status": "not_applicable"},
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="continue_failure")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    planner_rows = summary_payload["planner_rows"]

    assert call_order == ["prediction_planner", "goal"]
    assert len(planner_rows) == 2
    failed_row = next(row for row in planner_rows if row["planner_key"] == "prediction_planner")
    assert failed_row["status"] == "partial-failure"
    assert failed_row["most_likely_failure_reason"] == "worker crash"
    assert result["status"] == "unexpected_failure"
    assert result["exit_code"] == 2
    assert summary_payload["campaign"]["status"] == "unexpected_failure"
    assert summary_payload["campaign"]["accepted_unavailable_runs"] == 0
    assert summary_payload["campaign"]["unexpected_failed_runs"] == 1
    assert any(
        "most_likely_reason='worker crash'" in warning for warning in summary_payload["warnings"]
    )


def test_run_campaign_counts_existing_records_when_resumed_attempt_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Existing episode records should stay visible when a resumed attempt fails."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "campaign_resume_failure.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign_resume_failure",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "resume: true",
                "stop_on_failure: false",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(
        scenarios_or_path,
        out_path,
        schema_path,
        *,
        algo,
        benchmark_profile,
        **kwargs,
    ):
        """Write an episode artifact for release-readiness path tests."""
        del scenarios_or_path, schema_path, algo, benchmark_profile, kwargs
        Path(out_path).write_text(
            json.dumps(
                {
                    "scenario_id": "smoke",
                    "seed": 111,
                    "termination_reason": "success",
                    "metrics": {"success": 1.0, "collisions": 0.0, "snqi": 0.5},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "failed",
            "total_jobs": 1,
            "written": 0,
            "failed_jobs": 1,
            "failures": [{"scenario_id": "smoke", "seed": 111, "error": "resume crash"}],
            "preflight": {
                "status": "ok",
                "learned_policy_contract": {"status": "not_applicable"},
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="resume_failure")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    planner_row = summary_payload["planner_rows"][0]

    assert planner_row["status"] == "failed"
    assert planner_row["episodes"] == 1
    assert planner_row["most_likely_failure_reason"] == "resume crash"


def test_write_campaign_report_escapes_markdown_cells(tmp_path: Path) -> None:
    """Markdown report tables should escape raw cell separators from planner metadata."""
    report_path = tmp_path / "campaign_report.md"
    payload = {
        "campaign": {"campaign_id": "c1"},
        "warnings": [],
        "planner_rows": [
            {
                "planner_key": "planner|unsafe",
                "algo": "goal",
                "kinematics": "holonomic|vx_vy",
                "status": "ok",
                "started_at_utc": "now",
                "runtime_sec": 1.0,
                "episodes": 1,
                "episodes_per_second": 1.0,
                "success_mean": "1.0",
                "collisions_mean": "0.0",
                "snqi_mean": "0.5",
                "projection_rate": "0.0",
                "infeasible_rate": "0.0",
                "execution_mode": "native",
                "execution_detail": "direct_holonomic_world_velocity",
                "planner_command_space": "holonomic_vxy_world",
                "benchmark_command_space": "holonomic_vxy_world",
                "projection_policy": "world_velocity_passthrough",
                "readiness_status": "ok",
                "readiness_tier": "baseline-ready",
                "preflight_status": "ok",
                "learned_policy_contract_status": "not_applicable",
                "socnav_prereq_policy": "fail-fast",
            }
        ],
    }
    write_campaign_report(report_path, payload)
    report_text = report_path.read_text(encoding="utf-8")
    assert "planner\\|unsafe" in report_text
    assert "holonomic\\|vx_vy" in report_text
    assert "direct_holonomic_world_velocity" in report_text
    assert "world_velocity_passthrough" in report_text
    assert "## Accepted Unavailable/Excluded Planners" in report_text
    assert "## Unexpected Failed/Partial Planners" in report_text
    assert "No accepted unavailable/excluded planners." in report_text
    assert "No unexpected failed/partial planners." in report_text


@pytest.mark.parametrize("unexpected_status", ["failed", "partial-failure"])
def test_write_campaign_report_routes_non_success_rows_to_expected_sections(
    tmp_path: Path, unexpected_status: str
) -> None:
    """Report sections should separate accepted-unavailable rows from unexpected failures."""
    report_path = tmp_path / "campaign_report.md"
    payload = {
        "campaign": {"campaign_id": "c1"},
        "warnings": [],
        "planner_rows": [
            {
                "planner_key": "accepted_row",
                "algo": "goal",
                "kinematics": "differential_drive",
                "status": "not_available",
                "availability_reason": "missing optional dependency",
            },
            {
                "planner_key": "unexpected_row",
                "algo": "goal",
                "kinematics": "differential_drive",
                "status": unexpected_status,
                "most_likely_failure_reason": "worker crash",
            },
        ],
    }

    write_campaign_report(report_path, payload)
    report_text = report_path.read_text(encoding="utf-8")

    accepted_section = report_text.split("## Accepted Unavailable/Excluded Planners", 1)[1].split(
        "## Unexpected Failed/Partial Planners", 1
    )[0]
    unexpected_section = report_text.split("## Unexpected Failed/Partial Planners", 1)[1].split(
        "## Campaign Warnings", 1
    )[0]

    assert "| accepted_row | not_available | missing optional dependency |" in accepted_section
    assert "unexpected_row" not in accepted_section
    assert f"| unexpected_row | {unexpected_status} | worker crash |" in unexpected_section
    assert "accepted_row" not in unexpected_section


def test_planner_report_row_uses_nested_planner_kinematics_execution_mode() -> None:
    """Row builder should read execution_mode from nested planner_kinematics payload."""
    planner = PlannerSpec(key="prediction_planner", algo="prediction_planner")
    summary = {
        "status": "ok",
        "written": 1,
        "runtime_sec": 1.0,
        "episodes_per_second": 1.0,
        "algorithm_readiness": {"tier": "baseline-ready"},
        "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
        "algorithm_metadata_contract": {
            "planner_kinematics": {"execution_mode": "adapter"},
            "kinematics_feasibility": {
                "commands_evaluated": 4,
                "projection_rate": 0.25,
                "infeasible_rate": 0.25,
            },
        },
    }
    row = _planner_report_row(
        planner,
        summary,
        aggregates=None,
        kinematics="differential_drive",
    )
    assert row["execution_mode"] == "adapter"
    assert row["readiness_status"] == "adapter"


def test_planner_report_row_exposes_execution_detail_and_command_spaces() -> None:
    """Row builder should surface direct-world-velocity metadata explicitly."""
    planner = PlannerSpec(key="orca", algo="orca")
    summary = {
        "status": "ok",
        "written": 1,
        "runtime_sec": 1.0,
        "episodes_per_second": 1.0,
        "algorithm_readiness": {"tier": "baseline-ready"},
        "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
        "algorithm_metadata_contract": {
            "planner_kinematics": {
                "execution_mode": "adapter",
                "execution_detail": "direct_holonomic_world_velocity",
                "planner_command_space": "holonomic_vxy_world",
                "benchmark_command_space": "holonomic_vxy_world",
                "projection_policy": "world_velocity_passthrough",
            },
        },
    }
    row = _planner_report_row(
        planner,
        summary,
        aggregates=None,
        kinematics="holonomic",
    )
    assert row["execution_mode"] == "adapter"
    assert row["execution_detail"] == "direct_holonomic_world_velocity"
    assert row["planner_command_space"] == "holonomic_vxy_world"
    assert row["benchmark_command_space"] == "holonomic_vxy_world"
    assert row["projection_policy"] == "world_velocity_passthrough"


def test_build_actuation_envelope_summary_carries_amv_and_projection_metadata() -> None:
    """Compact actuation summaries should carry scenario AMV and planner projection metadata."""
    payload = _build_actuation_envelope_summary(
        campaign_id="issue1572",
        generated_at_utc="2026-05-27T09:00:00Z",
        profile=SyntheticActuationProfile(
            name="amv-actuation-stress-v0",
            profile_version="v0",
            claim_scope="synthetic-only",
            claim_boundary="diagnostic-only",
            max_linear_accel_m_s2=2.0,
            max_linear_decel_m_s2=2.5,
            max_yaw_rate_rad_s=1.2,
            max_angular_accel_rad_s2=4.0,
            latency_mode="one-step-delay",
            update_mode="5hz-hold",
        ),
        planner_rows=[
            {
                "planner_key": "social_force",
                "algo": "social_force",
                "planner_group": "core",
                "kinematics": "differential_drive",
                "status": "ok",
                "readiness_status": "adapter",
                "availability_status": "available",
                "benchmark_success": "true",
                "execution_mode": "adapter",
                "execution_detail": "adapter_projected_unicycle_vw",
                "planner_command_space": "unicycle_vw",
                "benchmark_command_space": "unicycle_vw",
                "projection_policy": "heading_safe_velocity_to_unicycle_vw",
                "success_mean": "1.0000",
                "command_clip_fraction_mean": "0.2500",
                "yaw_rate_saturation_fraction_mean": "0.5000",
                "signed_braking_peak_m_s2_mean": "-1.5000",
            }
        ],
        amv_summary={
            "status": "pass",
            "scenario_rows": [
                {
                    "name": "classic_overtaking_medium",
                    "scenario_family": "classic_crossing",
                    "amv": {
                        "use_case": "delivery_robot",
                        "context": "sidewalk",
                        "speed_regime": "walking_speed",
                        "maneuver_type": "overtake",
                    },
                }
            ],
        },
    )

    assert payload["amv_coverage_status"] == "pass"
    assert payload["scenario_amv_rows"][0]["amv"]["maneuver_type"] == "overtake"
    assert payload["rows"][0]["planner_command_space"] == "unicycle_vw"
    assert payload["rows"][0]["benchmark_command_space"] == "unicycle_vw"
    assert payload["rows"][0]["projection_policy"] == "heading_safe_velocity_to_unicycle_vw"
    assert payload["rows"][0]["projection_metadata_status"] == "explicit"


def test_planner_report_row_backfills_collision_means_from_termination_reason() -> None:
    """Row builder should not zero out collisions when aggregate metrics are sparse."""
    planner = PlannerSpec(key="ppo", algo="ppo")
    summary = {
        "status": "ok",
        "written": 2,
        "runtime_sec": 1.0,
        "episodes_per_second": 2.0,
        "algorithm_readiness": {"tier": "experimental"},
        "preflight": {"status": "ok", "learned_policy_contract": {"status": "pass"}},
        "algorithm_metadata_contract": {},
    }
    aggregates = {
        "ppo": {
            "success": {"mean": 0.0},
            "collisions": {"mean": 0.0},
            "total_collision_count": {"mean": 0.0},
            "snqi": {"mean": -0.5},
        }
    }
    records = [
        {"termination_reason": "collision", "metrics": {"snqi": -0.4}},
        {"termination_reason": "success", "metrics": {"snqi": -0.6}},
    ]

    row = _planner_report_row(
        planner,
        summary,
        aggregates=aggregates,
        kinematics="differential_drive",
        records=records,
    )

    assert row["success_mean"] == "0.5000"
    assert row["collisions_mean"] == "0.5000"
    assert row["total_collision_count_mean"] == "0.5000"
    assert row["success_ci_low"] == "nan"
    assert row["success_ci_high"] == "nan"
    assert row["collision_ci_low"] == "nan"
    assert row["collision_ci_high"] == "nan"


def test_planner_report_row_counts_existing_records_after_resume() -> None:
    """Resumed campaign rows should report total records, not only newly written rows."""
    planner = PlannerSpec(key="goal", algo="goal")
    summary = {
        "status": "ok",
        "written": 0,
        "episodes_total": 2,
        "runtime_sec": 1.0,
        "episodes_per_second": 0.0,
        "algorithm_readiness": {"tier": "baseline-ready"},
        "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
        "algorithm_metadata_contract": {},
    }
    records = [
        {"termination_reason": "success", "metrics": {"success": 1.0, "snqi": -0.1}},
        {"termination_reason": "timeout", "metrics": {"success": 0.0, "snqi": -0.2}},
    ]

    row = _planner_report_row(
        planner,
        summary,
        aggregates=None,
        kinematics="differential_drive",
        records=records,
    )

    assert row["episodes"] == 2


def test_planner_report_row_retains_clearance_and_proxemic_release_gate_fields() -> None:
    """Row builder retains min_clearance_m and proxemic_intrusion_rate (issue #4326).

    These dedicated release-gate fields aggregate from per-episode values that
    already exist in episode rows: min_clearance_m is the campaign-wide worst
    case (minimum) clearance, distinct from the mean-of-per-episode-minimums
    kept as min_clearance_mean, and proxemic_intrusion_rate is the mean
    per-episode personal-space intrusion fraction.
    """
    planner = PlannerSpec(key="ppo", algo="ppo")
    summary = {
        "status": "ok",
        "written": 2,
        "runtime_sec": 1.0,
        "episodes_per_second": 2.0,
        "algorithm_readiness": {"tier": "experimental"},
        "preflight": {"status": "ok", "learned_policy_contract": {"status": "pass"}},
        "algorithm_metadata_contract": {},
    }
    records = [
        {
            "termination_reason": "success",
            "metrics": {"min_clearance": 0.8, "social_proxemic_intrusion_frac": 0.10},
        },
        {
            "termination_reason": "success",
            "metrics": {"min_clearance": 0.3, "social_proxemic_intrusion_frac": 0.20},
        },
    ]

    row = _planner_report_row(
        planner,
        summary,
        aggregates=None,
        kinematics="differential_drive",
        records=records,
    )

    # Worst-case minimum clearance across the campaign, not a mean.
    assert row["min_clearance_m"] == "0.3000"
    # Mean-of-per-episode-minimums stays distinct and higher than the worst case.
    assert row["min_clearance_mean"] == "0.5500"
    # Mean per-episode intrusion fraction.
    assert row["proxemic_intrusion_rate"] == "0.1500"


def test_planner_report_row_release_gate_fields_fail_closed_without_source_values() -> None:
    """Campaigns that never recorded the fields stay not_evaluable (issue #4326).

    Past/degraded campaigns are not backfilled: with no per-episode clearance or
    proxemic values, both dedicated release-gate fields fail closed to ``nan``
    so downstream gates report ``not_evaluable`` rather than a fabricated value.
    """
    planner = PlannerSpec(key="goal", algo="goal")
    summary = {
        "status": "ok",
        "written": 1,
        "runtime_sec": 1.0,
        "episodes_per_second": 1.0,
        "algorithm_readiness": {"tier": "baseline-ready"},
        "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
        "algorithm_metadata_contract": {},
    }
    # Episode rows lacking clearance/proxemic metrics (older schema); no backfill.
    records = [{"termination_reason": "success", "metrics": {"success": 1.0}}]

    row = _planner_report_row(
        planner,
        summary,
        aggregates=None,
        kinematics="differential_drive",
        records=records,
    )

    assert row["min_clearance_m"] == "nan"
    assert row["proxemic_intrusion_rate"] == "nan"

    # Also fail closed on the zero-episode (degraded/failed) path.
    empty_row = _planner_report_row(
        planner,
        summary,
        aggregates=None,
        kinematics="differential_drive",
        records=[],
    )
    assert empty_row["min_clearance_m"] == "nan"
    assert empty_row["proxemic_intrusion_rate"] == "nan"


def test_planner_report_row_uses_episode_ci_placeholders_when_means_are_backfilled() -> None:
    """Backfilled success/collision means should not keep stale aggregate CIs."""
    planner = PlannerSpec(key="ppo", algo="ppo")
    summary = {
        "status": "ok",
        "written": 2,
        "runtime_sec": 1.0,
        "episodes_per_second": 2.0,
        "algorithm_readiness": {"tier": "experimental"},
        "preflight": {"status": "ok", "learned_policy_contract": {"status": "pass"}},
        "algorithm_metadata_contract": {},
    }
    aggregates = {
        "ppo": {
            "success": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
            "collisions": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
            "total_collision_count": {"mean": 0.0},
            "snqi": {"mean": -0.5, "mean_ci": [-0.6, -0.4]},
        }
    }
    records = [
        {"termination_reason": "success", "metrics": {"snqi": -0.4}},
        {"termination_reason": "collision", "metrics": {"snqi": -0.6}},
    ]

    row = _planner_report_row(
        planner,
        summary,
        aggregates=aggregates,
        kinematics="differential_drive",
        records=records,
    )

    assert row["success_mean"] == "0.5000"
    assert row["collisions_mean"] == "0.5000"
    assert row["success_ci_low"] == "nan"
    assert row["success_ci_high"] == "nan"
    assert row["collision_ci_low"] == "nan"
    assert row["collision_ci_high"] == "nan"


def test_planner_report_row_recomputes_all_means_for_mixed_algo_candidate() -> None:
    """Mixed scenario-algo candidates should report all metrics over every episode."""
    planner = PlannerSpec(key="scenario_adaptive_hybrid_orca_v1", algo="hybrid_rule_local_planner")
    summary = {
        "status": "ok",
        "written": 4,
        "runtime_sec": 1.0,
        "episodes_per_second": 4.0,
        "algorithm_readiness": {"tier": "experimental"},
        "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
        "algorithm_metadata_contract": {},
    }
    aggregates = {
        "hybrid_rule_local_planner": {
            "success": {"mean": 1.0},
            "collisions": {"mean": 0.0},
            "near_misses": {"mean": 3.0},
            "time_to_goal_norm": {"mean": 0.5},
            "comfort_exposure": {"mean": 0.01},
            "snqi": {"mean": 0.2},
        },
        "orca": {
            "success": {"mean": 0.0},
            "collisions": {"mean": 1.0},
            "near_misses": {"mean": 30.0},
            "time_to_goal_norm": {"mean": 0.9},
            "comfort_exposure": {"mean": 0.09},
            "snqi": {"mean": -0.6},
        },
    }
    records = [
        {
            "termination_reason": "success",
            "metrics": {
                "success": 1.0,
                "near_misses": 3.0,
                "time_to_goal_norm": 0.5,
                "comfort_exposure": 0.01,
                "snqi": 0.2,
            },
        },
        {
            "termination_reason": "success",
            "metrics": {
                "success": 1.0,
                "near_misses": 3.0,
                "time_to_goal_norm": 0.5,
                "comfort_exposure": 0.01,
                "snqi": 0.2,
            },
        },
        {
            "termination_reason": "success",
            "metrics": {
                "success": 1.0,
                "near_misses": 3.0,
                "time_to_goal_norm": 0.5,
                "comfort_exposure": 0.01,
                "snqi": 0.2,
            },
        },
        {
            "termination_reason": "collision",
            "metrics": {
                "success": 0.0,
                "near_misses": 30.0,
                "time_to_goal_norm": 0.9,
                "comfort_exposure": 0.09,
                "snqi": -0.6,
            },
        },
    ]

    row = _planner_report_row(
        planner,
        summary,
        aggregates=aggregates,
        kinematics="differential_drive",
        records=records,
    )

    assert row["success_mean"] == "0.7500"
    assert row["collisions_mean"] == "0.2500"
    assert row["near_misses_mean"] == "9.7500"
    assert row["time_to_goal_norm_mean"] == "0.6000"
    assert row["comfort_exposure_mean"] == "0.0300"
    assert row["snqi_mean"] == "0.0000"


def test_jsonable_repo_relative_normalizes_paths_for_stable_hashing(tmp_path: Path) -> None:
    """Hash-prep helper should normalize Path values to stable repo-relative strings."""
    repo_root = get_repository_root().resolve()
    payload = {
        "scenario_matrix_path": repo_root / "configs/scenarios/classic_interactions.yaml",
        "other_path": tmp_path / "external.yaml",
    }
    normalized = _jsonable_repo_relative(payload)
    assert normalized["scenario_matrix_path"] == "configs/scenarios/classic_interactions.yaml"
    assert str(normalized["other_path"]).endswith("/external.yaml")


def test_sanitize_git_remote_strips_credentials() -> None:
    """Git remote helper should remove embedded credentials from URL-form remotes."""
    remote = "https://user:token@example.com/org/repo.git"
    assert _sanitize_git_remote(remote) == "https://example.com/org/repo.git"
    assert _sanitize_git_remote("git@github.com:ll7/robot_sf_ll7.git") == (
        "git@github.com:ll7/robot_sf_ll7.git"
    )


def test_sanitize_csv_cell_prefixes_formula_like_values() -> None:
    """CSV sanitizer should neutralize spreadsheet formula prefixes."""
    assert _sanitize_csv_cell("=1+1") == "'=1+1"
    assert _sanitize_csv_cell("@SUM(A1:A2)") == "'@SUM(A1:A2)"
    assert _sanitize_csv_cell("\t=cmd|' /C calc'!A0") == "'\t=cmd|' /C calc'!A0"
    assert _sanitize_csv_cell(" \r\n+SUM(A1:A2)") == "' \r\n+SUM(A1:A2)"
    assert _sanitize_csv_cell("safe") == "safe"
    assert _sanitize_csv_cell(42) == 42


def test_stable_json_bytes_use_canonical_hash_encoding() -> None:
    """Hash payload helpers should share the same deterministic JSON bytes."""
    assert _stable_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_kinematics_matrix_or_default_documents_default_behavior() -> None:
    """Leaf helper should normalize labels and only supply the matrix default when empty."""
    assert _kinematics_matrix_or_default((" Holonomic ", "BICYCLE_DRIVE")) == (
        "holonomic",
        "bicycle_drive",
    )
    assert _kinematics_matrix_or_default(()) == ("differential_drive",)


def test_prepare_campaign_preflight_validates_campaign_config(tmp_path: Path) -> None:
    """Programmatic preflight entrypoint should enforce campaign invariants."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    cfg = CampaignConfig(
        name="invalid_preflight_cfg",
        scenario_matrix_path=scenario_abs,
        planners=(PlannerSpec(key="goal", algo="goal", planner_group_explicit=True),),
        seed_policy=SeedPolicy(mode="fixed-list", seeds=(111,)),
        paper_facing=True,
        paper_profile_version=None,
    )
    with pytest.raises(ValueError, match="paper_profile_version"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="invalid")


def test_prepare_campaign_preflight_checks_orca_rvo2_before_loading_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct preflight calls raise a typed error before scenario loading when ORCA lacks rvo2."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    cfg = CampaignConfig(
        name="orca_preflight_guard",
        scenario_matrix_path=scenario_path,
        planners=(PlannerSpec(key="orca", algo="orca"),),
        seed_policy=SeedPolicy(),
    )

    monkeypatch.setitem(sys.modules, "rvo2", None)

    def fail_if_scenarios_load(*_args, **_kwargs):
        raise AssertionError("_load_campaign_scenarios should not run before ORCA preflight")

    monkeypatch.setattr(
        camera_ready_campaign_module,
        "_load_campaign_scenarios",
        fail_if_scenarios_load,
    )

    with pytest.raises(OrcaRvo2PreflightError, match="rvo2"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="orca")


def test_run_campaign_checks_orca_rvo2_before_loading_optional_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct campaign runs raise a typed error before optional JSON loading when ORCA lacks rvo2."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    cfg = CampaignConfig(
        name="orca_run_guard",
        scenario_matrix_path=scenario_path,
        planners=(PlannerSpec(key="orca", algo="orca"),),
        seed_policy=SeedPolicy(),
    )

    monkeypatch.setitem(sys.modules, "rvo2", None)

    def fail_if_optional_json_loads(*_args, **_kwargs):
        raise AssertionError("load_optional_json should not run before ORCA preflight")

    monkeypatch.setattr(
        camera_ready_campaign_module,
        "load_optional_json",
        fail_if_optional_json_loads,
    )

    with pytest.raises(OrcaRvo2PreflightError, match="rvo2"):
        run_campaign(cfg, output_root=tmp_path / "out", label="orca")


def test_run_campaign_sanitizes_run_directory_keys(tmp_path: Path, monkeypatch) -> None:
    """Planner run directories should use sanitized planner/kinematics identifiers."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_sanitize.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: sanitize_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                'kinematics_matrix: ["holonomic/../unsafe"]',
                "planners:",
                '  - key: "../../planner|unsafe"',
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(
        scenarios_or_path,
        out_path,
        schema_path,
        *,
        algo,
        benchmark_profile,
        **kwargs,
    ):
        """Write an episode artifact under a sanitized planner output path."""
        del scenarios_or_path, schema_path, benchmark_profile, kwargs
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(
                {
                    "episode_id": f"e-{algo}-0",
                    "scenario_id": "mock",
                    "seed": 111,
                    "scenario_params": {"algo": algo, "metadata": {"archetype": "crossing"}},
                    "metrics": {"success": 1.0, "collisions": 0.0, "near_misses": 0.0},
                    "algorithm_metadata": {"algorithm": algo, "status": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "out_path": str(out_file),
            "algorithm_readiness": {
                "name": algo,
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
            "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="sanitize")
    campaign_root = Path(result["campaign_root"])
    runs_dir = campaign_root / "runs"
    run_dirs = [path.name for path in runs_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    expected = f"{_sanitize_name('../../planner|unsafe')}__{_sanitize_name('holonomic/../unsafe')}"
    assert run_dirs[0] == expected


def test_run_campaign_marks_skipped_preflight_as_not_available(tmp_path: Path, monkeypatch) -> None:
    """Skipped planner/kinematics combinations must fail closed in benchmark reports."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_skipped.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: skipped_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(*args, **kwargs):
        """Return a skipped preflight result without writing episodes."""
        del args, kwargs
        return {
            "total_jobs": 0,
            "written": 0,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {"status": "skipped", "compatibility_reason": "unsupported"},
            "algorithm_readiness": {
                "name": "goal",
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="skipped")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    assert summary_payload["planner_rows"][0]["status"] == "not_available"
    assert summary_payload["planner_rows"][0]["availability_status"] == "not_available"
    assert summary_payload["planner_rows"][0]["benchmark_success"] == "false"
    assert summary_payload["planner_rows"][0]["most_likely_failure_reason"] == "unsupported"
    assert summary_payload["campaign"]["successful_runs"] == 0
    assert summary_payload["campaign"]["benchmark_success"] is False
    assert result["benchmark_success"] is False


def test_run_campaign_success_ignores_not_available_experimental_when_core_ok(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Optional experimental planners should not make a core-successful campaign fail."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_core_with_optional_experimental.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: core_with_optional_experimental",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
                "    benchmark_profile: baseline-safe",
                "  - key: socnav_bench",
                "    algo: socnav_bench",
                "    planner_group: experimental",
                "    benchmark_profile: experimental",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(*args, **kwargs):
        """Return one successful core run and one missing optional dependency."""
        algo = kwargs["algo"]
        out_path = kwargs["out_path"]
        if algo == "goal":
            Path(out_path).write_text(
                json.dumps(
                    {
                        "episode_id": "e-goal-0",
                        "scenario_id": "mock",
                        "seed": 111,
                        "scenario_params": {
                            "algo": "goal",
                            "metadata": {"archetype": "crossing"},
                        },
                        "metrics": {"success": 1.0, "collisions": 0.0, "near_misses": 0.0},
                        "algorithm_metadata": {"algorithm": "goal", "status": "ok"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return {
                "total_jobs": 1,
                "written": 1,
                "failed_jobs": 0,
                "failures": [],
                "algorithm_readiness": {
                    "name": "goal",
                    "tier": "baseline-ready",
                    "profile": "baseline-safe",
                },
            }
        return {
            "total_jobs": 0,
            "written": 0,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {
                "status": "skipped",
                "compatibility_reason": "SocNavBench assets are not installed",
            },
            "algorithm_readiness": {
                "name": "socnav_bench",
                "tier": "experimental",
                "profile": "experimental",
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="core_optional")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    rows = {row["planner_key"]: row for row in summary_payload["planner_rows"]}

    assert rows["goal"]["status"] == "ok"
    assert rows["socnav_bench"]["status"] == "not_available"
    assert rows["socnav_bench"]["planner_group"] == "experimental"
    assert summary_payload["campaign"]["successful_runs"] == 1
    assert summary_payload["campaign"]["total_runs"] == 2
    assert summary_payload["campaign"]["core_successful_runs"] == 1
    assert summary_payload["campaign"]["core_total_runs"] == 1
    assert summary_payload["campaign"]["benchmark_success_basis"] == "core"
    assert summary_payload["campaign"]["benchmark_success"] is False
    assert summary_payload["campaign"]["campaign_execution_status"] == "completed"
    assert summary_payload["campaign"]["evidence_status"] == "partial"
    assert summary_payload["campaign"]["row_status_summary"] == {
        "successful_evidence_rows": 1,
        "accepted_unavailable_rows": 1,
        "unexpected_failed_rows": 0,
        "fallback_or_degraded_rows": 1,
    }
    assert "core_with_optional_experimental" in result["campaign_id"]
    assert result["benchmark_success_basis"] == "core"
    assert result["core_successful_runs"] == 1
    assert result["core_total_runs"] == 1
    assert result["benchmark_success"] is False
    assert result["status"] == "accepted_unavailable_only"
    assert result["campaign_execution_status"] == "completed"
    assert result["evidence_status"] == "partial"


def test_run_campaign_fails_if_core_run_is_unattempted_after_stop_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Stop-on-failure should not mask unfinished core groups as successful."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_core_stop.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: core_stop_unattempted",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "stop_on_failure: true",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
                "  - key: socnav_bench",
                "    algo: socnav_bench",
                "    planner_group: experimental",
                "  - key: social_force",
                "    algo: social_force",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    call_order: list[str] = []

    def _fake_run_batch(*args, **kwargs):
        """Stop execution on experimental planner before all core planners run."""
        del args
        algo = kwargs["algo"]
        call_order.append(algo)
        if algo == "goal":
            return {
                "total_jobs": 1,
                "written": 1,
                "failed_jobs": 0,
                "failures": [],
                "preflight": {
                    "status": "ok",
                    "learned_policy_contract": {"status": "not_applicable"},
                },
                "algorithm_readiness": {
                    "name": "goal",
                    "tier": "baseline-ready",
                    "profile": "baseline-safe",
                },
            }
        if algo == "socnav_bench":
            return {
                "total_jobs": 1,
                "written": 0,
                "failed_jobs": 1,
                "failures": [{"error": "experimental planner failed"}],
                "preflight": {
                    "status": "ok",
                    "learned_policy_contract": {"status": "not_applicable"},
                },
                "algorithm_readiness": {
                    "name": "socnav_bench",
                    "tier": "experimental",
                    "profile": "experimental",
                },
            }
        pytest.fail("run_batch must not execute remaining planners after stop_on_failure")

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="core_stop")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

    assert call_order == ["goal", "socnav_bench"]
    assert summary_payload["campaign"]["total_runs"] == 2
    assert summary_payload["campaign"]["successful_runs"] == 1
    assert summary_payload["campaign"]["core_successful_runs"] == 1
    assert summary_payload["campaign"]["core_total_runs"] == 1
    assert summary_payload["campaign"]["benchmark_success_basis"] == "core"
    assert summary_payload["campaign"]["benchmark_success"] is False
    assert result["core_total_runs"] == 1
    assert result["core_successful_runs"] == 1
    assert result["benchmark_success"] is False
    assert result["benchmark_success_basis"] == "core"


def test_run_campaign_marks_empty_run_set_as_non_success(tmp_path: Path, monkeypatch) -> None:
    """Campaigns with zero run entries must fail closed at campaign level."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_empty.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: empty_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = replace(load_campaign_config(config_path), planners=())

    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.run_batch",
        lambda *args, **kwargs: pytest.fail(
            "run_batch should not be called when no planners exist"
        ),
    )

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="empty")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

    assert summary_payload["planner_rows"] == []
    assert summary_payload["campaign"]["successful_runs"] == 0
    assert summary_payload["campaign"]["total_runs"] == 0
    assert summary_payload["campaign"]["benchmark_success"] is False
    assert result["benchmark_success"] is False


def test_run_campaign_enforces_snqi_contract_error_mode(tmp_path: Path, monkeypatch) -> None:
    """paper_facing + snqi_contract.enforcement=error should raise on failing contract."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_snqi_error.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: snqi_error_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "snqi_contract:",
                "  enabled: true",
                "  enforcement: error",
                "  rank_alignment_warn_threshold: 1.2",
                "  rank_alignment_fail_threshold: 1.1",
                "  outcome_separation_warn_threshold: 1.2",
                "  outcome_separation_fail_threshold: 1.1",
                "  calibration_seed: 123",
                "  calibration_trials: 10",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(*args, **kwargs):
        """Write an episode record whose benchmark metrics fail the gate."""
        del args
        out_path = Path(kwargs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "episode_id": "e-goal-0",
                    "scenario_id": "mock",
                    "seed": 111,
                    "scenario_params": {"algo": "goal", "metadata": {"archetype": "crossing"}},
                    "metrics": {
                        "success": 0.0,
                        "collisions": 1.0,
                        "near_misses": 2.0,
                        "time_to_goal_norm": 1.0,
                        "comfort_exposure": 0.8,
                        "force_exceed_events": 3.0,
                        "jerk_mean": 0.9,
                    },
                    "algorithm_metadata": {"algorithm": "goal", "status": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
            "algorithm_readiness": {
                "name": "goal",
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)

    with pytest.raises(RuntimeError, match="SNQI contract failed with enforcement=error"):
        run_campaign(cfg, output_root=tmp_path / "campaign_out", label="snqi_error")
    campaign_dirs = sorted((tmp_path / "campaign_out").glob("snqi_error_campaign_snqi_error_*"))
    assert campaign_dirs
    latest_campaign_dir = campaign_dirs[-1]
    assert (latest_campaign_dir / "reports" / "snqi_diagnostics.json").exists()
    assert (latest_campaign_dir / "reports" / "snqi_diagnostics.md").exists()
    assert (latest_campaign_dir / "reports" / "snqi_sensitivity.csv").exists()
    summary_payload = json.loads(
        (latest_campaign_dir / "reports" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    assert "publication_bundle" not in summary_payload


def test_run_campaign_surfaces_snqi_contract_warn_mode(tmp_path: Path, monkeypatch) -> None:
    """Warn-mode SNQI contract outcomes should be visible in campaign warnings."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_snqi_warn.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: snqi_warn_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "snqi_contract:",
                "  enabled: true",
                "  enforcement: warn",
                "  rank_alignment_warn_threshold: 1.2",
                "  rank_alignment_fail_threshold: 1.1",
                "  outcome_separation_warn_threshold: 1.2",
                "  outcome_separation_fail_threshold: 1.1",
                "  calibration_seed: 123",
                "  calibration_trials: 10",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(*args, **kwargs):
        """Write an episode record with failed status despite successful metrics."""
        del args
        out_path = Path(kwargs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "episode_id": "e-goal-0",
                    "scenario_id": "mock",
                    "seed": 111,
                    "scenario_params": {"algo": "goal", "metadata": {"archetype": "crossing"}},
                    "metrics": {
                        "success": 0.0,
                        "collisions": 1.0,
                        "near_misses": 2.0,
                        "time_to_goal_norm": 1.0,
                        "comfort_exposure": 0.8,
                        "force_exceed_events": 3.0,
                        "jerk_mean": 0.9,
                    },
                    "algorithm_metadata": {"algorithm": "goal", "status": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
            "algorithm_readiness": {
                "name": "goal",
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="snqi_warn")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    assert any("snqi_contract.enforcement=warn" in item for item in summary_payload["warnings"])


def test_run_campaign_parity_table_includes_ci_columns(tmp_path: Path, monkeypatch) -> None:
    """Parity artifacts should preserve available CI values."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_ci.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: ci_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(*args, **kwargs):
        """Write a successful episode record for CI report rendering."""
        del args
        out_path = Path(kwargs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "episode_id": "e-goal-0",
                    "scenario_id": "mock",
                    "seed": 111,
                    "scenario_params": {"algo": "goal", "metadata": {"archetype": "crossing"}},
                    "metrics": {"success": 1.0, "collisions": 0.0, "near_misses": 0.0},
                    "algorithm_metadata": {"algorithm": "goal", "status": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {"status": "ok"},
            "algorithm_readiness": {
                "name": "goal",
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
        }

    def _fake_compute_aggregates_with_ci(*args, **kwargs):
        """Return aggregate metrics with confidence intervals for Markdown output."""
        del args, kwargs
        return {
            "mock_group": {
                "success": {"mean": 1.0, "mean_ci": [0.8, 1.0]},
                "collisions": {"mean": 0.0, "mean_ci": [0.0, 0.2]},
                "snqi": {"mean": 0.7, "mean_ci": [0.6, 0.8]},
            },
            "_meta": {"warnings": [], "missing_algorithms": []},
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.compute_aggregates_with_ci",
        _fake_compute_aggregates_with_ci,
    )

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="ci")
    parity_csv = (
        Path(result["campaign_root"]) / "reports" / "kinematics_parity_table.csv"
    ).read_text(encoding="utf-8")
    assert "success_ci_low" in parity_csv
    assert "success_ci_high" in parity_csv
    assert "collision_ci_low" in parity_csv
    assert "collision_ci_high" in parity_csv
    assert "snqi_ci_low" in parity_csv
    assert "snqi_ci_high" in parity_csv


def test_load_campaign_config_accepts_planner_group_and_paper_profile(tmp_path: Path) -> None:
    """Paper-facing config should parse planner groups and profile fields."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "paper_campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: paper_cfg",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
                "    human_model_variant: default_social_force",
                "    human_model_source: robot_sf_native",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    assert cfg.paper_facing is True
    assert cfg.paper_profile_version == "paper-matrix-v1"
    assert cfg.planners[0].planner_group == "core"
    assert cfg.planners[0].human_model_variant == "default_social_force"
    assert cfg.planners[0].human_model_source == "robot_sf_native"


def test_load_campaign_config_rejects_invalid_planner_group(tmp_path: Path) -> None:
    """Planner group must be either core or experimental."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "invalid_group.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: invalid_group",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: invalid",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="planner_group"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_missing_paper_version(tmp_path: Path) -> None:
    """Paper-facing config requires explicit profile version."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "missing_version.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: missing_version",
                "paper_facing: true",
                "kinematics_matrix: [differential_drive]",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="paper_profile_version"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_non_differential_paper_kinematics(tmp_path: Path) -> None:
    """Paper-facing profile v1 should lock differential-drive-only matrix."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "bad_kinematics.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: bad_kinematics",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive, bicycle_drive]",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kinematics_matrix"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_implicit_planner_group_for_paper(tmp_path: Path) -> None:
    """Paper-facing configs should require explicit planner_group fields."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "implicit_group.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: implicit_group",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit planner_group"):
        load_campaign_config(config_path)


def test_prepare_campaign_preflight_writes_matrix_summary(tmp_path: Path) -> None:
    """Preflight preparation should emit validate/preview and matrix-summary artifacts."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "paper_campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: paper_cfg",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                "observation_mode: socnav_state",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
                "    human_model_variant: default_social_force",
                "    human_model_source: robot_sf_native",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="preflight")
    assert Path(prepared["validate_config_path"]).exists()
    assert Path(prepared["preview_scenarios_path"]).exists()
    assert Path(prepared["matrix_summary_json_path"]).exists()
    assert Path(prepared["matrix_summary_csv_path"]).exists()
    matrix_payload = json.loads(
        Path(prepared["matrix_summary_json_path"]).read_text(encoding="utf-8")
    )
    assert matrix_payload["rows"]
    first = matrix_payload["rows"][0]
    assert first["planner_group"] == "core"
    assert first["human_model_variant"] == "default_social_force"
    assert first["human_model_source"] == "robot_sf_native"
    assert first["kinematics"] == "differential_drive"
    assert first["observation_mode"] == "socnav_state"


def test_prepare_campaign_preflight_accepts_fixed_campaign_id(tmp_path: Path) -> None:
    """Fixed campaign ids should make interrupted campaign roots resumable."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: fixed_id",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(
        cfg,
        output_root=tmp_path / "out",
        label="ignored",
        campaign_id="Issue 832 S5 Resume",
    )

    assert prepared["campaign_id"] == "issue_832_s5_resume"
    assert Path(prepared["campaign_root"]).name == "issue_832_s5_resume"


def test_prepare_campaign_preflight_emits_route_clearance_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight should include informational route-clearance warnings with exact distances."""
    scenario_path = tmp_path / "scenarios.yaml"
    map_path = (get_repository_root() / "maps/svg_maps/classic_crossing.svg").resolve()
    scenario_path.write_text(
        "\n".join(
            [
                "- name: clearance_warn",
                f"  map_file: {map_path.as_posix()}",
                "  seeds: [111]",
                "  robot_config:",
                "    radius: 0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: clearance_contract",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Force a deterministic low-clearance condition independent of map geometry.
    # Positive-but-sub-threshold margin: center distance 1.0 with robot radius 0.8 -> +0.2 m
    # (below the 0.5 m warn threshold so a warning is still emitted, but above the 0.0 m
    # fail-closed bound so it does not trip RouteClearanceError; see issue #3628).
    fake_map_def = SimpleNamespace(
        robot_routes=[SimpleNamespace(waypoints=[(-1.0, 0.0), (0.0, 0.0)])],
        obstacles=[SimpleNamespace(vertices=[(1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (1.0, 1.0)])],
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.convert_map",
        lambda _path: fake_map_def,
    )

    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="clearance")
    validate_payload = json.loads(
        Path(prepared["validate_config_path"]).read_text(encoding="utf-8")
    )

    warnings = validate_payload.get("route_clearance_warnings")
    assert isinstance(warnings, list)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning["scenario"] == "clearance_warn"
    assert warning["warning_threshold_m"] == 0.5
    assert warning["warning_scope"] == "scenario"
    assert "min_center_distance_m" in warning
    assert "min_clearance_margin_m" in warning
    assert validate_payload["route_clearance_warning_count"] == 1

    manifest_payload = prepared["manifest_payload"]
    assert manifest_payload["route_clearance_warning_count"] == 1
    assert manifest_payload["route_clearance_warnings"] == warnings


def test_prepare_campaign_preflight_warns_when_route_clearance_map_parse_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Map parse failures should be visible without DEBUG logging."""
    broken_map_path = tmp_path / "broken.svg"
    broken_map_path.write_text("<svg><path d='broken'/></svg>\n", encoding="utf-8")
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "- name: parse_failure_case",
                f"  map_file: {broken_map_path.as_posix()}",
                "  seeds: [111]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: parse_visibility_contract",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_convert_map(_path: str) -> object:
        raise ValueError("invalid SVG path data")

    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.convert_map",
        fail_convert_map,
    )
    captured: list[str] = []

    def capture_message(message: object) -> None:
        captured.append(str(message))

    handle = logger.add(capture_message, level="WARNING", format="{level}:{message}")
    try:
        cfg = load_campaign_config(config_path)
        prepared = prepare_campaign_preflight(
            cfg,
            output_root=tmp_path / "out",
            label="parse",
        )
    finally:
        logger.remove(handle)

    validate_payload = json.loads(
        Path(prepared["validate_config_path"]).read_text(encoding="utf-8")
    )
    assert validate_payload["route_clearance_warning_count"] == 0

    log_text = "\n".join(captured)
    assert "WARNING:" in log_text
    assert "parse_failure_case" in log_text
    assert broken_map_path.as_posix() in log_text
    assert "invalid SVG path data" in log_text


def test_prepare_campaign_preflight_attaches_route_clearance_certifications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight should expose accepted route-clearance interpretations per warning row."""
    scenario_path = tmp_path / "scenarios.yaml"
    map_path = (get_repository_root() / "maps/svg_maps/classic_crossing.svg").resolve()
    scenario_path.write_text(
        "\n".join(
            [
                "- name: certified_clearance_warn",
                f"  map_file: {map_path.as_posix()}",
                "  seeds: [111]",
                "  robot_config:",
                "    radius: 0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    certification_path = tmp_path / "route_clearance_certifications.yaml"
    certification_path.write_text(
        "\n".join(
            [
                "schema_version: route-clearance-certifications.v1",
                "certifications:",
                "  certified_clearance_warn:",
                "    status: certified_stress_geometry",
                "    claim_scope: benchmark-ready stress geometry with caveat",
                "    rationale: Intentionally tight corridor fixture.",
                "    reviewed_on: '2026-05-09'",
                "    reviewed_by: ll7",
                "    issue: '1105'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: clearance_certification_contract",
                f"scenario_matrix: {scenario_path.as_posix()}",
                f"route_clearance_certifications: {certification_path.as_posix()}",
                "observation_noise:",
                "  profile: clearance_noise",
                "  pedestrian_false_negative_prob: 0.2",
                "  seed: 17",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Positive-but-sub-threshold margin: center distance 1.0 with robot radius 0.8 -> +0.2 m
    # (below the 0.5 m warn threshold so a warning is still emitted, but above the 0.0 m
    # fail-closed bound so it does not trip RouteClearanceError; see issue #3628).
    fake_map_def = SimpleNamespace(
        robot_routes=[SimpleNamespace(waypoints=[(-1.0, 0.0), (0.0, 0.0)])],
        obstacles=[SimpleNamespace(vertices=[(1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (1.0, 1.0)])],
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.convert_map",
        lambda _path: fake_map_def,
    )

    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="clearance")
    validate_payload = json.loads(
        Path(prepared["validate_config_path"]).read_text(encoding="utf-8")
    )

    warning = validate_payload["route_clearance_warnings"][0]
    assert warning["certification_status"] == "certified_stress_geometry"
    assert warning["certification_claim_scope"] == "benchmark-ready stress geometry with caveat"
    assert warning["certification_issue"] == "1105"
    assert validate_payload["route_clearance_warning_summary"] == {
        "warning_count": 1,
        "certified_warning_count": 1,
        "unresolved_warning_count": 0,
        "status_counts": {"certified_stress_geometry": 1},
        "unresolved_scenarios": [],
    }
    assert (
        validate_payload["route_clearance_certifications_path"]
        == certification_path.resolve().as_posix()
    )
    assert validate_payload["observation_noise"]["profile"] == "clearance_noise"
    assert validate_payload["observation_noise"]["enabled"] is True
    assert isinstance(validate_payload["observation_noise_hash"], str)
    assert len(validate_payload["observation_noise_hash"]) == 12

    manifest_payload = prepared["manifest_payload"]
    assert (
        manifest_payload["route_clearance_warning_summary"]
        == validate_payload["route_clearance_warning_summary"]
    )
    assert manifest_payload["observation_noise"] == validate_payload["observation_noise"]
    assert manifest_payload["observation_noise_hash"] == validate_payload["observation_noise_hash"]


def test_load_route_clearance_certifications_treats_null_optional_fields_as_missing(
    tmp_path: Path,
) -> None:
    """Optional null certification fields should stay unset rather than stringified."""
    certification_path = tmp_path / "route_clearance_certifications.yaml"
    certification_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "route-clearance-certifications.v1",
                "certifications": {
                    "certified_clearance_warn": {
                        "status": "certified_stress_geometry",
                        "claim_scope": "benchmark-ready stress geometry with caveat",
                        "rationale": "Intentionally tight corridor fixture.",
                        "reviewed_on": None,
                        "reviewed_by": None,
                        "issue": None,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    certifications = _load_route_clearance_certifications(certification_path)

    assert certifications["certified_clearance_warn"]["reviewed_on"] is None
    assert certifications["certified_clearance_warn"]["reviewed_by"] is None
    assert certifications["certified_clearance_warn"]["issue"] is None


def test_load_route_clearance_certifications_rejects_null_required_fields(tmp_path: Path) -> None:
    """Required certification strings should fail closed when YAML supplies null."""
    certification_path = tmp_path / "route_clearance_certifications.yaml"
    certification_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "route-clearance-certifications.v1",
                "certifications": {
                    "certified_clearance_warn": {
                        "status": "certified_stress_geometry",
                        "claim_scope": None,
                        "rationale": "Intentionally tight corridor fixture.",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires non-empty claim_scope and rationale"):
        _load_route_clearance_certifications(certification_path)


def test_load_route_clearance_certifications_rejects_case_insensitive_duplicates(
    tmp_path: Path,
) -> None:
    """Scenario keys should be unique even when they differ only by case."""
    certification_path = tmp_path / "route_clearance_certifications.yaml"
    certification_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "route-clearance-certifications.v1",
                "certifications": {
                    "Certified_Clearance_Warn": {
                        "status": "certified_stress_geometry",
                        "claim_scope": "benchmark-ready stress geometry with caveat",
                        "rationale": "Primary fixture.",
                    },
                    "certified_clearance_warn": {
                        "status": "excluded_from_planner_attribution",
                        "claim_scope": "excluded from planner attribution",
                        "rationale": "Duplicate scenario key with different case.",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate scenario name detected"):
        _load_route_clearance_certifications(certification_path)


def test_prepare_campaign_preflight_matrix_summary_is_deterministic(tmp_path: Path) -> None:
    """Matrix summary row ordering should be deterministic by group/key/kinematics."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "paper_order.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: paper_order",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: stream_gap",
                "    algo: stream_gap",
                "    planner_group: experimental",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="order")
    matrix_payload = json.loads(
        Path(prepared["matrix_summary_json_path"]).read_text(encoding="utf-8")
    )
    planner_keys = [row["planner_key"] for row in matrix_payload["rows"]]
    assert planner_keys == ["goal", "stream_gap"]


def test_prepare_campaign_preflight_writes_amv_and_comparability_artifacts(tmp_path: Path) -> None:
    """Preflight should emit AMV coverage and comparability artifacts."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "- name: amv_ok",
                "  map_file: maps/svg_maps/classic_crossing.svg",
                "  seeds: [111]",
                "  metadata:",
                "    archetype: classic_crossing",
                "  amv:",
                "    use_case: delivery_robot",
                "    context: sidewalk",
                "    speed_regime: walking_speed",
                "    maneuver_type: crossing",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: amv_contract",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                "amv_profile:",
                "  coverage_enforcement: warn",
                "  required_dimensions:",
                "    use_case: [delivery_robot]",
                "    context: [sidewalk]",
                "    speed_regime: [walking_speed]",
                "    maneuver_type: [crossing]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="amv")
    assert Path(prepared["amv_coverage_json_path"]).exists()
    assert Path(prepared["amv_coverage_md_path"]).exists()
    assert Path(prepared["comparability_json_path"]).exists()
    assert Path(prepared["comparability_md_path"]).exists()

    manifest = json.loads(
        (Path(prepared["campaign_root"]) / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["amv_coverage_status"] == "pass"
    assert manifest["comparability_mapping_version"] == "alyassi-comparability-v1"
    assert manifest["artifacts"]["amv_coverage_json"].endswith("reports/amv_coverage_summary.json")
    assert manifest["artifacts"]["comparability_json"].endswith("reports/comparability_matrix.json")


def test_prepare_campaign_preflight_enforces_amv_coverage_error_mode(tmp_path: Path) -> None:
    """Missing AMV dimensions should fail preflight when enforcement is error."""
    scenario_path = tmp_path / "scenarios_missing_amv.yaml"
    scenario_path.write_text(
        "- name: amv_missing\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_error.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: amv_error_contract",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "amv_profile:",
                "  coverage_enforcement: error",
                "  required_dimensions:",
                "    use_case: [delivery_robot]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    with pytest.raises(ValueError, match="AMV coverage contract validation failed"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="amv_error")


def test_load_campaign_config_rejects_invalid_amv_coverage_enforcement(tmp_path: Path) -> None:
    """AMV profile should reject unsupported enforcement values."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_bad_amv.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: bad_amv_cfg",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "amv_profile:",
                "  coverage_enforcement: maybe",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coverage_enforcement"):
        load_campaign_config(config_path)


def test_prepare_campaign_preflight_rejects_invalid_comparability_mapping_for_paper(
    tmp_path: Path,
) -> None:
    """Paper-facing preflight should fail fast on invalid comparability mapping schema."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    bad_mapping = tmp_path / "bad_mapping.yaml"
    bad_mapping.write_text("mapping_version: x\n", encoding="utf-8")
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: invalid_mapping",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                f"scenario_matrix: {scenario_path.as_posix()}",
                f"comparability_mapping: {bad_mapping.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    with pytest.raises(ValueError, match="scenario_family_mapping"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="bad_map")


def test_prepare_campaign_preflight_rejects_missing_planner_key_mapping_for_paper(
    tmp_path: Path,
) -> None:
    """Paper-facing preflight should fail when planner comparability coverage is incomplete."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    incomplete_mapping = tmp_path / "incomplete_mapping.yaml"
    incomplete_mapping.write_text(
        "\n".join(
            [
                "mapping_version: planner-coverage-test-v1",
                "scenario_family_mapping:",
                "  smoke: corridor",
                "metric_comparability:",
                "  success:",
                "    classification: comparable",
                "    alyassi_metric: success_rate",
                "    rationale: comparable",
                "planner_key_mapping:",
                "  prediction_planner_v2_full: predictive-planner-full",
                "  stream_gap: stream-gap",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: missing_planner_mapping",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                f"scenario_matrix: {scenario_path.as_posix()}",
                f"comparability_mapping: {incomplete_mapping.as_posix()}",
                "planners:",
                "  - key: prediction_planner_v2_full",
                "    algo: prediction_planner",
                "    planner_group: core",
                "  - key: prediction_planner_v2_xl_ego",
                "    algo: prediction_planner",
                "    planner_group: core",
                "  - key: stream_gap",
                "    algo: stream_gap",
                "    planner_group: experimental",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    with pytest.raises(ValueError, match="prediction_planner_v2_xl_ego"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="missing_planner")


class TestBuildScenarioAmvLookup:
    """Tests for _build_scenario_amv_lookup and AMV taxonomy in breakdown rows."""

    def test_build_scenario_amv_lookup_extracts_amv_from_scenario(self) -> None:
        """Lookup maps scenario names to their AMV taxonomy dimensions."""
        scenarios = [
            {"name": "corridor_low", "amv": {"use_case": "corridor", "context": "low_density"}},
            {
                "name": "crossing_high",
                "amv": {"use_case": "crossing", "speed_regime": "high_speed"},
            },
            {"name": "no_amv"},
        ]
        lookup = _build_scenario_amv_lookup(scenarios)
        assert lookup["corridor_low"] == {"use_case": "corridor", "context": "low_density"}
        assert lookup["crossing_high"] == {"use_case": "crossing", "speed_regime": "high_speed"}
        assert lookup["no_amv"] == {}

    def test_build_scenario_amv_lookup_from_metadata_amv(self) -> None:
        """Lookup falls back to metadata.amv when top-level amv is absent."""
        scenarios = [
            {
                "name": "meta_amv",
                "metadata": {"amv": {"use_case": "hallway", "maneuver_type": "turn"}},
            }
        ]
        lookup = _build_scenario_amv_lookup(scenarios)
        assert lookup["meta_amv"] == {"use_case": "hallway", "maneuver_type": "turn"}

    def test_extract_amv_taxonomy_prefers_top_level_over_metadata(self) -> None:
        """Top-level amv takes precedence over metadata.amv for overlapping keys."""
        scenario = {
            "name": "both",
            "amv": {"use_case": "top_level"},
            "metadata": {"amv": {"use_case": "metadata_level"}},
        }
        result = _extract_amv_taxonomy(scenario)
        assert result["use_case"] == "top_level"

    def test_extract_amv_taxonomy_ignores_empty_values(self) -> None:
        """Empty or whitespace-only AMV dimension values are excluded."""
        scenario = {
            "name": "sparse",
            "amv": {"use_case": "corridor", "context": "", "speed_regime": "   "},
        }
        result = _extract_amv_taxonomy(scenario)
        assert result == {"use_case": "corridor"}

    def test_extract_amv_taxonomy_returns_empty_when_no_amv(self) -> None:
        """Scenarios without any AMV block produce an empty taxonomy."""
        result = _extract_amv_taxonomy({"name": "no_amv_at_all"})
        assert result == {}

    def test_build_breakdown_rows_without_amv_lookup_preserves_existing_columns(self) -> None:
        """Empty input produces empty scenario and family rows."""
        scenario_rows, family_rows = _build_breakdown_rows([])
        assert scenario_rows == []
        assert family_rows == []

    def test_build_breakdown_rows_includes_amv_columns(self, tmp_path: Path) -> None:
        """AMV taxonomy columns appear in scenario and family breakdown rows."""
        episodes_path = tmp_path / "episodes.jsonl"
        episodes_path.write_text(
            json.dumps({"scenario_id": "corridor_low", "ped_collision_count": 0}) + "\n",
            encoding="utf-8",
        )
        run_entries = [
            {
                "planner": {"key": "orca", "algo": "orca"},
                "status": "ok",
                "episodes_path": str(episodes_path),
            }
        ]
        scenario_amv_lookup = {
            "corridor_low": {"use_case": "corridor", "context": "low_density"},
        }
        scenario_rows, family_rows = _build_breakdown_rows(
            run_entries,
            scenario_amv_lookup=scenario_amv_lookup,
        )
        assert len(scenario_rows) == 1
        row = scenario_rows[0]
        assert row["scenario_id"] == "corridor_low"
        assert row["use_case"] == "corridor"
        assert row["context"] == "low_density"
        assert row["speed_regime"] == ""
        assert row["maneuver_type"] == ""
        assert len(family_rows) == 1
        fam = family_rows[0]
        assert fam["use_case"] == "corridor"
        assert fam["context"] == "low_density"
        assert fam["speed_regime"] == ""
        assert fam["maneuver_type"] == ""

    def test_build_breakdown_rows_family_aggregates_multiple_amv_values(
        self, tmp_path: Path
    ) -> None:
        """Family rows aggregate distinct AMV dimension values with semicolons."""
        ep1_path = tmp_path / "ep1.jsonl"
        ep1_path.write_text(
            "\n".join(
                [
                    json.dumps({"scenario_id": "sc_a", "ped_collision_count": 0}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        ep2_path = tmp_path / "ep2.jsonl"
        ep2_path.write_text(
            "\n".join(
                [
                    json.dumps({"scenario_id": "sc_b", "ped_collision_count": 0}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run_entries = [
            {
                "planner": {"key": "orca", "algo": "orca"},
                "status": "ok",
                "episodes_path": str(ep1_path),
            },
            {
                "planner": {"key": "orca", "algo": "orca"},
                "status": "ok",
                "episodes_path": str(ep2_path),
            },
        ]
        scenario_amv_lookup = {
            "sc_a": {"use_case": "corridor", "context": "low_density"},
            "sc_b": {"use_case": "corridor", "context": "high_density"},
        }
        scenario_rows, family_rows = _build_breakdown_rows(
            run_entries,
            scenario_amv_lookup=scenario_amv_lookup,
        )
        assert len(scenario_rows) == 2
        corridor_scenarios_ids = {r["scenario_id"] for r in scenario_rows}
        assert "sc_a" in corridor_scenarios_ids
        assert "sc_b" in corridor_scenarios_ids
        for row in scenario_rows:
            assert row["use_case"] == "corridor"
            assert row["context"] in ("low_density", "high_density")
        assert len(family_rows) == 1
        assert family_rows[0]["use_case"] == "corridor"
        assert family_rows[0]["context"] == "high_density;low_density"

    def test_build_breakdown_rows_without_lookup_produces_empty_amv_columns(
        self, tmp_path: Path
    ) -> None:
        """Without a lookup, AMV columns exist but contain empty strings."""
        episodes_path = tmp_path / "episodes.jsonl"
        episodes_path.write_text(
            json.dumps({"scenario_id": "s1", "ped_collision_count": 0}) + "\n",
            encoding="utf-8",
        )
        run_entries = [
            {
                "planner": {"key": "orca", "algo": "orca"},
                "status": "ok",
                "episodes_path": str(episodes_path),
            }
        ]
        scenario_rows, _family_rows = _build_breakdown_rows(run_entries)
        assert len(scenario_rows) == 1
        row = scenario_rows[0]
        for dimension in ("use_case", "context", "speed_regime", "maneuver_type"):
            assert dimension in row, f"AMV dimension '{dimension}' missing from scenario row"
            assert row[dimension] == "", f"Expected empty string for {dimension} without AMV lookup"

    def test_scenario_breakdown_csv_contains_amv_headers_in_campaign(self, tmp_path: Path) -> None:
        """End-to-end: AMV columns flow from scenario definitions through the campaign config."""
        scenario_path = tmp_path / "scenarios.yaml"
        scenario_content = (
            "scenarios:\n"
            "  - name: corridor_amv\n"
            "    map_file: maps/svg_maps/francis2023_blind_corner.svg\n"
            "    robot_config:\n"
            "      type: differential_drive\n"
            "      radius: 0.3\n"
            "    ped_config:\n"
            "      robot_visible: false\n"
            "    amv:\n"
            "      use_case: corridor\n"
            "      context: low_density\n"
            "    simulation_config:\n"
            "      max_episode_steps: 200\n"
            "      steps_per_action: 4\n"
            "    robot_spawn_id: 0\n"
            "    robot_goal_id: 0\n"
        )
        scenario_path.write_text(scenario_content, encoding="utf-8")
        config_path = tmp_path / "campaign.yaml"
        config_content = (
            "\n".join(
                [
                    "name: amv_breakdown_test",
                    f"scenario_matrix: {scenario_path.as_posix()}",
                    "planners:",
                    "  - key: social_force",
                    "    algo: social_force",
                    "    planner_group: core",
                    "paper_facing: true",
                    "paper_profile_version: paper-seed-variability-v1",
                    f"comparability_mapping: {_get_comparability_path()}",
                ]
            )
            + "\n"
        )
        config_path.write_text(config_content, encoding="utf-8")
        cfg = load_campaign_config(config_path)
        scenarios = _load_campaign_scenarios(cfg)
        lookup = _build_scenario_amv_lookup(scenarios)
        assert "corridor_amv" in lookup
        assert lookup["corridor_amv"]["use_case"] == "corridor"
        assert lookup["corridor_amv"]["context"] == "low_density"


def _get_comparability_path() -> str:
    repo_root = get_repository_root()
    path = repo_root / "configs" / "benchmarks" / "alyassi_comparability_map_v1.yaml"
    return path.as_posix()


def test_run_campaign_fails_fast_on_missing_snqi_normalized_term(
    tmp_path: Path, monkeypatch
) -> None:
    """Missing normalized SNQI metric should fail sensitivity preflight."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "campaign_snqi_missing_metric.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: snqi_missing_metric_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "snqi_contract:",
                "  enabled: true",
                "  enforcement: error",
                "  rank_alignment_warn_threshold: 1.2",
                "  rank_alignment_fail_threshold: 1.1",
                "  outcome_separation_warn_threshold: 1.2",
                "  outcome_separation_fail_threshold: 1.1",
                "  calibration_seed: 123",
                "  calibration_trials: 10",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(*args, **kwargs):
        del args
        out_path = Path(kwargs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "episode_id": "e-goal-0",
                    "scenario_id": "mock",
                    "seed": 111,
                    "scenario_params": {"algo": "goal", "metadata": {"archetype": "crossing"}},
                    "metrics": {
                        "success": 0.0,
                        "collisions": 1.0,
                        "near_misses": 2.0,
                        "time_to_goal_norm": 1.0,
                        "comfort_exposure": 0.8,
                        "force_exceed_events": 3.0,
                    },
                    "algorithm_metadata": {"algorithm": "goal", "status": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
            "algorithm_readiness": {
                "name": "goal",
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)

    with pytest.raises(RuntimeError, match="SNQI sensitivity preflight failed"):
        run_campaign(cfg, output_root=tmp_path / "campaign_out", label="snqi_missing_metric")
