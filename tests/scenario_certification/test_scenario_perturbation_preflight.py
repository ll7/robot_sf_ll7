"""Tests for scenario perturbation manifest preflight contracts."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from robot_sf.scenario_certification.perturbation_preflight import (
    PERTURBATION_MANIFEST_SCHEMA_VERSION,
    materialize_perturbation_pilot_matrix,
    preflight_perturbation_manifest,
    preflight_to_dict,
)
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)


def _write_manifest(path: Path, *, max_route_offset_m: float = 0.5) -> Path:
    """Write a tiny perturbation manifest that targets an existing simple scenario."""
    payload = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test_perturbation_manifest",
        "scenario_config": "configs/scenarios/single/planner_sanity_simple.yaml",
        "seed_controls": {
            "baseline_seeds": [101, 102],
            "replay_seed_policy": "explicit",
        },
        "validity": {
            "require_scenario_certification": True,
            "max_route_offset_m": max_route_offset_m,
            "invalid_variant_evidence_policy": "exclude_from_success_evidence",
        },
        "variants": [
            {
                "variant_id": "planner_sanity_simple_noop",
                "scenario_id": "planner_sanity_simple",
                "family": "noop",
                "seeds": [101, 102],
            },
            {
                "variant_id": "planner_sanity_simple_route_offset",
                "scenario_id": "planner_sanity_simple",
                "family": "robot_route_offset",
                "seeds": [101, 102],
                "parameters": {
                    "dx_m": 0.25,
                    "dy_m": 0.0,
                    "max_magnitude_m": max_route_offset_m,
                },
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_pedestrian_route_manifest(path: Path, *, scenario_id: str) -> Path:
    """Write a perturbation manifest that targets pedestrian routes in a classic scenario."""
    payload = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test_pedestrian_route_perturbation_manifest",
        "scenario_config": "configs/scenarios/classic_interactions_francis2023.yaml",
        "seed_controls": {
            "baseline_seeds": [111],
            "replay_seed_policy": "explicit",
        },
        "validity": {
            "require_scenario_certification": True,
            "max_route_offset_m": 0.5,
            "invalid_variant_evidence_policy": "exclude_from_success_evidence",
        },
        "variants": [
            {
                "variant_id": f"{scenario_id}_noop",
                "scenario_id": scenario_id,
                "family": "noop",
                "seeds": [111],
            },
            {
                "variant_id": f"{scenario_id}_ped_route_offset",
                "scenario_id": scenario_id,
                "family": "pedestrian_route_offset",
                "seeds": [111],
                "parameters": {
                    "dx_m": 0.0,
                    "dy_m": 0.25,
                    "max_magnitude_m": 0.5,
                    "waypoint_selector": "all",
                },
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_start_delay_manifest(
    path: Path,
    *,
    scenario_id: str,
    dt_s: float = 0.5,
    max_abs_dt_s: float = 1.0,
) -> Path:
    """Write a perturbation manifest that offsets single-pedestrian start delay."""
    payload = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test_start_delay_perturbation_manifest",
        "scenario_config": "configs/scenarios/classic_interactions_francis2023.yaml",
        "seed_controls": {
            "baseline_seeds": [112],
            "replay_seed_policy": "explicit",
        },
        "validity": {
            "require_scenario_certification": True,
            "max_route_offset_m": 0.5,
            "max_start_delay_offset_s": 1.0,
            "invalid_variant_evidence_policy": "exclude_from_success_evidence",
        },
        "variants": [
            {
                "variant_id": f"{scenario_id}_noop",
                "scenario_id": scenario_id,
                "family": "noop",
                "seeds": [112],
            },
            {
                "variant_id": f"{scenario_id}_start_delay_offset",
                "scenario_id": scenario_id,
                "family": "single_pedestrian_start_delay_offset",
                "seeds": [112],
                "parameters": {
                    "dt_s": dt_s,
                    "max_abs_dt_s": max_abs_dt_s,
                    "pedestrian_id": "h3",
                },
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_speed_manifest(
    path: Path,
    *,
    scenario_id: str,
    speed_delta_m_s: float = 0.25,
    max_abs_speed_delta_m_s: float = 0.5,
) -> Path:
    """Write a perturbation manifest that offsets single-pedestrian speed."""
    payload = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test_speed_perturbation_manifest",
        "scenario_config": "configs/scenarios/classic_interactions_francis2023.yaml",
        "seed_controls": {
            "baseline_seeds": [112],
            "replay_seed_policy": "explicit",
        },
        "validity": {
            "require_scenario_certification": True,
            "max_route_offset_m": 0.5,
            "max_single_pedestrian_speed_delta_m_s": 0.5,
            "max_single_pedestrian_speed_m_s": 2.0,
            "invalid_variant_evidence_policy": "exclude_from_success_evidence",
        },
        "variants": [
            {
                "variant_id": f"{scenario_id}_noop",
                "scenario_id": scenario_id,
                "family": "noop",
                "seeds": [112],
            },
            {
                "variant_id": f"{scenario_id}_speed_offset",
                "scenario_id": scenario_id,
                "family": "single_pedestrian_speed_offset",
                "seeds": [112],
                "parameters": {
                    "speed_delta_m_s": speed_delta_m_s,
                    "max_abs_speed_delta_m_s": max_abs_speed_delta_m_s,
                    "pedestrian_id": "h3",
                },
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_wait_duration_manifest(
    path: Path,
    *,
    scenario_id: str,
    wait_delta_s: float = 0.5,
    max_abs_wait_delta_s: float = 1.0,
    max_wait_duration_offset_s: float = 1.0,
) -> Path:
    """Write a perturbation manifest that offsets single-pedestrian wait durations."""
    payload = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test_wait_duration_perturbation_manifest",
        "scenario_config": "configs/scenarios/classic_interactions_francis2023.yaml",
        "seed_controls": {
            "baseline_seeds": [240],
            "replay_seed_policy": "explicit",
        },
        "validity": {
            "require_scenario_certification": True,
            "max_route_offset_m": 0.5,
            "max_wait_duration_offset_s": max_wait_duration_offset_s,
            "invalid_variant_evidence_policy": "exclude_from_success_evidence",
        },
        "variants": [
            {
                "variant_id": f"{scenario_id}_noop",
                "scenario_id": scenario_id,
                "family": "noop",
                "seeds": [240],
            },
            {
                "variant_id": f"{scenario_id}_wait_duration_offset",
                "scenario_id": scenario_id,
                "family": "single_pedestrian_wait_duration_offset",
                "seeds": [240],
                "parameters": {
                    "wait_delta_s": wait_delta_s,
                    "max_abs_wait_delta_s": max_abs_wait_delta_s,
                    "pedestrian_id": "h1",
                },
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_pedestrian_density_manifest(
    path: Path,
    *,
    scenario_id: str,
    density_delta: float = 0.02,
    max_abs_density_delta: float = 0.03,
    manifest_delta_cap: float = 0.03,
    max_ped_density: float | None = 0.12,
) -> Path:
    """Write a perturbation manifest that offsets route-pedestrian density."""
    validity = {
        "require_scenario_certification": True,
        "max_route_offset_m": 0.5,
        "max_pedestrian_density_delta": manifest_delta_cap,
        "invalid_variant_evidence_policy": "exclude_from_success_evidence",
    }
    if max_ped_density is not None:
        validity["max_pedestrian_density"] = max_ped_density
    parameters = {
        "density_delta": density_delta,
        "max_abs_density_delta": max_abs_density_delta,
    }
    if max_ped_density is not None:
        parameters["max_ped_density"] = max_ped_density
    payload = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test_pedestrian_density_perturbation_manifest",
        "scenario_config": "configs/scenarios/classic_interactions_francis2023.yaml",
        "seed_controls": {
            "baseline_seeds": [171],
            "replay_seed_policy": "explicit",
        },
        "validity": validity,
        "variants": [
            {
                "variant_id": f"{scenario_id}_noop",
                "scenario_id": scenario_id,
                "family": "noop",
                "seeds": [171],
            },
            {
                "variant_id": f"{scenario_id}_density_offset",
                "scenario_id": scenario_id,
                "family": "pedestrian_density_offset",
                "seeds": [171],
                "parameters": parameters,
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_manifest_schema_accepts_noop_and_bounded_route_offset(tmp_path: Path) -> None:
    """The public schema should document the first supported perturbation surface."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.validate(manifest, schema)


def test_manifest_schema_accepts_bounded_pedestrian_route_offset(tmp_path: Path) -> None:
    """The public schema should expose the pedestrian route-offset family."""
    manifest_path = _write_pedestrian_route_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_high",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.validate(manifest, schema)


def test_manifest_schema_accepts_bounded_single_pedestrian_start_delay_offset(
    tmp_path: Path,
) -> None:
    """The public schema should expose the start-delay timing family."""
    manifest_path = _write_start_delay_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.validate(manifest, schema)


def test_manifest_schema_requires_manifest_timing_cap_for_start_delay_offset(
    tmp_path: Path,
) -> None:
    """Timing manifests should carry both variant-local and policy-level bounds."""
    manifest_path = _write_start_delay_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    del manifest["validity"]["max_start_delay_offset_s"]
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises(jsonschema.ValidationError, match="max_start_delay_offset_s"):
        jsonschema.validate(manifest, schema)


def test_manifest_schema_accepts_bounded_single_pedestrian_speed_offset(
    tmp_path: Path,
) -> None:
    """The public schema should expose the single-pedestrian speed family."""
    manifest_path = _write_speed_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.validate(manifest, schema)


def test_manifest_schema_requires_manifest_speed_cap_for_speed_offset(
    tmp_path: Path,
) -> None:
    """Speed manifests should carry both variant-local and policy-level bounds."""
    manifest_path = _write_speed_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    del manifest["validity"]["max_single_pedestrian_speed_delta_m_s"]
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises(jsonschema.ValidationError, match="max_single_pedestrian_speed_delta_m_s"):
        jsonschema.validate(manifest, schema)


def test_manifest_schema_accepts_bounded_single_pedestrian_wait_duration_offset(
    tmp_path: Path,
) -> None:
    """The public schema should expose the single-pedestrian wait-duration family."""
    manifest_path = _write_wait_duration_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_intersection_wait",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.validate(manifest, schema)


def test_manifest_schema_requires_manifest_wait_duration_cap_for_wait_duration_offset(
    tmp_path: Path,
) -> None:
    """Wait-duration manifests should carry both variant-local and policy-level bounds."""
    manifest_path = _write_wait_duration_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_intersection_wait",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    del manifest["validity"]["max_wait_duration_offset_s"]
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises(jsonschema.ValidationError, match="max_wait_duration_offset_s"):
        jsonschema.validate(manifest, schema)


def test_manifest_schema_accepts_bounded_pedestrian_density_offset(
    tmp_path: Path,
) -> None:
    """The public schema should expose the route-pedestrian density family."""
    manifest_path = _write_pedestrian_density_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_medium",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.validate(manifest, schema)


def test_manifest_schema_requires_manifest_density_cap_for_density_offset(
    tmp_path: Path,
) -> None:
    """Density manifests should carry both variant-local and policy-level bounds."""
    manifest_path = _write_pedestrian_density_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_medium",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    del manifest["validity"]["max_pedestrian_density_delta"]
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises(jsonschema.ValidationError, match="max_pedestrian_density_delta"):
        jsonschema.validate(manifest, schema)


def test_manifest_schema_rejects_cross_family_parameters(tmp_path: Path) -> None:
    """Family-specific parameter schemas should catch typo-shaped irrelevant fields."""
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    route_manifest_path = _write_manifest(tmp_path / "route_perturbations.yaml")
    route_manifest = yaml.safe_load(route_manifest_path.read_text(encoding="utf-8"))
    route_manifest["variants"][1]["parameters"]["dt_s"] = 0.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(route_manifest, schema)

    timing_manifest_path = _write_start_delay_manifest(
        tmp_path / "timing_perturbations.yaml",
        scenario_id="francis2023_join_group",
    )
    timing_manifest = yaml.safe_load(timing_manifest_path.read_text(encoding="utf-8"))
    timing_manifest["variants"][1]["parameters"]["dx_m"] = 0.0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(timing_manifest, schema)

    speed_manifest_path = _write_speed_manifest(
        tmp_path / "speed_perturbations.yaml",
        scenario_id="francis2023_join_group",
    )
    speed_manifest = yaml.safe_load(speed_manifest_path.read_text(encoding="utf-8"))
    speed_manifest["variants"][1]["parameters"]["dt_s"] = 0.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(speed_manifest, schema)

    wait_manifest_path = _write_wait_duration_manifest(
        tmp_path / "wait_perturbations.yaml",
        scenario_id="francis2023_intersection_wait",
    )
    wait_manifest = yaml.safe_load(wait_manifest_path.read_text(encoding="utf-8"))
    wait_manifest["variants"][1]["parameters"]["speed_delta_m_s"] = 0.25
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(wait_manifest, schema)

    density_manifest_path = _write_pedestrian_density_manifest(
        tmp_path / "density_perturbations.yaml",
        scenario_id="classic_group_crossing_medium",
    )
    density_manifest = yaml.safe_load(density_manifest_path.read_text(encoding="utf-8"))
    density_manifest["variants"][1]["parameters"]["dx_m"] = 0.0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(density_manifest, schema)


def test_preflight_rejects_manifest_that_violates_public_schema(tmp_path: Path) -> None:
    """Runtime preflight should enforce the same schema reviewers validate."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["validity"]["require_scenario_certification"] = False
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="validity.require_scenario_certification"):
        preflight_perturbation_manifest(manifest_path)


def test_preflight_certifies_noop_and_bounded_route_offset(tmp_path: Path) -> None:
    """Valid variants should carry scenario certificates and remain eligible evidence candidates."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml")

    report = preflight_perturbation_manifest(manifest_path)

    assert report.schema_version == "scenario_perturbation_preflight.v1"
    assert [result.variant_id for result in report.results] == [
        "planner_sanity_simple_noop",
        "planner_sanity_simple_route_offset",
    ]
    assert {result.validity_status for result in report.results} == {"valid"}
    assert {result.benchmark_evidence_status for result in report.results} == {
        "eligible_success_evidence_candidate"
    }
    assert all(result.certificate is not None for result in report.results)
    assert report.results[1].perturbation_summary["magnitude_m"] == pytest.approx(0.25)


def test_preflight_certifies_bounded_pedestrian_route_offset(tmp_path: Path) -> None:
    """Pedestrian route offsets should certify through the same fail-closed surface."""
    manifest_path = _write_pedestrian_route_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_high",
    )

    report = preflight_perturbation_manifest(manifest_path)

    assert [result.variant_id for result in report.results] == [
        "classic_group_crossing_high_noop",
        "classic_group_crossing_high_ped_route_offset",
    ]
    assert {result.validity_status for result in report.results} == {"valid"}
    assert report.results[1].family == "pedestrian_route_offset"
    assert report.results[1].perturbation_summary["target"]["waypoint_selector"] == "all"


def test_preflight_excludes_pedestrian_route_offset_without_ped_routes(tmp_path: Path) -> None:
    """A pedestrian-route perturbation must fail closed when no pedestrian routes exist."""
    manifest_path = _write_pedestrian_route_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.family == "pedestrian_route_offset"
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: pedestrian_route_offset selected no pedestrian routes"
    ]


def test_preflight_certifies_bounded_single_pedestrian_start_delay_offset(
    tmp_path: Path,
) -> None:
    """Single-pedestrian start-delay offsets should certify when the target exists."""
    manifest_path = _write_start_delay_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )

    report = preflight_perturbation_manifest(manifest_path)

    assert [result.variant_id for result in report.results] == [
        "francis2023_join_group_noop",
        "francis2023_join_group_start_delay_offset",
    ]
    assert {result.validity_status for result in report.results} == {"valid"}
    assert report.results[1].family == "single_pedestrian_start_delay_offset"
    assert report.results[1].perturbation_summary["dt_s"] == pytest.approx(0.5)
    assert report.results[1].perturbation_summary["target"]["pedestrian_id"] == "h3"


def test_preflight_excludes_start_delay_offset_without_single_pedestrians(
    tmp_path: Path,
) -> None:
    """A timing perturbation must fail closed when no single pedestrians exist."""
    manifest_path = _write_start_delay_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_high",
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.family == "single_pedestrian_start_delay_offset"
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: single_pedestrian_start_delay_offset selected no single pedestrians"
    ]


def test_preflight_excludes_unbounded_start_delay_offset_before_certification(
    tmp_path: Path,
) -> None:
    """Out-of-bounds timing perturbations should fail closed without certification."""
    manifest_path = _write_start_delay_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
        dt_s=1.25,
        max_abs_dt_s=1.0,
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "single_pedestrian_start_delay_offset abs_dt_s 1.250000 s exceeds variant max_abs_dt_s 1.000000 s"
    ]


def test_preflight_excludes_negative_start_delay_result(tmp_path: Path) -> None:
    """Negative timing offsets cannot make the selected start delay negative."""
    manifest_path = _write_start_delay_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
        dt_s=-0.5,
        max_abs_dt_s=1.0,
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: single_pedestrian_start_delay_offset would make pedestrian 'h3' start_delay_s negative"
    ]


def test_preflight_certifies_bounded_single_pedestrian_speed_offset(
    tmp_path: Path,
) -> None:
    """Single-pedestrian speed offsets should certify when the target exists."""
    manifest_path = _write_speed_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )

    report = preflight_perturbation_manifest(manifest_path)

    assert [result.variant_id for result in report.results] == [
        "francis2023_join_group_noop",
        "francis2023_join_group_speed_offset",
    ]
    assert {result.validity_status for result in report.results} == {"valid"}
    assert report.results[1].family == "single_pedestrian_speed_offset"
    assert report.results[1].perturbation_summary["speed_delta_m_s"] == pytest.approx(0.25)
    assert report.results[1].perturbation_summary["default_baseline_speed_m_s"] == pytest.approx(
        0.5
    )
    assert report.results[1].perturbation_summary["target"]["pedestrian_id"] == "h3"


def test_preflight_excludes_speed_offset_without_single_pedestrians(tmp_path: Path) -> None:
    """A speed perturbation must fail closed when no single pedestrians exist."""
    manifest_path = _write_speed_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_high",
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.family == "single_pedestrian_speed_offset"
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: single_pedestrian_speed_offset selected no single pedestrians"
    ]


def test_preflight_excludes_unbounded_speed_offset_before_certification(
    tmp_path: Path,
) -> None:
    """Out-of-bounds speed perturbations should fail closed without certification."""
    manifest_path = _write_speed_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
        speed_delta_m_s=0.75,
        max_abs_speed_delta_m_s=0.5,
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "single_pedestrian_speed_offset abs_speed_delta_m_s 0.750000 m/s exceeds "
        "variant max_abs_speed_delta_m_s 0.500000 m/s"
    ]


def test_preflight_excludes_non_positive_speed_result(tmp_path: Path) -> None:
    """Negative speed deltas cannot make the selected speed non-positive."""
    manifest_path = _write_speed_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
        speed_delta_m_s=-0.5,
        max_abs_speed_delta_m_s=0.5,
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: single_pedestrian_speed_offset would make pedestrian 'h3' speed_m_s non-positive"
    ]


def test_preflight_certifies_bounded_single_pedestrian_wait_duration_offset(
    tmp_path: Path,
) -> None:
    """Single-pedestrian wait-duration offsets should certify when target waits exist."""
    manifest_path = _write_wait_duration_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_intersection_wait",
    )

    report = preflight_perturbation_manifest(manifest_path)

    assert [result.variant_id for result in report.results] == [
        "francis2023_intersection_wait_noop",
        "francis2023_intersection_wait_wait_duration_offset",
    ]
    assert {result.validity_status for result in report.results} == {"valid"}
    assert report.results[1].family == "single_pedestrian_wait_duration_offset"
    assert report.results[1].perturbation_summary["wait_delta_s"] == pytest.approx(0.5)
    assert report.results[1].perturbation_summary["target"]["pedestrian_id"] == "h1"


def test_preflight_excludes_wait_duration_offset_without_wait_entries(
    tmp_path: Path,
) -> None:
    """A wait-duration perturbation must fail closed when selected pedestrians do not wait."""
    manifest_path = _write_wait_duration_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["variants"][1]["parameters"]["pedestrian_id"] = "h3"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.family == "single_pedestrian_wait_duration_offset"
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: single_pedestrian_wait_duration_offset selected no wait_at entries"
    ]


def test_preflight_excludes_wait_duration_offset_without_single_pedestrians(
    tmp_path: Path,
) -> None:
    """A wait-duration perturbation must fail closed for route-only pedestrian scenarios."""
    manifest_path = _write_wait_duration_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_high",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["variants"][1]["parameters"]["pedestrian_selector"] = "all"
    del manifest["variants"][1]["parameters"]["pedestrian_id"]
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.family == "single_pedestrian_wait_duration_offset"
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: single_pedestrian_wait_duration_offset selected no single pedestrians"
    ]


def test_preflight_excludes_unbounded_wait_duration_offset_before_certification(
    tmp_path: Path,
) -> None:
    """Out-of-bounds wait-duration perturbations should fail closed without certification."""
    manifest_path = _write_wait_duration_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_intersection_wait",
        wait_delta_s=1.25,
        max_abs_wait_delta_s=1.0,
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "single_pedestrian_wait_duration_offset abs_wait_delta_s 1.250000 s exceeds "
        "variant max_abs_wait_delta_s 1.000000 s"
    ]


def test_preflight_excludes_negative_wait_duration_result(tmp_path: Path) -> None:
    """Negative wait-duration offsets cannot make selected wait rules negative."""
    manifest_path = _write_wait_duration_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_intersection_wait",
        wait_delta_s=-2.5,
        max_abs_wait_delta_s=3.0,
        max_wait_duration_offset_s=3.0,
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: single_pedestrian_wait_duration_offset would make "
        "pedestrian 'h1' wait_at[0].wait_s negative"
    ]


def test_preflight_certifies_bounded_pedestrian_density_offset(
    tmp_path: Path,
) -> None:
    """Pedestrian density offsets should certify when route pedestrians can spawn."""
    manifest_path = _write_pedestrian_density_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_medium",
    )

    report = preflight_perturbation_manifest(manifest_path)

    assert [result.variant_id for result in report.results] == [
        "classic_group_crossing_medium_noop",
        "classic_group_crossing_medium_density_offset",
    ]
    assert {result.validity_status for result in report.results} == {"valid"}
    density_result = report.results[1]
    assert density_result.family == "pedestrian_density_offset"
    assert density_result.perturbation_summary["baseline_ped_density"] == pytest.approx(0.08)
    assert density_result.perturbation_summary["updated_ped_density"] == pytest.approx(0.10)
    assert density_result.perturbation_summary["pedestrian_route_count"] > 0


def test_preflight_excludes_density_offset_without_pedestrian_routes(
    tmp_path: Path,
) -> None:
    """Density offsets must fail closed when the source scenario has no route pedestrians."""
    manifest_path = _write_pedestrian_density_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_intersection_wait",
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.family == "pedestrian_density_offset"
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: pedestrian_density_offset selected no pedestrian routes"
    ]


def test_preflight_excludes_unbounded_density_offset_before_certification(
    tmp_path: Path,
) -> None:
    """Out-of-bounds density perturbations should fail closed without certification."""
    manifest_path = _write_pedestrian_density_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_medium",
        density_delta=0.04,
        max_abs_density_delta=0.03,
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "pedestrian_density_offset abs_density_delta 0.040000 exceeds "
        "variant max_abs_density_delta 0.030000"
    ]


def test_preflight_excludes_negative_density_result(tmp_path: Path) -> None:
    """Negative density offsets cannot make route-pedestrian density negative."""
    manifest_path = _write_pedestrian_density_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_medium",
        density_delta=-0.10,
        max_abs_density_delta=0.12,
        manifest_delta_cap=0.12,
        max_ped_density=None,
    )

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "preflight_error: pedestrian_density_offset would make ped_density negative"
    ]


def test_preflight_excludes_unbounded_route_offset_before_certification(tmp_path: Path) -> None:
    """Out-of-bounds perturbations should fail closed without becoming benchmark evidence."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml", max_route_offset_m=0.2)

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "robot_route_offset magnitude 0.250000 m exceeds variant max_magnitude_m 0.200000 m"
    ]
    payload = preflight_to_dict(report)
    assert payload["results"][1]["certificate"] is None


def test_materialize_pilot_matrix_writes_runnable_variant_scenarios(tmp_path: Path) -> None:
    """Preflight-eligible variants should become local scenario-matrix rows."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml")
    output_dir = tmp_path / "pilot"

    materialized = materialize_perturbation_pilot_matrix(
        manifest_path,
        output_dir=output_dir,
        seed_limit=1,
    )

    assert materialized.included_variants == (
        "planner_sanity_simple_noop",
        "planner_sanity_simple_route_offset",
    )
    assert materialized.excluded_variants == ()
    matrix_path = Path(materialized.scenario_matrix_path)
    generated_scenarios = load_scenarios(matrix_path)
    assert [scenario["name"] for scenario in generated_scenarios] == [
        "planner_sanity_simple_noop",
        "planner_sanity_simple_route_offset",
    ]
    assert {tuple(scenario["seeds"]) for scenario in generated_scenarios} == {(101,)}
    offset_scenario = generated_scenarios[1]
    metadata = offset_scenario["metadata"]["scenario_perturbation"]
    assert metadata["source_scenario_id"] == "planner_sanity_simple"
    assert metadata["evidence_boundary"] == "local_pilot_input_not_benchmark_evidence"

    original = load_scenarios("configs/scenarios/single/planner_sanity_simple.yaml")[0]
    original_config = build_robot_config_from_scenario(
        original,
        scenario_path=Path("configs/scenarios/single/planner_sanity_simple.yaml"),
    )
    offset_config = build_robot_config_from_scenario(
        offset_scenario,
        scenario_path=matrix_path,
    )
    _map_name, original_map = next(iter(original_config.map_pool.map_defs.items()))
    _map_name, offset_map = next(iter(offset_config.map_pool.map_defs.items()))
    original_point = original_map.robot_routes[0].waypoints[0]
    offset_point = offset_map.robot_routes[0].waypoints[0]
    assert offset_point[0] == pytest.approx(original_point[0] + 0.25)
    assert offset_point[1] == pytest.approx(original_point[1])


def test_materialize_pilot_matrix_writes_pedestrian_route_overrides(tmp_path: Path) -> None:
    """Pedestrian-route variants should populate the route-override ped_routes payload."""
    manifest_path = _write_pedestrian_route_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_high",
    )
    output_dir = tmp_path / "pilot"

    materialized = materialize_perturbation_pilot_matrix(
        manifest_path,
        output_dir=output_dir,
        seed_limit=1,
    )

    assert materialized.included_variants == (
        "classic_group_crossing_high_noop",
        "classic_group_crossing_high_ped_route_offset",
    )
    matrix_path = Path(materialized.scenario_matrix_path)
    generated_scenarios = load_scenarios(matrix_path)
    offset_scenario = generated_scenarios[1]
    route_override_path = matrix_path.parent / offset_scenario["route_overrides_file"]
    route_payload = yaml.safe_load(route_override_path.read_text(encoding="utf-8"))
    assert route_payload["route_payload"]["robot_routes"] == []
    assert route_payload["route_payload"]["ped_routes"]

    original = select_scenario(
        load_scenarios("configs/scenarios/classic_interactions_francis2023.yaml"),
        "classic_group_crossing_high",
    )
    original_config = build_robot_config_from_scenario(
        original,
        scenario_path=Path("configs/scenarios/classic_interactions_francis2023.yaml"),
    )
    offset_config = build_robot_config_from_scenario(
        offset_scenario,
        scenario_path=matrix_path,
    )
    _map_name, original_map = next(iter(original_config.map_pool.map_defs.items()))
    _map_name, offset_map = next(iter(offset_config.map_pool.map_defs.items()))
    assert offset_map.robot_routes[0].waypoints[0] == pytest.approx(
        original_map.robot_routes[0].waypoints[0]
    )
    original_point = original_map.ped_routes[0].waypoints[0]
    offset_point = offset_map.ped_routes[0].waypoints[0]
    assert offset_point[0] == pytest.approx(original_point[0])
    assert offset_point[1] == pytest.approx(original_point[1] + 0.25)


def test_materialize_pilot_matrix_writes_start_delay_overrides(tmp_path: Path) -> None:
    """Start-delay variants should preserve single-ped entries while updating delay."""
    manifest_path = _write_start_delay_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )
    output_dir = tmp_path / "pilot"

    materialized = materialize_perturbation_pilot_matrix(
        manifest_path,
        output_dir=output_dir,
        seed_limit=1,
    )

    assert materialized.included_variants == (
        "francis2023_join_group_noop",
        "francis2023_join_group_start_delay_offset",
    )
    matrix_path = Path(materialized.scenario_matrix_path)
    generated_scenarios = load_scenarios(matrix_path)
    offset_scenario = generated_scenarios[1]
    overrides = {entry["id"]: entry for entry in offset_scenario["single_pedestrians"]}
    assert overrides["h3"]["role"] == "join"
    assert overrides["h3"]["role_target_id"] == "h1"
    assert overrides["h3"]["start_delay_s"] == pytest.approx(0.5)

    offset_config = build_robot_config_from_scenario(
        offset_scenario,
        scenario_path=matrix_path,
    )
    _map_name, offset_map = next(iter(offset_config.map_pool.map_defs.items()))
    delays = {ped.id: ped.start_delay_s for ped in offset_map.single_pedestrians}
    assert delays["h3"] == pytest.approx(0.5)


def test_materialize_pilot_matrix_start_delay_selector_all(tmp_path: Path) -> None:
    """The timing family can intentionally phase-shift every single pedestrian."""
    manifest_path = _write_start_delay_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_leave_group",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest["variants"][1]["parameters"]
    del parameters["pedestrian_id"]
    parameters["pedestrian_selector"] = "all"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    materialized = materialize_perturbation_pilot_matrix(
        manifest_path,
        output_dir=tmp_path / "pilot",
        seed_limit=1,
    )

    assert materialized.included_variants == (
        "francis2023_leave_group_noop",
        "francis2023_leave_group_start_delay_offset",
    )
    matrix_path = Path(materialized.scenario_matrix_path)
    generated_scenarios = load_scenarios(matrix_path)
    offset_scenario = generated_scenarios[1]
    offset_config = build_robot_config_from_scenario(
        offset_scenario,
        scenario_path=matrix_path,
    )
    _map_name, offset_map = next(iter(offset_config.map_pool.map_defs.items()))
    delays = {ped.id: ped.start_delay_s for ped in offset_map.single_pedestrians}
    assert delays == {"h1": pytest.approx(0.5), "h2": pytest.approx(0.5), "h3": pytest.approx(0.5)}


def test_materialize_pilot_matrix_writes_speed_overrides(tmp_path: Path) -> None:
    """Speed variants should preserve single-ped entries while updating speed."""
    manifest_path = _write_speed_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_join_group",
    )
    output_dir = tmp_path / "pilot"

    materialized = materialize_perturbation_pilot_matrix(
        manifest_path,
        output_dir=output_dir,
        seed_limit=1,
    )

    assert materialized.included_variants == (
        "francis2023_join_group_noop",
        "francis2023_join_group_speed_offset",
    )
    matrix_path = Path(materialized.scenario_matrix_path)
    generated_scenarios = load_scenarios(matrix_path)
    offset_scenario = generated_scenarios[1]
    overrides = {entry["id"]: entry for entry in offset_scenario["single_pedestrians"]}
    assert overrides["h3"]["role"] == "join"
    assert overrides["h3"]["role_target_id"] == "h1"
    assert overrides["h3"]["speed_m_s"] == pytest.approx(0.75)

    offset_config = build_robot_config_from_scenario(
        offset_scenario,
        scenario_path=matrix_path,
    )
    _map_name, offset_map = next(iter(offset_config.map_pool.map_defs.items()))
    speeds = {ped.id: ped.speed_m_s for ped in offset_map.single_pedestrians}
    assert speeds["h3"] == pytest.approx(0.75)


def test_materialize_pilot_matrix_speed_selector_all(tmp_path: Path) -> None:
    """The speed family can intentionally shift every single pedestrian speed."""
    manifest_path = _write_speed_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_leave_group",
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest["variants"][1]["parameters"]
    del parameters["pedestrian_id"]
    parameters["pedestrian_selector"] = "all"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    materialized = materialize_perturbation_pilot_matrix(
        manifest_path,
        output_dir=tmp_path / "pilot",
        seed_limit=1,
    )

    assert materialized.included_variants == (
        "francis2023_leave_group_noop",
        "francis2023_leave_group_speed_offset",
    )
    matrix_path = Path(materialized.scenario_matrix_path)
    generated_scenarios = load_scenarios(matrix_path)
    offset_scenario = generated_scenarios[1]
    offset_config = build_robot_config_from_scenario(
        offset_scenario,
        scenario_path=matrix_path,
    )
    _map_name, offset_map = next(iter(offset_config.map_pool.map_defs.items()))
    speeds = {ped.id: ped.speed_m_s for ped in offset_map.single_pedestrians}
    assert speeds == {
        "h1": pytest.approx(0.75),
        "h2": pytest.approx(0.75),
        "h3": pytest.approx(0.75),
    }


def test_materialize_pilot_matrix_writes_wait_duration_overrides(tmp_path: Path) -> None:
    """Wait-duration variants should preserve single-ped entries while updating waits."""
    manifest_path = _write_wait_duration_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="francis2023_intersection_wait",
    )

    materialized = materialize_perturbation_pilot_matrix(
        manifest_path,
        output_dir=tmp_path / "pilot",
        seed_limit=1,
    )

    assert materialized.included_variants == (
        "francis2023_intersection_wait_noop",
        "francis2023_intersection_wait_wait_duration_offset",
    )
    matrix_path = Path(materialized.scenario_matrix_path)
    generated_scenarios = load_scenarios(matrix_path)
    offset_scenario = generated_scenarios[1]
    overrides = {entry["id"]: entry for entry in offset_scenario["single_pedestrians"]}
    assert overrides["h1"]["note"] == "intersection wait"
    assert overrides["h1"]["wait_at"] == [
        {
            "waypoint_index": 1,
            "wait_s": pytest.approx(2.5),
            "note": "yield before crossing",
        }
    ]

    offset_config = build_robot_config_from_scenario(
        offset_scenario,
        scenario_path=matrix_path,
    )
    _map_name, offset_map = next(iter(offset_config.map_pool.map_defs.items()))
    waits = {
        ped.id: [rule.wait_s for rule in ped.wait_at or []] for ped in offset_map.single_pedestrians
    }
    assert waits["h1"] == [pytest.approx(2.5)]


def test_materialize_pilot_matrix_writes_pedestrian_density_override(
    tmp_path: Path,
) -> None:
    """Density variants should write an executable scenario-level density override."""
    manifest_path = _write_pedestrian_density_manifest(
        tmp_path / "perturbations.yaml",
        scenario_id="classic_group_crossing_medium",
    )

    materialized = materialize_perturbation_pilot_matrix(
        manifest_path,
        output_dir=tmp_path / "pilot",
        seed_limit=1,
    )

    assert materialized.included_variants == (
        "classic_group_crossing_medium_noop",
        "classic_group_crossing_medium_density_offset",
    )
    matrix_path = Path(materialized.scenario_matrix_path)
    generated_scenarios = load_scenarios(matrix_path)
    offset_scenario = generated_scenarios[1]
    assert offset_scenario["simulation_config"]["ped_density"] == pytest.approx(0.10)
    metadata = offset_scenario["metadata"]["scenario_perturbation"]
    assert metadata["family"] == "pedestrian_density_offset"
    assert metadata["perturbation_summary"]["updated_ped_density"] == pytest.approx(0.10)

    offset_config = build_robot_config_from_scenario(
        offset_scenario,
        scenario_path=matrix_path,
    )
    assert offset_config.sim_config.peds_per_area_m2 == pytest.approx(0.10)


def test_materialize_pilot_matrix_skips_preflight_excluded_variants(tmp_path: Path) -> None:
    """Invalid variants should stay out of executable pilot inputs."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml", max_route_offset_m=0.2)

    materialized = materialize_perturbation_pilot_matrix(
        manifest_path,
        output_dir=tmp_path / "pilot",
    )

    assert materialized.included_variants == ("planner_sanity_simple_noop",)
    assert materialized.excluded_variants == ("planner_sanity_simple_route_offset",)
    generated_scenarios = load_scenarios(Path(materialized.scenario_matrix_path))
    assert [scenario["name"] for scenario in generated_scenarios] == ["planner_sanity_simple_noop"]
    route_files = list((tmp_path / "pilot" / "route_overrides").glob("*"))
    assert route_files == []


def test_materialize_pilot_matrix_clears_stale_generated_routes(tmp_path: Path) -> None:
    """A rerun should not leave old route files for newly excluded variants."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml")
    output_dir = tmp_path / "pilot"

    materialize_perturbation_pilot_matrix(manifest_path, output_dir=output_dir)
    assert len(list((output_dir / "route_overrides").glob("*.route_overrides.yaml"))) == 1

    _write_manifest(manifest_path, max_route_offset_m=0.2)
    materialized = materialize_perturbation_pilot_matrix(manifest_path, output_dir=output_dir)

    assert materialized.included_variants == ("planner_sanity_simple_noop",)
    assert materialized.excluded_variants == ("planner_sanity_simple_route_offset",)
    assert list((output_dir / "route_overrides").glob("*.route_overrides.yaml")) == []
    generated_scenarios = load_scenarios(Path(materialized.scenario_matrix_path))
    assert [scenario["name"] for scenario in generated_scenarios] == ["planner_sanity_simple_noop"]


def test_materialize_pilot_matrix_rejects_tracked_repo_output_location(tmp_path: Path) -> None:
    """Repository-local materialized pilot inputs should stay under ignored output/."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml")

    with pytest.raises(ValueError, match="must be under output/"):
        materialize_perturbation_pilot_matrix(
            manifest_path,
            output_dir=Path("docs") / "scenario_perturbation_pilot_test",
        )
