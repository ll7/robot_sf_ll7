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
