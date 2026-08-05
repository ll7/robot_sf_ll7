"""Tests for the issue #6644 oracle-first narrow-doorway geometry family."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.narrow_doorway_geometry_family import (
    build_variant_matrix,
    generate_variant_map,
    load_geometry_family_manifest,
    run_geometry_family_preflight,
    write_preflight_report,
)
from robot_sf.scenario_certification.v1 import RouteCertificate, ScenarioCertificate
from robot_sf.training.scenario_loader import load_scenarios

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "configs/benchmarks/issue_6644_narrow_doorway_geometry_family_v1.yaml"
_BASE_MAP = _REPO_ROOT / "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_certifier(scenario: dict[str, Any], scenario_path: Path) -> ScenarioCertificate:
    """Return a deterministic certificate keyed to the generated width."""
    metadata = scenario["metadata"]
    gap = float(metadata["gap_width_m"])
    radius = float(scenario["robot_config"]["radius"])
    feasible = gap > 2.0 * radius
    classification = "valid" if feasible else "geometrically_infeasible"
    checks = {
        "minimum_static_clearance_m": gap / 2.0 - radius,
        "shortest_path_length_m": 18.5,
        "inflated_collision_free_path": feasible,
    }
    return ScenarioCertificate(
        schema_version="scenario_cert.v1",
        scenario_id=str(scenario["name"]),
        source=str(scenario_path),
        classification=classification,
        benchmark_eligibility="eligible" if feasible else "excluded",
        reasons=[],
        checks={},
        route_certificates=[
            RouteCertificate(
                route_id="route-0",
                spawn_id=0,
                goal_id=0,
                classification=classification,
                benchmark_eligibility="eligible" if feasible else "excluded",
                reasons=[],
                checks=checks,
            )
        ],
    )


def _fake_episode_runner(
    scenario: dict[str, Any], seed: int, horizon: int | None, algo: str
) -> dict[str, Any]:
    return {"route_complete": True, "steps": 100, "termination_reason": "success"}


def test_manifest_and_matrix_cover_boundary_and_depth_axes() -> None:
    manifest = load_geometry_family_manifest(_MANIFEST)
    variants = build_variant_matrix(manifest)

    assert len(variants) == 15
    assert {item["gap_width_m"] for item in variants} == {0.8, 1.9, 2.0, 2.1, 2.2}
    assert {item["constriction_depth_m"] for item in variants} == {0.25, 1.0, 2.0}
    baseline = next(
        item
        for item in variants
        if item["gap_width_m"] == 2.0 and item["constriction_depth_m"] == 1.0
    )
    assert baseline["derived_clearance_margin_m"] == pytest.approx(0.0)
    assert baseline["expected_geometry_tier"] == "boundary_tangent"
    assert any(
        item["expected_geometry_tier"] == "geometrically_feasible_candidate" for item in variants
    )


def test_variant_map_changes_only_internal_gap_and_depth(tmp_path: Path) -> None:
    before = _sha256(_BASE_MAP)
    output = tmp_path / "variant.svg"
    generate_variant_map(
        _BASE_MAP,
        gap_width_m=2.1,
        constriction_depth_m=0.25,
        output_path=output,
    )
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "generated_variant",
                        "map_file": "variant.svg",
                        "simulation_config": {"max_episode_steps": 400},
                        "robot_config": {},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scenario = dict(load_scenarios(scenario_path)[0])
    from robot_sf.benchmark.narrow_doorway_radius_audit import derive_doorway_geometry

    geometry = derive_doorway_geometry(scenario_path, scenario)
    assert geometry.gap_width_m == pytest.approx(2.1)
    assert geometry.obstacle_rects[0]["width"] == pytest.approx(0.25)
    assert geometry.obstacle_rects[1]["width"] == pytest.approx(0.25)
    assert _sha256(_BASE_MAP) == before


def test_preflight_records_oracle_before_not_run_planner_lane(tmp_path: Path) -> None:
    report = run_geometry_family_preflight(
        _MANIFEST,
        output_dir=tmp_path / "variants",
        episode_runner=_fake_episode_runner,
        certifier=_fake_certifier,
    )

    assert report["go"] is True
    assert report["checks"]["oracle_available_for_every_variant"] is True
    assert report["checks"]["planner_records_are_not_run"] is True
    assert report["checks"]["no_campaign_evidence"] is True
    assert report["checks"]["variant_count"] == 15
    assert all(item["planner"]["status"] == "not_run" for item in report["variants"])
    assert {item["oracle"]["nominal_verdict"]["status"] for item in report["variants"]} == {
        "feasible",
        "infeasible_by_construction",
    }
    assert (
        sum(
            item["geometry"]["expected_geometry_tier"] == "boundary_tangent"
            for item in report["variants"]
        )
        == 3
    )
    assert (
        sum(
            item["geometry"]["expected_geometry_tier"] == "infeasible_by_construction"
            for item in report["variants"]
        )
        == 6
    )
    assert all(Path(item["assets"]["map_path"]).is_file() for item in report["variants"])

    report_path = tmp_path / "issue_6644_preflight.json"
    write_preflight_report(report, report_path)
    payload = report_path.read_text(encoding="utf-8")
    assert '"review_marker": "AI-GENERATED NEEDS-REVIEW"' in payload
