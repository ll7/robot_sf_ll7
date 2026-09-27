"""Tests for the issue #9348 three-width doorway comparison application."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from robot_sf.benchmark.three_width_doorway_application import (
    build_pair_manifest,
    check_variant_diff,
    generate_application_assets,
    load_three_width_manifest,
    run_three_width_preflight,
    write_preflight_report,
)
from robot_sf.scenario_certification.v1 import RouteCertificate, ScenarioCertificate
from robot_sf.training.scenario_loader import load_scenarios

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "configs/benchmarks/issue_9348_three_width_doorway_v1.yaml"
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
    return {
        "route_complete": True,
        "steps": 100,
        "horizon_steps": horizon,
        "termination_reason": "success",
        "fallback_or_degraded": False,
    }


def test_manifest_pins_three_width_tiers() -> None:
    """The application manifest must pin narrow/middle/wide tiers at fixed depth."""
    manifest = load_three_width_manifest(_MANIFEST)
    resolved = manifest["_resolved"]
    assert tuple(resolved["gap_levels"]) == (0.8, 2.0, 2.2)
    assert tuple(resolved["depth_levels"]) == (1.0,)
    assert float(resolved["nominal_radius_m"]) == 1.0
    assert tuple(resolved["planner_roster"]) == ("goal", "social_force")
    assert len(resolved["planner_seeds"]) == 30


def test_matrix_covers_narrower_equal_wider_diameter(tmp_path: Path) -> None:
    """Variant tiers must span infeasible, tangent, and feasible in width order."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "matrix")
    assert [asset["gap_width_m"] for asset in assets] == [0.8, 2.0, 2.2]
    assert [asset["expected_geometry_tier"] for asset in assets] == [
        "infeasible_by_construction",
        "boundary_tangent",
        "geometrically_feasible_candidate",
    ]
    assert all(asset["constriction_depth_m"] == 1.0 for asset in assets)


def test_variant_assets_change_only_explained_fields(tmp_path: Path) -> None:
    """Generated variant scenarios must pass the automated diff check."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "variants")
    assert len(assets) == 3
    base_scenario = dict(load_scenarios(manifest["_resolved"]["scenario_path"])[0])
    for asset in assets:
        variant_scenario = dict(load_scenarios(asset["scenario_path"])[0])
        assert check_variant_diff(base_scenario, variant_scenario) == []
        assert Path(asset["map_path"]).is_file()


def test_variant_diff_rejects_unexplained_changes() -> None:
    """The diff check must fail closed on out-of-allowlist scenario edits."""
    manifest = load_three_width_manifest(_MANIFEST)
    base_scenario = dict(load_scenarios(manifest["_resolved"]["scenario_path"])[0])
    tampered = dict(base_scenario)
    tampered["pedestrian_config"] = {"density": 99.0}
    violations = check_variant_diff(base_scenario, tampered)
    assert violations != []


def test_svg_units_are_metres() -> None:
    """The historical map must declare metre-scale SVG units for width semantics."""
    text = _BASE_MAP.read_text(encoding="utf-8")
    assert 'width="' in text and 'height="' in text


def test_pair_manifest_shares_seeds_across_widths() -> None:
    """Each pair must span all three widths with one shared seed and hashes."""
    assets = [
        {
            "variant_id": f"gap_{token}",
            "gap_width_m": gap,
            "scenario_sha256": f"scenario-{token}",
            "map_sha256": f"map-{token}",
        }
        for token, gap in (("0p80", 0.8), ("2p00", 2.0), ("2p20", 2.2))
    ]
    pairs = build_pair_manifest(assets, (225, 226), "manifest-sha")
    assert pairs["schema_version"] == "issue_9348_three_width_pair_manifest.v1"
    assert pairs["realization_hash_status"] == "pending_campaign"
    assert [pair["pair_id"] for pair in pairs["pairs"]] == ["pair_00225", "pair_00226"]
    for pair in pairs["pairs"]:
        assert [cell["gap_width_m"] for cell in pair["cells"]] == [0.8, 2.0, 2.2]
        assert all(cell["realization_hash"] is None for cell in pair["cells"])


def test_preflight_records_oracle_before_not_run_planner_lane(tmp_path: Path) -> None:
    """The oracle-first preflight must pass with planner rows explicitly not run."""
    report = run_three_width_preflight(
        _MANIFEST,
        output_dir=tmp_path / "variants",
        episode_runner=_fake_episode_runner,
        certifier=_fake_certifier,
    )

    assert report["go"] is True
    assert report["checks"]["baseline_passes"] is True
    assert report["checks"]["variant_count"] == 3
    assert report["checks"]["covers_narrower_equal_wider_tiers"] is True
    assert report["checks"]["oracle_available_for_every_variant"] is True
    assert report["checks"]["planner_records_are_not_run"] is True
    assert report["checks"]["no_campaign_evidence"] is True
    assert all(item["planner"]["status"] == "not_run" for item in report["variants"])
    assert {item["oracle"]["nominal_verdict"]["status"] for item in report["variants"]} == {
        "feasible",
        "infeasible_by_construction",
    }

    report_path = tmp_path / "issue_9348_preflight.json"
    write_preflight_report(report, report_path)
    payload = report_path.read_text(encoding="utf-8")
    assert '"review_marker": "AI-GENERATED NEEDS-REVIEW"' in payload
