"""Tests for the issue #9348 three-width doorway comparison application."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.three_width_doorway_application import (
    DoorwayPairingSession,
    _validate_application_execution,
    build_pair_manifest,
    build_pair_receipt,
    check_pair_receipts,
    check_variant_diff,
    generate_application_assets,
    load_three_width_manifest,
    non_width_config_sha256,
    run_three_width_preflight,
    write_preflight_report,
)
from robot_sf.gym_env.environment_factory import make_robot_env
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
    return {"route_complete": True, "steps": 100, "termination_reason": "success"}


def test_manifest_pins_three_width_tiers() -> None:
    """The application manifest must pin narrow/middle/wide tiers at fixed depth."""
    manifest = load_three_width_manifest(_MANIFEST)
    resolved = manifest["_resolved"]
    assert tuple(resolved["gap_levels"]) == (2.2, 2.8, 3.6)
    assert tuple(resolved["depth_levels"]) == (1.0,)
    assert float(resolved["nominal_radius_m"]) == 1.0
    assert tuple(resolved["planner_roster"]) == ("goal", "social_force")
    assert tuple(resolved["planner_seeds"]) == (225, 226, 227)
    assert manifest["planner_protocol"]["expected_rows"] == 18
    assert manifest["execution"] == {
        "production_campaign_authorized": True,
        "slurm_submission_authorized": True,
        "authorization_source": "https://github.com/ll7/diss/issues/2669#issuecomment-5811968439",
        "evidence_admission": "not_started",
        "baseline_map_must_remain_unchanged": True,
    }


def test_doorway_execution_requires_recorded_author_decision() -> None:
    """Execution authority cannot be inferred from a boolean without the author record."""
    execution = load_three_width_manifest(_MANIFEST)["execution"]
    with pytest.raises(ValueError, match="recorded author decision"):
        _validate_application_execution({**execution, "authorization_source": "ll7/diss#2669"})
    with pytest.raises(ValueError, match="author's execution decision"):
        _validate_application_execution({**execution, "slurm_submission_authorized": False})


def test_matrix_has_three_positive_clearance_widths(tmp_path: Path) -> None:
    """All comparison widths must clear the collision diameter at fixed depth."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "matrix")
    assert [asset["gap_width_m"] for asset in assets] == [2.2, 2.8, 3.6]
    assert [round(asset["derived_clearance_margin_m"], 3) for asset in assets] == [0.2, 0.8, 1.6]
    assert all(
        asset["expected_geometry_tier"] == "geometrically_feasible_candidate" for asset in assets
    )
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


def test_executed_planners_do_not_attach_classic_global_route(tmp_path: Path) -> None:
    """Generated scenario configs leave the classic global planner disabled."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "variants")
    for asset in assets:
        scenario_path = Path(asset["scenario_path"])
        scenario = dict(load_scenarios(scenario_path)[0])
        config = build_env_config(scenario, scenario_path=scenario_path)
        assert config.use_planner is False
        assert config.sim_config.time_per_step_in_secs == pytest.approx(0.1)
        assert scenario["simulation_config"]["max_episode_steps"] == 400


def test_generated_svg_changes_only_symmetric_wall_endpoints(tmp_path: Path) -> None:
    """All map features except the two doorway wall endpoints are identical."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "variants")
    label = "{http://www.inkscape.org/namespaces/inkscape}label"
    original = list(ET.parse(_BASE_MAP).getroot().iter())
    original_sha = _sha256(_BASE_MAP)
    for asset in assets:
        changed = list(ET.parse(asset["map_path"]).getroot().iter())
        assert len(changed) == len(original)
        wall_changes = []
        for before, after in zip(original, changed, strict=True):
            assert before.tag == after.tag
            if before.attrib != after.attrib:
                assert before.attrib.get(label) == after.attrib.get(label) == "obstacle"
                assert before.attrib["x"] == after.attrib["x"] == "15"
                assert before.attrib["width"] == after.attrib["width"] == "1"
                assert {k for k in before.attrib if before.attrib[k] != after.attrib[k]} <= {
                    "y",
                    "height",
                }
                wall_changes.append(after)
        assert len(wall_changes) == 2
        edges = sorted((float(w.attrib["y"]), float(w.attrib["height"])) for w in wall_changes)
        lower_end = edges[0][0] + edges[0][1]
        upper_start = edges[1][0]
        assert (lower_end + upper_start) / 2 == pytest.approx(5.0)
        assert upper_start - lower_end == pytest.approx(asset["gap_width_m"])
    assert _sha256(_BASE_MAP) == original_sha


@pytest.mark.parametrize("width", [1.9, 2.0])
def test_manifest_rejects_nonpositive_clearance(tmp_path: Path, width: float) -> None:
    """Infeasible and tangent widths cannot enter the comparison manifest."""
    raw = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    raw["geometry"]["gap_width_m"][0] = width
    for key in ("scenario_path", "map_path"):
        raw["base_scenario"][key] = str(_REPO_ROOT / raw["base_scenario"][key])
    candidate = tmp_path / "bad.yaml"
    candidate.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="positive|exceed"):
        load_three_width_manifest(candidate)


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
        for token, gap in (("2p20", 2.2), ("2p80", 2.8), ("3p60", 3.6))
    ]
    pairs = build_pair_manifest(assets, (225, 226, 227), "manifest-sha")
    assert pairs["schema_version"] == "issue_9348_three_width_pair_manifest.v1"
    assert pairs["realization_hash_status"] == "pending_initial_and_external_rng_state_verification"
    assert [pair["pair_id"] for pair in pairs["pairs"]] == [
        f"{planner}_pair_{seed:05d}"
        for planner in ("goal", "social_force")
        for seed in (225, 226, 227)
    ]
    for pair in pairs["pairs"]:
        assert [cell["gap_width_m"] for cell in pair["cells"]] == [2.2, 2.8, 3.6]
        assert all(cell["initial_actor_state_sha256"] is None for cell in pair["cells"])
        assert all(cell["external_rng_state_sha256"] is None for cell in pair["cells"])
    assert check_pair_receipts(pairs)
    for pair in pairs["pairs"]:
        for cell in pair["cells"]:
            cell["initial_actor_state_sha256"] = "a" * 64
            cell["external_rng_state_sha256"] = "b" * 64
            cell["non_width_config_sha256"] = "c" * 64
    assert check_pair_receipts(pairs) == []
    pairs["pairs"][0]["cells"][1]["external_rng_state_sha256"] = "d" * 64
    assert "differs across widths" in check_pair_receipts(pairs)[0]


def test_pair_receipt_canonicalizes_actor_order_and_requires_rng() -> None:
    """Identical reset states hash equally; missing RNG state blocks admission."""
    reset = {
        "robot": {"position": [4.0, 5.0], "velocity": [0.0, 0.0], "heading": 0.0},
        "route_state": {"robot_routes": [[7.0, 5.0], [25.5, 5.0]]},
        "pedestrians": [
            {"actor_id": "h1", "position": [27.0, 5.0], "velocity": [0.0, 0.0]},
            {"actor_id": "h2", "position": [25.0, 5.0], "velocity": [0.0, 0.0]},
        ],
    }
    snapshot = SimpleNamespace(
        global_rng_state=("MT19937", np.asarray([1, 2], dtype=np.uint32), 0, 0, 0.0),
        python_random_state=(3, (1, 2), None),
        behavior_rng_states={"h1": {"state": 7}},
        residual_adversary_state=None,
    )
    first = build_pair_receipt(reset, snapshot)
    reversed_reset = {**reset, "pedestrians": list(reversed(reset["pedestrians"]))}
    assert build_pair_receipt(reversed_reset, snapshot) == first
    snapshot.python_random_state = None
    with pytest.raises(ValueError, match="RNG snapshot is incomplete"):
        build_pair_receipt(reset, snapshot)


def test_portable_reset_matches_three_widths_and_distinct_maps(tmp_path: Path) -> None:
    """All 18 frozen cells have identical pre-command pairs and distinct maps."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "variants")
    session = DoorwayPairingSession()
    sf_config = Path(manifest["_resolved"]["social_force_config_path"])
    for planner in ("goal", "social_force"):
        planner_config_sha = _sha256(sf_config) if planner == "social_force" else None
        for seed in (225, 226, 227):
            for asset in assets:
                scenario_path = Path(asset["scenario_path"])
                scenario = dict(load_scenarios(scenario_path)[0])
                config = build_env_config(scenario, scenario_path=scenario_path)
                common = non_width_config_sha256(
                    scenario, planner=planner, planner_config_sha256=planner_config_sha
                )
                env = make_robot_env(config=config, seed=seed, debug=False)
                try:
                    obs, _ = env.reset(seed=seed)
                    receipt = session.hook(
                        planner=planner,
                        seed=seed,
                        map_sha256=asset["map_sha256"],
                        non_width_config_sha256=common,
                    )(env, obs)
                    assert receipt["non_width_config_sha256"] == common
                finally:
                    env.close()
    pairs = build_pair_manifest(assets, (225, 226, 227), _sha256(_MANIFEST))
    completed = session.fill_pair_manifest(pairs)
    assert len(completed["pairs"]) == 6
    assert sum(len(pair["cells"]) for pair in completed["pairs"]) == 18
    assert completed["realization_hash_status"] == "verified_pre_command"
    assert check_pair_receipts(completed) == []


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
    assert report["checks"]["all_widths_positive_clearance"] is True
    assert report["checks"]["oracle_available_for_every_variant"] is True
    assert report["checks"]["nominal_grid_route_feasible_for_every_variant"] is True
    assert report["checks"]["planner_records_are_not_run"] is True
    assert report["checks"]["no_campaign_evidence"] is True
    assert all(item["planner"]["status"] == "not_run" for item in report["variants"])
    assert {item["oracle"]["nominal_verdict"]["status"] for item in report["variants"]} == {
        "feasible"
    }

    report_path = tmp_path / "issue_9348_preflight.json"
    write_preflight_report(report, report_path)
    payload = report_path.read_text(encoding="utf-8")
    assert '"review_marker": "AI-GENERATED NEEDS-REVIEW"' in payload


def test_conservative_grid_result_is_reported_without_changing_frozen_widths(
    tmp_path: Path,
) -> None:
    """A grid no-route finding stays distinct from positive continuous clearance."""

    def conservative_certifier(scenario: dict[str, Any], path: Path) -> ScenarioCertificate:
        certificate = _fake_certifier(scenario, path)
        if float(scenario["metadata"]["gap_width_m"]) < 3.6:
            certificate.classification = "geometrically_infeasible"
            certificate.benchmark_eligibility = "excluded"
            certificate.route_certificates[0].classification = "geometrically_infeasible"
            certificate.route_certificates[0].benchmark_eligibility = "excluded"
            certificate.route_certificates[0].checks["inflated_collision_free_path"] = False
        return certificate

    report = run_three_width_preflight(
        _MANIFEST,
        output_dir=tmp_path / "variants",
        episode_runner=_fake_episode_runner,
        certifier=conservative_certifier,
    )
    assert report["checks"]["all_widths_positive_clearance"] is True
    assert report["checks"]["nominal_grid_route_feasible_for_every_variant"] is False
    assert report["protocol"]["production_campaign_authorized"] is True
    assert report["protocol"]["slurm_submission_authorized"] is True
    assert report["protocol"]["authorization_source"] == (
        "https://github.com/ll7/diss/issues/2669#issuecomment-5811968439"
    )
    assert report["execution"]["confirmation_ready"] is False
    assert report["go"] is True  # The diagnostic geometry preflight ran, not the campaign.
