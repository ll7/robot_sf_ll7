"""Tests for the issue #3207 footprint / clearance-semantics diagnostic."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.clearance_semantics import (
    CONSERVATIVE_BUFFER_BREACH,
    FOOTPRINT_CLEARANCE_SCHEMA,
    GEOMETRIC_BODY_CLEARANCE,
    PEDESTRIAN_CONTACT,
    PROXY_ENVELOPE_SURFACE_CLEARANCE,
    ClearanceGeometry,
    ClearanceThresholds,
    build_collision_threshold_sensitivity_table,
    build_footprint_clearance_manifest,
    classify_encounter,
    enumerate_footprint_sweep,
    evaluate_clearance,
    load_footprint_sweep_spec,
    write_footprint_clearance_manifest,
)
from robot_sf.benchmark.fidelity_sensitivity import load_fidelity_sensitivity_config

_CONFIG_PATH = "configs/research/fidelity_sensitivity_v1.yaml"

if TYPE_CHECKING:
    from pathlib import Path


def _repo_config() -> dict[str, object]:
    return load_fidelity_sensitivity_config(_CONFIG_PATH)


def _nominal_geometry() -> ClearanceGeometry:
    return ClearanceGeometry(
        robot_proxy_radius_m=1.00,
        pedestrian_radius_m=0.40,
        robot_body_radius_m=0.35,
        pedestrian_body_radius_m=0.25,
    )


def _nominal_thresholds() -> ClearanceThresholds:
    return ClearanceThresholds(
        contact_threshold_m=0.0,
        near_miss_threshold_m=0.20,
        conservative_buffer_m=0.30,
    )


def test_evaluate_clearance_separates_proxy_envelope_from_body_contact() -> None:
    """The maintainer example: proxy overlap without physical body contact."""
    evaluation = evaluate_clearance(
        _nominal_geometry(),
        _nominal_thresholds(),
        center_to_center_distance_m=1.37,
    )

    # 1.37 - (1.00 + 0.40) = -0.03 m envelope overlap.
    assert evaluation[PROXY_ENVELOPE_SURFACE_CLEARANCE] == pytest.approx(-0.03)
    # 1.37 - (0.35 + 0.25) = 0.77 m body clearance: no physical contact.
    assert evaluation[GEOMETRIC_BODY_CLEARANCE] == pytest.approx(0.77)
    assert evaluation[PEDESTRIAN_CONTACT] is False
    assert evaluation[CONSERVATIVE_BUFFER_BREACH] is True
    assert evaluation["encounter_class"] == "proxy_envelope_overlap"


def test_evaluate_clearance_reports_pedestrian_contact_only_on_body_overlap() -> None:
    """Pedestrian contact is driven by body radii, not planning proxy radii."""
    evaluation = evaluate_clearance(
        _nominal_geometry(),
        _nominal_thresholds(),
        center_to_center_distance_m=0.50,
    )

    # 0.50 - (0.35 + 0.25) = -0.10 m body overlap.
    assert evaluation[GEOMETRIC_BODY_CLEARANCE] == pytest.approx(-0.10)
    assert evaluation[PEDESTRIAN_CONTACT] is True
    assert evaluation["encounter_class"] == "pedestrian_contact"


def test_evaluate_clearance_obstacle_contact_is_optional() -> None:
    """Obstacle contact is None unless an obstacle surface distance is supplied."""
    without = evaluate_clearance(
        _nominal_geometry(), _nominal_thresholds(), center_to_center_distance_m=2.0
    )
    assert without["obstacle_contact"] is None

    with_obstacle = evaluate_clearance(
        _nominal_geometry(),
        _nominal_thresholds(),
        center_to_center_distance_m=2.0,
        obstacle_body_surface_distance_m=0.0,
    )
    assert with_obstacle["obstacle_contact"] is True


def test_classify_encounter_escalating_labels() -> None:
    """Encounter classification returns the tightest applicable severity band."""
    thresholds = _nominal_thresholds()
    assert (
        classify_encounter(
            proxy_envelope_surface_clearance_m=0.5,
            geometric_body_clearance_m=-0.01,
            thresholds=thresholds,
        )
        == "pedestrian_contact"
    )
    assert (
        classify_encounter(
            proxy_envelope_surface_clearance_m=-0.03,
            geometric_body_clearance_m=0.77,
            thresholds=thresholds,
        )
        == "proxy_envelope_overlap"
    )
    # 0.10 m proxy clearance is within the tighter near-miss band (<= 0.20 m).
    assert (
        classify_encounter(
            proxy_envelope_surface_clearance_m=0.10,
            geometric_body_clearance_m=1.0,
            thresholds=thresholds,
        )
        == "near_miss"
    )
    # 0.25 m is past the near-miss band but inside the wider conservative buffer (< 0.30 m).
    assert (
        classify_encounter(
            proxy_envelope_surface_clearance_m=0.25,
            geometric_body_clearance_m=1.0,
            thresholds=thresholds,
        )
        == "conservative_buffer_breach"
    )
    assert (
        classify_encounter(
            proxy_envelope_surface_clearance_m=0.9,
            geometric_body_clearance_m=1.5,
            thresholds=thresholds,
        )
        == "clear"
    )


def test_geometry_rejects_body_larger_than_proxy() -> None:
    """A body radius exceeding its planning proxy radius fails closed."""
    with pytest.raises(ValueError, match="robot_body_radius_m must not exceed"):
        ClearanceGeometry(
            robot_proxy_radius_m=0.30,
            pedestrian_radius_m=0.40,
            robot_body_radius_m=0.35,
            pedestrian_body_radius_m=0.25,
        )


def test_load_footprint_sweep_spec_from_repo_config() -> None:
    """The tracked config yields the expected footprint sweep grid."""
    spec = load_footprint_sweep_spec(_repo_config())

    assert spec.robot_proxy_radii_m == (0.60, 1.00, 1.40)
    assert spec.pedestrian_radii_m == (0.30, 0.40, 0.50)
    assert spec.encounter_center_distances_m == (1.10, 1.37, 1.80)
    assert spec.nominal_geometry.robot_proxy_radius_m == pytest.approx(1.00)


def test_load_footprint_sweep_spec_fails_closed_when_block_missing() -> None:
    """A missing footprint_semantics block raises rather than enumerating nothing."""
    config = _repo_config()
    config.pop("footprint_semantics")

    with pytest.raises(ValueError, match="missing required 'footprint_semantics' block"):
        load_footprint_sweep_spec(config)


def test_load_footprint_sweep_spec_fails_closed_on_wrong_schema() -> None:
    """A wrong footprint_semantics schema_version fails closed."""
    config = _repo_config()
    config["footprint_semantics"] = {"schema_version": "wrong.v0"}

    with pytest.raises(ValueError, match="schema_version must be"):
        load_footprint_sweep_spec(config)


def test_load_footprint_sweep_spec_fails_closed_on_empty_sweep_list() -> None:
    """An empty sweep list fails closed."""
    config = _repo_config()
    config["footprint_semantics"]["sweep"]["robot_proxy_radius_m"] = []

    with pytest.raises(ValueError, match="robot_proxy_radius_m must be a non-empty list"):
        load_footprint_sweep_spec(config)


def test_enumerate_footprint_sweep_is_full_grid() -> None:
    """The sweep enumerates the full proxy x pedestrian x distance grid."""
    spec = load_footprint_sweep_spec(_repo_config())

    cells = enumerate_footprint_sweep(spec)

    assert len(cells) == 3 * 3 * 3
    first = cells[0]
    assert set(first) == {
        "robot_proxy_radius_m",
        "pedestrian_radius_m",
        "center_to_center_distance_m",
        "evaluation",
    }


def test_threshold_sensitivity_flags_proxy_radius_dependence() -> None:
    """The table flags rows whose class depends on the proxy radius choice."""
    spec = load_footprint_sweep_spec(_repo_config())

    table = build_collision_threshold_sensitivity_table(spec)

    assert len(table) == 3 * 3  # pedestrian_radius x distance
    # At least one (pedestrian_radius, distance) row must flip class across proxy radii,
    # otherwise the diagnostic would show no footprint sensitivity at all.
    assert any(row["proxy_radius_sensitive"] for row in table)
    for row in table:
        assert row["distinct_class_count"] >= 1
        assert (row["distinct_class_count"] > 1) == row["proxy_radius_sensitive"]


def test_build_manifest_boundary_and_contract() -> None:
    """The manifest carries the diagnostic boundary and required-output contract."""
    manifest = build_footprint_clearance_manifest(
        _repo_config(), config_path=_CONFIG_PATH, git_head="abc1234"
    )

    assert manifest["schema_version"] == FOOTPRINT_CLEARANCE_SCHEMA
    assert manifest["issue"] == 3207
    assert manifest["dry_run"] is True
    assert manifest["evidence_status"] == "not_benchmark_evidence"
    assert "diagnostic only" in manifest["claim_boundary"]
    assert "does not change frozen-release" in manifest["claim_boundary"]
    assert manifest["cell_count"] == 27
    assert len(manifest["clearance_quantity_definitions"]) == 6
    assert "proxy_radius_sensitive" in manifest["required_outputs"]
    assert manifest["proxy_radius_sensitive_row_count"] >= 1


def test_write_manifest_is_deterministic_json(tmp_path: Path) -> None:
    """The manifest is written as reloadable deterministic JSON."""
    manifest = build_footprint_clearance_manifest(
        _repo_config(), config_path=_CONFIG_PATH, git_head="abc1234"
    )

    path = write_footprint_clearance_manifest(manifest, tmp_path)

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["schema_version"] == FOOTPRINT_CLEARANCE_SCHEMA
    assert reloaded["cell_count"] == 27
