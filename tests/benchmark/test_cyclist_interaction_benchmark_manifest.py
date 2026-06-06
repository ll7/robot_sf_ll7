"""Tests for the issue #2473 cyclist interaction benchmark manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs/benchmarks/cyclist_interaction_issue_2473.yaml"


def _manifest() -> dict[str, object]:
    """Load the #2473 cyclist interaction benchmark manifest."""
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_cyclist_interaction_manifest_references_existing_surfaces() -> None:
    """Every tracked current surface in the manifest should exist."""
    manifest = _manifest()
    surfaces = manifest["contract_surfaces"]
    assert isinstance(surfaces, dict)

    for rel_path in surfaces.values():
        assert (REPO_ROOT / str(rel_path)).exists(), rel_path


def test_scenario_families_cover_cyclist_specific_interaction_modes() -> None:
    """The suite should cover crossing, following, overtaking, merging, and shared-space cases."""
    manifest = _manifest()
    families = manifest["scenario_families"]
    assert isinstance(families, list)

    family_keys = {family["key"] for family in families}
    assert {
        "cyclist_crossing",
        "same_lane_following",
        "cyclist_overtake_pass_by",
        "bike_lane_merge_transition",
        "mixed_shared_space_encounter",
    }.issubset(family_keys)
    statuses = {family["required_runtime_status"] for family in families}
    assert "proxy_surface_only" in statuses
    assert "future_cyclist_actor_required" in statuses
    for family in families:
        assert family["initial_case_ids"], family["key"]
        assert (REPO_ROOT / str(family["nearest_existing_proxy"])).exists()


def test_dynamics_assumptions_distinguish_cyclists_from_pedestrians() -> None:
    """Cyclist assumptions should name speed, acceleration, heading, lane, and turn constraints."""
    manifest = _manifest()
    dynamics = manifest["cyclist_dynamics_assumptions"]
    assert isinstance(dynamics, dict)

    future_fields = set(dynamics["future_explicit_fields"])
    differences = set(dynamics["non_pedestrian_differences"])
    speed_range = dynamics["candidate_ranges"]["speed_m_s"]
    assert {"actor_type", "speed_m_s", "heading_rad", "route_intent"}.issubset(future_fields)
    assert "min_turn_radius_m" in future_fields
    assert "higher_nominal_speed" in differences
    assert "nonholonomic_or_turn_radius_constraints" in differences
    assert speed_range["low"] >= 2.0
    assert speed_range["high"] > speed_range["low"]


def test_required_metrics_and_trace_fields_include_cyclist_specific_evidence() -> None:
    """The direction should specify cyclist-specific metrics and future trace fields."""
    manifest = _manifest()
    metrics = manifest["required_metrics"]
    trace_fields = manifest["required_trace_fields"]
    assert isinstance(metrics, dict)
    assert isinstance(trace_fields, dict)

    assert "success" in metrics["existing_metrics"]
    assert "relative_closing_speed_m_s" in metrics["cyclist_specific_metrics"]
    assert "cyclist_ttc_violation_count" in metrics["cyclist_specific_metrics"]
    assert "outcome.collision_event" in trace_fields["existing_trace_like_fields"]
    assert "actor_type_by_id" in trace_fields["future_cyclist_trace_fields"]
    assert "pass_clearance_by_step" in trace_fields["future_cyclist_trace_fields"]


def test_fallback_policy_rejects_proxy_surfaces_as_benchmark_evidence() -> None:
    """Adjacent proxy surfaces should not be counted as cyclist benchmark evidence."""
    manifest = _manifest()
    reporting = manifest["fallback_and_reporting_policy"]
    claim_boundary = str(manifest["claim_boundary"]).lower()
    first_spike = manifest["first_smoke_or_spike"]
    assert isinstance(reporting, dict)
    assert isinstance(first_spike, dict)

    assert manifest["benchmark_evidence"] is False
    assert "does not prove" in claim_boundary
    assert "cyclist_realism_claim" in reporting["proxy_surfaces_not_allowed_for"]
    assert (
        "no fallback/degraded/not_available row counted as benchmark success"
        in reporting["benchmark_success_requires"]
    )
    assert first_spike["type"] == "one_executable_cyclist_actor_smoke"
