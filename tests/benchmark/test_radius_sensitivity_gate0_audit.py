"""Tests for the Gate 0 post-hoc feasibility-audit decision (issue #6640).

These tests pin the post-hoc-versus-replay boundary for the collision-envelope radius campaign
(parent #6600) and enforce the immutable stop conditions: trajectory-dependent planner,
obstacle-contact, feasibility, and collision outcomes are always replay-required; the
re-derivable set is currently empty because the frozen release has unresolved effective-radius and
map-provenance gaps; the decision must not read as a radius sweep.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.constants import COLLISION_DIST, NEAR_MISS_DIST
from robot_sf.benchmark.radius_sensitivity_gate0_audit import (
    CAMPAIGN_ARMS_M,
    CAMPAIGN_BASELINE_ARM_M,
    GATE0_DECISION_SCHEMA,
    RADIUS_AWARE_CLEARANCE_METRICS,
    RE_DERIVABLE,
    REPLAY_REQUIRED,
    TRAJECTORY_DEPENDENT_CATEGORIES,
    build_gate0_decision,
    build_outcome_registry,
    load_gate0_decision,
    validate_gate0_decision,
    write_gate0_decision,
)

COMMITTED_DECISION = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "context"
    / "radius_sensitivity_gate0_audit_issue_6640.json"
)


def _outcomes_by_id(decision):
    return {o["outcome_id"]: o for o in decision["outcomes"]}


def test_decision_schema_and_provenance():
    decision = build_gate0_decision()
    assert decision["schema_version"] == GATE0_DECISION_SCHEMA
    assert decision["issue"] == 6640
    assert decision["parent_issue"] == 6600
    assert decision["validity_study_issue"] == 3207
    assert decision["gate"] == "gate0_post_hoc_feasibility_audit"
    assert decision["next_gate"] == "gate1_binding_canary"
    assert "diagnostic" in decision["claim_boundary"].lower()


def test_campaign_axis_is_collision_envelope_radius_with_three_arms():
    decision = build_gate0_decision()
    campaign = decision["campaign"]
    assert campaign["axis"] == "robot_collision_envelope_radius_m"
    assert campaign["arms_m"] == [0.5, 0.8, 1.0]
    assert CAMPAIGN_ARMS_M == (0.5, 0.8, 1.0)
    assert CAMPAIGN_BASELINE_ARM_M == 1.0
    assert campaign["baseline_arm_m"] == 1.0
    assert campaign["horizon_steps"] == 600


def test_frozen_release_pointer_matches_tracked_evidence():
    decision = build_gate0_decision()
    frozen = decision["frozen_release"]
    assert frozen["release_tag"] == "0.0.3.post1"
    assert frozen["episode_rows"] == 20160
    assert frozen["arms"] == 14
    assert frozen["rows_per_arm"] == 1440
    assert frozen["execution_commit"] == "a307ef276d701f8d14dead1aa0513f44ee97c0b0"
    assert frozen["artifact_pointer"].endswith("artifact_pointer.json")
    # The audit inspects the row schema and metric contract, not the bundle bytes.
    assert (
        "schema" in frozen["row_location"].lower() or "contract" in frozen["row_location"].lower()
    )


def test_metric_contract_grounding_matches_frozen_constants():
    decision = build_gate0_decision()
    contract = decision["metric_contract"]
    # Radius-aware clearance family is exactly the four metrics that subtract (robot+ped) radius.
    assert set(contract["radius_aware_clearance_metrics"]) == set(RADIUS_AWARE_CLEARANCE_METRICS)
    assert set(RADIUS_AWARE_CLEARANCE_METRICS) == {
        "human_collisions",
        "near_misses",
        "min_clearance",
        "mean_clearance",
    }
    assert "robot_radius_m + ped_radius_m" in contract["clearance_formula"]
    # Frozen constants are referenced verbatim, not redefined.
    assert contract["collision_constants"]["COLLISION_DIST_m"] == COLLISION_DIST
    assert contract["collision_constants"]["NEAR_MISS_DIST_m"] == NEAR_MISS_DIST
    for metric in ("wall_collisions", "agent_collisions"):
        assert (
            contract["fixed_threshold_collision_metrics"][metric]["threshold_m"] == COLLISION_DIST
        )
        assert (
            contract["fixed_threshold_collision_metrics"][metric]["uses_radius_in_formula"] is False
        )


def test_radius_default_inconsistency_is_recorded_as_gate0_finding():
    decision = build_gate0_decision()
    inconsistency = decision["metric_contract"]["radius_default_inconsistency"]
    # metrics.py EpisodeData default is 1.0 m; runner.py default is 0.3 m -> not uniform.
    assert inconsistency["metrics_episode_data_default_robot_radius_m"] == 1.0
    assert inconsistency["runner_default_robot_radius_m"] == 0.3
    assert (
        inconsistency["metrics_episode_data_default_robot_radius_m"]
        != (inconsistency["runner_default_robot_radius_m"])
    )
    assert "Gate 1" in inconsistency["finding"]


def test_threshold_sensitivity_requires_replay_is_documented():
    decision = build_gate0_decision()
    note = decision["metric_contract"]["threshold_sensitivity_requires_replay"]
    assert "threshold_sensitivity.py" in note
    assert "replay" in note.lower()


def test_every_outcome_has_unique_id_and_valid_classification():
    decision = build_gate0_decision()
    ids = [o["outcome_id"] for o in decision["outcomes"]]
    assert len(ids) == len(set(ids))
    assert len(decision["outcomes"]) >= 20
    for outcome in decision["outcomes"]:
        assert outcome["classification"] in {RE_DERIVABLE, REPLAY_REQUIRED}
        assert isinstance(outcome["caveats"], list)


@pytest.mark.parametrize(
    "outcome_id",
    [
        "human_collisions_count",
        "near_misses_count",
        "wall_collisions_count",
        "agent_collisions_count",
        "binary_success",
        "total_collision_count",
        "ped_collision_count",
        "simulator_obstacle_contact",
        "geometric_body_pedestrian_contact",
        "trajectory_feasibility_traversal_executed",
        "planner_behaviour_decisions",
        "planner_rankings_success_typed_collisions_snqi",
        "scenario_family_conclusions_transitions",
        "snqi_per_episode",
    ],
)
def test_collision_contact_feasibility_planner_outcomes_are_replay_required(outcome_id):
    """Stop condition #3: these outcome classes must never be re-derivable."""
    decision = build_gate0_decision()
    outcomes = _outcomes_by_id(decision)
    assert outcomes[outcome_id]["classification"] == REPLAY_REQUIRED
    assert outcomes[outcome_id]["is_collision_contact_feasibility_or_planner_outcome"] is True


@pytest.mark.parametrize(
    "outcome_id",
    ["min_clearance_scalar", "mean_clearance_scalar"],
)
def test_clearance_scalars_are_replay_required_but_not_collision_outcomes(outcome_id):
    """Clearance distances are replay-required (trajectory-dependent) but are distances, not
    collision/contact outcomes, so they must not carry the collision-outcome flag."""
    decision = build_gate0_decision()
    outcome = _outcomes_by_id(decision)[outcome_id]
    assert outcome["classification"] == REPLAY_REQUIRED
    assert outcome["is_collision_contact_feasibility_or_planner_outcome"] is False


def test_no_trajectory_dependent_category_is_re_derivable():
    """Stop condition #3 via category: trajectory-dependent categories are replay-required."""
    decision = build_gate0_decision()
    for outcome in decision["outcomes"]:
        if outcome["category"] in TRAJECTORY_DEPENDENT_CATEGORIES:
            assert outcome["classification"] == REPLAY_REQUIRED, outcome["outcome_id"]


def test_radius_aware_clearance_metrics_are_all_replay_required():
    """Each radius-aware clearance metric maps to a replay-required outcome."""
    decision = build_gate0_decision()
    outcomes = _outcomes_by_id(decision)
    expected = {
        "human_collisions": "human_collisions_count",
        "near_misses": "near_misses_count",
        "min_clearance": "min_clearance_scalar",
        "mean_clearance": "mean_clearance_scalar",
    }
    for metric, outcome_id in expected.items():
        assert metric in RADIUS_AWARE_CLEARANCE_METRICS
        assert outcomes[outcome_id]["classification"] == REPLAY_REQUIRED


def test_unretained_radius_and_map_provenance_are_replay_required():
    """Gate 0 must fail closed when exact radius or map provenance is absent."""
    decision = build_gate0_decision()
    outcomes = _outcomes_by_id(decision)
    provenance_blocked = {
        "retained_radius_and_threshold_parameters",
        "static_map_geometry_feasibility_margin",
    }
    for outcome_id in provenance_blocked:
        outcome = outcomes[outcome_id]
        assert outcome["classification"] == REPLAY_REQUIRED
        assert outcome["source_geometry_retained_in_frozen_rows"] is False
        assert outcome["is_collision_contact_feasibility_or_planner_outcome"] is False

    assert [o for o in decision["outcomes"] if o["classification"] == RE_DERIVABLE] == []


def test_static_geometry_margin_records_unpinned_map_provenance():
    decision = build_gate0_decision()
    outcomes = _outcomes_by_id(decision)
    static = outcomes["static_map_geometry_feasibility_margin"]
    caveats = " ".join(static["caveats"]).lower()
    assert "matrix checksum" in caveats
    assert "map asset" in caveats
    gaps = decision["metric_contract"]["frozen_provenance_gaps"]
    assert gaps["effective_robot_and_pedestrian_radius_retained"] is False
    assert gaps["map_asset_bytes_pinned"] is False


def test_summary_counts_match_outcomes():
    decision = build_gate0_decision()
    summary = decision["summary"]
    total = len(decision["outcomes"])
    re_derivable = sum(1 for o in decision["outcomes"] if o["classification"] == RE_DERIVABLE)
    replay = total - re_derivable
    assert summary["total_outcomes"] == total
    assert summary["re_derivable_count"] == re_derivable
    assert summary["replay_required_count"] == replay
    assert summary["replay_required_count"] == 24
    assert summary["re_derivable_count"] == 0


def test_decision_is_deterministic():
    first = build_gate0_decision()
    second = build_gate0_decision()
    assert first == second
    # Registry order is stable.
    ids = [o["outcome_id"] for o in first["outcomes"]]
    assert ids[0] == "human_collisions_count"
    assert ids[-1] == "static_map_geometry_feasibility_margin"


def test_committed_json_matches_builder_and_validates():
    """The checked-in artifact must equal the deterministic builder output."""
    assert COMMITTED_DECISION.exists(), COMMITTED_DECISION
    committed = load_gate0_decision(COMMITTED_DECISION)
    rebuilt = build_gate0_decision()
    assert committed == rebuilt


def test_write_round_trips(tmp_path):
    out = write_gate0_decision(tmp_path / "nested" / "decision.json")
    assert out.exists()
    reloaded = load_gate0_decision(out)
    assert reloaded == build_gate0_decision()


def test_validator_rejects_collision_outcome_marked_re_derivable():
    decision = build_gate0_decision()
    tampered = copy.deepcopy(decision)
    for outcome in tampered["outcomes"]:
        if outcome["outcome_id"] == "human_collisions_count":
            outcome["classification"] = RE_DERIVABLE
    with pytest.raises(ValueError, match="must be 'replay-required'"):
        validate_gate0_decision(tampered)


def test_validator_rejects_re_derivable_without_retained_geometry():
    decision = build_gate0_decision()
    tampered = copy.deepcopy(decision)
    for outcome in tampered["outcomes"]:
        if outcome["outcome_id"] == "retained_radius_and_threshold_parameters":
            outcome["classification"] = RE_DERIVABLE
            outcome["source_geometry_retained_in_frozen_rows"] = False
    with pytest.raises(ValueError, match="must retain its source geometry"):
        validate_gate0_decision(tampered)


def test_validator_rejects_pinned_looking_provenance_without_gate_evidence():
    decision = build_gate0_decision()
    tampered = copy.deepcopy(decision)
    for outcome in tampered["outcomes"]:
        if outcome["outcome_id"] == "static_map_geometry_feasibility_margin":
            outcome["classification"] = RE_DERIVABLE
            outcome["source_geometry_retained_in_frozen_rows"] = True
    with pytest.raises(ValueError, match="without exact retained provenance"):
        validate_gate0_decision(tampered)


def test_validator_rejects_provenance_gap_flip():
    decision = build_gate0_decision()
    tampered = copy.deepcopy(decision)
    tampered["metric_contract"]["frozen_provenance_gaps"]["map_asset_bytes_pinned"] = True
    with pytest.raises(ValueError, match="map_asset_bytes_pinned must be false"):
        validate_gate0_decision(tampered)


def test_validator_rejects_over_broad_re_derivable_set():
    """Stop condition #5: an unapproved re-derivable entry must be rejected."""
    decision = build_gate0_decision()
    tampered = copy.deepcopy(decision)
    extra = copy.deepcopy(tampered["outcomes"][0])
    extra["outcome_id"] = "bogus_re_derivable"
    extra["classification"] = RE_DERIVABLE
    extra["category"] = "metadata_parameter"
    extra["is_collision_contact_feasibility_or_planner_outcome"] = False
    extra["source_geometry_retained_in_frozen_rows"] = True
    extra["caveats"] = []
    tampered["outcomes"].append(extra)
    with pytest.raises(ValueError, match="imply a sweep"):
        validate_gate0_decision(tampered)


def test_validator_rejects_bad_schema_and_empty_outcomes():
    bad_schema = build_gate0_decision()
    bad_schema["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema_version"):
        validate_gate0_decision(bad_schema)
    empty = build_gate0_decision()
    empty["outcomes"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        validate_gate0_decision(empty)


@pytest.mark.parametrize(
    "field",
    [
        "outcome",
        "category",
        "radius_binding",
        "source_geometry_retained_in_frozen_rows",
        "is_collision_contact_feasibility_or_planner_outcome",
        "rationale",
        "caveats",
    ],
)
def test_validator_rejects_outcome_missing_required_schema_field(field):
    decision = build_gate0_decision()
    tampered = copy.deepcopy(decision)
    del tampered["outcomes"][0][field]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_gate0_decision(tampered)


def test_validator_rejects_malformed_outcome_field_types():
    decision = build_gate0_decision()
    tampered = copy.deepcopy(decision)
    tampered["outcomes"][0]["is_collision_contact_feasibility_or_planner_outcome"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        validate_gate0_decision(tampered)


def test_registry_is_consistent_with_decision():
    registry = build_outcome_registry()
    decision = build_gate0_decision()
    assert [o.outcome_id for o in registry] == [o["outcome_id"] for o in decision["outcomes"]]
    # JSON-serialisable and stable under json round-trip.
    json.dumps(decision)
