"""Versioned contact provenance preserves exact collisions and historical defaults."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from robot_sf.benchmark.event_ledger import (
    build_event_ledger,
    reconcile_event_ledger,
    validate_record_event_ledger,
)
from robot_sf.benchmark.map_runner.map_runner_episode import (
    _CollisionEventContext,
    _step_collision_events,
)


def _record() -> dict:
    return {
        "scenario_id": "contact-fixture",
        "seed": 113,
        "algo": "stand_still",
        "metrics": {"collisions": 1.0},
        "outcome": {"collision_event": True, "route_complete": False},
        "termination_reason": "collision",
        "scenario_params": {"collision_attribution_version": "v1"},
        "spawn_validity": {
            "reset_overlap": False,
            "reset_clearance_status": "available",
            "respawn_overlap_collisions": [],
        },
    }


def _step_event(previous_robot_pos: np.ndarray | None, *, step_idx: int = 2) -> dict:
    context = _CollisionEventContext(
        dt_seconds=0.1,
        map_def=None,
        robot_radius=0.5,
        ped_radius=0.4,
        capture_robot_speed=True,
    )
    events = _step_collision_events(
        step_idx=step_idx,
        robot_pos=np.array([1.0, 0.0]),
        previous_robot_pos=previous_robot_pos,
        ped_positions=np.array([[1.2, 0.0]]),
        previous_ped_positions=np.array([[1.5, 0.0]]),
        meta={"is_pedestrian_collision": True},
        context=context,
    )
    return events[0]


@pytest.mark.parametrize(
    ("previous_position", "expected_class", "expected_speed"),
    [
        (np.array([1.0, 0.0]), "pedestrian_on_stationary_robot", 0.0),
        (np.array([0.8, 0.0]), "moving_robot_pedestrian_contact", 2.0),
        (None, "unresolved_missing_robot_speed", None),
    ],
)
def test_step_displacement_classifies_contact_without_changing_exact_collision(
    previous_position: np.ndarray | None, expected_class: str, expected_speed: float | None
) -> None:
    """Measured robot displacement distinguishes stationary, moving, and unknown."""
    event = _step_event(previous_position)
    record = _record()
    ledger = build_event_ledger(record, collision_events=[event])
    contact = ledger["contact_provenance"]["contacts"][0]
    assert contact["contact_class"] == expected_class
    if expected_speed is None:
        assert contact["robot_speed_at_contact_m_s"] is None
    else:
        assert contact["robot_speed_at_contact_m_s"] == pytest.approx(expected_speed)
    assert ledger["exact_events"]["collision"] is True
    assert ledger["reconciliation"]["collision_metric_value"] == 1.0
    assert len(ledger["collision_events"]) == 1
    assert ledger["contact_provenance"]["planner_causation_admitted"] is False
    assert reconcile_event_ledger(ledger) == []


def test_respawn_match_takes_precedence_and_reset_is_separate() -> None:
    """A matched respawn contact and reset overlap retain distinct provenance."""
    record = _record()
    record["spawn_validity"]["reset_overlap"] = True
    record["spawn_validity"]["respawn_overlap_collisions"] = [
        {"group_id": "g1", "ped_row": 0, "collision_time_s": 0.3}
    ]
    ledger = build_event_ledger(record, collision_events=[_step_event(np.array([1.0, 0.0]))])
    block = ledger["contact_provenance"]
    assert block["reset_overlap"] is True
    assert block["classification_status"] == "invalid_start"
    assert block["contacts"][0]["contact_class"] == "respawn_contact"
    assert block["contacts"][0]["respawn_match"]["group_id"] == "g1"
    assert len(ledger["collision_events"]) == 1


def test_first_step_contact_with_reset_overlapping_row_is_reset_class() -> None:
    """A measured first-step collision with an overlapping reset row is setup-caused."""
    record = _record()
    record["spawn_validity"]["reset_overlap"] = True
    record["spawn_validity"]["reset_clearance"] = {
        "overlap": True,
        "overlapping_pedestrian_rows": [0],
    }
    ledger = build_event_ledger(
        record, collision_events=[_step_event(np.array([1.0, 0.0]), step_idx=0)]
    )
    contact = ledger["contact_provenance"]["contacts"][0]
    assert contact["reset_overlap_match"] is True
    assert contact["contact_class"] == "reset_overlap_contact"
    assert ledger["contact_provenance"]["classification_status"] == "invalid_start"
    assert reconcile_event_ledger(ledger) == []


def test_historical_default_event_and_ledger_do_not_gain_attribution() -> None:
    """The opt-in cannot alter default raw events or historical ledger bytes."""
    legacy_context = _CollisionEventContext(
        dt_seconds=0.1, map_def=None, robot_radius=0.5, ped_radius=0.4
    )
    legacy_event = _step_collision_events(
        step_idx=2,
        robot_pos=np.array([1.0, 0.0]),
        previous_robot_pos=np.array([1.0, 0.0]),
        ped_positions=np.array([[1.2, 0.0]]),
        previous_ped_positions=np.array([[1.5, 0.0]]),
        meta={"is_pedestrian_collision": True},
        context=legacy_context,
    )[0]
    assert "robot_speed_at_contact_m_s" not in legacy_event
    record = _record()
    del record["scenario_params"]["collision_attribution_version"]
    baseline = build_event_ledger(record, collision_events=[legacy_event])
    with_extra_runtime_key = deepcopy(legacy_event)
    with_extra_runtime_key["robot_speed_at_contact_m_s"] = 0.0
    assert build_event_ledger(record, collision_events=[with_extra_runtime_key]) == baseline
    assert "contact_provenance" not in baseline


def test_forged_stationary_class_with_moving_speed_fails_reconciliation() -> None:
    """The opt-in block cannot relabel a measured moving contact as stationary."""
    ledger = build_event_ledger(_record(), collision_events=[_step_event(np.array([0.8, 0.0]))])
    ledger["contact_provenance"]["contacts"][0]["contact_class"] = "pedestrian_on_stationary_robot"
    assert any("class inconsistent with speed" in issue for issue in reconcile_event_ledger(ledger))


def test_missing_step_index_cannot_admit_a_contact_class() -> None:
    """Saved speed needs a step index so the release trace can audit its source."""
    event = _step_event(np.array([1.0, 0.0]))
    del event["contact_step_index"]
    ledger = build_event_ledger(_record(), collision_events=[event])
    contact = ledger["contact_provenance"]["contacts"][0]
    assert contact["contact_class"] == "unresolved_missing_step_index"
    assert ledger["contact_provenance"]["classification_status"] == (
        "unresolved_contact_measurement"
    )
    assert reconcile_event_ledger(ledger) == []


def test_opt_in_row_rejects_missing_block_and_unknown_version() -> None:
    """A candidate row cannot silently omit or invent the attribution version."""
    record = _record()
    record["event_ledger"] = build_event_ledger(record)
    del record["event_ledger"]["contact_provenance"]
    assert "opt-in contact_provenance.v1 block is missing" in validate_record_event_ledger(record)
    record["scenario_params"]["collision_attribution_version"] = "v2"
    assert any(
        "unsupported collision_attribution_version" in issue
        for issue in validate_record_event_ledger(record)
    )
