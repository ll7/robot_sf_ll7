"""Versioned contact provenance preserves exact collisions and historical defaults."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.event_ledger import (
    build_event_ledger,
    reconcile_event_ledger,
    validate_record_event_ledger,
)
from robot_sf.benchmark.map_runner import map_runner_episode
from robot_sf.benchmark.map_runner.map_runner_episode import (
    _CollisionEventContext,
    _step_collision_events,
)
from robot_sf.benchmark.map_runner.map_runner_jsonl import write_validated_to_handle


def _record() -> dict:
    return {
        "scenario_id": "contact-fixture",
        "seed": 1001,
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
    assert (
        ledger["collision_events"][0]["robot_speed_at_contact_m_s"]
        == (contact["robot_speed_at_contact_m_s"])
    )
    assert ledger["collision_events"][0]["contact_step_index"] == 2
    assert contact["robot_speed_measurement_window"] == "simulation_step"
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


@pytest.mark.parametrize("match_kind", ["reset", "respawn"])
def test_two_partner_event_with_one_setup_match_stays_unresolved(match_kind: str) -> None:
    """One setup-matched pedestrian cannot label another simultaneous contact."""
    context = _CollisionEventContext(
        dt_seconds=0.1,
        map_def=None,
        robot_radius=0.5,
        ped_radius=0.4,
        capture_robot_speed=True,
    )
    step_idx = 0 if match_kind == "reset" else 2
    event = _step_collision_events(
        step_idx=step_idx,
        robot_pos=np.array([1.0, 0.0]),
        previous_robot_pos=np.array([1.0, 0.0]),
        ped_positions=np.array([[1.1, 0.0], [1.2, 0.0]]),
        previous_ped_positions=np.array([[1.1, 0.0], [1.2, 0.0]]),
        meta={"is_pedestrian_collision": True},
        context=context,
    )[0]
    assert event["contact_partner_ids"] == ["0", "1"]
    record = _record()
    if match_kind == "reset":
        record["spawn_validity"].update(
            reset_overlap=True,
            reset_clearance={"overlap": True, "overlapping_pedestrian_rows": [1]},
        )
    else:
        record["spawn_validity"]["respawn_overlap_collisions"] = [
            {"group_id": "g1", "ped_row": 1, "collision_time_s": 0.3}
        ]
    ledger = build_event_ledger(record, collision_events=[event])
    contact = ledger["contact_provenance"]["contacts"][0]
    assert contact["contact_class"] == "unresolved_multi_pedestrian_attribution"
    assert ledger["exact_events"]["collision"] is True
    assert ledger["reconciliation"]["collision_metric_value"] == 1.0
    assert reconcile_event_ledger(ledger) == []
    if match_kind == "respawn":
        assert ledger["contact_provenance"]["classification_status"] == (
            "unresolved_contact_measurement"
        )
    forged = deepcopy(ledger)
    forged["contact_provenance"]["contacts"][0]["contact_class"] = (
        "reset_overlap_contact" if match_kind == "reset" else "respawn_contact"
    )
    assert any("class inconsistent" in issue for issue in reconcile_event_ledger(forged))


def test_versioned_scenario_marker_saves_measured_contact_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The episode marker enables runtime speed capture in a saved contact row."""

    class ContactEnv:
        def __init__(self) -> None:
            self.simulator = SimpleNamespace(
                robot_pos=[np.array([0.0, 0.0])],
                ped_pos=np.array([[2.0, 0.0]]),
                goal_pos=[np.array([3.0, 0.0])],
                map_def=SimpleNamespace(obstacles=[], bounds=(0.0, 0.0, 10.0, 10.0)),
                robot_vel=[np.array([0.0, 0.0])],
                get_pedestrian_forces=lambda: np.zeros((1, 2)),
            )
            self.action_space = None

        def _observation(self) -> dict:
            return {
                "robot": {"position": self.simulator.robot_pos[0].tolist(), "heading": [0.0]},
                "goal": {"current": [3.0, 0.0]},
                "pedestrians": {"positions": self.simulator.ped_pos},
            }

        def reset(self, seed: int | None = None) -> tuple[dict, dict]:
            assert seed == 1001
            return self._observation(), {}

        def step(self, _action: object) -> tuple[dict, float, bool, bool, dict]:
            self.simulator.robot_pos[0] = np.array([0.2, 0.0])
            self.simulator.ped_pos = np.array([[0.3, 0.0]])
            return (
                self._observation(),
                0.0,
                True,
                False,
                {"meta": {"is_pedestrian_collision": True}},
            )

        def close(self) -> None:
            return None

    config = SimpleNamespace(
        sim_config=SimpleNamespace(
            max_sim_steps=600, time_per_step_in_secs=0.1, robot_radius=0.5, ped_radius=0.4
        ),
        robot_config=SimpleNamespace(radius=0.5),
    )
    monkeypatch.setattr(map_runner_episode, "_build_env_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(map_runner_episode, "make_robot_env", lambda **_kwargs: ContactEnv())
    monkeypatch.setattr(map_runner_episode, "sample_obstacle_points", lambda *_args: None)
    monkeypatch.setattr(map_runner_episode, "compute_shortest_path_length", lambda *_args: 3.0)
    monkeypatch.setattr(
        map_runner_episode,
        "compute_all_metrics",
        lambda *_args, **_kwargs: {"success": 0.0, "collisions": 1.0},
    )
    monkeypatch.setattr(
        map_runner_episode,
        "post_process_metrics",
        lambda metrics, **_kwargs: metrics,
    )

    record = map_runner_episode.run_map_episode(
        {
            "name": "contact-marker-1001",
            "collision_attribution_version": "v1",
            "simulation_config": {"max_episode_steps": 1},
        },
        seed=1001,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=Path(__file__),
        policy_builder=lambda *_args, **_kwargs: (lambda _obs: (1.0, 0.0), {"algorithm": "goal"}),
    )
    out_path = tmp_path / "contact_episode.jsonl"
    schema_path = (
        Path(__file__).resolve().parents[2] / "robot_sf/benchmark/schemas/episode.schema.v1.json"
    )
    with out_path.open("w", encoding="utf-8") as handle:
        write_validated_to_handle(handle, json.loads(schema_path.read_text()), record)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["scenario_params"]["collision_attribution_version"] == "v1"
    ledger = saved["event_ledger"]
    assert ledger["contact_provenance"]["schema_version"] == "contact_provenance.v1"
    source = ledger["collision_events"][0]
    contact = ledger["contact_provenance"]["contacts"][0]
    assert source["robot_speed_at_contact_m_s"] == pytest.approx(2.0)
    assert source["contact_step_index"] == 0
    assert contact["robot_speed_at_contact_m_s"] == pytest.approx(2.0)
    assert contact["contact_step_index"] == 0
    assert contact["contact_class"] == "moving_robot_pedestrian_contact"
    assert validate_record_event_ledger(saved) == []


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


def test_saved_speed_cannot_diverge_from_collision_source() -> None:
    """Changing the classified speed alone fails exact-ledger reconciliation."""
    record = _record()
    ledger = build_event_ledger(record, collision_events=[_step_event(np.array([0.8, 0.0]))])
    ledger["contact_provenance"]["contacts"][0]["robot_speed_at_contact_m_s"] = 0.0
    assert any(
        "robot speed differs from saved collision source" in issue
        for issue in reconcile_event_ledger(ledger)
    )
    record["event_ledger"] = ledger
    assert any(
        "opt-in contact provenance differs from saved event source" in issue
        for issue in validate_record_event_ledger(record)
    )


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


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("contact_partner_ids", 3, "saved contact partner IDs must be a list"),
        ("collision_time", "bad", "collision_time must be finite"),
    ],
)
def test_corrupt_saved_event_returns_violations(field: str, value: object, expected: str) -> None:
    """Malformed opt-in rows fail validation without crashing its recomputation."""
    record = _record()
    record["event_ledger"] = build_event_ledger(
        record, collision_events=[_step_event(np.array([1.0, 0.0]))]
    )
    record["event_ledger"]["collision_events"][0][field] = value
    assert any(expected in issue for issue in validate_record_event_ledger(record))
