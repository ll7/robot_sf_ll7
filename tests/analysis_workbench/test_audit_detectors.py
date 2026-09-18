"""Deterministic BA-01 detector and registry contracts."""

from __future__ import annotations

import json

import pytest

from robot_sf.analysis_workbench.audit_contracts import (
    SIGNAL_STATUSES,
    record_to_dict,
    validate_record,
)
from robot_sf.analysis_workbench.audit_detectors import (
    ADVISORY_DETECTOR_IDS,
    DETERMINISTIC_DETECTOR_IDS,
    DetectorError,
    DetectorRegistry,
    DetectorSpec,
    default_registry,
    detect,
    normalize_detector_ids,
    registry_document,
)


def _frame(
    index: int,
    position: tuple[float, float] = (0.0, 0.0),
    *,
    heading: float = 0.0,
    angular_velocity: float = 0.0,
    desired: float | None = None,
    executed: float | None = None,
) -> dict[str, object]:
    action: dict[str, float] = {
        "linear_velocity": 0.0,
        "angular_velocity": angular_velocity,
    }
    planner: dict[str, object] = {"selected_action": action}
    if desired is not None:
        planner["desired_action"] = {"linear_velocity": desired}
    if executed is not None:
        planner["executed_action"] = {"linear_velocity": executed}
    return {
        "time_s": float(index),
        "robot": {"position": list(position), "heading": heading},
        "planner": planner,
    }


def _trace(
    positions: list[tuple[float, float]],
    *,
    headings: list[float] | None = None,
    angular: list[float] | None = None,
) -> dict[str, object]:
    headings = headings or [0.0] * len(positions)
    angular = angular or [0.0] * len(positions)
    return {
        "frames": [
            _frame(
                index,
                position,
                heading=headings[index],
                angular_velocity=angular[index],
            )
            for index, position in enumerate(positions)
        ]
    }


def _row(episode_id: str = "episode") -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "planner_id": "planner-a",
        "scenario_id": "scenario-a",
        "config_id": "config-a",
        "seed": 101,
        "provenance": {
            "campaign_id": "campaign-a",
            "source_commit": "a" * 40,
            "config_identity": "config-a",
        },
        "metrics": {"clearance_m": 1.0, "speed_m_s": 0.5, "force_N": 1.0},
        "outcome": {"label": "success"},
        "progress_m": 2.0,
        "duration_s": 2.0,
        "initial_state": {"robot": {"position": [0.0, 0.0], "radius": 0.5}},
        "trace": {"frames": [_frame(0), _frame(1, (2.0, 0.0))]},
    }


def _signal(detector_id: str, row: dict[str, object], **kwargs):
    result = detect(detector_id, row, **kwargs)
    assert result.status in SIGNAL_STATUSES
    payload = record_to_dict(result)
    validate_record(payload)
    json.dumps(payload, allow_nan=False)
    return result


def test_registry_is_versioned_closed_and_deterministic() -> None:
    registry = default_registry()
    assert registry.ids == tuple(sorted(registry.ids))
    assert set(DETERMINISTIC_DETECTOR_IDS) == {
        item.detector_id for item in registry if not item.advisory
    }
    assert set(ADVISORY_DETECTOR_IDS) == {item.detector_id for item in registry if item.advisory}
    assert len(registry) == 14
    assert registry.digest == default_registry().digest
    document = registry_document()
    assert document["schema_version"] == "audit-detector-registry.v1"
    assert all(
        {"required_capabilities", "optional_capabilities", "cohort_definition", "parameters"}
        <= set(item)
        for item in document["detectors"]
    )


def test_registry_rejects_duplicates_and_unstable_order() -> None:
    spec = DetectorSpec("b", "family", "description")
    with pytest.raises(DetectorError):
        DetectorRegistry(detectors=(spec, spec))
    with pytest.raises(DetectorError):
        DetectorRegistry(detectors=(DetectorSpec("z", "family", "description"), spec))


def test_custom_registry_selection_is_supported_and_aliases_are_deduplicated() -> None:
    custom = DetectorRegistry(
        version="audit-detector-registry.custom",
        detectors=(DetectorSpec("custom_detector", "custom", "test"),),
    )
    assert normalize_detector_ids(["custom_detector"], registry=custom) == ("custom_detector",)
    assert normalize_detector_ids(["goal_adjacent_timeout", "completion_goal_geometry"]) == (
        "goal_adjacent_timeout",
    )


@pytest.mark.parametrize(
    ("detector_id", "positive", "normal", "unavailable", "false_positive"),
    [
        (
            "goal_adjacent_timeout",
            lambda: {
                **_row(),
                "outcome": {"timeout_event": True, "collision_event": False},
                "goal": [0.0, 0.0],
                "completion_radius_m": 0.5,
                "trace": _trace([(3.0, 0.0)] * 3),
            },
            lambda: {
                **_row(),
                "outcome": {"timeout_event": True, "collision_event": False},
                "goal": [0.0, 0.0],
                "completion_radius_m": 0.5,
                "trace": _trace([(10.0, 0.0)] * 3),
            },
            lambda: {**_row(), "outcome": {"timeout_event": True}},
            lambda: {
                **_row(),
                "outcome": {"label": "success"},
                "goal": [0.0, 0.0],
                "completion_radius_m": 0.5,
                "trace": _trace([(0.1, 0.0)] * 3),
            },
        ),
        (
            "initial_reset_anomaly",
            lambda: {
                **_row(),
                "initial_state": {
                    "robot": {"position": [0.0, 0.0], "radius": 0.5},
                    "agents": [{"position": [0.5, 0.0], "radius": 0.5}],
                },
            },
            lambda: {
                **_row(),
                "initial_state": {
                    "robot": {"position": [0.0, 0.0], "radius": 0.5},
                    "agents": [{"position": [5.0, 0.0], "radius": 0.5}],
                },
            },
            lambda: {**_row(), "requires_initial_state": True, "initial_state": None},
            lambda: {
                **_row(),
                "initial_state": {
                    "robot": {"position": [0.0, 0.0], "radius": 0.2},
                    "agents": [{"position": [5.0, 0.0], "radius": 0.2}],
                },
            },
        ),
        (
            "stuck_no_progress",
            lambda: {**_row(), "progress_m": 0.0, "duration_s": 3.0, "mean_speed_m_s": 0.0},
            lambda: {**_row(), "progress_m": 1.0, "duration_s": 3.0},
            lambda: _row_without("progress_m", "duration_s", "trace"),
            lambda: {
                **_row(),
                "progress_m": 0.0,
                "duration_s": 3.0,
                "expected_waiting": True,
            },
        ),
        (
            "oscillation_limit_cycle",
            lambda: {
                **_row(),
                "trace": _trace(
                    [(0.0, 0.0)] * 6,
                    angular=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                ),
            },
            lambda: {
                **_row(),
                "trace": _trace(
                    [(float(index), 0.0) for index in range(6)],
                    angular=[0.2] * 6,
                ),
            },
            lambda: {"episode_id": "episode"},
            lambda: {
                **_row(),
                "trace": _trace(
                    [(float(index) * 2.0, 0.0) for index in range(6)],
                    angular=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                ),
            },
        ),
        (
            "outcome_metric_contradiction",
            lambda: {
                **_row(),
                "outcome": {"success": True, "timeout": True},
                "termination_contract": {
                    "schema_version": "termination-contract.v1",
                    "mutually_exclusive": True,
                },
            },
            lambda: {
                **_row(),
                "outcome": {"success": True, "collision": False, "timeout": False},
                "termination_contract": {
                    "schema_version": "termination-contract.v1",
                    "mutually_exclusive": True,
                },
            },
            lambda: {"episode_id": "episode"},
            lambda: {
                **_row(),
                "outcome": {"success": True, "collision": True},
                "termination_contract": {
                    "schema_version": "termination-contract.v1",
                    "success_and_collision_valid": True,
                },
            },
        ),
        (
            "extreme_measurements",
            lambda: {**_row(), "metrics": {"clearance_m": -0.1}},
            lambda: {**_row(), "metrics": {"clearance_m": 1.0, "ttc_s": None}},
            lambda: _row_without("metrics"),
            lambda: {**_row(), "metrics": {"ttc_s": None, "pet_s": None, "clearance_m": 1.0}},
        ),
        (
            "actuator_mismatch",
            lambda: {
                **_row(),
                "commands": [
                    {"desired": {"linear_velocity": 1.0}, "executed": {"linear_velocity": 0.0}}
                ],
            },
            lambda: {
                **_row(),
                "commands": [
                    {"desired": {"linear_velocity": 0.5}, "executed": {"linear_velocity": 0.5}}
                ],
            },
            lambda: _row_without("commands", "trace"),
            lambda: {
                **_row(),
                "commands": [
                    {
                        "desired": {"linear_velocity": 0.5},
                        "executed": {"linear_velocity": 0.5000001},
                    }
                ],
            },
        ),
        (
            "telemetry_integrity",
            lambda: {
                **_row(),
                "telemetry_contract": {"required_fields": ["missing_telemetry"]},
            },
            lambda: {**_row(), "telemetry_contract": {"required_fields": ["trace.frames"]}},
            lambda: {"episode_id": "episode"},
            _row,
        ),
        (
            "provenance_consistency",
            lambda: {**_row(), "requires_provenance": True, "provenance": {}},
            _row,
            lambda: {"episode_id": "episode"},
            lambda: {
                **_row(),
                "provenance": {},
            },
        ),
        (
            "common_mode_anomaly",
            lambda: {**_row(), "outcome": {"timeout": True}},
            lambda: {**_row(), "outcome": {"success": True}},
            _row,
            lambda: {**_row(), "config_id": "different", "outcome": {"timeout": True}},
        ),
        (
            "planner_disagreement",
            lambda: {**_row(), "outcome": {"label": "success"}},
            lambda: {**_row(), "outcome": {"label": "success"}},
            _row,
            lambda: {**_row(), "scenario_id": "different", "outcome": {"label": "timeout"}},
        ),
        (
            "seed_outlier",
            lambda: {**_row(), "metrics": {"loss": 10.0}},
            lambda: {**_row(), "metrics": {"loss": 1.0}},
            _row,
            lambda: {**_row(), "seed": 102, "metrics": {"loss": 1.0}},
        ),
    ],
)
def test_deterministic_detector_family_contracts(
    detector_id: str,
    positive,
    normal,
    unavailable,
    false_positive,
) -> None:
    positive_row = positive()
    normal_row = normal()
    unavailable_row = unavailable()
    false_positive_row = false_positive()
    cohort = [normal_row, positive_row]
    kwargs = {"cohort": cohort}
    if detector_id in {"goal_adjacent_timeout", "oscillation_limit_cycle"}:
        kwargs["config"] = {
            "tail_steps": 3,
            "minimum_reversals": 3,
        }
    if detector_id == "common_mode_anomaly":
        positive_row["outcome"] = {"timeout": True}
        cohort = [
            {**positive_row, "planner_id": "planner-a"},
            {**positive_row, "episode_id": "peer", "planner_id": "planner-b"},
        ]
        kwargs["cohort"] = cohort
        normal_row["outcome"] = {"success": True}
        normal_cohort = [
            {**normal_row, "planner_id": "planner-a"},
            {**normal_row, "episode_id": "peer", "planner_id": "planner-b"},
        ]
        kwargs["cohort"] = cohort
    elif detector_id == "planner_disagreement":
        cohort = [
            positive_row,
            {
                **positive_row,
                "episode_id": "peer",
                "planner_id": "planner-b",
                "outcome": {"label": "timeout"},
            },
        ]
        normal_cohort = [
            normal_row,
            {**normal_row, "episode_id": "peer", "planner_id": "planner-b"},
        ]
        kwargs["cohort"] = cohort
    elif detector_id == "seed_outlier":
        cohort = [
            positive_row,
            *[
                {
                    **positive_row,
                    "episode_id": f"peer-{index}",
                    "seed": index,
                    "metrics": {"loss": 1.0},
                }
                for index in range(3)
            ],
        ]
        normal_cohort = [
            normal_row,
            *[
                {
                    **normal_row,
                    "episode_id": f"peer-{index}",
                    "seed": index,
                }
                for index in range(3)
            ],
        ]
        kwargs["cohort"] = cohort
    else:
        kwargs["cohort"] = cohort
        normal_cohort = cohort
    positive_kwargs = kwargs
    normal_kwargs = {**kwargs, "cohort": normal_cohort}
    flagged = _signal(detector_id, positive_row, **positive_kwargs)
    assert flagged.status == "flagged", detector_id
    normal_signal = _signal(detector_id, normal_row, **normal_kwargs)
    assert normal_signal.status != "flagged", detector_id
    unavailable_kwargs = {**normal_kwargs, "cohort": [unavailable_row]}
    unavailable_signal = _signal(detector_id, unavailable_row, **unavailable_kwargs)
    assert unavailable_signal.status in {"unavailable", "error"}, detector_id
    false_signal = _signal(detector_id, false_positive_row, **normal_kwargs)
    assert false_signal.status != "flagged", detector_id


def _row_without(*keys: str) -> dict[str, object]:
    value = _row()
    for key in keys:
        value.pop(key, None)
    return value


def test_cross_planner_and_cohort_detectors_use_compatible_keys() -> None:
    target = {**_row("target"), "outcome": {"label": "success"}, "metrics": {"loss": 10.0}}
    peer = {
        **_row("peer"),
        "planner_id": "planner-b",
        "outcome": {"timeout": True},
        "metrics": {"loss": 1.0},
    }
    common = _signal(
        "common_mode_anomaly",
        {**target, "outcome": {"timeout": True}},
        cohort=[{**target, "outcome": {"timeout": True}}, peer],
    )
    assert common.status == "flagged"
    disagreement = _signal("planner_disagreement", target, cohort=[target, peer])
    assert disagreement.status == "flagged"
    outlier = _signal(
        "seed_outlier",
        target,
        cohort=[target]
        + [
            {
                **peer,
                "planner_id": "planner-a",
                "episode_id": f"peer-{index}",
                "seed": index,
                "metrics": {"loss": 1.0},
            }
            for index in range(3)
        ],
    )
    assert outlier.status == "flagged"
    assert outlier.measured["seed_values_are_keys_only"] is True


def test_planner_disagreement_normalizes_boolean_outcomes() -> None:
    target = {**_row("target"), "outcome": {"success": True}}
    peer = {
        **_row("peer"),
        "planner_id": "planner-b",
        "outcome": {"timeout": True},
    }
    signal = _signal("planner_disagreement", target, cohort=[target, peer])
    assert signal.status == "flagged"
    assert signal.measured["outcome"] == "success"
    assert signal.measured["peer_outcomes"] == ["timeout"]


def test_extreme_detector_accepts_operational_metric_sidecar() -> None:
    row = {
        **_row(),
        "metrics": {},
        "operational_metrics": {"minimum_clearance": -0.1},
    }
    signal = _signal("extreme_measurements", row)
    assert signal.status == "flagged"


def test_advisory_outliers_disclose_features_scaling_and_small_cohort() -> None:
    peers = [
        {
            **_row(f"peer-{index}"),
            "metrics": {"loss": 1.0},
            "trace": _trace([(0.0, 0.0), (1.0, 0.0)]),
        }
        for index in range(3)
    ]
    target = {
        **_row("target"),
        "metrics": {"loss": 10.0},
        "trace": _trace([(0.0, 0.0), (10.0, 0.0)]),
    }
    cohort = peers + [target]
    for detector_id in ADVISORY_DETECTOR_IDS:
        signal = _signal(detector_id, target, cohort=cohort)
        assert signal.status == "flagged", detector_id
        assert signal.measured["priority_score_uncalibrated"] >= 0
        assert signal.evidence[-1]["scaling"]
        assert "cohort" in signal.evidence[-1]
        assert "not probability" in json.dumps(signal.evidence[-1])
        small = _signal(detector_id, target, cohort=[target])
        assert small.status == "unavailable"


def test_goal_adjacent_timeout_is_not_a_geometrical_impossibility_claim() -> None:
    row = {
        **_row(),
        "outcome": {"timeout_event": True, "collision_event": False},
        "goal": [0.0, 0.0],
        "completion_radius_m": 0.5,
        "trace": _trace([(3.0, 0.0)] * 3),
    }
    signal = _signal("goal_adjacent_timeout", row, config={"tail_steps": 3})
    assert signal.status == "flagged"
    assert signal.measured["infeasibility_claimed"] is False
    assert "not a geometrical impossibility claim" in json.dumps(signal.evidence)


def test_goal_signal_retains_active_and_visible_goal_context_without_reinterpreting_it() -> None:
    row = {
        **_row(),
        "outcome": {"timeout_event": True, "collision_event": False},
        "goal": [0.0, 0.0],
        "visible_goal": {"position": [0.1, 0.0]},
        "active_waypoint": {"position": [1.0, 0.0]},
        "completion_radius_m": 0.5,
        "trace": _trace([(3.0, 0.0)] * 3),
    }
    signal = _signal("goal_adjacent_timeout", row, config={"tail_steps": 3})
    assert signal.status == "flagged"
    assert signal.measured["active_waypoint_xy_m"] == [1.0, 0.0]
    assert signal.measured["visible_goal_xy_m"] == [0.1, 0.0]


def test_malformed_top_level_command_stream_is_an_error() -> None:
    signal = _signal(
        "actuator_mismatch",
        {**_row(), "desired_action": {"linear_velocity": 1.0}, "executed_action": 0.0},
    )
    assert signal.status == "error"
    assert signal.reason_code == "executed_action_malformed"


def test_nonfinite_promised_telemetry_is_an_error_and_not_a_clear_signal() -> None:
    row = {
        **_row(),
        "telemetry_contract": {"required_fields": ["telemetry.value"]},
        "telemetry": {"value": float("nan")},
    }
    signal = _signal("telemetry_integrity", row)
    assert signal.status == "error"
    assert signal.reason_code == "promised_telemetry_nonfinite"


@pytest.mark.parametrize(
    "field",
    [
        "row_status",
        "status",
        "availability_status",
        "readiness_status",
        "campaign_execution_status",
    ],
)
def test_direct_detector_calls_fail_closed_for_non_native_status(field: str) -> None:
    signal = _signal("extreme_measurements", {**_row(), field: "degraded"})
    assert signal.status == "unavailable"
    assert "non_admissible_execution_status" in signal.reason_code


def test_direct_detector_ignores_descriptive_terminal_status_with_native_row_status() -> None:
    signal = _signal(
        "extreme_measurements",
        {**_row(), "row_status": "native", "status": "collision"},
    )
    assert signal.status in {"clear", "flagged"}


def test_canonical_analysis_trace_controls_feed_actuator_and_turn_detectors() -> None:
    frames = []
    for index, turn_rate in enumerate((1.0, -1.0, 1.0, -1.0, 1.0, -1.0)):
        frames.append(
            {
                "step": index,
                "time_s": float(index),
                "robot": {"position": [0.0, 0.0], "heading": 0.0, "radius_m": 0.25},
                "pedestrians": [],
                "controls": {
                    "requested": {"linear_m_s": 0.2, "turn_rate_rad_s": turn_rate},
                    "applied": {"linear_m_s": 0.2, "turn_rate_rad_s": turn_rate},
                },
            }
        )
    row = {
        "episode_id": "canonical-controls",
        "algorithm_metadata": {
            "analysis_trace": {"schema_version": "analysis-trace.v1", "steps": frames}
        },
    }
    actuator = _signal("actuator_mismatch", row)
    oscillation = _signal("oscillation_limit_cycle", row, config={"minimum_reversals": 3})
    assert actuator.status == "clear"
    assert oscillation.status == "flagged"


def test_outcome_contradiction_requires_typed_exclusivity_contract() -> None:
    row = {"episode_id": "outcome", "outcome": {"success": True, "timeout": True}}
    assert _signal("outcome_metric_contradiction", row).status == "unavailable"
    typed = {
        **row,
        "termination_contract": {
            "schema_version": "termination-contract.v1",
            "mutually_exclusive": True,
        },
    }
    assert _signal("outcome_metric_contradiction", typed).status == "flagged"
    malformed = {
        **typed,
        "termination_contract": {
            "schema_version": "termination-contract.v1",
            "mutually_exclusive": "yes",
        },
    }
    malformed_signal = _signal("outcome_metric_contradiction", malformed)
    assert malformed_signal.status == "error"


def test_initial_geometry_uses_radius_m_and_retains_threshold_margin() -> None:
    overlap = {
        "episode_id": "geometry",
        "initial_state": {
            "robot": {"position": [0.0, 0.0], "radius_m": 0.5},
            "pedestrians": [{"position": [0.75, 0.0], "radius_m": 0.25}],
        },
    }
    signal = _signal("initial_reset_anomaly", overlap, config={"minimum_separation_m": 0.1})
    assert signal.status == "flagged"
    assert signal.measured["minimum_separation_m"] == 0.1
    assert signal.threshold["minimum_separation_m"] == 0.1
    missing_radius = {
        **overlap,
        "initial_state": {
            "robot": {"position": [0.0, 0.0]},
            "pedestrians": [],
        },
    }
    assert _signal("initial_reset_anomaly", missing_radius).status == "unavailable"
    incomplete_actor = {
        **overlap,
        "initial_state": {
            "robot": {"position": [0.0, 0.0], "radius_m": 0.5},
            "pedestrians": [{"position": [0.75, 0.0]}],
        },
    }
    assert _signal("initial_reset_anomaly", incomplete_actor).status == "unavailable"


def test_path_length_is_not_substituted_for_route_progress() -> None:
    signal = _signal(
        "stuck_no_progress",
        {"episode_id": "path-length", "path_length": 0.0, "duration_s": 10.0},
    )
    assert signal.status == "unavailable"
    assert signal.status != "flagged"


def test_empty_and_truncated_trace_cannot_be_reported_clear() -> None:
    empty = _signal(
        "telemetry_integrity",
        {"episode_id": "empty", "trace": {"frames": []}},
    )
    assert empty.status == "unavailable"
    truncated = _signal(
        "telemetry_integrity",
        {
            "episode_id": "truncated",
            "trace_coverage": {"status": "unavailable"},
            "trace": {"frames": [{"time_s": 0.0, "robot": {"position": [0.0, 0.0]}}]},
        },
    )
    assert truncated.status == "unavailable"


def test_detector_rejects_non_mapping_rows() -> None:
    with pytest.raises(DetectorError):
        detect("stuck_no_progress", [])
