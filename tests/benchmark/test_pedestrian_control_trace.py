"""Tests for per-pedestrian control-trace logging."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.pedestrian_control_trace import (
    PEDESTRIAN_CONTROL_TRACE_LABELS_KEY,
    PEDESTRIAN_CONTROL_TRACE_SCHEMA,
    attach_pedestrian_control_trace,
    build_generated_population_control_trace_labels,
    build_pedestrian_control_trace,
    has_pedestrian_control_trace_metadata,
)


def _scenario() -> dict[str, object]:
    return {
        "name": "heterogeneous_trace_smoke",
        "single_pedestrians": [
            {"id": "ped_cautious", "metadata": {"archetype": "cautious"}},
            {"id": "ped_hurried", "metadata": {"archetype": "hurried"}},
        ],
    }


def test_build_pedestrian_control_trace_records_archetype_controls() -> None:
    """Recorder emits finite per-pedestrian controls aligned with authored archetypes."""

    positions = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.1, 0.0], [1.0, 0.2]],
        ],
        dtype=float,
    )
    forces = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.3, 0.4], [0.0, 0.5]],
        ],
        dtype=float,
    )

    trace = build_pedestrian_control_trace(
        scenario=_scenario(),
        ped_positions=positions,
        ped_forces=forces,
        dt=0.1,
    )

    assert trace["schema_version"] == PEDESTRIAN_CONTROL_TRACE_SCHEMA
    assert trace["dt"] == pytest.approx(0.1)
    assert trace["pedestrian_count"] == 2
    assert trace["step_count"] == 2
    cautious, hurried = trace["pedestrians"]
    assert cautious["id"] == "ped_cautious"
    assert cautious["archetype"] == "cautious"
    assert cautious["steps"][1]["vx_m_s"] == pytest.approx(1.0)
    assert cautious["steps"][1]["force_norm"] == pytest.approx(0.5)
    assert hurried["id"] == "ped_hurried"
    assert hurried["archetype"] == "hurried"
    assert hurried["steps"][1]["speed_m_s"] == pytest.approx(2.0)


def test_build_pedestrian_control_trace_computes_clearance() -> None:
    """Recorder computes per-step clearance_m when robot positions are supplied."""

    positions = np.array(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ],
        dtype=float,
    )
    robot_positions = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    trace = build_pedestrian_control_trace(
        scenario=_scenario(),
        ped_positions=positions,
        ped_forces=None,
        dt=0.1,
        robot_positions=robot_positions,
        robot_radius=0.3,
        ped_radius=0.2,
    )

    cautious, hurried = trace["pedestrians"]
    assert cautious["steps"][0]["clearance_m"] == pytest.approx(0.5)
    assert cautious["steps"][1]["clearance_m"] == pytest.approx(0.5)
    assert hurried["steps"][0]["clearance_m"] == pytest.approx(0.5)
    assert hurried["steps"][1]["clearance_m"] == pytest.approx(math.sqrt(5) - 0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"robot_positions": np.zeros((1, 2), dtype=float)}, "step count must match"),
        ({"robot_positions": np.zeros((2, 3), dtype=float)}, "must have shape"),
        (
            {"robot_positions": np.zeros((2, 2), dtype=float), "robot_radius": -0.1},
            "must be non-negative",
        ),
        (
            {"robot_positions": np.zeros((2, 2), dtype=float), "ped_radius": -0.1},
            "must be non-negative",
        ),
    ],
)
def test_build_pedestrian_control_trace_rejects_invalid_robot_params(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Invalid robot positions or radii raise ValueError."""

    positions = np.zeros((2, 2, 2), dtype=float)
    params = {
        "scenario": _scenario(),
        "ped_positions": positions,
        "ped_forces": None,
        "dt": 0.1,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_pedestrian_control_trace(**params)


def test_has_pedestrian_control_trace_metadata_detects_archetypes() -> None:
    """Episode integration should only attach the trace for explicitly labeled pedestrians."""

    assert has_pedestrian_control_trace_metadata(_scenario()) is True
    assert (
        has_pedestrian_control_trace_metadata(
            {PEDESTRIAN_CONTROL_TRACE_LABELS_KEY: [{"simulator_index": 0, "archetype": "cautious"}]}
        )
        is True
    )
    assert has_pedestrian_control_trace_metadata({"single_pedestrians": [{"id": "ped"}]}) is False


def test_attach_pedestrian_control_trace_adds_metadata_payload() -> None:
    """Episode integration helper attaches the versioned trace when labels are present."""

    metadata: dict[str, object] = {}
    attach_pedestrian_control_trace(
        metadata,
        scenario=_scenario(),
        ped_positions=np.zeros((1, 2, 2), dtype=float),
        ped_forces=None,
        dt=0.1,
    )

    trace = metadata["pedestrian_control_trace"]
    assert trace["schema_version"] == PEDESTRIAN_CONTROL_TRACE_SCHEMA
    assert trace["pedestrians"][0]["archetype"] == "cautious"
    assert trace["pedestrians"][1]["archetype"] == "hurried"


def test_attach_pedestrian_control_trace_skips_unlabeled_scenarios() -> None:
    """Episode integration helper preserves existing records without archetype metadata."""

    metadata: dict[str, object] = {"existing": True}
    attach_pedestrian_control_trace(
        metadata,
        scenario={"single_pedestrians": [{"id": "ped"}]},
        ped_positions=np.zeros((1, 1, 2), dtype=float),
        ped_forces=None,
        dt=0.1,
    )

    assert metadata == {"existing": True}


def test_generated_population_trace_labels_feed_control_trace() -> None:
    """Generated population records can label simulator-indexed trace rows."""

    labels = build_generated_population_control_trace_labels(
        [
            {"simulator_index": 1, "archetype": "hurried", "desired_speed_factor": 1.4},
            {"simulator_index": 0, "archetype": "cautious", "desired_speed_factor": 0.7},
        ],
        source="unit.generated_population",
    )

    trace = build_pedestrian_control_trace(
        scenario={PEDESTRIAN_CONTROL_TRACE_LABELS_KEY: labels},
        ped_positions=np.zeros((1, 2, 2), dtype=float),
        ped_forces=None,
        dt=0.1,
    )

    assert trace["archetype_source"] == f"scenario.{PEDESTRIAN_CONTROL_TRACE_LABELS_KEY}"
    assert trace["pedestrians"][0]["archetype"] == "cautious"
    assert trace["pedestrians"][0]["desired_speed_factor"] == pytest.approx(0.7)
    assert trace["pedestrians"][0]["source"] == "unit.generated_population"
    assert trace["pedestrians"][1]["archetype"] == "hurried"
    assert trace["pedestrians"][1]["desired_speed_factor"] == pytest.approx(1.4)


def test_generated_population_trace_labels_fail_closed_on_count_mismatch() -> None:
    """Incomplete generated-label feeds report the exact label contract mismatch."""

    labels = build_generated_population_control_trace_labels(
        [{"simulator_index": 0, "archetype": "cautious"}],
        source="unit.generated_population",
    )

    with pytest.raises(ValueError, match="pedestrian_control_trace_labels length"):
        build_pedestrian_control_trace(
            scenario={PEDESTRIAN_CONTROL_TRACE_LABELS_KEY: labels},
            ped_positions=np.zeros((1, 2, 2), dtype=float),
            ped_forces=None,
            dt=0.1,
        )


def test_explicit_empty_generated_population_trace_labels_fail_closed() -> None:
    """An explicit generated-label field is a contract, not an unlabeled scenario."""

    with pytest.raises(ValueError, match="pedestrian_control_trace_labels length"):
        build_pedestrian_control_trace(
            scenario={PEDESTRIAN_CONTROL_TRACE_LABELS_KEY: []},
            ped_positions=np.zeros((1, 1, 2), dtype=float),
            ped_forces=None,
            dt=0.1,
        )


def test_generated_population_trace_labels_accept_zero_pedestrian_trace() -> None:
    """Empty generated-label feeds are valid only when simulator has no pedestrians."""

    trace = build_pedestrian_control_trace(
        scenario={PEDESTRIAN_CONTROL_TRACE_LABELS_KEY: []},
        ped_positions=np.zeros((1, 0, 2), dtype=float),
        ped_forces=None,
        dt=0.1,
    )

    assert trace["pedestrian_count"] == 0
    assert trace["pedestrians"] == []
    # An explicit (empty) label feed must be attributed to the labels key, not to
    # single_pedestrians metadata, even when it yields zero pedestrians.
    assert trace["archetype_source"] == f"scenario.{PEDESTRIAN_CONTROL_TRACE_LABELS_KEY}"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.0}, "dt must be positive"),
        ({"ped_positions": np.zeros((1, 2), dtype=float)}, "ped_positions must have shape"),
        (
            {
                "ped_forces": np.zeros((2, 1, 2), dtype=float),
            },
            "ped_forces must match",
        ),
    ],
)
def test_build_pedestrian_control_trace_rejects_invalid_trace_shape(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Trace shape and timing contracts fail before emitting partial labels."""

    params = {
        "scenario": {"single_pedestrians": [{"metadata": {"archetype": "cautious"}}]},
        "ped_positions": np.zeros((1, 1, 2), dtype=float),
        "ped_forces": None,
        "dt": 0.1,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_pedestrian_control_trace(**params)


def test_build_pedestrian_control_trace_rejects_excess_single_metadata() -> None:
    """Single-pedestrian metadata cannot exceed simulator pedestrian count."""

    with pytest.raises(ValueError, match="metadata exceeds simulator pedestrian count"):
        build_pedestrian_control_trace(
            scenario={
                "single_pedestrians": [
                    {"metadata": {"archetype": "cautious"}},
                    {"metadata": {"archetype": "hurried"}},
                ]
            },
            ped_positions=np.zeros((1, 1, 2), dtype=float),
            ped_forces=None,
            dt=0.1,
        )


@pytest.mark.parametrize(
    ("records", "source", "message"),
    [
        ([{"simulator_index": 0, "archetype": "cautious"}], "", "source must be non-empty"),
        ([object()], "unit", "population_records\\[0\\] must be mapping"),
        ([{"archetype": "cautious"}], "unit", "simulator_index missing"),
        ([{"simulator_index": "bad", "archetype": "cautious"}], "unit", "must be integer"),
        ([{"simulator_index": -1, "archetype": "cautious"}], "unit", "must be >= 0"),
        (
            [
                {"simulator_index": 0, "archetype": "cautious"},
                {"simulator_index": 0, "archetype": "hurried"},
            ],
            "unit",
            "duplicate simulator_index 0",
        ),
        ([{"simulator_index": 0}], "unit", "missing archetype"),
    ],
)
def test_build_generated_population_control_trace_labels_rejects_bad_records(
    records: list[object],
    source: str,
    message: str,
) -> None:
    """Generated label construction fails closed on malformed population records."""

    with pytest.raises(ValueError, match=message):
        build_generated_population_control_trace_labels(records, source=source)


@pytest.mark.parametrize(
    ("labels", "pedestrian_count", "message"),
    [
        ("bad", 1, "must be sequence"),
        ([object()], 1, "pedestrian_control_trace_labels\\[0\\] must be mapping"),
        ([{"archetype": "cautious"}], 1, "simulator_index missing"),
        ([{"simulator_index": "bad", "archetype": "cautious"}], 1, "must be integer"),
        ([{"simulator_index": 2, "archetype": "cautious"}], 1, "out of range"),
        (
            [
                {"simulator_index": 0, "archetype": "cautious"},
                {"simulator_index": 0, "archetype": "hurried"},
            ],
            2,
            "duplicate simulator_index 0",
        ),
    ],
)
def test_explicit_generated_population_trace_labels_reject_malformed_labels(
    labels: object,
    pedestrian_count: int,
    message: str,
) -> None:
    """Scenario-provided generated labels report exact malformed label fields."""

    with pytest.raises(ValueError, match=message):
        build_pedestrian_control_trace(
            scenario={PEDESTRIAN_CONTROL_TRACE_LABELS_KEY: labels},
            ped_positions=np.zeros((1, pedestrian_count, 2), dtype=float),
            ped_forces=None,
            dt=0.1,
        )


@pytest.mark.parametrize("bad_value", [math.nan, math.inf])
def test_build_pedestrian_control_trace_rejects_non_finite_positions(bad_value: float) -> None:
    """Non-finite simulator trace values fail closed instead of leaking to JSON."""

    positions = np.array([[[bad_value, 0.0]]], dtype=float)
    with pytest.raises(ValueError, match="ped_positions"):
        build_pedestrian_control_trace(
            scenario={"single_pedestrians": [{"metadata": {"archetype": "cautious"}}]},
            ped_positions=positions,
            ped_forces=None,
            dt=0.1,
        )


def test_build_pedestrian_control_trace_rejects_missing_archetype() -> None:
    """A control trace without per-pedestrian archetypes cannot feed per-archetype analysis."""

    positions = np.zeros((1, 1, 2), dtype=float)
    with pytest.raises(ValueError, match="archetype metadata"):
        build_pedestrian_control_trace(
            scenario={"single_pedestrians": [{"id": "ped_unlabeled"}]},
            ped_positions=positions,
            ped_forces=None,
            dt=0.1,
        )
