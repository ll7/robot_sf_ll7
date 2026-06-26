"""Tests for per-pedestrian control-trace logging."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.pedestrian_control_trace import (
    PEDESTRIAN_CONTROL_TRACE_SCHEMA,
    attach_pedestrian_control_trace,
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


def test_has_pedestrian_control_trace_metadata_detects_archetypes() -> None:
    """Episode integration should only attach the trace for explicitly labeled pedestrians."""

    assert has_pedestrian_control_trace_metadata(_scenario()) is True
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
