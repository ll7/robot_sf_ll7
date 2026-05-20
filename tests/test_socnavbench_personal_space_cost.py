"""Tests for SocNavBench personal-space velocity handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOCNAV_ROOT = ROOT / "third_party" / "socnavbench"


@pytest.fixture(autouse=True)
def _socnav_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(SOCNAV_ROOT))
    monkeypatch.chdir(SOCNAV_ROOT)


def _trajectory_at(x: float, y: float):
    from trajectory.trajectory import Trajectory

    return Trajectory.from_pos3_array(np.array([[[x, y, 0.0]]], dtype=np.float32))


def _sim_state_for_agent(*, heading: float, speed: float = 0.0):
    from simulators.sim_state import AgentState, SimState
    from trajectory.trajectory import SystemConfig

    agent = AgentState(
        name="ped_0",
        goal_config=None,
        start_config=None,
        current_config=SystemConfig.from_pos3([0.0, 0.0, heading], v=speed),
    )
    return SimState(environment=None, pedestrians={"ped_0": agent}, robots={})


def _personal_space_value(
    *,
    heading: float,
    speed: float = 0.0,
    x: float = 0.0,
    y: float = 1.0,
    velocity_aware: bool = False,
):
    from dotmap import DotMap
    from objectives.personal_space_cost import PersonalSpaceCost

    params = DotMap(
        psc_scale=1.0,
        use_agent_velocity=velocity_aware,
    )
    objective = PersonalSpaceCost(params)
    values = objective.evaluate_objective(
        _trajectory_at(x, y),
        {0: _sim_state_for_agent(heading=heading, speed=speed)},
    )
    return float(values[0, 0])


def test_default_personal_space_uses_heading_derived_unit_vector() -> None:
    """Default mode should ignore stored agent speed."""
    default_value = _personal_space_value(
        heading=0.0,
        speed=2.0,
        velocity_aware=False,
    )
    baseline_value = _personal_space_value(
        heading=0.0,
        speed=0.0,
        velocity_aware=False,
    )

    assert default_value == pytest.approx(baseline_value)


def test_velocity_aware_personal_space_uses_agent_speed() -> None:
    """Enabled mode should scale the personal-space Gaussian by agent speed."""
    from metrics.cost_functions import asym_gauss_from_vel

    disabled_value = _personal_space_value(
        heading=0.0,
        speed=2.0,
        velocity_aware=False,
    )
    enabled_value = _personal_space_value(
        heading=0.0,
        speed=2.0,
        velocity_aware=True,
    )
    expected_enabled_value = float(
        asym_gauss_from_vel(
            x=0.0,
            y=1.0,
            velx=2.0,
            vely=0.0,
            xc=0.0,
            yc=0.0,
        )
    )

    assert enabled_value != pytest.approx(disabled_value)
    assert enabled_value == pytest.approx(expected_enabled_value)


def test_velocity_aware_personal_space_falls_back_for_zero_speed() -> None:
    """Zero-speed agents should use the default heading-derived unit vector."""
    disabled_value = _personal_space_value(
        heading=0.0,
        speed=0.0,
        velocity_aware=False,
    )
    enabled_zero_value = _personal_space_value(
        heading=0.0,
        speed=0.0,
        velocity_aware=True,
    )

    assert enabled_zero_value == pytest.approx(disabled_value)


def test_velocity_aware_personal_space_falls_back_when_speed_unavailable() -> None:
    """Missing speed tensors should preserve default behavior."""
    from dotmap import DotMap
    from objectives.personal_space_cost import PersonalSpaceCost
    from simulators.sim_state import AgentState, SimState
    from trajectory.trajectory import SystemConfig

    agent = AgentState(
        name="ped_0",
        goal_config=None,
        start_config=None,
        current_config=SystemConfig.from_pos3([0.0, 0.0, 0.0]),
    )
    agent.current_config._speed_nk1 = None
    objective = PersonalSpaceCost(DotMap(psc_scale=1.0, use_agent_velocity=True))

    values = objective.evaluate_objective(
        _trajectory_at(0.0, 1.0),
        {0: SimState(environment=None, pedestrians={"ped_0": agent}, robots={})},
    )

    assert float(values[0, 0]) == pytest.approx(
        _personal_space_value(heading=0.0, velocity_aware=False)
    )
