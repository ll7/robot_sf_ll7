"""Regression coverage for configurable robot control-to-actuation latency."""

from __future__ import annotations

import pytest

from robot_sf.sim.sim_config import SimulationSettings


def test_action_latency_defaults_to_zero_steps_for_backward_compatibility() -> None:
    """The default environment contract keeps immediate action execution."""
    settings = SimulationSettings()

    assert settings.resolved_action_latency_steps == 0
    assert settings.action_latency_metadata() == {
        "configured_steps": 0,
        "configured_ms": None,
        "effective_steps": 0,
        "effective_ms": 0.0,
    }


def test_action_latency_ms_rounds_up_to_an_honest_whole_step_delay() -> None:
    """A millisecond request never understates the delay realizable by a fixed-step loop."""
    settings = SimulationSettings(time_per_step_in_secs=0.1, action_latency_ms=250.0)

    assert settings.resolved_action_latency_steps == 3
    assert settings.action_latency_metadata() == {
        "configured_steps": 0,
        "configured_ms": 250.0,
        "effective_steps": 3,
        "effective_ms": 300.0,
    }


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"action_latency_steps": -1}, "must be >= 0"),
        ({"action_latency_steps": 1.5}, "must be an integer"),
        ({"action_latency_ms": -1.0}, "must be finite and >= 0"),
        (
            {"action_latency_steps": 1, "action_latency_ms": 100.0},
            "cannot both be configured",
        ),
    ],
)
def test_invalid_action_latency_settings_fail_closed(
    kwargs: dict[str, float | int], error: str
) -> None:
    """Ambiguous or invalid delay settings cannot silently change episode semantics."""
    with pytest.raises((TypeError, ValueError), match=error):
        SimulationSettings(**kwargs)
