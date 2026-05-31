"""Tests for the diagnostic adaptive proxemic-profile selector."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.adaptive_proxemic_selector import (
    AdaptiveProxemicSelectorAdapter,
    build_adaptive_proxemic_selector_config,
)


def _obs(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    goal: tuple[float, float] = (2.0, 0.0),
    ped_positions: list[tuple[float, float]] | None = None,
) -> dict:
    """Build a compact structured observation for selector tests."""
    ped_positions = [] if ped_positions is None else ped_positions
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.zeros((len(ped_positions), 2), dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": 0.25,
        },
        "sim": {"timestep": 0.1},
    }


def test_adaptive_selector_uses_open_profile_when_local_context_is_clear() -> None:
    """Clear low-density context should select the open fixed profile."""
    planner = AdaptiveProxemicSelectorAdapter(build_adaptive_proxemic_selector_config({}))

    linear, angular = planner.plan(_obs())

    assert isinstance(linear, float)
    assert isinstance(angular, float)
    diagnostics = planner.diagnostics()
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["selected_profile_counts"] == {"open": 1}
    assert diagnostics["last_selection"]["selected_profile"] == "open"
    assert diagnostics["last_selection"]["trigger_reason"] == "clear_low_density"
    assert diagnostics["last_selection"]["source_candidate"] == "proxemic_profile_open_issue_1676"


def test_adaptive_selector_uses_conservative_profile_near_humans() -> None:
    """Immediate human proximity should select the conservative fixed profile."""
    planner = AdaptiveProxemicSelectorAdapter(build_adaptive_proxemic_selector_config({}))

    planner.plan(_obs(ped_positions=[(0.55, 0.0)]))

    diagnostics = planner.diagnostics()
    assert diagnostics["selected_profile_counts"] == {"conservative": 1}
    assert diagnostics["last_selection"]["selected_profile"] == "conservative"
    assert diagnostics["last_selection"]["trigger_reason"] == "near_human"
    assert (
        diagnostics["last_selection"]["source_candidate"]
        == "proxemic_profile_conservative_issue_1676"
    )


def test_adaptive_selector_exposes_density_and_low_progress_reasons() -> None:
    """Density and low-progress signals should be visible in diagnostic reason counts."""
    planner = AdaptiveProxemicSelectorAdapter(build_adaptive_proxemic_selector_config({}))

    planner.plan(_obs(ped_positions=[(1.1, 0.0), (1.2, 0.2), (1.3, -0.2)]))
    low_progress_obs = _obs(ped_positions=[(2.8, 0.0)])
    low_progress_obs["route_arc_progress_windows"] = {"3s": 0.0}
    planner.plan(low_progress_obs)

    diagnostics = planner.diagnostics()
    assert diagnostics["selected_profile_counts"] == {"conservative": 1, "open": 1}
    assert diagnostics["trigger_reason_counts"] == {
        "high_local_density": 1,
        "low_progress_clear_space": 1,
    }
    assert diagnostics["last_selection"]["selected_profile"] == "open"
    assert diagnostics["last_selection"]["trigger_reason"] == "low_progress_clear_space"


def test_adaptive_selector_uses_neutral_profile_for_constrained_passage() -> None:
    """A narrow passage signal should choose the neutral profile, not a new parameterization."""
    planner = AdaptiveProxemicSelectorAdapter(build_adaptive_proxemic_selector_config({}))
    obs = _obs(ped_positions=[(2.0, 0.0)])
    obs["route_corridor"] = {"corridor_width_estimate": 0.85}

    planner.plan(obs)

    diagnostics = planner.diagnostics()
    assert diagnostics["selected_profile_counts"] == {"neutral": 1}
    assert diagnostics["last_selection"]["selected_profile"] == "neutral"
    assert diagnostics["last_selection"]["trigger_reason"] == "constrained_passage"
    assert (
        diagnostics["last_selection"]["source_candidate"] == "proxemic_profile_neutral_issue_1676"
    )


def test_adaptive_selector_ignores_nonfinite_pedestrian_count() -> None:
    """Nonfinite pedestrian counts should fall back to the available rows."""
    planner = AdaptiveProxemicSelectorAdapter(build_adaptive_proxemic_selector_config({}))
    obs = _obs(ped_positions=[(0.55, 0.0)])
    obs["pedestrians"]["count"] = np.asarray([float("nan")], dtype=float)

    planner.plan(obs)

    diagnostics = planner.diagnostics()
    assert diagnostics["last_selection"]["selected_profile"] == "conservative"
    assert diagnostics["last_selection"]["trigger_reason"] == "near_human"


def test_adaptive_selector_requires_positive_constrained_width() -> None:
    """Zero or negative corridor widths are invalid signals, not constrained passages."""
    planner = AdaptiveProxemicSelectorAdapter(build_adaptive_proxemic_selector_config({}))
    obs = _obs(ped_positions=[(2.0, 0.0)])
    obs["route_corridor"] = {"corridor_width_estimate": 0.0}

    planner.plan(obs)

    diagnostics = planner.diagnostics()
    assert diagnostics["last_selection"]["trigger_reason"] == "clear_low_density"


def test_adaptive_selector_defaults_blank_claim_boundary() -> None:
    """Blank registry values should keep the diagnostic-only claim boundary."""
    config = build_adaptive_proxemic_selector_config({"claim_boundary": ""})

    assert config.claim_boundary == "diagnostic_only"


def test_adaptive_selector_rejects_non_diagnostic_claim_boundary() -> None:
    """The selector is diagnostic-only until a future issue promotes it."""
    with pytest.raises(ValueError, match="diagnostic-only"):
        build_adaptive_proxemic_selector_config({"claim_boundary": "benchmark_candidate"})
