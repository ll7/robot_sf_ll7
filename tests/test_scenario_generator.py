"""Tests for the deterministic scenario generator.

Focus areas:
 - Determinism: identical params + seed -> identical outputs.
 - Density mapping: low/med/high -> expected agent counts.
 - Obstacles: layout sizes per obstacle kind.
 - Flow + goal_topology: swap reverses ordering; circulate rotates.
 - Group assignment fraction honored.
 - Speed variance metadata selection.
"""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.scenario_generator import generate_scenario

BASE_PARAMS = {
    "id": "TEST",
    "density": "low",
    "flow": "uni",
    "obstacle": "open",
    "groups": 0.0,
    "speed_var": "low",
    "goal_topology": "point",
    "robot_context": "embedded",
    "repeats": 1,
}


def test_determinism_same_seed():
    p = {**BASE_PARAMS}
    a = generate_scenario(p, seed=123)
    b = generate_scenario(p, seed=123)
    np.testing.assert_allclose(a.state, b.state)
    assert a.obstacles == b.obstacles
    assert a.groups == b.groups
    assert a.metadata["n_agents"] == b.metadata["n_agents"]


def test_variation_different_seed():
    p = {**BASE_PARAMS}
    a = generate_scenario(p, seed=1)
    c = generate_scenario(p, seed=2)
    # Position rows should differ for at least one agent
    assert not np.allclose(a.state[:, :2], c.state[:, :2])


def test_density_counts():
    for density, expected in {"low": 10, "med": 25, "high": 40}.items():
        p = {**BASE_PARAMS, "density": density}
        scen = generate_scenario(p, seed=42)
        assert scen.state.shape[0] == expected, (
            f"density {density} -> {scen.state.shape[0]} != {expected}"
        )


def test_obstacle_layouts():
    kinds_expected = {"open": 0, "bottleneck": 2, "maze": 3}
    for kind, exp in kinds_expected.items():
        p = {**BASE_PARAMS, "obstacle": kind}
        scen = generate_scenario(p, seed=99)
        assert len(scen.obstacles) == exp, f"obstacle {kind} length {len(scen.obstacles)} != {exp}"


def test_goal_topology_swap_and_circulate():
    # swap reverses ordering of positions for goals
    p_swap = {**BASE_PARAMS, "flow": "uni", "goal_topology": "swap"}
    scen_swap = generate_scenario(p_swap, seed=5)
    positions = scen_swap.state[:, :2]
    goals = scen_swap.state[:, 4:6]
    np.testing.assert_allclose(goals, positions[::-1])

    p_circ = {**BASE_PARAMS, "flow": "uni", "goal_topology": "circulate"}
    scen_circ = generate_scenario(p_circ, seed=5)
    positions_c = scen_circ.state[:, :2]
    goals_c = scen_circ.state[:, 4:6]
    # circulate: roll by 1
    np.testing.assert_allclose(goals_c, np.roll(positions_c, shift=1, axis=0))


def test_group_assignment_fraction():
    # Use high density for better sample size
    p = {**BASE_PARAMS, "density": "high", "groups": 0.4}
    scen = generate_scenario(p, seed=7)
    grouped = sum(g >= 0 for g in scen.groups)
    expected = int(round(40 * 0.4))
    assert grouped == expected, f"grouped {grouped} != {expected}"


def test_speed_variance_metadata():
    p_low = {**BASE_PARAMS, "speed_var": "low"}
    scen_low = generate_scenario(p_low, seed=11)
    assert np.isclose(scen_low.metadata["speed_std"], 0.2)
    p_high = {**BASE_PARAMS, "speed_var": "high"}
    scen_high = generate_scenario(p_high, seed=11)
    assert np.isclose(scen_high.metadata["speed_std"], 0.5)
