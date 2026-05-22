"""Tests for the deterministic scenario generator.

Focus areas:
 - Determinism: identical params + seed -> identical outputs.
 - Density mapping: low/med/high -> expected agent counts.
 - Obstacles: layout sizes per obstacle kind.
 - Flow + goal_topology: swap reverses ordering; circulate rotates.
 - Group assignment fraction honored.
 - Speed variance metadata selection.
 - Property-style structural invariants across parametrized inputs (no Hypothesis dep;
   uses pytest.mark.parametrize with deterministic generated cases).
  - Metamorphic tests with explicitly documented invariants.

Scope note: These tests cover valid parameter combinations only. Invalid
parameter values (e.g., unknown density strings, negative group fractions)
are outside the current test scope; the generator handles missing keys via
defaults but does not perform input sanitization.

Dependency decision (issue #1435): Hypothesis is not in pyproject.toml.
We use pytest.mark.parametrize with deterministic generated parameter
combinations instead of adding a Hypothesis dev-dependency. This keeps
test dependencies minimal while still covering generated-input invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.scenario_generator import (
    AREA_HEIGHT,
    AREA_WIDTH,
    GeneratedScenario,
    generate_scenario,
)

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
    """TODO docstring. Document this function."""
    p = {**BASE_PARAMS}
    a = generate_scenario(p, seed=123)
    b = generate_scenario(p, seed=123)
    np.testing.assert_allclose(a.state, b.state)
    assert a.obstacles == b.obstacles
    assert a.groups == b.groups
    assert a.metadata["n_agents"] == b.metadata["n_agents"]


def test_variation_different_seed():
    """TODO docstring. Document this function."""
    p = {**BASE_PARAMS}
    a = generate_scenario(p, seed=1)
    c = generate_scenario(p, seed=2)
    # Position rows should differ for at least one agent
    assert not np.allclose(a.state[:, :2], c.state[:, :2])


def test_density_counts():
    """TODO docstring. Document this function."""
    for density, expected in {"low": 10, "med": 25, "high": 40}.items():
        p = {**BASE_PARAMS, "density": density}
        scen = generate_scenario(p, seed=42)
        assert scen.state.shape[0] == expected, (
            f"density {density} -> {scen.state.shape[0]} != {expected}"
        )


def test_obstacle_layouts():
    """TODO docstring. Document this function."""
    kinds_expected = {"open": 0, "bottleneck": 2, "maze": 3}
    for kind, exp in kinds_expected.items():
        p = {**BASE_PARAMS, "obstacle": kind}
        scen = generate_scenario(p, seed=99)
        assert len(scen.obstacles) == exp, f"obstacle {kind} length {len(scen.obstacles)} != {exp}"


def test_goal_topology_swap_and_circulate():
    # swap reverses ordering of positions for goals
    """TODO docstring. Document this function."""
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
    """TODO docstring. Document this function."""
    p = {**BASE_PARAMS, "density": "high", "groups": 0.4}
    scen = generate_scenario(p, seed=7)
    grouped = sum(g >= 0 for g in scen.groups)
    expected = round(40 * 0.4)
    assert grouped == expected, f"grouped {grouped} != {expected}"


def test_speed_variance_metadata():
    """TODO docstring. Document this function."""
    p_low = {**BASE_PARAMS, "speed_var": "low"}
    scen_low = generate_scenario(p_low, seed=11)
    assert np.isclose(scen_low.metadata["speed_std"], 0.2)
    p_high = {**BASE_PARAMS, "speed_var": "high"}
    scen_high = generate_scenario(p_high, seed=11)
    assert np.isclose(scen_high.metadata["speed_std"], 0.5)


def test_flow_goal_geometries():
    # bi-directional
    """TODO docstring. Document this function."""
    p_bi = {**BASE_PARAMS, "flow": "bi", "goal_topology": "point", "density": "med"}
    scen_bi = generate_scenario(p_bi, seed=21)
    pos_bi = scen_bi.state[:, 0:2]
    goals_bi = scen_bi.state[:, 4:6]
    n = pos_bi.shape[0]
    half = n // 2
    # First half rightward
    assert np.allclose(goals_bi[:half, 0], AREA_WIDTH - 0.2)
    assert np.allclose(goals_bi[:half, 1], pos_bi[:half, 1])
    # Second half leftward
    assert np.allclose(goals_bi[half:, 0], 0.2)
    assert np.allclose(goals_bi[half:, 1], pos_bi[half:, 1])

    # cross flow
    p_cross = {**BASE_PARAMS, "flow": "cross", "goal_topology": "point", "density": "med"}
    scen_cross = generate_scenario(p_cross, seed=22)
    pos_c = scen_cross.state[:, 0:2]
    goals_c = scen_cross.state[:, 4:6]
    n_c = pos_c.shape[0]
    half_c = n_c // 2
    # Horizontal movers
    np.testing.assert_allclose(goals_c[:half_c, 0], AREA_WIDTH - pos_c[:half_c, 0])
    np.testing.assert_allclose(goals_c[:half_c, 1], pos_c[:half_c, 1])
    # Vertical movers
    np.testing.assert_allclose(goals_c[half_c:, 0], pos_c[half_c:, 0])
    np.testing.assert_allclose(goals_c[half_c:, 1], AREA_HEIGHT - pos_c[half_c:, 1])

    # merge flow (all share central goal point)
    p_merge = {**BASE_PARAMS, "flow": "merge", "goal_topology": "point", "density": "med"}
    scen_merge = generate_scenario(p_merge, seed=23)
    goals_m = scen_merge.state[:, 4:6]
    cx = AREA_WIDTH * 0.6
    cy = AREA_HEIGHT / 2
    assert np.allclose(goals_m[:, 0], cx)
    assert np.allclose(goals_m[:, 1], cy)


# ---------------------------------------------------------------------------
# Property-style deterministic generated test (pytest parametrization).
# No Hypothesis dependency needed -- we enumerate the valid parameter space.
# ---------------------------------------------------------------------------

# Valid parameter values for cross-product generation.
# Excludes robot_context and repeats (trivially stored, not used by generator).
_DENSITIES = ["low", "med", "high"]
_FLOWS = ["uni", "bi", "cross", "merge"]
_OBSTACLES = ["open", "bottleneck", "maze"]
_SPEED_VARS = ["low", "high"]
_GOAL_TOPOLOGIES = ["point", "swap", "circulate"]


def _gen_param_combos():
    """Yield (density, flow, obstacle, speed_var, goal_topology) combos.

    We generate the full cross-product (3x4x3x2x3 = 216) deliberately:
    the generator is pure Python + numpy and each invocation is < 1 ms.
    """
    for d in _DENSITIES:
        for f in _FLOWS:
            for o in _OBSTACLES:
                for s in _SPEED_VARS:
                    for g in _GOAL_TOPOLOGIES:
                        yield pytest.param(d, f, o, s, g, id=f"{d}_{f}_{o}_{s}_{g}")


_DENSITY_EXPECTED = {"low": 10, "med": 25, "high": 40}
_OBS_EXPECTED = {"open": 0, "bottleneck": 2, "maze": 3}


@pytest.mark.parametrize("density,flow,obstacle,speed_var,goal_topology", _gen_param_combos())
def test_scenario_structural_invariants(density, flow, obstacle, speed_var, goal_topology):
    """Property: every valid parameter combination produces a structurally valid scenario.

    Invariants checked (all pure, deterministic, no simulator stochasticity):
      - State shape is (n_agents, 7) where n_agents matches the density table.
      - Positions are within the arena bounding box [0.5, 9.5] x [0.5, 5.5].
      - Initial velocities (cols 2,3) are zero; tau (col 6) is 1.0.
      - Groups list length equals n_agents; elements are ints (-1 or >=0).
      - Metadata contains selected original param keys (density, flow,
        obstacle, speed_var, goal_topology) that influence generation,
        plus derived fields (n_agents, area, seed, speed_std).
      - Obstacle count matches the expected layout for the obstacle kind.
      - Obstacle segments are within the arena [0, 10] x [0, 6].
    """
    seed = 42
    params = {
        "id": f"prop_{density}_{flow}_{obstacle}_{speed_var}_{goal_topology}",
        "density": density,
        "flow": flow,
        "obstacle": obstacle,
        "groups": 0.0,
        "speed_var": speed_var,
        "goal_topology": goal_topology,
        "robot_context": "embedded",
        "repeats": 1,
    }
    scen = generate_scenario(params, seed=seed)

    expected_n = _DENSITY_EXPECTED[density]
    state = scen.state
    assert state.shape == (expected_n, 7), f"state shape {state.shape} != ({expected_n}, 7)"

    # Positions within arena bounds
    assert np.all(state[:, 0] >= 0.5) and np.all(state[:, 0] <= 9.5), "x out of bounds"
    assert np.all(state[:, 1] >= 0.5) and np.all(state[:, 1] <= 5.5), "y out of bounds"

    # Initial velocities and tau
    assert np.all(state[:, 2] == 0.0), "vx not zero"
    assert np.all(state[:, 3] == 0.0), "vy not zero"
    assert np.all(state[:, 6] == 1.0), "tau not 1.0"

    # Groups
    assert len(scen.groups) == expected_n
    assert all(isinstance(g, int) for g in scen.groups)
    assert all(g >= -1 for g in scen.groups)

    # Metadata
    meta = scen.metadata
    assert meta["n_agents"] == expected_n
    assert meta["area"] == 60.0
    assert meta["seed"] == seed
    assert meta["density"] == density
    assert meta["flow"] == flow
    assert meta["obstacle"] == obstacle
    assert meta["speed_var"] == speed_var
    assert meta["goal_topology"] == goal_topology
    assert meta["speed_std"] == (0.2 if speed_var == "low" else 0.5)

    # Obstacle count and bounds
    obs_count = _OBS_EXPECTED[obstacle]
    assert len(scen.obstacles) == obs_count, (
        f"obstacle {obstacle}: {len(scen.obstacles)} != {obs_count}"
    )
    for x1, y1, x2, y2 in scen.obstacles:
        assert 0.0 <= x1 <= 10.0 and 0.0 <= x2 <= 10.0
        assert 0.0 <= y1 <= 6.0 and 0.0 <= y2 <= 6.0


# ---------------------------------------------------------------------------
# Metamorphic tests with explicitly documented invariants.
# ---------------------------------------------------------------------------

# These tests verify relations where a transformation of inputs (e.g. changing
# only obstacle kind or only seed) has a constrained, predictable effect on
# outputs.  They **do not** assert anything about stochastic simulator
# behaviour -- only about the deterministic generator's structural invariants.


def test_metamorphic_obstacle_independence():
    """Metamorphic relation: obstacle layout is independent of agent generation.

    Documented invariant:
        For a fixed seed and non-obstacle parameters, changing only the
        obstacle type MUST preserve agent positions (state[:, 0:2]), agent
        goals (state[:, 4:6]), group assignments (groups list), and all
        metadata fields except "obstacle" and "id".

        The only parts of the GeneratedScenario that may differ are:
        `obstacles` (the layout segments) and `simulator` (which embeds
        the obstacle geometry).

    This follows from the implementation: _sample_positions and _assign_groups
    consume the RNG before _build_obstacles is called, and _build_obstacles
    does not consume the RNG at all.

    Non-goal: does not assert anything about simulator execution outcomes.
    """
    seed = 77
    base_params = {
        "id": "meta_obs",
        "density": "med",
        "flow": "bi",
        "obstacle": "open",  # will be overridden
        "groups": 0.2,
        "speed_var": "low",
        "goal_topology": "point",
        "robot_context": "embedded",
        "repeats": 1,
    }
    reference: dict[str, GeneratedScenario] = {}

    for obs_kind in ("open", "bottleneck", "maze"):
        p = {**base_params, "obstacle": obs_kind, "id": f"meta_obs_{obs_kind}"}
        reference[obs_kind] = generate_scenario(p, seed=seed)

    ref = reference["open"]
    for obs_kind in ("bottleneck", "maze"):
        other = reference[obs_kind]
        # Agent positions and goals must be identical
        np.testing.assert_allclose(
            other.state[:, 0:2],
            ref.state[:, 0:2],
            err_msg=f"positions differ between open and {obs_kind}",
        )
        np.testing.assert_allclose(
            other.state[:, 4:6],
            ref.state[:, 4:6],
            err_msg=f"goals differ between open and {obs_kind}",
        )
        # Group assignments must be identical
        assert other.groups == ref.groups, f"groups differ between open and {obs_kind}"
        # Metadata must agree except for obstacle and id
        for k in ref.metadata:
            if k in ("obstacle", "id"):
                continue
            assert other.metadata[k] == ref.metadata[k], (
                f"metadata key '{k}' differs between open and {obs_kind}"
            )
        # Obstacle layouts must differ
        assert other.obstacles != ref.obstacles, f"obstacles should differ for {obs_kind} vs open"


def test_metamorphic_seed_structural_stability():
    """Metamorphic relation: seed changes preserve structure, vary positions.

    Documented invariant:
        For fixed scenario parameters, changing only the seed:
          (a) MUST preserve structural metadata: n_agents, area, speed_std,
              and obstacle list are identical across seeds.
              (group_count is RNG-driven and may vary with seed.)
          (b) MUST produce different agent positions for at least one agent
              (i.e. the seed actually controls the position RNG).

    This is a stronger version of test_determinism_same_seed /
    test_variation_different_seed: it verifies that the seed only affects
    the stochastic parts of generation (positions, desired speeds, groups),
    not the deterministic structural properties.

    Non-goal: does not test simulator reproducibility or episode outcomes.
    """
    params = {
        "id": "meta_seed_stab",
        "density": "high",
        "flow": "cross",
        "obstacle": "maze",
        "groups": 0.4,
        "speed_var": "high",
        "goal_topology": "circulate",
        "robot_context": "embedded",
        "repeats": 1,
    }
    seeds = [0, 1, 5, 42, 999]
    scenarios = [generate_scenario(params, seed=s) for s in seeds]

    ref = scenarios[0]

    # (a) Structural fields must be identical across all seeds
    structural_keys = ["n_agents", "area", "speed_std"]
    for i, sc in enumerate(scenarios[1:], start=1):
        for k in structural_keys:
            assert sc.metadata[k] == ref.metadata[k], (
                f"seed {seeds[i]}: metadata '{k}' {sc.metadata[k]} != {ref.metadata[k]}"
            )
        assert sc.obstacles == ref.obstacles, f"seed {seeds[i]}: obstacles differ"
        assert sc.metadata["flow"] == "cross"
        assert sc.metadata["density"] == "high"
        assert sc.metadata["goal_topology"] == "circulate"

    # (b) Positions must differ for at least one pair of seeds
    positions_differ = False
    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            if not np.allclose(scenarios[i].state[:, :2], scenarios[j].state[:, :2]):
                positions_differ = True
                break
        if positions_differ:
            break
    assert positions_differ, "all seeds produced identical positions -- seed not driving RNG"
