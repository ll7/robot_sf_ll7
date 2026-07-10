"""Tests for issue #4974: field-observation-informed ped-robot interaction.

Covers both halves of the issue:

Model side (calibration surface):
  * Per-scenario ``simulation_config.prf_config`` override exposes the ped-robot
    force coefficient, effective radius, activation distance, and active flag.
  * The ``hesitating`` response law maps to ``hesitating_response_multiplier``
    (the "optional hesitation state").
  * Backward compatibility: defaults are unchanged when fields are omitted.

Scenario side:
  * The new archetype file loads, has the three required archetypes, valid
    structure, existing map references, and uses the new calibration surface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.ped_npc.ped_archetypes import assign_archetype_labels
from robot_sf.ped_npc.ped_robot_force import PedRobotForce, PedRobotForceConfig
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
)

WORKTREE = Path(__file__).resolve().parents[2]
ARCHETYPE_FILE = (
    WORKTREE / "configs" / "scenarios" / "archetypes" / "issue_4974_field_observed_ped_robot.yaml"
)


class _DummyPedState:
    """Minimal mock of PySocialForce ``PedState`` for force unit tests."""

    def __init__(self, positions: np.ndarray, radius: float = 0.3):
        self._pos = positions
        self.agent_radius = radius

    def pos(self) -> np.ndarray:
        return self._pos

    def size(self) -> int:
        return self._pos.shape[0]


# ---------------------------------------------------------------------------
# Model side: prf_config calibration surface
# ---------------------------------------------------------------------------


def _build_config(scenario_name: str) -> SimulationSettings:
    """Load one scenario from the issue #4974 archetype file into sim settings."""
    scenarios = load_scenarios(ARCHETYPE_FILE, base_dir=ARCHETYPE_FILE)
    scenario = next(s for s in scenarios if s["name"] == scenario_name)
    robot_cfg = build_robot_config_from_scenario(scenario, scenario_path=ARCHETYPE_FILE)
    return robot_cfg.sim_config


def test_prf_config_override_applies_coefficient_and_radius() -> None:
    """Per-scenario prf_config overrides reach SimulationSettings.prf_config."""
    sim_cfg = _build_config("issue_4974_robot_blocks_narrow_path")
    prf = sim_cfg.prf_config
    assert prf.force_multiplier == pytest.approx(15.0)
    assert prf.robot_radius == pytest.approx(1.2)
    assert prf.activation_threshold == pytest.approx(2.5)
    assert prf.is_active is True


def test_prf_config_override_preserves_omitted_defaults() -> None:
    """A scenario that omits a prf_config field keeps the field default."""
    scenarios = load_scenarios(ARCHETYPE_FILE, base_dir=ARCHETYPE_FILE)
    # Build a minimal override payload that sets only force_multiplier.
    scenario = {**scenarios[0]}
    scenario = {**scenario}
    scenario["simulation_config"] = {"prf_config": {"force_multiplier": 7.5}}
    robot_cfg = build_robot_config_from_scenario(scenario, scenario_path=ARCHETYPE_FILE)
    prf = robot_cfg.sim_config.prf_config
    assert prf.force_multiplier == pytest.approx(7.5)
    # Omitted fields fall back to PedRobotForceConfig defaults.
    assert prf.robot_radius == pytest.approx(1.0)
    assert prf.activation_threshold == pytest.approx(2.0)
    assert prf.is_active is True


def test_prf_config_override_rejects_unknown_keys() -> None:
    """Unknown prf_config keys fail closed."""
    scenario = {
        "name": "bogus",
        "map_file": "../../../maps/svg_maps/classic_doorway.svg",
        "simulation_config": {"prf_config": {"not_a_field": 1.0}},
        "robot_config": {},
        "metadata": {},
        "seeds": [1, 2, 3],
    }
    with pytest.raises(ValueError, match="prf_config contains unknown keys"):
        build_robot_config_from_scenario(scenario, scenario_path=ARCHETYPE_FILE)


def test_prf_config_override_rejects_negative_radius() -> None:
    """Negative effective radius fails closed via positive-float coercion."""
    scenario = {
        "name": "bogus",
        "map_file": "../../../maps/svg_maps/classic_doorway.svg",
        "simulation_config": {"prf_config": {"robot_radius": -1.0}},
        "robot_config": {},
        "metadata": {},
        "seeds": [1, 2, 3],
    }
    with pytest.raises(ValueError, match="robot_radius must be > 0"):
        build_robot_config_from_scenario(scenario, scenario_path=ARCHETYPE_FILE)


# ---------------------------------------------------------------------------
# Model side: hesitating response law (optional hesitation state)
# ---------------------------------------------------------------------------


def test_hesitating_response_multiplier_default_preserves_behavior() -> None:
    """Default hesitating_response_multiplier is 1.0 (no-op vs reactive default)."""
    sim = SimulationSettings()
    assert sim.hesitating_response_multiplier == pytest.approx(1.0)


def test_hesitating_response_multiplier_validation_rejects_negative() -> None:
    """Negative hesitating multipliers are rejected at construction time."""
    with pytest.raises(ValueError, match="hesitating_response_multiplier must be a finite value"):
        SimulationSettings(hesitating_response_multiplier=-0.1)


def test_hesitating_response_multiplier_reaches_scenario_config() -> None:
    """The hesitation archetype's multiplier and composition reach sim settings."""
    sim_cfg = _build_config("issue_4974_pedestrian_hesitation")
    assert sim_cfg.hesitating_response_multiplier == pytest.approx(1.8)
    assert sim_cfg.response_law_composition == {
        "reactive": pytest.approx(0.6),
        "hesitating": pytest.approx(0.4),
    }


def test_ped_robot_force_applies_hesitating_multiplier_above_reactive() -> None:
    """A hesitating pedestrian (multiplier > 1) feels stronger repulsion than reactive (1.0)."""
    peds = _DummyPedState(np.array([[1.0, 0.0]], dtype=float))
    config = PedRobotForceConfig(
        is_active=True,
        robot_radius=0.5,
        activation_threshold=5.0,
        force_multiplier=10.0,
    )

    reactive = PedRobotForce(
        config, peds, get_robot_pos=lambda: np.array([0.0, 0.0], dtype=float)
    )()
    reactive_mag = float(np.linalg.norm(reactive[0]))

    hesitating = PedRobotForce(
        config,
        peds,
        get_robot_pos=lambda: np.array([0.0, 0.0], dtype=float),
        get_ped_response_multipliers=lambda: np.array([1.8], dtype=float),
    )()
    hesitating_mag = float(np.linalg.norm(hesitating[0]))

    assert hesitating_mag == pytest.approx(1.8 * reactive_mag, rel=1e-6)


def test_hesitating_label_assignment_is_deterministic() -> None:
    """Seeded hesitating/reactive assignment is deterministic with the right counts."""
    comp = {"reactive": 0.6, "hesitating": 0.4}
    labels_1 = assign_archetype_labels(10, comp, seed=4974)
    labels_2 = assign_archetype_labels(10, comp, seed=4974)
    assert np.array_equal(labels_1, labels_2)
    assert int(np.count_nonzero(labels_1 == "hesitating")) == 4
    assert int(np.count_nonzero(labels_1 == "reactive")) == 6


# ---------------------------------------------------------------------------
# Backward compatibility: existing scenarios unaffected
# ---------------------------------------------------------------------------


def test_existing_scenario_without_prf_override_uses_defaults() -> None:
    """A scenario without prf_config keeps the default PedRobotForceConfig."""
    classic_doorway = WORKTREE / "configs" / "scenarios" / "archetypes" / "classic_doorway.yaml"
    scenarios = load_scenarios(classic_doorway, base_dir=classic_doorway)
    sim_cfg = build_robot_config_from_scenario(
        scenarios[0], scenario_path=classic_doorway
    ).sim_config
    default = PedRobotForceConfig()
    prf = sim_cfg.prf_config
    assert prf.force_multiplier == pytest.approx(default.force_multiplier)
    assert prf.robot_radius == pytest.approx(default.robot_radius)
    assert prf.activation_threshold == pytest.approx(default.activation_threshold)
    assert prf.is_active == default.is_active


# ---------------------------------------------------------------------------
# Scenario side: archetype file structure
# ---------------------------------------------------------------------------

REQUIRED_SCENARIO_KEYS = {
    "name",
    "map_file",
    "simulation_config",
    "robot_config",
    "metadata",
    "seeds",
}
EXPECTED_ARCHETYPES = {
    "robot_blocks_narrow_path",
    "pedestrian_hesitation",
    "give_way_asymmetry",
}


@pytest.fixture(scope="module")
def scenarios() -> list[dict]:
    """Load the issue #4974 archetype file once for the scenario-side tests."""
    return list(load_scenarios(ARCHETYPE_FILE, base_dir=ARCHETYPE_FILE))


def test_archetype_file_loads(scenarios: list[dict]) -> None:
    """The archetype file parses and defines three scenarios."""
    assert len(scenarios) == 3


def test_archetype_names_match_field_observed_set(scenarios: list[dict]) -> None:
    """The three required field-observed archetypes are present and named correctly."""
    archetypes = {s["metadata"]["archetype"] for s in scenarios}
    assert archetypes == EXPECTED_ARCHETYPES


def test_each_scenario_has_required_structure(scenarios: list[dict]) -> None:
    """Each scenario has required keys and an existing map file."""
    for scenario in scenarios:
        missing = REQUIRED_SCENARIO_KEYS - scenario.keys()
        assert not missing, f"Scenario {scenario.get('name')} missing keys: {missing}"
        map_file = (ARCHETYPE_FILE.parent / scenario["map_file"]).resolve()
        assert map_file.exists(), f"Map file does not exist: {map_file}"


def test_each_scenario_uses_calibration_surface(scenarios: list[dict]) -> None:
    """Each scenario exercises the new prf_config calibration surface."""
    for scenario in scenarios:
        prf = scenario["simulation_config"].get("prf_config")
        assert isinstance(prf, dict), f"{scenario['name']} must define simulation_config.prf_config"
        assert "force_multiplier" in prf
        assert "robot_radius" in prf


def test_hesitation_and_giveway_use_hesitating_response_law(scenarios: list[dict]) -> None:
    """The hesitation and give-way archetypes declare a hesitating response-law mix."""
    by_name = {s["metadata"]["archetype"]: s for s in scenarios}
    for archetype in ("pedestrian_hesitation", "give_way_asymmetry"):
        comp = by_name[archetype]["simulation_config"]["response_law_composition"]
        assert "hesitating" in comp, f"{archetype} must include a hesitating response law"
        mult = by_name[archetype]["simulation_config"]["hesitating_response_multiplier"]
        assert mult > 1.0, (
            f"{archetype} hesitating_response_multiplier should model larger clearance"
        )


def test_scenario_metadata_marks_exploratory_no_benchmark_claim(scenarios: list[dict]) -> None:
    """All archetypes honestly label themselves exploratory with no benchmark claim."""
    for scenario in scenarios:
        meta = scenario["metadata"]
        assert meta["evidence_tier"] == "exploratory"
        assert meta["benchmark_evidence"] is False
        assert "claim_boundary" in meta
        assert "4974" in str(meta.get("issue", ""))


def test_scenario_seeds_are_int_lists(scenarios: list[dict]) -> None:
    """Each scenario carries an integer seed list of length >= 3."""
    for scenario in scenarios:
        seeds = scenario["seeds"]
        assert isinstance(seeds, list) and len(seeds) >= 3
        assert all(isinstance(seed, int) for seed in seeds)
