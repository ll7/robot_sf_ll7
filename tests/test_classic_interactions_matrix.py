"""Validation tests for classic interaction scenario matrix.

Ensures that:
  * YAML loads successfully.
  * Each scenario has required keys.
  * Referenced SVG map files exist.
  * Density values fall within expected set.
  * Group scenarios only set groups when archetype == group_crossing.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_FILE = ROOT / "configs" / "scenarios" / "classic_interactions.yaml"

REQUIRED_SCENARIO_KEYS = {
    "name",
    "map_file",
    "simulation_config",
    "robot_config",
    "metadata",
    "seeds",
}
# Recommended canonical density triad retained for benchmarking summaries.
# Tests now accept any positive ped_density but emit a warning when a value
# falls outside the recommended inclusive range [0.02, 0.08]. This enables
# exploratory sweeps while preserving guidance for benchmark stability.
RECOMMENDED_DENSITIES = {0.02, 0.05, 0.08}
RECOMMENDED_RANGE = (0.02, 0.08)


@pytest.mark.parametrize("path", [SCENARIO_FILE])
def test_yaml_exists(path: Path) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
    """
    assert path.exists(), f"Scenario file missing: {path}"


def test_yaml_parses() -> None:
    """TODO docstring. Document this function."""
    scenarios = load_scenarios(SCENARIO_FILE, base_dir=SCENARIO_FILE)
    assert scenarios, "No scenarios defined"


def test_each_scenario_structure_and_files() -> None:
    """TODO docstring. Document this function."""
    scenarios = load_scenarios(SCENARIO_FILE, base_dir=SCENARIO_FILE)
    for scenario in scenarios:
        missing = REQUIRED_SCENARIO_KEYS - scenario.keys()
        assert not missing, f"Scenario {scenario.get('name')} missing keys: {missing}"
        # Resolve map_file relative to the scenario file's directory
        map_file = (SCENARIO_FILE.parent / scenario["map_file"]).resolve()
        assert map_file.exists(), f"Map file does not exist: {map_file}"
        # simulation_config density check
        sim_cfg = scenario["simulation_config"]
        assert isinstance(sim_cfg, dict), "simulation_config must be a mapping"
        density = sim_cfg.get("ped_density")
        if density is not None:
            assert density >= 0, f"ped_density must be non-negative (got {density})"
            if density == 0:
                warnings.warn(
                    f"""Scenario {scenario["name"]} ped_density=0.0 means no pedestrians spawn. This is allowed for certain empty-crowd baselines but should be used intentionally.""",
                    UserWarning,
                    stacklevel=2,
                )
            low, high = RECOMMENDED_RANGE
            if not (low <= density <= high):
                warnings.warn(
                    f"""Scenario {scenario["name"]} ped_density={density} is outside the recommended [{low}, {high}] range. This is allowed, but may reduce comparability with canonical benchmark results.""",
                    UserWarning,
                    stacklevel=2,
                )
        # groups usage rule
        groups_val = sim_cfg.get("groups", 0.0)
        archetype = scenario["metadata"].get("archetype")
        if archetype == "group_crossing":
            assert groups_val == 0.5, (
                f"group_crossing scenario must have groups=0.5 (got {groups_val})"
            )
        else:
            assert groups_val in (0.0, None), (
                f"Non-group archetype should not set groups (got {groups_val})"
            )


def test_seed_lists_have_length() -> None:
    """TODO docstring. Document this function."""
    scenarios = load_scenarios(SCENARIO_FILE, base_dir=SCENARIO_FILE)
    for scenario in scenarios:
        seeds = scenario.get("seeds")
        assert isinstance(seeds, list) and len(seeds) >= 3, (
            f"Scenario {scenario.get('name')} should have >=3 seeds"
        )
