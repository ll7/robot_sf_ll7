"""Validation tests for classic interaction scenario matrix.

Ensures that:
  * YAML loads successfully.
  * Each scenario has required keys.
  * Referenced SVG map files exist.
  * Density values fall within expected set.
  * Group scenarios only set groups when archetype == group_crossing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
ALLOWED_DENSITIES = {0.02, 0.05, 0.08}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("path", [SCENARIO_FILE])
def test_yaml_exists(path: Path) -> None:
    assert path.exists(), f"Scenario file missing: {path}"


def test_yaml_parses() -> None:
    data = load_yaml(SCENARIO_FILE)
    assert "scenarios" in data and isinstance(data["scenarios"], list)
    assert data["scenarios"], "No scenarios defined"


def test_each_scenario_structure_and_files() -> None:
    data = load_yaml(SCENARIO_FILE)
    for scenario in data["scenarios"]:
        missing = REQUIRED_SCENARIO_KEYS - scenario.keys()
        assert not missing, f"Scenario {scenario.get('name')} missing keys: {missing}"
        map_file = ROOT / scenario["map_file"]
        assert map_file.exists(), f"Map file does not exist: {map_file}"
        # simulation_config density check
        sim_cfg = scenario["simulation_config"]
        assert isinstance(sim_cfg, dict), "simulation_config must be a mapping"
        density = sim_cfg.get("ped_density")
        if density is not None:
            assert density in ALLOWED_DENSITIES, (
                f"Unexpected ped_density {density} in {scenario['name']}"
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
    data = load_yaml(SCENARIO_FILE)
    for scenario in data["scenarios"]:
        seeds = scenario.get("seeds")
        assert isinstance(seeds, list) and len(seeds) >= 3, (
            f"Scenario {scenario.get('name')} should have >=3 seeds"
        )
