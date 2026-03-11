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
# Tests now accept any non-negative ped_density, but require explicit
# ``metadata.density_advisory`` annotations for intentional outlier values.
RECOMMENDED_DENSITIES = {0.02, 0.05, 0.08}
RECOMMENDED_RANGE = (0.02, 0.08)
ZERO_DENSITY_ADVISORY = "zero_baseline_route_spawn"
LOW_DENSITY_ADVISORY = "low_density_exploration"
HIGH_DENSITY_ADVISORY = "high_density_stress"


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
    """Validate per-scenario schema, map references, and density advisory contracts.

    This test ensures each scenario includes required keys, references an existing
    map file, uses non-negative pedestrian density, and sets
    ``metadata.density_advisory`` when densities intentionally fall outside the
    recommended range.
    """
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
        metadata = scenario["metadata"]
        assert isinstance(metadata, dict), "metadata must be a mapping"
        density = sim_cfg.get("ped_density")
        if density is not None:
            assert density >= 0, f"ped_density must be non-negative (got {density})"
            advisory = metadata.get("density_advisory")
            if density == 0:
                assert advisory == ZERO_DENSITY_ADVISORY, (
                    f"Scenario {scenario['name']} uses ped_density=0.0 and must set "
                    f"metadata.density_advisory={ZERO_DENSITY_ADVISORY!r}."
                )
            elif density < RECOMMENDED_RANGE[0]:
                assert advisory == LOW_DENSITY_ADVISORY, (
                    f"Scenario {scenario['name']} uses ped_density={density} and must set "
                    f"metadata.density_advisory={LOW_DENSITY_ADVISORY!r}."
                )
            elif density > RECOMMENDED_RANGE[1]:
                assert advisory == HIGH_DENSITY_ADVISORY, (
                    f"Scenario {scenario['name']} uses ped_density={density} and must set "
                    f"metadata.density_advisory={HIGH_DENSITY_ADVISORY!r}."
                )
            else:
                assert advisory in (None, ""), (
                    f"Scenario {scenario['name']} is in the recommended density range and should "
                    "not set metadata.density_advisory."
                )
        # groups usage rule
        groups_val = sim_cfg.get("groups", 0.0)
        archetype = metadata.get("archetype")
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
