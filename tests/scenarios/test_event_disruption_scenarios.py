"""Scenario-loader coverage for issue #4759 event-disruption scenarios."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.scenario_schema import validate_scenario_list
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ARCHETYPE = REPO_ROOT / "configs/scenarios/single/event_disruption_public_exit.yaml"
SCENARIO_MANIFEST = REPO_ROOT / "configs/scenarios/event_disruption.yaml"


class TestEventDisruptionArchetypeLoads:
    """The archetype YAML is parseable and has the expected fields."""

    def test_archetype_manifest_loads(self) -> None:
        """Archetype file loads through existing scenario loader."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        assert len(scenarios) >= 1

    def test_scenario_family_is_event_disruption(self) -> None:
        """Scenario carries event_disruption family metadata."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        assert scenario.get("scenario_family") == "event_disruption"
        assert (scenario.get("metadata") or {}).get("scenario_family") == "event_disruption"

    def test_expected_failure_modes_present(self) -> None:
        """Expected failure-mode metadata lists event-relevant modes."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        failure_modes = scenario["metadata"]["expected_failure_modes"]
        assert isinstance(failure_modes, list)
        assert len(failure_modes) >= 2
        expected_sub_modes = {"fallback_brake", "stuck", "near_miss"}
        assert expected_sub_modes.issubset(set(failure_modes))

    def test_map_resolves(self) -> None:
        """The scenario references a valid map."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        assert "map_file" in scenario or "map_id" in scenario

    def test_has_blockage_and_hazard(self) -> None:
        """Scenario defines at least one blockage/hazard region."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        ps = scenario.get("platform_semantics", {})
        assert ps.get("status") == "metadata_only"
        regions = ps.get("regions", [])
        assert len(regions) >= 1
        kinds = {r.get("kind") for r in regions}
        assert "hazard" in kinds

    def test_has_multiple_pedestrian_routes(self) -> None:
        """Scenario defines at least two pedestrians with routes."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        peds = scenario.get("single_pedestrians", [])
        assert len(peds) >= 3
        ped_ids = {p["id"] for p in peds}
        assert ped_ids == {"p1", "p2", "p3"}

    def test_non_cooperative_pedestrian_present(self) -> None:
        """At least one non-cooperative (zero-speed, proxemic-hold) pedestrian."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        peds = scenario.get("single_pedestrians", [])
        blocker = [
            p
            for p in peds
            if (p.get("metadata") or {}).get("intent_label") == "non_cooperative_blocker"
        ]
        assert len(blocker) == 1
        assert "wait_at" in blocker[0]

    def test_density_above_baseline(self) -> None:
        """Pedestrian density is set for stress testing."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        ped_density = scenario.get("simulation_config", {}).get("ped_density", 0.0)
        assert ped_density > 0.05

    def test_proxy_boundary_in_metadata(self) -> None:
        """Metadata explicitly states this is an ODD-stress proxy, not a real event model."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        meta = scenario["metadata"]
        assert meta.get("not_real_world_event_model") is True
        assert meta.get("odd_stress_proxy") == "public_event_disruption"

    def test_schema_validation_passes(self) -> None:
        """Scenario entry passes JSON-schema validation."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        errors = validate_scenario_list([dict(s) for s in scenarios])
        assert errors == [], f"Schema validation errors: {errors}"

    def test_build_robot_config_succeeds(self) -> None:
        """build_robot_config_from_scenario runs without error."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_ARCHETYPE)


class TestEventDisruptionManifest:
    """The manifest wrapper loads and expands correctly."""

    def test_manifest_loads_included_scenario(self) -> None:
        """Manifest include expands to the archetype scenario."""
        scenarios = load_scenarios(SCENARIO_MANIFEST)
        assert len(scenarios) >= 1
        names = {s.get("name") for s in scenarios}
        assert "event_disruption_public_exit_blocked_path_v1" in names

    def test_seeds_present(self) -> None:
        """Scenario has multiple seeds for reproducible evaluation."""
        scenarios = load_scenarios(SCENARIO_ARCHETYPE)
        scenario = next(
            (
                s
                for s in scenarios
                if s.get("name") == "event_disruption_public_exit_blocked_path_v1"
            ),
            None,
        )
        assert scenario is not None
        seeds = scenario.get("seeds")
        assert isinstance(seeds, list)
        assert len(seeds) >= 2
