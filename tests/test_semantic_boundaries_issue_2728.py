"""Tests for issue 2728: 2D semantic boundary metadata.

Covers parser success, unsupported token failure, scenario load,
and validation summary contract.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = REPO_ROOT / "maps" / "svg_maps" / "issue_2728_semantic_boundaries.svg"
SCENARIO_PATH = (
    REPO_ROOT / "configs" / "scenarios" / "single" / "issue_2728_semantic_boundaries.yaml"
)


class TestSemanticBoundaryDataclass:
    """Tests for the SemanticBoundary dataclass."""

    def test_defaults(self):
        """Default flags are all False."""
        from robot_sf.nav.nav_types import SemanticBoundary

        sb = SemanticBoundary(coordinates=((0, 0), (1, 1)), label="test", id_="t1")
        assert sb.vehicle_blocking is False
        assert sb.pedestrian_passable is False
        assert sb.occluding is False
        assert sb.spawn_edge is False

    def test_all_flags(self):
        """All flags can be set to True."""
        from robot_sf.nav.nav_types import SemanticBoundary

        sb = SemanticBoundary(
            coordinates=((0, 0),),
            label="test",
            id_="t1",
            vehicle_blocking=True,
            pedestrian_passable=True,
            occluding=True,
            spawn_edge=True,
        )
        assert sb.vehicle_blocking
        assert sb.pedestrian_passable
        assert sb.occluding
        assert sb.spawn_edge

    def test_frozen(self):
        """Dataclass is frozen (immutable)."""
        from robot_sf.nav.nav_types import SemanticBoundary

        sb = SemanticBoundary(coordinates=(), label="x", id_="x")
        with pytest.raises(AttributeError):
            sb.label = "y"  # type: ignore[misc]


class TestParseSemanticBoundaryLabel:
    """Tests for SvgMapConverter._parse_semantic_boundary_label."""

    def test_single_flag(self):
        """Single flag is parsed correctly."""
        from robot_sf.nav.svg_map_parser import SvgMapConverter

        name, flags = SvgMapConverter._parse_semantic_boundary_label(
            "semantic_boundary_vehicle_blocking"
        )
        assert name == "vehicle_blocking"
        assert flags == frozenset({"vehicle_blocking"})

    def test_named_single_flag(self):
        """Named single flag labels are parsed correctly."""
        from robot_sf.nav.svg_map_parser import SvgMapConverter

        name, flags = SvgMapConverter._parse_semantic_boundary_label(
            "semantic_boundary_wall__vehicle_blocking"
        )
        assert name == "wall"
        assert flags == frozenset({"vehicle_blocking"})

    def test_multiple_flags(self):
        """Multiple flags are parsed correctly."""
        from robot_sf.nav.svg_map_parser import SvgMapConverter

        name, flags = SvgMapConverter._parse_semantic_boundary_label(
            "semantic_boundary_vehicle_blocking__occluding"
        )
        assert name == "vehicle_blocking"
        assert flags == frozenset({"vehicle_blocking", "occluding"})

    def test_named_multiple_flags(self):
        """Named multi-flag labels are parsed correctly."""
        from robot_sf.nav.svg_map_parser import SvgMapConverter

        name, flags = SvgMapConverter._parse_semantic_boundary_label(
            "semantic_boundary_sep__vehicle_blocking__occluding"
        )
        assert name == "sep"
        assert flags == frozenset({"vehicle_blocking", "occluding"})

    def test_no_flags(self):
        """No flags produces empty frozenset."""
        from robot_sf.nav.svg_map_parser import SvgMapConverter

        name, flags = SvgMapConverter._parse_semantic_boundary_label("semantic_boundary_barrier")
        assert name == "barrier"
        assert flags == frozenset()

    def test_unsupported_token_raises(self):
        """Unsupported token raises ValueError."""
        from robot_sf.nav.svg_map_parser import SvgMapConverter

        with pytest.raises(ValueError, match="Unsupported semantic boundary token"):
            SvgMapConverter._parse_semantic_boundary_label("semantic_boundary_x__bogus_flag")

    def test_all_supported_flags(self):
        """All supported flags are accepted."""
        from robot_sf.nav.nav_types import SUPPORTED_SEMANTIC_BOUNDARY_FLAGS
        from robot_sf.nav.svg_map_parser import SvgMapConverter

        label = "semantic_boundary_all__" + "__".join(sorted(SUPPORTED_SEMANTIC_BOUNDARY_FLAGS))
        name, flags = SvgMapConverter._parse_semantic_boundary_label(label)
        assert name == "all"
        assert flags == SUPPORTED_SEMANTIC_BOUNDARY_FLAGS


class TestSvgMapParserSemanticBoundaries:
    """Tests for parsing semantic boundaries from SVG maps."""

    def test_fixture_map_loads(self):
        """Fixture map loads without error."""
        from robot_sf.nav.svg_map_parser import convert_map

        map_def = convert_map(str(MAP_PATH))
        assert map_def is not None

    def test_fixture_map_has_two_boundaries(self):
        """Fixture map has exactly two semantic boundaries."""
        from robot_sf.nav.svg_map_parser import convert_map

        map_def = convert_map(str(MAP_PATH))
        assert map_def is not None
        assert len(map_def.semantic_boundaries) == 2

    def test_separator_flags(self):
        """Separator has vehicle_blocking and occluding flags."""
        from robot_sf.nav.svg_map_parser import convert_map

        map_def = convert_map(str(MAP_PATH))
        assert map_def is not None
        sep = next(b for b in map_def.semantic_boundaries if b.id_ == "separator")
        assert sep.vehicle_blocking is True
        assert sep.occluding is True
        assert sep.pedestrian_passable is False
        assert sep.spawn_edge is False

    def test_ped_spawn_edge_flags(self):
        """Ped spawn edge has pedestrian_passable and spawn_edge flags."""
        from robot_sf.nav.svg_map_parser import convert_map

        map_def = convert_map(str(MAP_PATH))
        assert map_def is not None
        ped = next(b for b in map_def.semantic_boundaries if b.id_ == "ped_spawn_edge")
        assert ped.pedestrian_passable is True
        assert ped.spawn_edge is True
        assert ped.vehicle_blocking is False
        assert ped.occluding is False

    def test_backward_compat_empty_semantic_boundaries(self):
        """MapDefinition without semantic_boundaries should still work."""
        from robot_sf.nav.map_config import MapDefinition

        md = MapDefinition(
            width=10,
            height=10,
            obstacles=[],
            robot_spawn_zones=[((0, 0), (1, 0), (1, 1))],
            ped_spawn_zones=[],
            robot_goal_zones=[((9, 9), (10, 9), (10, 10))],
            bounds=[(0, 10, 0, 0), (0, 10, 10, 10), (0, 0, 0, 10), (10, 10, 0, 10)],
            robot_routes=[],
            ped_goal_zones=[],
            ped_crowded_zones=[],
            ped_routes=[],
        )
        assert md.semantic_boundaries == []


class TestScenarioLoad:
    """Tests for loading the scenario YAML."""

    def test_scenario_file_exists(self):
        """Scenario file exists."""
        assert SCENARIO_PATH.exists()

    def test_scenario_loads(self):
        """Scenario YAML loads without error."""
        import yaml

        with open(SCENARIO_PATH) as f:
            data = yaml.safe_load(f)
        assert "scenarios" in data
        assert len(data["scenarios"]) == 2

    def test_scenario_metadata_has_claim_boundary(self):
        """Both scenarios have diagnostic_only_not_benchmark claim boundary."""
        import yaml

        with open(SCENARIO_PATH) as f:
            data = yaml.safe_load(f)
        for scenario in data["scenarios"]:
            assert scenario["metadata"]["claim_boundary"] == "diagnostic_only_not_benchmark"

    def test_scenario_metadata_has_expectations(self):
        """Both scenarios have semantic_boundary_expectations."""
        import yaml

        with open(SCENARIO_PATH) as f:
            data = yaml.safe_load(f)
        for scenario in data["scenarios"]:
            assert "semantic_boundary_expectations" in scenario["metadata"]
            expectations = scenario["metadata"]["semantic_boundary_expectations"]
            assert len(expectations) == 2


class TestValidationSummaryContract:
    """Tests for the validation script output contract."""

    def test_validation_runs(self):
        """Validation script runs successfully."""
        from scripts.validation.validate_semantic_boundaries_issue_2728 import run_validation

        results = run_validation()
        assert results["ok"] is True
        assert results["evidence_classification"] == "diagnostic_only_not_benchmark"
        assert len(results["checks"]) > 0

    def test_all_checks_pass(self):
        """All validation checks pass."""
        from scripts.validation.validate_semantic_boundaries_issue_2728 import run_validation

        results = run_validation()
        failed = [c for c in results["checks"] if not c["passed"]]
        assert not failed, f"Failed checks: {failed}"

    def test_separator_route_avoidance(self):
        """Robot route avoids the vehicle_blocking separator."""
        from scripts.validation.validate_semantic_boundaries_issue_2728 import run_validation

        results = run_validation()
        route_check = next(
            c for c in results["checks"] if c["name"] == "robot_route_avoids_separator"
        )
        assert route_check["passed"]

    def test_ped_start_near_boundary(self):
        """Pedestrian start is near the spawn-edge boundary."""
        from scripts.validation.validate_semantic_boundaries_issue_2728 import run_validation

        results = run_validation()
        ped_check = next(c for c in results["checks"] if c["name"] == "ped_start_near_spawn_edge")
        assert ped_check["passed"]
