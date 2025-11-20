"""Tests for map verification module.

This module contains unit and integration tests for the map verification
workflow, covering:
- Map inventory loading and filtering
- Rule validation (geometry, metadata, spawn points)
- Environment instantiation testing
- CLI argument parsing and execution
- Manifest output generation
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

from robot_sf.maps.verification import (
    MapRecord,
    VerificationResult,
    VerificationRunSummary,
)
from robot_sf.maps.verification.map_inventory import (
    load_map_inventory,
    get_ci_enabled_maps,
    _extract_tags_from_filename,
)
from robot_sf.maps.verification.scope_resolver import resolve_scope
from robot_sf.maps.verification.context import (
    VerificationContext,
    create_artifact_path,
    get_default_output_dir,
)


class TestDataClasses:
    """Test the core data structures."""
    
    def test_map_record_creation(self):
        """Test MapRecord can be instantiated with required fields."""
        record = MapRecord(
            map_id="test_map",
            file_path=Path("maps/svg_maps/test_map.svg"),
            tags={"classic", "ci_enabled"},
            ci_enabled=True,
            metadata={"spawn_points": [(0, 0)]},
            last_modified=datetime.now(),
        )
        
        assert record.map_id == "test_map"
        assert record.ci_enabled is True
        assert "classic" in record.tags
    
    def test_verification_result_creation(self):
        """Test VerificationResult can be instantiated."""
        result = VerificationResult(
            map_id="test_map",
            status="pass",
            rule_ids=["geometry_check", "metadata_check"],
            duration_ms=150.5,
            factory_used="robot",
            message="All checks passed",
            timestamp=datetime.now(),
        )
        
        assert result.status == "pass"
        assert result.duration_ms > 0
        assert len(result.rule_ids) == 2
    
    def test_verification_run_summary_creation(self):
        """Test VerificationRunSummary aggregation."""
        summary = VerificationRunSummary(
            run_id="test-run-001",
            git_sha="abc123",
            total_maps=10,
            passed=8,
            failed=1,
            warned=1,
            slow_maps=["large_map"],
            artifact_path="output/validation/test.json",
            started_at=datetime.now(),
            finished_at=datetime.now(),
        )
        
        assert summary.total_maps == summary.passed + summary.failed + summary.warned
        assert len(summary.slow_maps) == 1


class TestMapVerificationStub:
    """Placeholder tests for verification logic (to be implemented)."""
    
    def test_placeholder_always_passes(self):
        """This test always passes and should be replaced with real tests."""
        assert True


class TestMapInventory:
    """Tests for map inventory loading."""
    
    def test_load_map_inventory_default_dir(self):
        """Test loading maps from default directory."""
        maps = load_map_inventory()
        
        # Should find SVG maps in the default location
        assert isinstance(maps, list)
        # We expect at least some maps in the repo
        assert len(maps) > 0
        
        # Check structure of returned maps
        for map_record in maps:
            assert isinstance(map_record, MapRecord)
            assert map_record.file_path.exists()
            assert map_record.file_path.suffix == ".svg"
    
    def test_get_ci_enabled_maps(self):
        """Test filtering for CI-enabled maps."""
        all_maps = load_map_inventory()
        ci_maps = get_ci_enabled_maps(all_maps)
        
        # CI maps should be a subset
        assert len(ci_maps) <= len(all_maps)
        
        # All returned maps should be CI-enabled
        for map_record in ci_maps:
            assert map_record.ci_enabled is True
    
    def test_extract_tags_from_filename(self):
        """Test tag extraction from filenames."""
        # Classic maps
        tags = _extract_tags_from_filename("classic_crossing")
        assert "classic" in tags
        
        # Debug maps
        tags = _extract_tags_from_filename("debug_05")
        assert "debug" in tags
        
        # Test maps
        tags = _extract_tags_from_filename("test_spawn_in_obstacle")
        assert "test" in tags
        
        # Pedestrian maps
        tags = _extract_tags_from_filename("ped_metrics_example")
        assert "pedestrian_focused" in tags


class TestScopeResolver:
    """Tests for scope filtering."""
    
    def test_resolve_scope_all(self):
        """Test resolving 'all' scope."""
        maps = resolve_scope("all")
        assert isinstance(maps, list)
        assert len(maps) > 0
    
    def test_resolve_scope_ci(self):
        """Test resolving 'ci' scope."""
        maps = resolve_scope("ci")
        assert isinstance(maps, list)
        # Should only include CI-enabled maps
        for m in maps:
            assert m.ci_enabled is True
    
    def test_resolve_scope_changed(self):
        """Test resolving 'changed' scope."""
        # This may return empty if no files are changed
        maps = resolve_scope("changed")
        assert isinstance(maps, list)


class TestVerificationContext:
    """Tests for verification context."""
    
    def test_context_creation(self):
        """Test creating a verification context."""
        ctx = VerificationContext(mode="local")
        
        assert ctx.mode == "local"
        assert ctx.is_local_mode is True
        assert ctx.is_ci_mode is False
        assert isinstance(ctx.run_id, str)
        assert isinstance(ctx.git_sha, str)
        assert ctx.output_path.parent.exists()
    
    def test_ci_mode_context(self):
        """Test CI mode context."""
        ctx = VerificationContext(mode="ci")
        
        assert ctx.mode == "ci"
        assert ctx.is_ci_mode is True
        assert ctx.is_local_mode is False
    
    def test_create_artifact_path(self):
        """Test artifact path creation."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            path = create_artifact_path(base, "test-run-001")
            
            assert path.parent == base
            assert path.name.startswith("verification_")
    
    def test_get_default_output_dir(self):
        """Test getting default output directory."""
        output_dir = get_default_output_dir()
        assert output_dir == Path("output/validation")


class TestMapVerificationStubOld:
    """Original placeholder tests (to be removed after full implementation)."""
    
    def test_placeholder_always_passes_old(self):
        """This test always passes and should be replaced with real tests."""
        assert True


class TestRules:
    """Tests for validation rules."""
    
    def test_file_exists_rule_pass(self):
        """Test FileExistsRule with existing file."""
        from robot_sf.maps.verification.rules import FileExistsRule
        from robot_sf.maps.verification.map_inventory import load_map_inventory
        
        maps = load_map_inventory()
        if maps:
            rule = FileExistsRule()
            result = rule.validate(maps[0])
            assert result.passed is True
            assert result.rule_id == "file_exists"
    
    def test_apply_rules(self):
        """Test applying default rules to maps."""
        from robot_sf.maps.verification.rules import apply_rules
        from robot_sf.maps.verification.map_inventory import load_map_inventory
        
        maps = load_map_inventory()
        if maps:
            results = apply_rules(maps[0])
            assert len(results) > 0
            # Check structure
            for result in results:
                assert hasattr(result, 'rule_id')
                assert hasattr(result, 'passed')
                assert hasattr(result, 'message')


class TestRunner:
    """Tests for verification runner."""
    
    def test_verify_single_map(self):
        """Test verifying a single map."""
        from robot_sf.maps.verification.runner import verify_single_map
        from robot_sf.maps.verification.map_inventory import load_map_inventory
        from robot_sf.maps.verification.context import VerificationContext
        
        maps = load_map_inventory()
        if maps:
            ctx = VerificationContext(mode="local")
            result = verify_single_map(maps[0], ctx)
            
            assert result.map_id == maps[0].map_id
            assert result.status in ["pass", "fail", "warn"]
            assert result.duration_ms >= 0
            assert len(result.rule_ids) > 0
    
    def test_verify_maps(self):
        """Test verifying multiple maps."""
        from robot_sf.maps.verification.runner import verify_maps
        from robot_sf.maps.verification.map_inventory import load_map_inventory
        from robot_sf.maps.verification.context import VerificationContext
        
        maps = load_map_inventory()[:5]  # Test with first 5 maps
        if maps:
            ctx = VerificationContext(mode="local")
            results = verify_maps(maps, ctx)
            
            assert len(results) == len(maps)
            # All should have passed with valid SVG files
            for result in results:
                assert result.status in ["pass", "fail", "warn"]


# Integration tests for CLI will be added here
class TestCLI:
    """Tests for command-line interface (stub)."""
    
    def test_cli_import(self):
        """Test that CLI module can be imported."""
        # This will fail if the module has syntax errors
        import scripts.validation.verify_maps as verify_cli
        assert hasattr(verify_cli, 'main')
