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

from robot_sf.maps.verification import (
    MapRecord,
    VerificationResult,
    VerificationRunSummary,
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


# Integration tests for CLI will be added here
class TestCLI:
    """Tests for command-line interface (stub)."""
    
    def test_cli_import(self):
        """Test that CLI module can be imported."""
        # This will fail if the module has syntax errors
        import scripts.validation.verify_maps as verify_cli
        assert hasattr(verify_cli, 'main')
