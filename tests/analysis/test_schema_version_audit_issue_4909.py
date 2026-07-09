"""Focused tests for the schema_version audit script — Issue #4909."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "analysis" / "audit_schema_version_coverage.py"


@pytest.fixture(scope="module")
def audit_result() -> dict:
    """Run the audit script once and parse the JSON inventory."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    out_path = REPO_ROOT / "output" / "schema_version_audit_inventory.json"
    assert out_path.exists(), "Inventory file not created"
    return json.loads(out_path.read_text())


def test_script_runs_without_error(audit_result: dict) -> None:
    """Verify the audit script produces output without errors."""
    assert "total_distinct_values" in audit_result
    assert audit_result["total_distinct_values"] > 0


def test_all_five_buckets_present(audit_result: dict) -> None:
    """Verify all five classification buckets appear in the inventory."""
    expected_buckets = {
        "jsonschema-validated",
        "named-const-backed",
        "test-asserted",
        "provenance-write-only-by-design",
        "genuinely-orphaned-needs-validator",
    }
    actual = set(audit_result["classification_counts"].keys())
    assert expected_buckets == actual, f"Missing or extra buckets: {expected_buckets ^ actual}"


def test_entries_have_required_fields(audit_result: dict) -> None:
    """Verify every entry has value, classification, and writers."""
    for entry in audit_result["entries"]:
        assert "value" in entry, "Missing 'value' in entry"
        assert "classification" in entry, f"Missing 'classification' in {entry['value']}"
        assert "writers" in entry, f"Missing 'writers' in {entry['value']}"
        assert len(entry["writers"]) > 0, f"No writers for {entry['value']}"


def test_negative_test_fixtures_excluded(audit_result: dict) -> None:
    """Verify negative test fixtures are excluded from the inventory."""
    values = {e["value"] for e in audit_result["entries"]}
    for excluded in ["wrong", "bogus", "invalid", "unexpected", "WrongLedger.v1", "bogus.v9"]:
        assert excluded not in values, f"Negative test fixture '{excluded}' should be excluded"


def test_issue_emitters_classified(audit_result: dict) -> None:
    """The five emitters named in issue #4909 should all appear in the inventory."""
    issue_values = {
        "static-deadlock-distance-to-goal-delta.v1",
        "safety-shield-stats.v1",
        "forecast_dataset_row.v1",
        "topology_route_progress_summary.v1",
        "release-badge-validation-report.v1",
    }
    inventory_values = {e["value"] for e in audit_result["entries"]}
    missing = issue_values - inventory_values
    assert not missing, f"Issue emitters missing from inventory: {missing}"


def test_jsonschema_validated_entry_has_backing(audit_result: dict) -> None:
    """Verify jsonschema-validated entries have jsonschema backing info."""
    for entry in audit_result["entries"]:
        if entry["classification"] == "jsonschema-validated":
            assert entry.get("jsonschema_backing"), (
                f"{entry['value']} classified as jsonschema-validated but has no backing"
            )


def test_orphaned_count_matches_sum(audit_result: dict) -> None:
    """Verify classification counts sum to total entries."""
    total = sum(audit_result["classification_counts"].values())
    assert total == audit_result["total_distinct_values"]
    assert total == len(audit_result["entries"])


def test_const_backed_values_have_constant_info(audit_result: dict) -> None:
    """Verify named-const-backed entries have constant backing info."""
    for entry in audit_result["entries"]:
        if entry["classification"] == "named-const-backed":
            assert entry.get("constant_backing"), (
                f"{entry['value']} classified as named-const-backed but has no constant info"
            )
