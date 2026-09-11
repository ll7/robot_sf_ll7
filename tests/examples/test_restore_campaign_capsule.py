"""Focused tests for the offline campaign-capsule example (issue #8900)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "examples/fixtures/campaign_capsule_v1.json"
DRIVER = REPO_ROOT / "examples/advanced/40_restore_campaign_capsule.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("restore_campaign_capsule", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


def test_restore_summary_is_deterministic_and_cleans_owned_root(tmp_path: Path) -> None:
    """The positive path validates the capsule and removes both owned roots."""
    first = driver.summarize_capsule(FIXTURE, tmp_path / "first")
    second = driver.summarize_capsule(FIXTURE, tmp_path / "second")

    assert first == second
    assert first["status"] == "verified"
    assert first["execution_status"] == "not_run"
    assert first["evidence_status"] == "diagnostic_only"
    assert first["promotable"] is False
    assert first["summary"] == {
        "members": 7,
        "missing_rows": 0,
        "row_status_counts": {"native": 2},
        "rows": 2,
    }
    assert first["validation"]["lineage"]["row_artifact_count"] == 2
    assert first["validation"]["lineage"]["report_artifact_count"] == 1
    assert first["cleanup"]["ownership_verified"] is True
    assert not (tmp_path / "first").exists()
    assert str(tmp_path) not in json.dumps(first)


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("stale_manifest", "manifest_digest_mismatch"),
        ("missing_row", "row_count_mismatch"),
        ("duplicate_row", "duplicate_row"),
        ("wrong_source_identity", "source_identity_mismatch"),
        ("wrong_config_identity", "config_identity_mismatch"),
        ("checksum_mismatch", "full_digest_mismatch"),
        ("path_escape", "path_escape"),
        ("unsupported_schema", "unsupported_schema"),
        ("incomplete_copy", "missing_member"),
    ],
)
def test_negative_cases_fail_closed(tmp_path: Path, case: str, error: str) -> None:
    """Requested stale, identity, integrity, path, schema, and copy failures stay diagnostic."""
    root = tmp_path / case
    result = driver.summarize_capsule(FIXTURE, root, case=case)

    assert result["status"] == "failed"
    assert result["error"] == {"code": error}
    assert result["execution_status"] == "not_run"
    assert result["evidence_status"] == "diagnostic_only"
    assert result["cleanup"] == {
        "marker": ".robot_sf_offline_capsule_owned",
        "ownership_verified": True,
        "status": "removed",
    }
    assert not root.exists()
    assert not (tmp_path / "outside.json").exists()


def test_cleanup_requires_exact_ownership_marker(tmp_path: Path) -> None:
    """A root without the example marker is not deleted."""
    root = tmp_path / "unowned"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(driver.CapsuleError) as caught:
        driver._cleanup(root)

    assert caught.value.code == "cleanup_ownership"
    assert sentinel.read_text(encoding="utf-8") == "keep"
