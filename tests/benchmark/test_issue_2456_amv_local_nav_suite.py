"""Contract tests for issue #2456 AMV Local Navigation Evaluation Suite proposal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_2456_amv_local_nav_suite_v0.yaml"

REQUIRED_DIMENSIONS: set[str] = {
    "actuation_feasibility",
    "braking_margin",
    "yaw_rate_saturation",
    "command_smoothness",
    "latency_sensitivity",
    "low_speed_stability",
    "narrow_corridor_progress",
    "sidewalk_crossing",
}

BOUNDARY_ISSUES: set[int] = {1559, 1585, 2000, 2001}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_suite_declares_proposal_evidence_tier() -> None:
    """Verify the suite is explicitly proposal-only, not benchmark evidence."""
    payload = _load_yaml(CONFIG_PATH)
    assert payload["evidence_tier"] == "proposal"
    assert payload["benchmark_evidence"] is False
    assert payload["status"] == "proposed"


def test_suite_has_versioned_suite_id() -> None:
    """Verify the suite exposes a stable identifier and initial version."""
    payload = _load_yaml(CONFIG_PATH)
    assert payload["suite_id"] == "amv_local_nav_suite_v0"
    assert payload["suite_version"] == "0"


def test_suite_declares_all_required_dimensions() -> None:
    """Verify the manifest names the required AMV-specific dimensions."""
    payload = _load_yaml(CONFIG_PATH)
    declared = set(payload["evaluation_dimensions"].keys())
    assert declared == REQUIRED_DIMENSIONS, (
        f"Expected dimensions {REQUIRED_DIMENSIONS}, got {declared}"
    )


def test_each_dimension_has_evidence_boundary_and_metrics() -> None:
    """Verify every dimension declares evidence boundaries and metric needs."""
    payload = _load_yaml(CONFIG_PATH)
    dims = payload["evaluation_dimensions"]
    for name, dimension in dims.items():
        assert "evidence_boundary" in dimension, f"{name} missing evidence_boundary"
        assert dimension["evidence_boundary"] in ("synthetic", "proxy"), (
            f"{name} has unexpected boundary: {dimension['evidence_boundary']}"
        )
        assert "required_metrics" in dimension, f"{name} missing required_metrics"
        assert len(dimension["required_metrics"]) > 0, f"{name} has empty metrics"
        for metric in dimension["required_metrics"]:
            assert "name" in metric, f"{name} metric missing name"
            assert "status" in metric, f"{name} metric {metric.get('name')} missing status"
            assert metric["status"] in ("existing", "proposed"), (
                f"{name} metric {metric['name']} has unexpected status"
            )


def test_suite_links_boundary_issues_without_claiming_resolution() -> None:
    """Verify unresolved boundary issues are linked without closure claims."""
    payload = _load_yaml(CONFIG_PATH)
    related = payload["related_issues"]
    declared_issues = {entry["issue"] for entry in related}
    for issue_num in BOUNDARY_ISSUES:
        assert issue_num in declared_issues, (
            f"Boundary issue #{issue_num} not found in related_issues"
        )
    for entry in related:
        assert "status" in entry, f"Issue #{entry['issue']} missing status field"
        assert "claim" in entry, f"Issue #{entry['issue']} missing claim field"
        assert entry["status"] != "resolved", (
            f"Issue #{entry['issue']} claims resolved status; "
            f"must not claim resolution of boundary issues"
        )


def test_hardware_calibrated_is_blocked() -> None:
    """Verify hardware-calibrated AMV evidence remains blocked."""
    payload = _load_yaml(CONFIG_PATH)
    classification = payload["evidence_boundary_classification"]
    assert "hardware_calibrated" in classification
    hc = classification["hardware_calibrated"]
    assert hc["status"] == "blocked"
    assert hc["dimensions"] == []
    assert "block_reason" in hc
    assert 1585 in hc["gate_issues"]
    assert 2000 in hc["gate_issues"]


def test_evidence_boundary_classification_covers_all_dimensions() -> None:
    """Verify synthetic/proxy classifications partition the dimensions."""
    payload = _load_yaml(CONFIG_PATH)
    classification = payload["evidence_boundary_classification"]
    synthetic_dims = set(classification["synthetic"]["dimensions"])
    proxy_dims = set(classification["proxy"]["dimensions"])
    all_classified = synthetic_dims | proxy_dims
    declared = set(payload["evaluation_dimensions"].keys())
    assert all_classified == declared, (
        f"Dimension coverage mismatch: classified {all_classified}, declared {declared}"
    )
    assert synthetic_dims & proxy_dims == set(), (
        f"Dimension overlap between synthetic and proxy: {synthetic_dims & proxy_dims}"
    )
    hc_dims = set(classification["hardware_calibrated"]["dimensions"])
    assert hc_dims == set(), "hardware_calibrated should have no dimensions"


def test_claim_boundary_excludes_hardware_and_paper_claims() -> None:
    """Verify the claim boundary excludes unsupported AMV realism claims."""
    payload = _load_yaml(CONFIG_PATH)
    cb = payload["claim_boundary"].lower()
    assert "does not provide" in cb
    assert "does not" in cb
    assert "not" in cb.lower()
    assert "hardware-calibrated" in cb.lower()
    assert "paper-facing" in cb.lower()


def test_fallback_policy_is_fail_closed() -> None:
    """Verify fallback and degraded modes are not successful evidence."""
    payload = _load_yaml(CONFIG_PATH)
    fp = payload["fallback_policy"]
    assert fp["fallback_is_success"] is False
    assert fp["degraded_is_success"] is False


def test_scenario_classes_defined() -> None:
    """Verify sidewalk and crossing scenario classes are defined."""
    payload = _load_yaml(CONFIG_PATH)
    classes = payload["scenario_classes"]
    assert "sidewalk" in classes
    assert "crossing" in classes
    for name, cls in classes.items():
        assert "description" in cls, f"Scenario class {name} missing description"
        assert "covered_dimensions" in cls, f"Scenario class {name} missing covered_dimensions"
        assert len(cls["covered_dimensions"]) > 0, (
            f"Scenario class {name} has empty covered_dimensions"
        )
