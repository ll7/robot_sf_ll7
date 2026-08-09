"""Tests for the dependency-review workflow's license policy coverage."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_dependency_review_covers_vendored_build_and_license_surfaces() -> None:
    """Dependency-review changes must include the vendored package manifests."""
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github" / "workflows" / "dependency-review.yml").read_text(encoding="utf-8")
    )
    trigger = workflow.get("on") or workflow[True]
    assert "workflow_dispatch" in trigger
    assert "pull_request" in trigger
    paths = set(trigger["pull_request"]["paths"])
    assert {
        "third_party/python-rvo2/pyproject.toml",
        "third_party/python-rvo2/setup.py",
        "third_party/socnavbench/LICENSING.yaml",
    } <= paths

    allow_licenses = workflow["jobs"]["review"]["steps"][-1]["with"]["allow-licenses"]
    allowed = {item.strip() for item in allow_licenses.split(",")}
    assert {"Apache-2.0", "MIT", "GPL-3.0-only"} <= allowed
