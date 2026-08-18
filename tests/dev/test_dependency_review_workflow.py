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
    expected_paths = {
        "third_party/python-rvo2/pyproject.toml",
        "third_party/python-rvo2/setup.py",
        "third_party/python-rvo2/requirements.txt",
        "third_party/socnavbench/LICENSING.yaml",
    }
    assert expected_paths <= paths

    review_steps = workflow["jobs"]["review"]["steps"]
    dependency_review_steps = [
        step
        for step in review_steps
        if step.get("uses", "").startswith("actions/dependency-review-action@")
    ]
    assert len(dependency_review_steps) == 1
    allow_licenses = dependency_review_steps[0]["with"]["allow-licenses"]
    allowed = {item.strip() for item in allow_licenses.split(",")}
    assert allowed == {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "LGPL-2.1-only",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "LGPL-3.0-or-later",
        "MIT",
        "MPL-2.0",
        "PSF-2.0",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
    }
