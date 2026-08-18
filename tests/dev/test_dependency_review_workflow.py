"""Tests for the dependency-review workflow's license policy coverage."""

from __future__ import annotations

from pathlib import Path

import yaml


def _dependency_review_step(workflow: dict[str, object]) -> dict[str, object]:
    """Find the dependency-review action without relying on step position."""
    steps = workflow["jobs"]["review"]["steps"]  # type: ignore[index]
    matches = [
        step
        for step in steps
        if step.get("uses", "").split("@", maxsplit=1)[0] == "actions/dependency-review-action"
    ]
    assert len(matches) == 1, f"expected one dependency-review action, got {len(matches)}"
    return matches[0]


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

    steps = workflow["jobs"]["review"]["steps"]  # type: ignore[index]
    dependency_review_step = _dependency_review_step(workflow)
    dependency_review_index = steps.index(dependency_review_step)
    later_steps = steps[dependency_review_index + 1 :]
    assert later_steps, "evidence steps must remain covered by this regression"
    assert any(
        step.get("uses", "").split("@", maxsplit=1)[0] == "actions/upload-artifact"
        for step in later_steps
    ), "the test must exercise a workflow with later artifact steps"

    allow_licenses = dependency_review_step["with"]["allow-licenses"]  # type: ignore[index]
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
