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
        "pyproject.toml",
        "uv.lock",
        "fast-pysf/pyproject.toml",
        "fast-pysf/uv.lock",
        "third_party/python-rvo2/pyproject.toml",
        "third_party/python-rvo2/setup.py",
        "third_party/python-rvo2/requirements.txt",
        "third_party/python-rvo2/LICENSE",
        "third_party/python-rvo2/UPSTREAM.md",
        "third_party/python-rvo2/LOCAL_CHANGES.patch",
        "third_party/socnavbench/LICENSE",
        "third_party/socnavbench/LICENSES/Apache-2.0.txt",
        "third_party/socnavbench/LICENSING.yaml",
        "third_party/socnavbench/UPSTREAM.md",
        "scripts/tools/check_dependency_license_inventory.py",
        "tests/tools/test_check_dependency_license_inventory.py",
    }
    assert expected_paths <= paths

    allow_licenses = workflow["jobs"]["review"]["steps"][-1]["with"]["allow-licenses"]
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


def test_packaging_workflow_covers_archive_license_and_provenance_inputs() -> None:
    """Archive checks must rerun when any shipped legal/provenance input changes."""
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github" / "workflows" / "packaging-extras.yml").read_text(encoding="utf-8")
    )
    trigger = workflow.get("on") or workflow[True]
    paths = set(trigger["pull_request"]["paths"])
    expected_paths = {
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "fast-pysf/LICENSE",
        "third_party/python-rvo2/LICENSE",
        "third_party/python-rvo2/UPSTREAM.md",
        "third_party/python-rvo2/LOCAL_CHANGES.patch",
        "third_party/socnavbench/LICENSE",
        "third_party/socnavbench/LICENSES/Apache-2.0.txt",
        "third_party/socnavbench/LICENSING.yaml",
        "third_party/socnavbench/UPSTREAM.md",
        "scripts/tools/check_distribution_licenses.py",
        "tests/tools/test_check_distribution_licenses.py",
        ".github/workflows/packaging-extras.yml",
    }
    assert expected_paths <= paths
