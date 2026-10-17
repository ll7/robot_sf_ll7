"""Affected PR selection requires default admission at the exact matrix head."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_pr_shards_enable_affected_selection_only_with_default_admission():
    """A live fast-file registry must not restrict tests before affected selection."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["fast-feedback"]
    # Default admission supersedes the intermediate stack's blanket slow opt-in.
    # The selector's conservative decision now owns admission of affected slow tests.
    conftest = (ROOT / "tests/conftest.py").read_text()
    assert "github.event.pull_request.base.sha" in job["env"]["ROBOT_SF_AFFECTED_BASE_REF"]
    # Anchor the assignment so retained _LEGACY_FAST_FILES data cannot trigger
    # the guard; reject both ordinary and annotated live module-level assignments.
    assert re.search(r"^_FAST_FILES(?:\s*:[^=\n]+)?\s*=", conftest, re.MULTILINE) is None
    checkout = next(step for step in job["steps"] if step.get("name") == "Checkout")
    assert checkout["with"]["fetch-depth"] == 0
    assert "github.event.pull_request.head.sha" in checkout["with"]["ref"]


def test_selector_is_prepared_once_for_the_exact_pr_range():
    """Avoid six repeated scans while preserving the exact PR base/head range."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    setup = workflow["jobs"]["dispatch-ownership"]
    prepare = next(
        (s for s in setup["steps"] if s.get("name") == "Prepare conservative test admission"), None
    )
    assert prepare is not None
    assert '--base "$BASE_SHA"' in prepare["run"] and '--head "$HEAD_SHA"' in prepare["run"]
    assert "--json-output output/ci/affected-selection.json" in prepare["run"]
    runs = "\n".join(
        str(s.get("run", "")) for j in workflow["jobs"].values() for s in j.get("steps", [])
    )
    assert runs.count("python scripts/dev/affected_test_selection.py") == 1
    checkout = setup["steps"][0]["with"]
    assert checkout["ref"] == "${{ github.sha }}" and checkout["fetch-depth"] == 0
    upload = next(
        s for s in setup["steps"] if s.get("name") == "Upload conservative test admission"
    )
    assert upload["with"]["if-no-files-found"] == "error"


def test_dispatch_ownership_helper_tests_use_workflow_checkout():
    """Workflow-added helper tests must exist even on older PR branches."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    setup = workflow["jobs"]["dispatch-ownership"]
    checkout = setup["steps"][0]
    helper_test = next(
        s
        for s in setup["steps"]
        if s.get("name") == "Test CI fallback decisions without project dependencies"
    )

    assert checkout["name"] == "Checkout gate source"
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert "github.event.pull_request.head.sha" not in checkout["with"]["ref"]
    assert "tests/ci/runner_fallback.test.cjs" in helper_test["run"]
