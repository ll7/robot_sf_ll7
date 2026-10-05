"""Deterministic admission and cleanup at the browser and child boundaries."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("version", ["v22.7.0", "v22.20.0", "v24.15.0"])
def test_browser_runtime_checks_supported_version(monkeypatch, version):
    from tests.support.browser_runtime import require_node_runtime

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, version + "\n", "")
    )
    assert require_node_runtime() == version


def test_child_completion_and_hang_cleanup(tmp_path):
    from tests.support.subprocess_events import capture_completion

    result = capture_completion([sys.executable, "-c", 'print("ready")'])
    assert result.returncode == 0 and result.stdout.strip() == "ready"
    with pytest.raises(subprocess.TimeoutExpired):
        capture_completion(
            [sys.executable, "-c", "import time; time.sleep(120)"], hang_guard_seconds=0.1
        )


@pytest.mark.parametrize("ci_flag", ["CI", "GITHUB_ACTIONS"])
def test_missing_node_fails_in_ci(monkeypatch, tmp_path, ci_flag):
    """An empty executable search path must not silently skip a CI witness."""
    from tests.support.browser_runtime import require_node_runtime

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv(ci_flag, "true")
    monkeypatch.setenv("PATH", str(tmp_path))
    try:
        with pytest.raises(pytest.fail.Exception, match="Node.*missing.*CI"):
            require_node_runtime()
    except pytest.skip.Exception as exc:
        pytest.fail(f"Missing Node silently skipped in CI: {exc}")

    monkeypatch.delenv(ci_flag)
    with pytest.raises(pytest.skip.Exception, match="Node.*absent locally"):
        require_node_runtime()


@pytest.mark.parametrize("ci_flag", [None, "CI", "GITHUB_ACTIONS"])
@pytest.mark.parametrize("version", ["v20.18.0", "v22.6.0", "unknown"])
def test_installed_unsupported_node_fails(monkeypatch, tmp_path, ci_flag, version):
    """Exercise the version subprocess with an executable reporting a controlled version."""
    from tests.support.browser_runtime import require_node_runtime

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    if ci_flag:
        monkeypatch.setenv(ci_flag, "true")
    node = tmp_path / "node"
    node.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version}'\n", encoding="utf-8")
    node.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    try:
        with pytest.raises(pytest.fail.Exception, match="requires Node >= 22.7"):
            require_node_runtime()
    except pytest.skip.Exception as exc:
        pytest.fail(f"Installed unsupported Node silently skipped: {exc}")


def test_browser_ci_jobs_install_declared_node_before_tests():
    """The real CI setup must install the declared runtime before browser witnesses run."""
    root = Path(__file__).resolve().parents[2]
    action = yaml.safe_load((root / ".github/actions/setup-ci-python/action.yml").read_text())
    node_steps = [
        step
        for step in action["runs"]["steps"]
        if step.get("uses", "").startswith("actions/setup-node@")
    ]
    assert len(node_steps) == 1, "Shared CI setup must install Node"
    node_step = node_steps[0]
    assert node_step["with"]["node-version-file"] == ".nvmrc"
    assert action["inputs"].get("node", {}).get("default") == "false"
    assert node_step.get("if") == "${{ inputs.node == 'true' }}"
    assert not node_step.get("continue-on-error")
    version = tuple(int(part) for part in (root / ".nvmrc").read_text().strip().split("."))
    assert len(version) == 3 and version >= (22, 7, 0)


ROOT = Path(__file__).resolve().parents[2]
SETUP_CALLS = [
    (path.name, job_name, job)
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml"))
    for job_name, job in yaml.safe_load(path.read_text()).get("jobs", {}).items()
    if any(step.get("uses") == "./.github/actions/setup-ci-python" for step in job.get("steps", []))
]


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "job"),
    SETUP_CALLS,
    ids=[f"{workflow}:{name}" for workflow, name, _ in SETUP_CALLS],
)
def test_ci_node_opt_in_matches_browser_test_execution(workflow_name, job_name, job):
    """Every real action caller opts in exactly when its commands can run browser tests."""
    # These routes run the suite or changed tests, including browser witnesses.
    browser_routes = (
        "scripts/dev/ci_driver.sh test",
        "scripts/dev/run_tests_parallel.sh",
        "scripts/dev/run_xdist_race_validation.sh",
        "scripts/validation/check_new_tests_fail_on_base.py",
        "tests/render",
        "tests/test_review_workbench.py",
    )
    steps = job["steps"]
    test_indexes = [
        i
        for i, step in enumerate(steps)
        if any(route in step.get("run", "") for route in browser_routes)
    ]
    setup_indexes = [
        i for i, step in enumerate(steps) if step.get("uses") == "./.github/actions/setup-ci-python"
    ]
    for index in setup_indexes:
        setup = steps[index]
        opted_in = setup.get("with", {}).get("node", "false")
        expected = "true" if test_indexes else "false"
        assert opted_in == expected, f"{workflow_name}:{job_name} requires node={expected}"
        if test_indexes:
            assert index < min(test_indexes)
            assert not setup.get("if") and not setup.get("continue-on-error")
