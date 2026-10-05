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
    assert not node_step.get("if") and not node_step.get("continue-on-error")
    version = tuple(int(part) for part in (root / ".nvmrc").read_text().strip().split("."))
    assert len(version) == 3 and version >= (22, 7, 0)
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    for job_name in ("fast-feedback", "smoke-artifacts", "new-tests-fail-on-base"):
        steps = workflow["jobs"][job_name]["steps"]
        setup_index = next(
            i
            for i, step in enumerate(steps)
            if step.get("uses") == "./.github/actions/setup-ci-python"
        )
        assert not steps[setup_index].get("if")
        assert not steps[setup_index].get("continue-on-error")
        assert any(step.get("run") for step in steps[setup_index + 1 :])
