"""Deterministic admission and cleanup at the browser and child boundaries."""

import ast
import os
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
    from tests.dev.test_pr_ready_preflight import _wait_for_process_exit
    from tests.support.subprocess_events import capture_completion

    result = capture_completion([sys.executable, "-c", 'print("ready")'])
    assert result.returncode == 0 and result.stdout.strip() == "ready"
    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        capture_completion(["sh", "-c", "sleep 120 & echo $!; wait"], hang_guard_seconds=1)
    grandchild_pid = int(exc_info.value.output.strip())
    _wait_for_process_exit(grandchild_pid)


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
BROWSER_WITNESSES = (
    "tests/render/test_audit_workbench.py::test_browser_controller_runtime_covers_normal_and_missing_media_cases",
    "tests/render/test_audit_workbench.py::test_browser_controller_runtime_failure_reports_redacted_bounded_diagnostics",
    "tests/render/test_review_editor.py::test_python_generated_model_can_save_storyboard_in_browser_runtime",
    "tests/render/test_review_editor.py::test_node_browser_controller_honours_typing_shortcut_suppression",
    "tests/render/test_review_sessions.py::test_node_browser_runtime_is_offline_and_has_no_implicit_start",
    "tests/test_review_workbench.py::test_canonical_launch_mounts_diagnostic_audit_extension_and_browser_runtime",
)
SETUP_CALLS = [
    (path.name, job_name, job)
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml"))
    for job_name, job in yaml.safe_load(path.read_text()).get("jobs", {}).items()
    if any(step.get("uses") == "./.github/actions/setup-ci-python" for step in job.get("steps", []))
]


@pytest.mark.parametrize(
    ("owner_result", "run_full_ci", "cancelled", "expected"),
    [
        ("success", "true", False, True),
        ("success", "false", False, False),
        ("failure", "true", False, False),
        ("success", "true", True, False),
    ],
)
def test_browser_witnesses_require_successful_uncancelled_owner(
    owner_result, run_full_ci, cancelled, expected
):
    """A stale positive output or cancellation cannot admit hosted browser work."""
    from tests.ci.test_self_hosted_routing import _evaluate_node

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    expression = workflow["jobs"]["browser-witnesses"]["if"]
    expression = expression.removeprefix("${{").removesuffix("}}")
    expression = expression.replace("&&", "and").replace("!cancelled()", "not cancelled()")
    decision = _evaluate_node(
        ast.parse(
            expression.strip().replace("dispatch-ownership", "dispatch_ownership"), mode="eval"
        ),
        {
            "cancelled": cancelled,
            "needs": {
                "dispatch_ownership": {
                    "result": owner_result,
                    "outputs": {"run_full_ci": run_full_ci},
                }
            },
        },
    )
    assert decision is expected


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
    if workflow_name == "ci.yml" and job_name == "fast-feedback":
        # Shards still run Python assertions; the hosted lane owns Node witnesses.
        deselections = job.get("env", {}).get("PYTEST_ADDOPTS", "").split()
        assert set(deselections) == {f"--deselect={nodeid}" for nodeid in BROWSER_WITNESSES}
        workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
        browser_job = workflow["jobs"]["browser-witnesses"]
        assert browser_job["runs-on"] == "ubuntu-latest"
        assert browser_job["needs"] == "dispatch-ownership"
        assert browser_job["if"] == job["if"]
        assert "browser-witnesses" in workflow["jobs"]["ci"]["needs"]
        aggregate = next(
            step
            for step in workflow["jobs"]["ci"]["steps"]
            if step.get("name") == "Check split job results"
        )
        assert (
            aggregate["env"]["BROWSER_WITNESSES_RESULT"] == "${{ needs.browser-witnesses.result }}"
        )
        guard = aggregate["run"].split("python scripts/dev/check_ci_needs.py", 1)[0]
        for result in ("success", "failure", "skipped", "cancelled", ""):
            completed = subprocess.run(
                ["bash", "-e", "-c", guard],
                env={**os.environ, "BROWSER_WITNESSES_RESULT": result},
                check=False,
            )
            assert (completed.returncode == 0) == (result in {"success", "cancelled"})
        checkout = browser_job["steps"][0]["with"]
        assert "github.event.pull_request.head.sha" in checkout["ref"]
        commands = "\n".join(step.get("run", "") for step in browser_job["steps"])
        assert "uv run pytest" in commands
        assert all(nodeid in commands for nodeid in BROWSER_WITNESSES)
    test_indexes = [
        i
        for i, step in enumerate(steps)
        if any(route in step.get("run", "") for route in browser_routes)
    ]
    if workflow_name == "ci.yml" and job_name == "fast-feedback":
        test_indexes = []
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
