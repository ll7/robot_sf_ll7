"""Guard the shared CI phase contract across the shell driver and workflow."""

from __future__ import annotations

import re
import shlex
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
CI_DRIVER = ROOT / "scripts" / "dev" / "ci_driver.sh"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
WHEEL_INSTALL_SMOKE = ROOT / "scripts" / "validation" / "wheel_install_smoke.sh"
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
CI_JOB_TIMEOUTS = {
    "fast-feedback": 45,
    "smoke-artifacts": 30,
    "xdist-scratch-isolation": 15,
    "wheel-smoke-install": 20,
    "examples-smoke": 30,
    "ci": 5,
}
PHASE_PATTERN = re.compile(
    r"(?:^|\s)(?:\./)?scripts/dev/ci_driver\.sh(?P<args>(?:\s+--?[a-z0-9_-]+|\s+[a-z0-9_-]+)*)",
    re.MULTILINE,
)
USES_PATTERN = re.compile(r"^\s*(?:-\s+)?uses:\s+(?P<value>\S+)(?:\s+#\s*(?P<comment>\S+))?\s*$")
PINNED_ACTION_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
READABLE_ACTION_TAG_PATTERN = re.compile(r"^v[0-9][A-Za-z0-9_.-]*$")


def _driver_phases() -> set[str]:
    """Return phases advertised by the shell CI driver."""
    result = subprocess.run(
        [str(CI_DRIVER), "--list-phases"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _extract_workflow_phases(run_text: str) -> set[str]:
    """Return all ci_driver phase arguments referenced in a workflow run block."""

    normalized = re.sub(r"\\\s*\n", " ", run_text)
    referenced_phases: set[str] = set()
    for match in PHASE_PATTERN.finditer(normalized):
        args = match.group("args")
        if not args or not args.strip():
            continue
        for token in shlex.split(args, comments=True, posix=True):
            if token and not token.startswith("-"):
                referenced_phases.add(token)
    return referenced_phases


def _workflow_phases() -> set[str]:
    """Return CI driver phases referenced by the GitHub Actions workflow."""
    workflow = yaml.safe_load(_workflow_text()) or {}

    referenced_phases: set[str] = set()
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            run_text = step.get("run")
            if not run_text:
                continue
            referenced_phases.update(_extract_workflow_phases(run_text))

    return referenced_phases


def _workflow_job_phases(job_name: str) -> set[str]:
    """Return CI driver phases referenced by a specific workflow job."""
    workflow = yaml.safe_load(_workflow_text())
    steps = workflow["jobs"][job_name].get("steps", [])

    referenced_phases: set[str] = set()
    for step in steps:
        run_text = step.get("run")
        if not run_text:
            continue
        referenced_phases.update(_extract_workflow_phases(run_text))
    return referenced_phases


def _workflow_text() -> str:
    """Return the raw CI workflow text."""
    return CI_WORKFLOW.read_text(encoding="utf-8")


def _workflow_files() -> list[Path]:
    """Return tracked GitHub Actions workflow YAML files."""
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def _action_ref_failures_for_line(workflow_file: Path, line_number: int, line: str) -> list[str]:
    """Return action-ref pinning failures for one workflow line."""
    match = USES_PATTERN.match(line)
    if not match:
        return [f"{workflow_file.relative_to(ROOT)}:{line_number}: malformed uses line"]

    value = match.group("value")
    if value.startswith("./"):
        return []

    failures: list[str] = []
    action, separator, ref = value.partition("@")
    if not separator:
        return [f"{workflow_file.relative_to(ROOT)}:{line_number}: missing action ref"]
    if not PINNED_ACTION_SHA_PATTERN.fullmatch(ref):
        failures.append(
            f"{workflow_file.relative_to(ROOT)}:{line_number}: {action}@{ref} is not pinned"
        )

    comment = match.group("comment")
    if comment is None or not READABLE_ACTION_TAG_PATTERN.fullmatch(comment):
        failures.append(
            f"{workflow_file.relative_to(ROOT)}:{line_number}: missing readable version comment"
        )
    return failures


def _workflow_jobs() -> dict[str, Any]:
    """Return the parsed CI workflow jobs mapping."""
    workflow = yaml.safe_load(_workflow_text()) or {}
    return workflow.get("jobs", {})


def _pyproject() -> dict[str, Any]:
    """Return parsed project metadata."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_extract_workflow_phases_handles_multiple_phase_args() -> None:
    """Capture all phase arguments from one workflow run stanza."""

    run_text = """
    scripts/dev/ci_driver.sh lint   test
    ./scripts/dev/ci_driver.sh   --verbose smoke artifact-policy
    """

    assert _extract_workflow_phases(run_text) == {
        "lint",
        "test",
        "smoke",
        "artifact-policy",
    }


def test_extract_workflow_phases_handles_line_continuations() -> None:
    """Capture phases when a workflow command uses shell line continuations."""

    run_text = """
    scripts/dev/ci_driver.sh \
      lint \
      typecheck
    """

    assert _extract_workflow_phases(run_text) == {"lint", "typecheck"}


def test_ci_workflow_only_references_known_driver_phases() -> None:
    """Fail if the workflow drifts to a non-existent shared CI phase.

    This matters because issue 533 makes the shell driver the canonical source
    for CI validation phase names, so the workflow must only reference phases
    the driver advertises.
    """

    driver_phases = _driver_phases()
    workflow_phases = _workflow_phases()

    assert workflow_phases, "workflow no longer references scripts/dev/ci_driver.sh"
    assert workflow_phases <= driver_phases, (
        f"unknown workflow phases: {sorted(workflow_phases - driver_phases)}"
    )


def test_ci_workflow_splits_fast_feedback_from_smoke_artifacts() -> None:
    """Keep fast PR feedback independent from the heavier smoke/artifact lane."""
    workflow = yaml.safe_load(_workflow_text())

    assert {"fast-feedback", "smoke-artifacts"} <= set(workflow["jobs"])
    assert _workflow_job_phases("fast-feedback") == {"lint", "typecheck", "test"}
    assert _workflow_job_phases("smoke-artifacts") == {"smoke", "artifact-policy"}
    assert "needs" not in workflow["jobs"]["fast-feedback"]


def test_ci_workflow_examples_smoke_is_independent_and_required_by_aggregate() -> None:
    """Keep example smoke tests in a separately timed job required by aggregate CI."""
    workflow = yaml.safe_load(_workflow_text())

    assert "examples-smoke" in workflow["jobs"]
    assert _workflow_job_phases("examples-smoke") == {"examples-smoke"}
    assert "needs" not in workflow["jobs"]["examples-smoke"]
    assert "examples-smoke" in workflow["jobs"]["ci"]["needs"]


def test_ci_workflow_wheel_smoke_is_independent_and_required_by_aggregate() -> None:
    """Keep wheel install smoke in a separately timed job required by aggregate CI."""
    workflow = yaml.safe_load(_workflow_text())
    wheel_smoke_steps = workflow["jobs"]["wheel-smoke-install"]["steps"]
    wheel_smoke_run_blocks = [step.get("run", "") for step in wheel_smoke_steps]

    assert "wheel-smoke-install" in workflow["jobs"]
    assert any("uv build" in run_block for run_block in wheel_smoke_run_blocks)
    assert any("wheel_install_smoke.sh" in run_block for run_block in wheel_smoke_run_blocks)
    assert "needs" not in workflow["jobs"]["wheel-smoke-install"]
    assert "wheel-smoke-install" in workflow["jobs"]["ci"]["needs"]


def test_wheel_install_smoke_uses_dependency_resolution_and_runtime_env_step() -> None:
    """Wheel smoke should catch missing package metadata and core runtime dependencies."""
    smoke_text = WHEEL_INSTALL_SMOKE.read_text(encoding="utf-8")

    assert '"${PIP_BIN}" install --no-cache-dir "${WHEEL_PATH}"' in smoke_text
    assert "--no-deps" not in smoke_text
    assert "wheel_with_dependency_resolution" in smoke_text
    assert "make_crowd_sim_env" in smoke_text
    assert "env.reset(seed=123)" in smoke_text
    assert "env.step()" in smoke_text
    assert "PYTHONPATH= PYTHONNOUSERSITE=1" in smoke_text


def test_wheel_install_smoke_tests_optional_extras_independently() -> None:
    """Optional extras should be installable one at a time from the built wheel."""
    smoke_text = WHEEL_INSTALL_SMOKE.read_text(encoding="utf-8")

    assert "ROBOT_SF_WHEEL_INSTALL_SMOKE_EXTRAS" in smoke_text
    assert "progress analysis analytics viz" in smoke_text
    assert '"${extra_pip}" install --no-cache-dir "${WHEEL_PATH}[${extra}]"' in smoke_text
    assert '"extras": json.loads(extras_status_json)' in smoke_text


def test_wheel_metadata_vendors_compatible_fast_pysf_package() -> None:
    """Clean wheel installs must not resolve the incompatible PyPI pysocialforce package."""
    project = _pyproject()
    dependency_names = {
        Requirement(dependency).name for dependency in project["project"]["dependencies"]
    }
    wheel_force_include = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    sdist_includes = project["tool"]["hatchling"]["build"]["targets"]["sdist"]["include"]

    assert "pysocialforce" not in dependency_names
    assert wheel_force_include["fast-pysf/pysocialforce"] == "pysocialforce"
    assert "/fast-pysf/pysocialforce" in sdist_includes


def test_ci_driver_test_phase_excludes_separately_timed_examples() -> None:
    """Keep example smoke timing out of the fast-feedback pytest phase."""
    driver_text = CI_DRIVER.read_text(encoding="utf-8")

    assert '"$SCRIPT_DIR/run_tests_parallel.sh" --ignore=tests/examples' in driver_text
    assert "uv run python scripts/validation/run_examples_smoke.py --skip-perf-tests" in driver_text


def test_ci_workflow_does_not_download_apt_fast_at_runtime() -> None:
    """Avoid bespoke apt-fast bootstrap in hosted CI."""
    workflow_text = _workflow_text()

    assert "apt-fast" not in workflow_text
    assert "raw.githubusercontent.com/ilikenwf/apt-fast" not in workflow_text


def test_workflow_action_refs_are_pinned_with_readable_version_comments() -> None:
    """Reject mutable action refs while keeping the intended upstream version visible."""
    failures: list[str] = []

    for workflow_file in _workflow_files():
        for line_number, line in enumerate(
            workflow_file.read_text(encoding="utf-8").splitlines(), 1
        ):
            if "uses:" not in line:
                continue
            failures.extend(_action_ref_failures_for_line(workflow_file, line_number, line))

    assert not failures, "\n".join(failures)


def test_action_ref_parser_handles_list_items_and_local_actions() -> None:
    """Parse workflow action refs without false failures for YAML lists or local actions."""
    workflow_file = WORKFLOWS_DIR / "ci.yml"
    pinned_sha = "34e114876b0b11c390a56381ad16ebd13914f8d5"

    assert (
        _action_ref_failures_for_line(
            workflow_file,
            1,
            f"      - uses: actions/checkout@{pinned_sha} # v4",
        )
        == []
    )
    assert (
        _action_ref_failures_for_line(workflow_file, 2, "      - uses: ./.github/actions/cache")
        == []
    )


def test_ci_workflow_jobs_have_explicit_timeout_bounds() -> None:
    """Keep every CI job bounded against hung GitHub Actions runs."""
    jobs = _workflow_jobs()

    assert set(CI_JOB_TIMEOUTS) == set(jobs)
    for job_name, expected_timeout in CI_JOB_TIMEOUTS.items():
        assert jobs[job_name].get("timeout-minutes") == expected_timeout


def test_ci_driver_passes_shell_syntax() -> None:
    """The shared CI driver must be parseable by bash."""
    result = subprocess.run(
        ["bash", "-n", str(CI_DRIVER)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_ci_driver_artifact_policy_uses_no_sync() -> None:
    """artifact-policy must not re-enter dependency sync after an earlier sync failure.

    If the initial ``uv sync`` step hangs or fails, the always-run artifact-policy
    cleanup step would otherwise trigger another sync and hang the job.  Using
    ``uv run --no-sync`` keeps the guard bounded.
    """
    driver_text = CI_DRIVER.read_text(encoding="utf-8")
    assert "uv run --no-sync python scripts/tools/check_artifact_root.py" in driver_text
