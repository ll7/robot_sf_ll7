"""Guard the shared CI phase contract across the shell driver and workflow."""

from __future__ import annotations

import os
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
CI_SETUP_ACTION = ROOT / ".github" / "actions" / "setup-ci-python" / "action.yml"
CODEQL_WORKFLOW = ROOT / ".github" / "workflows" / "codeql.yml"
WHEEL_INSTALL_SMOKE = ROOT / "scripts" / "validation" / "wheel_install_smoke.sh"
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
CI_JOB_TIMEOUTS = {
    "fast-feedback": 45,
    "coverage-gate": 10,
    "compat-matrix": 30,
    "fast-pysf-compat": 10,
    "smoke-artifacts": 30,
    "reproducibility-check": 10,
    "xdist-scratch-isolation": 15,
    "wheel-smoke-install": 20,
    "examples-smoke": 30,
    "notebooks-smoke": 30,
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


def test_main_push_workflows_queue_while_pull_request_runs_supersede() -> None:
    """Keep post-merge validation from being starved while retaining fast PR feedback.

    GitHub Actions groups all runs for ``refs/heads/main`` together.  The explicit
    expression queues those main runs, but still cancels superseded pull-request runs.
    """

    expected = "${{ github.ref != 'refs/heads/main' }}"
    for workflow_file in (CI_WORKFLOW, CODEQL_WORKFLOW):
        workflow = yaml.safe_load(workflow_file.read_text(encoding="utf-8")) or {}
        concurrency = workflow.get("concurrency", {})
        assert isinstance(concurrency, dict)
        assert concurrency.get("cancel-in-progress") == expected


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


def test_torch_213_constraint_is_shared_by_training_and_gpu_extras() -> None:
    """Keep both Torch installation paths on the tested 2.13 release line (issue #5556)."""
    project = _pyproject()["project"]
    optional_dependencies = project["optional-dependencies"]
    training_requirement = next(
        requirement
        for value in optional_dependencies["training"]
        if (requirement := Requirement(value)).name == "torch"
    )
    gpu_requirement = next(
        requirement
        for value in optional_dependencies["gpu"]
        if (requirement := Requirement(value)).name == "torch"
    )
    expected = Requirement("torch>=2.13.0,<2.14.0").specifier

    assert training_requirement.specifier == expected
    assert gpu_requirement.specifier == expected
    assert gpu_requirement.extras == {"cuda"}

    lock_data = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_project = next(item for item in lock_data["package"] if item["name"] == "robot-sf")
    locked_metadata = locked_project.get("metadata", {})
    assert isinstance(locked_metadata, dict)
    requires_dist = locked_metadata.get("requires-dist", [])
    assert isinstance(requires_dist, list)
    locked_torch = {
        (item.get("marker"), tuple(item.get("extras", []))): item
        for item in requires_dist
        if isinstance(item, dict) and item.get("name") == "torch"
    }

    training_lock = locked_torch.get(("extra == 'training'", ()))
    gpu_lock = locked_torch.get(("extra == 'gpu'", ("cuda",)))
    assert training_lock is not None, "uv.lock is missing the training Torch entry"
    assert gpu_lock is not None, "uv.lock is missing the gpu Torch entry"
    assert training_lock.get("specifier") == ">=2.13.0,<2.14.0"
    assert gpu_lock.get("specifier") == ">=2.13.0,<2.14.0"


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


def test_ci_workflow_combines_sharded_main_coverage_before_enforcing_floor() -> None:
    """Keep main fast by combining complete coverage from four full-suite shards."""
    workflow = yaml.safe_load(_workflow_text())
    fast_feedback = workflow["jobs"]["fast-feedback"]
    coverage_gate = workflow["jobs"]["coverage-gate"]
    fast_feedback_steps = fast_feedback["steps"]
    coverage_steps = coverage_gate["steps"]
    upload_step = next(
        step for step in fast_feedback_steps if step.get("name") == "Upload coverage shard"
    )
    floor_step = next(
        step for step in coverage_steps if step.get("name") == "Enforce absolute coverage floor"
    )
    baseline_step = next(
        step for step in coverage_steps if step.get("name") == "Compare coverage with baseline"
    )
    combine_step = next(
        step for step in coverage_steps if step.get("name") == "Combine coverage shards"
    )

    assert fast_feedback["strategy"]["matrix"]["shard"] == [1, 2, 3, 4]
    assert fast_feedback["env"]["PYTEST_SHARD_COUNT"] == 4
    assert (
        "github.event_name != 'pull_request'" in fast_feedback["env"]["ROBOT_SF_SHARD_INCLUDE_SLOW"]
    )
    assert "github.event_name != 'pull_request'" in fast_feedback["env"]["ROBOT_SF_PYTEST_COVERAGE"]
    assert "matrix.shard" in fast_feedback["env"]["COVERAGE_FILE"]
    assert upload_step["if"] == "${{ github.event_name != 'pull_request' }}"
    assert upload_step["with"]["include-hidden-files"] is True
    assert coverage_gate["needs"] == "fast-feedback"
    assert "github.event_name != 'pull_request'" in coverage_gate["if"]
    assert "coverage combine output/coverage" in combine_step["run"]
    assert "coverage json" in combine_step["run"]
    assert "coverage html" in combine_step["run"]
    assert "if" not in floor_step
    assert "continue-on-error" not in floor_step
    assert "--minimum-total 85.0" in floor_step["run"]
    assert "--absolute-only" in floor_step["run"]
    assert "--current output/coverage/coverage.json" in floor_step["run"]
    assert baseline_step["continue-on-error"] is True
    assert "--threshold 1.0" in baseline_step["run"]
    assert "--fail-on-decrease" not in baseline_step["run"]
    assert "coverage-gate" in workflow["jobs"]["ci"]["needs"]


def test_parallel_test_driver_supports_full_sharded_coverage() -> None:
    """Require explicit slow-test and coverage controls for main's sharded pass."""
    script_text = (ROOT / "scripts" / "dev" / "run_tests_parallel.sh").read_text(encoding="utf-8")

    assert "ROBOT_SF_SHARD_INCLUDE_SLOW" in script_text
    assert '"$include_slow" != "1"' in script_text
    assert '[[ "$coverage_enabled" == "1" ]]' in script_text
    assert '[[ -z "${COVERAGE_FILE:-}" ]]' in script_text
    assert 'cmd+=("--cov=robot_sf" "--cov-report=")' in script_text


def test_ci_workflow_requires_the_proven_core_compatibility_matrix() -> None:
    """Gate aggregate CI on the proven Linux/macOS and Python compatibility matrix."""
    workflow = yaml.safe_load(_workflow_text())
    compat_matrix = workflow["jobs"]["compat-matrix"]
    steps = compat_matrix.get("steps", [])
    setup_step = next(
        (step for step in steps if step.get("uses") == "./.github/actions/setup-ci-python"),
        None,
    )
    assert setup_step is not None, "setup-ci-python step not found in compat-matrix job"
    test_steps = [step.get("run", "") for step in steps]

    assert "continue-on-error" not in compat_matrix
    assert compat_matrix["strategy"]["fail-fast"] is False
    assert compat_matrix["strategy"]["matrix"] == {
        "os": ["ubuntu-latest", "macos-latest"],
        "python": ["3.11", "3.13"],
    }
    assert setup_step["with"] == {
        "python-version": "${{ matrix.python }}",
        "sync-args": "--frozen",
    }
    assert any(
        "pytest tests/common tests/contract tests/factories tests/gym_env tests/maps" in step
        and "tests/nav tests/ped_npc tests/render tests/scenarios" in step
        and "tests/sensor tests/sim tests/unit" in step
        and '-q -m "not slow"' in step
        for step in test_steps
    )
    aggregate = workflow["jobs"]["ci"]
    assert "compat-matrix" in aggregate["needs"]
    aggregate_steps = [step.get("run", "") for step in aggregate.get("steps", [])]
    assert any('needs.compat-matrix.result }}" != "success"' in step for step in aggregate_steps)


def test_ci_setup_action_supports_core_matrix_dependencies_on_macos() -> None:
    """Keep the shared setup action usable by the backend-light macOS matrix."""
    action = yaml.safe_load(CI_SETUP_ACTION.read_text(encoding="utf-8"))
    steps = action.get("runs", {}).get("steps", [])
    system_packages_step = next(
        (step for step in steps if step.get("name") == "System packages for headless"),
        None,
    )
    sync_step = next(
        (step for step in steps if step.get("name", "").startswith("Sync dependencies")),
        None,
    )

    assert system_packages_step is not None, "System packages step not found"
    assert sync_step is not None, "Sync dependencies step not found"
    assert action["inputs"]["sync-args"]["default"] == "--all-extras --frozen"
    assert system_packages_step["if"] == "runner.os == 'Linux'"
    assert sync_step["env"]["CI_STEP_TIMEOUT_SECONDS"] == (
        "${{ runner.os == 'Linux' && '1200' || '' }}"
    )
    assert "${{ inputs.sync-args }}" in sync_step["run"]


def test_ci_workflow_examples_smoke_is_independent_and_required_by_aggregate() -> None:
    """Keep example smoke tests in a separately timed job required by aggregate CI."""
    workflow = yaml.safe_load(_workflow_text())

    assert "examples-smoke" in workflow["jobs"]
    assert _workflow_job_phases("examples-smoke") == {"examples-smoke"}
    assert "needs" not in workflow["jobs"]["examples-smoke"]
    assert "examples-smoke" in workflow["jobs"]["ci"]["needs"]


def test_ci_workflow_notebooks_smoke_is_independent_and_required_by_aggregate() -> None:
    """Keep notebook execution in a separately timed job required by aggregate CI."""
    workflow = yaml.safe_load(_workflow_text())

    assert "notebooks-smoke" in workflow["jobs"]
    assert _workflow_job_phases("notebooks-smoke") == {"notebooks-smoke"}
    assert "needs" not in workflow["jobs"]["notebooks-smoke"]
    assert "notebooks-smoke" in workflow["jobs"]["ci"]["needs"]


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


def test_ci_driver_lint_reports_all_failures_without_short_circuiting(tmp_path: Path) -> None:
    """The lint phase must enumerate every failure, not stop at the first (issue #5960).

    Reproduces the exact bug class from the issue: a branch with BOTH a format
    violation (ruff format) and a new broad exception must report BOTH in one run.
    We stub ``uv``/``python`` so the real ``run_lint_phase`` logic in ci_driver.sh
    executes without a Python environment, then assert that all failing checks ran
    and the phase still exits non-zero.
    """

    # Stub `uv` so `uv run <tool> <args>` proxies to the local interpreter (or a
    # failing stub). We map each known lint check to a deterministic pass/fail so we
    # can force BOTH ruff format AND broad-exception failures simultaneously.
    stub_dir = tmp_path / "ci_driver_lint_stub"
    stub_dir.mkdir(parents=True, exist_ok=True)
    uv_stub = stub_dir / "uv"
    python_stub = stub_dir / "python"

    # `uv run` forwards to our python stub. We detect which check is running via the
    # args so we can make the broad-exception check fail.
    uv_stub.write_text(
        '#!/usr/bin/env bash\nset -- "${@:3}"\nexec "' + str(python_stub) + '" "$@"\n',
        encoding="utf-8",
    )
    python_stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "# ruff format --check -> fail (format violation)\n"
        "if 'format' in sys.argv and '--check' in sys.argv:\n"
        "    sys.stderr.write('would reformat file.py\\n')\n"
        "    sys.exit(1)\n"
        "# check_broad_exceptions -> fail (new broad exception)\n"
        "if 'check_broad_exceptions' in sys.argv[0] or sys.argv[-1].endswith('check_broad_exceptions.py'):\n"
        "    sys.stderr.write('broad exception ratchet increased\\n')\n"
        "    sys.exit(1)\n"
        "# ruff check -> pass; version alignment -> pass\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    uv_stub.chmod(0o755)
    python_stub.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}:{env.get('PATH', '')}"
    env["CI_DRIVER_GITHUB_REF"] = "refs/heads/feature"
    env["CI_DRIVER_EVENT_NAME"] = "pull_request"

    result = subprocess.run(
        [str(CI_DRIVER), "lint"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        env=env,
    )

    combined = result.stdout + result.stderr
    # Both independent failures must appear in a single run.
    assert "would reformat" in combined, f"format failure not surfaced:\n{combined}"
    assert "broad exception ratchet increased" in combined, (
        f"broad-exception failure not surfaced:\n{combined}"
    )
    # The phase must still fail overall.
    assert result.returncode != 0, f"lint phase exited zero despite failures:\n{combined}"


def test_ci_driver_lint_passes_when_all_checks_pass(tmp_path: Path) -> None:
    """The lint phase must exit zero when every check passes (issue #5960)."""

    stub_dir = tmp_path / "ci_driver_lint_stub_pass"
    stub_dir.mkdir(parents=True, exist_ok=True)
    uv_stub = stub_dir / "uv"
    python_stub = stub_dir / "python"

    uv_stub.write_text(
        '#!/usr/bin/env bash\nset -- "${@:3}"\nexec "' + str(python_stub) + '" "$@"\n',
        encoding="utf-8",
    )
    python_stub.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n", encoding="utf-8")
    uv_stub.chmod(0o755)
    python_stub.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}:{env.get('PATH', '')}"
    env["CI_DRIVER_GITHUB_REF"] = "refs/heads/main"
    env["CI_DRIVER_EVENT_NAME"] = "push"

    result = subprocess.run(
        [str(CI_DRIVER), "lint"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
