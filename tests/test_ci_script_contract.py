"""Guard CI shell wrappers against drift in shared script contracts."""

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_DRIVER = ROOT / "scripts" / "dev" / "ci_driver.sh"
PYPROJECT = ROOT / "pyproject.toml"
RUN_TESTS_PARALLEL = ROOT / "scripts" / "dev" / "run_tests_parallel.sh"
RUN_CI_LOCAL = ROOT / "scripts" / "dev" / "run_ci_local.sh"
PR_READY_CHECK = ROOT / "scripts" / "dev" / "pr_ready_check.sh"


def test_ci_driver_smoke_uses_runtime_schema_and_output_matrix_path() -> None:
    """Keep smoke preflight aligned with the runtime benchmark invocation."""

    script_text = CI_DRIVER.read_text(encoding="utf-8")

    assert 'local schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json"' in script_text
    assert '--schema "$schema_path"' in script_text
    assert 'local matrix_path="output/benchmarks/ci_smoke/matrix.yaml"' in script_text
    assert 'cat > "$matrix_path"' in script_text
    assert '--matrix "$matrix_path"' in script_text
    assert "cat > matrix.yaml" not in script_text


def test_ci_driver_test_phase_uses_shared_parallel_test_wrapper() -> None:
    """Preserve the shared pytest wrapper and default testpaths in the CI driver."""

    script_text = CI_DRIVER.read_text(encoding="utf-8")
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    testpaths = pyproject["tool"]["pytest"]["ini_options"]["testpaths"]

    assert '"$SCRIPT_DIR/run_tests_parallel.sh"' in script_text
    assert '"$SCRIPT_DIR/run_tests_parallel.sh" tests' not in script_text
    assert "tests" in testpaths
    assert "fast-pysf/tests" in testpaths
    assert "uv run pytest -q -n auto --max-worker-restart=0" not in script_text


def test_run_tests_parallel_exposes_xdist_distribution_mode() -> None:
    """Keep test scheduling configurable without changing the collected test paths."""

    script_text = RUN_TESTS_PARALLEL.read_text(encoding="utf-8")

    assert 'dist_mode="${PYTEST_XDIST_DIST:-load}"' in script_text
    assert "Invalid PYTEST_XDIST_DIST value" in script_text
    assert 'cmd=(uv run pytest -n "$worker_spec" --dist "$dist_mode")' in script_text
    assert "PYTEST_XDIST_DIST=load|worksteal|loadscope|loadfile|loadgroup" in script_text


def test_run_tests_parallel_validates_dist_mode_before_resolving_workers() -> None:
    """Invalid dist mode must fail before resolve_pytest_workers.py is invoked."""

    script_text = RUN_TESTS_PARALLEL.read_text(encoding="utf-8")

    dist_validation = "Invalid PYTEST_XDIST_DIST value"
    worker_resolution = 'uv run python "$SCRIPT_DIR/resolve_pytest_workers.py"'

    assert dist_validation in script_text
    assert worker_resolution in script_text
    assert script_text.find(dist_validation) < script_text.find(worker_resolution)


def test_run_tests_parallel_invalid_dist_fails_before_worker_resolution() -> None:
    """Invalid dist mode should exit before validating or resolving worker count."""

    env = {
        **os.environ,
        "PYTEST_XDIST_DIST": "invalid-mode",
        "PYTEST_NUM_WORKERS": "definitely-not-a-worker-count",
    }
    result = subprocess.run(
        [str(RUN_TESTS_PARALLEL), "tests/test_ci_script_contract.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert (
        "Invalid PYTEST_XDIST_DIST value 'invalid-mode' "
        "(expected load|worksteal|loadscope|loadfile|loadgroup)."
    ) in result.stderr
    assert "Resolved pytest-xdist workers" not in result.stderr
    assert "resolve_pytest_workers.py" not in result.stderr


def test_ci_driver_typecheck_phase_is_explicitly_advisory() -> None:
    """Typecheck phase should report findings without becoming a merge gate."""

    script_text = CI_DRIVER.read_text(encoding="utf-8")

    assert "Ty type check (advisory; reports findings but exits zero)" in script_text
    assert "Running ty in advisory mode (--exit-zero)" in script_text
    assert "findings are reported but do not fail this phase" in script_text
    assert "uvx ty check . --exit-zero" in script_text


def test_run_ci_local_loads_default_phases_from_ci_driver() -> None:
    """Avoid duplicating the canonical CI phase list in the local wrapper."""

    script_text = RUN_CI_LOCAL.read_text(encoding="utf-8")

    assert "scripts/dev/ci_driver.sh --list-phases" in script_text
    assert 'mapfile -t default_phases < <("$SCRIPT_DIR/ci_driver.sh" --list-phases)' in script_text
    assert "mapfile -t phases < <(load_default_phases)" in script_text
    assert 'phases=("lint" "typecheck" "test" "smoke" "artifact-policy")' not in script_text


def test_run_ci_local_exposes_fast_repeat_mode_with_timed_setup() -> None:
    """Local CI repeats should be able to skip setup after dependencies are current."""

    script_text = RUN_CI_LOCAL.read_text(encoding="utf-8")
    normalized_script = " ".join(script_text.replace("\\\n", " ").split())

    assert "--no-setup" in script_text
    assert 'run_setup="1"' in script_text
    assert 'run_setup="0"' in script_text
    assert 'if [[ "$run_setup" == "1" ]]' in script_text
    assert (
        'bash "$SCRIPT_DIR/ci_step_timer.sh" "Sync dependencies (locked)" '
        "uv sync --all-extras --frozen"
    ) in normalized_script
    assert (
        'bash "$SCRIPT_DIR/ci_step_timer.sh" "Migrate legacy artifacts into canonical root" '
        "uv run python scripts/tools/migrate_artifacts.py"
    ) in normalized_script


def test_pr_ready_check_records_freshness_after_successful_gates() -> None:
    """The PR-ready wrapper should record the freshness stamp its consumers require.

    `gh-pr-opener` treats a matching readiness stamp as the handoff proof, so a
    successful `pr_ready_check.sh` run should write that stamp without requiring
    a separate manual command.
    """

    script_text = PR_READY_CHECK.read_text(encoding="utf-8")

    expected_gates = [
        '"$SCRIPT_DIR/ruff_fix_format.sh"',
        '"$SCRIPT_DIR/run_tests_parallel.sh"',
        '"$SCRIPT_DIR/check_changed_coverage.sh"',
        '"$SCRIPT_DIR/check_docstring_todos_diff.sh"',
        '"$SCRIPT_DIR/check_docstring_todos_ratchet.sh"',
    ]
    for gate in expected_gates:
        assert gate in script_text

    freshness_call = 'uv run python "$SCRIPT_DIR/pr_ready_freshness.py" write'
    assert freshness_call in script_text
    assert script_text.rfind(freshness_call) > max(
        script_text.rfind(gate) for gate in expected_gates
    )
