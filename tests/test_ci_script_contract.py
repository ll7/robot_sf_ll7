"""Guard CI shell wrappers against drift in shared script contracts.

Help-behaviour contract
-----------------------
Selected ``scripts/dev/*.sh`` helpers with --help / -h usage support must
handle both forms as cheap success paths: exit 0, print usage to stdout,
and return before sourcing common_setup.sh or invoking heavy dependencies
(uv, ruff, pytest, gh, etc.).

Covered scripts (11 total):
  pr_ready_check.sh, gh_comment.sh, run_worktree_shared_venv.sh,
  run_tests_parallel.sh, run_xdist_race_validation.sh, run_ci_local.sh, local_signoff.sh,
  ci_driver.sh, check_runtime_requirements.sh, check_carla_runtime.sh,
  bootstrap_worktree.sh

Also covered (in tests/dev/): ci_step_timer.sh

Excluded by policy (SLURM/training-oriented, not general-purpose helpers):
  auxme_partition_status.sh, sbatch_*.sh

Excluded (no usage/help support at all):
  ruff_fix_format.sh, check_changed_coverage.sh, check_docstring_todos_diff.sh,
  check_docstring_todos_ratchet.sh, check_docs_proof_consistency_diff.sh,
  common_setup.sh (sourced, not invoked directly)
"""

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CI_DRIVER = ROOT / "scripts" / "dev" / "ci_driver.sh"
GH_COMMENT = ROOT / "scripts" / "dev" / "gh_comment.sh"
PYPROJECT = ROOT / "pyproject.toml"
RUN_TESTS_PARALLEL = ROOT / "scripts" / "dev" / "run_tests_parallel.sh"
RUN_XDIST_RACE_VALIDATION = ROOT / "scripts" / "dev" / "run_xdist_race_validation.sh"
RUN_CI_LOCAL = ROOT / "scripts" / "dev" / "run_ci_local.sh"
LOCAL_SIGNOFF = ROOT / "scripts" / "dev" / "local_signoff.sh"
PR_READY_CHECK = ROOT / "scripts" / "dev" / "pr_ready_check.sh"
PR_BODY_CONTRACTS_WORKFLOW = ROOT / ".github" / "workflows" / "pr-body-contracts.yml"
RUN_WORKTREE_SHARED_VENV = ROOT / "scripts" / "dev" / "run_worktree_shared_venv.sh"
BOOTSTRAP_WORKTREE = ROOT / "scripts" / "dev" / "bootstrap_worktree.sh"
CHECK_RUNTIME_REQUIREMENTS = ROOT / "scripts" / "dev" / "check_runtime_requirements.sh"
CHECK_CARLA_RUNTIME = ROOT / "scripts" / "dev" / "check_carla_runtime.sh"


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
    assert "--lane core|optional|all" in script_text
    assert "ROBOT_SF_TEST_LANE=core|optional|all" in script_text
    assert "Resolved pytest lane:" in script_text
    assert "normalize_pytest_target_path()" in script_text
    assert "${path%%::*}" in script_text
    assert "core_test_paths=(" in script_text
    assert "tests/adversarial" in script_text
    assert "tests/analysis_workbench" in script_text
    assert "tests/scenario_certification" in script_text
    assert "explicit_test_targets=(" in script_text
    assert 'cmd+=("--ignore=$optional_test_path")' in script_text
    assert 'pytest_args+=("$optional_test_path")' in script_text
    assert 'append_unique_pytest_arg "$core_test_path"' in script_text
    assert "changed_top_level_core_test_paths=()" in script_text
    assert ":(top,glob)tests/test_*.py" in script_text
    assert 'append_unique_pytest_arg "$changed_test_path"' in script_text
    assert "Core pytest lane cannot run optional-extra path" in script_text


def test_pytest_coverage_is_explicit_opt_in() -> None:
    """Default pytest runs should stay fast while the wrapper preserves coverage opt-in."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    script_text = RUN_TESTS_PARALLEL.read_text(encoding="utf-8")
    pr_ready_text = PR_READY_CHECK.read_text(encoding="utf-8")

    assert "--cov=robot_sf" not in addopts
    assert "--cov-report=html" not in addopts
    assert "--cov-report=json" not in addopts
    assert "ROBOT_SF_PYTEST_COVERAGE" in script_text
    assert "${coverage_requested,,}" not in script_text
    assert 'cmd+=("--cov=robot_sf" "--cov-report=html" "--cov-report=json")' in script_text
    assert (
        'ROBOT_SF_PYTEST_COVERAGE=1 ROBOT_SF_TEST_LANE=core "$SCRIPT_DIR/run_tests_parallel.sh" --lane core'
        in pr_ready_text
    )
    assert 'optional_pytest_addopts="${PYTEST_ADDOPTS:-}"' in pr_ready_text
    assert "--cov-append" in pr_ready_text


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


def test_run_tests_parallel_core_lane_includes_changed_top_level_core_tests(tmp_path: Path) -> None:
    """New top-level core tests must reach PR-readiness pytest collection (issue #5108)."""
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    optional_allowlist = repo / "tests" / "support" / "optional_test_allowlist.txt"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()
    optional_allowlist.parent.mkdir(parents=True)

    for script_name in ("run_tests_parallel.sh", "common_setup.sh"):
        source = ROOT / "scripts" / "dev" / script_name
        target = script_dir / script_name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)
    optional_allowlist.write_text("tests/test_optional_top_level.py\n", encoding="utf-8")

    captured_args = repo / "captured-pytest-args.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "$1" == "run" && "$2" == "python" ]]; then',
                '  printf "1\\n"',
                "  exit 0",
                "fi",
                'printf "%s\\n" "$*" > "$UV_CAPTURED_ARGS"',
            ]
        ),
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "base fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "tests" / "test_new_top_level.py").write_text(
        "def test_new(): pass\n", encoding="utf-8"
    )
    (repo / "tests" / "test_optional_top_level.py").write_text(
        "def test_optional(): pass\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "tests"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "add top-level tests"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [str(script_dir / "run_tests_parallel.sh"), "--lane", "core", "--no-ordering"],
        cwd=repo,
        env={
            **os.environ,
            "BASE_REF": "HEAD~1",
            "PYTEST_NUM_WORKERS": "1",
            "PYTEST_FAST_FAIL": "0",
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "UV_CAPTURED_ARGS": str(captured_args),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    pytest_args = captured_args.read_text(encoding="utf-8")
    assert "tests/test_new_top_level.py" in pytest_args
    assert "tests/test_optional_top_level.py" not in pytest_args


def test_xdist_race_validation_wraps_parallel_tests_and_artifact_scan() -> None:
    """The stress route should force high xdist concurrency and scan shared outputs."""

    script_text = RUN_XDIST_RACE_VALIDATION.read_text(encoding="utf-8")

    assert 'workers="${XDIST_RACE_WORKERS:-32}"' in script_text
    assert 'export PYTEST_NUM_WORKERS="$workers"' in script_text
    assert 'export PYTEST_XDIST_DIST="${PYTEST_XDIST_DIST:-worksteal}"' in script_text
    assert 'export PYTEST_FAST_FAIL="${PYTEST_FAST_FAIL:-0}"' in script_text
    assert 'export PYTEST_ORDER_MODE="${PYTEST_ORDER_MODE:-none}"' in script_text
    assert "run_compact_validation.py" in script_text
    assert '"$SCRIPT_DIR/run_tests_parallel.sh" "${pytest_args[@]}"' in script_text
    assert "check_xdist_race_artifacts.py" in script_text
    assert "--baseline-json" in script_text


def test_xdist_race_validation_rejects_invalid_worker_value() -> None:
    """Invalid stress worker counts should fail before running uv or pytest."""

    result = subprocess.run(
        [str(RUN_XDIST_RACE_VALIDATION), "--workers", "12bad", "tests/dev"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "--workers must be a positive integer or 'auto'." in result.stderr
    assert "uv run" not in result.stderr


def test_xdist_race_validation_rejects_missing_option_value() -> None:
    """Stress wrapper options should fail cleanly when the value is omitted."""

    result = subprocess.run(
        [str(RUN_XDIST_RACE_VALIDATION), "--workers"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "--workers requires a non-empty value." in result.stderr
    assert "shift" not in result.stderr


def test_ci_driver_typecheck_phase_is_explicitly_advisory() -> None:
    """Typecheck phase should report findings without becoming a merge gate."""

    script_text = CI_DRIVER.read_text(encoding="utf-8")

    assert "Ty type check (advisory; reports findings but exits zero)" in script_text
    assert "Running ty in advisory mode (--exit-zero)" in script_text
    assert "findings are reported but do not fail this phase" in script_text
    assert "uvx ty check . --exit-zero" in script_text


def test_ci_driver_test_phase_runs_benchmark_reconciliation_guard() -> None:
    """Fast-feedback must not silently skip frozen-trace reconciliation tests."""

    script_text = CI_DRIVER.read_text(encoding="utf-8")

    assert "run_fast_feedback_benchmark_reconciliation_guard()" in script_text
    assert "Running fast-feedback benchmark reconciliation guard" in script_text
    assert "tests/benchmark/test_frozen_trace_reconciliation.py" in script_text
    assert "tests/benchmark/test_safety_wrapper_ablation_manifest.py" in script_text
    assert '[[ "$shard_index" != "1" ]]' in script_text
    assert "$SCRIPT_DIR/check_event_ledger_reconciliation_guard.sh" in script_text
    assert "run_fast_feedback_benchmark_reconciliation_guard" in script_text
    assert '"$SCRIPT_DIR/run_tests_parallel.sh" --ignore=tests/examples' in script_text


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
        'uv run python "$SCRIPT_DIR/check_pr_followups.py" "${followup_args[@]}"',
        '"$SCRIPT_DIR/ruff_fix_format.sh"',
        '"$SCRIPT_DIR/run_tests_parallel.sh"',
        '"$SCRIPT_DIR/check_changed_coverage.sh"',
        '"$SCRIPT_DIR/check_docstring_todos_diff.sh"',
        '"$SCRIPT_DIR/check_docstring_todos_ratchet.sh"',
    ]
    for gate in expected_gates:
        assert gate in script_text
    assert (
        'ROBOT_SF_PYTEST_COVERAGE=1 ROBOT_SF_TEST_LANE=core "$SCRIPT_DIR/run_tests_parallel.sh" --lane core'
        in script_text
    )
    assert (
        'PYTEST_ADDOPTS="$optional_pytest_addopts" ROBOT_SF_PYTEST_COVERAGE=1 ROBOT_SF_TEST_LANE=optional "$SCRIPT_DIR/run_tests_parallel.sh" --lane optional'
        in script_text
    )
    assert "Optional-extra changed files requiring the predictive lane" in script_text
    assert "No changed files require the optional-extra lane." in script_text

    freshness_call = 'uv run python "$SCRIPT_DIR/pr_ready_freshness.py" "${freshness_args[@]}"'
    assert 'freshness_args=(write --base-ref "$BASE_REF")' in script_text
    assert freshness_call in script_text
    assert script_text.rfind(freshness_call) > max(
        script_text.rfind(gate) for gate in expected_gates
    )
    assert "followup_args=()" in script_text
    assert "followup_args+=(--require-body)" in script_text
    assert script_text.find("followup_args+=(--require-body)") < script_text.find(
        'uv run python "$SCRIPT_DIR/check_pr_followups.py" "${followup_args[@]}"'
    )


def test_pr_ready_check_exposes_final_committed_head_mode() -> None:
    """Final PR proof should fail closed on dirty trees and mark clean-tree stamps."""
    script_text = PR_READY_CHECK.read_text(encoding="utf-8")

    assert "PR_READY_MODE" in script_text
    assert "PR_READY_FINAL" in script_text
    assert ",," not in script_text
    assert "tr '[:upper:]' '[:lower:]'" in script_text
    assert "final) pr_ready_final=1" in script_text
    assert "interim) pr_ready_final=0" in script_text
    assert "Final PR readiness requires a clean non-ignored worktree" in script_text
    assert "recording interim PR readiness from a dirty non-ignored worktree" in script_text
    assert "--require-clean-tree" in script_text
    assert "pr_ready_freshness.py" in script_text


def test_pr_body_contracts_workflow_runs_strict_pr_body_checker() -> None:
    """The live PR workflow should enforce body, follow-up, and domain-review contracts."""
    workflow_text = PR_BODY_CONTRACTS_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in workflow_text
    assert "gh api --paginate" in workflow_text
    assert "pr_changed_files.txt" in workflow_text
    assert "scripts/dev/check_pr_followups.py" in workflow_text
    for flag in (
        "--github-event-path",
        "--changed-files-file",
        "--require-body",
        "--require-substantive-body",
        "--require-open-issues",
    ):
        assert flag in workflow_text


def test_pr_ready_check_final_mode_preflights_analytics_dependencies(tmp_path: Path) -> None:
    """Final PR proof should fail early when analytics extras are missing."""
    repo = tmp_path / "repo"
    stale_repo = tmp_path / "stale-repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()

    for script_name in ("pr_ready_check.sh", "common_setup.sh"):
        source = ROOT / "scripts" / "dev" / script_name
        target = script_dir / script_name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "$1" == "run" && "$2" == "python" ]]; then',
                "  echo 'duckdb, pyarrow'",
                "  exit 1",
                "fi",
                "echo 'unexpected uv invocation' >&2",
                "exit 99",
            ]
        ),
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "test fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    stale_repo.mkdir()
    subprocess.run(["git", "init"], cwd=stale_repo, check=True, capture_output=True, text=True)
    (stale_repo / "untracked-marker").write_text("stale outer checkout\n", encoding="utf-8")

    result = subprocess.run(
        [str(script_dir / "pr_ready_check.sh")],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PR_READY_MODE": "final",
            "BASE_REF": "origin/main",
            "REPO_ROOT": str(stale_repo),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "Final PR readiness requires analytics dependencies" in result.stderr
    assert "uv sync --all-extras" in result.stderr
    assert "duckdb, pyarrow" in result.stderr
    assert "ruff_fix_format" not in result.stderr


def test_pr_ready_check_help_long() -> None:
    """pr_ready_check.sh --help prints usage and exits 0."""
    result = subprocess.run(
        [str(PR_READY_CHECK), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "BASE_REF" in result.stdout
    assert "PR_READY_MODE" in result.stdout
    assert "PR_READY_FINAL" in result.stdout
    assert "PR_READY_PR_BODY_FILE" in result.stdout


def test_pr_ready_check_help_short() -> None:
    """pr_ready_check.sh -h prints usage and exits 0."""
    result = subprocess.run(
        [str(PR_READY_CHECK), "-h"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "BASE_REF" in result.stdout


def test_pr_ready_check_help_does_not_invoke_gates(tmp_path: Path) -> None:
    """--help should exit 0 before reaching heavy gate commands (uv, ruff, pytest)."""
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()

    for script_name in ("pr_ready_check.sh", "common_setup.sh"):
        source = ROOT / "scripts" / "dev" / script_name
        target = script_dir / script_name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\necho "uv should not be called for --help" >&2\nexit 99\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "test fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [str(script_dir / "pr_ready_check.sh"), "--help"],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "BASE_REF" in result.stdout
    assert "PR_READY_MODE" in result.stdout
    assert "PR_READY_FINAL" in result.stdout
    assert "PR_READY_PR_BODY_FILE" in result.stdout
    assert "uv should not be called" not in result.stderr


def test_worktree_shared_venv_helper_pins_current_checkout_imports() -> None:
    """Shared-venv validation must import from the active worktree, not the owning checkout."""
    script_text = RUN_WORKTREE_SHARED_VENV.read_text(encoding="utf-8")

    assert 'repo_root="$(git rev-parse --show-toplevel)"' in script_text
    assert 'main_repo_root="$(cd "$git_common_dir/.." && pwd)"' in script_text
    assert 'venv_path="${venv_override:-$main_repo_root/.venv}"' in script_text
    assert 'export UV_PROJECT_ENVIRONMENT="$venv_path"' in script_text
    assert "export UV_NO_SYNC=1" in script_text
    assert 'export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"' in script_text
    assert 'exec uv run "${cmd[@]}"' in script_text


def test_worktree_shared_venv_helper_isolates_linked_worktree_coverage() -> None:
    """Linked worktrees should not share the default coverage database."""
    script_text = RUN_WORKTREE_SHARED_VENV.read_text(encoding="utf-8")

    assert 'if [[ -z "${COVERAGE_FILE:-}" && "$git_common_dir" != "$repo_root/.git" ]]' in (
        script_text
    )
    assert "git hash-object --stdin" in script_text
    assert "cut -c1-12" in script_text
    assert 'export COVERAGE_FILE="$repo_root/output/coverage/.coverage.${worktree_id}"' in (
        script_text
    )


def test_worktree_shared_venv_helper_has_valid_shell_and_help() -> None:
    """The shared-venv helper should be shell-valid and document its safety boundary."""
    syntax = subprocess.run(
        ["bash", "-n", str(RUN_WORKTREE_SHARED_VENV)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = subprocess.run(
        [str(RUN_WORKTREE_SHARED_VENV), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert help_result.returncode == 0
    assert "PYTHONPATH=$PWD" in help_result.stdout
    assert "UV_PROJECT_ENVIRONMENT" in help_result.stdout
    assert "UV_NO_SYNC=1" in help_result.stdout
    assert "COVERAGE_FILE" in help_result.stdout
    assert "full local .venv" in help_result.stdout


def test_worktree_shared_venv_helper_fails_for_missing_shared_env(tmp_path: Path) -> None:
    """A missing shared env should fail before uv can fall back to an unintended checkout."""
    missing_venv = tmp_path / "missing-venv"

    result = subprocess.run(
        [str(RUN_WORKTREE_SHARED_VENV), "--venv", str(missing_venv), "--", "python", "-V"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert f"Shared virtualenv not found or incomplete: {missing_venv}" in result.stderr
    assert "Create it with 'uv sync --all-extras'" in result.stderr


def test_worktree_shared_venv_helper_reports_relative_missing_env() -> None:
    """Relative missing env paths should still use the helper's actionable error."""
    missing_venv = Path("does/not/exist")

    result = subprocess.run(
        [str(RUN_WORKTREE_SHARED_VENV), "--venv", str(missing_venv), "--", "python", "-V"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert f"Shared virtualenv not found or incomplete: {ROOT / missing_venv}" in result.stderr
    assert "cd:" not in result.stderr


def test_worktree_shared_venv_helper_has_freshness_check_wiring() -> None:
    """The shared-venv helper must guard the reused env against source drift (issue #5023)."""
    script_text = RUN_WORKTREE_SHARED_VENV.read_text(encoding="utf-8")

    # The freshness gate compares the vendored pysocialforce install against fast-pysf source.
    assert "check_shared_venv_freshness()" in script_text
    assert 'local src_pkg="$repo_root/fast-pysf/pysocialforce"' in script_text
    assert 'for candidate in "$venv"/lib/python*/site-packages/pysocialforce' in script_text
    assert 'cmp -s "$src_file" "$installed_file"' in script_text
    assert "Shared virtualenv is stale relative to this checkout" in script_text
    assert "diverging module: pysocialforce/" in script_text
    assert "uv sync --all-extras --reinstall-package robot-sf" in script_text
    # Standalone commands with a verified no-project-import boundary can skip project drift safely.
    assert "--standalone" in script_text
    assert "use --standalone for a command verified not to import project packages" in script_text
    assert 'if [[ -z "$standalone" ]]; then' in script_text
    # The gate is skippable for advanced users with a confirmed-matching env.
    assert "--no-freshness-check" in script_text
    assert "ROBOT_SF_VENV_FRESHNESS_CHECK:-" in script_text
    assert "ROBOT_SF_VENV_FRESHNESS_CHECK=skip" in script_text


def _make_freshness_fixture_repo(
    tmp_path: Path,
    *,
    installed_scene: str,
) -> tuple[Path, Path, dict[str, str]]:
    """Build a git repo + shared venv whose installed pysocialforce/scene.py is ``installed_scene``.

    The worktree source fast-pysf/pysocialforce/scene.py carries a newer API (normalize_integration_scheme)
    while the installed copy may or may not match it. A fake ``uv`` on PATH proves whether the helper
    reached the underlying command or failed earlier in the freshness gate.
    """
    repo = tmp_path / "repo"
    fake_bin = repo / "fake-bin"
    venv = repo / "shared-venv"
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    installed_pkg = site_packages / "pysocialforce"
    src_pkg = repo / "fast-pysf" / "pysocialforce"
    fake_bin.mkdir(parents=True)
    installed_pkg.mkdir(parents=True)
    src_pkg.mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)

    # Worktree source scene.py carries the newer API the helper must detect drift against.
    newer_scene = "def normalize_integration_scheme(value=None):\n    return value\n"
    (src_pkg / "scene.py").write_text(newer_scene, encoding="utf-8")
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    # Installed copy is whatever the caller passes (matching = fresh, divergent = stale).
    (installed_pkg / "scene.py").write_text(installed_scene, encoding="utf-8")
    (installed_pkg / "__init__.py").write_text("", encoding="utf-8")

    # The helper only checks venv presence via bin/python executability.
    py = venv / "bin" / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    py.chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf "uv-reached %s\\n" "$*" >&2\n'
        'printf "pythonpath=%s\\n" "${PYTHONPATH-}" >&2\nexit 7\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "freshness fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }
    env.pop("PYTHONPATH", None)
    return repo, venv, env


def test_worktree_shared_venv_freshness_check_fails_early_on_stale_env(
    tmp_path: Path,
) -> None:
    """A stale shared env (missing the newer source API) must fail before uv runs (issue #5023)."""
    repo, venv, env = _make_freshness_fixture_repo(
        tmp_path,
        installed_scene="# stale install without normalize_integration_scheme\n",
    )

    result = subprocess.run(
        [
            str(RUN_WORKTREE_SHARED_VENV),
            "--venv",
            str(venv),
            "--",
            "python",
            "-c",
            "from pysocialforce.scene import normalize_integration_scheme",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "Shared virtualenv is stale relative to this checkout" in result.stderr
    assert "diverging module: pysocialforce/scene.py" in result.stderr
    assert "uv sync --all-extras --reinstall-package robot-sf" in result.stderr
    # The freshness gate must fire before the underlying command is executed.
    assert "uv-reached" not in result.stderr


def test_worktree_shared_venv_freshness_check_passes_on_fresh_env(
    tmp_path: Path,
) -> None:
    """A shared env whose installed pysocialforce matches the source must proceed to the command."""
    matching_scene = "def normalize_integration_scheme(value=None):\n    return value\n"
    repo, venv, env = _make_freshness_fixture_repo(tmp_path, installed_scene=matching_scene)

    result = subprocess.run(
        [
            str(RUN_WORKTREE_SHARED_VENV),
            "--venv",
            str(venv),
            "--",
            "python",
            "-V",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    # The freshness gate passes, so the helper reaches the underlying command (fake uv exits 7).
    assert result.returncode == 7
    assert "uv-reached" in result.stderr
    assert "Shared virtualenv is stale" not in result.stderr


def test_worktree_shared_venv_freshness_check_flag_bypasses_stale_env(
    tmp_path: Path,
) -> None:
    """--no-freshness-check lets a confirmed-matching env skip the drift gate."""
    repo, venv, env = _make_freshness_fixture_repo(
        tmp_path,
        installed_scene="# stale install without normalize_integration_scheme\n",
    )

    result = subprocess.run(
        [
            str(RUN_WORKTREE_SHARED_VENV),
            "--venv",
            str(venv),
            "--no-freshness-check",
            "--",
            "python",
            "-V",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 7
    assert "uv-reached" in result.stderr
    assert "Shared virtualenv is stale" not in result.stderr


def test_worktree_shared_venv_standalone_mode_bypasses_stale_project_env(
    tmp_path: Path,
) -> None:
    """--standalone reaches dependency-light tools without exposing project source."""
    repo, venv, env = _make_freshness_fixture_repo(
        tmp_path,
        installed_scene="# stale install without normalize_integration_scheme\n",
    )

    result = subprocess.run(
        [
            str(RUN_WORKTREE_SHARED_VENV),
            "--venv",
            str(venv),
            "--standalone",
            "--",
            "python",
            "scripts/dev/check_docs_evidence_integrity.py",
            "--help",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 7
    assert "uv-reached" in result.stderr
    assert "Shared virtualenv is stale" not in result.stderr
    assert "pythonpath=\n" in result.stderr


def test_worktree_shared_venv_freshness_check_env_var_bypasses_stale_env(
    tmp_path: Path,
) -> None:
    """ROBOT_SF_VENV_FRESHNESS_CHECK=skip lets a confirmed-matching env skip the drift gate."""
    repo, venv, env = _make_freshness_fixture_repo(
        tmp_path,
        installed_scene="# stale install without normalize_integration_scheme\n",
    )
    env = {**env, "ROBOT_SF_VENV_FRESHNESS_CHECK": "skip"}

    result = subprocess.run(
        [
            str(RUN_WORKTREE_SHARED_VENV),
            "--venv",
            str(venv),
            "--",
            "python",
            "-V",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 7
    assert "uv-reached" in result.stderr
    assert "Shared virtualenv is stale" not in result.stderr


def test_gh_comment_has_valid_shell_syntax() -> None:
    """gh_comment.sh should pass bash -n syntax check."""
    syntax = subprocess.run(
        ["bash", "-n", str(GH_COMMENT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_gh_comment_top_level_help_long() -> None:
    """gh_comment.sh --help prints usage and exits 0."""
    result = subprocess.run(
        [str(GH_COMMENT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "pr <number>" in result.stdout
    assert "issue <number>" in result.stdout


def test_gh_comment_top_level_help_short() -> None:
    """gh_comment.sh -h prints usage and exits 0."""
    result = subprocess.run(
        [str(GH_COMMENT), "-h"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_gh_comment_pr_help() -> None:
    """gh_comment.sh pr --help prints usage and exits 0."""
    result = subprocess.run(
        [str(GH_COMMENT), "pr", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "pr <number>" in result.stdout


def test_gh_comment_issue_help() -> None:
    """gh_comment.sh issue --help prints usage and exits 0."""
    result = subprocess.run(
        [str(GH_COMMENT), "issue", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "issue <number>" in result.stdout


def test_gh_comment_no_args_exits_2() -> None:
    """gh_comment.sh with no arguments prints usage to stdout and exits 2."""
    result = subprocess.run(
        [str(GH_COMMENT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stdout


def test_gh_comment_invalid_target_exits_2() -> None:
    """gh_comment.sh with invalid target prints error to stderr and exits 2."""
    result = subprocess.run(
        [str(GH_COMMENT), "invalid"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "target must be 'pr' or 'issue'" in result.stderr
    assert "Usage:" in result.stdout


# Help-behaviour contract tests.

HELP_COVERED_SCRIPTS = [
    PR_READY_CHECK,
    GH_COMMENT,
    RUN_WORKTREE_SHARED_VENV,
    RUN_TESTS_PARALLEL,
    RUN_XDIST_RACE_VALIDATION,
    RUN_CI_LOCAL,
    LOCAL_SIGNOFF,
    CI_DRIVER,
    CHECK_RUNTIME_REQUIREMENTS,
    CHECK_CARLA_RUNTIME,
]


def _script_name(path: Path) -> str:
    return path.name


def _make_help_fixture_repo(
    tmp_path: Path,
    script_names: tuple[str, ...],
) -> tuple[Path, Path, dict[str, str]]:
    """Create a tiny repo where help paths prove they do not invoke uv-backed setup."""

    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()

    for script_name in script_names:
        source = ROOT / "scripts" / "dev" / script_name
        target = script_dir / script_name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\necho "uv should not be called for --help" >&2\nexit 99\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "test fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }
    return repo, script_dir, env


@pytest.mark.parametrize("script", HELP_COVERED_SCRIPTS, ids=_script_name)
def test_help_long_usage(script: Path) -> None:
    """Every contract-covered script exits 0 with Usage: for --help."""
    result = subprocess.run(
        [str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"{script.name} --help failed: {result.stderr}"
    assert "Usage:" in result.stdout


@pytest.mark.parametrize("script", HELP_COVERED_SCRIPTS, ids=_script_name)
def test_help_short_usage(script: Path) -> None:
    """Every contract-covered script exits 0 with Usage: for -h."""
    result = subprocess.run(
        [str(script), "-h"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"{script.name} -h failed: {result.stderr}"
    assert "Usage:" in result.stdout


# Cheap help: --help must not invoke heavy gates.


def test_ci_driver_help_does_not_invoke_phases(tmp_path: Path) -> None:
    """ci_driver.sh --help exits 0 before sourcing common_setup or running phases."""
    repo, script_dir, env = _make_help_fixture_repo(tmp_path, ("ci_driver.sh", "common_setup.sh"))
    result = subprocess.run(
        [str(script_dir / "ci_driver.sh"), "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "uv should not be called" not in result.stderr


def test_local_signoff_refuses_dirty_worktree_before_signing(tmp_path: Path) -> None:
    """local_signoff.sh refuses dirty commits before gh or validation can sign them."""
    repo, script_dir, env = _make_help_fixture_repo(
        tmp_path,
        ("local_signoff.sh", "common_setup.sh"),
    )
    (repo / "dirty.txt").write_text("not committed\n", encoding="utf-8")

    result = subprocess.run(
        [str(script_dir / "local_signoff.sh"), "--dry-run", "--no-setup", "artifact-policy"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "refusing to sign HEAD" in result.stderr
    assert "uv should not be called" not in result.stderr


def test_run_tests_parallel_help_does_not_invoke_pytest(tmp_path: Path) -> None:
    """run_tests_parallel.sh --help exits 0 before sourcing common_setup."""
    repo, script_dir, env = _make_help_fixture_repo(
        tmp_path,
        ("run_tests_parallel.sh", "common_setup.sh"),
    )
    result = subprocess.run(
        [str(script_dir / "run_tests_parallel.sh"), "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "COVERAGE_FILE" in result.stdout
    assert "uv should not be called" not in result.stderr


def test_run_xdist_race_validation_help_does_not_invoke_pytest(tmp_path: Path) -> None:
    """run_xdist_race_validation.sh --help exits 0 before invoking uv."""
    repo, script_dir, env = _make_help_fixture_repo(
        tmp_path,
        ("run_xdist_race_validation.sh", "common_setup.sh"),
    )
    result = subprocess.run(
        [str(script_dir / "run_xdist_race_validation.sh"), "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "XDIST_RACE_WORKERS" in result.stdout
    assert "uv should not be called" not in result.stderr


def test_run_ci_local_help_does_not_invoke_setup(tmp_path: Path) -> None:
    """run_ci_local.sh --help exits 0 before sourcing common_setup or running phases."""
    repo, script_dir, env = _make_help_fixture_repo(
        tmp_path,
        ("run_ci_local.sh", "common_setup.sh", "ci_driver.sh"),
    )
    result = subprocess.run(
        [str(script_dir / "run_ci_local.sh"), "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "uv should not be called" not in result.stderr


# ---------------------------------------------------------------------------
# bootstrap_worktree.sh contract tests (issue #5091)
# ---------------------------------------------------------------------------


def test_bootstrap_worktree_script_exists() -> None:
    """bootstrap_worktree.sh must exist and be executable."""
    assert BOOTSTRAP_WORKTREE.exists(), f"Missing: {BOOTSTRAP_WORKTREE}"
    assert BOOTSTRAP_WORKTREE.stat().st_mode & 0o111, "bootstrap_worktree.sh is not executable"


def test_bootstrap_worktree_shell_syntax_is_valid() -> None:
    """bootstrap_worktree.sh must pass bash -n (no syntax errors)."""
    result = subprocess.run(
        ["bash", "-n", str(BOOTSTRAP_WORKTREE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_worktree_help_long() -> None:
    """bootstrap_worktree.sh --help prints usage and exits 0."""
    result = subprocess.run(
        [str(BOOTSTRAP_WORKTREE), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "uv venv .venv" in result.stdout  # must document the explicit venv-create step
    assert "uv sync --all-extras" in result.stdout  # must document the sync step
    assert "source .venv/bin/activate" in result.stdout


def test_bootstrap_worktree_help_short() -> None:
    """bootstrap_worktree.sh -h prints usage and exits 0."""
    result = subprocess.run(
        [str(BOOTSTRAP_WORKTREE), "-h"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_bootstrap_worktree_rejects_unknown_flag() -> None:
    """bootstrap_worktree.sh rejects unknown flags with exit 2."""
    result = subprocess.run(
        [str(BOOTSTRAP_WORKTREE), "--not-a-real-flag"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


def test_bootstrap_worktree_rejects_multiple_worktree_directories() -> None:
    """bootstrap_worktree.sh accepts at most one explicit worktree directory."""
    result = subprocess.run(
        [str(BOOTSTRAP_WORKTREE), "one", "two"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "expected at most one WORKTREE_DIR" in result.stderr


def test_bootstrap_worktree_rejects_extra_without_name() -> None:
    """bootstrap_worktree.sh should reject an incomplete --extra option before syncing."""
    result = subprocess.run(
        [str(BOOTSTRAP_WORKTREE), "--extra"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "--extra requires a name" in result.stderr


def test_bootstrap_worktree_forwards_repeatable_extras_to_uv_sync(tmp_path: Path) -> None:
    """Named bootstrap extras must reach uv sync for training-specific worktrees."""
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    captured_args = repo / "uv-sync-args.txt"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()

    (script_dir / "bootstrap_worktree.sh").write_text(
        BOOTSTRAP_WORKTREE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (script_dir / "bootstrap_worktree.sh").chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "$1" == "venv" ]]; then',
                '  mkdir -p "$2/bin"',
                '  printf "#!/usr/bin/env bash\\nexit 0\\n" > "$2/bin/python"',
                '  chmod 0755 "$2/bin/python"',
                "  exit 0",
                "fi",
                'if [[ "$1" == "sync" ]]; then',
                '  printf "%s\\n" "$*" > "$UV_CAPTURED_ARGS"',
                "  exit 0",
                "fi",
                'echo "unexpected uv invocation: $*" >&2',
                "exit 99",
            ]
        ),
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "bootstrap extra fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            str(script_dir / "bootstrap_worktree.sh"),
            "--no-symlink-machine",
            "--extra",
            "training",
            "--extra",
            "gpu",
        ],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "UV_CAPTURED_ARGS": str(captured_args),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert captured_args.read_text(encoding="utf-8") == "sync --extra training --extra gpu\n"


def test_bootstrap_worktree_targets_an_explicit_linked_worktree(tmp_path: Path) -> None:
    """An explicit target runs the bootstrap flow in that linked worktree."""
    main_repo = tmp_path / "main-repo"
    linked_worktree = tmp_path / "linked-worktree"
    script_dir = main_repo / "scripts" / "dev"
    fake_bin = main_repo / "fake-bin"
    captured_cwds = main_repo / "uv-cwds.txt"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()

    script = script_dir / "bootstrap_worktree.sh"
    script.write_text(BOOTSTRAP_WORKTREE.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$PWD" >> "$UV_CAPTURED_CWDS"',
                'if [[ "$1" == "venv" ]]; then',
                '  mkdir -p "$2/bin"',
                '  printf "#!/usr/bin/env bash\\nexit 0\\n" > "$2/bin/python"',
                '  chmod 0755 "$2/bin/python"',
                "  exit 0",
                "fi",
                'if [[ "$1" == "sync" ]]; then',
                "  exit 0",
                "fi",
                'echo "unexpected uv invocation: $*" >&2',
                "exit 99",
            ]
        ),
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=main_repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.invalid"],
        cwd=main_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=main_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=main_repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "bootstrap target fixture"],
        cwd=main_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "UV_CAPTURED_CWDS": str(captured_cwds),
    }
    non_linked_result = subprocess.run(
        [str(script), "--no-symlink-machine", str(main_repo)],
        cwd=main_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert non_linked_result.returncode == 2
    assert "must be a linked Git worktree" in non_linked_result.stderr
    assert not captured_cwds.exists()

    subprocess.run(
        ["git", "worktree", "add", "-b", "target-worktree", str(linked_worktree)],
        cwd=main_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [str(script), "--no-symlink-machine", str(linked_worktree)],
        cwd=main_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (linked_worktree / ".venv" / "bin" / "python").is_file()
    assert captured_cwds.read_text(encoding="utf-8") == f"{linked_worktree}\n{linked_worktree}\n"


def test_bootstrap_worktree_creates_venv_before_sync() -> None:
    """bootstrap_worktree.sh must call `uv venv .venv` before `uv sync --all-extras` in code.

    Searches only the code body (after the show_help function definition) to avoid
    false positives from comment or help-text occurrences of these strings.
    """
    script_text = BOOTSTRAP_WORKTREE.read_text(encoding="utf-8")

    # Isolate the code body: everything after the show_help function definition ends.
    # The help function closes with `}` on its own line; the main code follows.
    help_end_marker = "\nshow_help"
    code_start = script_text.find(help_end_marker)
    assert code_start != -1, "Could not locate show_help function in bootstrap_worktree.sh"
    # Advance past the show_help block to the arg-parsing / main code body.
    body = script_text[code_start:]

    venv_create = "uv venv .venv"
    sync_cmd = "uv sync --all-extras"
    assert venv_create in body, "bootstrap_worktree.sh code body must contain 'uv venv .venv'"
    assert sync_cmd in body, "bootstrap_worktree.sh code body must contain 'uv sync --all-extras'"
    assert body.find(venv_create) < body.find(sync_cmd), (
        "In the code body, 'uv venv .venv' must appear before 'uv sync --all-extras'"
    )


def test_bootstrap_worktree_fails_closed_on_missing_python(tmp_path: Path) -> None:
    """bootstrap_worktree.sh must exit 1 with an actionable message when .venv/bin/python
    is absent after uv sync (the core fail-closed contract for issue #5091).

    Simulated with a fake `uv` that prints the expected sync output but does NOT
    create .venv/bin/python, reproducing the exact failure mode from the issue.
    """
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()

    (script_dir / "bootstrap_worktree.sh").write_text(
        BOOTSTRAP_WORKTREE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (script_dir / "bootstrap_worktree.sh").chmod(0o755)

    # Fake uv: prints plausible sync output but never creates .venv/bin/python.
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "$1" == "venv" ]]; then',
                '  mkdir -p "$2"',
                "  exit 0",
                "fi",
                'if [[ "$1" == "sync" ]]; then',
                '  echo "Resolved 302 packages in 1ms"',
                '  echo "Checked 256 packages in 12ms"',
                "  exit 0",
                "fi",
                'echo "unexpected uv invocation: $*" >&2',
                "exit 99",
            ]
        ),
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "bootstrap test fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [str(script_dir / "bootstrap_worktree.sh"), "--no-symlink-machine"],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1, (
        f"Expected exit 1 (fail-closed) but got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert ".venv/bin/python" in result.stderr
    assert "uv venv .venv" in result.stderr
    assert "uv sync --all-extras" in result.stderr


def test_bootstrap_worktree_succeeds_when_python_present(tmp_path: Path) -> None:
    """bootstrap_worktree.sh exits 0 when .venv/bin/python exists after uv sync."""
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()

    (script_dir / "bootstrap_worktree.sh").write_text(
        BOOTSTRAP_WORKTREE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (script_dir / "bootstrap_worktree.sh").chmod(0o755)

    # Fake uv: creates .venv/bin/python on `uv venv .venv`, succeeds on sync.
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "$1" == "venv" ]]; then',
                '  venv_dir="$2"',
                '  mkdir -p "$venv_dir/bin"',
                "  # Create a working python stub so the -x check passes.",
                '  printf "#!/usr/bin/env bash\\nexit 0\\n" > "$venv_dir/bin/python"',
                '  chmod 0755 "$venv_dir/bin/python"',
                "  exit 0",
                "fi",
                'if [[ "$1" == "sync" ]]; then',
                '  echo "Resolved 302 packages in 1ms"',
                '  echo "Checked 256 packages in 12ms"',
                "  exit 0",
                "fi",
                'echo "unexpected uv invocation: $*" >&2',
                "exit 99",
            ]
        ),
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "bootstrap success fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [str(script_dir / "bootstrap_worktree.sh"), "--no-symlink-machine"],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        f"Expected success but got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert ".venv/bin/python is ready" in result.stdout
    assert "source .venv/bin/activate" in result.stdout


def test_bootstrap_worktree_help_does_not_invoke_uv(tmp_path: Path) -> None:
    """bootstrap_worktree.sh --help exits 0 before invoking uv."""
    repo, script_dir, env = _make_help_fixture_repo(
        tmp_path,
        ("bootstrap_worktree.sh",),
    )
    result = subprocess.run(
        [str(script_dir / "bootstrap_worktree.sh"), "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "uv should not be called" not in result.stderr
