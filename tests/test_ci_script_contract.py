"""Guard CI shell wrappers against drift in shared script contracts.

Help-behaviour contract
-----------------------
Selected ``scripts/dev/*.sh`` helpers with --help / -h usage support must
handle both forms as cheap success paths: exit 0, print usage to stdout,
and return before sourcing common_setup.sh or invoking heavy dependencies
(uv, ruff, pytest, gh, etc.).

Covered scripts (13 total):
  pr_ready_check.sh, gh_comment.sh, gh_pr_merge.sh, run_worktree_shared_venv.sh,
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_DRIVER = ROOT / "scripts" / "dev" / "ci_driver.sh"
GH_COMMENT = ROOT / "scripts" / "dev" / "gh_comment.sh"
GH_PR_MERGE = ROOT / "scripts" / "dev" / "gh_pr_merge.sh"
PYPROJECT = ROOT / "pyproject.toml"
RUN_TESTS_PARALLEL = ROOT / "scripts" / "dev" / "run_tests_parallel.sh"
OPTIONAL_ALLOWLIST = ROOT / "tests" / "support" / "optional_test_allowlist.txt"
RUN_XDIST_RACE_VALIDATION = ROOT / "scripts" / "dev" / "run_xdist_race_validation.sh"
RUN_CI_LOCAL = ROOT / "scripts" / "dev" / "run_ci_local.sh"
LOCAL_SIGNOFF = ROOT / "scripts" / "dev" / "local_signoff.sh"
PR_READY_CHECK = ROOT / "scripts" / "dev" / "pr_ready_check.sh"
PR_BODY_CONTRACTS_WORKFLOW = ROOT / ".github" / "workflows" / "pr-body-contracts.yml"
RUN_WORKTREE_SHARED_VENV = ROOT / "scripts" / "dev" / "run_worktree_shared_venv.sh"
COMMON_SETUP = ROOT / "scripts" / "dev" / "common_setup.sh"
RUFF_FIX_FORMAT = ROOT / "scripts" / "dev" / "ruff_fix_format.sh"
BOOTSTRAP_WORKTREE = ROOT / "scripts" / "dev" / "bootstrap_worktree.sh"
CHECK_RUNTIME_REQUIREMENTS = ROOT / "scripts" / "dev" / "check_runtime_requirements.sh"
CHECK_CARLA_RUNTIME = ROOT / "scripts" / "dev" / "check_carla_runtime.sh"
CI_INSTALL_HEADLESS_PACKAGES = ROOT / "scripts" / "dev" / "ci_install_headless_packages.sh"
EVIDENCE_REGISTRY_RATCHET = ROOT / "scripts" / "dev" / "evidence_registry_ratchet.py"
COVERAGE_GUIDE = ROOT / "docs" / "coverage_guide.md"
DEV_GUIDE = ROOT / "docs" / "dev_guide.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


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
    assert "cmd=(uv run pytest)" in script_text
    assert 'if [[ "$pytest_execution_mode" == "xdist" ]]; then' in script_text
    assert 'cmd+=(-n "$worker_spec" --dist "$dist_mode")' in script_text
    assert 'if [[ "$worker_spec" == "1" ]]; then' in script_text
    assert "PYTEST_XDIST_DIST=load|worksteal|loadscope|loadfile|loadgroup" in script_text
    assert "--lane core|optional|all" in script_text
    assert "ROBOT_SF_TEST_LANE=core|optional|all" in script_text
    assert "Resolved pytest lane:" in script_text
    assert 'dependency_profile="all-extras"' in script_text
    assert 'dependency_profile="core"' in script_text
    assert 'preflight_check_worktree_dependency_profile "$dependency_profile"' in script_text
    assert "Repair with: cd" in script_text
    assert "normalize_pytest_target_path()" in script_text
    assert "${path%%::*}" in script_text
    assert "core_test_paths=(" in script_text
    assert "tests/adversarial" in script_text
    assert "tests/analysis_workbench" in script_text
    assert "tests/ped_npc" in script_text
    assert "tests/scenario_certification" in script_text
    assert "explicit_test_targets=(" in script_text
    assert 'cmd+=("--ignore=$optional_test_path")' in script_text
    assert 'pytest_args+=("$optional_test_path")' in script_text
    assert 'append_unique_pytest_arg "$core_test_path"' in script_text
    assert "changed_top_level_core_test_paths=()" in script_text
    assert "-- tests fast-pysf/tests" in script_text
    assert 'append_unique_pytest_arg "$changed_test_path"' in script_text
    assert "Core pytest lane cannot run optional-extra path" in script_text


def test_run_tests_parallel_emits_duration_store_flags_for_sharded_runs() -> None:
    """Sharded runs must record pytest-split durations so CI can balance later shards."""

    script_text = RUN_TESTS_PARALLEL.read_text(encoding="utf-8")

    splits_line = 'cmd+=("--splits" "$shard_count" "--group" "$shard_index")'
    store_line = 'cmd+=("--store-durations" "--durations-path" ".test_durations")'
    assert splits_line in script_text
    assert store_line in script_text
    # Duration-store flags belong to the sharding block, not default unsharded runs.
    assert script_text.find(splits_line) < script_text.find(store_line)

    # ``--clean-durations`` keeps the CI cache fresh: each shard uploads only the
    # tests it ran so the workflow-level merge unions disjoint stores instead of
    # freezing on the restored aggregate's stale values. It must stay gated on
    # ``CI=true`` so local sharded runs keep pytest-split's retain-and-update
    # default behavior.
    ci_gate = 'if [[ "${CI:-}" == "true" ]]; then'
    clean_flag = 'cmd+=("--clean-durations")'
    assert ci_gate in script_text
    assert clean_flag in script_text
    assert script_text.find(store_line) < script_text.find(ci_gate)
    assert script_text.find(ci_gate) < script_text.find(clean_flag)


def test_run_tests_parallel_allows_only_empty_fast_only_shards() -> None:
    """A PR shard may contain only excluded slow tests, but real failures stay red."""

    script_text = RUN_TESTS_PARALLEL.read_text(encoding="utf-8")

    empty_shard_guard = (
        'if [[ "$pytest_exit" -eq 5 && "$sharding_active" == "1" && "$include_slow" == "0" ]]'
    )
    assert empty_shard_guard in script_text
    assert 'grep -Fq "no tests ran" "$pytest_log"' in script_text
    assert "fast-only shard collected no tests" in script_text


def test_run_tests_parallel_empty_shard_guard_executes_only_the_safe_case(tmp_path: Path) -> None:
    """Exercise the exit-5 guard and prove rejected cases clean up their logs."""

    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()
    (repo / "tests" / "support").mkdir(parents=True)
    (repo / "tests" / "support" / "optional_test_allowlist.txt").write_text("", encoding="utf-8")
    venv_python = repo / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    venv_python.chmod(0o755)
    for script_name in ("run_tests_parallel.sh", "common_setup.sh"):
        source = ROOT / "scripts" / "dev" / script_name
        target = script_dir / script_name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)

    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "-C",
            str(repo),
            "add",
            ".",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "-C",
            str(repo),
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "$1" == run && "$2" == python ]]; then\n'
        '  case "$3" in\n'
        "    *resolve_pytest_workers.py) printf '1\\n' ;;\n"
        "    *diagnose_xdist_crash.py) exit 0 ;;\n"
        '    *) echo "unexpected helper: $*" >&2; exit 99 ;;\n'
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "$1" == run && "$2" == pytest ]]; then\n'
        '  cat "$FIXTURE_OUTPUT"\n'
        '  exit "$FIXTURE_EXIT"\n'
        "fi\n"
        'echo "unexpected uv invocation: $*" >&2\n'
        "exit 99\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    temp_root = tmp_path / "tmp space"
    temp_root.mkdir()
    output = tmp_path / "pytest-output.txt"

    cases = (
        ("empty-fast-shard", 5, "no tests ran\n", 2, 1, 0),
        ("ordinary-exit-5", 5, "collected 1 item\n", 2, 1, 0),
        ("test-failure", 1, "no tests ran\n", 2, 1, 0),
        ("collection-error", 2, "ERROR collecting test_bad.py\n", 2, 1, 0),
        ("full-suite-empty", 5, "no tests ran\n", 2, 1, 1),
        ("unsharded-empty", 5, "no tests ran\n", 1, 1, 0),
    )
    for name, pytest_exit, pytest_output, shard_count, shard_index, include_slow in cases:
        output.write_text(pytest_output, encoding="utf-8")
        env = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PYTEST_NUM_WORKERS": "1",
            "PYTEST_FAST_FAIL": "0",
            "PYTEST_ORDER_MODE": "none",
            "PYTEST_SHARD_COUNT": str(shard_count),
            "PYTEST_SHARD_INDEX": str(shard_index),
            "ROBOT_SF_SHARD_INCLUDE_SLOW": str(include_slow),
            "PR_READY_SKIP_PREFLIGHT": "1",
            "PYTEST_DEBUG_TEMPROOT": "fixture",
            "TMPDIR": str(temp_root),
            "COVERAGE_FILE": str(tmp_path / "coverage" / f"{name}.coverage"),
            "FIXTURE_OUTPUT": str(output),
            "FIXTURE_EXIT": str(pytest_exit),
        }
        result = subprocess.run(
            [str(script_dir / "run_tests_parallel.sh")],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        expected = 0 if name == "empty-fast-shard" else pytest_exit
        assert result.returncode == expected, (name, result.stdout, result.stderr)
        assert not list(temp_root.glob("pytest_run.*.log")), name
        assert not list(temp_root.glob("pytest_serial.*.log")), name

    missing_coverage_env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTEST_NUM_WORKERS": "1",
        "PYTEST_FAST_FAIL": "0",
        "PYTEST_ORDER_MODE": "none",
        "PYTEST_SHARD_COUNT": "2",
        "PYTEST_SHARD_INDEX": "1",
        "ROBOT_SF_SHARD_INCLUDE_SLOW": "0",
        "ROBOT_SF_PYTEST_COVERAGE": "1",
        "PR_READY_SKIP_PREFLIGHT": "1",
        "PYTEST_DEBUG_TEMPROOT": "fixture",
        "TMPDIR": str(temp_root),
        "FIXTURE_OUTPUT": str(output),
        "FIXTURE_EXIT": "0",
    }
    missing_coverage_env.pop("COVERAGE_FILE", None)
    result = subprocess.run(
        [str(script_dir / "run_tests_parallel.sh")],
        cwd=repo,
        env=missing_coverage_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "Sharded coverage requires a unique COVERAGE_FILE per shard." in result.stderr


def test_ci_fast_feedback_authenticates_gh_release_downloads() -> None:
    """Keep public release hydration on the authenticated Actions API path."""

    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["jobs"]["fast-feedback"]["env"]["GH_TOKEN"] == "${{ github.token }}"


def test_ci_workflow_persists_merged_pytest_duration_store() -> None:
    """Keep CI duration restore, per-shard upload, and aggregate-save wiring intact."""

    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    fast_feedback = workflow["jobs"]["fast-feedback"]
    fast_feedback_steps = fast_feedback["steps"]
    duration_restore = next(
        step
        for step in fast_feedback_steps
        if step.get("name") == "Restore test durations for pytest-split balancing"
    )
    duration_upload = next(
        step for step in fast_feedback_steps if step.get("name") == "Upload test durations"
    )

    assert duration_restore["uses"].startswith("actions/cache/restore@")
    assert duration_restore["continue-on-error"] is True
    assert duration_restore["with"]["path"] == ".test_durations"
    assert "${{ github.run_id }}" in duration_restore["with"]["key"]
    assert "${{ github.run_attempt }}" in duration_restore["with"]["key"]
    assert "test-durations-${{ runner.os }}-" in duration_restore["with"]["restore-keys"]
    assert duration_upload["if"] == "always()"
    assert duration_upload["continue-on-error"] is True
    assert duration_upload["with"] == {
        "name": "pytest-durations-${{ matrix.shard }}",
        "path": ".test_durations",
        "if-no-files-found": "ignore",
        "include-hidden-files": True,
    }

    aggregate = workflow["jobs"]["ci"]
    assert "fast-feedback" in aggregate["needs"]
    duration_checkout = next(
        step
        for step in aggregate["steps"]
        if step.get("name") == "Checkout for duration-cache update"
    )
    assert duration_checkout["id"] == "checkout_duration_cache"
    assert "always()" in duration_checkout["if"]
    assert duration_checkout["continue-on-error"] is True
    duration_download = next(
        step for step in aggregate["steps"] if step.get("name") == "Download test-duration shards"
    )
    duration_merge = next(
        step for step in aggregate["steps"] if step.get("name") == "Merge test durations"
    )
    duration_save = next(
        step for step in aggregate["steps"] if step.get("name") == "Save merged test durations"
    )

    assert duration_download["with"] == {
        "pattern": "pytest-durations-*",
        "path": ".duration-artifacts",
    }
    assert "always()" in duration_download["if"]
    assert "steps.checkout_duration_cache.outcome == 'success'" in duration_download["if"]
    assert duration_merge["id"] == "merge-test-durations"
    assert duration_merge["continue-on-error"] is True
    assert "always()" in duration_merge["if"]
    assert "steps.checkout_duration_cache.outcome == 'success'" in duration_merge["if"]
    # The inline merge program is replaced by the tested helper.
    assert "merge_test_durations.py" in duration_merge["run"]
    assert "--artifact-dir .duration-artifacts" in duration_merge["run"]
    assert "--output .test_durations" in duration_merge["run"]
    assert "Expected exactly one pytest duration store" not in duration_merge["run"]
    assert "Overlapping pytest duration stores" not in duration_merge["run"]
    assert "merged.update(durations)" not in duration_merge["run"]
    assert duration_save["continue-on-error"] is True
    assert "always()" in duration_save["if"]
    assert "steps.checkout_duration_cache.outcome == 'success'" in duration_save["if"]
    assert "steps.merge-test-durations.outcome == 'success'" in duration_save["if"]
    assert duration_save["with"]["path"] == ".test_durations"
    assert "${{ github.run_id }}" in duration_save["with"]["key"]
    assert "${{ github.run_attempt }}" in duration_save["with"]["key"]


def test_determinism_gate_reuses_the_model_preflight_cache() -> None:
    """The full determinism gate must not redownload models in a parallel job."""

    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    determinism_gate = workflow["jobs"]["determinism-gate"]
    assert determinism_gate["needs"] == "exact-repeat-model-preflight"
    assert (
        determinism_gate["env"]["PPO_ALGO_CONFIG"]
        == "configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml"
    )
    key_step = next(
        step
        for step in determinism_gate["steps"]
        if step.get("name") == "Derive model-cache key from registry-pinned digests"
    )
    assert key_step["id"] == "model-cache-key"
    assert "model_cache_key.py" in key_step["run"]
    assert "--config" in key_step["run"]
    assert "${PPO_ALGO_CONFIG}" in key_step["run"]
    # The inline registry/preflight snippet is replaced by the tested helper.
    assert "required_model_ids_for_config" not in key_step["run"]
    assert "hashlib.sha256" not in key_step["run"]
    restore_step = next(
        step
        for step in determinism_gate["steps"]
        if step.get("name") == "Restore exact-repeat model cache"
    )
    assert restore_step["uses"].startswith("actions/cache/restore@")
    assert restore_step["with"] == {
        "path": "output/model_cache",
        "key": "model-cache-exact-repeat-ppo-${{ steps.model-cache-key.outputs.key }}",
    }


def test_ci_aggregate_uses_declarative_needs_checker() -> None:
    """The ci aggregate job must route result checks through check_ci_needs.py."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    aggregate = workflow["jobs"]["ci"]
    result_step = next(
        step for step in aggregate["steps"] if step.get("name") == "Check split job results"
    )
    assert "check_ci_needs.py" in result_step["run"]
    assert "--event-name" in result_step["run"]
    assert "${{ github.event_name }}" in result_step["run"]
    assert "--results" in result_step["run"]
    assert "python scripts/dev/check_ci_needs.py" in result_step["run"]
    assert "uv run python scripts/dev/check_ci_needs.py" not in result_step["run"]
    assert '--results "$CI_NEEDS_JSON"' in result_step["run"]
    assert result_step.get("env", {}).get("CI_NEEDS_JSON") == "${{ toJSON(needs) }}"
    checkout_index = next(
        index
        for index, step in enumerate(aggregate["steps"])
        if step.get("id") == "checkout_duration_cache"
    )
    result_index = aggregate["steps"].index(result_step)
    assert checkout_index < result_index
    # The handwritten per-dependency if-blocks are gone.
    assert 'needs.fast-feedback.result"' not in result_step["run"]
    assert "needs.determinism-gate.result" not in result_step["run"]
    # Every required job stays in the aggregate needs set.
    for job in (
        "fast-feedback",
        "changed-coverage-gate",
        "coverage-gate",
        "compat-matrix",
        "fast-pysf-compat",
        "smoke-artifacts",
        "scenario-validation",
        "xdist-scratch-isolation",
        "wheel-smoke-install",
        "examples-smoke",
        "notebooks-smoke",
        "determinism-gate",
        "exact-repeat-model-preflight",
    ):
        assert job in aggregate["needs"]


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
    assert "Resolved pytest workers" not in result.stderr
    assert "Resolved pytest-xdist workers" not in result.stderr
    assert "resolve_pytest_workers.py" not in result.stderr


def test_run_tests_parallel_fails_before_worker_resolution_on_incomplete_profile(
    tmp_path: Path,
) -> None:
    """A partial fresh-worktree venv is setup evidence, not a collection failure (#7726)."""
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()
    (repo / "tests" / "support").mkdir(parents=True)
    (repo / "tests" / "support" / "optional_test_allowlist.txt").write_text("", encoding="utf-8")
    for script_name in ("run_tests_parallel.sh", "common_setup.sh"):
        source = ROOT / "scripts" / "dev" / script_name
        target = script_dir / script_name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)

    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    venv_python = repo / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'Worktree optional dependency preflight: missing_optional (all-extras)\\n'\n"
        "printf 'Missing optional imports: pandas\\n'\n"
        "exit 2\n",
        encoding="utf-8",
    )
    venv_python.chmod(0o755)
    uv_called = repo / "uv-called.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" > "$UV_CALLED"\nexit 99\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    result = subprocess.run(
        [str(script_dir / "run_tests_parallel.sh"), "--lane", "all"],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "UV_CALLED": str(uv_called),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    diagnostic = result.stdout + result.stderr
    assert "dependency profile 'all-extras' is incomplete" in diagnostic
    assert "Missing optional imports: pandas" in diagnostic
    assert "pytest was not started" in diagnostic
    assert "uv sync --all-extras" in diagnostic
    assert not uv_called.exists()


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
    venv_python = repo / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    venv_python.chmod(0o755)

    captured_args = repo / "captured-pytest-args.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "$1" == "run" && "$2" == "python" ]]; then',
                '  echo "worker resolver must not rewrite explicit serial mode" >&2',
                "  exit 98",
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
    ped_npc_test = repo / "tests" / "ped_npc" / "test_population.py"
    ped_npc_test.parent.mkdir(parents=True)
    ped_npc_test.write_text("def test_population(): pass\n", encoding="utf-8")
    nested_core_test = repo / "tests" / "new_nested" / "test_nested.py"
    nested_core_test.parent.mkdir(parents=True)
    nested_core_test.write_text("def test_nested(): pass\n", encoding="utf-8")
    fast_pysf_test = repo / "fast-pysf" / "tests" / "test_forces.py"
    fast_pysf_test.parent.mkdir(parents=True)
    fast_pysf_test.write_text("def test_fast(): pass\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tests", "fast-pysf"], cwd=repo, check=True, capture_output=True, text=True
    )
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
    padded_pytest_args = f" {pytest_args} "
    assert " -n " not in padded_pytest_args
    assert " --dist " not in padded_pytest_args
    assert "in-process serial (pytest-xdist disabled)" in result.stderr
    assert "tests/test_new_top_level.py" in pytest_args
    assert "tests/test_optional_top_level.py" not in pytest_args
    assert "tests/ped_npc" in pytest_args
    assert "tests/new_nested/test_nested.py" in pytest_args
    assert "fast-pysf/tests" in pytest_args


def test_run_tests_parallel_keeps_ped_npc_in_core_lane() -> None:
    """The core lane must collect pedestrian-constructor tests (issue #5753)."""
    script_text = RUN_TESTS_PARALLEL.read_text(encoding="utf-8")
    core_paths = script_text.split("core_test_paths=(", maxsplit=1)[1].split(")", maxsplit=1)[0]

    assert "\n  tests/ped_npc\n" in core_paths
    assert "tests/ped_npc" not in OPTIONAL_ALLOWLIST.read_text(encoding="utf-8")


def test_run_tests_parallel_serial_fallback_is_single_worker_and_fail_closed(
    tmp_path: Path,
) -> None:
    """Coverage-finalization fallback must be true no-xdist and fail closed (#6526)."""
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    (repo / "tests" / "dev").mkdir(parents=True)
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()
    (repo / "tests" / "support").mkdir(parents=True)
    (repo / "tests" / "support" / "optional_test_allowlist.txt").write_text(
        "tests/optional\n", encoding="utf-8"
    )

    for script_name in ("run_tests_parallel.sh", "common_setup.sh"):
        source = ROOT / "scripts" / "dev" / script_name
        target = script_dir / script_name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)
    venv_python = repo / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    venv_python.chmod(0o755)

    captured_args = repo / "captured-pytest-args.txt"
    invocation_count = repo / "pytest-invocations.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "$1" == "run" && "$2" == "python" ]]; then',
                '  case "$3" in',
                '    *resolve_pytest_workers.py) printf "2\\n" ;;',
                "    *diagnose_xdist_crash.py)",
                '      if [[ " $* " == *" --serialized-ok false "* ]]; then',
                '        echo "serial diagnostic observed" >&2',
                "      else",
                '        echo "parallel diagnostic observed" >&2',
                "      fi",
                "      ;;",
                '    *) echo "unexpected python helper: $*" >&2; exit 99 ;;',
                "  esac",
                "  exit 0",
                "fi",
                'if [[ "$1" == "run" && "$2" == "pytest" ]]; then',
                "  count=0",
                '  [[ -f "$UV_COUNT_FILE" ]] && count=$(<"$UV_COUNT_FILE")',
                "  count=$((count + 1))",
                '  printf "%s\\n" "$count" > "$UV_COUNT_FILE"',
                '  printf "%s\\n" "$*" >> "$UV_CAPTURED_ARGS"',
                '  echo "sqlite3.OperationalError: unable to open database file" >&2',
                '  echo "Segmentation fault (core dumped)" >&2',
                "  exit 1",
                "fi",
                'echo "unexpected uv invocation: $*" >&2',
                "exit 99",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    result = subprocess.run(
        [str(script_dir / "run_tests_parallel.sh"), "--no-ordering", "tests/dev"],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PR_READY_SERIAL_FALLBACK": "1",
            "PYTEST_NUM_WORKERS": "2",
            "PYTEST_FAST_FAIL": "0",
            "PYTEST_ORDER_MODE": "none",
            "UV_CAPTURED_ARGS": str(captured_args),
            "UV_COUNT_FILE": str(invocation_count),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    calls = captured_args.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "-n 2" in calls[0]
    assert "--dist load" in calls[0]
    padded_serial_call = f" {calls[1]} "
    assert " -n " not in padded_serial_call
    assert " --dist " not in padded_serial_call
    assert "-n 2" not in calls[1]
    assert "pytest-xdist disabled" in result.stderr
    assert "parallel diagnostic observed" in result.stderr
    assert "serial diagnostic observed" in result.stderr


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
    assert "uvx ty@0.0.58 check . --exit-zero" in script_text


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
        '"$SCRIPT_DIR/ruff_fix_format.sh" "${format_changed_files[@]}"',
        '"$SCRIPT_DIR/run_tests_parallel.sh"',
        '"$SCRIPT_DIR/check_changed_coverage.sh"',
        '"$SCRIPT_DIR/check_docstring_todos_diff.sh"',
        '"$SCRIPT_DIR/check_docstring_todos_ratchet.sh"',
        'uv run python "$SCRIPT_DIR/check_optional_import_pr_freshness.py" --base-ref "$BASE_REF"',
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
    assert 'git diff --name-only --diff-filter=ACDMRT "$BASE_REF...HEAD"' in script_text
    assert "format_changed_files=()" in script_text
    assert '[[ "$changed_file" == *.py && -f "$changed_file" ]]' in script_text

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


def test_pr_ready_check_captures_validated_base_sha_for_drift_guard(tmp_path: Path) -> None:
    """Issue #5782: the readiness gate must capture the concrete base SHA it validates
    against and run a final base-drift recheck before recording the freshness stamp.

    The drift guard compares the captured base SHA against the (moving) base ref
    again before the stamp is written, so a readiness stamp cannot stay green
    through a silent origin/main advance during the long lanes.
    """
    script_text = PR_READY_CHECK.read_text(encoding="utf-8")

    # The gate resolves the concrete base commit before the expensive lanes.
    assert (
        'VALIDATED_BASE_SHA="$(git rev-parse --verify --quiet "${BASE_REF}^{commit}"' in script_text
    )
    assert "Validated base SHA for this run" in script_text

    # The freshness stamp now records the resolved base SHA.
    assert 'freshness_args+=(--base-sha "$VALIDATED_BASE_SHA")' in script_text

    # A final lightweight drift recheck runs before the stamp is written.
    assert 'uv run python "$SCRIPT_DIR/check_base_drift.py"' in script_text
    assert "--validated-base-sha" in script_text
    assert "--changed-files" in script_text
    assert "revalidate against" in script_text
    # The drift recheck must sit before the freshness stamp write.
    drift_index = script_text.find('uv run python "$SCRIPT_DIR/check_base_drift.py"')
    freshness_write_index = script_text.find('uv run python "$SCRIPT_DIR/pr_ready_freshness.py"')
    assert 0 < drift_index < freshness_write_index, (
        "base-drift recheck must precede the stamp write"
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


def test_pr_ready_check_final_mode_runs_evidence_hygiene_contract() -> None:
    """Final local readiness must invoke the hosted evidence-hygiene contract (issue #7812)."""
    script_text = PR_READY_CHECK.read_text(encoding="utf-8")

    assert "pr_contract_check.py" in script_text
    assert "--changed-files-file" in script_text
    assert "Issue #7812" in script_text
    # The gate is bound to final readiness only.
    assert 'if [[ "$pr_ready_final" == "1" ]]; then' in script_text
    # The evidence-hygiene invocation must appear after the followups gate.
    followups_index = script_text.find("check_pr_followups.py")
    contract_index = script_text.find("pr_contract_check.py")
    assert 0 < followups_index < contract_index


def test_pr_body_contracts_workflow_runs_strict_pr_body_checker() -> None:
    """The live PR workflow should enforce body, follow-up, and domain-review contracts."""
    workflow_text = PR_BODY_CONTRACTS_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in workflow_text
    assert "scripts/ci/collect_pr_files.py" in workflow_text
    assert "--out-changed-files" in workflow_text
    assert "gh api --paginate" not in workflow_text
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
                "  echo 'duckdb, pyarrow, pandas'",
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
    assert "duckdb, pyarrow, pandas" in result.stderr
    assert "ruff_fix_format" not in result.stderr


def test_pr_ready_check_rejects_process_substitution_body_paths(tmp_path: Path) -> None:
    """Process substitution fails at the wrapper boundary; regular files reach preflight."""
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
        "#!/usr/bin/env bash\necho 'duckdb, pyarrow, pandas' >&2\nexit 1\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=agent@example.invalid",
            "-c",
            "user.name=Agent",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "BASE_REF": "HEAD",
        "PR_READY_MODE": "final",
    }

    process_substitution = subprocess.run(
        [
            "bash",
            "-c",
            'body_source=<(printf "## Summary\\nbody\\n"); PR_READY_PR_BODY_FILE="$body_source" "$1"',
            "bash",
            str(script_dir / "pr_ready_check.sh"),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert process_substitution.returncode == 2
    assert (
        "PR_READY_PR_BODY_FILE must name an existing readable regular file"
        in process_substitution.stderr
    )
    assert "Process-substitution paths are not supported" in process_substitution.stderr
    assert "mktemp" in process_substitution.stderr
    assert "Final PR readiness requires analytics dependencies" not in process_substitution.stderr

    body_file = tmp_path / "pr-body.md"
    body_file.write_text("## Summary\nbody\n", encoding="utf-8")
    regular_file = subprocess.run(
        [str(script_dir / "pr_ready_check.sh")],
        cwd=repo,
        env={**env, "PR_READY_PR_BODY_FILE": str(body_file)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert regular_file.returncode == 2
    assert "Final PR readiness requires analytics dependencies" in regular_file.stderr
    assert (
        "PR_READY_PR_BODY_FILE must name an existing readable regular file"
        not in regular_file.stderr
    )


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
    assert 'if [[ -n "$venv_override" ]]; then' in script_text
    assert 'elif [[ -x "$repo_root/.venv/bin/python" ]]; then' in script_text
    assert 'venv_path="$repo_root/.venv"' in script_text
    assert 'venv_path="$main_repo_root/.venv"' in script_text
    assert 'export UV_PROJECT_ENVIRONMENT="$venv_path"' in script_text
    assert "export UV_NO_SYNC=1" in script_text
    assert (
        'export PYTHONPATH="$repo_root:$repo_root/fast-pysf${PYTHONPATH:+:$PYTHONPATH}"'
        in script_text
    )
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
    assert "PYTHONPATH=$PWD:$PWD/fast-pysf" in help_result.stdout
    assert "UV_PROJECT_ENVIRONMENT" in help_result.stdout
    assert "UV_NO_SYNC=1" in help_result.stdout
    assert "COVERAGE_FILE" in help_result.stdout
    assert "full local .venv" in help_result.stdout
    assert "--profile NAME" in help_result.stdout
    assert "all-extras" in help_result.stdout


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
    """The shared-venv helper recognizes checkout-local source as authoritative (issue #6003)."""
    script_text = RUN_WORKTREE_SHARED_VENV.read_text(encoding="utf-8")

    # The freshness gate recognizes the checkout-local source as authoritative.
    assert "check_shared_venv_freshness()" in script_text
    assert 'local src_pkg="$repo_root/fast-pysf/pysocialforce"' in script_text
    assert "checkout source authoritative" in script_text
    # Standalone commands with a verified no-project-import boundary remain supported.
    assert "--standalone" in script_text
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
    venv = repo / ".venv"
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
    (repo / ".gitignore").write_text(".venv/\n", encoding="utf-8")

    # The helper only checks venv presence via bin/python executability.
    py = venv / "bin" / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    py.chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf "uv-reached %s\\n" "$*" >&2\n'
        'printf "venv=%s\\n" "${UV_PROJECT_ENVIRONMENT-}" >&2\n'
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


def test_worktree_shared_venv_ignores_stale_installed_copy(
    tmp_path: Path,
) -> None:
    """A checkout source package must win over a stale installed copy (issue #6003)."""
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

    assert result.returncode == 7
    assert "uv-reached" in result.stderr
    assert "Shared virtualenv is stale" not in result.stderr


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


def test_worktree_shared_venv_prefers_initialized_local_env(tmp_path: Path) -> None:
    """A linked worktree selects its usable local env before a stale main env (issue #5984)."""
    matching_scene = "def normalize_integration_scheme(value=None):\n    return value\n"
    repo, main_venv, env = _make_freshness_fixture_repo(
        tmp_path,
        installed_scene="# stale main install\n",
    )
    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    local_venv = worktree / ".venv"
    local_site_packages = local_venv / "lib" / "python3.12" / "site-packages" / "pysocialforce"
    local_site_packages.mkdir(parents=True)
    (local_venv / "bin").mkdir()
    local_python = local_venv / "bin" / "python"
    local_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    local_python.chmod(0o755)
    (local_site_packages / "scene.py").write_text(matching_scene, encoding="utf-8")
    (local_site_packages / "__init__.py").write_text("", encoding="utf-8")

    result = subprocess.run(
        [str(RUN_WORKTREE_SHARED_VENV), "--", "python", "-V"],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 7
    assert f"venv={local_venv}" in result.stderr
    assert f"venv={main_venv}" not in result.stderr
    assert "Shared virtualenv is stale" not in result.stderr


def test_worktree_shared_venv_explicit_override_shadows_incomplete_local_env(
    tmp_path: Path,
) -> None:
    """Issue #7823: an explicit --venv stays authoritative over an incomplete local .venv."""
    matching_scene = "def normalize_integration_scheme(value=None):\n    return value\n"
    repo, main_venv, env = _make_freshness_fixture_repo(tmp_path, installed_scene=matching_scene)
    # Add a diagnostic only for this precedence assertion; other freshness fixtures stay unchanged.
    fake_uv = repo / "fake-bin" / "uv"
    fake_uv.write_text(
        fake_uv.read_text(encoding="utf-8").replace(
            "exit 7",
            'echo "virtual_env=${VIRTUAL_ENV-}" >&2; exit 7',
            1,
        ),
        encoding="utf-8",
    )
    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    # An incomplete worktree-local .venv must not shadow the explicit shared override.
    incomplete_local_venv = worktree / ".venv"
    (incomplete_local_venv / "bin").mkdir(parents=True)
    (incomplete_local_venv / "bin" / "python").write_text("not executable\n", encoding="utf-8")

    result = subprocess.run(
        [str(RUN_WORKTREE_SHARED_VENV), "--venv", str(main_venv), "--", "python", "-V"],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 7
    assert f"venv={main_venv}" in result.stderr
    assert f"virtual_env={main_venv}" in result.stderr
    assert f"pythonpath={worktree}:{worktree / 'fast-pysf'}" in result.stderr
    wrapper_text = RUN_WORKTREE_SHARED_VENV.read_text(encoding="utf-8")
    assert 'export VIRTUAL_ENV="$venv_path"' in wrapper_text
    assert (
        'export PYTHONPATH="$repo_root:$repo_root/fast-pysf${PYTHONPATH:+:$PYTHONPATH}"'
        in wrapper_text
    )
    assert "Virtualenv not found or incomplete" not in result.stderr
    # The explicit shared override is authoritative: the command ran with the
    # shared env (uv-reached exit 7) rather than failing on the incomplete local.
    assert "uv-reached" in result.stderr


def _make_common_setup_venv_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Build a tiny Git repo whose public Ruff helper exposes selected virtualenv state."""
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    fake_bin = repo / "fake-bin"
    local_venv = repo / ".venv"
    script_dir.mkdir(parents=True)
    fake_bin.mkdir()
    (local_venv / "bin").mkdir(parents=True)

    (script_dir / "common_setup.sh").write_text(
        COMMON_SETUP.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (script_dir / "ruff_fix_format.sh").write_text(
        RUFF_FIX_FORMAT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (local_venv / "bin" / "activate").write_text(
        'export VIRTUAL_ENV="$REPO_ROOT/.venv"\n', encoding="utf-8"
    )
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ -n "${VIRTUAL_ENV:-}" && -n "${UV_PROJECT_ENVIRONMENT:-}" '
        '&& "$VIRTUAL_ENV" != "$UV_PROJECT_ENVIRONMENT" ]]; then\n'
        '  printf "virtualenv mismatch warning\\n" >&2\n'
        "fi\n"
        'printf "virtual_env=%s uv_project_environment=%s\\n" '
        '"${VIRTUAL_ENV:-}" "${UV_PROJECT_ENVIRONMENT:-}"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}
    return repo, env


def test_common_setup_restores_local_env_for_unmarked_foreign_virtualenv(tmp_path: Path) -> None:
    """Issue #7830: an unrelated active env must not override repository-local setup."""
    repo, env = _make_common_setup_venv_fixture(tmp_path)
    env["VIRTUAL_ENV"] = "/tmp/unrelated-foreign-venv"
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env.pop("ROBOT_SF_EXPLICIT_VENV_OVERRIDE", None)

    result = subprocess.run(
        ["bash", str(repo / "scripts" / "dev" / "ruff_fix_format.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert f"virtual_env={repo / '.venv'} uv_project_environment=" in result.stdout
    assert "mismatch warning" not in result.stderr


def test_common_setup_preserves_marked_explicit_shared_override(tmp_path: Path) -> None:
    """Issue #7830: a marked explicit override survives a real nested helper without warnings."""
    repo, env = _make_common_setup_venv_fixture(tmp_path)
    shared_venv = tmp_path / "shared-venv"
    env["VIRTUAL_ENV"] = str(shared_venv)
    env["UV_PROJECT_ENVIRONMENT"] = str(shared_venv)
    env["ROBOT_SF_EXPLICIT_VENV_OVERRIDE"] = str(shared_venv)

    result = subprocess.run(
        ["bash", str(repo / "scripts" / "dev" / "ruff_fix_format.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert f"virtual_env={shared_venv} uv_project_environment={shared_venv}" in result.stdout
    assert result.stderr == ""


def test_worktree_shared_venv_marks_explicit_override_for_nested_helpers() -> None:
    """The wrapper must bind the preservation marker to the exact selected environment."""
    script_text = RUN_WORKTREE_SHARED_VENV.read_text(encoding="utf-8")

    assert 'export ROBOT_SF_EXPLICIT_VENV_OVERRIDE="$venv_path"' in script_text


def test_worktree_shared_venv_falls_back_to_main_env_without_local_env(tmp_path: Path) -> None:
    """A linked worktree without an executable local env selects the fresh main env (issue #5984)."""
    matching_scene = "def normalize_integration_scheme(value=None):\n    return value\n"
    repo, main_venv, env = _make_freshness_fixture_repo(tmp_path, installed_scene=matching_scene)
    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    incomplete_local_venv = worktree / ".venv" / "bin"
    incomplete_local_venv.mkdir(parents=True)
    (incomplete_local_venv / "python").write_text("not executable\n", encoding="utf-8")

    result = subprocess.run(
        [str(RUN_WORKTREE_SHARED_VENV), "--", "python", "-V"],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 7
    assert f"venv={main_venv}" in result.stderr
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


def test_gh_comment_pr_uses_rest_api(tmp_path: Path) -> None:
    """PR comment publication must use REST validation and issue comments, not ``gh pr comment``."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "gh-calls.txt"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GH_COMMENT_CALLS"\n'
        "printf '%s\\n' '{\"id\": 1, \"number\": 6529}'\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    body_file = tmp_path / "comment.md"
    body_file.write_text("REST comment body\n", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GH_COMMENT_CALLS"] = str(calls)

    result = subprocess.run(
        [
            str(GH_COMMENT),
            "pr",
            "6529",
            "--repo",
            "ll7/robot_sf_ll7",
            "--body-file",
            str(body_file),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    assert call_lines[0].startswith("api repos/ll7/robot_sf_ll7/pulls/6529")
    assert "api --method POST repos/ll7/robot_sf_ll7/issues/6529/comments" in call_lines[1]
    assert "-F body=@" in call_lines[1]
    assert all("pr comment" not in call for call in call_lines)


def test_gh_comment_current_resolves_pr_via_rest(tmp_path: Path) -> None:
    """``--current`` must find the branch PR through REST instead of ``gh pr view``."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "gh-calls.txt"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GH_COMMENT_CALLS"\n'
        'if [[ "$*" == *state=open* ]]; then\n'
        "  printf '%s\\n' '6529'\n"
        "else\n"
        "  printf '%s\\n' '{\"id\": 1, \"number\": 6529}'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    body_file = tmp_path / "comment.md"
    body_file.write_text("REST current comment\n", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GH_COMMENT_CALLS"] = str(calls)

    result = subprocess.run(
        [
            str(GH_COMMENT),
            "pr",
            "--current",
            "--repo",
            "ll7/robot_sf_ll7",
            "--body-file",
            str(body_file),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    local_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    upstream = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    tracked_branch = local_branch
    if not tracked_branch:
        tracked_branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME", "")
    if upstream.returncode == 0 and "/" in upstream.stdout.strip():
        tracked_branch = upstream.stdout.strip().split("/", maxsplit=1)[1]
    assert len(call_lines) == 3
    assert f"pulls?state=open&head=ll7:{tracked_branch}" in call_lines[0]
    assert "pulls/6529" in call_lines[1]
    assert "issues/6529/comments" in call_lines[2]
    assert all("gh pr" not in call for call in call_lines)


def test_gh_comment_current_uses_event_ref_for_detached_checkout(tmp_path: Path) -> None:
    """A detached CI checkout uses the event ref for REST lookup."""
    repo = tmp_path / "detached-repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch", "main"],
        cwd=repo,
        check=True,
        timeout=30,
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, timeout=30)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=contract-test",
            "-c",
            "user.email=contract-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
        timeout=30,
    )
    subprocess.run(["git", "checkout", "--quiet", "--detach", "HEAD"], cwd=repo, check=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "gh-calls.txt"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GH_COMMENT_CALLS"\n'
        'if [[ "$*" == *state=open* ]]; then\n'
        "  printf '%s\\n' '6529'\n"
        "else\n"
        "  printf '%s\\n' '{\"id\": 1, \"number\": 6529}'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    body_file = tmp_path / "comment.md"
    body_file.write_text("REST detached comment\n", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GH_COMMENT_CALLS"] = str(calls)
    env.pop("GITHUB_HEAD_REF", None)
    env["GITHUB_REF_NAME"] = "main"

    result = subprocess.run(
        [
            str(GH_COMMENT),
            "pr",
            "--current",
            "--repo",
            "ll7/robot_sf_ll7",
            "--body-file",
            str(body_file),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(call_lines) == 3
    assert "pulls?state=open&head=ll7:main" in call_lines[0]


def test_gh_comment_current_falls_back_to_local_branch_without_upstream(tmp_path: Path) -> None:
    """An unpublished linked worktree uses its local branch for REST lookup."""
    main_repo = tmp_path / "main-repo"
    main_repo.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch", "main"],
        cwd=main_repo,
        check=True,
        timeout=30,
    )
    (main_repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=main_repo, check=True, timeout=30)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=contract-test",
            "-c",
            "user.email=contract-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=main_repo,
        check=True,
        timeout=30,
    )
    repo = tmp_path / "unpublished-worktree"
    subprocess.run(
        ["git", "worktree", "add", "--quiet", "-b", "unpublished", str(repo), "HEAD"],
        cwd=main_repo,
        check=True,
        timeout=30,
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "gh-calls.txt"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GH_COMMENT_CALLS"\n'
        'if [[ "$*" == *state=open* ]]; then\n'
        "  printf '%s\\n' '6529'\n"
        "else\n"
        "  printf '%s\\n' '{\"id\": 1, \"number\": 6529}'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    body_file = tmp_path / "comment.md"
    body_file.write_text("REST unpublished comment\n", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GH_COMMENT_CALLS"] = str(calls)

    result = subprocess.run(
        [
            str(GH_COMMENT),
            "pr",
            "--current",
            "--repo",
            "ll7/robot_sf_ll7",
            "--body-file",
            str(body_file),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(call_lines) == 3
    assert "pulls?state=open&head=ll7:unpublished" in call_lines[0]
    assert "pulls/6529" in call_lines[1]
    assert "issues/6529/comments" in call_lines[2]


def test_gh_comment_issue_uses_rest_api(tmp_path: Path) -> None:
    """Issue comment publication must use REST validation and issue comments, not ``gh issue comment``.

    Mirrors ``test_gh_comment_pr_uses_rest_api`` so the issue path is provably
    quota-independent: under a mocked environment where GraphQL is exhausted but
    REST remains available, it invokes only ``gh api`` REST calls (no
    ``gh issue comment``, no GraphQL) to validate the target and publish.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "gh-calls.txt"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GH_COMMENT_CALLS"\n'
        "printf '%s\\n' '{\"id\": 1, \"number\": 6843}'\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    body_file = tmp_path / "comment.md"
    body_file.write_text("REST issue comment body\n", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GH_COMMENT_CALLS"] = str(calls)

    result = subprocess.run(
        [
            str(GH_COMMENT),
            "issue",
            "6843",
            "--repo",
            "ll7/robot_sf_ll7",
            "--body-file",
            str(body_file),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    # REST validation lookup (GET repos/<owner>/<repo>/issues/<number>) precedes the POST.
    assert call_lines[0].startswith("api repos/ll7/robot_sf_ll7/issues/6843")
    assert "--method POST" not in call_lines[0]
    # Publication uses the REST issue-comments endpoint with the body file.
    assert "api --method POST repos/ll7/robot_sf_ll7/issues/6843/comments" in call_lines[1]
    assert "-F body=@" in call_lines[1]
    # The issue path must never fall back to ``gh issue comment`` (which routes
    # through GraphQL under quota exhaustion) nor issue any GraphQL call.
    assert all("issue comment" not in call for call in call_lines)
    assert all("graphql" not in call.lower() for call in call_lines)


def test_gh_comment_issue_fail_closed_on_missing(tmp_path: Path) -> None:
    """Issue path must exit nonzero and skip the POST when the target is missing/unknown.

    When the REST issue lookup reports a missing or unknown target, the script
    fails closed (nonzero exit) and never publishes a comment, so a degraded
    GraphQL lookup cannot masquerade as a successful publication.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "gh-calls.txt"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GH_COMMENT_CALLS"\n'
        # A publication POST must never be reached when validation fails; if it
        # somehow is, surface a distinct non-matching exit so the assertion fails.
        'case "$*" in\n'
        '  *"--method POST"*) echo "POST reached despite failed validation" >&2; exit 5 ;;\n'
        '  *) echo "HTTP 404: Not Found" >&2; exit 1 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    body_file = tmp_path / "comment.md"
    body_file.write_text("must not be posted\n", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GH_COMMENT_CALLS"] = str(calls)

    result = subprocess.run(
        [
            str(GH_COMMENT),
            "issue",
            "9999999",
            "--repo",
            "ll7/robot_sf_ll7",
            "--body-file",
            str(body_file),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "could not be resolved through the REST API" in result.stderr
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    # Only the REST validation lookup is recorded; no POST and no ``gh issue comment``.
    assert any(call.startswith("api repos/ll7/robot_sf_ll7/issues/9999999") for call in call_lines)
    assert not any("--method POST" in call for call in call_lines)
    assert not any("issue comment" in call for call in call_lines)


def _run_headless_package_install_fixture(
    tmp_path: Path,
    *,
    update_output: str,
    update_rc: int,
    install_rc: int,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run the headless package helper against deterministic fake APT commands."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "apt-calls.txt"
    install_marker = tmp_path / "install-called"

    fake_dpkg_query = fake_bin / "dpkg-query"
    fake_dpkg_query.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    fake_dpkg_query.chmod(0o755)

    fake_sudo = fake_bin / "sudo"
    fake_sudo.write_text(
        '#!/usr/bin/env bash\nset -eu\nprintf \'%s\\n\' "$*" >> "$APT_CALLS"\nexec "$@"\n',
        encoding="utf-8",
    )
    fake_sudo.chmod(0o755)

    fake_apt_get = fake_bin / "apt-get"
    fake_apt_get.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'if [[ "$*" == *" update" ]]; then\n'
        "  printf '%s\\n' \"$APT_UPDATE_OUTPUT\" >&2\n"
        '  exit "$APT_UPDATE_RC"\n'
        "fi\n"
        'if [[ "$*" == *" install "* ]]; then\n'
        '  : > "$APT_INSTALL_MARKER"\n'
        '  exit "$APT_INSTALL_RC"\n'
        "fi\n"
        'echo "unexpected apt-get invocation: $*" >&2\n'
        "exit 99\n",
        encoding="utf-8",
    )
    fake_apt_get.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "APT_CALLS": str(calls),
            "APT_INSTALL_MARKER": str(install_marker),
            "APT_INSTALL_RC": str(install_rc),
            "APT_UPDATE_OUTPUT": update_output,
            "APT_UPDATE_RC": str(update_rc),
        }
    )
    result = subprocess.run(
        ["bash", str(CI_INSTALL_HEADLESS_PACKAGES), "libglib2.0-0"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result, install_marker, calls


def test_ci_install_headless_ignores_unrelated_third_party_403(tmp_path: Path) -> None:
    """A Microsoft-source 403 must not prevent a usable package install."""
    result, install_marker, calls = _run_headless_package_install_fixture(
        tmp_path,
        update_output=(
            "Err:1 https://packages.microsoft.com/repos/azure-cli noble InRelease\n"
            "  403 Forbidden [IP: 13.107.246.40 443]\n"
            "Hit:2 http://archive.ubuntu.com/ubuntu noble InRelease"
        ),
        update_rc=100,
        install_rc=0,
    )

    assert result.returncode == 0, result.stderr
    assert install_marker.exists()
    assert "warning=ignored_third_party_apt_403 hosts=packages.microsoft.com" in result.stdout
    assert len([line for line in result.stdout.splitlines() if "warning=" in line]) == 1
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    assert any(line.endswith(" update") for line in call_lines)
    assert any(" install " in line for line in call_lines)


def test_ci_install_headless_preserves_requested_install_failure(tmp_path: Path) -> None:
    """Ignoring a third-party 403 must not hide a failed requested install."""
    result, install_marker, _ = _run_headless_package_install_fixture(
        tmp_path,
        update_output=(
            "Err:1 https://packages.microsoft.com/ubuntu/24.04/prod noble InRelease\n"
            "  403 Forbidden"
        ),
        update_rc=100,
        install_rc=42,
    )

    assert result.returncode == 42
    assert install_marker.exists()
    assert "warning=ignored_third_party_apt_403 hosts=packages.microsoft.com" in result.stdout


def test_ci_install_headless_fails_closed_on_non_ignorable_update_error(tmp_path: Path) -> None:
    """An official-source update failure must still stop before installation."""
    result, install_marker, _ = _run_headless_package_install_fixture(
        tmp_path,
        update_output=(
            "Err:1 http://archive.ubuntu.com/ubuntu noble InRelease\n  500  Internal Server Error"
        ),
        update_rc=100,
        install_rc=0,
    )

    assert result.returncode == 100
    assert not install_marker.exists()
    assert "error=apt_update_failed rc=100" in result.stderr


def test_ci_install_headless_fails_closed_on_mixed_update_errors(tmp_path: Path) -> None:
    """A tolerated third-party 403 must not mask another source failure."""
    result, install_marker, _ = _run_headless_package_install_fixture(
        tmp_path,
        update_output=(
            "Err:1 https://packages.microsoft.com/repos/azure-cli noble InRelease\n"
            "  403 Forbidden\n"
            "Err:2 http://archive.ubuntu.com/ubuntu noble InRelease\n"
            "  500  Internal Server Error"
        ),
        update_rc=100,
        install_rc=0,
    )

    assert result.returncode == 100
    assert not install_marker.exists()
    assert "error=apt_update_failed rc=100" in result.stderr
    assert "sources=" in result.stderr
    assert "unknown" not in result.stderr


def test_gh_comment_succeeds_with_empty_post_response(tmp_path: Path) -> None:
    """Successful REST POST with empty response body must exit 0 (issue #6891).

    ``gh api`` may fail to parse an empty or malformed response body even when
    the REST POST succeeded and the comment was created.  The helper must
    suppress response output and return the POST exit code deterministically,
    preventing a duplicate retry surface.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "gh-calls.txt"
    fake_gh = fake_bin / "gh"
    # Validation succeeds. Without --silent, emulate the CLI's parse failure
    # after a successful POST; --silent is the contract under test.
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GH_COMMENT_CALLS"\n'
        'case "$*" in\n'
        '  *"--method POST"*)\n'
        '    if [[ "$*" == *"--silent"* ]]; then exit 0; fi\n'
        '    echo "unexpected end of JSON input" >&2\n'
        "    exit 1\n"
        "    ;;\n"
        "  *) printf '%s\\n' '{\"id\":1}' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    body_file = tmp_path / "comment.md"
    body_file.write_text("posted despite empty response\n", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{os.pathsep}{env['PATH']}"
    env["GH_COMMENT_CALLS"] = str(calls)

    result = subprocess.run(
        [
            str(GH_COMMENT),
            "issue",
            "6877",
            "--repo",
            "ll7/robot_sf_ll7",
            "--body-file",
            str(body_file),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    assert any("--method POST" in call for call in call_lines)
    assert any("--silent" in call for call in call_lines)
    assert any("repos/ll7/robot_sf_ll7/issues/6877/comments" in call for call in call_lines)


def test_gh_comment_fails_on_nonzero_post_exit(tmp_path: Path) -> None:
    """A nonzero REST POST exit code must propagate (not be swallowed by empty-response handling).

    The helper must not hide a genuine POST failure behind the empty-response
    guard introduced for issue #6891.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "gh-calls.txt"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GH_COMMENT_CALLS"\n'
        'case "$*" in\n'
        '  *"--method POST"*) echo "HTTP 500: Internal Server Error" >&2; exit 1 ;;\n'
        "  *) printf '%s\\n' '{\"id\":1}' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    body_file = tmp_path / "comment.md"
    body_file.write_text("should fail\n", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{os.pathsep}{env['PATH']}"
    env["GH_COMMENT_CALLS"] = str(calls)

    result = subprocess.run(
        [
            str(GH_COMMENT),
            "issue",
            "6877",
            "--repo",
            "ll7/robot_sf_ll7",
            "--body-file",
            str(body_file),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    assert any("--method POST" in call for call in call_lines)


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
# evidence_registry_ratchet.py contract tests (issue #5994)
# ---------------------------------------------------------------------------


def test_evidence_registry_ratchet_is_directly_executable() -> None:
    """Keep the canonical evidence-registry ratchet invocation executable."""
    assert EVIDENCE_REGISTRY_RATCHET.exists(), f"Missing: {EVIDENCE_REGISTRY_RATCHET}"
    assert EVIDENCE_REGISTRY_RATCHET.stat().st_mode & 0o111, (
        "evidence_registry_ratchet.py is not executable"
    )


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
    assert "UV_NO_SYNC=1" in result.stdout
    assert "env -u UV_NO_SYNC" in result.stdout


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
                '  printf "# fake activation\\n" > "$2/bin/activate"',
                "  exit 0",
                "fi",
                'if [[ "$1" == "sync" ]]; then',
                '  if [[ -n "${UV_NO_SYNC:-}" ]]; then echo "UV_NO_SYNC was not cleared" >&2; exit 98; fi',
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
                '  printf "# fake activation\\n" > "$2/bin/activate"',
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
    assert 'env -u UV_NO_SYNC UV_PROJECT_ENVIRONMENT="$local_venv" uv sync' in body
    assert 'activate_marker="# robot_sf bootstrap: preserve selected extras for uv run"' in body


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
                '  printf "# fake activation\\n" > "$venv_dir/bin/activate"',
                "  exit 0",
                "fi",
                'if [[ "$1" == "sync" ]]; then',
                '  if [[ -n "${UV_NO_SYNC:-}" ]]; then echo "UV_NO_SYNC was not cleared" >&2; exit 98; fi',
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
    activation = repo / ".venv" / "bin" / "activate"
    activation_text = activation.read_text(encoding="utf-8")
    assert activation_text.count("export UV_NO_SYNC=1") == 1

    second_result = subprocess.run(
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
    assert second_result.returncode == 0, second_result.stderr
    assert activation.read_text(encoding="utf-8").count("export UV_NO_SYNC=1") == 1

    clean_env = {key: value for key, value in os.environ.items() if key != "UV_NO_SYNC"}
    activation_result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\n" "$UV_NO_SYNC"', "activate", str(activation)],
        cwd=repo,
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert activation_result.returncode == 0, activation_result.stderr
    assert activation_result.stdout.strip() == "1"


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


def test_coverage_docs_match_effective_source_scope() -> None:
    """Docs must distinguish coverage.py defaults from pytest-cov's effective source."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    coverage_run = pyproject["tool"]["coverage"]["run"]

    cov_guide_text = COVERAGE_GUIDE.read_text(encoding="utf-8")
    dev_guide_text = DEV_GUIDE.read_text(encoding="utf-8")
    wrapper_text = RUN_TESTS_PARALLEL.read_text(encoding="utf-8")

    assert 'source = ["robot_sf", "fast-pysf/pysocialforce"]' in cov_guide_text
    assert "fast-pysf/pysocialforce" in coverage_run["source"]
    assert 'cmd+=("--cov=robot_sf" "--cov-report=html" "--cov-report=json")' in wrapper_text
    assert "Only the `robot_sf/` package" in cov_guide_text
    assert "not included in the local wrapper report" in cov_guide_text
    assert "measure only the `robot_sf/` package" in dev_guide_text
    assert "not included in wrapper reports" in dev_guide_text

    assert '"fast-pysf/tests/*"' in cov_guide_text
    assert '"fast-pysf/examples/*"' in cov_guide_text
    config_section = cov_guide_text.split("## Configuration")[1].split("### Customization")[0]
    assert '"fast-pysf/*"' not in config_section

    assert "branch = true" in config_section
    assert 'data_file = "output/coverage/.coverage"' in config_section
    assert "skip_covered = false" in config_section
    assert "skip_empty = false" in config_section
    assert '"raise NotImplementedError"' in config_section

    assert "fast-pysf/tests/*" in dev_guide_text


def test_coverage_docs_match_ci_workflow_contract() -> None:
    """Docs must accurately reflect CI workflow coverage gate semantics."""
    cov_guide_text = COVERAGE_GUIDE.read_text(encoding="utf-8")
    dev_guide_text = DEV_GUIDE.read_text(encoding="utf-8")
    ci_workflow_text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "COVERAGE_CORE:" in ci_workflow_text
    assert "ctrace" in ci_workflow_text
    assert "sysmon" in ci_workflow_text
    assert "--minimum-total 85.0" in ci_workflow_text
    assert "coverage combine output/coverage" in ci_workflow_text
    assert 'ROBOT_SF_PYTEST_COVERAGE: "1"' in ci_workflow_text
    assert "changed-coverage-gate:" in ci_workflow_text
    assert '--base-sha "$BASE_SHA"' in ci_workflow_text
    assert '--head-sha "$HEAD_SHA"' in ci_workflow_text
    assert "--json-output output/coverage/changed-coverage-result.json" in ci_workflow_text
    assert "github.event.pull_request.base.sha" in ci_workflow_text
    assert "github.event.pull_request.head.sha" in ci_workflow_text
    assert "github.event.merge_group.base_sha" in ci_workflow_text
    assert '"changed-coverage.v1"' in (
        ROOT / "scripts" / "coverage" / "check_changed_files_coverage.py"
    ).read_text(encoding="utf-8")

    assert "sysmon" in cov_guide_text
    assert "--minimum-total 85.0" in cov_guide_text
    assert "coverage combine" in cov_guide_text
    assert "pull request" in cov_guide_text.lower()
    assert "coverage-gate" in cov_guide_text

    assert "coverage-gate" in dev_guide_text
    assert "changed-coverage-gate" in dev_guide_text
    assert "changed-coverage.v1" in dev_guide_text
    assert "85.0%" in dev_guide_text or "85.0" in dev_guide_text


def test_gh_pr_merge_wrapper_has_valid_shell_syntax() -> None:
    """gh_pr_merge.sh should pass bash -n syntax check."""
    syntax = subprocess.run(
        ["bash", "-n", str(GH_PR_MERGE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_gh_pr_merge_wrapper_help() -> None:
    """gh_pr_merge.sh --help and -h print usage and exit 0."""
    for flag in ("--help", "-h"):
        result = subprocess.run(
            [str(GH_PR_MERGE), flag],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Usage:" in result.stdout
        assert "--match-head-commit" in result.stdout


def test_gh_pr_merge_wrapper_refuses_without_exact_head_binding() -> None:
    """The REST fallback must never run without a full expected head SHA."""
    result = subprocess.run(
        [str(GH_PR_MERGE), "1234", "--match-head-commit", "short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "requires a full 40-char SHA" in result.stderr
