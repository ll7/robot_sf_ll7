"""Train evidence must cover a complete, clean checkout at the requested head."""

import json
import subprocess

import pytest

from tests.support.ci_wrapper_fixture import captured_calls, run_wrapper, wrapper_repository


def head(repo):
    """Resolve the fixture commit through the same Git identity used by the wrapper."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def invoke(repo, env, *args):
    """Require an exact head and place evidence in the fixture's ignored output directory."""
    return run_wrapper(
        repo,
        env,
        "--train-suite",
        "--expect-head",
        head(repo),
        "--receipt-file",
        "output/train.json",
        *args,
    )


@pytest.mark.parametrize(
    "path", ["tests/test_untracked.py", "conftest.py", "robot_sf/untracked.py"]
)
def test_train_suite_refuses_untracked_inputs(tmp_path, path):
    """Uncommitted collected or imported Python must never become head-bound evidence."""
    repo, env = wrapper_repository(tmp_path)
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("pass\n")
    result = invoke(repo, env)
    assert result.returncode == 2
    assert "clean tree including untracked files" in result.stderr
    assert not captured_calls(repo)


def test_train_suite_refuses_dirty_tracked_inputs(tmp_path):
    """A tracked edit also invalidates the claimed commit tree before pytest starts."""
    repo, env = wrapper_repository(tmp_path)
    (repo / "tests/test_one.py").write_text("pass\n")
    result = invoke(repo, env)
    assert result.returncode == 2
    assert "clean tree including untracked files" in result.stderr
    assert not captured_calls(repo)


def test_train_suite_requires_expected_head(tmp_path):
    """A command without the reviewed commit cannot supply train evidence."""
    repo, env = wrapper_repository(tmp_path)
    result = run_wrapper(repo, env, "--train-suite")
    assert result.returncode == 2
    assert "--expect-head is required" in result.stderr
    assert not captured_calls(repo)


def test_train_suite_refuses_stale_head(tmp_path):
    """A different requested commit must be refused before test execution."""
    repo, env = wrapper_repository(tmp_path)
    result = run_wrapper(repo, env, "--train-suite", "--expect-head", "0" * 40)
    assert result.returncode == 2
    assert "expected head differs" in result.stderr
    assert not captured_calls(repo)


@pytest.mark.parametrize("partial", ["sparse", "skip-worktree", "assume-unchanged"])
def test_train_suite_refuses_partial_checkout(tmp_path, partial):
    """Git's hidden or omitted working files cannot witness a complete commit tree."""
    repo, env = wrapper_repository(tmp_path)
    if partial == "sparse":
        subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=repo, check=True)
    else:
        subprocess.run(
            ["git", "update-index", "--" + partial, "tests/test_one.py"], cwd=repo, check=True
        )
        (repo / "tests/test_one.py").unlink()
    result = invoke(repo, env)
    assert result.returncode == 2
    assert "complete checkout" in result.stderr
    assert not captured_calls(repo)


def test_train_suite_refuses_partial_selectors_in_clean_tree(tmp_path):
    """A clean fixture makes selector rejection independent of the dirty-tree guard."""
    repo, env = wrapper_repository(tmp_path)
    result = invoke(repo, env, "tests/ci")
    assert result.returncode == 2
    assert "no pytest selectors" in result.stderr
    assert "clean tree" not in result.stderr
    assert not captured_calls(repo)


@pytest.mark.parametrize(
    "addopts", ["-k test_one", "-m slow", "--lf", "--deselect=tests/test_one.py::test_one"]
)
def test_train_suite_normalizes_partial_environment_and_records_proof(tmp_path, addopts):
    """The real shell runs both roots with timeout support and retains counts and source identity."""
    repo, env = wrapper_repository(tmp_path)
    env.update(
        PYTEST_ADDOPTS=addopts, PYTEST_SHARD_COUNT="6", PYTEST_SHARD_INDEX="2", PYTEST_FAST_FAIL="1"
    )
    result = invoke(repo, env)
    assert result.returncode == 0, result.stderr
    call = next(call for call in captured_calls(repo) if call["args"][:2] == ["run", "pytest"])
    args = call["args"]
    assert call["addopts"] is None and call["shards"] is None
    assert "tests" in args and "fast-pysf/tests" in args
    assert "--maxfail=0" in args and "addopts=" not in args
    assert args[args.index("-p") + 1] == "timeout" and "--timeout=300" in args
    for partial in ("-m", "-k", "--lf", "--splits", "-x", "--failed-first"):
        assert partial not in args
    receipt = json.loads((repo / "output/train.json").read_text())
    assert receipt["complete"] is True
    assert receipt["head_sha"] == head(repo)
    assert (
        receipt["tree_sha"]
        == subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip()
    )
    assert receipt["counts"] == {"passed": 2, "skipped": 1}
    assert receipt["pytest_exit_code"] == receipt["exit_code"] == 0
    assert (repo / "output/train.log").is_file()
    assert "2 passed, 1 skipped" in result.stdout


@pytest.mark.parametrize(
    "change, reason",
    [
        ("MOVE_HEAD", "expected head differs"),
        ("DIRTY_AFTER", "clean tree including untracked files"),
    ],
)
def test_train_suite_rechecks_binding_after_pytest(tmp_path, change, reason):
    """A successful process cannot certify a checkout that moved or became dirty during it."""
    repo, env = wrapper_repository(tmp_path)
    env[change] = "1"
    result = invoke(repo, env)
    assert result.returncode == 2, result.stderr
    receipt = json.loads((repo / "output/train.json").read_text())
    assert receipt["complete"] is False and receipt["exit_code"] == 2
    assert receipt["pytest_exit_code"] == 0
    assert reason in receipt["binding_error"]


def test_train_suite_failure_is_not_complete_evidence(tmp_path):
    """Retain a failed pytest result without turning its summary into a complete passing receipt."""
    repo, env = wrapper_repository(tmp_path)
    env["FAKE_PYTEST_EXIT"] = "1"
    result = invoke(repo, env)
    assert result.returncode == 1, result.stderr
    receipt = json.loads((repo / "output/train.json").read_text())
    assert receipt["complete"] is False
    assert receipt["pytest_exit_code"] == receipt["exit_code"] == 1


def test_train_shards_consume_one_bound_selection_artifact():
    """The setup result must reach the real shell instead of being rescanned per shard."""
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text()
    )
    job = workflow["jobs"]["fast-feedback"]
    assert "github.event.pull_request.base.sha" in job["env"]["ROBOT_SF_AFFECTED_BASE_REF"]
    assert "output/ci/affected-selection.json" in job["env"]["ROBOT_SF_AFFECTED_SELECTION_FILE"]
    download = next(
        (s for s in job["steps"] if s.get("name") == "Download conservative test admission"), None
    )
    assert download is not None
    assert download["with"]["name"] == "conservative-test-admission"
    assert download["with"]["path"] == "output/ci"


@pytest.mark.parametrize("disable_autoload", [False, True])
def test_wrapper_timeout_works_with_real_pytest_in_both_plugin_modes(tmp_path, disable_autoload):
    """A real pytest subprocess catches option rejection and duplicate plugin registration."""
    repo, env = wrapper_repository(tmp_path)
    env.update(
        REAL_PYTEST="1",
        PYTEST_NUM_WORKERS="1",
        PYTEST_SHARD_COUNT="1",
        ROBOT_SF_SHARD_INCLUDE_SLOW="1",
    )
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    if disable_autoload:
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = run_wrapper(repo, env, "tests", "fast-pysf/tests")
    assert result.returncode == 0, result.stdout + result.stderr


def test_train_suite_preserves_project_pytest_configuration(tmp_path):
    """Complete roots must retain import and strict-config semantics from the commit."""
    repo, env = wrapper_repository(tmp_path)
    (repo / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = ["--import-mode=importlib", "--strict-markers", "--strict-config", "--durations=10"]\n'
    )
    (repo / "tests/test_one.py").write_text(
        "def test_one(pytestconfig):\n"
        '    assert pytestconfig.getoption("importmode") == "importlib"\n'
        '    assert pytestconfig.getoption("strict_markers") is True\n'
        '    assert pytestconfig.getoption("strict_config") is True\n'
        '    assert pytestconfig.getoption("durations") == 10\n'
    )
    subprocess.run(["git", "add", "pyproject.toml", "tests/test_one.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "project pytest configuration"], cwd=repo, check=True)
    env.update(REAL_PYTEST="1", PYTEST_NUM_WORKERS="1")
    result = invoke(repo, env)
    assert result.returncode == 0, result.stdout + result.stderr
