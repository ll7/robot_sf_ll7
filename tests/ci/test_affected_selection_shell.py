"""Affected admission must reach the real shell command, including full fallback."""

from tests.support.ci_wrapper_fixture import captured_calls, run_wrapper, wrapper_repository


def test_shell_invokes_selector_and_does_not_filter_full_fallback(tmp_path):
    """Dropping the shell call or retaining not-slow excludes required witnesses."""
    repo, env = wrapper_repository(tmp_path)
    result = run_wrapper(
        repo, {**env, "ROBOT_SF_AFFECTED_BASE_REF": "HEAD", "SELECTION_MODE": "full"}, "tests"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = captured_calls(repo)
    selector = next(c["args"] for c in calls if "affected_test_selection.py" in " ".join(c["args"]))
    assert selector[-4:] == ["--base", "HEAD", "--format", "mode"]
    pytest_args = next(c["args"] for c in calls if c["args"][:2] == ["run", "pytest"])
    assert "-m" not in pytest_args and "not slow or affected" not in pytest_args


def test_shell_applies_affected_marker_instead_of_full_suite(tmp_path):
    """The real selector's mapped decision must reach pytest's marker expression."""
    import os
    import subprocess
    from pathlib import Path

    repo, env = wrapper_repository(tmp_path)
    (repo / "model.py").write_text("VALUE = 1\n")
    (repo / "tests/test_one.py").write_text(
        "import pytest\npytestmark = pytest.mark.slow\nfrom model import VALUE\ndef test_one(): assert False, 'mapped slow executed'\n"
    )
    (repo / "tests/test_unrelated.py").write_text(
        "import pytest\npytestmark = pytest.mark.slow\ndef test_unrelated(): assert False, 'unrelated slow executed'\n"
    )
    (repo / "tests/conftest.py").write_text("""
from tests import conftest as production

def pytest_collection_modifyitems(config, items):
    # Run the actual hook with this synthetic checkout as its filesystem root.
    original = production.__file__
    try:
        production.__file__ = __file__
        production.pytest_collection_modifyitems(config, items)
    finally:
        production.__file__ = original
""")
    subprocess.run(
        [
            "git",
            "add",
            "model.py",
            "tests/test_one.py",
            "tests/test_unrelated.py",
            "tests/conftest.py",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "mapped input"], cwd=repo, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "model.py").write_text("VALUE = 2\n")
    subprocess.run(["git", "add", "model.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "changed module"], cwd=repo, check=True)
    result = run_wrapper(
        repo,
        {
            **env,
            "ROBOT_SF_AFFECTED_BASE_REF": base,
            "REAL_PYTEST": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]) + os.pathsep + str(repo),
        },
        "tests",
        "-n",
        "0",
        "-q",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    args = next(c["args"] for c in captured_calls(repo) if c["args"][:2] == ["run", "pytest"])
    assert args[args.index("-m") + 1] == "not slow or affected"
    assert "mapped slow executed" in result.stderr
    assert "unrelated slow executed" not in result.stderr
