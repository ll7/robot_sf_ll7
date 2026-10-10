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
