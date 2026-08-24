"""Tests for the shared FakeSubprocess test fixture."""

from __future__ import annotations

import subprocess

import pytest

from tests.conftest import FakeSubprocess


def test_fake_subprocess_records_calls() -> None:
    """FakeSubprocess records call history accurately."""
    fake = FakeSubprocess()
    fake.set_default(stdout="ok")

    fake(["git", "status"])
    fake(["git", "branch", "-a"])

    assert fake.call_count == 2
    assert fake.calls == [["git", "status"], ["git", "branch", "-a"]]
    assert fake.last_call == ["git", "branch", "-a"]
    assert fake.called(["git", "status"])
    assert fake.called("git")
    assert not fake.called(["gh", "pr"])


def test_fake_subprocess_exact_and_prefix_matching() -> None:
    """Handlers match exact commands or command prefixes."""
    fake = FakeSubprocess()
    fake.register(["git", "status", "--porcelain"], stdout=" M file.txt\n")
    fake.register(["git", "rev-parse"], stdout="deadbeef\n")
    fake.register("gh", stdout='[{"number": 123}]')

    res1 = fake(["git", "status", "--porcelain"])
    res2 = fake(["git", "rev-parse", "HEAD"])
    res3 = fake(["gh", "pr", "list"])

    assert res1.stdout == " M file.txt\n"
    assert res1.returncode == 0
    assert res2.stdout == "deadbeef\n"
    assert res3.stdout == '[{"number": 123}]'


def test_fake_subprocess_callable_and_exception_handlers() -> None:
    """Handlers can be callables or raise configured exceptions."""
    fake = FakeSubprocess()
    fake.register("sstat", FileNotFoundError("sstat missing"))
    fake.register(
        lambda cmd: "--custom" in cmd,
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 42, stdout="custom-handled"),
    )

    with pytest.raises(FileNotFoundError, match="sstat missing"):
        fake(["sstat", "-j", "123"])

    custom_res = fake(["mytool", "--custom", "arg"])
    assert custom_res.returncode == 42
    assert custom_res.stdout == "custom-handled"


def test_fake_subprocess_unmatched_raises_assertion_error() -> None:
    """Unmatched command without default raises AssertionError."""
    fake = FakeSubprocess()
    fake.register(["known", "cmd"], stdout="ok")

    with pytest.raises(AssertionError, match="Unexpected subprocess command"):
        fake(["unknown", "cmd"])


def test_fake_subprocess_custom_passthrough_object() -> None:
    """Arbitrary return objects pass through untouched."""
    fake = FakeSubprocess()
    custom_obj = {"status": "success", "data": [1, 2, 3]}
    fake.register(["custom", "cmd"], custom_obj)

    result = fake(["custom", "cmd"])
    assert result == custom_obj


def test_fake_subprocess_kwargs_tracking() -> None:
    """FakeSubprocess records kwargs passed to calls."""
    fake = FakeSubprocess()
    fake.set_default(stdout="ok")

    fake(["git", "status"], timeout=30, cwd="/tmp")
    fake(["git", "log"], check=True)

    assert len(fake.kwargs_history) == 2
    assert fake.kwargs_history[0] == {"timeout": 30, "cwd": "/tmp"}
    assert fake.last_kwargs == {"check": True}
